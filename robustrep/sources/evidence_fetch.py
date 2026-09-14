"""Fetch evidence URIs (http/ipfs/data) and classify every distinct URI into the cache.

Evidence URIs come from on-chain data written by arbitrary, untrusted parties, so
the fetcher here is a security boundary, not just an HTTP client:

- **SSRF guard** (``_is_safe_url``): only ``http``/``https`` on the standard ports
  (80/443) to a host that is not ``localhost`` (a trailing FQDN dot is stripped
  first, so ``localhost.`` is caught too), is not a bare IP literal (an evidence
  URI must name a resolvable host -- see below), is pure ASCII, and does not
  *resolve* (via an injectable ``resolver``) to any address ``_is_disallowed_ip``
  rejects (private/loopback/link-local/multicast/reserved/unspecified/non-global,
  including CGNAT and 6to4-embedded targets). Refusal never raises -- it just
  yields ``None`` from the fetch, which ``robustrep.evidence.classify`` treats as
  "unfetchable" (level 1), the conservative outcome.
- **Bare IP literals are refused outright**, even public ones: requiring a
  resolvable hostname means *every* fetch goes through the resolver check above
  -- there is no "it's just an IP, skip DNS" path for an attacker to route
  around it.
- **Non-ASCII hosts are refused**: Python's resolver (``socket.getaddrinfo``,
  IDNA2003) and ``requests``/``urllib3`` (IDNA2008/UTS-46) can encode the same
  Unicode hostname to *different* ASCII (punycode) labels for certain code
  points. A hostname that is safe under one encoding is not guaranteed safe
  under the other, so non-ASCII hosts are refused rather than trusted to a
  resolver check that might be answering about a different host than the one
  ``requests`` will actually connect to.
- **Bounded manual redirects**: automatic redirects are disabled
  (``allow_redirects=False``); up to ``MAX_REDIRECTS`` 3xx hops are followed by
  hand, re-running the SSRF guard against each ``Location`` before following it.
  Every response (200, 3xx, 4xx/5xx, or one that fails mid-read) is closed
  before the function returns or recurses into the next hop.
- **Bounded size/time**: at most ``MAX_BYTES`` bytes are read per response
  (regardless of any ``Content-Length`` claim -- we just want the beginning),
  clamped a second time after the read as a belt-and-suspenders check against a
  non-conforming stream, and every request uses a ``(connect, read)`` timeout.

``classify_all`` walks every URI referenced by ``feedback`` rows that is not yet
in the evidence cache, classifies it, and persists the level -- one bad URI (an
unexpected exception out of ``classify``) is caught, logged, and recorded as
level 1 rather than aborting the whole batch.

**Known limitation -- DNS rebinding (not mitigated in v0.1):** the guard above
resolves the host and checks *those* addresses, but the actual connection is
made by ``requests``/``urllib3``, which resolves the host *again* independently.
A DNS-rebinding attacker (answering a public IP on the first lookup and a
private one on the second, timed to land between the two resolutions) can pass
the guard and cause a blind GET to a private address. The blast radius is
bounded -- no response content is ever exposed to the caller or stored, only a
0-3 evidence level -- but this is still a real gap. Proper mitigation (resolve
once, then fetch via a pinned-IP transport adapter with the original hostname
kept for TLS SNI/Host) is scheduled for v0.2; see
``test_dns_rebinding_not_mitigated_in_v0_1`` in the test suite, which documents
this with a ``strict=True`` xfail so it starts failing (as a reminder to update
docs/tests) the moment it's actually fixed.
"""
from __future__ import annotations

import base64
import ipaddress
import logging
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Sequence
from urllib.parse import unquote, urljoin, urlsplit

import requests

from ..evidence import classify
from ..store import Store

IPFS_GATEWAYS = ("https://ipfs.io/ipfs/", "https://cloudflare-ipfs.com/ipfs/")
MAX_BYTES = 200_000
MAX_REDIRECTS = 3
ALLOWED_PORTS = (80, 443)
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 10
DEFAULT_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)
USER_AGENT = "robustrep/0.1"

_log = logging.getLogger(__name__)

Resolver = Callable[[str], Sequence[str]]

# 6to4 (RFC 3056): embeds an IPv4 address in bits 16-47 of the IPv6 address.
# Python's ipaddress module does not treat 2002::/16 itself as non-global, so
# without unwrapping it a 6to4 literal could smuggle a private/loopback IPv4
# target straight past `is_global` -- see `_is_disallowed_ip`.
_SIX_TO_FOUR = ipaddress.ip_network("2002::/16")


def _default_resolver(host: str) -> list[str]:
    """Resolve ``host`` to its IP addresses via the system resolver."""
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def _is_disallowed_ip(ip_str: str) -> bool:
    """True if ``ip_str`` is not a valid IP literal, or is one that must never
    be fetched from a server processing untrusted URIs.

    ``is_multicast or is_reserved or not is_global`` covers private, loopback,
    link-local, reserved, unspecified and CGNAT (100.64.0.0/10) addresses in
    one check; ``is_reserved`` additionally catches NAT64 (``64:ff9b::/96``
    and the RFC 8215 local-use ``64:ff9b:1::/48``), both of which Python's
    ipaddress module reports as ``is_global=True`` despite embedding an IPv4
    address that itself may be private/loopback. 6to4 (2002::/16) is
    similarly ``is_global=True`` but *not* ``is_reserved``, so it is unwrapped
    separately and its embedded IPv4 re-checked (see ``_SIX_TO_FOUR``).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    if ip.is_multicast or ip.is_reserved or not ip.is_global:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip in _SIX_TO_FOUR:
        embedded = ipaddress.IPv4Address(ip.packed[2:6])
        return _is_disallowed_ip(str(embedded))
    return False


def _is_safe_url(url: str, resolver: Resolver = _default_resolver) -> bool:
    """SSRF allowlist check for one URL. Never raises; returns ``False`` (and
    logs at DEBUG with just the host, never the full URL/path) for anything
    disallowed: non-http(s) scheme, missing/localhost/non-ASCII/IP-literal
    host, a non-standard port, or a host that resolves (via ``resolver``) to
    any address ``_is_disallowed_ip`` rejects. See the module docstring for
    the full rationale, including the IP-literal and non-ASCII refusals.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        _log.debug("evidence fetch refused: unparseable URL")
        return False
    if parts.scheme not in ("http", "https"):
        _log.debug("evidence fetch refused: scheme %r not http/https", parts.scheme)
        return False
    host = parts.hostname
    if host:
        host = host.rstrip(".")  # normalize a trailing FQDN root dot, e.g. "localhost."
    if not host:
        _log.debug("evidence fetch refused: no host")
        return False
    if not host.isascii():
        _log.debug("evidence fetch refused: non-ASCII host")
        return False
    default_port = 443 if parts.scheme == "https" else 80
    try:
        port = parts.port
    except ValueError:
        # Malformed port (out of range, non-numeric): refuse rather than
        # silently falling back to the scheme's default port.
        _log.debug("evidence fetch refused: malformed port for host %s", host)
        return False
    if (port or default_port) not in ALLOWED_PORTS:
        _log.debug("evidence fetch refused: non-standard port for host %s", host)
        return False
    if host.lower() == "localhost":
        _log.debug("evidence fetch refused: localhost host")
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        _log.debug("evidence fetch refused: bare IP literal host %s", host)
        return False
    try:
        addrs = resolver(host)
    except Exception:
        _log.debug("evidence fetch refused: DNS resolution failed for host %s", host)
        return False
    if not addrs or any(_is_disallowed_ip(a) for a in addrs):
        _log.debug("evidence fetch refused: disallowed resolved address for host %s", host)
        return False
    return True


def resolve_uri(uri: str) -> Optional[str]:
    """Best-effort resolution of an evidence URI to one fetchable http(s) URL.

    ``ipfs://<path>`` resolves through the first configured gateway;
    ``http(s)://`` URLs pass through unchanged; anything else (notably
    ``data:`` URIs, which are not fetched over HTTP) returns ``None``.
    """
    if uri.startswith("ipfs://"):
        return IPFS_GATEWAYS[0] + uri[len("ipfs://"):]
    if uri.startswith("http://") or uri.startswith("https://"):
        return uri
    return None


def _decode_data_uri(uri: str) -> Optional[str]:
    """Decode a ``data:`` URI's payload as text, capped at ``MAX_BYTES`` bytes.

    Supports both percent-encoded payloads and ``;base64,`` payloads (the
    ``base64`` marker is matched case-insensitively, per RFC 2397 examples in
    the wild using ``;BASE64,``). Returns ``None`` if the URI has no ``,``
    separator or the base64 payload is invalid.
    """
    header, sep, payload = uri[len("data:"):].partition(",")
    if not sep:
        return None
    if header.lower().endswith(";base64"):
        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception:
            return None
    else:
        raw = unquote(payload).encode("utf-8", errors="replace")
    return raw[:MAX_BYTES].decode("utf-8", errors="replace")


def _fetch_url(url: str, session, resolver: Resolver, timeout, redirects_left: int = MAX_REDIRECTS
                ) -> Optional[str]:
    """Fetch one http(s) URL through the SSRF guard, following up to
    ``redirects_left`` 3xx hops manually (re-checking each ``Location``), and
    reading at most ``MAX_BYTES`` bytes of the (final) response body.

    The response is always closed (in a ``finally``) before this call returns
    or recurses into the next redirect hop, regardless of which path (success,
    redirect, HTTP error, or a body-read failure) was taken.
    """
    if not _is_safe_url(url, resolver):
        return None
    sess = session or requests.Session()
    try:
        r = sess.get(url, timeout=timeout, stream=True, allow_redirects=False,
                      headers={"user-agent": USER_AGENT})
    except Exception:
        _log.debug("evidence fetch failed for %s", url, exc_info=True)
        return None
    next_url: Optional[str] = None
    text: Optional[str] = None
    try:
        if 300 <= r.status_code < 400:
            loc = r.headers.get("location") if r.headers else None
            if not loc or redirects_left <= 0:
                return None
            next_url = urljoin(url, loc)
        else:
            r.raise_for_status()
            raw = r.raw.read(MAX_BYTES, decode_content=True)
            raw = raw[:MAX_BYTES]  # belt-and-suspenders: don't trust a non-conforming stream
            text = raw.decode("utf-8", errors="replace")
    except Exception:
        _log.debug("evidence fetch failed reading body for %s", url, exc_info=True)
        return None
    finally:
        r.close()
    if next_url is not None:
        return _fetch_url(next_url, session, resolver, timeout, redirects_left - 1)
    return text


def http_fetch_text(uri: str, session=None, resolver: Resolver = _default_resolver,
                     timeout=DEFAULT_TIMEOUT) -> Optional[str]:
    """Fetch and decode text content for one evidence URI, or ``None`` on any
    failure/refusal (never raises -- this satisfies ``classify``'s
    ``fetch_text`` contract of handling its own errors).

    Handles ``data:`` URIs directly, tries every ``IPFS_GATEWAYS`` entry in
    order for ``ipfs://`` URIs (falling through to the next on failure), and
    fetches ``http(s)://`` URLs through the SSRF-guarded, redirect-bounded,
    size-capped ``_fetch_url``. A non-``str`` or empty ``uri`` returns ``None``
    immediately.
    """
    if not isinstance(uri, str) or not uri:
        return None
    if uri.startswith("data:"):
        return _decode_data_uri(uri)
    if uri.startswith("ipfs://"):
        tail = uri[len("ipfs://"):]
        for gateway in IPFS_GATEWAYS:
            text = _fetch_url(gateway + tail, session, resolver, timeout)
            if text is not None:
                return text
        return None
    url = resolve_uri(uri)
    if url is None:
        return None
    return _fetch_url(url, session, resolver, timeout)


def _classify_uri(uri: str, parties: set, fetch_text: Callable, tx_parties: Callable, session
                   ) -> tuple[int, str]:
    """Classify one URI, returning ``(level, note)``.

    ``note`` is ``"unfetchable"`` when ``fetch_text`` returned ``None`` (the
    URI could not be retrieved at all -- a later re-run can target these
    specifically), ``""`` when it was retrieved (regardless of what level that
    yielded), and ``"fetch-error"`` if ``classify`` itself raised (defensive:
    ``fetch_text``'s contract is to never raise, but a bug elsewhere in
    ``classify``, e.g. in ``tx_parties`` handling, should still degrade
    gracefully rather than abort the batch).
    """
    fetched_none = False

    def _fetch(u):
        nonlocal fetched_none
        text = fetch_text(u, session=session)
        if text is None:
            fetched_none = True
        return text

    try:
        level = classify(uri, _fetch, tx_parties, parties)
    except Exception:
        _log.error("classify_all: classify failed for %s", uri, exc_info=True)
        return 1, "fetch-error"
    return level, ("unfetchable" if fetched_none else "")


def classify_all(store: Store, fetch_text: Callable = http_fetch_text,
                  tx_parties: Callable[[str], Optional[set]] = lambda h: None,
                  log_every: int = 500, workers: int = 8) -> int:
    """Classify every distinct URI referenced by ``feedback`` that is not yet in
    the evidence cache, and persist each result via ``store.upsert_evidence``.

    ``parties`` passed to ``classify`` for each URI is the union of every
    rater address (``feedback.client``) and agent-owner address that used it.

    The pending ``(uri, parties)`` rows are read into a list up front (a
    ~34k-row batch is fine in memory), then classified concurrently across
    ``workers`` threads (``concurrent.futures.ThreadPoolExecutor``) -- each
    URI's fetch+classify (``_classify_uri``, including its own per-URI
    exception guard) runs in a worker thread. ``store.upsert_evidence`` is
    called ONLY from this (the main) thread, as each future completes via
    ``as_completed``: ``Store`` wraps a single ``sqlite3`` connection opened
    with the default ``check_same_thread=True`` and is not safe to write from
    multiple threads. ``workers=1`` behaves like the previous sequential
    implementation (results identical; the order of store writes is not
    guaranteed either way).

    Each worker thread lazily creates and reuses its own ``requests.Session``
    (via thread-local storage) for every ``fetch_text`` call it makes --
    sessions are never shared across threads, since a single
    ``requests.Session`` is not guaranteed safe for concurrent use. Every
    created session is closed once all URIs have been processed.

    Logs progress at INFO every ``log_every`` URIs *completed* (regardless of
    completion order); ``log_every <= 0`` disables progress logging entirely
    (also guards against a ``ZeroDivisionError`` from ``% log_every``).
    Returns the number of URIs processed.
    """
    # feedback.client and agents.owner are 0x-hex addresses, which never
    # contain a comma, so GROUP_CONCAT's default "," separator can't collide
    # with an address value and corrupt the split-back-apart below.
    q = """SELECT f.feedback_uri, GROUP_CONCAT(DISTINCT f.client), GROUP_CONCAT(DISTINCT a.owner)
           FROM feedback f LEFT JOIN agents a ON a.agent_id=f.agent_id
           WHERE f.feedback_uri<>'' AND NOT EXISTS (SELECT 1 FROM evidence_cache e WHERE e.uri=f.feedback_uri)
           GROUP BY f.feedback_uri"""
    # Read every pending row up front (rather than streaming the cursor): the
    # correlated NOT EXISTS sub-select re-reads evidence_cache on every row,
    # which is only safe to interleave with writes when nothing is written
    # back until the whole pending set has been captured -- concurrent
    # workers writing mid-scan (as they complete, out of order) could
    # otherwise race the cursor's own re-evaluation of NOT EXISTS.
    rows = store.conn.execute(q).fetchall()

    thread_local = threading.local()
    sessions: list = []
    sessions_lock = threading.Lock()

    def _thread_session():
        sess = getattr(thread_local, "session", None)
        if sess is None:
            sess = requests.Session()
            thread_local.session = sess
            with sessions_lock:
                sessions.append(sess)
        return sess

    def _classify_one(uri: str, parties: set):
        session = _thread_session()
        return _classify_uri(uri, parties, fetch_text, tx_parties, session)

    processed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_uri = {}
            for uri, clients, owners in rows:
                parties = {p for p in (clients or "").split(",") + (owners or "").split(",") if p}
                future = executor.submit(_classify_one, uri, parties)
                future_to_uri[future] = uri
            for future in as_completed(future_to_uri):
                uri = future_to_uri[future]
                level, note = future.result()
                store.upsert_evidence(uri, level, note)
                processed += 1
                if log_every > 0 and processed % log_every == 0:
                    _log.info("classify_all: processed %d URIs", processed)
    finally:
        for sess in sessions:
            sess.close()
    return processed
