"""Fetch evidence URIs (http/ipfs/data) and classify every distinct URI into the cache.

Evidence URIs come from on-chain data written by arbitrary, untrusted parties, so
the fetcher here is a security boundary, not just an HTTP client:

- **SSRF guard** (``_is_safe_url``): only ``http``/``https`` on the standard ports
  (80/443) to a host that is not ``localhost``, not an IP literal, and does not
  *resolve* (via an injectable ``resolver``) to any private/loopback/link-local/
  multicast/reserved/unspecified address. Refusal never raises -- it just yields
  ``None`` from the fetch, which ``robustrep.evidence.classify`` treats as
  "unfetchable" (level 1), the conservative outcome.
- **Bounded manual redirects**: automatic redirects are disabled
  (``allow_redirects=False``); up to ``MAX_REDIRECTS`` 3xx hops are followed by
  hand, re-running the SSRF guard against each ``Location`` before following it.
- **Bounded size/time**: at most ``MAX_BYTES`` bytes are read per response
  (regardless of any ``Content-Length`` claim -- we just want the beginning), and
  every request uses a ``(connect, read)`` timeout.

``classify_all`` walks every URI referenced by ``feedback`` rows that is not yet
in the evidence cache, classifies it, and persists the level -- one bad URI (an
unexpected exception out of ``classify``) is caught, logged, and recorded as
level 1 rather than aborting the whole batch.
"""
from __future__ import annotations

import base64
import ipaddress
import logging
import socket
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


def _default_resolver(host: str) -> list[str]:
    """Resolve ``host`` to its IP addresses via the system resolver."""
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def _is_disallowed_ip(ip_str: str) -> bool:
    """True if ``ip_str`` is not a literal IP, or is one that must never be
    fetched from a server processing untrusted URIs (private, loopback,
    link-local, multicast, reserved or unspecified)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified)


def _is_safe_url(url: str, resolver: Resolver = _default_resolver) -> bool:
    """SSRF allowlist check for one URL. Never raises; returns ``False`` (and
    logs at DEBUG with just the host, never the full URL/path) for anything
    disallowed: non-http(s) scheme, missing/localhost/IP-literal host, a
    non-standard port, or a host that resolves (via ``resolver``) to any
    private/loopback/link-local/multicast/reserved/unspecified address."""
    try:
        parts = urlsplit(url)
    except ValueError:
        _log.debug("evidence fetch refused: unparseable URL")
        return False
    if parts.scheme not in ("http", "https"):
        _log.debug("evidence fetch refused: scheme %r not http/https", parts.scheme)
        return False
    host = parts.hostname
    if not host:
        _log.debug("evidence fetch refused: no host")
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
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_disallowed_ip(str(literal)):
            _log.debug("evidence fetch refused: disallowed literal IP host %s", host)
            return False
        return True
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

    Supports both percent-encoded payloads and ``;base64,`` payloads. Returns
    ``None`` if the URI has no ``,`` separator or the base64 payload is invalid.
    """
    header, sep, payload = uri[len("data:"):].partition(",")
    if not sep:
        return None
    if header.endswith(";base64"):
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
    reading at most ``MAX_BYTES`` bytes of the (final) response body."""
    if not _is_safe_url(url, resolver):
        return None
    sess = session or requests.Session()
    try:
        r = sess.get(url, timeout=timeout, stream=True, allow_redirects=False,
                      headers={"user-agent": USER_AGENT})
    except Exception:
        _log.debug("evidence fetch failed for %s", url, exc_info=True)
        return None
    if 300 <= r.status_code < 400:
        loc = r.headers.get("location") if r.headers else None
        if not loc or redirects_left <= 0:
            return None
        return _fetch_url(urljoin(url, loc), session, resolver, timeout, redirects_left - 1)
    try:
        r.raise_for_status()
        raw = r.raw.read(MAX_BYTES, decode_content=True)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        _log.debug("evidence fetch failed reading body for %s", url, exc_info=True)
        return None


def http_fetch_text(uri: str, session=None, resolver: Resolver = _default_resolver,
                     timeout=DEFAULT_TIMEOUT) -> Optional[str]:
    """Fetch and decode text content for one evidence URI, or ``None`` on any
    failure/refusal (never raises -- this satisfies ``classify``'s
    ``fetch_text`` contract of handling its own errors).

    Handles ``data:`` URIs directly, tries every ``IPFS_GATEWAYS`` entry in
    order for ``ipfs://`` URIs (falling through to the next on failure), and
    fetches ``http(s)://`` URLs through the SSRF-guarded, redirect-bounded,
    size-capped ``_fetch_url``.
    """
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


def classify_all(store: Store, fetch_text: Callable = http_fetch_text,
                  tx_parties: Callable[[str], Optional[set]] = lambda h: None,
                  log_every: int = 500) -> int:
    """Classify every distinct URI referenced by ``feedback`` that is not yet in
    the evidence cache, and persist each result via ``store.upsert_evidence``.

    ``parties`` passed to ``classify`` for each URI is the union of every
    rater address (``feedback.client``) and agent-owner address that used it.

    Opens one ``requests.Session`` for the whole call (connection-pool reuse
    across every URI) and passes it to ``fetch_text`` as the ``session``
    keyword. A per-URI exception escaping ``classify`` (``fetch_text`` itself
    must never raise per its contract, but a defensive catch is kept here
    anyway) is logged at ERROR and recorded as level 1 with note
    ``"fetch-error"`` rather than aborting the batch -- a single bad URI must
    never abort a run over 100k URIs. Logs progress at INFO every
    ``log_every`` URIs. Returns the number of URIs processed.
    """
    q = """SELECT f.feedback_uri, GROUP_CONCAT(DISTINCT f.client), GROUP_CONCAT(DISTINCT a.owner)
           FROM feedback f LEFT JOIN agents a ON a.agent_id=f.agent_id
           WHERE f.feedback_uri<>'' AND NOT EXISTS (SELECT 1 FROM evidence_cache e WHERE e.uri=f.feedback_uri)
           GROUP BY f.feedback_uri"""
    rows = store.conn.execute(q).fetchall()
    session = requests.Session()
    try:
        for i, (uri, clients, owners) in enumerate(rows, 1):
            parties = {p for p in (clients or "").split(",") + (owners or "").split(",") if p}
            try:
                level, note = classify(uri, lambda u: fetch_text(u, session=session), tx_parties, parties), ""
            except Exception:
                _log.error("classify_all: classify failed for %s", uri, exc_info=True)
                level, note = 1, "fetch-error"
            store.upsert_evidence(uri, level, note)
            if i % log_every == 0:
                _log.info("classify_all: processed %d/%d URIs", i, len(rows))
    finally:
        session.close()
    return len(rows)
