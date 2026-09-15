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
- **The vetted host must be the connected host** (``_vetted_host`` /
  ``_parsers_agree`` / ``_canonical_url``): the guard vets what ``urlsplit``
  sees, but two further transforms sit between it and the socket, and each has
  been a full bypass of every check above:

  * ``urllib3.util.parse_url`` treats a backslash as an authority terminator,
    so ``http://127.0.0.1:6379\\@example.com/`` vets as ``example.com`` port 80
    and connects to ``127.0.0.1:6379``;
  * ``requests.PreparedRequest.prepare_url`` runs ``requote_uri()`` *after*
    this guard has run: it percent-DECODES unreserved characters in the
    authority, so ``http://169.254.169.25%34/`` vets as the (unresolvable-
    looking) host ``169.254.169.25%34`` and connects to ``169.254.169.254``;
    and it percent-ENCODES others (``" ` ^ < > { | }`` and friends), again
    connecting somewhere other than what was vetted.

  Four layers close this, most load-bearing first: the vetted hostname must
  match ``_HOST_ALLOWED`` -- LDH labels and dots only, no ``%``, no underscore,
  nothing ``requote_uri`` would rewrite -- which is what guarantees the rebuilt
  host survives requests' requoting *byte for byte*; a URL containing
  whitespace/C0/C1 controls anywhere, or an authority containing ``\\``, ``@``,
  whitespace or a control character, is refused outright
  (``_URL_FORBIDDEN``/``_AUTHORITY_FORBIDDEN``); the scheme/host/port urllib3
  reads must match ``urlsplit``'s (defense in depth against the *next* parser
  differential); and the request is issued against a URL *rebuilt* from the
  vetted components (``_canonical_url``), never the attacker's raw string.
  ``test_vetted_host_is_the_host_requests_would_connect_to`` pins the
  end-to-end invariant through a real ``requests`` ``PreparedRequest``.
  ``_URL_FORBIDDEN`` also refuses a literal space anywhere in a URL or a
  ``Location`` -- including in a path or query, where a browser would just
  encode it. That costs some legitimate traffic (the URI is recorded
  unfetchable, level 1) and is deliberate: what is vetted and what is sent
  stay byte-identical.
- **Bounded manual redirects**: automatic redirects are disabled
  (``allow_redirects=False``); up to ``MAX_REDIRECTS`` 3xx hops are followed by
  hand, re-running the SSRF guard against each ``Location`` before following it.
  The *raw* header value is screened for whitespace/control characters first:
  ``urljoin`` (via ``urlsplit``) strips TAB/CR/LF before the guard would ever
  see them, quietly turning ``http://ot\\rher.example/x`` into an allowed URL
  rather than refusing it.
  Every response (200, 3xx, 4xx/5xx, or one that fails mid-read) is closed
  before the function returns or recurses into the next hop.
- **Bounded size/time**: at most ``MAX_BYTES`` bytes are read per response
  (regardless of any ``Content-Length`` claim -- we just want the beginning),
  clamped a second time after the read as a belt-and-suspenders check against a
  non-conforming stream, and every request uses a ``(connect, read)`` timeout.
  A decompression bomb is bounded by that read itself: urllib3 2.x caps the
  *decoded* output of ``raw.read(amt, decode_content=True)`` at ``amt``, so no
  more than ``MAX_BYTES`` decompressed bytes are ever materialized. Requests
  additionally ask for ``accept-encoding: identity`` as belt-and-braces -- a
  hostile server is free to ignore that header and gzip anyway, which is why it
  is not the control the bound rests on.

``classify_all`` walks every URI referenced by ``feedback`` rows that is not yet
in the evidence cache, classifies it, and persists the level -- one bad URI (an
unexpected exception out of ``classify``) is caught, logged, and recorded as
level 1 rather than aborting the whole batch.

**H3 (security review): tx-hash lookup budget.** ``robustrep.evidence.classify``
already caps lookups *per URI* at ``MAX_TX_LOOKUPS_PER_URI``, but that alone
does not bound a whole ``fetch`` run: an attacker can post many distinct
evidence URIs, each packed with hashes up to that per-URI cap, and still
multiply out to a large number of RPC calls across the batch. ``classify_all``
additionally shares one ``_LookupBudget`` across every URI and every worker
thread for the run (``max_total_lookups``, default 20,000): once it is spent,
the wrapped ``tx_parties`` stops calling the underlying function at all and
returns ``None`` for every further hash, which ``classify`` treats as
"unverified" (its documented, conservative degrade path) rather than raising.
Exactly one WARNING is logged for the whole run, the moment the budget is
exhausted. A URI classified while the budget was exhausted may therefore hold
a false negative (level 2 instead of the 3 a later, unchecked hash would have
verified) -- such a URI is persisted with note ``"lookup-budget"`` instead of
an ordinary result, and ``Store.pending_uris(include_notes=("lookup-budget",))``
(used by ``classify_all`` itself) treats it as still pending, so the next
``fetch`` run -- with its own fresh budget -- reclassifies it rather than
leaving a starved, possibly-wrong level cached forever.

**Known limitation -- DNS rebinding (the residual unmitigated gap in v0.1):**
the guard above resolves the host and checks *those* addresses, but the actual
connection is made by ``requests``/``urllib3``, which resolves the host *again*
independently. A DNS-rebinding attacker (answering a public IP on the first
lookup and a private one on the second, timed to land between the two
resolutions) can pass the guard and cause a GET to a private address.

The blast radius is *not* nil: the fetched text is consumed by
``robustrep.evidence.classify``, which scans it for a transaction hash and for
task-id JSON keys, and the outcome is persisted per URI as ``(level, note)``.
A bypass is therefore a four-state oracle about the target --
``(1, "unfetchable")`` (no response), ``(1, "")`` (responded, no markers),
``(2, "")`` (response contained a 0x-hash or task-id key) and ``(3, "")``
(a hash in the response involves the rater/owner addresses) -- not a blind
request. That is precisely why the parser cross-check and canonical rebuild
above exist: "the response body is never returned to the caller" is not a
sufficient reason to tolerate a reachable bypass. Proper mitigation of the
remaining DNS gap (resolve once, then fetch via a pinned-IP transport adapter
with the original hostname kept for TLS SNI/Host) is scheduled for v0.2; see
``test_dns_rebinding_not_mitigated_in_v0_1`` in the test suite, which documents
this with a ``strict=True`` xfail so it starts failing (as a reminder to update
docs/tests) the moment it's actually fixed.
"""
from __future__ import annotations

import base64
import ipaddress
import logging
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Sequence
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import requests
from urllib3.util import parse_url as _urllib3_parse_impl

from ..evidence import classify
from ..store import Store

IPFS_GATEWAYS = ("https://ipfs.io/ipfs/", "https://dweb.link/ipfs/")
MAX_BYTES = 200_000
MAX_REDIRECTS = 3
ALLOWED_PORTS = (80, 443)
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 10
DEFAULT_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)
USER_AGENT = "robustrep/0.1"
# H3: total eth_getTransactionByHash calls allowed across one classify_all
# run, shared by every URI and every worker thread -- see the module
# docstring and `_LookupBudget`.
DEFAULT_MAX_TOTAL_LOOKUPS = 20_000

_log = logging.getLogger(__name__)

Resolver = Callable[[str], Sequence[str]]

# Characters that must never appear in a URL's authority. `\\` and `@` are where
# `urlsplit` and urllib3's `parse_url` disagree about where the authority ends
# (see the module docstring); whitespace and C0/C1 controls are request-smuggling
# material. An evidence URI needs none of them, so they are refused rather than
# normalized -- normalizing would just pick a winner between the two parsers.
_AUTHORITY_FORBIDDEN = re.compile(r"[\\@\s\x00-\x1f\x7f-\x9f]")
# The same character classes anywhere in the URL: `urlsplit` silently strips
# tab/CR/LF *before* parsing, so an authority-only check never sees them.
_URL_FORBIDDEN = re.compile(r"[\s\x00-\x1f\x7f-\x9f]")
# What a vetted hostname may contain: LDH labels (letters/digits/hyphen) joined
# by dots, no leading/trailing dot or hyphen. This is an allowlist on purpose --
# `requests` requotes the URL *after* this guard runs, and every character it
# would rewrite (`%`, quotes, backticks, `^`, braces, ...) is simply not in the
# set, so the vetted host and the connected host cannot diverge. Underscores are
# refused as collateral: they are not LDH, and are not worth an exception.
_HOST_ALLOWED = re.compile(r"\A[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\Z")

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


def _urllib3_parse(url: str):
    """Module-level indirection over ``urllib3.util.parse_url``.

    Named (rather than called through the import directly) so the cross-parser
    check below can be exercised against a parser that deliberately disagrees:
    the differential this guards against is by definition one the *installed*
    urllib3 may not currently exhibit, and the guard must not depend on that.
    """
    return _urllib3_parse_impl(url)


def _canonical_url(url: str) -> str:
    """Rebuild ``url`` from the components the guard vetted.

    A request must never carry the attacker's raw string: anything the guard's
    parser ignored or normalized away is something a parser further down the
    stack might still act on. The rebuilt URL has a lower-cased scheme and
    host with the FQDN root dot and a redundant default port dropped, an
    explicit ``"/"`` path, and no fragment (fragments are never sent on the
    wire anyway -- keeping one would only preserve bytes to smuggle).

    Only ever called on a URL ``_is_safe_url`` accepted, so the host is a
    plain ASCII hostname -- never a bracketed IP literal, which this would not
    re-bracket.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").rstrip(".").lower()
    default_port = 443 if scheme == "https" else 80
    netloc = host if parts.port in (None, default_port) else f"{host}:{parts.port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def _vetted_host(parts) -> Optional[str]:
    """The hostname of an already-split URL, or ``None`` (logged at DEBUG,
    host only) when it is one this fetcher must never touch: missing,
    non-ASCII, outside the ``_HOST_ALLOWED`` LDH allowlist, ``localhost`` (any
    trailing FQDN root dot stripped first), or a bare IP literal.

    The allowlist is the load-bearing one: it is what makes the vetted host
    byte-identical to the host ``requests`` will connect to after its own
    ``requote_uri`` pass (see the module docstring). The non-ASCII check runs
    first only to give that common case its own reason in the log.
    """
    host = parts.hostname
    if host:
        host = host.rstrip(".")  # normalize a trailing FQDN root dot, e.g. "localhost."
    if not host:
        _log.debug("evidence fetch refused: no host")
        return None
    if not host.isascii():
        _log.debug("evidence fetch refused: non-ASCII host")
        return None
    if not _HOST_ALLOWED.match(host):
        _log.debug("evidence fetch refused: host outside the LDH allowlist %r", host)
        return None
    if host.lower() == "localhost":
        _log.debug("evidence fetch refused: localhost host")
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host
    _log.debug("evidence fetch refused: bare IP literal host %r", host)
    return None


def _parsers_agree(url: str, scheme: str, host: str, port: int) -> bool:
    """True if urllib3 reads the same scheme/host/port out of ``url`` as
    ``urlsplit`` did (``scheme``/``host``/``port``, already normalized by the
    caller, ``port`` with the scheme default applied).

    Defense in depth, not the primary control: the *last* transform before the
    socket is ``requests.PreparedRequest.prepare_url`` -> ``requote_uri``, not
    either of these parsers, and what keeps its output identical to the vetted
    host is ``_HOST_ALLOWED`` in ``_vetted_host``. What this check adds is
    cheap insurance against the *next* parser differential (the backslash case
    in the module docstring was one): where two parsers read one string
    differently, refuse rather than pick a winner. A URL urllib3 cannot parse
    at all is refused for the same reason.
    """
    try:
        u3 = _urllib3_parse(url)
    except Exception:
        _log.debug("evidence fetch refused: urllib3 cannot parse URL for host %r", host)
        return False
    u3_host = (u3.host or "").rstrip(".").strip("[]").lower()
    default_port = 443 if scheme == "https" else 80
    return ((u3.scheme or "").lower() == scheme
            and u3_host == host.lower()
            and (u3.port or default_port) == port)


def _is_safe_url(url: str, resolver: Resolver = _default_resolver) -> bool:
    """SSRF allowlist check for one URL. Never raises; returns ``False`` (and
    logs at DEBUG with just the host, never the full URL/path) for anything
    disallowed: non-http(s) scheme, a forbidden character in the URL or its
    authority, missing/localhost/non-ASCII/IP-literal host, a non-standard
    port, a scheme/host/port urllib3 reads differently than ``urlsplit``, or a
    host that resolves (via ``resolver``) to any address ``_is_disallowed_ip``
    rejects. See the module docstring for the full rationale.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        _log.debug("evidence fetch refused: unparseable URL")
        return False
    if parts.scheme not in ("http", "https"):
        _log.debug("evidence fetch refused: scheme %r not http/https", parts.scheme)
        return False
    if _URL_FORBIDDEN.search(url) or _AUTHORITY_FORBIDDEN.search(parts.netloc):
        _log.debug("evidence fetch refused: forbidden character in URL or authority")
        return False
    host = _vetted_host(parts)
    if host is None:
        return False
    default_port = 443 if parts.scheme == "https" else 80
    try:
        # Malformed port (out of range, non-numeric): refuse rather than
        # silently falling back to the scheme's default port.
        port = parts.port or default_port
    except ValueError:
        _log.debug("evidence fetch refused: malformed port for host %r", host)
        return False
    if port not in ALLOWED_PORTS:
        _log.debug("evidence fetch refused: non-standard port for host %r", host)
        return False
    if not _parsers_agree(url, parts.scheme, host, port):
        _log.debug("evidence fetch refused: parser disagreement for host %r", host)
        return False
    try:
        addrs = resolver(host)
    except Exception:
        _log.debug("evidence fetch refused: DNS resolution failed for host %r", host)
        return False
    if not addrs or any(_is_disallowed_ip(a) for a in addrs):
        _log.debug("evidence fetch refused: disallowed resolved address for host %r", host)
        return False
    return True


def _ipfs_tail(uri: str) -> Optional[str]:
    """The content path of an ``ipfs://`` URI, or ``None`` if it must not be
    pasted onto a gateway prefix.

    A gateway URL is built by concatenation, so the tail decides what path on
    the gateway *host* is fetched: ``ipfs://../api/v0/id`` would escape
    ``/ipfs/`` and hit the gateway's own API, and a leading ``/`` would address
    its root. Refused, bluntly: empty, absolute, any ``..`` anywhere, and any
    ``%`` at all. The last one matters because checking the raw tail is not
    sufficient on its own -- ``requests``'s ``requote_uri`` decodes ``%2e``
    back to ``.`` after this runs, so ``ipfs://%2e%2e/%2e%2e/api/v0/id`` would
    reach the wire as ``/ipfs/../../api/v0/id``. CIDs are base58/base32 and
    paths under them need no escaping (whitespace is refused anyway), so
    nothing legitimate is lost.
    """
    tail = uri[len("ipfs://"):]
    if not tail or tail.startswith("/") or ".." in tail or "%" in tail:
        _log.debug("evidence fetch refused: unsafe ipfs path %r", tail)
        return None
    return tail


def resolve_uri(uri: str) -> Optional[str]:
    """Best-effort resolution of an evidence URI to one fetchable http(s) URL.

    ``ipfs://<path>`` resolves through the first configured gateway (for a
    path ``_ipfs_tail`` accepts); ``http(s)://`` URLs pass through unchanged;
    anything else (notably ``data:`` URIs, which are not fetched over HTTP)
    returns ``None``.
    """
    if uri.startswith("ipfs://"):
        tail = _ipfs_tail(uri)
        return None if tail is None else IPFS_GATEWAYS[0] + tail
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


def _next_hop(url: str, loc: str) -> Optional[str]:
    """Absolute URL for a ``Location`` header value, or ``None`` if it must not
    be followed at all.

    The *raw* header value is screened before ``urljoin`` touches it:
    ``urljoin`` (via ``urlsplit``) strips TAB/CR/LF, so a hostile ``Location``
    would otherwise be silently normalized into a different, allowed URL --
    ``http://ot\\rher.example/x`` becoming ``http://other.example/x`` -- instead
    of being refused. The joined result is still re-run through
    ``_is_safe_url`` by the hop that follows it.
    """
    if _URL_FORBIDDEN.search(loc):
        _log.debug("evidence fetch refused: forbidden character in Location header")
        return None
    return urljoin(url, loc)


def _fetch_url(url: str, session, resolver: Resolver, timeout, redirects_left: int = MAX_REDIRECTS
                ) -> Optional[str]:
    """Fetch one http(s) URL through the SSRF guard, following up to
    ``redirects_left`` 3xx hops manually (each ``Location`` screened by
    ``_next_hop``, then re-guarded by this function), and reading at most
    ``MAX_BYTES`` bytes of the (final) response body.

    The URL requested is always the canonical one rebuilt from the vetted
    components (``_canonical_url``), never the raw input string -- on redirect
    hops too, since each hop re-enters here. ``accept-encoding: identity`` is
    belt-and-braces only: the bound on decompressed bytes comes from urllib3
    2.x capping the decoded output of ``raw.read(amt, decode_content=True)``.

    A ``session`` of ``None`` means "make your own": that session is then this
    function's to close (in a ``finally``, and after any redirect hops have
    reused it), since a leaked ``requests.Session`` leaks its connection pool.
    A caller-supplied session is never closed here -- ``classify_all`` reuses
    one per worker thread across many URIs and closes them itself.
    """
    if not _is_safe_url(url, resolver):
        return None
    url = _canonical_url(url)
    if session is not None:
        return _fetch_vetted(url, session, resolver, timeout, redirects_left)
    sess = requests.Session()
    try:
        return _fetch_vetted(url, sess, resolver, timeout, redirects_left)
    finally:
        sess.close()


def _fetch_vetted(url: str, sess, resolver: Resolver, timeout, redirects_left: int
                   ) -> Optional[str]:
    """Issue the request for an ``url`` that ``_fetch_url`` has already guarded
    and canonicalized, with ``sess`` as the (caller- or self-owned) transport.

    Split out of ``_fetch_url`` purely so that function can own a session's
    lifetime in a ``finally`` without nesting this whole body one level deeper.
    Redirect hops re-enter through ``_fetch_url`` (guard, then canonicalize)
    carrying the same session.

    The response is always closed (in a ``finally``) before this returns or
    recurses into the next hop, on every path (success, redirect, HTTP error or
    body-read failure).
    """
    try:
        r = sess.get(url, timeout=timeout, stream=True, allow_redirects=False,
                      headers={"user-agent": USER_AGENT, "accept-encoding": "identity"})
    except Exception:
        _log.debug("evidence fetch failed for %r", url, exc_info=True)
        return None
    next_url: Optional[str] = None
    text: Optional[str] = None
    try:
        if 300 <= r.status_code < 400:
            loc = r.headers.get("location") if r.headers else None
            # No Location, no budget left, or a refused one: all mean "stop".
            next_url = _next_hop(url, loc) if loc and redirects_left > 0 else None
            if next_url is None:
                return None
        else:
            r.raise_for_status()
            raw = r.raw.read(MAX_BYTES, decode_content=True)
            raw = raw[:MAX_BYTES]  # belt-and-suspenders: don't trust a non-conforming stream
            text = raw.decode("utf-8", errors="replace")
    except Exception:
        _log.debug("evidence fetch failed reading body for %r", url, exc_info=True)
        return None
    finally:
        r.close()
    if next_url is not None:
        return _fetch_url(next_url, sess, resolver, timeout, redirects_left - 1)
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
        tail = _ipfs_tail(uri)
        if tail is None:
            return None
        for gateway in IPFS_GATEWAYS:
            text = _fetch_url(gateway + tail, session, resolver, timeout)
            if text is not None:
                return text
        return None
    url = resolve_uri(uri)
    if url is None:
        return None
    return _fetch_url(url, session, resolver, timeout)


def _classify_uri(uri: str, parties: set, fetch_text: Callable, tx_parties: Callable, session,
                   was_starved: Optional[Callable[[], bool]] = None) -> tuple[int, str]:
    """Classify one URI, returning ``(level, note)``.

    ``note`` is ``"unfetchable"`` when ``fetch_text`` returned ``None`` (the
    URI could not be retrieved at all -- a later re-run can target these
    specifically); ``"fetch-error"`` if ``classify`` itself raised (defensive:
    ``fetch_text``'s contract is to never raise, but a bug elsewhere in
    ``classify``, e.g. in ``tx_parties`` handling, should still degrade
    gracefully rather than abort the batch); ``"lookup-budget"`` (H3) when the
    URI *was* fetched and classified but ``was_starved()`` reports that the
    shared per-run tx lookup budget ran out partway through its hash list --
    the returned level may be a false negative (a verifying hash could have
    sat past the point the budget cut off at), so it is persisted as
    retryable rather than as an ordinary final result (see
    ``Store.pending_uris``); ``""`` for a normal, fully-checked result.

    ``was_starved``, when given, must be a callable that reports (and resets)
    whether the *this-thread* budget-wrapped ``tx_parties`` was refused for
    lack of budget since it was last checked -- see ``_LookupBudget.wrap``.
    It is called once before ``classify`` (discarding any stale flag left by
    a previous URI this same worker thread handled) and once after.
    """
    fetched_none = False

    def _fetch(u):
        nonlocal fetched_none
        text = fetch_text(u, session=session)
        if text is None:
            fetched_none = True
        return text

    if was_starved is not None:
        was_starved()

    try:
        level = classify(uri, _fetch, tx_parties, parties)
    except Exception:
        _log.error("classify_all: classify failed for %r", uri, exc_info=True)
        if was_starved is not None:
            was_starved()
        return 1, "fetch-error"
    if fetched_none:
        return level, "unfetchable"
    if was_starved is not None and was_starved():
        return level, "lookup-budget"
    return level, ""


class _LookupBudget:
    """Thread-safe counter that wraps ``tx_parties`` to cap the total number
    of RPC lookups spent across one ``classify_all`` run (H3, see module
    docstring).

    ``spend()`` is called once per candidate hash, from whichever worker
    thread ``robustrep.evidence.classify`` happens to be running in, so the
    decrement-and-check has to be atomic -- a plain ``if self._remaining > 0``
    followed by a decrement would let two threads both pass the check for the
    last unit of budget. ``_exhausted`` is a lock-free fast path checked
    before taking the lock at all: it only ever flips ``False`` -> ``True``
    (never back), so every call after the one that actually exhausts the
    budget can skip the lock entirely, at the cost of nothing worse than a
    handful of calls racing the exhausting one itself still taking the slow,
    always-correct locked path. Once exhausted, ``wrap``'s callable stops
    calling the underlying ``tx_parties`` entirely and returns ``None`` (the
    same "unverified" signal a real RPC miss would give), and exactly one
    WARNING is logged for the whole run -- logged *outside* the lock (holding
    a lock across a logging call would serialize every other thread's
    lookups behind a slow handler for no benefit), by whichever call is the
    one that drove ``_remaining`` to zero.
    """

    def __init__(self, total: int):
        self._total = max(total, 0)
        self._remaining = self._total
        self._lock = threading.Lock()
        self._exhausted = False
        self.spent = 0

    def spend(self) -> bool:
        """Atomically consume one unit of budget; True if one was available,
        False if the budget is (now, or already) exhausted."""
        if self._exhausted:
            return False
        just_exhausted = False
        with self._lock:
            if self._remaining <= 0:
                self._exhausted = True
                return False
            self._remaining -= 1
            self.spent += 1
            if self._remaining <= 0:
                self._exhausted = True
                just_exhausted = True
        if just_exhausted:
            _log.warning(
                "evidence: tx lookup budget of %d exhausted; remaining hashes treated as unverified",
                self._total)
        return True

    def wrap(self, tx_parties: Callable[[str], Optional[set]]
             ) -> tuple[Callable[[str], Optional[set]], Callable[[], bool]]:
        """Return ``(budgeted_tx_parties, was_starved)``.

        ``budgeted_tx_parties`` is a ``tx_parties``-shaped callable that
        spends one unit of this budget per call and, once exhausted, calls
        ``tx_parties`` no further (returning ``None`` instead).

        ``was_starved()`` reports, and resets, whether *this calling thread's*
        most recent run of ``budgeted_tx_parties`` calls included one refused
        for lack of budget. It is thread-local rather than a single shared
        flag because ``classify_all`` gives each worker thread one URI to
        classify at a time (never two interleaved ``classify()`` calls on the
        same thread), so "since this thread last checked" is exactly "during
        the URI this thread is classifying right now" -- which is precisely
        what ``_classify_uri`` needs to decide whether *that* URI's result is
        retryable.
        """
        local = threading.local()

        def _budgeted(h: str) -> Optional[set]:
            if not self.spend():
                local.starved = True
                return None
            return tx_parties(h)

        def _was_starved() -> bool:
            starved = getattr(local, "starved", False)
            local.starved = False
            return starved

        return _budgeted, _was_starved


def classify_all(store: Store, fetch_text: Callable = http_fetch_text,
                  tx_parties: Callable[[str], Optional[set]] = lambda h: None,
                  log_every: int = 500, workers: int = 8,
                  max_total_lookups: int = DEFAULT_MAX_TOTAL_LOOKUPS) -> int:
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

    ``max_total_lookups`` (H3, see module docstring) bounds the total number
    of ``tx_parties`` calls across the *whole run*, shared by every URI and
    every worker thread via a single ``_LookupBudget``; the number actually
    spent (out of the clamped total, never negative even if a caller passes
    one) is logged at INFO once the run completes, in a ``finally`` so it is
    logged even if the run itself raised. A URI whose classification was cut
    short by the budget is persisted with note ``"lookup-budget"`` rather
    than ``""``/an ordinary level (see ``_classify_uri``), and is picked back
    up as pending on the *next* call via
    ``store.pending_uris(include_notes=("lookup-budget",))`` below -- a fresh
    run gets a fresh budget, so a URI that was starved once is not starved
    forever.

    Returns the number of URIs processed.
    """
    # "lookup-budget"-noted URIs are retried (see Store.pending_uris); a
    # URI cached with any other note (a normal result, "unfetchable",
    # "fetch-error") is not retried by classify_all itself.
    rows = store.pending_uris(include_notes=("lookup-budget",))

    budget = _LookupBudget(max_total_lookups)
    budgeted_tx_parties, was_starved = budget.wrap(tx_parties)

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
        return _classify_uri(uri, parties, fetch_text, budgeted_tx_parties, session, was_starved)

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
        _log.info("classify_all: tx lookup budget spent %d/%d", budget.spent, budget._total)
    return processed
