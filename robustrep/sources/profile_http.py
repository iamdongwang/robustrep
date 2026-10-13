"""Hygiene shared by every rater-profile HTTP client.

``rater_profile`` (Etherscan V2, Blockscout v2) and ``alchemy_profile``
(Alchemy) each talk to a different third party for the same one fact -- the
first incoming transaction of an address -- and each of them must apply the
same three rules from the security review, in exactly the same way, so that a
reviewer who has checked one client has checked them all:

- **M1** -- the provider's API key must never reach a log or an exception.
  Every helper here that builds an error message names only a fixed label
  (``"Etherscan"``, ``"Blockscout"``, ``"Alchemy"``), the target address and a
  numeric HTTP status -- never the request URL or params (Etherscan carries
  the key in the query string, Alchemy in the URL *path*), never the raw
  ``requests`` exception (whose text embeds the URL), and never any body
  text. ``status_code`` is the only thing a client may extract from a
  ``requests`` exception.
- **M2** -- response bodies are decoded through ``capped_json`` (over
  ``http_util.read_json_capped``) with an ``MAX_PROFILE_RESPONSE_BYTES`` cap,
  never ``requests``' unbounded ``Response.json()``; an oversized body is a
  non-retryable ``RuntimeError`` naming only the label, the address and the
  limit.
- **M4** -- a first-transaction timestamp that is not a plausible epoch
  second (``<= 0`` or past ``MAX_PLAUSIBLE_TS``) is refused via
  ``implausible``/``plausible_ts`` and logged once through ``warn_bad_ts``
  rather than cached: one bad value in the raters table makes
  ``robustrep.sybil._validate_meta`` raise on every later ``score``/``report``
  run.
- **L1** -- the funder (the ``from`` of the first incoming transaction) is
  cached verbatim and never revisited, so it must be a real address:
  ``funder_address`` accepts only a ``str`` matching ``^0x[0-9a-fA-F]{40}$``
  (lowercased on the way in, the same shape ``base_erc8004._addr`` enforces
  for raters) and refuses anything else -- ``null``, a number, an object, a
  malformed or over-long string -- with one WARNING. The client then treats
  the entry exactly like an implausible timestamp (M4): ``first_tx`` returns
  ``None``, so the address is recorded as unprofiled (no funder, no
  timestamp) rather than carrying an attacker-chosen value into the store.

The retry loop itself is shared as well: ``fetch_with_retries`` is the one
request/retry loop every client runs its HTTP call through, so the throttle
sleep, the response close, the transient/permanent classification, the
``Retry-After`` handling and the terminal error exist exactly once. Its
vocabulary: ``is_transient_http`` decides what is worth another attempt,
``retry_after_seconds`` reads (and caps) a 429's ``Retry-After``, ``RETRY`` /
``REJECTED`` are the sentinels a client's callbacks answer with, and
``raise_exhausted`` is the single terminal error for an address whose every
attempt failed. ``ProfilePlanError`` is the one non-retryable failure every
client may raise that ``enrich_raters`` treats specially: the configured
plan/key/network cannot serve this chain at all, so it aborts on the first
address instead of logging the same rejection once per rater.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

import requests

from .http_util import ResponseTooLarge, read_json_capped, short_for_log

logger = logging.getLogger(__name__)

# Largest profile-API response body ever buffered (M2). A first-transaction
# lookup asks for one page of one address's transactions; anything past this
# is a broken or hostile endpoint, not a bigger answer worth parsing.
MAX_PROFILE_RESPONSE_BYTES = 8 * 1024 * 1024

# Upper bound on a plausible first-transaction timestamp: 2100-01-01T00:00:00Z
# (M4). Anything above this -- or at/below zero -- is a third-party bug or an
# attempt to poison the raters cache, not a real epoch second.
MAX_PLAUSIBLE_TS = 4102444800

# The one shape a cached funder may have (L1): the lowercase form of what
# ``base_erc8004._addr`` enforces for rater addresses, mixed case accepted on
# input because explorers return checksum-cased addresses.
_FUNDER_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# Longest backoff sleep honored from a 429 response's Retry-After header,
# in seconds -- caps a misbehaving/adversarial server from stalling a run.
RETRY_AFTER_CAP = 60.0

# Longest exponential-backoff sleep between attempts, in seconds
# (``min(2 ** attempt, BACKOFF_CAP)``).
BACKOFF_CAP = 30

# Per-attempt socket timeout handed to ``requests`` by every client, in seconds.
REQUEST_TIMEOUT = 30

# Sentinel an ``on_response``/``on_http_error`` callback returns to
# ``fetch_with_retries`` to mean "transient -- try again", as opposed to a
# real result (which may legitimately be ``None``). Not an exception, so the
# throttle/backoff bookkeeping stays in the one loop.
RETRY = object()

# Sentinel an ``on_http_error`` callback returns to mean "no special
# handling for this status -- raise the default permanent-rejection error".
REJECTED = object()


class ProfilePlanError(RuntimeError):
    """The configured plan, key or network of a rater-profile source cannot
    serve this chain (Etherscan's free plan does not cover Base; an Alchemy
    app whose key is rejected or whose Base network is not enabled).
    Non-retryable: no number of retries or different address fixes this,
    only a different plan or a different ``--profile-source``.
    ``rater_profile.enrich_raters`` treats this specially -- it aborts on the
    first address instead of failing every one -- see its docstring."""


def warn_bad_ts(address: str, value) -> None:
    """One WARNING for a first-transaction timestamp this pipeline refuses to
    cache (M4). Both the address and the offending value are attacker-influenced,
    so both are logged with ``%r``."""
    logger.warning("profile_http: implausible first-tx timestamp for %r: %r - leaving address unprofiled",
                    address, short_for_log(value))


def funder_address(value, address: str) -> Optional[str]:
    """``value`` as a lowercase ``0x`` + 40-hex address, or ``None`` (with one
    WARNING) when it is not a string of that exact shape (L1). Both the
    address and the offending value are attacker-influenced, so both are
    logged with ``%r`` and the value is bounded through ``short_for_log``.
    A ``None`` here must make the client's ``first_tx`` return ``None`` --
    the same outcome as an implausible timestamp (M4)."""
    if isinstance(value, str) and _FUNDER_RE.fullmatch(value):
        return value.lower()
    logger.warning("profile_http: funder for %r is not an address: %r - leaving address unprofiled",
                    address, short_for_log(value))
    return None


def implausible(ts: Optional[int]) -> bool:
    """True for a timestamp that must never reach ``Store.upsert_rater``: one
    negative/zero or past ``MAX_PLAUSIBLE_TS`` wedges every later ``score``/
    ``report`` run (``robustrep.sybil._validate_meta`` raises on it, and there
    is no way to un-cache it short of ``fetch --reprofile-raters``)."""
    return ts is None or ts <= 0 or ts > MAX_PLAUSIBLE_TS


def plausible_ts(value, address: str) -> Optional[int]:
    """``value`` as epoch seconds, or ``None`` (with one WARNING) if it is
    unparseable or implausible -- see ``implausible`` (M4)."""
    try:
        ts = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: JSON's 1e400 is standards-conformant and decodes to
        # float("inf"), which int() refuses (I2).
        ts = None
    if implausible(ts):
        warn_bad_ts(address, value)
        return None
    return ts


def capped_json(r, address: str, label: str):
    """Decode a streamed response body, refusing to buffer more than
    ``MAX_PROFILE_RESPONSE_BYTES`` (M2).

    An oversized body raises ``RuntimeError`` immediately -- the same
    non-retryable, scrubbed error path every client already uses for a
    permanent HTTP rejection, naming only the label and the target address so
    neither the API key nor any body text can reach a log. A body that simply
    is not JSON still raises ``ValueError``, which callers keep treating as a
    transient failure."""
    try:
        return read_json_capped(r, MAX_PROFILE_RESPONSE_BYTES)
    except ResponseTooLarge:
        raise RuntimeError(f"{label} response for {address} exceeds the "
                           f"{MAX_PROFILE_RESPONSE_BYTES} byte limit") from None


def raise_exhausted(label: str, address: str, last_kind: str, last_status, retries: int):
    """Raise the terminal error for an address whose every attempt failed.

    ``last_kind`` names what the *last* attempt failed on: ``"json"`` (an
    HTTP 200 whose body never parsed as JSON) raises ``ValueError``;
    ``"body"`` (an HTTP 200 whose body the client's ``on_response`` judged
    transient -- a JSON-RPC rate-limit error, Etherscan's "Max rate limit
    reached") raises ``RuntimeError`` saying so, rather than a misleading
    "HTTP error (status n/a)"; ``"http"`` raises ``RuntimeError`` naming the
    last HTTP status, or ``n/a`` when the failures never produced a response.
    Shared by every client so they stay identical in the one respect that
    matters: the message carries only the source label, the address and a
    status code -- never the request URL or params (which carry the API key)
    and never any body text."""
    if last_kind == "json":
        raise ValueError(f"non-JSON {label} response for {address}")
    if last_kind == "body":
        raise RuntimeError(f"{label} body-level error for {address} after {retries} attempts")
    status_label = last_status if last_status is not None else "n/a"
    raise RuntimeError(f"{label} HTTP error for {address} after {retries} attempts "
                        f"(status {status_label})")


def status_code(e: requests.RequestException) -> Optional[int]:
    """HTTP status code carried by a ``requests`` exception, or ``None`` if
    the failure happened before any response was received (connection
    error, timeout). The only thing a client may extract from such an
    exception: its text embeds the request URL, hence the API key (M1)."""
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None) if resp is not None else None


def is_transient_http(code: Optional[int]) -> bool:
    """429 and 5xx are transient; so is no response at all (connection
    errors, timeouts); any other HTTP status is a permanent rejection."""
    if code is None:
        return True
    return code == 429 or 500 <= code < 600


def retry_after_seconds(e: requests.RequestException) -> Optional[float]:
    """Seconds named by the response's ``Retry-After`` header, capped at
    ``RETRY_AFTER_CAP``, or ``None`` if there is no response, no such
    header, or its value isn't a non-negative number. Only the numeric
    (seconds) form of the header is honored -- the HTTP-date form is
    ignored rather than trusted to `sleep()` against a possibly-skewed
    remote clock."""
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    value = resp.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, RETRY_AFTER_CAP)


def fetch_with_retries(label: str, address: str, do_request: Callable[[], object],
                       on_response: Callable[[dict], object], *, retries: int, gap: float,
                       sleep: Callable[[float], None],
                       on_http_error: Optional[Callable[[int, object], object]] = None):
    """The one request/retry loop behind every rater-profile client's lookup.

    ``do_request()`` performs the HTTP call (with ``stream=True`` -- M2) and
    returns the response; ``on_response(body)`` interprets a decoded JSON
    object body and returns the final value (``None`` is a legitimate
    result), returns ``RETRY`` for a body-level transient failure, or raises.
    ``on_http_error(code, response)``, when given, is consulted for a
    non-transient HTTP status (any 4xx but 429) *before* the default
    rejection: it returns a final value (Blockscout maps 404 to "unknown
    address"), ``RETRY`` to treat the status as transient after all,
    ``REJECTED`` to fall through to the default, or raises (Alchemy
    classifies a 401/403 body as a configuration error). ``response`` is the
    streamed response, still open, so the hook may read its body through
    ``capped_json``; it may be ``None`` when the request itself raised.

    What this loop owns, identically for every client:

    - the response is closed and the throttle ``sleep(gap)`` is paid in
      ``finally`` on **every** path -- success, retry, permanent rejection,
      callback exception, even ``do_request`` raising -- so a storm of
      errors cannot exceed the configured request rate;
    - classification via ``status_code``/``is_transient_http``: 429/5xx and
      no-response failures (connection error, timeout) are retried, any other
      4xx is rejected at once as ``RuntimeError(f"{label} HTTP {code} for
      {address}")`` raised ``from None``, so the ``requests`` exception (whose
      text embeds the URL, hence the key -- M1) is never chained; the hook is
      likewise called *outside* the ``except`` block, so nothing it raises
      chains it either;
    - backoff between attempts: a 429's numeric ``Retry-After`` (capped at
      ``RETRY_AFTER_CAP``) when present, else ``min(2 ** attempt,
      BACKOFF_CAP)``; no backoff after the final attempt;
    - a non-JSON body (``capped_json`` raising ``ValueError``) counts as
      transient of kind ``"json"``; an oversized one is the non-retryable
      ``RuntimeError`` ``capped_json`` raises; a body that is JSON but not an
      object raises ``ValueError`` at once (malformed, not transient);
    - ``last_status`` is reset when a later attempt fails at body level, so
      the terminal message never reports a status from an earlier attempt;
    - the terminal ``raise_exhausted`` once ``retries`` attempts are spent.

    Nothing raised here carries anything but ``label``, ``address`` and an
    integer status (M1).
    """
    last_status: Optional[int] = None
    last_kind = "http"
    for attempt in range(retries):
        transient = False
        retry_after: Optional[float] = None
        rejected: Optional[int] = None
        r = None
        try:
            try:
                r = do_request()
                r.raise_for_status()
            except requests.RequestException as e:
                # The status code is the only thing read off the exception:
                # its text embeds the request URL, hence the key (M1).
                code = status_code(e)
                if is_transient_http(code):
                    transient, last_status, last_kind = True, code, "http"
                    if code == 429:
                        retry_after = retry_after_seconds(e)
                else:
                    rejected = code
            else:
                try:
                    body = capped_json(r, address, label)
                except ValueError:
                    transient, last_kind, last_status = True, "json", None
                else:
                    if not isinstance(body, dict):
                        raise ValueError(f"{label} response is not a JSON object for {address}")
                    outcome = on_response(body)
                    if outcome is not RETRY:
                        return outcome
                    transient, last_kind, last_status = True, "body", None
            if rejected is not None:
                # Deliberately outside the `except` above: the requests
                # exception is no longer being handled, so neither the
                # default error nor anything the hook raises chains it.
                outcome = REJECTED if on_http_error is None else on_http_error(rejected, r)
                if outcome is REJECTED:
                    raise RuntimeError(f"{label} HTTP {rejected} for {address}") from None
                if outcome is not RETRY:
                    return outcome
                transient, last_status, last_kind = True, rejected, "http"
        finally:
            # The body is streamed (M2), so the response must be closed
            # explicitly -- on every path, including the error ones.
            if r is not None:
                r.close()
            # Rate-limit throttle: always paid, success or failure, so a
            # string of errors can't be used to blow past the configured rps.
            sleep(gap)

        if transient and attempt < retries - 1:
            # Honor a 429's Retry-After header (capped) over the usual
            # exponential backoff, when the server told us how long to wait.
            sleep(retry_after if retry_after is not None else min(2 ** attempt, BACKOFF_CAP))

    raise_exhausted(label, address, last_kind, last_status, retries)
