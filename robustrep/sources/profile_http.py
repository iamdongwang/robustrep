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

The retry vocabulary is shared too: ``is_transient_http`` decides what is
worth another attempt, ``retry_after_seconds`` reads (and caps) a 429's
``Retry-After``, and ``raise_exhausted`` is the single terminal error for an
address whose every attempt failed. ``ProfilePlanError`` is the one
non-retryable failure every client may raise that ``enrich_raters`` treats
specially: the configured plan/key/network cannot serve this chain at all, so
it aborts on the first address instead of logging the same rejection once per
rater.
"""
from __future__ import annotations

import logging
from typing import Optional

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

# Longest backoff sleep honored from a 429 response's Retry-After header,
# in seconds -- caps a misbehaving/adversarial server from stalling a run.
RETRY_AFTER_CAP = 60.0


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

    ``ValueError`` when the last attempt's body never parsed as JSON,
    ``RuntimeError`` otherwise (naming the last HTTP status, or ``n/a`` when
    the failures never produced a response). Shared by every client so they
    stay identical in the one respect that matters: the message carries only
    the source label, the address and a status code -- never the request URL
    or params (which carry the API key) and never any body text."""
    if last_kind == "json":
        raise ValueError(f"non-JSON {label} response for {address}")
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
