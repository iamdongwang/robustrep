"""First-seen time and funding source per rater, from Alchemy.

A third rater-profile source beside the two in ``rater_profile``, added
because the free ones stopped serving Base: Blockscout put its Base API behind
a Cloudflare challenge on 2026-10-09, and its developer portal states
"Starting October 1, 2026, Base, Polygon and ZkSync will no longer be
available on Free tier"; Etherscan's free plan never covered Base via the V2
API. Alchemy's free plan does -- 30M compute units (CU) per month at
300 CU/s, and ``alchemy_getAssetTransfers`` costs 120 CU per call, so the
sustainable rate is 300 / 120 = 2.5 requests/sec (``DEFAULT_ALCHEMY_RPS``)
and a month covers 250,000 lookups; the ~13,000 raters on Base profile in
about an hour and a half.

``AlchemyClient.first_tx`` asks, in one JSON-RPC call, for the earliest
*incoming external* transfer to the address (``alchemy_getAssetTransfers``
with ``toAddress``, ``category: ["external"]``, ``order: "asc"``,
``maxCount: "0x1"``, ``withMetadata`` for the block timestamp). Verified
2026-10-13 on 12 raters already profiled through Blockscout: the funder
address and the first-seen timestamp were identical for all 12 (12/12), so
the two definitions -- Blockscout's oldest ``filter=to`` transaction and
Alchemy's earliest incoming external transfer -- are treated as equivalent.

Mixed-source caveat: a store profiled partly through Blockscout (before
2026-10) and partly through Alchemy holds rows from both, and
``sync_state.rater_profile_mode`` records only the source of the *last* run.
That is acceptable precisely because the two definitions were verified
equivalent; a report that states ``rater profile mode: alchemy`` therefore
describes the method, not which rows came from which host.

Same duck-typed interface as the other clients -- ``first_tx(address)``, a
``source`` attribute (``"alchemy"``) and ``gap`` (seconds between requests,
read by the CLI for its ETA) -- so it drops into ``rater_profile.
enrich_raters`` and ``default_client`` unchanged. The remote is treated as
hostile, with the shared ``profile_http`` helpers applying the security
review's rules, plus one twist of Alchemy's own:

- **M1** -- the API key is the last path segment of the request URL
  (``.../v2/<key>``), not a query parameter or a header. Every ``requests``
  exception, and urllib3's DEBUG request line, therefore carries it. The
  constructor installs ``http_util.install_key_redaction`` on urllib3's
  loggers, and this module never interpolates a ``requests`` exception, the
  URL or any response text into a raised message or a log record: only
  ``profile_http.status_code`` is read off an exception, and a JSON-RPC
  error's server-chosen message is logged at DEBUG only, length-bounded
  through ``http_util.short_for_log``. ``AlchemyConfigError`` carries a fixed
  message rather than the server's.
- **M2** -- the body is streamed and decoded through
  ``profile_http.capped_json`` (8 MiB cap); an oversized body is a
  non-retryable ``RuntimeError`` naming only the limit and the address.
- **M4** -- the block timestamp is parsed by ``blockscout_util.
  parse_timestamp`` (Alchemy serves the same ISO8601 shape,
  ``2026-02-22T21:16:19.000Z``) and refused via ``profile_http.implausible``
  rather than cached when it is not a plausible epoch second.
- **L1** -- the funder (``from``) must be a ``0x`` + 40-hex address
  (``profile_http.funder_address``); anything else is refused with one
  WARNING and ``first_tx`` returns ``None``, exactly as for an implausible
  timestamp -- the address is recorded as unprofiled, never with a
  server-chosen string as its funder.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import requests

from . import blockscout_util
from .http_util import install_key_redaction, short_for_log
from .profile_http import (
    REJECTED,
    REQUEST_TIMEOUT,
    RETRY,
    ProfilePlanError,
    capped_json,
    fetch_with_retries,
    funder_address,
    implausible,
    warn_bad_ts,
)

logger = logging.getLogger(__name__)

# Alchemy's Base mainnet JSON-RPC endpoint; the API key is appended as the
# last path segment (``{ALCHEMY_BASE_URL}/{key}``) -- see M1 in the module
# docstring for what that means for logging.
ALCHEMY_BASE_URL = "https://base-mainnet.g.alchemy.com/v2"

# Free-plan arithmetic: 300 compute units per second, 120 per
# alchemy_getAssetTransfers call -> 2.5 calls/sec sustained.
ALCHEMY_FREE_CU_PER_SECOND = 300
ALCHEMY_CU_PER_GET_ASSET_TRANSFERS = 120
DEFAULT_ALCHEMY_RPS = ALCHEMY_FREE_CU_PER_SECOND / ALCHEMY_CU_PER_GET_ASSET_TRANSFERS

# Substrings (case-insensitive) of a JSON-RPC error message meaning the key
# or the app's network configuration is rejected -- a configuration problem
# no retry or different address fixes (AlchemyConfigError). Observed live
# 2026-10-13: "Must be authenticated!" (HTTP 401, bad key) and "BASE_MAINNET
# is not enabled for this app. ..." (HTTP 403).
_CONFIG_REJECTED = ("authenticated", "not enabled", "api key")
# Substrings meaning the app is over its compute-unit budget right now --
# transient, retried with backoff like an HTTP 429. Deliberately not the bare
# "rate": it matched "enumerate", "operate", "moderate" and sent permanent
# errors around the retry loop.
_RATE_LIMITED = ("rate limit", "rate-limit", "throughput", "capacity")
# JSON-RPC error code Alchemy uses for the same condition.
_RATE_LIMIT_CODE = 429
# HTTP status for a rejected key: a configuration error whatever the body
# says (observed live 2026-10-13 with the "Must be authenticated!" body).
_HTTP_UNAUTHENTICATED = 401

# The one message AlchemyConfigError ever carries: fixed, so no server text
# (and nothing that could embed the key) reaches a log through it.
CONFIG_REJECTED_MESSAGE = ("Alchemy rejected the API key or the Base network is not enabled for this "
                           "app -- check ALCHEMY_API_KEY and that Base Mainnet is enabled for the app, "
                           "or choose another --profile-source")


class AlchemyConfigError(ProfilePlanError):
    """Alchemy rejected the API key, or the app behind it does not have the
    Base network enabled. Non-retryable: only a different key, a different
    app configuration or a different ``--profile-source`` fixes this. A
    ``ProfilePlanError``, so ``enrich_raters`` aborts on the first address
    instead of logging the same rejection once per rater. The message is
    always ``CONFIG_REJECTED_MESSAGE`` -- never the server's text."""


class AlchemyClient:
    """Client for Alchemy's ``alchemy_getAssetTransfers`` "first incoming
    transfer" lookup on Base.

    One POST per address to ``{base_url}/{api_key}`` with a JSON-RPC body
    asking for the earliest external transfer *to* the address (see
    ``_request_body``). The reply's first ``transfers`` entry gives the
    funder (``from``), the block (``blockNum``, hex) and the block timestamp
    (``metadata.blockTimestamp``, ISO8601); an empty ``transfers`` list
    means the address has never received an external transfer and yields
    ``None``.

    Transient failures -- HTTP 429/5xx, connection/timeout errors, a
    non-JSON body, or a JSON-RPC error body that names a rate/capacity limit
    (or carries code 429) -- are retried up to ``retries`` times with
    exponential backoff (``min(2**attempt, 30)`` seconds; a 429 whose
    response carries a numeric ``Retry-After`` header uses that instead,
    capped at 60s -- see ``profile_http.retry_after_seconds``), then raise
    ``RuntimeError`` (or ``ValueError`` when the last body never parsed as
    JSON) naming the address and the last HTTP status. HTTP 401, or any
    other non-transient 4xx whose body is a JSON-RPC error rejecting the key
    or the network, raises ``AlchemyConfigError`` immediately (see
    ``_classify_http_error``); any other HTTP 4xx raises ``RuntimeError``
    immediately; a JSON-RPC error in an HTTP 200 body that rejects the key
    or the network raises ``AlchemyConfigError`` immediately; any other
    JSON-RPC error raises ``RuntimeError`` immediately. None of those
    consume a retry. A per-request throttle sleep (``gap = 1/rps``) runs
    after every attempt regardless of outcome (in ``finally``), same as the
    other clients, so the configured rate is respected even while erroring.

    Never logs or interpolates the raw exception, the request URL (whose
    path carries the API key) or any response text anywhere -- only a
    numeric HTTP status code and the target address appear in any error
    message this raises; a JSON-RPC error's server text appears only at
    DEBUG, length-bounded.
    """

    # Used by enrich_raters to label rater_profile_mode.
    source = "alchemy"

    def __init__(self, api_key: Optional[str], base_url: str = ALCHEMY_BASE_URL, session=None,
                 rps: float = DEFAULT_ALCHEMY_RPS, retries: int = 5,
                 sleep: Callable[[float], None] = time.sleep):
        if not api_key or not api_key.strip():
            raise ValueError("AlchemyClient requires an API key (ALCHEMY_API_KEY)")
        # M1: the key is in the URL path, and urllib3 logs the full request
        # line at DEBUG. Install the redaction filter before the first request.
        install_key_redaction(api_key)
        self.base_url = base_url
        # The one place the key is used. Never logged, never interpolated into
        # an exception -- see the module docstring.
        self._url = f"{base_url}/{api_key}"
        self.session = session or requests.Session()
        self.retries, self.sleep = retries, sleep
        self.gap = 1.0 / rps

    def first_tx(self, address: str) -> Optional[tuple[int, int, str]]:
        """Return ``(block, timestamp, funder_from_address)`` for the earliest
        incoming external transfer to ``address``, or ``None`` if it has none,
        its block timestamp is implausible (M4, logged once) or its ``from``
        is not an address (L1, logged once). Raises
        ``ValueError`` naming ``address`` for a malformed body -- not a JSON
        object, ``result`` not an object, ``transfers`` not a list, an entry
        missing ``blockNum``/``from``/``metadata.blockTimestamp`` or with a
        non-hex ``blockNum`` -- or, after retries, for a body that never
        parsed as JSON. See the class docstring for the full error
        contract."""
        address = address.lower()
        payload = _request_body(address)
        return fetch_with_retries(
            "Alchemy", address,
            lambda: self.session.post(self._url, json=payload, timeout=REQUEST_TIMEOUT, stream=True),
            lambda body: self._parse_body(body, address),
            retries=self.retries, gap=self.gap, sleep=self.sleep,
            on_http_error=lambda code, r: _classify_http_error(code, r, address))

    def _parse_body(self, body: dict, address: str):
        """Interpret one decoded JSON-RPC response body. Returns the
        ``first_tx`` result (a tuple, or ``None`` for "no transfers"), or
        ``RETRY`` for a transient body-level error. Raises ``ValueError``
        for a malformed body and ``RuntimeError``/``AlchemyConfigError`` for
        a non-transient JSON-RPC error (see ``first_tx``)."""
        error = body.get("error")
        if error is not None:  # ``"error": null`` beside a result is a success
            return _classify_error(error, address)
        result = body.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"Alchemy response result is not an object for {address}")
        transfers = result.get("transfers")
        if not isinstance(transfers, list):
            raise ValueError(f"Alchemy response 'transfers' is not a list for {address}")
        if not transfers:
            return None
        return _parse_transfer(transfers[0], address)


def _request_body(address: str) -> dict:
    """The JSON-RPC body for one lookup: the earliest (``order: asc``,
    ``maxCount: 1``) external transfer *to* ``address`` over the whole chain,
    with block metadata so the timestamp comes back in the same call.
    ``excludeZeroValue: false`` so a zero-value call to the address still
    counts as its first incoming transfer, matching Blockscout's
    ``filter=to`` definition (a contract deployment would not: it has
    ``to = null`` and can never match ``toAddress``)."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [{
            "fromBlock": "0x0",
            "toBlock": "latest",
            "toAddress": address,
            "category": ["external"],
            "order": "asc",
            "maxCount": "0x1",
            "withMetadata": True,
            "excludeZeroValue": False,
        }],
    }


def _error_kind(error, address: str) -> str:
    """``"rate"``, ``"config"`` or ``"other"`` for a JSON-RPC ``error``
    member of any shape. Rate limiting is checked *first*: Alchemy's own
    throttling text can read "API key rate limit exceeded", which names the
    key and so would otherwise be mistaken for a rejected one and abort the
    whole profiling step over a transient condition. The server's text is
    remote-chosen, so it is logged at DEBUG only, length-bounded, and never
    interpolated into anything raised."""
    code = error.get("code") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    low = message.lower() if isinstance(message, str) else ""
    logger.debug("alchemy_profile: JSON-RPC error for %s (code %r): %r",
                 address, short_for_log(code), short_for_log(message))
    if code == _RATE_LIMIT_CODE or any(s in low for s in _RATE_LIMITED):
        return "rate"
    if any(s in low for s in _CONFIG_REJECTED):
        return "config"
    return "other"


def _classify_error(error, address: str):
    """Map a JSON-RPC ``error`` member in an HTTP 200 body onto the retry
    contract (see ``_error_kind``): rate/capacity limits return ``RETRY``;
    key/network rejections raise ``AlchemyConfigError`` (fixed message);
    anything else raises ``RuntimeError`` naming only the address."""
    kind = _error_kind(error, address)
    if kind == "rate":
        return RETRY
    if kind == "config":
        raise AlchemyConfigError(CONFIG_REJECTED_MESSAGE)
    raise RuntimeError(f"Alchemy JSON-RPC error for {address}")


def _classify_http_error(code: int, r, address: str):
    """``fetch_with_retries``' ``on_http_error`` hook: decide whether a
    non-transient HTTP status is a configuration error. Alchemy answers a
    rejected key with HTTP 401 and a network the app has not enabled with
    HTTP 403, each carrying a JSON-RPC error body (observed live
    2026-10-13); without this, the first address would fail as a plain
    ``RuntimeError`` and ``enrich_raters`` would go on to fail every one of
    the ~13,000 others the same way.

    HTTP 401 raises ``AlchemyConfigError`` regardless of body -- and without
    reading it. Any other
    non-transient 4xx raises it when the body (streamed and capped like a
    success body -- M2) is a JSON-RPC error ``_error_kind`` calls
    ``"config"``; a body that is not JSON, not an object, or any other
    error returns ``REJECTED`` so the loop raises its fixed ``Alchemy HTTP
    {code} for {address}``. ``r`` may be ``None`` when the session itself
    raised. The server text, as always, reaches only the DEBUG log."""
    if code == _HTTP_UNAUTHENTICATED:
        # Decided on the status alone, *before* touching the body: a bad key is
        # a bad key whatever the server attached, and an oversized body must
        # not demote this abort into a per-address limit error (I-1).
        raise AlchemyConfigError(CONFIG_REJECTED_MESSAGE)
    body = None
    if r is not None:
        try:
            body = capped_json(r, address, "Alchemy")
        except ValueError:
            body = None  # not JSON: nothing to classify beyond the status
    error = body.get("error") if isinstance(body, dict) else None
    kind = _error_kind(error, address) if error is not None else "other"
    if kind == "config":
        raise AlchemyConfigError(CONFIG_REJECTED_MESSAGE)
    return REJECTED


def _parse_transfer(tx, address: str) -> Optional[tuple[int, int, str]]:
    """``(block, ts, funder)`` for one ``transfers`` entry, or ``None`` when
    its ``metadata.blockTimestamp`` is unparseable or implausible (M4 -- see
    ``profile_http.implausible``; the WARNING is ``warn_bad_ts``) or its
    ``from`` is not an address (L1 -- ``profile_http.funder_address``; the
    timestamp is checked first, so a refused entry logs one WARNING, not
    two). Raises ``ValueError`` naming only the address for an entry
    missing the fields every transfer has, or whose ``blockNum`` is not a
    hex string -- that is a malformed response, not a suspect value. The
    cause is chained for a debugger but deliberately not interpolated: it
    may quote server text."""
    try:
        block = int(tx["blockNum"], 16)
        raw_ts = tx["metadata"]["blockTimestamp"]
        raw_from = tx["from"]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed Alchemy transfer entry for {address}") from e
    epoch = blockscout_util.parse_timestamp(raw_ts)
    if epoch is None or implausible(epoch):
        warn_bad_ts(address, raw_ts)
        return None
    funder = funder_address(raw_from, address)
    return None if funder is None else (block, epoch, funder)
