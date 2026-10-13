"""First-seen time and funding source per rater.

Sybil clustering (see ``robustrep.sybil``) needs, per rater address, when it
first appeared on-chain and who funded it (the ``from`` of its first incoming
transfer) -- or, since 2026-10, who *controls* it: measured 2026-10-13 on live
Base data, 230/230 sampled unprofiled raters were smart-contract accounts that
never received a transfer at all, so ``enrich_raters`` runs a second pass over
them that reads the account's controlling key (``account_owner``) and stores it
in the same ``funder`` column, as ``owner:<address>``, with ``funder_kind =
"owner"``. The first pass is unchanged. Public RPC nodes expose no per-address index for the transfer, so two
HTTP clients live here (a third, Alchemy, lives in ``alchemy_profile`` and is
what ``default_client("auto", ...)`` picks whenever ``ALCHEMY_API_KEY`` is set
and no Etherscan key is -- since 2026-10 it is the only one whose free plan
still serves Base):

- **Blockscout** (``BlockscoutV2Client``): Base's free Blockscout v2 REST
  API (``GET /api/v2/addresses/{address}/transactions?filter=to``). No API
  key. This is the default (``--profile-source auto`` picks it when no
  Etherscan key is configured). The legacy Etherscan-compatible
  ``?module=account&action=txlist`` endpoint (``EtherscanClient.
  blockscout()``) is rate-banned -- measured 2026-09-14, it answers HTTP 429
  even at 1 request/sec after a short burst -- so it is kept only for
  callers that specifically need that shape, not as the default.
- **Etherscan V2** (``EtherscanClient(api_key)``/``client_from_env``): needs
  ``ETHERSCAN_API_KEY``. Etherscan's *free* plan does not cover Base via the
  V2 API -- it replies with a "Free API access is not supported for this
  chain" error on every address -- so an Etherscan key here only helps with a
  paid plan.

Rater profiles: Alchemy (free plan covers Base) whenever ``ALCHEMY_API_KEY``
is set; Blockscout v2 (keyless, but its free tier dropped Base on
2026-10-01) is the keyless fallback; Etherscan V2 requires a paid plan for
Base.

Without a client at all (``--profile-source none``), ``enrich_raters`` cannot
determine a funder, and rather than writing placeholder rows it leaves the
raters table untouched: ``robustrep.sybil.profiles_from_records`` already
falls back to the minimum feedback ``ts`` per rater when no meta row exists,
so an unprofiled address gets that same first-seen value for free, and --
unlike a written row -- stays visible to ``Store.distinct_clients()`` so a
*later* run with a real client can still profile it instead of being
permanently stuck with the approximate value. ``enrich_raters`` records which
mode ran via ``Store.set_sync("rater_profile_mode", ...)`` so downstream
reports can state which mode produced the profile.

Both clients treat their remote as hostile, per the security review. The
rules they share with every other rater-profile client (``alchemy_profile``
included) live in ``profile_http`` and are applied through its helpers:

- **M1** -- an Etherscan API key must never reach a log. The key already stays
  out of this module's own messages (``profile_http.status_code`` is the only
  thing read off a ``requests`` exception), but ``urllib3`` logs the whole
  request line (query string included) at DEBUG, so ``EtherscanClient``
  installs an ``http_util.install_key_redaction`` filter on urllib3's loggers.
- **M2** -- response bodies are read through ``profile_http.capped_json``
  (over ``http_util.read_json_capped``) with an 8 MiB cap instead of
  ``requests``' unbounded ``Response.json()``.
- **M4** -- a first-transaction timestamp that is not a plausible epoch second
  (``<= 0`` or past ``profile_http.MAX_PLAUSIBLE_TS``) is refused rather than
  cached (``profile_http.plausible_ts``/``implausible``): one bad value written
  to the raters table makes ``robustrep.sybil._validate_meta`` raise on every
  later ``score``/``report`` run.
- **L1** -- the funder (``from``) is cached verbatim and never revisited, so
  it must be a ``0x`` + 40-hex address (``profile_http.funder_address``);
  anything else is refused with one WARNING and the address is left
  unprofiled, exactly as for an implausible timestamp.
- **L5** -- Blockscout's ``next_page_params`` are echoed back as query params
  on the next request, so they are allowlisted (see
  ``blockscout_util.safe_next_page_params``) rather than forwarded verbatim.

``MAX_PLAUSIBLE_TS``, ``MAX_PROFILE_RESPONSE_BYTES`` and ``ProfilePlanError``
are re-exported here for callers that import them from this module. They are
*names*, not the values the clients read: the clients look them up in
``profile_http`` at call time, so a test that wants to lower
``MAX_PROFILE_RESPONSE_BYTES`` or ``MAX_PLAUSIBLE_TS`` must monkeypatch
``robustrep.sources.profile_http``, not this module.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from ..store import FUNDER_KIND_NONE, FUNDER_KIND_OWNER, FUNDER_KIND_TRANSFER, Store
from . import blockscout_util
from .account_owner import account_owners, owner_funder
from .alchemy_profile import AlchemyClient
from .base_erc8004 import BatchState
from .http_util import install_key_redaction
from .profile_http import (
    MAX_PLAUSIBLE_TS,
    MAX_PROFILE_RESPONSE_BYTES,
    REJECTED,
    REQUEST_TIMEOUT,
    RETRY,
    ProfilePlanError,
    fetch_with_retries,
    funder_address,
    implausible,
    plausible_ts,
    warn_bad_ts,
)
from .rpc import scrub_rpc_error

logger = logging.getLogger(__name__)

# Addresses per JSON-RPC batch in the owner pass (see ``account_owner.
# account_owners``). Not a unit of failure: every address's outcome is its own
# (a failed lookup leaves that one row NULL for the next run).
OWNER_PASS_BATCH = 50

ETHERSCAN_V2 = "https://api.etherscan.io/v2/api"
# Base's Blockscout instance, exposing an Etherscan-compatible `?module=
# account&action=txlist` endpoint with no API key and no `chainid` param
# (it's already scoped to Base). See EtherscanClient.blockscout().
#
# WARNING: measured 2026-09-14, this legacy endpoint is rate-banned -- it
# answers HTTP 429 even at 1 request/sec after a short burst. Use
# BlockscoutV2Client (the v2 REST API, stable at 2 req/s sustained) instead;
# EtherscanClient.blockscout() is kept only for callers that still need the
# Etherscan-compatible shape.
BLOCKSCOUT_BASE = "https://base.blockscout.com/api"
# Blockscout's v2 REST API base, e.g. GET {BLOCKSCOUT_V2_BASE}/addresses/
# {address}/transactions?filter=to. See BlockscoutV2Client.
BLOCKSCOUT_V2_BASE = "https://base.blockscout.com/api/v2"
BASE_CHAIN_ID = 8453
# Default request rate (Etherscan free-tier limit; also a reasonable default
# against Blockscout); shared by EtherscanClient's throttle and by callers
# (e.g. the CLI) that need to estimate wall-clock time for a run via
# `estimate_seconds` before it starts.
DEFAULT_RPS = 4.0

# Substrings (case-insensitive) of an error ``result``/``message`` that
# indicate the request itself is permanently rejected (bad key) -- retrying
# cannot help, so these raise immediately instead of consuming the retry
# budget.
_NON_RETRYABLE = ("api key",)
# Substrings meaning "this address genuinely has no transactions" -- not an
# error at all, so these return ``None`` immediately without retrying.
_NO_TX = ("no transactions found",)
# Substrings meaning the configured Etherscan plan doesn't cover this chain
# at all (e.g. the free plan does not include Base via the V2 API) -- a
# configuration problem no retry or different address will fix. See
# EtherscanPlanError.
_PLAN_UNSUPPORTED = ("free api access is not supported",)
# Substrings meaning the request address itself is malformed -- a caller bug,
# not a transient failure.
_INVALID_ADDRESS = ("invalid address",)

# Sentinel returned internally by ``BlockscoutV2Client._fetch_page`` to mean
# "HTTP 404 -- unknown address", distinguishing it from a real page body.
_NOT_FOUND = object()

# Blockscout v2 page requests always filter for incoming transactions (the
# ``to`` side) -- we only care who funded ``address``, not what it sent.
_BLOCKSCOUT_V2_FILTER = {"filter": "to"}


class EtherscanPlanError(ProfilePlanError):
    """The configured Etherscan API plan does not cover this chain (e.g. the
    free plan does not support Base via the V2 API). Non-retryable: no
    number of retries or different address fixes this, only a different
    plan or a different ``--profile-source``. A ``ProfilePlanError``, so
    ``enrich_raters`` treats it specially -- see its docstring."""


class EtherscanClient:
    """Minimal client for an Etherscan-compatible "first transaction" lookup.

    Works against both Etherscan V2 (``base_url=ETHERSCAN_V2``, the default)
    and Base's Blockscout instance (``EtherscanClient.blockscout()``), which
    exposes the same ``module=account&action=txlist`` shape with no API key.

    Looks up the earliest transaction for an address via
    ``module=account&action=txlist&sort=asc&offset=1``. Transient failures --
    HTTP 429/5xx, connection/timeout errors, a non-JSON body, or a rate-limit
    message -- are retried up to ``retries`` times with exponential backoff
    (a 429 whose response carries a ``Retry-After`` header uses that instead,
    capped at 60s -- see ``profile_http.retry_after_seconds``); any other HTTP 4xx, an
    invalid API key, an unsupported plan, or a malformed address raise
    immediately, without consuming a retry. A per-request throttle sleep
    (``1/rps``) runs after every attempt regardless of outcome, so the
    configured request rate is respected even while retrying or erroring.

    Never logs or interpolates the raw exception, response body, or request
    URL/params anywhere (all of those may carry the API key in the query
    string) -- only a numeric HTTP status code and the target address appear
    in any error message this raises.

    Profiling ``n`` addresses this way costs wall-clock time of roughly
    ``n / rps`` seconds, excluding retry backoff -- e.g. about 6.9 hours for
    100,000 addresses at the default 4 requests/sec (see ``estimate_seconds``).
    """

    def __init__(self, api_key: Optional[str] = None, chain_id: Optional[int] = BASE_CHAIN_ID,
                 base_url: str = ETHERSCAN_V2, session=None, rps: float = DEFAULT_RPS,
                 retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        self.key, self.chain_id, self.base_url = api_key, chain_id, base_url
        # M1: urllib3 logs the full request line (query string, hence the key)
        # at DEBUG. Install the redaction filter before the first request.
        install_key_redaction(api_key)
        self.session = session or requests.Session()
        self.retries, self.sleep = retries, sleep
        self.gap = 1.0 / rps
        # Used by enrich_raters to label rater_profile_mode ("blockscout" vs
        # "etherscan") without hardcoding a URL comparison there too.
        self.source = "blockscout" if base_url == BLOCKSCOUT_BASE else "etherscan"

    @classmethod
    def blockscout(cls, rps: float = 4.0, **kw) -> "EtherscanClient":
        """Build a client for Base's legacy Blockscout ``?module=account&
        action=txlist`` endpoint: no API key, no ``chainid`` param (the
        endpoint is already Base-only).

        LEGACY / RATE-BANNED: measured 2026-09-14, this endpoint answers
        HTTP 429 even at 1 request/sec after a short burst. Prefer
        ``BlockscoutV2Client`` (the v2 REST API, stable at 2 req/s
        sustained) -- ``default_client`` already does. Kept only for
        callers that specifically need the Etherscan-compatible shape."""
        return cls(base_url=BLOCKSCOUT_BASE, chain_id=None, api_key=None, rps=rps, **kw)

    def first_tx(self, address: str) -> Optional[tuple[int, int, str]]:
        """Return ``(block, timestamp, funder_from_address)`` for the earliest
        transaction of ``address``, or ``None`` if it genuinely has none yet.

        Raises ``RuntimeError`` naming ``address`` if every retry is exhausted
        on a transient error (HTTP 429/5xx, connection/timeout, a non-JSON
        body, or a rate-limit message), or immediately -- without retrying --
        for a non-retryable one: any other HTTP 4xx, or an invalid/rejected
        API key. Raises ``EtherscanPlanError`` (a ``profile_http.
        ProfilePlanError``, itself a ``RuntimeError``; ``enrich_raters``
        aborts on the first address for any ``ProfilePlanError``)
        immediately if the configured plan does not cover this chain (e.g.
        Etherscan's free plan on Base). Raises ``ValueError`` naming
        ``address`` for a malformed response: a body that isn't a JSON object
        (e.g. ``null`` or a bare list), a missing ``status`` field, a
        ``status: "1"`` body whose ``result`` isn't a list, a rejected
        (malformed) address, a transaction entry missing required fields, or
        (after retries) a body that never parses as JSON.
        """
        address = address.lower()
        params = self._params(address)
        return fetch_with_retries(
            "Etherscan", address,
            lambda: self.session.get(self.base_url, params=params, timeout=REQUEST_TIMEOUT, stream=True),
            lambda body: self._parse_body(body, address),
            retries=self.retries, gap=self.gap, sleep=self.sleep)

    def _params(self, address: str) -> dict:
        """Query params for one ``txlist`` lookup. The API key appears here and
        nowhere else in this class -- it must never reach a log message, an
        exception's text, or a logged URL (M1; ``http_util.install_key_redaction``
        covers the one place we do not control, urllib3's own request line)."""
        params = dict(module="account", action="txlist", address=address, page=1, offset=1, sort="asc")
        if self.chain_id is not None:
            params["chainid"] = self.chain_id
        if self.key:
            params["apikey"] = self.key
        return params

    def _parse_body(self, body: dict, address: str):
        """Interpret one decoded JSON response body. Returns the ``first_tx``
        result (a tuple, or ``None`` for "no transactions"), or ``RETRY`` for
        a transient body-level error. Raises ``ValueError``/``RuntimeError``
        (including ``EtherscanPlanError``) directly for non-transient errors
        (see ``first_tx``)."""
        if "status" not in body:
            raise ValueError(f"Etherscan response missing 'status' for {address}")
        status, result = body.get("status"), body.get("result")
        if status == "1":
            if not isinstance(result, list):
                raise ValueError(f"Etherscan response result is not a list for {address}")
            if not result:
                return None
            return self._parse_tx(result[0], address)
        if isinstance(result, list):
            return None  # status "0" with an (empty) list result: genuinely no tx.
        text = "" if result is None else str(result)
        message = str(body.get("message") or "")
        low, mlow = text.lower(), message.lower()
        if any(s in low or s in mlow for s in _NO_TX):
            return None
        if any(s in low or s in mlow for s in _PLAN_UNSUPPORTED):
            raise EtherscanPlanError(
                f"Etherscan plan does not support this chain for {address} -- upgrade to a paid "
                "Etherscan plan, or use --profile-source blockscout (free, no key) instead")
        if any(s in low or s in mlow for s in _INVALID_ADDRESS):
            raise ValueError(f"malformed address {address}: rejected as an invalid address format")
        if any(s in low or s in mlow for s in _NON_RETRYABLE):
            raise RuntimeError(f"Etherscan API Key rejected for {address}")
        return RETRY

    @staticmethod
    def _parse_tx(tx: dict, address: str) -> Optional[tuple[int, int, str]]:
        """``(block, ts, funder)`` for one Etherscan tx entry, or ``None`` when
        its ``timeStamp`` is unparseable or implausible (M4 -- see
        ``profile_http.plausible_ts``: refusing the profile keeps a poisoned value out of
        the raters cache, where it would wedge every later scoring run) or
        its ``from`` is not an address (L1 -- ``profile_http.funder_address``).
        Still raises ``ValueError`` for an entry missing the fields every tx
        has -- that is a malformed response, not a suspect value."""
        try:
            block = int(tx["blockNumber"])
            raw_ts = tx["timeStamp"]
            raw_from = tx["from"]
        except (KeyError, TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"malformed Etherscan tx entry for {address}") from e
        ts = plausible_ts(raw_ts, address)
        if ts is None:
            return None
        funder = funder_address(raw_from, address)
        return None if funder is None else (block, ts, funder)


class BlockscoutV2Client:
    """Client for Base's Blockscout v2 REST API "first transaction" lookup.

    Replaces ``EtherscanClient.blockscout()`` (the legacy Etherscan-compatible
    ``?module=account&action=txlist`` endpoint), which measured 2026-09-14 is
    rate-banned: it answers HTTP 429 even at 1 request/sec after a short
    burst. This client instead pages through
    ``GET {base_url}/addresses/{address}/transactions?filter=to``. Measured
    2026-09-14: this v2 endpoint sustains ~1 request/sec (the default
    ``rps``); at 2 requests/sec it starts answering HTTP 429 after a few
    hundred requests. A 429 whose response carries a ``Retry-After`` header
    is honored (see ``first_tx``) rather than the usual exponential backoff,
    and the default ``retries`` (5, up from ``EtherscanClient``'s 3) gives a
    run more room to absorb occasional transient 429s at the sustained rate.

    Same duck-typed interface as ``EtherscanClient`` -- a ``first_tx``
    method and a ``source`` attribute -- so it's a drop-in for
    ``enrich_raters``.

    Items come back newest-first, one page at a time; ``next_page_params``
    (when present) names the query params for the next page. The oldest
    incoming transaction -- what we want, since it names the funder -- is
    therefore the last item of the last page. Paging stops after
    ``max_pages`` (default 5) pages: if the address's incoming history is
    still not exhausted at that point, ``first_tx`` returns ``None`` rather
    than guessing from a partial history and risking recording the wrong
    funder (a DEBUG log records the truncation).

    An unknown address (HTTP 404) returns ``None``, same as "no
    transactions yet". A non-JSON body or one missing the ``items``/
    ``next_page_params`` shape raises ``ValueError`` naming the address.
    Transient failures on a single page fetch -- HTTP 429/5xx,
    connection/timeout errors, or a non-JSON body -- are retried up to
    ``retries`` times with exponential backoff (a 429 whose response carries
    a ``Retry-After`` header uses that instead, capped at 60s), then raise
    ``RuntimeError`` naming the address and status; any other HTTP 4xx
    raises immediately, without consuming a retry. A per-request throttle
    sleep (``1/rps``) runs after every attempt regardless of outcome (in
    ``finally``), same as ``EtherscanClient``.

    Never logs or interpolates the raw exception or response body anywhere
    -- only a numeric HTTP status code and the target address appear in any
    error message this raises.
    """

    def __init__(self, base_url: str = BLOCKSCOUT_V2_BASE, session=None, rps: float = 1.0,
                 retries: int = 5, max_pages: int = 5, sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url
        self.session = session or requests.Session()
        self.retries, self.max_pages, self.sleep = retries, max_pages, sleep
        self.gap = 1.0 / rps
        # Used by enrich_raters to label rater_profile_mode -- always
        # "blockscout" for this client (unlike EtherscanClient, which also
        # serves the Etherscan V2 base URL).
        self.source = "blockscout"

    def first_tx(self, address: str) -> Optional[tuple[int, int, str]]:
        """Return ``(block, timestamp, funder_from_address)`` for the
        earliest incoming transaction of ``address``, or ``None`` if it
        genuinely has none yet, the address is unknown (HTTP 404), or its
        history could not be fully paged within ``max_pages``. See the
        class docstring for the full error contract."""
        address = address.lower()
        url = f"{self.base_url}/addresses/{address}/transactions"
        params: dict = dict(_BLOCKSCOUT_V2_FILTER)
        last_items: list = []
        for _page_num in range(self.max_pages):
            body = self._fetch_page(url, params, address)
            if body is _NOT_FOUND:
                return None
            items = body.get("items")
            if not isinstance(items, list):
                raise ValueError(f"Blockscout response missing 'items' for {address}")
            last_items = items
            next_params = body.get("next_page_params")
            if not next_params:
                if not last_items:
                    return None
                return self._parse_tx(last_items[-1], address)
            safe = blockscout_util.safe_next_page_params(next_params, address)
            if not safe:
                # Every pagination key the server sent was rejected (L5).
                # Re-requesting page 1 would loop, and guessing the next page
                # would risk recording the wrong funder -- so stop here.
                return None
            params = dict(safe)
            params.update(_BLOCKSCOUT_V2_FILTER)

        logger.debug(
            "rater_profile: Blockscout history for %s not exhausted after %d page(s) - "
            "skipping rather than risk recording the wrong funder", address, self.max_pages)
        return None

    def _fetch_page(self, url: str, params: dict, address: str):
        """Fetch and JSON-decode one page through
        ``profile_http.fetch_with_retries``. Returns the decoded body (a
        dict), or the ``_NOT_FOUND`` sentinel for HTTP 404 (the one status
        this client maps to a value rather than an error). Raises
        ``ValueError``/``RuntimeError`` for other failures (see
        ``first_tx``'s docstring)."""
        return fetch_with_retries(
            "Blockscout", address,
            lambda: self.session.get(url, params=params, timeout=REQUEST_TIMEOUT, stream=True),
            lambda body: body,
            retries=self.retries, gap=self.gap, sleep=self.sleep,
            on_http_error=lambda code, r: _NOT_FOUND if code == 404 else REJECTED)

    @staticmethod
    def _parse_tx(tx: dict, address: str) -> Optional[tuple[int, int, str]]:
        """``(block, ts, funder)`` for one Blockscout tx entry, or ``None``
        when its ``timestamp`` is unparseable or implausible (M4 -- see
        ``_parse_blockscout_timestamp``) or its ``from.hash`` is not an
        address (L1 -- ``profile_http.funder_address``). Still raises
        ``ValueError`` for an entry missing the fields every tx has."""
        try:
            block = int(tx["block_number"])
            raw_ts = tx["timestamp"]
            raw_from = tx["from"]["hash"]
        except (KeyError, TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"malformed Blockscout tx entry for {address}") from e
        ts = _parse_blockscout_timestamp(raw_ts, address)
        if ts is None:
            return None
        funder = funder_address(raw_from, address)
        return None if funder is None else (block, ts, funder)


def _parse_blockscout_timestamp(ts, address: str) -> Optional[int]:
    """Epoch seconds for a Blockscout ISO8601 timestamp, or ``None`` (with one
    WARNING) if it is unparseable or implausible -- see ``profile_http.implausible`` (M4).
    The parsing itself lives in ``blockscout_util.parse_timestamp``; what is
    decided here is whether the result is a value we are willing to cache."""
    epoch = blockscout_util.parse_timestamp(ts)
    if epoch is None or implausible(epoch):
        warn_bad_ts(address, ts)
        return None
    return epoch


def client_from_env() -> Optional[EtherscanClient]:
    """Build an Etherscan V2 ``EtherscanClient`` from the ``ETHERSCAN_API_KEY``
    environment variable, or ``None`` if it is unset/blank (use
    ``EtherscanClient.blockscout()`` or the offline fallback instead)."""
    key = (os.environ.get("ETHERSCAN_API_KEY") or "").strip()
    return EtherscanClient(key) if key else None


def default_client(source: str, key: Optional[str], rps: Optional[float] = None,
                    retries: Optional[int] = None, alchemy_key: Optional[str] = None):
    """Build the rater-profile HTTP client for ``--profile-source source``,
    given ``key`` (an explicit ``--etherscan-key``/``ETHERSCAN_API_KEY``
    value, or ``None``) and ``alchemy_key`` (``ALCHEMY_API_KEY``, or
    ``None``; whitespace-only counts as absent).

    - ``"none"``: no client -- ``enrich_raters`` runs its offline fallback
      (writes no rows). ``rps``/``retries`` are ignored -- there is no
      client to configure.
    - ``"blockscout"``: ``BlockscoutV2Client()`` -- Base's Blockscout v2
      REST API, keyless. Since 2026-10 Blockscout's free tier no longer
      serves Base (the API sits behind a Cloudflare challenge), so this only
      works against a paid Blockscout plan or another instance. (The legacy
      ``EtherscanClient.blockscout()`` Etherscan-compatible endpoint is
      rate-banned -- see its docstring -- so it is not built here; construct
      it explicitly if needed.)
    - ``"etherscan"``: Etherscan V2 using ``key``. Raises ``ValueError`` if
      ``key`` is falsy -- Etherscan V2 always needs a key, and unlike
      ``"auto"`` this source was requested explicitly, so silently falling
      back would hide the missing configuration.
    - ``"alchemy"``: ``alchemy_profile.AlchemyClient(alchemy_key)`` --
      Alchemy's free plan covers Base. Raises ``ValueError`` if
      ``alchemy_key`` is blank, for the same reason as ``"etherscan"``.
    - ``"auto"`` (the CLI default), in precedence order: Etherscan V2 when
      ``key`` is given (presumably a paid plan, since the free plan doesn't
      cover Base -- see the module docstring); else Alchemy when
      ``alchemy_key`` is given; else ``BlockscoutV2Client()``.

    ``rps``/``retries``, when given (not ``None``), are forwarded to
    whichever client is built, overriding its own default request rate /
    retry count (``BlockscoutV2Client``'s 1.0 rps / 5 retries,
    ``EtherscanClient``'s ``DEFAULT_RPS`` (4.0) / 3 retries, or
    ``AlchemyClient``'s ``DEFAULT_ALCHEMY_RPS`` (2.5) / 5 retries). ``None``
    (the default) leaves the client's own default in place -- so a caller
    that never asks about ``rps``/``retries`` (e.g. an existing test double)
    gets exactly the previous behavior.

    Raises ``ValueError`` for any other ``source`` value.
    """
    kw: dict = {}
    if rps is not None:
        kw["rps"] = rps
    if retries is not None:
        kw["retries"] = retries
    alchemy_key = (alchemy_key or "").strip() or None

    if source == "none":
        return None
    if source == "blockscout":
        return BlockscoutV2Client(**kw)
    if source == "etherscan":
        if not key:
            raise ValueError("--profile-source etherscan requires --etherscan-key or ETHERSCAN_API_KEY")
        return EtherscanClient(key, **kw)
    if source == "alchemy":
        if not alchemy_key:
            raise ValueError("--profile-source alchemy requires ALCHEMY_API_KEY")
        return AlchemyClient(alchemy_key, **kw)
    if source == "auto":
        if key:
            return EtherscanClient(key, **kw)
        if alchemy_key:
            return AlchemyClient(alchemy_key, **kw)
        return BlockscoutV2Client(**kw)
    raise ValueError(f"unknown --profile-source: {source!r}")


def estimate_seconds(n_addresses: int, rps: float) -> float:
    """Rough wall-clock time (seconds) for ``enrich_raters`` to profile
    ``n_addresses`` at ``rps`` requests/sec, excluding retry backoff:
    ``n_addresses / rps`` -- e.g. ``estimate_seconds(100_000, 4.0)`` is about
    25,000s (~6.9h). For the CLI to print a progress estimate before a long
    enrichment run."""
    if rps <= 0:
        raise ValueError("rps must be > 0")
    return n_addresses / rps


@dataclass
class _Progress:
    """Bookkeeping for one ``enrich_raters`` run, shared by both passes:
    addresses that failed (listed in the final ``RuntimeError``), the
    distinct addresses settled (``written``; its size is the returned count
    -- an address pass 1 left NULL and pass 2 then settled is one address,
    not two) and client calls attempted -- the first-address abort rule for
    ``ProfilePlanError`` keys off the latter, so it holds whichever pass
    makes the run's first client call."""
    failures: list[str] = field(default_factory=list)
    written: set[str] = field(default_factory=set)
    attempted: int = 0

    @property
    def processed(self) -> int:
        return len(self.written)


def _lookup(progress: _Progress, client, address: str, **kw) -> tuple[bool, Optional[tuple[int, int, str]]]:
    """One ``client.first_tx(address, **kw)`` under the per-address failure
    rules: returns ``(True, result)`` on success, ``(False, None)`` after
    logging and recording a failure. A ``ProfilePlanError`` on the run's
    very first client call is re-raised instead (see ``enrich_raters``)."""
    try:
        got = client.first_tx(address, **kw)
    except ProfilePlanError:
        if progress.attempted == 0:
            raise
        logger.error("rater_profile: failed to enrich %s: plan does not support this chain", address)
        progress.failures.append(address)
        return False, None
    except Exception as e:
        logger.error("rater_profile: failed to enrich %s: %s", address, e)
        progress.failures.append(address)
        return False, None
    finally:
        progress.attempted += 1
    return True, got


def _transfer_pass(store: Store, client, addrs: list[str], progress: _Progress) -> None:
    """Pass 1: the first incoming transfer of every address in ``addrs``. A
    found funder is written with ``funder_kind="transfer"``; an address with
    no transfer gets a row with no funder and a NULL kind, which is exactly
    what ``Store.raters_needing_owner_lookup`` hands to pass 2."""
    found = 0
    for a in addrs:
        ok, got = _lookup(progress, client, a)
        if not ok:
            continue
        if got is None:
            store.upsert_rater(a, None, None)
        else:
            _, ts, funder = got
            store.upsert_rater(a, ts, funder, FUNDER_KIND_TRANSFER)
            found += 1
        progress.written.add(a)
    logger.info("rater_profile: pass 1 (first transfer): %d address(es), %d funder(s) found, %d failed",
                len(addrs), found, len(progress.failures))


def _settle(store: Store, address: str, owner: Optional[str], progress: _Progress) -> str:
    """Write pass 2's verdict for one address and name it: ``"owner"`` when a
    controlling key was found (stored as ``owner_funder(owner)``, the owner
    namespace -- see ``account_owner``), else ``"none"``.

    There is deliberately no second client call. Through the branch's first
    draft an address with no owner was re-asked of Alchemy with internal
    transfers admitted; a paymaster's gas advance *is* an internal transfer,
    so that retry could have bound passkey wallets to shared ERC-4337
    infrastructure (security review M3), and measured on 30 + 12 live
    addresses it changed nothing.

    The row's ``first_seen_ts`` is left exactly as it is (``Store.set_funder``
    updates only ``funder``/``funder_kind``): a legacy row may hold a measured
    timestamp, which pass 2 must not overwrite (I2), and a row pass 1 wrote
    this run holds NULL, which stays NULL on purpose -- the account's creation
    time is not what the owner call returns, and ``robustrep.sybil.
    profiles_from_records`` already falls back to the rater's earliest
    feedback time for a NULL. Writing that derivable approximation into the
    cache would make it indistinguishable from a measured value and violate
    the M4 rule that the raters table never holds an approximate timestamp."""
    if owner is not None:
        store.set_funder(address, owner_funder(owner), FUNDER_KIND_OWNER)
        progress.written.add(address)
        return "owner"
    store.set_funder(address, None, FUNDER_KIND_NONE)
    progress.written.add(address)
    return "none"


def _owner_pass(store: Store, rpc, progress: _Progress, state: Optional[BatchState] = None,
                echo: Optional[Callable[[str], None]] = None) -> None:
    """Pass 2 over ``Store.raters_needing_owner_lookup()``: the smart account's
    controlling key via ``account_owners`` (batched with per-entry errors, so
    one contract's answer never decides its neighbours') -> ``funder_kind=
    "owner"``; nothing -> ``"none"``, never retried (``_settle``; the
    ``client`` is not asked again). An address whose own lookup died on a transport error
    (an ``RpcError`` in its ``account_owners`` slot) is recorded failed by
    itself -- one ERROR line, its row stays NULL for the next run -- and
    every other address of the chunk is still settled. ``state`` is the
    ``fetch`` command's shared ``BatchState`` (as ``cli._step_owners`` gets
    it), so batching found unsupported earlier in the run is not rediscovered
    here. ``echo``, when given, is told ``"owner pass: N address(es)
    pending"`` before the pass starts (the CLI prints it)."""
    pending = store.raters_needing_owner_lookup()
    if echo is not None:
        echo(f"owner pass: {len(pending)} address(es) pending")
    if not pending:
        return
    counts = {"owner": 0, "none": 0}
    failed_before = len(progress.failures)
    state = state or BatchState()
    for i in range(0, len(pending), OWNER_PASS_BATCH):
        chunk = pending[i:i + OWNER_PASS_BATCH]
        owners = account_owners(rpc, chunk, batch_size=OWNER_PASS_BATCH, state=state)
        for a in chunk:
            found = owners.get(a)
            if isinstance(found, Exception):
                logger.error("rater_profile: owner lookup failed for %s: %s", a, scrub_rpc_error(found))
                progress.failures.append(a)
                continue
            counts[_settle(store, a, found, progress)] += 1
    logger.info("rater_profile: pass 2 (owner): %d address(es) (owner: %d, none: %d), %d failed",
                len(pending), counts["owner"], counts["none"], len(progress.failures) - failed_before)


def enrich_raters(store: Store, client: Optional["EtherscanClient | BlockscoutV2Client | AlchemyClient"],
                  addresses: Optional[list[str]] = None, *, rpc=None,
                  batch_state: Optional[BatchState] = None,
                  echo: Optional[Callable[[str], None]] = None) -> int:
    """Fill the raters table for every client address not yet profiled, in two
    passes.

    **Pass 1** (with ``client``) looks up each address's first incoming
    transfer: a funder is written with ``funder_kind="transfer"``; an address
    with no transfer gets a row with no funder and a NULL kind. A failure on
    one address does not lose the others: every address is attempted,
    successes are upserted, and if any failed a ``RuntimeError`` listing them
    (count + first 5) is raised at the end -- but only after the successful
    ones have been persisted and the mode recorded.

    **Pass 2** (with ``rpc``, an ``RpcClient``) runs over every row whose
    funder is empty and whose kind is still NULL (``Store.
    raters_needing_owner_lookup`` -- this run's pass-1 misses and any legacy
    rows): since 2026-10 nearly every new ERC-8004 rater on Base is a
    smart-contract account that never received a transfer (230/230 sampled
    2026-10-13), so its *controlling key* is read instead
    (``account_owner.account_owners``: ``ownerAtIndex``, then ``owner()``) and
    stored as the funder -- in the owner namespace, ``owner:<address>``, so it
    can never equal a transfer-kind funder -- with ``funder_kind="owner"`` and
    ``first_seen_ts`` left NULL (see ``_settle`` for why). What has no owner
    is written as ``"none"`` so no later run retries it; the ``client`` is not
    asked a second time (see ``_settle``). Per-address failures in pass 2 are collected like pass 1's:
    an address whose own RPC lookup failed is recorded by itself and its
    chunk-mates are still settled (see ``_owner_pass``). ``batch_state``, when
    given, is the caller's shared ``base_erc8004.BatchState`` (the ``fetch``
    command hands the one its agent-owner step used); ``echo``, when given,
    is called once with ``"owner pass: N address(es) pending"`` as pass 2
    starts (the CLI prints it). Without ``rpc`` there is no pass 2 and pass-1
    misses keep their NULL kind for a later run that has one.

    Sets ``rater_profile_mode`` to ``client.source`` (``"blockscout"``,
    ``"etherscan"`` or ``"alchemy"``; ``"etherscan"`` for any client without a
    ``source`` attribute, for compatibility with simple duck-typed test
    doubles) when every address succeeded, or ``"<source>-partial"`` when some
    failed.

    Exception: if the run's *first* client call raises ``ProfilePlanError``
    (the configured plan/key/network cannot serve this chain -- see
    ``profile_http.ProfilePlanError``; ``EtherscanPlanError`` is one), that
    error is re-raised immediately instead of being counted as a per-address
    failure and continuing -- a plan-level rejection will fail identically
    for every remaining address, so there is no reason to burn through (and
    log an ERROR for) all of them just to say so 13,000 times. A plan error
    on any *later* call is still just one more per-address failure, since by
    then real progress may have been made.

    Without ``client`` (offline fallback), writes *no* rows at all and runs
    no pass 2 whatever ``rpc`` is -- see the module docstring for why sticky
    fallback rows would be worse than none. Just records
    ``rater_profile_mode = "fallback"``, logs one WARNING if any feedback
    still lacks a cached block timestamp (``Store.n_missing_block_ts() > 0``,
    meaning ``robustrep.sources.base_erc8004.fill_block_timestamps`` should
    be run first), and returns the number of still-unprofiled addresses as an
    informational count (nothing was written).

    ``addresses``, when given, is used verbatim instead of calling
    ``store.distinct_clients()`` -- for a caller (e.g. the CLI) that already
    computed the target list, so it isn't queried from the store twice.

    Returns the number of distinct addresses settled across both passes with
    a client (an address pass 1 left NULL and pass 2 then settled counts
    once), or the number of still-unprofiled addresses in fallback mode; 0 (no HTTP
    calls, no store writes, no mode change) when every client address is
    already profiled and nothing awaits the owner pass.
    """
    addrs = store.distinct_clients() if addresses is None else addresses

    if client is None:
        if not addrs:
            return 0
        n_missing = store.n_missing_block_ts()
        if n_missing > 0:
            logger.warning(
                "rater_profile: %d block(s) have no cached timestamp - "
                "run fill_block_timestamps first for accurate fallback first-seen times",
                n_missing)
        store.set_sync("rater_profile_mode", "fallback")
        return len(addrs)

    source = getattr(client, "source", "etherscan")
    progress = _Progress()
    if addrs:
        _transfer_pass(store, client, addrs, progress)
    if rpc is not None:
        _owner_pass(store, rpc, progress, batch_state, echo)

    if progress.failures:
        store.set_sync("rater_profile_mode", f"{source}-partial")
        failures = progress.failures
        shown = ", ".join(failures[:5])
        more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
        raise RuntimeError(f"rater_profile: {len(failures)} address(es) failed to enrich: {shown}{more}")
    if progress.processed == 0:
        return 0
    store.set_sync("rater_profile_mode", source)
    return progress.processed
