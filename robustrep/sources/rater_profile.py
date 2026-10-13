"""First-seen time and funding source per rater.

Sybil clustering (see ``robustrep.sybil``) needs, per rater address, when it
first appeared on-chain and who funded it (the ``from`` of its first outgoing
transaction). Public RPC nodes expose no per-address index for this, so two
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
from typing import Callable, Optional

import requests

from ..store import Store
from . import blockscout_util
from .alchemy_profile import AlchemyClient
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

logger = logging.getLogger(__name__)

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


def enrich_raters(store: Store, client: Optional["EtherscanClient | BlockscoutV2Client | AlchemyClient"],
                   addresses: Optional[list[str]] = None) -> int:
    """Fill the raters table for every client address not yet profiled.

    With ``client``, looks up each address's first transaction. A failure on
    one address does not lose the others: every address is attempted,
    successes are upserted, and if any failed a ``RuntimeError`` listing them
    (count + first 5) is raised at the end -- but only after the successful
    ones have been persisted and the mode recorded. Sets
    ``rater_profile_mode`` to ``client.source`` (``"blockscout"``,
    ``"etherscan"`` or ``"alchemy"``; ``"etherscan"`` for any client without a ``source``
    attribute, for compatibility with simple duck-typed test doubles) when
    every address succeeded, or ``"<source>-partial"`` when some failed.

    Exception: if the *first* address attempted raises ``ProfilePlanError``
    (the configured plan/key/network cannot serve this chain -- see
    ``profile_http.ProfilePlanError``; ``EtherscanPlanError`` is one), that error is re-raised immediately instead of being counted
    as a per-address failure and continuing -- a plan-level rejection will
    fail identically for every remaining address, so there is no reason to
    burn through (and log an ERROR for) all of them just to say so 13,000
    times. A plan error on any *later* address is still just one more
    per-address failure, since by then real progress may have been made.

    Without ``client`` (offline fallback), writes *no* rows at all -- see the
    module docstring for why sticky fallback rows would be worse than none.
    Just records ``rater_profile_mode = "fallback"``, logs one WARNING if any
    feedback still lacks a cached block timestamp (``Store.n_missing_block_ts()
    > 0``, meaning ``robustrep.sources.base_erc8004.fill_block_timestamps``
    should be run first), and returns the number of still-unprofiled
    addresses as an informational count (nothing was written).

    ``addresses``, when given, is used verbatim instead of calling
    ``store.distinct_clients()`` -- for a caller (e.g. the CLI) that already
    computed the target list, so it isn't queried from the store twice.

    Returns the number of addresses successfully processed with a client, or
    the number of still-unprofiled addresses in fallback mode; 0 (no HTTP
    calls, no store writes, no mode change) when every client address is
    already profiled.
    """
    addrs = store.distinct_clients() if addresses is None else addresses
    if not addrs:
        return 0

    if client is None:
        n_missing = store.n_missing_block_ts()
        if n_missing > 0:
            logger.warning(
                "rater_profile: %d block(s) have no cached timestamp - "
                "run fill_block_timestamps first for accurate fallback first-seen times",
                n_missing)
        store.set_sync("rater_profile_mode", "fallback")
        return len(addrs)

    source = getattr(client, "source", "etherscan")
    failures: list[str] = []
    processed = 0
    for idx, a in enumerate(addrs):
        try:
            got = client.first_tx(a)
        except ProfilePlanError:
            if idx == 0:
                raise
            logger.error("rater_profile: failed to enrich %s: plan does not support this chain", a)
            failures.append(a)
            continue
        except Exception as e:
            logger.error("rater_profile: failed to enrich %s: %s", a, e)
            failures.append(a)
            continue
        if got is None:
            store.upsert_rater(a, None, None)
        else:
            _, ts, funder = got
            store.upsert_rater(a, ts, funder)
        processed += 1

    if failures:
        store.set_sync("rater_profile_mode", f"{source}-partial")
        shown = ", ".join(failures[:5])
        more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
        raise RuntimeError(f"rater_profile: {len(failures)} address(es) failed to enrich: {shown}{more}")
    store.set_sync("rater_profile_mode", source)
    return processed
