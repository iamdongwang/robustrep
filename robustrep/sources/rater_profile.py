"""First-seen time and funding source per rater.

Sybil clustering (see ``robustrep.sybil``) needs, per rater address, when it
first appeared on-chain and who funded it (the ``from`` of its first outgoing
transaction). Public RPC nodes expose no per-address index for this, so
``EtherscanClient`` fetches it from an Etherscan-compatible "first
transaction" endpoint. Two such endpoints are supported:

- **Blockscout** (``EtherscanClient.blockscout()``): Base's free,
  Etherscan-compatible Blockscout instance. No API key, no per-chain
  ``chainid`` parameter (the endpoint is already Base-only). This is the
  default (``--profile-source auto`` picks it when no Etherscan key is
  configured).
- **Etherscan V2** (``EtherscanClient(api_key)``/``client_from_env``): needs
  ``ETHERSCAN_API_KEY``. Etherscan's *free* plan does not cover Base via the
  V2 API -- it replies with a "Free API access is not supported for this
  chain" error on every address -- so an Etherscan key here only helps with a
  paid plan.

Rater profiles: Blockscout (free, no key) by default; Etherscan V2 requires a
paid plan for Base.

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
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

import requests

from ..store import Store

logger = logging.getLogger(__name__)

ETHERSCAN_V2 = "https://api.etherscan.io/v2/api"
# Base's Blockscout instance, exposing an Etherscan-compatible `?module=
# account&action=txlist` endpoint with no API key and no `chainid` param
# (it's already scoped to Base). See EtherscanClient.blockscout().
BLOCKSCOUT_BASE = "https://base.blockscout.com/api"
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

# Sentinel returned by ``EtherscanClient._parse_body`` for a body-level error
# (e.g. Etherscan's own "Max rate limit reached" message) that should be
# retried like a transport-level failure, as opposed to a real result (which
# may legitimately be ``None``).
_RETRY = object()


class EtherscanPlanError(RuntimeError):
    """The configured Etherscan API plan does not cover this chain (e.g. the
    free plan does not support Base via the V2 API). Non-retryable: no
    number of retries or different address fixes this, only a different
    plan or a different ``--profile-source``. ``enrich_raters`` treats this
    specially -- see its docstring."""


class EtherscanClient:
    """Minimal client for an Etherscan-compatible "first transaction" lookup.

    Works against both Etherscan V2 (``base_url=ETHERSCAN_V2``, the default)
    and Base's Blockscout instance (``EtherscanClient.blockscout()``), which
    exposes the same ``module=account&action=txlist`` shape with no API key.

    Looks up the earliest transaction for an address via
    ``module=account&action=txlist&sort=asc&offset=1``. Transient failures --
    HTTP 429/5xx, connection/timeout errors, a non-JSON body, or a rate-limit
    message -- are retried up to ``retries`` times with exponential backoff;
    any other HTTP 4xx, an invalid API key, an unsupported plan, or a
    malformed address raise immediately, without consuming a retry. A
    per-request throttle sleep (``1/rps``) runs after every attempt
    regardless of outcome, so the configured request rate is respected even
    while retrying or erroring.

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
        self.session = session or requests.Session()
        self.retries, self.sleep = retries, sleep
        self.gap = 1.0 / rps
        # Used by enrich_raters to label rater_profile_mode ("blockscout" vs
        # "etherscan") without hardcoding a URL comparison there too.
        self.source = "blockscout" if base_url == BLOCKSCOUT_BASE else "etherscan"

    @classmethod
    def blockscout(cls, rps: float = 4.0, **kw) -> "EtherscanClient":
        """Build a client for Base's free Blockscout instance: no API key,
        no ``chainid`` param (the endpoint is already Base-only)."""
        return cls(base_url=BLOCKSCOUT_BASE, chain_id=None, api_key=None, rps=rps, **kw)

    def first_tx(self, address: str) -> Optional[tuple[int, int, str]]:
        """Return ``(block, timestamp, funder_from_address)`` for the earliest
        transaction of ``address``, or ``None`` if it genuinely has none yet.

        Raises ``RuntimeError`` naming ``address`` if every retry is exhausted
        on a transient error (HTTP 429/5xx, connection/timeout, a non-JSON
        body, or a rate-limit message), or immediately -- without retrying --
        for a non-retryable one: any other HTTP 4xx, or an invalid/rejected
        API key. Raises ``EtherscanPlanError`` (a ``RuntimeError`` subclass)
        immediately if the configured plan does not cover this chain (e.g.
        Etherscan's free plan on Base). Raises ``ValueError`` naming
        ``address`` for a malformed response: a body that isn't a JSON object
        (e.g. ``null`` or a bare list), a missing ``status`` field, a
        ``status: "1"`` body whose ``result`` isn't a list, a rejected
        (malformed) address, a transaction entry missing required fields, or
        (after retries) a body that never parses as JSON.
        """
        address = address.lower()
        params = dict(module="account", action="txlist", address=address, page=1, offset=1, sort="asc")
        if self.chain_id is not None:
            params["chainid"] = self.chain_id
        if self.key:
            params["apikey"] = self.key
        last_status: Optional[int] = None
        last_kind = "http"
        for attempt in range(self.retries):
            transient = False
            try:
                try:
                    r = self.session.get(self.base_url, params=params, timeout=30)
                    r.raise_for_status()
                except requests.RequestException as e:
                    code = self._status_code(e)
                    if not self._is_transient_http(code):
                        raise RuntimeError(f"Etherscan HTTP {code} for {address}") from None
                    transient, last_status, last_kind = True, code, "http"
                else:
                    try:
                        body = r.json()
                    except ValueError:
                        transient, last_kind = True, "json"
                    else:
                        if not isinstance(body, dict):
                            raise ValueError(f"Etherscan response is not a JSON object for {address}")
                        outcome = self._parse_body(body, address)
                        if outcome is _RETRY:
                            transient, last_kind = True, "body"
                        else:
                            return outcome
            finally:
                # Rate-limit throttle: always paid, success or failure, so a
                # string of errors can't be used to blow past the configured rps.
                self.sleep(self.gap)

            if transient and attempt < self.retries - 1:
                self.sleep(min(2 ** attempt, 30))

        if last_kind == "json":
            raise ValueError(f"non-JSON Etherscan response for {address}")
        status_label = last_status if last_status is not None else "n/a"
        raise RuntimeError(
            f"Etherscan HTTP error for {address} after {self.retries} attempts (status {status_label})")

    def _parse_body(self, body: dict, address: str):
        """Interpret one decoded JSON response body. Returns the ``first_tx``
        result (a tuple, or ``None`` for "no transactions"), or ``_RETRY`` for
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
        return _RETRY

    @staticmethod
    def _status_code(e: requests.RequestException) -> Optional[int]:
        resp = getattr(e, "response", None)
        return getattr(resp, "status_code", None) if resp is not None else None

    @staticmethod
    def _is_transient_http(code: Optional[int]) -> bool:
        """429 and 5xx are transient; so is no response at all (connection
        errors, timeouts); any other HTTP status is a permanent rejection."""
        if code is None:
            return True
        return code == 429 or 500 <= code < 600

    @staticmethod
    def _parse_tx(tx: dict, address: str) -> tuple[int, int, str]:
        try:
            block = int(tx["blockNumber"])
            ts = int(tx["timeStamp"])
            frm = str(tx["from"]).lower()
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"malformed Etherscan tx entry for {address}: {e}") from e
        return block, ts, frm


def client_from_env() -> Optional[EtherscanClient]:
    """Build an Etherscan V2 ``EtherscanClient`` from the ``ETHERSCAN_API_KEY``
    environment variable, or ``None`` if it is unset/blank (use
    ``EtherscanClient.blockscout()`` or the offline fallback instead)."""
    key = (os.environ.get("ETHERSCAN_API_KEY") or "").strip()
    return EtherscanClient(key) if key else None


def default_client(source: str, key: Optional[str]) -> Optional[EtherscanClient]:
    """Build the rater-profile HTTP client for ``--profile-source source``,
    given ``key`` (an explicit ``--etherscan-key``/``ETHERSCAN_API_KEY``
    value, or ``None``).

    - ``"none"``: no client -- ``enrich_raters`` runs its offline fallback
      (writes no rows).
    - ``"blockscout"``: ``EtherscanClient.blockscout()`` -- Base's free,
      keyless Blockscout instance.
    - ``"etherscan"``: Etherscan V2 using ``key``. Raises ``ValueError`` if
      ``key`` is falsy -- Etherscan V2 always needs a key, and unlike
      ``"auto"`` this source was requested explicitly, so silently falling
      back would hide the missing configuration.
    - ``"auto"`` (the CLI default): Etherscan V2 when ``key`` is given
      (presumably a paid plan, since the free plan doesn't cover Base -- see
      the module docstring), else Blockscout.

    Raises ``ValueError`` for any other ``source`` value.
    """
    if source == "none":
        return None
    if source == "blockscout":
        return EtherscanClient.blockscout()
    if source == "etherscan":
        if not key:
            raise ValueError("--profile-source etherscan requires --etherscan-key or ETHERSCAN_API_KEY")
        return EtherscanClient(key)
    if source == "auto":
        return EtherscanClient(key) if key else EtherscanClient.blockscout()
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


def enrich_raters(store: Store, client: Optional[EtherscanClient],
                   addresses: Optional[list[str]] = None) -> int:
    """Fill the raters table for every client address not yet profiled.

    With ``client``, looks up each address's first transaction. A failure on
    one address does not lose the others: every address is attempted,
    successes are upserted, and if any failed a ``RuntimeError`` listing them
    (count + first 5) is raised at the end -- but only after the successful
    ones have been persisted and the mode recorded. Sets
    ``rater_profile_mode`` to ``client.source`` (``"blockscout"`` or
    ``"etherscan"``; ``"etherscan"`` for any client without a ``source``
    attribute, for compatibility with simple duck-typed test doubles) when
    every address succeeded, or ``"<source>-partial"`` when some failed.

    Exception: if the *first* address attempted raises ``EtherscanPlanError``
    (the configured Etherscan plan does not cover this chain -- see its
    docstring), that error is re-raised immediately instead of being counted
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
        except EtherscanPlanError:
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
