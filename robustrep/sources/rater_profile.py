"""First-seen time and funding source per rater.

Sybil clustering (see ``robustrep.sybil``) needs, per rater address, when it
first appeared on-chain and who funded it (the ``from`` of its first outgoing
transaction). Public RPC nodes expose no per-address index for this, so
``EtherscanClient`` fetches it from the Etherscan V2 API when an API key is
available (``client_from_env``/``ETHERSCAN_API_KEY``). Without a key,
``enrich_raters`` cannot determine a funder at all, and rather than writing
placeholder rows it leaves the raters table untouched: ``robustrep.sybil.
profiles_from_records`` already falls back to the minimum feedback ``ts`` per
rater when no meta row exists, so an unprofiled address gets that same
first-seen value for free, and -- unlike a written row -- stays visible to
``Store.distinct_clients()`` so a *later* run with a real API key can still
profile it via Etherscan instead of being permanently stuck with the
approximate value. ``enrich_raters`` records which mode ran via
``Store.set_sync("rater_profile_mode", ...)`` so downstream reports can state
which mode produced the profile.
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
BASE_CHAIN_ID = 8453

# Substrings (case-insensitive) of an Etherscan V2 error ``result``/``message``
# that indicate the request itself is permanently rejected (bad key) -- retrying
# cannot help, so these raise immediately instead of consuming the retry budget.
_NON_RETRYABLE = ("api key",)
# Substrings meaning "this address genuinely has no transactions" -- not an
# error at all, so these return ``None`` immediately without retrying.
_NO_TX = ("no transactions found",)

# Sentinel returned by ``EtherscanClient._parse_body`` for a body-level error
# (e.g. Etherscan's own "Max rate limit reached" message) that should be
# retried like a transport-level failure, as opposed to a real result (which
# may legitimately be ``None``).
_RETRY = object()


class EtherscanClient:
    """Minimal client for the Etherscan V2 "first transaction" lookup.

    Looks up the earliest transaction for an address via
    ``module=account&action=txlist&sort=asc&offset=1``. Transient failures --
    HTTP 429/5xx, connection/timeout errors, a non-JSON body, or an
    Etherscan-level rate-limit message -- are retried up to ``retries`` times
    with exponential backoff; any other HTTP 4xx or an invalid API key raise
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

    def __init__(self, api_key: str, chain_id: int = BASE_CHAIN_ID, session=None, rps: float = 4.0,
                 retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        self.key, self.chain_id = api_key, chain_id
        self.session = session or requests.Session()
        self.retries, self.sleep = retries, sleep
        self.gap = 1.0 / rps

    def first_tx(self, address: str) -> Optional[tuple[int, int, str]]:
        """Return ``(block, timestamp, funder_from_address)`` for the earliest
        transaction of ``address``, or ``None`` if it genuinely has none yet.

        Raises ``RuntimeError`` naming ``address`` if every retry is exhausted
        on a transient error (HTTP 429/5xx, connection/timeout, a non-JSON
        body, or an Etherscan rate-limit message), or immediately -- without
        retrying -- for a non-retryable one: any other HTTP 4xx, or an
        invalid/rejected API key. Raises ``ValueError`` naming ``address`` for
        a malformed response: a body that isn't a JSON object (e.g. ``null``
        or a bare list), a missing ``status`` field, a ``status: "1"`` body
        whose ``result`` isn't a list, a transaction entry missing required
        fields, or (after retries) a body that never parses as JSON.
        """
        address = address.lower()
        params = dict(chainid=self.chain_id, module="account", action="txlist", address=address,
                      page=1, offset=1, sort="asc", apikey=self.key)
        last_status: Optional[int] = None
        last_kind = "http"
        for attempt in range(self.retries):
            transient = False
            try:
                try:
                    r = self.session.get(ETHERSCAN_V2, params=params, timeout=30)
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
        directly for non-transient errors (see ``first_tx``)."""
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
    """Build an ``EtherscanClient`` from the ``ETHERSCAN_API_KEY`` environment
    variable, or ``None`` if it is unset/blank (use the offline fallback)."""
    key = (os.environ.get("ETHERSCAN_API_KEY") or "").strip()
    return EtherscanClient(key) if key else None


def estimate_seconds(n_addresses: int, rps: float) -> float:
    """Rough wall-clock time (seconds) for ``enrich_raters`` to profile
    ``n_addresses`` via Etherscan at ``rps`` requests/sec, excluding retry
    backoff: ``n_addresses / rps`` -- e.g. ``estimate_seconds(100_000, 4.0)``
    is about 25,000s (~6.9h). For the CLI to print a progress estimate before
    a long enrichment run."""
    if rps <= 0:
        raise ValueError("rps must be > 0")
    return n_addresses / rps


def enrich_raters(store: Store, client: Optional[EtherscanClient]) -> int:
    """Fill the raters table for every client address not yet profiled.

    With ``client``, looks up each address's first transaction via Etherscan
    V2. A failure on one address does not lose the others: every address is
    attempted, successes are upserted, and if any failed a ``RuntimeError``
    listing them (count + first 5) is raised at the end -- but only after the
    successful ones have been persisted and the mode recorded. Sets
    ``rater_profile_mode`` to ``"etherscan"`` when every address succeeded, or
    ``"etherscan-partial"`` when some failed.

    Without ``client`` (offline fallback), writes *no* rows at all -- see the
    module docstring for why sticky fallback rows would be worse than none.
    Just records ``rater_profile_mode = "fallback"``, logs one WARNING if any
    feedback still lacks a cached block timestamp (``Store.n_missing_block_ts()
    > 0``, meaning ``robustrep.sources.base_erc8004.fill_block_timestamps``
    should be run first), and returns the number of still-unprofiled
    addresses as an informational count (nothing was written).

    Returns the number of addresses successfully processed with a client, or
    the number of still-unprofiled addresses in fallback mode; 0 (no HTTP
    calls, no store writes, no mode change) when every client address is
    already profiled.
    """
    addrs = store.distinct_clients()
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

    failures: list[str] = []
    processed = 0
    for a in addrs:
        try:
            got = client.first_tx(a)
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
        store.set_sync("rater_profile_mode", "etherscan-partial")
        shown = ", ".join(failures[:5])
        more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
        raise RuntimeError(f"rater_profile: {len(failures)} address(es) failed to enrich: {shown}{more}")
    store.set_sync("rater_profile_mode", "etherscan")
    return processed
