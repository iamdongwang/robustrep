"""First-seen time and funding source per rater.

Sybil clustering (see ``robustrep.sybil``) needs, per rater address, when it
first appeared on-chain and who funded it (the ``from`` of its first outgoing
transaction). Public RPC nodes expose no per-address index for this, so
``EtherscanClient`` fetches it from the Etherscan V2 API when an API key is
available (``client_from_env``/``ETHERSCAN_API_KEY``). Without a key,
``enrich_raters`` falls back to first-seen = the rater's earliest feedback
timestamp in our own data and funder = ``None`` (clustering then relies only
on the time window + Jaccard signal). ``enrich_raters`` records which mode ran
via ``Store.set_sync("rater_profile_mode", ...)`` so downstream reports can
state which mode produced the profile.
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


class EtherscanClient:
    """Minimal client for the Etherscan V2 "first transaction" lookup.

    Looks up the earliest transaction for an address via
    ``module=account&action=txlist&sort=asc&offset=1``. Retries transient
    failures (e.g. rate limiting) up to ``retries`` times with exponential
    backoff, rotates to no other endpoint (Etherscan V2 is a single host, only
    ``chainid`` varies), and raises immediately -- without consuming a retry --
    on a non-retryable error such as an invalid API key. Never logs request
    parameters (which include the API key).
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
        on a transient error, or immediately for a non-retryable one (e.g. an
        invalid API key). Raises ``ValueError`` naming ``address`` if
        Etherscan returns a transaction entry missing required fields.
        """
        address = address.lower()
        params = dict(chainid=self.chain_id, module="account", action="txlist", address=address,
                      page=1, offset=1, sort="asc", apikey=self.key)
        last_error: Optional[str] = None
        for attempt in range(self.retries):
            r = self.session.get(ETHERSCAN_V2, params=params, timeout=30)
            r.raise_for_status()
            body = r.json()
            self.sleep(self.gap)
            result = body.get("result")
            if body.get("status") == "1":
                if not isinstance(result, list) or not result:
                    return None
                return self._parse_tx(result[0], address)
            if isinstance(result, list):
                # status "0" with an (empty) list result: genuinely no tx.
                return None
            text = "" if result is None else str(result)
            message = str(body.get("message") or "")
            low = text.lower()
            if any(s in low for s in _NO_TX):
                return None
            if any(s in low or s in message.lower() for s in _NON_RETRYABLE):
                raise RuntimeError(f"Etherscan API key error for {address}: {text or message}")
            last_error = text or message or "unknown error"
            if attempt < self.retries - 1:
                self.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"Etherscan request for {address} failed after {self.retries} attempts: {last_error}")

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


def enrich_raters(store: Store, client: Optional[EtherscanClient]) -> int:
    """Fill the raters table for every client address not yet profiled.

    With ``client``, looks up each address's first transaction via Etherscan
    V2. A failure on one address does not lose the others: every address is
    attempted, successes are upserted, and if any failed a ``RuntimeError``
    listing them (count + first 5) is raised at the end. Without ``client``,
    falls back to the earliest feedback timestamp in our own data (funder
    left ``None``); if that timestamp is 0 (block time not cached yet -- see
    ``Store.missing_block_ts``), a single WARNING is logged regardless of how
    many addresses are affected.

    Records which mode ran via ``store.set_sync("rater_profile_mode", ...)``
    ("etherscan" or "fallback") so a report can state it. Returns the number
    of addresses successfully processed; 0 (no HTTP calls, no store writes)
    when every client address is already profiled.
    """
    addrs = store.distinct_clients()
    if not addrs:
        return 0

    if client is None:
        recs = store.load_records()
        first_seen = recs.groupby("rater")["ts"].min()
        warned = False
        for a in addrs:
            ts = int(first_seen.get(a, 0))
            if ts == 0 and not warned:
                logger.warning(
                    "rater %s has first-seen ts=0 (no cached block timestamp) - "
                    "run fill_block_timestamps first", a)
                warned = True
            store.upsert_rater(a, ts, None)
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
    store.set_sync("rater_profile_mode", "etherscan")
    if failures:
        shown = ", ".join(failures[:5])
        more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
        raise RuntimeError(f"rater_profile: {len(failures)} address(es) failed to enrich: {shown}{more}")
    return processed
