"""Evidence level (0..3) for a rating and the level -> weight mapping."""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

import pandas as pd

from .config import Config

TX_RE = re.compile(r"0x[0-9a-fA-F]{64}(?![0-9a-fA-F])")
# Deliberately strict (double-quoted JSON keys only): misses fall to level 1,
# the conservative direction.
TASK_KEY_RE = re.compile(r'"(?:taskId|task_id|jobId|job_id|orderId|order_id)"\s*:')

FetchText = Callable[[str], Optional[str]]
TxParties = Callable[[str], Optional[set[str]]]  # tx hash -> {from, to} lowercased, or None

_log = logging.getLogger(__name__)


def weights_for(levels: pd.Series, cfg: Config) -> pd.Series:
    """Map evidence levels (0..3) to weights per cfg.evidence_weights.

    Preserves the input index. Raises ValueError if any level is outside
    [0, 1, 2, 3] or null.
    """
    bad = sorted(set(levels.dropna()) - {0, 1, 2, 3})
    if bad:
        raise ValueError(f"evidence_level values {bad} not in [0, 1, 2, 3]")
    if levels.isna().any():
        raise ValueError("evidence_level contains null")
    w = cfg.evidence_weights
    return levels.astype(int).map(lambda l: w[l]).astype(float)


def _verified(h: str, tx_parties: TxParties, parties_lc: set[str]) -> bool:
    try:
        found = tx_parties(h)
        return bool(found) and bool({a.lower() for a in found if a} & parties_lc)
    except Exception:
        _log.debug("tx_parties(%s) failed; treating as unverified", h, exc_info=True)
        return False


def classify(uri: Optional[str], fetch_text: FetchText, tx_parties: TxParties, parties: set[str]) -> int:
    """Evidence level 0..3 per the spec table.

    parties: lowercased addresses of rater and ratee; falsy entries ignored.

    Callable contracts: `fetch_text` exceptions propagate to the caller (its
    own fetcher must handle its own errors); `tx_parties` exceptions, or a
    None/falsy return, are tolerated and degrade that hash to unverified
    (logged at DEBUG), never raising out of `classify`.
    """
    if not uri or not uri.strip():
        return 0
    text = fetch_text(uri)
    if text is None:
        return 1
    hashes = TX_RE.findall(text)
    if not hashes and not TASK_KEY_RE.search(text):
        return 1
    parties_lc = {p.lower() for p in parties if p}
    for h in dict.fromkeys(x.lower() for x in hashes):
        if _verified(h, tx_parties, parties_lc):
            return 3
    return 2
