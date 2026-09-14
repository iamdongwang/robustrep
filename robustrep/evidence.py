"""Evidence level (0..3) for a rating and the level -> weight mapping."""
from __future__ import annotations

import re
from typing import Callable, Optional

import pandas as pd

from .config import Config

TX_RE = re.compile(r"0x[0-9a-fA-F]{64}")
TASK_KEY_RE = re.compile(r'"(?:taskId|task_id|jobId|job_id|orderId|order_id)"\s*:')

FetchText = Callable[[str], Optional[str]]
TxParties = Callable[[str], Optional[set]]  # tx hash -> {from, to} lowercased, or None


def weights_for(levels: pd.Series, cfg: Config) -> pd.Series:
    w = cfg.evidence_weights
    if levels.empty:
        return pd.Series([], dtype=float, index=levels.index)
    return levels.astype(int).clip(0, 3).map(lambda l: w[l]).astype(float)


def classify(uri: Optional[str], fetch_text: FetchText, tx_parties: TxParties, parties: set) -> int:
    """parties: lowercased addresses of rater and ratee. Returns 0..3 per the spec table."""
    if not uri:
        return 0
    text = fetch_text(uri)
    if text is None:
        return 1
    hashes = TX_RE.findall(text)
    if not hashes and not TASK_KEY_RE.search(text):
        return 1
    parties_lc = {p.lower() for p in parties}
    for h in hashes:
        try:
            found = tx_parties(h)
        except Exception:
            found = None
        if found and {a.lower() for a in found} & parties_lc:
            return 3
    return 2
