"""Map raw values to [0, 1] per (tag, scale) group. Groups never mix."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import scale_decimals

RULE_BINARY = "binary"
RULE_PERCENT = "percent"
RULE_UNIT = "unit"
RULE_CONSTANT = "constant"
RULE_MINMAX = "minmax"


def _group_scores(real: pd.Series, decimals: int) -> tuple[np.ndarray, str]:
    """Map one (tag, scale) group's real values to [0, 1].

    Tries these rules in order and returns the first that applies, along
    with its name. Order matters:

    1. binary   - every value is 0 or 1: passed through unchanged. Tried
       first so a {0, 1}-only group is never divided by 100 by the
       "percent" rule below.
    2. percent  - integer scale (decimals == 0) with all values in
       [0, 100]: divided by 100. Tried before "unit" so an integer-scaled
       group already confined to [0, 1] (e.g. whole-number percentages
       0 or 1) is still treated as a percentage rather than assumed
       already-normalized.
    3. unit     - all values already fall in [0, 1]: passed through
       unchanged.
    4. constant - every value in the group is identical (and none of the
       above matched): scored as a neutral 0.5. Tried after "percent" so a
       constant all-100 d0 group scores 1.0 via "percent", while a
       constant all-500 d0 group has no natural anchor and scores 0.5 as
       a neutral value.
    5. minmax   - fallback: linearly rescale the group's [min, max] to
       [0, 1].
    """
    vals = real.to_numpy(dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if bool(np.isin(vals, (0.0, 1.0)).all()):
        return vals, RULE_BINARY
    if decimals == 0 and lo >= 0 and hi <= 100:
        return vals / 100.0, RULE_PERCENT
    if lo >= 0 and hi <= 1:
        return vals, RULE_UNIT
    if hi == lo:
        return np.full_like(vals, 0.5), RULE_CONSTANT
    return (vals - lo) / (hi - lo), RULE_MINMAX


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add `score` (float, in [0, 1]) and `norm_rule` (str) columns.

    Input must have passed validate_records. Groups are formed strictly by
    (tag, scale) and never mixed - rows are selected by position, not by
    index label, so frames with duplicate index labels (e.g. the result of
    `pd.concat`) are handled correctly. Resets the index; any existing
    `score`/`norm_rule` columns are overwritten.
    """
    out = df.reset_index(drop=True)
    decimals = out["scale"].map(scale_decimals)
    real = out["value"] / (10.0 ** decimals)
    scores = np.full(len(out), np.nan)
    rules = np.full(len(out), None, dtype=object)
    for (_, scale), pos in out.groupby(["tag", "scale"], sort=False).indices.items():
        group_scores, rule = _group_scores(real.iloc[pos], scale_decimals(scale))
        scores[pos] = group_scores
        rules[pos] = rule
    out["score"] = scores
    out["norm_rule"] = rules
    return out
