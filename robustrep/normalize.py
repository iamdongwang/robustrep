"""Map raw values to [0, 1] per (tag, scale) group. Groups never mix."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import scale_decimals


def _group_scores(real: pd.Series, decimals: int) -> pd.Series:
    vals = real.to_numpy(dtype=float)
    uniq = set(np.unique(vals))
    lo, hi = float(vals.min()), float(vals.max())
    if uniq <= {0.0, 1.0}:
        return real.astype(float)
    if decimals == 0 and lo >= 0 and hi <= 100:
        return real / 100.0
    if lo >= 0 and hi <= 1:
        return real.astype(float)
    if hi == lo:
        return pd.Series(0.5, index=real.index)
    return (real - lo) / (hi - lo)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `score` column in [0, 1]. Input must have passed validate_records."""
    out = df.copy()
    out["score"] = np.nan
    if out.empty:
        return out
    decimals = out["scale"].map(scale_decimals)
    real = out["value"] / (10.0 ** decimals)
    for (_, scale), idx in out.groupby(["tag", "scale"]).groups.items():
        out.loc[idx, "score"] = _group_scores(real.loc[idx], scale_decimals(scale))
    return out
