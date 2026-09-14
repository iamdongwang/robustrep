"""Column definitions and validation for the unified record table."""
from __future__ import annotations

import pandas as pd

RECORD_COLUMNS = ["rater", "ratee", "value", "scale", "tag", "ts", "evidence_uri", "source"]
OPTIONAL_DEFAULTS = {"evidence_level": 0, "cluster": None, "revoked": 0}
RESULT_COLUMNS = [
    "ratee", "robust_score", "ci_low", "ci_high", "n_clusters", "n_raw",
    "zero_evidence_ratio", "sybil_flag", "naive_mean", "insufficient",
]


def make_scale(decimals: int) -> str:
    return f"d{int(decimals)}"


def scale_decimals(scale: str) -> int:
    if not isinstance(scale, str) or not scale.startswith("d") or not scale[1:].isdigit():
        raise ValueError(f"bad scale {scale!r}; expected like 'd0'")
    return int(scale[1:])


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with required columns checked and optional columns filled."""
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = df.copy()
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    out["cluster"] = out["cluster"].where(out["cluster"].notna(), out["rater"])
    out["evidence_level"] = out["evidence_level"].fillna(0).astype(int)
    out["revoked"] = out["revoked"].fillna(0).astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="raise").astype(float)
    out["rater"] = out["rater"].astype(str)
    out["ratee"] = out["ratee"].astype(str)
    out["tag"] = out["tag"].fillna("").astype(str)
    return out.reset_index(drop=True)
