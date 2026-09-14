"""Column definitions and validation for the unified record table."""
from __future__ import annotations

import numpy as np
import pandas as pd

RECORD_COLUMNS = ["rater", "ratee", "value", "scale", "tag", "ts", "evidence_uri", "source"]
OPTIONAL_DEFAULTS = {"evidence_level": 0, "cluster": None, "revoked": 0}
RESULT_COLUMNS = [
    "ratee", "robust_score", "ci_low", "ci_high", "n_clusters", "n_raw",
    "zero_evidence_ratio", "sybil_flag", "naive_mean", "insufficient",
]

REQUIRED_NON_NULL = ["rater", "ratee", "value", "ts"]
VALID_LEVELS = {0, 1, 2, 3}
VALID_REVOKED = {0, 1}


# ERC-8004 valueDecimals is uint8 (0-255); the EIP text recommends 0-18. Values up to 255 are
# chain-valid and must not be dropped.
MAX_SCALE_DECIMALS = 255


def make_scale(decimals: int) -> str:
    if not isinstance(decimals, (int, np.integer)) or isinstance(decimals, bool) or decimals < 0:
        raise ValueError(f"decimals must be a non-negative integer, got {decimals!r}")
    if decimals > MAX_SCALE_DECIMALS:
        raise ValueError(
            f"decimals must be 0-{MAX_SCALE_DECIMALS} (ERC-8004 valueDecimals is uint8), got {decimals!r}")
    return f"d{int(decimals)}"


def scale_decimals(scale: str) -> int:
    if not isinstance(scale, str) or not scale.startswith("d") or not scale[1:].isdigit():
        raise ValueError(f"bad scale {scale!r}; expected like 'd0'")
    decimals = int(scale[1:])
    if decimals > MAX_SCALE_DECIMALS:
        raise ValueError(
            f"bad scale {scale!r}; decimals must be 0-{MAX_SCALE_DECIMALS} (ERC-8004 valueDecimals is uint8)")
    return decimals


def _coerce_numeric(out: pd.DataFrame, col: str, dtype: str) -> None:
    try:
        out[col] = pd.to_numeric(out[col], errors="raise")
    except (ValueError, TypeError) as e:
        raise ValueError(f"column {col!r}: {e}") from e
    finite = np.isfinite(out[col].to_numpy(dtype=float))
    if not finite.all():
        bad = out.index[~finite].tolist()[:5]
        raise ValueError(f"column {col!r}: non-finite values at rows {bad}")
    out[col] = out[col].astype(dtype)


def _check_domain(out: pd.DataFrame, col: str, allowed: set) -> None:
    vals = out[col].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError(f"column {col!r}: non-finite values")
    if not np.array_equal(vals, np.floor(vals)):
        raise ValueError(f"column {col!r}: non-integer values")
    bad = sorted(set(vals.astype(int)) - allowed)
    if bad:
        raise ValueError(f"column {col!r}: values {bad} not in {sorted(allowed)}")
    out[col] = out[col].astype(int)


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Boundary check + coercion. Returns a new frame; the input is never modified.

    ts is truncated to integer seconds.
    """
    if df.columns.duplicated().any():
        raise ValueError(f"duplicated columns: {df.columns[df.columns.duplicated()].tolist()}")
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = df.copy()
    for col in REQUIRED_NON_NULL:
        if out[col].isna().any():
            rows = out.index[out[col].isna()].tolist()[:5]
            raise ValueError(f"column {col!r}: null values at rows {rows}")
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    out["rater"] = out["rater"].astype(str)
    out["ratee"] = out["ratee"].astype(str)
    out["tag"] = out["tag"].fillna("").astype(str)
    out["scale"] = out["scale"].fillna("").astype(str)
    out["source"] = out["source"].fillna("").astype(str)
    bad_scales = []
    for s in out["scale"].unique():
        try:
            scale_decimals(s)
        except ValueError:
            bad_scales.append(s)
    if bad_scales:
        rows = out.index[out["scale"].isin(bad_scales)].tolist()[:5]
        raise ValueError(f"column 'scale': bad values {bad_scales[:5]} at rows {rows}")
    _coerce_numeric(out, "value", "float")
    _coerce_numeric(out, "ts", "int64")
    for col in ("evidence_level", "revoked"):
        try:
            out[col] = pd.to_numeric(out[col].where(out[col].notna(), 0), errors="raise")
        except (ValueError, TypeError) as e:
            raise ValueError(f"column {col!r}: {e}") from e
    _check_domain(out, "evidence_level", VALID_LEVELS)
    _check_domain(out, "revoked", VALID_REVOKED)
    out["cluster"] = out["cluster"].where(out["cluster"].notna(), out["rater"]).astype(str)
    return out.reset_index(drop=True)
