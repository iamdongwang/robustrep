"""scores.json for GitHub Pages: machine-readable, one object per ratee.

NaN (robust_score/ci_low/ci_high on an `insufficient` ratee) round-trips to
JSON `null`, never `NaN` (which is not valid JSON): `pandas.DataFrame.to_json`
already does this by default. Rows are sorted by `robust_score` descending,
NaN last, so a viewer/consumer gets a ranked list without re-sorting.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy
import pandas as pd

from .. import __version__

SCHEMA_VERSION = 1

# Integer-valued result columns: written as JSON ints, never floats
# (pandas.to_json would otherwise emit e.g. "n_clusters": 3.0 once the frame
# has been re-sorted through a copy that lost the original int dtype).
_INT_COLUMNS = ("n_clusters", "n_raw", "sybil_flag", "insufficient")


def export_json(scores: pd.DataFrame, block: int, out: Path, config: Optional[dict] = None) -> Path:
    """Write `scores` (RESULT_COLUMNS) to `out` as JSON: {schema_version, block,
    generated_at, config, versions, scores}. Rows are sorted by robust_score
    descending with NaN (insufficient ratees) last. `config` (e.g. bootstrap_n,
    evidence_weights, the sybil_* thresholds -- the scoring parameters actually
    used) is written verbatim next to `versions`, defaulting to `{}` when not
    given. Parent directories are created as needed.
    """
    ordered = scores.sort_values("robust_score", ascending=False, na_position="last").copy()
    for col in _INT_COLUMNS:
        if col in ordered.columns:
            ordered[col] = ordered[col].astype(int)
    rows = json.loads(ordered.to_json(orient="records"))

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "block": block,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config or {},
        "versions": {"robustrep": __version__, "numpy": numpy.__version__, "pandas": pd.__version__},
        "scores": rows,
    }
    out.write_text(json.dumps(payload, indent=None))
    return out
