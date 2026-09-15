"""scores.json for GitHub Pages: machine-readable, one object per ratee.

NaN (robust_score/ci_low/ci_high on an `insufficient` ratee) round-trips to
JSON `null`, never `NaN` (which is not valid JSON): `pandas.DataFrame.to_json`
already does this by default. Rows are sorted by `robust_score` descending,
NaN last, so a viewer/consumer gets a ranked list without re-sorting.

This file is published to a public CDN, so two properties are enforced here
rather than assumed of every caller upstream:

- **No address ever reaches it** (`_ADDRESS_RE`). Nothing in `RESULT_COLUMNS`
  carries one today -- ratees are agent ids, and raters are aggregated away
  into counts -- but "today's columns" is not a property anyone re-checks when
  adding an explain-style field (a "top raters" list, a cluster membership
  dump). The finished JSON text is scanned once and the export fails loudly
  instead of publishing a de-anonymising row.
- **`versions` names every dependency that can change the numbers**, not just
  the three the scorer imports directly: matplotlib draws the figures the
  report reads from, and requests/urllib3/eth_abi decide which evidence a
  fetch run could retrieve and decode in the first place. The interpreter
  version goes in too -- a report is only reproducible against the stack that
  produced it.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import __version__

SCHEMA_VERSION = 1

# Every dependency whose version can move a published number or figure, in
# `versions` order. Imported lazily (inside `_versions`) so importing this
# module stays cheap for callers that never export.
_VERSIONED_MODULES = ("numpy", "pandas", "matplotlib", "requests", "urllib3", "eth_abi")

# An EVM address as it appears in JSON text. Deliberately not anchored and not
# `\b`-delimited: the point is to catch an address *anywhere* in the payload,
# including inside a longer string. A 64-hex tx hash does not match -- the
# lookahead rejects a 41st hex digit -- which keeps a legitimately published
# hash from failing the export.
_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")

# Integer-valued result columns: written as JSON ints, never floats
# (pandas.to_json would otherwise emit e.g. "n_clusters": 3.0 once the frame
# has been re-sorted through a copy that lost the original int dtype).
_INT_COLUMNS = ("n_clusters", "n_raw", "sybil_flag", "insufficient")


def _module_version(name: str) -> str:
    """The installed version of `name`, preferring the module's own
    `__version__` and falling back to its distribution metadata (a module is
    not required to expose `__version__`; every one of these is installed as a
    distribution)."""
    module = importlib.import_module(name)
    version = getattr(module, "__version__", None)
    return str(version) if version else importlib.metadata.version(name)


def _versions() -> dict:
    """`{name: version}` for robustrep, the interpreter, and every dependency
    in `_VERSIONED_MODULES` -- see the module docstring for why the list is
    wider than the scorer's own imports."""
    versions = {"robustrep": __version__, "python": platform.python_version()}
    for name in _VERSIONED_MODULES:
        versions[name] = _module_version(name)
    return versions


def export_json(scores: pd.DataFrame, block: int, out: Path, config: Optional[dict] = None,
                cluster_stats: Optional[dict] = None) -> Path:
    """Write `scores` (RESULT_COLUMNS) to `out` as JSON: {schema_version, block,
    generated_at, config, cluster_stats, versions, scores}. Rows are sorted by
    robust_score descending with NaN (insufficient ratees) last. `config` (e.g.
    bootstrap_n, evidence_weights, the sybil_* thresholds and pair budgets --
    the scoring parameters actually used) is written verbatim next to
    `versions`, defaulting to `{}` when not given.

    Raises `ValueError` if the finished JSON text contains anything shaped
    like an EVM address (`_ADDRESS_RE`): this file is published, and no
    published field is allowed to carry one. The message never repeats the
    address it found -- the point is to keep it out of the logs too.

    `cluster_stats` (a `robustrep.sybil.ClusterStats` as a dict) says whether
    sybil clustering hit a pair budget on this run and what it skipped; a
    consumer comparing two exports needs it to tell a real change from a
    budget-limited one. Also `{}` when not given. Both are additive keys, so
    `schema_version` stays 1: a v1 consumer that ignores them still reads
    every field it knew about. Parent directories are created as needed.
    """
    ordered = scores.sort_values("robust_score", ascending=False, na_position="last").copy()
    for col in _INT_COLUMNS:
        if col in ordered.columns:
            ordered[col] = ordered[col].astype(int)
    rows = json.loads(ordered.to_json(orient="records"))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "block": block,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config or {},
        "cluster_stats": cluster_stats or {},
        "versions": _versions(),
        "scores": rows,
    }
    text = json.dumps(payload, indent=None)
    if _ADDRESS_RE.search(text):
        raise ValueError("export would publish an address")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out
