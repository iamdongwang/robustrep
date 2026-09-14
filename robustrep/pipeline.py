"""score(): validate -> normalize -> evidence weights -> sybil collapse -> weighted median + bootstrap CI."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .aggregate import Aggregator, weighted_median
from .config import Config
from .evidence import weights_for
from .normalize import normalize
from .schema import RESULT_COLUMNS, validate_records


def prepare(records: pd.DataFrame, cfg: Config, clusters: Optional[dict] = None) -> pd.DataFrame:
    """Validate, drop revoked rows, normalize, and attach weight/cluster columns.

    Returns a new per-record frame with `score`, `weight`, and `cluster`
    columns; never mutates `records`. If `clusters` is given, each rater's
    cluster is `clusters.get(rater, rater)` -- a rater missing from the dict
    is its own singleton cluster. If `clusters` is falsy (None or empty),
    the `cluster` column produced by `validate_records` (defaulting to
    `rater`) is left untouched.
    """
    df = validate_records(records)
    df = df[df["revoked"] == 0].copy()
    df = normalize(df)
    df["weight"] = weights_for(df["evidence_level"], cfg)
    if clusters:
        df["cluster"] = df["rater"].map(lambda r: clusters.get(r, r))
    return df


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    """One vote per (ratee, tag, cluster): median score, mean weight, record count."""
    return (df.groupby(["ratee", "tag", "cluster"], as_index=False)
              .agg(score=("score", "median"), weight=("weight", "mean"), n=("score", "size")))


def _total(scores: np.ndarray, weights: np.ndarray, tag_codes: np.ndarray) -> float:
    """Per-tag weighted median of votes, then weighted median across tags,
    each tag weighted by its number of (already-collapsed) votes."""
    tag_scores, tag_w = [], []
    for code in np.unique(tag_codes):
        m = tag_codes == code
        tag_scores.append(weighted_median(scores[m], weights[m]))
        tag_w.append(float(m.sum()))
    return weighted_median(np.array(tag_scores), np.array(tag_w))


def _ci(scores: np.ndarray, weights: np.ndarray, tag_codes: np.ndarray, cfg: Config,
        rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap CI over the vote array via one vectorized `rng.integers` draw
    (no per-resample pandas operations, so this stays cheap at ~28k ratees)."""
    n = len(scores)
    if n == 1 or cfg.bootstrap_n == 0:
        s = _total(scores, weights, tag_codes)
        return s, s
    idx = rng.integers(0, n, size=(cfg.bootstrap_n, n))
    boots = np.array([_total(scores[i], weights[i], tag_codes[i]) for i in idx])
    a = (1 - cfg.ci_level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return float(lo), float(hi)


def score(records: pd.DataFrame, cfg: Config = Config(), clusters: Optional[dict] = None,
          aggregator: Optional[Aggregator] = None) -> pd.DataFrame:
    """Score every ratee in `records`. One row per ratee, columns = RESULT_COLUMNS.

    Pipeline: validate_records -> drop revoked -> normalize -> evidence
    weights -> cluster assignment (from `clusters`, else each rater its own
    cluster) -> collapse to one vote per (ratee, tag, cluster) -> per-ratee
    `robust_score` = weighted median across tags of the per-tag weighted
    median of votes (tag weight = number of clusters/votes in that tag) ->
    bootstrap CI over the *votes* (collapsed per-cluster values), not the
    raw per-record rows, so sybil-farmed raters within one cluster cannot
    inflate the resample. `records` is never mutated.

    Determinism: a single `numpy.random.Generator` is created once per
    `score()` call (seeded from `cfg.bootstrap_seed`) and advanced once per
    ratee, in a fixed `sort=True` ratee order -- so two calls with the same
    `records` and `cfg` produce bit-identical output, independent of the
    input row order or of how many ratees are being scored.

    A ratee with fewer than `cfg.min_clusters` distinct clusters is
    `insufficient` (robust_score/ci_low/ci_high are NaN). `sybil_flag` is 1
    when the largest cluster's share of that ratee's raw records is >=
    `cfg.sybil_flag_share`. `naive_mean` is the unweighted mean of raw
    per-record scores, for comparison against the robust estimate.

    `aggregator` is reserved for future non-default aggregators; v1 always
    uses the weighted median and this parameter is currently ignored.
    """
    del aggregator
    df = prepare(records, cfg, clusters)
    if df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    votes = collapse(df)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    rows = []
    for ratee, raw in df.groupby("ratee", sort=True):
        v = votes[votes["ratee"] == ratee]
        n_clusters = int(v["cluster"].nunique())
        share = raw.groupby("cluster").size().max() / len(raw)
        row = dict(ratee=ratee, n_clusters=n_clusters, n_raw=int(len(raw)),
                   zero_evidence_ratio=float((raw["evidence_level"] == 0).mean()),
                   sybil_flag=int(share >= cfg.sybil_flag_share),
                   naive_mean=float(raw["score"].mean()))
        if n_clusters < cfg.min_clusters:
            row.update(robust_score=np.nan, ci_low=np.nan, ci_high=np.nan, insufficient=1)
        else:
            s_arr = v["score"].to_numpy(dtype=float)
            w_arr = v["weight"].to_numpy(dtype=float)
            t_arr = pd.factorize(v["tag"])[0]
            s = _total(s_arr, w_arr, t_arr)
            lo, hi = _ci(s_arr, w_arr, t_arr, cfg, rng)
            row.update(robust_score=s, ci_low=min(lo, s), ci_high=max(hi, s), insufficient=0)
        rows.append(row)
    out = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    for col in ("n_clusters", "n_raw", "sybil_flag", "insufficient"):
        out[col] = out[col].astype(int)
    return out
