"""score(): validate -> normalize -> evidence weights -> sybil collapse -> weighted median + bootstrap CI."""
from __future__ import annotations

import zlib
from typing import Optional

import numpy as np
import pandas as pd

from .aggregate import Aggregator, bootstrap_ci, weighted_median
from .config import Config
from .evidence import weights_for
from .normalize import normalize
from .schema import RESULT_COLUMNS, validate_records

# Explicit dtypes for the empty-input result frame: building a DataFrame from
# an empty list of row-dicts would leave every column `object`, which breaks
# callers that check dtype (e.g. is_integer_dtype) before checking row count.
_RESULT_DTYPES = {
    "ratee": object,
    "robust_score": float,
    "ci_low": float,
    "ci_high": float,
    "n_clusters": int,
    "n_raw": int,
    "zero_evidence_ratio": float,
    "sybil_flag": int,
    "naive_mean": float,
    "insufficient": int,
}


def prepare(records: pd.DataFrame, cfg: Config, clusters: Optional[dict] = None) -> pd.DataFrame:
    """Validate, drop revoked rows, normalize, and attach weight/cluster columns.

    Returns a new per-record frame with `score`, `weight`, and `cluster`
    columns; never mutates `records`. If `clusters` is given, each rater's
    cluster is `str(clusters.get(rater, rater))` -- a rater missing from the
    dict is its own singleton cluster, and cluster ids are coerced to `str`
    so e.g. `{"r0": 7, "r1": "7"}` collapse into the same cluster rather than
    being kept apart by an int/str type mismatch. If `clusters` is falsy
    (None or empty), the `cluster` column produced by `validate_records`
    (defaulting to `rater`) is left untouched.
    """
    df = validate_records(records)
    df = df[df["revoked"] == 0].copy()
    df = normalize(df, cfg.norm_fit_share)
    df["weight"] = weights_for(df["evidence_level"], cfg)
    if clusters:
        df["cluster"] = df["rater"].map(lambda r: clusters.get(r, r)).astype(str)
    return df


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    """One vote per (ratee, tag, cluster): median score, mean weight."""
    return (df.groupby(["ratee", "tag", "cluster"], as_index=False)
              .agg(score=("score", "median"), weight=("weight", "mean")))


def _total(scores: np.ndarray, weights: np.ndarray, tag_codes: np.ndarray) -> float:
    """Per-tag weighted median of votes, then weighted median across tags.

    Each tag's cross-tag weight is the SUM of its votes' evidence weights,
    not merely their count. Count-weighting is exploitable: an attacker
    wanting to sink a well-evidenced score can simply file a handful of
    evidence-free (weight ~0.1) ratings under a brand-new tag -- with count
    weighting, that fresh, unevidenced tag would carry exactly as much
    cross-tag influence as an established tag backed by real evidence, no
    matter how thin its actual evidentiary mass is. Weighting by summed
    evidence weight instead means an evidence-free tag stays lightweight
    regardless of how many low-evidence votes are stuffed into it.
    """
    tag_scores, tag_w = [], []
    for code in np.unique(tag_codes):
        m = tag_codes == code
        tag_scores.append(weighted_median(scores[m], weights[m]))
        tag_w.append(float(weights[m].sum()))
    return weighted_median(np.array(tag_scores), np.array(tag_w))


def _ci(scores: np.ndarray, weights: np.ndarray, tag_codes: np.ndarray, cfg: Config,
        rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap CI over the vote array.

    A ratee voted on under a single tag (the common case) takes the fully
    vectorized `bootstrap_ci` path -- with one tag, `_total` reduces to a
    plain `weighted_median(scores, weights)`, so a single (chunked)
    multinomial draw computes the whole bootstrap distribution with no
    per-resample Python/pandas work. A ratee with votes under more than one
    tag needs the two-level (per-tag, then across-tag) statistic recomputed
    on every resample -- a resample can change which tags are even present
    in it -- so that path loops `_total` once per resample. Both paths take
    the caller-supplied `rng`.
    """
    n = len(scores)
    if n == 1 or cfg.bootstrap_n == 0:
        s = _total(scores, weights, tag_codes)
        return s, s
    if len(np.unique(tag_codes)) == 1:
        return bootstrap_ci(scores, weights, cfg.bootstrap_n, rng, cfg.ci_level)
    idx = rng.integers(0, n, size=(cfg.bootstrap_n, n))
    boots = np.array([_total(scores[i], weights[i], tag_codes[i]) for i in idx])
    a = (1 - cfg.ci_level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return float(lo), float(hi)


def _ratee_rng(cfg: Config, ratee: str) -> np.random.Generator:
    """A generator seeded from `(cfg.bootstrap_seed, crc32(ratee))`.

    Uses `zlib.crc32`, never the builtin `hash()` (randomized per process
    via `PYTHONHASHSEED` for `str`, so it would break cross-run
    determinism). Deterministic per ratee and independent of iteration
    order or of which other ratees are being scored in the same call.
    """
    return np.random.default_rng(
        np.random.SeedSequence([cfg.bootstrap_seed, zlib.crc32(str(ratee).encode())]))


def score(records: pd.DataFrame, cfg: Config = Config(), clusters: Optional[dict] = None,
          aggregator: Optional[Aggregator] = None) -> pd.DataFrame:
    """Score every ratee in `records`. One row per ratee, columns = RESULT_COLUMNS.

    Pipeline: validate_records -> drop revoked -> normalize -> evidence
    weights -> cluster assignment (from `clusters`, else each rater its own
    cluster) -> collapse to one vote per (ratee, tag, cluster) -> per-ratee
    `robust_score` = weighted median across tags of the per-tag weighted
    median of votes, each tag weighted by the SUM of its votes' evidence
    weights (see `_total`) -> bootstrap CI over the *votes* (collapsed
    per-cluster values), not the raw per-record rows, so sybil-farmed
    raters within one cluster cannot inflate the resample. `records` is
    never mutated.

    Determinism: each ratee gets its own `numpy.random.Generator`, seeded
    from `(cfg.bootstrap_seed, crc32(ratee))` (see `_ratee_rng`) -- so a
    ratee's row is bit-identical across calls regardless of input row
    order, regardless of which other ratees are present in the same call,
    and regardless of the order in which ratees happen to be processed.

    A ratee with fewer than `cfg.min_clusters` distinct clusters is
    `insufficient` (robust_score/ci_low/ci_high are NaN). `sybil_flag` is 1
    when the largest cluster's share of that ratee's raw records is >=
    `cfg.sybil_flag_share`. `naive_mean` is the unweighted mean of that
    ratee's normalized, uncollapsed, per-record scores (i.e. before the
    per-cluster vote collapse), kept for comparison against the robust
    estimate.

    `aggregator`, when given (not None), raises `NotImplementedError`: v1
    always uses the weighted median; custom aggregators arrive in v2.
    """
    if aggregator is not None:
        raise NotImplementedError("custom aggregators arrive in v2")
    df = prepare(records, cfg, clusters)
    if df.empty:
        empty = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in _RESULT_DTYPES.items()})
        return empty[RESULT_COLUMNS]
    votes = collapse(df)
    # Group votes by ratee once, up front: an O(1) dict lookup per ratee below
    # instead of rescanning the whole `votes` frame per ratee (which would be
    # O(n_ratees * n_votes) and too slow at ~28k ratees).
    votes_by_ratee = dict(tuple(votes.groupby("ratee", sort=False)))

    # Per-ratee stats, vectorized once over the whole frame (one groupby
    # pass covering all three "ratee"-keyed aggregates) rather than
    # recomputed from a raw per-ratee group in a Python loop.
    stats = df.groupby("ratee").agg(
        n_raw=("score", "size"),
        naive_mean=("score", "mean"),
        zero_evidence_ratio=("evidence_level", lambda s: (s == 0).mean()),
    )
    cluster_counts = df.groupby(["ratee", "cluster"]).size()
    max_cluster_share = cluster_counts.groupby(level=0).max() / stats["n_raw"]

    rows = []
    for ratee in stats.index:
        v = votes_by_ratee[ratee]
        n_clusters = int(v["cluster"].nunique())
        row = dict(ratee=ratee, n_clusters=n_clusters, n_raw=int(stats.at[ratee, "n_raw"]),
                   zero_evidence_ratio=float(stats.at[ratee, "zero_evidence_ratio"]),
                   sybil_flag=int(max_cluster_share.loc[ratee] >= cfg.sybil_flag_share),
                   naive_mean=float(stats.at[ratee, "naive_mean"]))
        if n_clusters < cfg.min_clusters:
            row.update(robust_score=np.nan, ci_low=np.nan, ci_high=np.nan, insufficient=1)
        else:
            s_arr = v["score"].to_numpy(dtype=float)
            w_arr = v["weight"].to_numpy(dtype=float)
            t_arr = pd.factorize(v["tag"])[0]
            s = _total(s_arr, w_arr, t_arr)
            lo, hi = _ci(s_arr, w_arr, t_arr, cfg, _ratee_rng(cfg, ratee))
            row.update(robust_score=s, ci_low=min(lo, s), ci_high=max(hi, s), insufficient=0)
        rows.append(row)
    out = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    for col in ("n_clusters", "n_raw", "sybil_flag", "insufficient"):
        out[col] = out[col].astype(int)
    return out
