"""Per-record breakdown of how one ratee's score was formed."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .aggregate import weighted_median_index
from .config import Config
from .pipeline import collapse, prepare

COLUMNS = ["rater", "tag", "score", "norm_rule", "evidence_level", "weight", "cluster", "collapsed",
           "contribution", "is_median_vote"]

# Mirrors pipeline._RESULT_DTYPES: gives the empty-input result frame real dtypes
# instead of `object` on every column, so callers can check dtype before checking
# row count the same way they do for score()'s empty frame.
_EXPLAIN_DTYPES = {
    "rater": object,
    "tag": object,
    "score": float,
    "norm_rule": object,
    "evidence_level": int,
    "weight": float,
    "cluster": object,
    "collapsed": bool,
    "contribution": float,
    "is_median_vote": bool,
}


def _selected_vote(votes: pd.DataFrame) -> tuple[str, str]:
    """Return the (tag, cluster) of the vote selected by score()'s weighted-median chain.

    Mirrors `pipeline._total` exactly: for each tag, the vote chosen by
    `weighted_median_index` among that tag's votes (its score is that tag's
    per-tag weighted median); then the tag chosen by `weighted_median_index`
    over those per-tag scores, weighted by each tag's summed vote weight.
    """
    tags = votes["tag"].to_numpy()
    scores = votes["score"].to_numpy(dtype=float)
    weights = votes["weight"].to_numpy(dtype=float)
    clusters = votes["cluster"].to_numpy()
    uniq_tags = pd.unique(tags)
    tag_scores = np.empty(len(uniq_tags))
    tag_weights = np.empty(len(uniq_tags))
    tag_vote_idx = np.empty(len(uniq_tags), dtype=int)
    for i, t in enumerate(uniq_tags):
        m = np.flatnonzero(tags == t)
        local_idx = weighted_median_index(scores[m], weights[m])
        global_idx = m[local_idx]
        tag_vote_idx[i] = global_idx
        tag_scores[i] = scores[global_idx]
        tag_weights[i] = weights[m].sum()
    winner_tag_i = weighted_median_index(tag_scores, tag_weights)
    winner_vote_idx = tag_vote_idx[winner_tag_i]
    return tags[winner_vote_idx], clusters[winner_vote_idx]


def explain(records: pd.DataFrame, ratee: str, cfg: Config = Config(),
            clusters: Optional[dict] = None) -> pd.DataFrame:
    """One row per non-revoked record of `ratee`. `contribution` sums to 1 over the ratee.

    `contribution` is the record's WEIGHT SHARE of the ratee's total evidence mass --
    it mirrors exactly how `score()` weights votes within a tag and tags against each
    other -- it is *not* a claim that the record moved (or could move) the final score:
    the aggregate is a weighted median, so a record can carry a large contribution and
    still have no effect on `robust_score` (its vote wasn't the one selected), and
    conversely a small-contribution record can belong to the selected vote. Use
    `is_median_vote` (below) to see which records actually determined the score.

    contribution = (record weight / sum of record weights in its (tag, cluster))
                   * (that vote's weight / sum of all vote weights for the ratee).

    `is_median_vote` (bool): True for every record belonging to the single vote that
    `score()`'s weighted-median chain actually selects for this ratee -- the vote
    chosen, within its tag, by the per-tag weighted median, in the tag chosen, across
    tags, by the weighted median of per-tag scores (tag weight = summed vote weight;
    see `pipeline._total`). Exactly one (tag, cluster) vote is selected, so exactly the
    records collapsed into that one vote are marked True; all others are False.

    `records` must be the same frame you pass (or would pass) to `score()` --
    normalization is dataset-relative (`normalize()` groups by (tag, scale) across the
    whole input, see `robustrep.normalize`), so filtering `records` down to one ratee
    before calling `explain` can change `score`/`norm_rule` for the surviving rows.
    Pass `ratee` exactly as it appears in `score()`'s output (both are coerced via
    `str()`, so e.g. an int ratee matches its stringified form). Runtime is O(len(records)):
    a single pass through `prepare`/`collapse`, no per-ratee rescans.

    `records` is never mutated.
    """
    df = prepare(records, cfg, clusters)
    df = df[df["ratee"] == str(ratee)].copy()
    if df.empty:
        empty = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in _EXPLAIN_DTYPES.items()})
        return empty[COLUMNS]
    votes = collapse(df)
    v_total = float(votes["weight"].sum())
    vote_w = votes.set_index(["tag", "cluster"])["weight"]
    key = pd.MultiIndex.from_arrays([df["tag"], df["cluster"]])
    in_cluster_w = df.groupby(["tag", "cluster"])["weight"].transform("sum").to_numpy()
    cluster_size = df.groupby(["tag", "cluster"])["weight"].transform("size").to_numpy()
    df["collapsed"] = cluster_size > 1
    df["contribution"] = (df["weight"].to_numpy() / in_cluster_w) * (vote_w.reindex(key).to_numpy() / v_total)
    sel_tag, sel_cluster = _selected_vote(votes)
    df["is_median_vote"] = (df["tag"].to_numpy() == sel_tag) & (df["cluster"].to_numpy() == sel_cluster)
    return df[COLUMNS].reset_index(drop=True)
