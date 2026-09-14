"""Per-record breakdown of how one ratee's score was formed."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import Config
from .pipeline import collapse, prepare

COLUMNS = ["rater", "tag", "score", "norm_rule", "evidence_level", "weight", "cluster", "collapsed",
           "contribution"]


def explain(records: pd.DataFrame, ratee: str, cfg: Config = Config(),
            clusters: Optional[dict] = None) -> pd.DataFrame:
    """One row per non-revoked record of `ratee`. `contribution` sums to 1 over the ratee.

    contribution = (record weight / sum of record weights in its (tag, cluster))
                   * (that vote's weight / sum of all vote weights for the ratee).
    This mirrors score(): votes are weighted by evidence mass within a tag and tags by their
    total evidence mass. `records` is never mutated.
    """
    df = prepare(records, cfg, clusters)
    df = df[df["ratee"] == str(ratee)].copy()
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    votes = collapse(df)
    v_total = float(votes["weight"].sum())
    vote_w = votes.set_index(["tag", "cluster"])["weight"]
    key = pd.MultiIndex.from_arrays([df["tag"], df["cluster"]])
    in_cluster_w = df.groupby(["tag", "cluster"])["weight"].transform("sum").to_numpy()
    cluster_size = df.groupby(["tag", "cluster"])["weight"].transform("size").to_numpy()
    df["collapsed"] = cluster_size > 1
    df["contribution"] = (df["weight"].to_numpy() / in_cluster_w) * (vote_w.reindex(key).to_numpy() / v_total)
    return df[COLUMNS].reset_index(drop=True)
