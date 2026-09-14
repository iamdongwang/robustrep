import math

import pandas as pd

from robustrep import Config
from robustrep.explain import COLUMNS, explain
from robustrep.pipeline import score


def test_explain_marks_collapsed_and_weights(records_factory):
    df = records_factory([dict(rater="h1", ratee="A", value=80, evidence_level=3),
                          dict(rater="s1", ratee="A", value=0), dict(rater="s2", ratee="A", value=0)])
    out = explain(df, "A", Config(bootstrap_n=0), clusters={"s1": "S", "s2": "S", "h1": "h1"})
    out = out.set_index("rater")
    assert out.loc["h1", "weight"] == 1.0 and out.loc["s1", "weight"] == 0.1
    assert bool(out.loc["s1", "collapsed"]) and not bool(out.loc["h1", "collapsed"])
    assert math.isclose(out["contribution"].sum(), 1.0)
    # h1 is one vote of weight 1.0; S is one vote of weight 0.1 -> h1 share = 1.0/1.1
    assert math.isclose(out.loc["h1", "contribution"], 1.0 / 1.1)


def test_explain_unknown_ratee_is_empty(records_factory):
    out = explain(records_factory([dict(rater="a", ratee="A", value=1)]), "Z", Config(bootstrap_n=0))
    assert out.empty
    assert list(out.columns) == COLUMNS
    expected_dtypes = dict(rater=object, tag=object, score=float, norm_rule=object,
                            evidence_level=int, weight=float, cluster=object, collapsed=bool,
                            contribution=float, is_median_vote=bool)
    for col, dt in expected_dtypes.items():
        assert out[col].dtype == dt, f"{col}: expected {dt}, got {out[col].dtype}"


def test_contributions_sum_to_one_multi_tag(records_factory):
    df = records_factory([
        dict(rater="h1", ratee="A", value=80, evidence_level=3, tag="quality"),
        dict(rater="h2", ratee="A", value=60, evidence_level=2, tag="quality"),
        dict(rater="s1", ratee="A", value=0, tag="speed"),
        dict(rater="s2", ratee="A", value=0, tag="speed"),
    ])
    out = explain(df, "A", Config(bootstrap_n=0),
                  clusters={"h1": "h1", "h2": "h2", "s1": "S", "s2": "S"})
    assert math.isclose(out["contribution"].sum(), 1.0)
    assert (out["contribution"] >= 0).all()


def test_revoked_records_excluded(records_factory):
    df = records_factory([
        dict(rater="h1", ratee="A", value=80, evidence_level=3),
        dict(rater="h2", ratee="A", value=10, revoked=1),
    ])
    out = explain(df, "A", Config(bootstrap_n=0))
    assert "h2" not in set(out["rater"])
    assert list(out["rater"]) == ["h1"]


def test_columns_and_order(records_factory):
    df = records_factory([dict(rater="h1", ratee="A", value=80, evidence_level=3)])
    out = explain(df, "A", Config(bootstrap_n=0))
    assert list(out.columns) == COLUMNS


def test_explain_consistent_with_score(records_factory):
    # (a) single-tag case: the records marked is_median_vote are the ones score()
    # actually picked -- the median of their scores equals robust_score.
    cfg = Config(bootstrap_n=0, min_clusters=1)
    df = records_factory([
        dict(rater="h1", ratee="A", value=80, evidence_level=3),
        dict(rater="h2", ratee="A", value=10, evidence_level=0),
    ])
    out = explain(df, "A", cfg, clusters={"h1": "h1", "h2": "h2"})
    top = out.sort_values("contribution", ascending=False).iloc[0]
    assert top["rater"] == "h1"
    assert top["weight"] == out["weight"].max()
    robust_score = score(df, cfg, clusters={"h1": "h1", "h2": "h2"}
                          ).set_index("ratee").loc["A", "robust_score"]
    marked = out[out["is_median_vote"]]
    assert not marked.empty
    assert math.isclose(marked["score"].median(), robust_score)
    assert marked["rater"].tolist() == ["h1"]

    # (b) reviewer's example: one tag, three votes -- a (score 0.1, weight 1.0),
    # b (score 0.5, weight 0.1), c (score 0.9, weight 1.0). weighted median of
    # [0.1, 0.5, 0.9] under [1.0, 0.1, 1.0] is 0.5 -> only rater b is selected.
    df2 = records_factory([
        dict(rater="a", ratee="B", value=10, scale="d2", evidence_level=3, tag="quality"),
        dict(rater="b", ratee="B", value=50, scale="d2", evidence_level=0, tag="quality"),
        dict(rater="c", ratee="B", value=90, scale="d2", evidence_level=3, tag="quality"),
    ])
    out2 = explain(df2, "B", Config(bootstrap_n=0))
    robust_score2 = score(df2, Config(bootstrap_n=0)).set_index("ratee").loc["B", "robust_score"]
    assert math.isclose(robust_score2, 0.5)
    assert out2.loc[out2["is_median_vote"], "rater"].tolist() == ["b"]
    assert int(out2["is_median_vote"].sum()) == 1


def test_is_median_vote_multi_tag(records_factory):
    # tag q: 3 votes, scores 0.2/0.4/0.9, evidence levels 3/3/0 -> weights 1.0/1.0/0.1
    # tag u: 2 votes, scores 0.6/0.8, evidence levels 1/1 -> weights 0.3/0.3
    # distinct raters/clusters (no `clusters` passed -> each rater its own cluster).
    # per-tag weighted median: q -> 0.4 (weight 2.1), u -> 0.6 (weight 0.6); across
    # tags, weighted median of [0.4, 0.6] under [2.1, 0.6] is 0.4 -> rater q2's vote.
    df = records_factory([
        dict(rater="q1", ratee="C", value=20, evidence_level=3, tag="q"),
        dict(rater="q2", ratee="C", value=40, evidence_level=3, tag="q"),
        dict(rater="q3", ratee="C", value=90, evidence_level=0, tag="q"),
        dict(rater="u1", ratee="C", value=60, evidence_level=1, tag="u"),
        dict(rater="u2", ratee="C", value=80, evidence_level=1, tag="u"),
    ])
    cfg = Config(bootstrap_n=0)
    out = explain(df, "C", cfg)
    marked = out[out["is_median_vote"]]
    # exactly one (tag, cluster) vote is marked
    assert marked[["tag", "cluster"]].drop_duplicates().shape[0] == 1
    assert marked["rater"].tolist() == ["q2"]
    robust_score = score(df, cfg).set_index("ratee").loc["C", "robust_score"]
    assert math.isclose(robust_score, 0.4)
    assert math.isclose(marked["score"].median(), robust_score)


def test_ratee_int_accepted(records_factory):
    df = records_factory([dict(rater="h1", ratee=1, value=80, evidence_level=3)])
    out = explain(df, 1, Config(bootstrap_n=0))
    assert not out.empty
    assert list(out["rater"]) == ["h1"]


def test_input_not_mutated(records_factory):
    df = records_factory([dict(rater="h1", ratee="A", value=80, evidence_level=3)])
    before = df.copy(deep=True)
    explain(df, "A", Config(bootstrap_n=0))
    pd.testing.assert_frame_equal(df, before)
