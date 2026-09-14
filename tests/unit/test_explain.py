import math

import pandas as pd

from robustrep import Config
from robustrep.explain import COLUMNS, explain


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
    df = records_factory([
        dict(rater="h1", ratee="A", value=80, evidence_level=3),
        dict(rater="h2", ratee="A", value=10, evidence_level=0),
    ])
    out = explain(df, "A", Config(bootstrap_n=0),
                  clusters={"h1": "h1", "h2": "h2"})
    top = out.sort_values("contribution", ascending=False).iloc[0]
    assert top["rater"] == "h1"
    assert top["weight"] == out["weight"].max()


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
