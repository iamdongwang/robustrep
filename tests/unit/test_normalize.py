import numpy as np
import pandas as pd

from robustrep.normalize import normalize
from robustrep.schema import validate_records


def _n(records_factory, rows):
    return normalize(validate_records(records_factory(rows)))


def test_binary_group_passes_through(records_factory):
    df = _n(records_factory, [dict(rater="a", ratee="1", value=1), dict(rater="b", ratee="1", value=0)])
    assert list(df["score"]) == [1.0, 0.0]


def test_percent_group_divides_by_100(records_factory):
    df = _n(records_factory, [dict(rater="a", ratee="1", value=87), dict(rater="b", ratee="2", value=40)])
    assert list(df["score"]) == [0.87, 0.40]


def test_decimals_applied_then_minmax(records_factory):
    rows = [dict(rater="a", ratee="1", value=9000, scale="d2", tag="uptime"),
            dict(rater="b", ratee="2", value=9500, scale="d2", tag="uptime"),
            dict(rater="c", ratee="3", value=10000, scale="d2", tag="uptime")]
    df = _n(records_factory, rows)
    assert np.allclose(df["score"], [0.0, 0.5, 1.0])
    assert list(df["norm_rule"]) == ["minmax", "minmax", "minmax"]


def test_unit_range_passthrough(records_factory):
    rows = [dict(rater="a", ratee="1", value=50, scale="d2", tag="u"),
            dict(rater="b", ratee="2", value=75, scale="d2", tag="u")]
    df = _n(records_factory, rows)
    assert np.allclose(df["score"], [0.5, 0.75])
    assert list(df["norm_rule"]) == ["unit", "unit"]


def test_negative_range_uses_minmax(records_factory):
    rows = [dict(rater="a", ratee="1", value=-10, tag="neg"),
            dict(rater="b", ratee="2", value=0, tag="neg"),
            dict(rater="c", ratee="3", value=10, tag="neg")]
    df = _n(records_factory, rows)
    assert np.allclose(df["score"], [0.0, 0.5, 1.0])
    assert list(df["norm_rule"]) == ["minmax", "minmax", "minmax"]


def test_constant_group_is_half(records_factory):
    rows = [dict(rater="a", ratee="1", value=500, tag="rev"), dict(rater="b", ratee="2", value=500, tag="rev")]
    df = _n(records_factory, rows)
    assert list(df["score"]) == [0.5, 0.5]


def test_groups_do_not_mix(records_factory):
    rows = [dict(rater="a", ratee="1", value=1, tag="bin"), dict(rater="b", ratee="1", value=87, tag="q")]
    df = _n(records_factory, rows)
    assert list(df["score"]) == [1.0, 0.87]


def test_norm_rule_recorded(records_factory):
    df = _n(records_factory, [dict(rater="a", ratee="1", value=1), dict(rater="b", ratee="1", value=0)])
    assert list(df["norm_rule"]) == ["binary", "binary"]

    df = _n(records_factory, [dict(rater="a", ratee="1", value=87), dict(rater="b", ratee="2", value=40)])
    assert list(df["norm_rule"]) == ["percent", "percent"]

    rows = [dict(rater="a", ratee="1", value=9977, scale="d2", tag="uptime"),
            dict(rater="b", ratee="2", value=9000, scale="d2", tag="uptime")]
    df = _n(records_factory, rows)
    assert list(df["norm_rule"]) == ["minmax", "minmax"]

    rows = [dict(rater="a", ratee="1", value=500, tag="rev"), dict(rater="b", ratee="2", value=500, tag="rev")]
    df = _n(records_factory, rows)
    assert list(df["norm_rule"]) == ["constant", "constant"]

    rows = [dict(rater="a", ratee="1", value=1, tag="bin"), dict(rater="b", ratee="1", value=87, tag="q")]
    df = _n(records_factory, rows)
    assert list(df["norm_rule"]) == ["binary", "percent"]


def test_duplicate_index_groups_do_not_mix(records_factory):
    a = validate_records(records_factory([dict(rater="a", ratee="1", value=1, tag="quality"),
                                           dict(rater="b", ratee="2", value=0, tag="quality")]))
    b = validate_records(records_factory([dict(rater="c", ratee="3", value=1, tag="delivery"),
                                           dict(rater="d", ratee="4", value=0, tag="delivery")]))
    combined = pd.concat([a, b])
    assert list(combined.index) == [0, 1, 0, 1]
    df = normalize(combined)
    assert list(df["score"]) == [1.0, 0.0, 1.0, 0.0]


def test_empty_frame_returns_score_column(records_factory):
    validated = validate_records(records_factory([dict(rater="a", ratee="1", value=87)]))
    empty = validated.iloc[0:0]
    df = normalize(empty)
    assert "score" in df.columns
    assert "norm_rule" in df.columns
    assert len(df) == 0


def test_input_not_mutated(records_factory):
    validated = validate_records(records_factory([dict(rater="a", ratee="1", value=87),
                                                    dict(rater="b", ratee="2", value=40)]))
    before = validated.copy()
    normalize(validated)
    pd.testing.assert_frame_equal(validated, before)
