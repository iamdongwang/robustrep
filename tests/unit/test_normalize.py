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
    rows = [dict(rater="a", ratee="1", value=9977, scale="d2", tag="uptime"),
            dict(rater="b", ratee="2", value=9000, scale="d2", tag="uptime")]
    df = _n(records_factory, rows)
    assert np.isclose(df["score"].max(), 1.0) and np.isclose(df["score"].min(), 0.0)


def test_constant_group_is_half(records_factory):
    rows = [dict(rater="a", ratee="1", value=500, tag="rev"), dict(rater="b", ratee="2", value=500, tag="rev")]
    df = _n(records_factory, rows)
    assert list(df["score"]) == [0.5, 0.5]


def test_groups_do_not_mix(records_factory):
    rows = [dict(rater="a", ratee="1", value=1, tag="bin"), dict(rater="b", ratee="1", value=87, tag="q")]
    df = _n(records_factory, rows)
    assert list(df["score"]) == [1.0, 0.87]


def test_empty_frame_returns_score_column(records_factory):
    validated = validate_records(records_factory([dict(rater="a", ratee="1", value=87)]))
    empty = validated.iloc[0:0]
    df = normalize(empty)
    assert "score" in df.columns
    assert len(df) == 0


def test_input_not_mutated(records_factory):
    validated = validate_records(records_factory([dict(rater="a", ratee="1", value=87),
                                                    dict(rater="b", ratee="2", value=40)]))
    before = validated.copy()
    normalize(validated)
    pd.testing.assert_frame_equal(validated, before)
