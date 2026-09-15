import numpy as np
import pandas as pd
import pytest

from robustrep.config import Config
from robustrep.normalize import (
    RULE_BINARY,
    RULE_CONSTANT,
    RULE_PERCENT,
    RULE_RANK,
    RULE_UNIT,
    normalize,
)
from robustrep.schema import validate_records


def _n(records_factory, rows):
    return normalize(validate_records(records_factory(rows)))


def test_binary_group_passes_through(records_factory):
    df = _n(records_factory, [dict(rater="a", ratee="1", value=1), dict(rater="b", ratee="1", value=0)])
    assert list(df["score"]) == [1.0, 0.0]


def test_percent_group_divides_by_100(records_factory):
    df = _n(records_factory, [dict(rater="a", ratee="1", value=87), dict(rater="b", ratee="2", value=40)])
    assert list(df["score"]) == [0.87, 0.40]


def test_decimals_applied_then_percent(records_factory):
    # real values 90.0/95.0/100.0 under a d2 scale: "unit" does not fit, but the
    # any-decimals "percent" rung does, so these are absolute percentages rather
    # than a group-relative rank.
    rows = [dict(rater="a", ratee="1", value=9000, scale="d2", tag="uptime"),
            dict(rater="b", ratee="2", value=9500, scale="d2", tag="uptime"),
            dict(rater="c", ratee="3", value=10000, scale="d2", tag="uptime")]
    df = _n(records_factory, rows)
    assert np.allclose(df["score"], [0.90, 0.95, 1.00])
    assert list(df["norm_rule"]) == ["percent", "percent", "percent"]


def test_unit_range_passthrough(records_factory):
    rows = [dict(rater="a", ratee="1", value=50, scale="d2", tag="u"),
            dict(rater="b", ratee="2", value=75, scale="d2", tag="u")]
    df = _n(records_factory, rows)
    assert np.allclose(df["score"], [0.5, 0.75])
    assert list(df["norm_rule"]) == ["unit", "unit"]


def test_negative_outlier_tolerated_and_clipped_under_percent(records_factory):
    # n=3 tolerates one out-of-range record, so the -10 no longer drags the group
    # off "percent": it is clipped to 0 and the two honest values keep their
    # absolute percent scores.
    rows = [dict(rater="a", ratee="1", value=-10, tag="neg"),
            dict(rater="b", ratee="2", value=0, tag="neg"),
            dict(rater="c", ratee="3", value=10, tag="neg")]
    df = _n(records_factory, rows)
    assert np.allclose(df["score"], [0.0, 0.0, 0.1])
    assert list(df["norm_rule"]) == ["percent", "percent", "percent"]


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
    assert list(df["norm_rule"]) == ["percent", "percent"]

    rows = [dict(rater="a", ratee="1", value=500, tag="rev"), dict(rater="b", ratee="2", value=500, tag="rev")]
    df = _n(records_factory, rows)
    assert list(df["norm_rule"]) == ["constant", "constant"]

    rows = [dict(rater="a", ratee="1", value=1, tag="bin"), dict(rater="b", ratee="1", value=87, tag="q")]
    df = _n(records_factory, rows)
    assert list(df["norm_rule"]) == ["binary", "percent"]


def test_duplicate_index_groups_do_not_mix(records_factory):
    a = validate_records(records_factory([dict(rater="a", ratee="1", value=1, tag="quality"),
                                           dict(rater="b", ratee="1", value=0, tag="quality")]))
    b = validate_records(records_factory([dict(rater="c", ratee="1", value=87, tag="delivery"),
                                           dict(rater="d", ratee="1", value=40, tag="delivery")]))
    combined = pd.concat([a, b])
    assert list(combined.index) == [0, 1, 0, 1]
    df = normalize(combined)
    assert list(df["score"]) == [1.0, 0.0, 0.87, 0.40]
    assert list(df["norm_rule"]) == ["binary", "binary", "percent", "percent"]


def test_large_decimals_no_overflow(records_factory):
    df = _n(records_factory, [dict(rater="a", ratee="1", value=5, scale="d255", tag="x")])
    assert df["norm_rule"].iloc[0] == "unit" and np.isfinite(df["score"].iloc[0])


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


# --- C2: attacker-chosen int128 values must not steer the rule ladder ----------
#
# `value` is an arbitrary int128 chosen by the rater, so a single record can sit
# 10^36 away from every honest one. The tests below pin the fit-share/clipping
# behaviour that keeps such a record from deciding which rule the whole (tag,
# scale) group is scored under.


def _frame(values, scale="d0", tag="quality"):
    """One record per value, all inside a single (tag, scale) group.

    `value` is the raw on-chain integer; `scale` ("d0", "d2", ...) divides it by
    10**decimals to give the real value the rule ladder sees.
    """
    return validate_records(pd.DataFrame(
        [dict(rater=f"r{i}", ratee=str(i), value=v, scale=scale, tag=tag,
              ts=0, evidence_uri=None, source="test")
         for i, v in enumerate(values)]))


def test_one_extreme_value_does_not_change_rule_or_honest_scores():
    honest = [10, 50, 90, 100, 0, 75, 25, 60, 40, 80]
    base = normalize(_frame(honest))
    poisoned = normalize(_frame(honest + [2**127 - 1]))
    assert set(base["norm_rule"]) == {RULE_PERCENT}
    assert set(poisoned["norm_rule"]) == {RULE_PERCENT}
    np.testing.assert_allclose(poisoned["score"].iloc[:10], base["score"])
    assert poisoned["score"].iloc[10] == 1.0


def test_two_extreme_values_do_not_flatten_honest_scores():   # the exact C2 attack
    honest = [10, 50, 90, 100, 0, 75, 25, 60, 40, 80, 30, 70, 20, 65, 55, 45, 35, 85, 95, 5]
    poisoned = normalize(_frame(honest + [2**127 - 1, -(2**127)]))
    assert set(poisoned["norm_rule"]) == {RULE_PERCENT}
    assert len(set(poisoned["score"].iloc[:20].round(6))) == 20
    assert poisoned["score"].iloc[20] == 1.0 and poisoned["score"].iloc[21] == 0.0


def test_fallback_is_percentile_rank_not_minmax():
    out = normalize(_frame([1000, 2000, 3000, 10**30]))
    assert set(out["norm_rule"]) == {RULE_RANK}
    np.testing.assert_allclose(out["score"], [0.0, 1 / 3, 2 / 3, 1.0])


def test_rank_uses_average_ranks_for_ties():
    # no absolute rung fits (nothing is in [0, 100]), so the group reaches "rank".
    out = normalize(_frame([5000, 5000, 500000]))
    assert set(out["norm_rule"]) == {RULE_RANK}
    np.testing.assert_allclose(out["score"], [0.25, 0.25, 1.0])


def test_binary_group_maps_stray_value_as_percent():
    # The binary rung fits (one record outside {0, 1}, within the n=11 tolerance),
    # but the stray 7 is NOT clipped into {0, 1}: it takes the next absolute rung
    # for a d0 scale, "percent", so it scores 0.07. The 0s and 1s keep the binary map.
    out = normalize(_frame([0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 7]))
    assert set(out["norm_rule"]) == {RULE_BINARY}
    assert out["score"].iloc[-1] == 0.07
    np.testing.assert_allclose(out["score"].iloc[:10], [0, 1, 1, 0, 1, 1, 0, 1, 1, 1])


def test_binary_fitting_d2_group_maps_strays_by_unit_rule():
    # Same group shape under a d2 scale: the next absolute rule is "unit", not
    # "percent", so a stray real 0.6 stays 0.6 and a stray real 5 clips to 1.0.
    raw = [0, 100] * 9 + [60, 500]          # real: nine 0.0, nine 1.0, 0.6, 5.0
    out = normalize(_frame(raw, scale="d2"))
    assert set(out["norm_rule"]) == {RULE_BINARY}
    assert out["score"].iloc[-2] == 0.6
    assert out["score"].iloc[-1] == 1.0
    np.testing.assert_allclose(out["score"].iloc[:18], [0.0, 1.0] * 9)


def test_binary_flood_cannot_level_honest_percent_values():
    # The levelling attack: 50 all-zero rows carry an honest percent group onto the
    # binary rung (5 records outside {0, 1}, exactly the n=55 tolerance). Both rungs
    # involved map each record from its own value, so the honest 80s still score 0.8
    # rather than clipping to 1.0. Note this is not a general guarantee: a flood past
    # the tolerance can still re-level, via "rank" or the d0 binary/percent 100x
    # boundary -- see the module docstring in robustrep.normalize.
    out = normalize(_frame([80] * 5 + [0] * 50))
    assert set(out["norm_rule"]) == {RULE_BINARY}
    np.testing.assert_allclose(out["score"].iloc[:5], [0.8] * 5)
    assert (out["score"].iloc[5:] == 0.0).all()


def test_two_outliers_exceed_the_small_group_tolerance():
    # the designed limit: n=10 tolerates one out-of-range record, not two, so a
    # pair of extremes does drag the group onto the group-relative "rank" rung.
    honest = [10, 50, 90, 100, 0, 75, 25, 60]
    out = normalize(_frame(honest + [2**127 - 1, -(2**127)]))
    assert set(out["norm_rule"]) == {RULE_RANK}


def test_small_group_tolerates_one_outlier_under_percent():
    # 489 of 654 real (tag, scale) groups on Base hold <= 8 records, where a bare
    # 0.9 share test still lets ONE record flip the rule. The tolerance is a count,
    # so an attacker needs at least two records at every group size >= 2.
    honest = [10, 50, 90, 100, 0, 75, 25, 60]       # n=8
    base = normalize(_frame(honest))
    poisoned = normalize(_frame(honest + [2**127 - 1]))
    assert set(base["norm_rule"]) == {RULE_PERCENT}
    assert set(poisoned["norm_rule"]) == {RULE_PERCENT}
    np.testing.assert_allclose(poisoned["score"].iloc[:8], base["score"])
    assert poisoned["score"].iloc[8] == 1.0


def test_small_binary_group_tolerates_one_stray():
    # tag `execution_success` on Base: 8 records all value 1. One stray v=50 no
    # longer flips the group to "percent" (which would re-level honest 1.0 -> 0.01).
    out = normalize(_frame([1, 1, 0, 1, 1, 0, 1, 1, 50]))
    assert set(out["norm_rule"]) == {RULE_BINARY}
    np.testing.assert_allclose(out["score"].iloc[:8], [1, 1, 0, 1, 1, 0, 1, 1])
    assert out["score"].iloc[-1] == 0.5


def test_single_record_group_is_constant():
    # n=1 tolerates nothing, so no ranged rule fits and the lone record is neutral.
    out = normalize(_frame([500]))
    assert set(out["norm_rule"]) == {RULE_CONSTANT} and out["score"].iloc[0] == 0.5


def test_two_record_identical_group_is_constant():
    out = normalize(_frame([500, 500]))
    assert set(out["norm_rule"]) == {RULE_CONSTANT} and (out["score"] == 0.5).all()


def test_percent_rung_covers_decimal_scales():
    # F4: genuine percent data carried at d2 (real 99.77, 98.5, ...) has an
    # absolute rung of its own instead of falling through to "rank".
    raw = [9977, 9850, 10000, 9525, 0, 5000, 7550, 8880, 1230, 6660]
    out = normalize(_frame(raw, scale="d2"))
    assert set(out["norm_rule"]) == {RULE_PERCENT}
    np.testing.assert_allclose(
        out["score"], [0.9977, 0.985, 1.0, 0.9525, 0.0, 0.5, 0.755, 0.888, 0.123, 0.666])


def test_unit_wins_over_decimal_percent():
    # "unit" is tried before the any-decimals "percent" rung, so a d2 group already
    # confined to [0, 1] is passed through rather than divided by 100 again.
    raw = [50, 25, 75, 10, 90, 33, 66, 5, 95, 40]   # real 0.50, 0.25, ... all in [0, 1]
    out = normalize(_frame(raw, scale="d2"))
    assert set(out["norm_rule"]) == {RULE_UNIT}
    np.testing.assert_allclose(
        out["score"], [0.50, 0.25, 0.75, 0.10, 0.90, 0.33, 0.66, 0.05, 0.95, 0.40])


def test_unit_group_with_negative_outlier_clips_to_zero():
    # real values 0.20..0.95 plus -50, expressed under scale d2
    raw = [20, 35, 50, 65, 80, 95, 25, 40, 55, 70, -5000]
    out = normalize(_frame(raw, scale="d2"))
    assert set(out["norm_rule"]) == {RULE_UNIT}
    np.testing.assert_allclose(out["score"].iloc[:10],
                               [0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 0.25, 0.40, 0.55, 0.70])
    assert out["score"].iloc[-1] == 0.0


def test_constant_group_is_neutral():
    out = normalize(_frame([500, 500, 500]))
    assert set(out["norm_rule"]) == {RULE_CONSTANT} and (out["score"] == 0.5).all()


def test_normalize_validates_fit_share():
    # normalize() is callable directly, not only through Config, so it range-checks
    # `fit_share` with the same validator Config uses.
    with pytest.raises(ValueError):
        normalize(_frame([10, 20, 30]), fit_share=0.5)
    with pytest.raises(ValueError):
        normalize(_frame([10, 20, 30]), fit_share=1.1)


def test_fit_share_is_configurable_and_validated():
    # n=10 tolerates 1 outlier at the 0.9 default but 2 at 0.75, so the same pair
    # of extremes falls to "rank" under the default and stays "percent" at 0.75.
    rows = [10, 50, 90, 100, 0, 75, 25, 60] + [2**127 - 1, -(2**127)]
    assert set(normalize(_frame(rows))["norm_rule"]) == {RULE_RANK}
    out = normalize(_frame(rows), fit_share=0.75)
    assert set(out["norm_rule"]) == {RULE_PERCENT}
    with pytest.raises(ValueError):
        Config(norm_fit_share=0.5)
    with pytest.raises(ValueError):
        Config(norm_fit_share=1.1)
