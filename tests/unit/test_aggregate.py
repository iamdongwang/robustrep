import numpy as np
import pytest

from robustrep.aggregate import Aggregator, WeightedMedian, weighted_median


def test_weighted_median_equal_weights_is_median():
    assert weighted_median(np.array([0.1, 0.9, 0.5]), np.ones(3)) == 0.5


def test_weighted_median_respects_weights():
    # heavy weight on 0.9 pulls the median there
    assert weighted_median(np.array([0.1, 0.9]), np.array([1.0, 3.0])) == 0.9


def test_single_value():
    assert weighted_median(np.array([0.42]), np.array([0.1])) == 0.42


def test_one_outlier_cannot_move_median():
    vals = np.array([0.8] * 9 + [0.0])
    assert weighted_median(vals, np.ones(10)) == 0.8


def test_bootstrap_ci_contains_point_and_is_ordered(rng):
    agg = WeightedMedian(n_boot=200, seed=0)
    vals = rng.uniform(0.6, 0.8, size=50)
    score, lo, hi = agg.aggregate(vals, np.ones(50))
    assert lo <= score <= hi and 0.6 <= lo and hi <= 0.8


def test_aggregator_is_abstract():
    with pytest.raises(TypeError):
        Aggregator()


def test_weighted_median_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        weighted_median(np.array([]), np.array([]))


def test_weighted_median_rejects_bad_weights():
    with pytest.raises(ValueError):
        weighted_median(np.array([0.1, 0.2]), np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        weighted_median(np.array([0.1, 0.2]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        weighted_median(np.array([0.1, np.nan]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        weighted_median(np.array([0.1, 0.2]), np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        weighted_median(np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.0]))


def test_weighted_median_rejects_inf_value():
    with pytest.raises(ValueError, match="non-finite"):
        weighted_median(np.array([0.1, np.inf]), np.ones(2))


def test_weighted_median_rejects_inf_weight():
    with pytest.raises(ValueError, match="non-finite"):
        weighted_median(np.array([0.1, 0.2]), np.array([1.0, np.inf]))


def test_weighted_median_ties_and_even_counts():
    # lower-median convention: side="left"
    assert weighted_median(np.array([0.2, 0.8]), np.ones(2)) == 0.2
    assert weighted_median(np.array([0.2, 0.8]), np.array([1.0, 1.0001])) == 0.8


def test_weighted_median_ignores_input_order():
    vals = np.array([0.3, 0.1, 0.9, 0.5, 0.7])
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    order = np.array([3, 0, 4, 1, 2])
    assert weighted_median(vals, weights) == weighted_median(vals[order], weights[order])


def test_weighted_median_matches_numpy_median_for_odd_equal_weights(rng):
    vals = rng.uniform(0, 1, size=101)
    assert weighted_median(vals, np.ones(101)) == np.median(vals)


def test_bootstrap_is_deterministic(rng):
    vals = rng.uniform(0, 1, size=30)
    weights = np.ones(30)
    a1 = WeightedMedian(n_boot=100, seed=42)
    a2 = WeightedMedian(n_boot=100, seed=42)
    a3 = WeightedMedian(n_boot=100, seed=43)
    r1 = a1.aggregate(vals, weights)
    r2 = a2.aggregate(vals, weights)
    r3 = a3.aggregate(vals, weights)
    assert r1 == r2
    assert r1 != r3


def test_aggregate_zero_boot_returns_point():
    agg = WeightedMedian(n_boot=0, seed=0)
    score, lo, hi = agg.aggregate(np.array([0.1, 0.5, 0.9]), np.ones(3))
    assert lo == hi == score


def test_aggregate_ci_level_validated():
    with pytest.raises(ValueError):
        WeightedMedian(ci_level=1.5)
    with pytest.raises(ValueError):
        WeightedMedian(n_boot=-1)


def test_aggregate_does_not_mutate_inputs():
    vals = np.array([0.3, 0.1, 0.9])
    weights = np.array([1.0, 2.0, 1.0])
    vals_copy, weights_copy = vals.copy(), weights.copy()
    agg = WeightedMedian(n_boot=50, seed=0)
    agg.aggregate(vals, weights)
    np.testing.assert_array_equal(vals, vals_copy)
    np.testing.assert_array_equal(weights, weights_copy)
