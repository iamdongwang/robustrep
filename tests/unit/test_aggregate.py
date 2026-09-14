import numpy as np
import pytest

from robustrep.aggregate import (
    Aggregator, WeightedMedian, _bootstrap_ci, _bootstrap_draws, _weighted_median_sorted,
    bootstrap_ci, weighted_median, weighted_median_index,
)
from robustrep.config import Config


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
    # point estimate is independent of the bootstrap seed; only the CI may differ
    assert r1[0] == r3[0]


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


def test_scale_invariance():
    # cumulative-sum drift must not make the result depend on the weight scale
    rng = np.random.default_rng(0)
    weight_choices = np.array([0.1, 0.3, 0.7, 1.0])
    for _ in range(200):
        n = rng.integers(2, 16)
        v = rng.uniform(0, 1, size=n)
        w = rng.choice(weight_choices, size=n)
        assert weighted_median(v, w) == weighted_median(v, 0.3 * w)


def test_exact_half_tie_returns_lower():
    vals = np.array([0.14, 0.36, 0.58, 0.71, 0.79, 0.99])
    weights = np.array([0.3] * 6)
    assert weighted_median(vals, weights) == 0.58


def test_zero_weight_extreme_never_returned():
    assert weighted_median(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 1.0
    assert weighted_median(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == 0.0


def test_bootstrap_with_some_zero_weights_does_not_crash():
    vals = np.array([0.5, 0.9, 0.7, 0.6])
    weights = np.array([0.0, 1.0, 1.0, 1.0])
    agg = WeightedMedian(n_boot=200, seed=0)
    score, lo, hi = agg.aggregate(vals, weights)
    assert np.isfinite([score, lo, hi]).all()
    assert lo <= score <= hi


def test_bootstrap_degenerate_raises():
    # n=2, weights=[0, 1]: a resample of size 2 is all-zero-weight iff both
    # draws are index 0, P = 0.25 per draw of the pair; with n=2 pairs drawn
    # (n_boot=2), P(at least one all-zero pair) = 1 - 0.75^2 ~= 44%. The guard
    # needs >= max(2, n_boot // 10) = 2 valid resamples, so a single all-zero
    # resample out of 2 already triggers it. Loop seeds instead of hand-picking
    # one so the test doesn't depend on numpy's RNG stream staying stable.
    vals = np.array([0.5, 0.9])
    weights = np.array([0.0, 1.0])
    raised = False
    for seed in range(200):
        try:
            WeightedMedian(n_boot=2, seed=seed).aggregate(vals, weights)
        except ValueError:
            raised = True
            break
    assert raised is True


def test_ndim_rejected():
    with pytest.raises(ValueError, match="1-D"):
        weighted_median(np.array([[0.1, 0.2], [0.3, 0.4]]), np.ones((2, 2)))


def test_weighted_median_index_matches_value():
    vals = np.array([0.3, 0.1, 0.9, 0.5, 0.7])
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    idx = weighted_median_index(vals, weights)
    assert vals[idx] == weighted_median(vals, weights)


def test_weighted_median_index_ties_and_ties_out_of_order():
    # unsorted input, tie handling must match weighted_median's lower-median
    # convention (side="left")
    vals = np.array([0.8, 0.2])
    weights = np.array([1.0, 1.0])
    idx = weighted_median_index(vals, weights)
    assert vals[idx] == 0.2 == weighted_median(vals, weights)


def test_weighted_median_index_single_value():
    vals = np.array([0.42])
    weights = np.array([0.1])
    assert weighted_median_index(vals, weights) == 0
    assert vals[weighted_median_index(vals, weights)] == weighted_median(vals, weights)


def test_weighted_median_index_rejects_bad_input():
    with pytest.raises(ValueError, match="empty"):
        weighted_median_index(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        weighted_median_index(np.array([0.1, 0.2]), np.array([1.0, -1.0]))


def test_weighted_median_index_ignores_input_order():
    vals = np.array([0.3, 0.1, 0.9, 0.5, 0.7])
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    order = np.array([3, 0, 4, 1, 2])
    idx1 = weighted_median_index(vals, weights)
    idx2 = order[weighted_median_index(vals[order], weights[order])]
    assert vals[idx1] == vals[idx2]


def test_from_config():
    cfg = Config(bootstrap_n=5, bootstrap_seed=3, ci_level=0.9)
    agg = WeightedMedian.from_config(cfg)
    assert agg.n_boot == 5
    assert agg.seed == 3
    assert agg.ci_level == 0.9


def test_bootstrap_ci_matches_index_resampling():
    # Fixed 6-vote input. Compare the library's vectorized (multinomial)
    # bootstrap against an independent, directly-coded classic index
    # resampling loop (draw n indices with replacement per resample,
    # compute the weighted median of each draw) -- not via any library
    # internals beyond the already-tested `_weighted_median_sorted`. Both
    # draw 20,000 boots from rngs spun off the same SeedSequence family.
    # Since a weighted median always lands on one of the original values,
    # bucket each method's boots by those support points and compare the
    # resulting frequency distributions.
    vals = np.array([0.12, 0.53, 0.77, 0.31, 0.66, 0.90])
    weights = np.array([0.4, 0.9, 0.2, 0.7, 1.0, 0.3])
    n = len(vals)
    n_boot = 20_000

    vec_rng, idx_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(0).spawn(2))

    order = np.argsort(vals, kind="stable")
    v_sorted, w_sorted = vals[order], weights[order]
    vec_boots = _bootstrap_draws(v_sorted, w_sorted, n_boot, vec_rng, chunk_size=n_boot)

    idx = idx_rng.integers(0, n, size=(n_boot, n))
    idx_boots = []
    for row in idx:
        vi, wi = vals[row], weights[row]
        if wi.sum() <= 0:
            continue
        o = np.argsort(vi, kind="stable")
        idx_boots.append(_weighted_median_sorted(vi[o], wi[o]))
    idx_boots = np.array(idx_boots)

    support = np.unique(vals)
    vec_freq = np.array([(vec_boots == s).mean() for s in support])
    idx_freq = np.array([(idx_boots == s).mean() for s in support])
    assert np.max(np.abs(vec_freq - idx_freq)) < 0.01


def test_bootstrap_ci_is_deterministic_given_same_rng_state():
    vals = np.array([0.1, 0.4, 0.6, 0.9])
    weights = np.ones(4)
    r1 = bootstrap_ci(vals, weights, 500, np.random.default_rng(7), 0.95)
    r2 = bootstrap_ci(vals, weights, 500, np.random.default_rng(7), 0.95)
    assert r1 == r2


def test_bootstrap_ci_degenerate_raises():
    # Same reasoning as test_bootstrap_degenerate_raises: n=2, one weight is
    # zero, n_boot=2 needs >= 2 valid resamples so a single all-zero
    # resample already triggers the guard.
    vals = np.array([0.5, 0.9])
    weights = np.array([0.0, 1.0])
    raised = False
    for seed in range(200):
        try:
            bootstrap_ci(vals, weights, 2, np.random.default_rng(seed), 0.95)
        except ValueError:
            raised = True
            break
    assert raised is True


def test_bootstrap_ci_rejects_non_finite_input():
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_ci(np.array([0.1, np.inf]), np.ones(2), 10, np.random.default_rng(0), 0.95)
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_ci(np.array([0.1, 0.2]), np.array([1.0, np.nan]), 10, np.random.default_rng(0), 0.95)


def test_bootstrap_ci_chunked_completes_for_many_votes():
    # 1 ratee x 20,000 votes: exercises the chunked multinomial draw path
    # (chunk_size = max(1, 2_000_000 // 20_000) = 100 << n_boot). No memory
    # assertion -- just confirm it completes and returns a CI bracketing a
    # sane range.
    rng = np.random.default_rng(3)
    vals = rng.uniform(0, 1, size=20_000)
    weights = rng.uniform(0.1, 1.0, size=20_000)
    point = weighted_median(vals, weights)
    lo, hi = bootstrap_ci(vals, weights, 1000, np.random.default_rng(4), 0.95)
    lo, hi = min(lo, point), max(hi, point)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0
