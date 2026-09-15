"""Aggregators turn (values, weights) into (score, ci_low, ci_high).

The arithmetic mean has breakdown point zero: a single extreme rating can move
it arbitrarily far. The weighted median has breakdown point ~0.5 (roughly half
the total weight must be adversarial to move it), so it is the default
aggregator here. `Aggregator` is an abstract base so Bayesian and graph-based
aggregators can be added later behind the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import Config

# Half-weight threshold is nudged down by this relative fraction so that
# float64 cumsum drift at an exact tie doesn't step one element too far
# (which would break scale invariance and bias the result upward).
_TIE_EPS = 1e-9

# Minimum fraction of bootstrap resamples that must carry non-zero total
# weight for the CI to be considered meaningful.
MIN_VALID_BOOT_FRACTION = 0.1

# Cap on the number of (resample x vote) cells materialized at once by a
# chunked bootstrap draw, so peak memory stays bounded even for a ratee with
# thousands of votes: a chunk holds at most this many cells regardless of how
# large `n_boot` is. Public because `robustrep.pipeline._ci` chunks its
# multi-tag bootstrap against the same budget (see M3 there); one constant so
# the two paths can never bound memory differently.
MAX_BOOT_CHUNK_CELLS = 2_000_000


def _validate(values: np.ndarray, weights: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Shared input validation for `weighted_median` and `bootstrap_ci`.

    Coerces both to 1-D float64 arrays and validates: 1-D, non-empty, equal
    length, all-finite (no NaN/+/-inf), weights non-negative, and weights
    summing to > 0. Returns the coerced `(values, weights)`. Raises
    ValueError (messages prefixed with `name`, the caller's own name) on any
    violation.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError(f"{name}: values/weights must be 1-D")
    if values.shape[0] == 0:
        raise ValueError(f"{name}: values/weights must not be empty")
    if values.shape != weights.shape:
        raise ValueError(f"{name}: values and weights must have the same length")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError(f"{name}: values/weights must not contain non-finite entries (NaN/inf)")
    if (weights < 0).any():
        raise ValueError(f"{name}: weights must be non-negative")
    if weights.sum() <= 0:
        raise ValueError(f"{name}: weights must sum to > 0")
    return values, weights


def _weighted_median_pos(cum: np.ndarray) -> int:
    """Position in an ascending cumulative-weight array selected by the lower-weighted-
    median convention: the first index at which cumulative weight reaches >= half the
    total (`np.searchsorted(..., side="left")`), with the half-weight threshold nudged
    down by `_TIE_EPS` to absorb float64 cumsum drift at exact ties. Shared by
    `_weighted_median_sorted` and `weighted_median_index` so this tie-tolerance logic
    lives in exactly one place. `cum` must be non-empty and non-decreasing (a cumsum).
    """
    half = 0.5 * cum[-1]
    return int(np.searchsorted(cum, half * (1 - _TIE_EPS), side="left"))


def _weighted_median_sorted(v: np.ndarray, w: np.ndarray) -> float:
    """Weighted median of already-sorted, already-validated inputs.

    Assumes `v` is sorted ascending and `w` is finite, non-negative, and sums
    to > 0. Not for external use (no validation) — see `weighted_median`.
    """
    return float(v[_weighted_median_pos(np.cumsum(w))])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median of `values` under `weights`.

    Tie/even-count convention: this is the *lower* weighted median. Sorting
    values ascending and walking cumulative weight, the returned value is the
    first one at which cumulative weight reaches >= half the total weight
    (`np.searchsorted(..., side="left")`, with the half-weight threshold
    nudged down by a small epsilon to absorb float64 cumsum drift at exact
    ties). For an even count with equal weights this picks the smaller of the
    two middle values (e.g. [0.2, 0.8] with equal weights returns 0.2),
    matching a robust, deterministic, order-independent convention rather
    than numpy's mean-of-two-middles.

    Raises ValueError if `values`/`weights` are not 1-D, are empty,
    mismatched in length, contain NaN or +/-inf, or if any weight is
    negative or all weights are zero (see `_validate`).
    """
    values, weights = _validate(values, weights, "weighted_median")
    order = np.argsort(values, kind="stable")
    return _weighted_median_sorted(values[order], weights[order])


def weighted_median_index(values: np.ndarray, weights: np.ndarray) -> int:
    """Return the *original* index of the element `weighted_median` would return.

    Same validation, stable sort, and tie tolerance as `weighted_median` (see
    `_validate` and `_weighted_median_sorted`) -- `values[weighted_median_index(values,
    weights)] == weighted_median(values, weights)` always holds. Useful when a caller
    needs to know *which* input element was selected (e.g. `explain()` marking the
    vote/tag chosen by the weighted-median chain), not just its value.
    """
    values, weights = _validate(values, weights, "weighted_median_index")
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order])
    return int(order[_weighted_median_pos(cum)])


def _bootstrap_draws(
    v_sorted: np.ndarray,
    w_sorted: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> np.ndarray:
    """The array of valid weighted-median bootstrap draws (length <= n_boot).

    `v_sorted`/`w_sorted` must already be sorted ascending by value. Draws
    are generated `chunk_size` resamples at a time via
    `rng.multinomial(n, [1/n]*n, size=chunk_size)` (see `bootstrap_ci` for
    why a multinomial count draw is equivalent to classic index resampling
    with replacement) and accumulated, so peak memory is bounded by
    `chunk_size * n` regardless of `n_boot`. A resample whose weights sum to
    zero has an undefined weighted median and is dropped from the result
    rather than counted.
    """
    n = len(v_sorted)
    pvals = np.full(n, 1.0 / n)
    boots = []
    remaining = n_boot
    while remaining > 0:
        take = min(chunk_size, remaining)
        remaining -= take
        counts = rng.multinomial(n, pvals, size=take)
        cum = np.cumsum(counts * w_sorted, axis=1)
        valid = cum[:, -1] > 0
        cum_valid = cum[valid]
        half = 0.5 * cum_valid[:, -1:]
        idx = (cum_valid >= half * (1 - _TIE_EPS)).argmax(axis=1)
        boots.append(v_sorted[idx])
    return np.concatenate(boots) if boots else np.array([], dtype=float)


def bootstrap_ci(
    values: np.ndarray,
    weights: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    ci_level: float,
) -> tuple[float, float]:
    """Vectorized bootstrap (lo, hi) quantiles of the weighted median.

    Equivalent to resampling n `(value, weight)` pairs with replacement
    `n_boot` times and taking the weighted median of each resample, but
    without a Python-level loop over resamples. A resample's weighted
    median depends only on how many times each *position* was drawn (its
    resample count), not on the order of the draws, so after sorting
    `values` once (stable), `rng.multinomial(n, [1/n]*n, size=chunk)`
    produces a batch of resamples' per-position counts in a single call --
    equivalent to drawing n indices with replacement, once per resample --
    and the cumulative sum of `counts * sorted_weights` along each row gives
    every resample's weighted median in one vectorized pass (see
    `_bootstrap_draws`). Processed in chunks of at most
    `max(1, 2_000_000 // n)` resamples so peak memory stays bounded even for
    a ratee with thousands of votes; this is what keeps the per-ratee
    bootstrap affordable at ~28k ratees.

    Inputs are validated exactly like `weighted_median` (see `_validate`).
    A resample whose weights sum to zero has an undefined weighted median
    and is skipped rather than counted. If fewer than `max(2, n_boot // 10)`
    resamples remain valid, raises ValueError instead of silently reporting
    a CI built from too few (or zero) samples. Skipping all-zero resamples
    conditions the CI on non-zero weight; unreachable in-pipeline since
    Config forbids zero weights.
    """
    values, weights = _validate(values, weights, "bootstrap_ci")
    n = len(values)
    order = np.argsort(values, kind="stable")
    v_sorted, w_sorted = values[order], weights[order]

    chunk_size = max(1, MAX_BOOT_CHUNK_CELLS // n)
    boots = _bootstrap_draws(v_sorted, w_sorted, n_boot, rng, chunk_size)

    min_valid = max(2, int(n_boot * MIN_VALID_BOOT_FRACTION))
    if boots.size < min_valid:
        raise ValueError("bootstrap degenerate: too many zero-weight resamples")

    alpha = (1 - ci_level) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return float(lo), float(hi)


def _bootstrap_ci(
    values: np.ndarray,
    weights: np.ndarray,
    n_boot: int,
    seed: int,
    ci_level: float,
) -> tuple[float, float]:
    """Bootstrap (lo, hi) quantiles of the weighted median, seeded by a plain
    integer `seed`. Thin wrapper around `bootstrap_ci` (the one bootstrap
    implementation) for callers that don't manage their own
    `numpy.random.Generator`.
    """
    return bootstrap_ci(values, weights, n_boot, np.random.default_rng(seed), ci_level)


class Aggregator(ABC):
    """Abstract base for turning per-ratee (values, weights) into a score with a CI.

    Concrete implementations (e.g. `WeightedMedian`, and future Bayesian or
    graph-based aggregators) all expose the same `aggregate` interface so the
    pipeline can swap aggregation strategies without changing callers.
    """

    @abstractmethod
    def aggregate(self, values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
        """Return (score, ci_low, ci_high)."""


class WeightedMedian(Aggregator):
    """Weighted-median aggregator with a bootstrap confidence interval.

    `aggregate` resamples (values, weights) pairs with replacement `n_boot`
    times, computes the weighted median of each resample, and reports the
    `ci_level` central interval of the bootstrap distribution (widened if
    needed so it always contains the point estimate). Note: with tied or
    coarse-grained values the percentile bootstrap can legitimately return a
    zero-width interval (every resample lands on the same value) — consumers
    should read the CI together with the underlying vote count rather than
    treating a narrow interval alone as high confidence.
    """

    def __init__(self, n_boot: int = 1000, seed: int = 0, ci_level: float = 0.95):
        if n_boot < 0:
            raise ValueError("WeightedMedian: n_boot must be >= 0")
        if not 0 < ci_level < 1:
            raise ValueError("WeightedMedian: ci_level must be in (0, 1)")
        self.n_boot, self.seed, self.ci_level = n_boot, seed, ci_level

    @classmethod
    def from_config(cls, cfg: Config) -> "WeightedMedian":
        """Build a WeightedMedian aggregator from a pipeline Config."""
        return cls(n_boot=cfg.bootstrap_n, seed=cfg.bootstrap_seed, ci_level=cfg.ci_level)

    def aggregate(self, values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        point = weighted_median(values, weights)
        if len(values) == 1 or self.n_boot == 0:
            return point, point, point
        lo, hi = _bootstrap_ci(values, weights, self.n_boot, self.seed, self.ci_level)
        # defensive; not expected to fire
        return point, float(min(lo, point)), float(max(hi, point))
