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


def _weighted_median_sorted(v: np.ndarray, w: np.ndarray) -> float:
    """Weighted median of already-sorted, already-validated inputs.

    Assumes `v` is sorted ascending and `w` is finite, non-negative, and sums
    to > 0. Not for external use (no validation) — see `weighted_median`.
    """
    cum = np.cumsum(w)
    half = 0.5 * cum[-1]
    return float(v[np.searchsorted(cum, half * (1 - _TIE_EPS), side="left")])


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
    negative or all weights are zero.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("weighted_median: values/weights must be 1-D")
    if values.shape[0] == 0:
        raise ValueError("weighted_median: values/weights must not be empty")
    if values.shape != weights.shape:
        raise ValueError("weighted_median: values and weights must have the same length")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("weighted_median: values/weights must not contain non-finite entries (NaN/inf)")
    if (weights < 0).any():
        raise ValueError("weighted_median: weights must be non-negative")

    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    if np.cumsum(w)[-1] <= 0:
        raise ValueError("weighted_median: weights must sum to > 0")
    return _weighted_median_sorted(v, w)


def bootstrap_ci_sorted(
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
    `values` once (stable), `rng.multinomial(n, [1/n]*n, size=n_boot)`
    produces every resample's per-position counts in a single call --
    equivalent to drawing n indices with replacement n_boot times -- and
    the cumulative sum of `counts * sorted_weights` along each row gives
    every resample's weighted median in one vectorized pass. This is what
    keeps the per-ratee bootstrap affordable at ~28k ratees.

    Same degenerate-resample handling as before: a resample whose weights
    sum to zero has an undefined weighted median and is skipped rather than
    counted. If fewer than `max(2, n_boot // 10)` resamples remain valid,
    raises ValueError instead of silently reporting a CI built from too few
    (or zero) samples. Skipping all-zero resamples conditions the CI on
    non-zero weight; unreachable in-pipeline since Config forbids zero
    weights.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(values)
    order = np.argsort(values, kind="stable")
    v_sorted, w_sorted = values[order], weights[order]

    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    cum = np.cumsum(counts * w_sorted, axis=1)
    totals = cum[:, -1]
    valid = totals > 0

    min_valid = max(2, int(n_boot * MIN_VALID_BOOT_FRACTION))
    if int(valid.sum()) < min_valid:
        raise ValueError("bootstrap degenerate: too many zero-weight resamples")

    cum_valid = cum[valid]
    half = 0.5 * cum_valid[:, -1:]
    idx = (cum_valid >= half * (1 - _TIE_EPS)).argmax(axis=1)
    boots = v_sorted[idx]

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
    integer `seed`. Thin wrapper around `bootstrap_ci_sorted` (the one
    bootstrap implementation) for callers that don't manage their own
    `numpy.random.Generator`.
    """
    return bootstrap_ci_sorted(values, weights, n_boot, np.random.default_rng(seed), ci_level)


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
