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


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median of `values` under `weights`.

    Tie/even-count convention: this is the *lower* weighted median. Sorting
    values ascending and walking cumulative weight, the returned value is the
    first one at which cumulative weight reaches >= half the total weight
    (`np.searchsorted(..., side="left")`). For an even count with equal
    weights this picks the smaller of the two middle values (e.g. [0.2, 0.8]
    with equal weights returns 0.2), matching a robust, deterministic,
    order-independent convention rather than numpy's mean-of-two-middles.

    Raises ValueError if `values`/`weights` are empty, mismatched in length,
    contain NaN, or if any weight is negative or all weights are zero.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape[0] == 0:
        raise ValueError("weighted_median: values/weights must not be empty")
    if values.shape != weights.shape:
        raise ValueError("weighted_median: values and weights must have the same length")
    if np.isnan(values).any() or np.isnan(weights).any():
        raise ValueError("weighted_median: values/weights must not contain NaN")
    if (weights < 0).any():
        raise ValueError("weighted_median: weights must be non-negative")
    if weights.sum() <= 0:
        raise ValueError("weighted_median: weights must sum to > 0")

    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1], side="left")])


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
    needed so it always contains the point estimate).
    """

    def __init__(self, n_boot: int = 1000, seed: int = 0, ci_level: float = 0.95):
        if n_boot < 0:
            raise ValueError("WeightedMedian: n_boot must be >= 0")
        if not 0 < ci_level < 1:
            raise ValueError("WeightedMedian: ci_level must be in (0, 1)")
        self.n_boot, self.seed, self.ci_level = n_boot, seed, ci_level

    def aggregate(self, values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        point = weighted_median(values, weights)
        if len(values) == 1 or self.n_boot == 0:
            return point, point, point
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(values), size=(self.n_boot, len(values)))
        boots = np.array([weighted_median(values[i], weights[i]) for i in idx])
        alpha = (1 - self.ci_level) / 2
        lo, hi = np.quantile(boots, [alpha, 1 - alpha])
        return point, float(min(lo, point)), float(max(hi, point))
