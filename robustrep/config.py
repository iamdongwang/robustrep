"""Tunable parameters. Defaults are provisional until the sensitivity analysis in the report confirms them."""
from __future__ import annotations

from dataclasses import dataclass

# Blocks to lag behind the chain head before ingesting: guards against a re-org
# near the tip leaving phantom feedback rows behind. Single source of truth for
# both Config.confirmations' default and robustrep.sources.base_erc8004.sync_feedback's
# default, so the two can never silently drift apart.
DEFAULT_CONFIRMATIONS = 20

# Share of a (tag, scale) group's values that must land inside a normalization
# rule's range for that rule to be selected (see robustrep.normalize). Single
# source of truth for both Config.norm_fit_share's default and normalize()'s own
# default, so the two can never silently drift apart.
DEFAULT_NORM_FIT_SHARE = 0.9


def validate_norm_fit_share(value: float) -> None:
    """Raise ValueError unless `value` is in (0.5, 1.0].

    Shared by `Config.__post_init__` and `robustrep.normalize.normalize`, which
    can be called directly with its own `fit_share` rather than through a
    Config. <= 0.5 would let a minority of a group pick that group's
    normalization rule, which is the attacker-controlled breakdown the
    parameter exists to close; it is also what makes the `constant` rung's
    median test safe (see `robustrep.normalize._group_scores`).

    1.0 is allowed and is strictly all-or-nothing: every value in a group must
    fall inside a rule's range for that rule to apply, with no small-group
    tolerance. It is the only way to ask for the pre-tolerance behaviour.
    """
    if not 0.5 < value <= 1.0:
        raise ValueError("norm_fit_share must be in (0.5, 1.0]")


@dataclass(frozen=True)
class Config:
    # evidence level 0..3 -> weight
    evidence_weights: tuple[float, float, float, float] = (0.1, 0.3, 0.7, 1.0)
    # normalization: share of a (tag, scale) group that must fit a rule's range
    # for that rule to be chosen; out-of-range values are clipped, not fitted.
    # Small groups always tolerate one outlier (see robustrep.normalize._fits)
    # EXCEPT at exactly 1.0, which is strictly all-or-nothing.
    norm_fit_share: float = DEFAULT_NORM_FIT_SHARE
    # sybil clustering
    sybil_window_s: int = 24 * 3600
    sybil_jaccard: float = 0.8
    sybil_max_group: int = 2000  # max raters per ratee considered when generating candidate sybil pairs
    sybil_max_pairs: int = 5_000_000  # global cap on candidate pairs; cluster_raters fails fast past this
    sybil_flag_share: float = 0.5
    # aggregation
    min_clusters: int = 3
    bootstrap_n: int = 1000
    bootstrap_seed: int = 0
    ci_level: float = 0.95
    # data source
    rpc_urls: tuple[str, ...] = ("https://mainnet.base.org",)
    chunk_blocks: int = 2000
    user_agent: str = "robustrep/0.1 (+https://github.com/iamdongwang/robustrep)"
    confirmations: int = DEFAULT_CONFIRMATIONS

    def __post_init__(self) -> None:
        if len(self.evidence_weights) != 4:
            raise ValueError("evidence_weights must have exactly 4 entries")
        if any(w <= 0 for w in self.evidence_weights):
            raise ValueError("evidence_weights must be > 0 (zero weights make bootstrap resamples degenerate)")
        if not 0 < self.ci_level < 1:
            raise ValueError("ci_level must be in (0, 1)")
        if self.bootstrap_n < 0:
            raise ValueError("bootstrap_n must be >= 0")
        if self.min_clusters < 1:
            raise ValueError("min_clusters must be >= 1")
        validate_norm_fit_share(self.norm_fit_share)
        if not 0 < self.sybil_jaccard <= 1:
            raise ValueError("sybil_jaccard must be in (0, 1]")  # 0 would match every pair's Jaccard signal
        if self.sybil_max_pairs < 1:
            raise ValueError("sybil_max_pairs must be >= 1")
        if not 0 <= self.sybil_flag_share <= 1:
            raise ValueError("sybil_flag_share must be in [0, 1]")
        if self.sybil_window_s < 0:
            raise ValueError("sybil_window_s must be >= 0")
        if self.chunk_blocks < 1:
            raise ValueError("chunk_blocks must be >= 1")
        if len(self.rpc_urls) < 1:
            raise ValueError("rpc_urls must have at least 1 entry")
        if self.confirmations < 0:
            raise ValueError("confirmations must be >= 0")
