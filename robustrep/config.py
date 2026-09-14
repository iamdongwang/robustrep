"""Tunable parameters. Defaults are provisional until the sensitivity analysis in the report confirms them."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # evidence level 0..3 -> weight
    evidence_weights: tuple[float, float, float, float] = (0.1, 0.3, 0.7, 1.0)
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
    # blocks to lag behind the chain head before ingesting: guards against a re-org
    # near the tip leaving phantom feedback rows behind.
    confirmations: int = 20

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
