"""Tunable parameters. All defaults are the ones reported in the paper's sensitivity analysis."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # evidence level 0..3 -> weight
    evidence_weights: tuple[float, float, float, float] = (0.1, 0.3, 0.7, 1.0)
    # sybil clustering
    sybil_window_s: int = 24 * 3600
    sybil_jaccard: float = 0.8
    sybil_max_block: int = 2000  # skip ratees with more raters than this when generating pairs
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
