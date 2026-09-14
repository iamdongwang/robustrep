"""Figure 5: how stable is the top-N ranking under reasonable parameter changes."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import Config  # noqa: E402
from ..pipeline import score  # noqa: E402
from ..sybil import cluster_raters, profiles_from_records  # noqa: E402

# The 7 variants required by the report (spec Sec 5.5b): base plus three axes
# (evidence-weight shape, sybil Jaccard threshold, sybil time window), each
# perturbed in both directions.
VARIANTS = {
    "base": {},
    "weights_flatter": {"evidence_weights": (0.3, 0.5, 0.8, 1.0)},
    "weights_steeper": {"evidence_weights": (0.02, 0.1, 0.5, 1.0)},
    "jaccard_0.6": {"sybil_jaccard": 0.6},
    "jaccard_0.9": {"sybil_jaccard": 0.9},
    "window_6h": {"sybil_window_s": 6 * 3600},
    "window_72h": {"sybil_window_s": 72 * 3600},
}


def _top_ranks(records: pd.DataFrame, cfg: Config, meta: Optional[pd.DataFrame], top_n: int):
    clusters = cluster_raters(profiles_from_records(records, meta), cfg)
    s = score(records, cfg, clusters=clusters).dropna(subset=["robust_score"])
    return s.set_index("ratee")["robust_score"].rank(ascending=False, method="first"), s


def sensitivity_table(records: pd.DataFrame, cfg: Config, meta: Optional[pd.DataFrame] = None,
                      top_n: int = 100) -> pd.DataFrame:
    """Rank-stability of the top-`top_n` ratees (by base robust_score) between the
    base config and each variant in `VARIANTS`. `bootstrap_n` is forced to 0
    (point estimates only; sensitivity is about ranking stability, not CI width).

    The correlation is Spearman's rho computed WITHOUT scipy: Spearman is just the
    Pearson correlation of the two rank sequences, so `pandas.Series.rank().corr()`
    (default Pearson) gives the identical statistic with no extra dependency.

    NaN, never a fabricated 1.0, when fewer than 2 ratees overlap between the base
    top-N and a variant's scored set (e.g. every ratee that mattered became
    `insufficient` under that variant) -- there isn't enough data there to claim
    any correlation, let alone perfect stability.
    """
    cfg = replace(cfg, bootstrap_n=0)
    base_rank, base_scores = _top_ranks(records, cfg, meta, top_n)
    top = base_scores.nlargest(top_n, "robust_score")["ratee"]
    rows = []
    for name, kw in VARIANTS.items():
        rank, _ = _top_ranks(records, replace(cfg, **kw), meta, top_n)
        joined = pd.concat([base_rank.reindex(top), rank.reindex(top)], axis=1, keys=["a", "b"]).dropna()
        if len(joined) > 1:
            rho = joined["a"].rank().corr(joined["b"].rank())
            rho = float(rho) if pd.notna(rho) else float("nan")
        else:
            rho = float("nan")
        rows.append(dict(variant=name, spearman_top=rho))
    return pd.DataFrame(rows)


def fig_sensitivity(table: pd.DataFrame, out: Path) -> None:
    """Fig 5: bar chart of `sensitivity_table`'s Spearman rho per variant.

    A variant with no defined rho (`NaN` -- insufficient overlap with the base
    top-N) is drawn as a zero-height bar annotated "insufficient overlap" rather
    than silently omitted or plotted as a fabricated value.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    if table.empty:
        ax.text(0.5, 0.5, "no sensitivity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        values = table["spearman_top"]
        bars = ax.bar(table["variant"], values.fillna(0.0))
        for bar, v in zip(bars, values):
            if pd.isna(v):
                ax.annotate("insufficient overlap", (bar.get_x() + bar.get_width() / 2, 0),
                           ha="center", va="bottom", rotation=90, fontsize=7)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Spearman rho (top-N vs base)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Fig 5. ranking stability")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
