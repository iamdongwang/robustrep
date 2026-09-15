"""Figure 5: how stable is the top-N ranking under reasonable parameter changes.

Tie-aware by construction: `robust_score` clusters heavily at round values (in
the real cut, 1,067 of 5,170 scored agents tie at exactly 1.0), so "the top
100" is not a well-defined set unless ties are included -- an arbitrary
top-N cut through a multi-hundred-agent tie group, re-drawn independently for
the base run and every variant, churns Spearman's rho toward 0 from tie
order alone, never from any real ranking change. Every top set here is
defined as tie-inclusive: every agent whose score is >= the N-th highest
score, ties included, so a variant that only reshuffles order *within* a tie
group leaves the top set (and the correlation) untouched.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import Config  # noqa: E402
from ..pipeline import score  # noqa: E402
from ..sybil import cluster_raters_with_stats, profiles_from_records  # noqa: E402

# The 9 variants required by the report (spec Sec 5.5b): base plus four axes
# (evidence-weight shape, sybil Jaccard threshold, sybil time window,
# normalization fit share), each perturbed in both directions. The fit share's
# upper end is 1.0, which is strictly all-or-nothing -- no small-group outlier
# tolerance -- so `fit_1.0` also measures what that tolerance is worth.
VARIANTS = {
    "base": {},
    "weights_flatter": {"evidence_weights": (0.3, 0.5, 0.8, 1.0)},
    "weights_steeper": {"evidence_weights": (0.02, 0.1, 0.5, 1.0)},
    "jaccard_0.6": {"sybil_jaccard": 0.6},
    "jaccard_0.9": {"sybil_jaccard": 0.9},
    "window_6h": {"sybil_window_s": 6 * 3600},
    "window_72h": {"sybil_window_s": 72 * 3600},
    "fit_0.8": {"norm_fit_share": 0.8},
    "fit_1.0": {"norm_fit_share": 1.0},
}


def _scored_series(records: pd.DataFrame, cfg: Config,
                   meta: Optional[pd.DataFrame]) -> tuple[pd.Series, bool]:
    """(`robust_score` indexed by `ratee`, was-clustering-budget-limited).

    Scored agents only (`insufficient` ratees are dropped -- they never had
    enough independent evidence for a score to mean anything, tie-inclusive or
    not). Every variant re-clusters from scratch, and a variant can hit a sybil
    pair budget the base run did not (`window_72h` widens the window, which
    multiplies candidate pairs), so each run reports whether its clustering was
    budget-limited -- a ranking change that is really a budget artefact must
    not read as a parameter effect.
    """
    clusters, stats = cluster_raters_with_stats(profiles_from_records(records, meta), cfg)
    s = score(records, cfg, clusters=clusters).dropna(subset=["robust_score"])
    return s.set_index("ratee")["robust_score"], stats.budget_limited


def _tie_inclusive_top(scores: pd.Series, top_n: int) -> set:
    """Every ratee whose score is >= the `top_n`-th highest score, ties
    included -- e.g. if 30 ratees tie for 8th-highest, all 30 are "in the top
    10", not an arbitrary 3 of them. The whole scored set if it has `top_n`
    or fewer members."""
    if scores.empty:
        return set()
    if len(scores) <= top_n:
        return set(scores.index)
    threshold = scores.nlargest(top_n).min()
    return set(scores.index[scores >= threshold])


def _ranks_with_missing_lowest(scores: pd.Series, universe: list) -> pd.Series:
    """Average-method rank (descending; ties share the mean rank) of every
    member of `scores`, reindexed onto `universe`. A `universe` member absent
    from `scores` (insufficient, or never rated, under that run) gets that
    run's lowest rank -- one worse than the worst rank actually achieved (n+1
    for n scored ratees, never tied with an actual last-place ratee, which
    would otherwise collapse a small run's ranks to zero variance) -- so it
    reads as "worse than everyone actually ranked here" rather than vanishing
    (the old intersect-and-dropna behaviour) or fabricating a mid-pack rank."""
    ranks = scores.rank(ascending=False, method="average")
    lowest = float(len(scores) + 1) if len(scores) else 1.0
    return ranks.reindex(universe).fillna(lowest)


def tie_aware_sensitivity(base_scores: pd.Series, variant_scores: pd.Series, top_n: int) -> dict:
    """Compare two runs' scores (each a `robust_score` Series indexed by
    `ratee`) over their tie-inclusive top-`top_n` sets.

    Returns `spearman_union` (Pearson correlation of average ranks, computed
    over the UNION of the two tie-inclusive top sets -- an agent missing/
    insufficient in one run gets that run's lowest rank rather than being
    dropped; `NaN` only when the union has fewer than 2 members, i.e. there
    isn't enough data to define a correlation at all), `top_set_jaccard`
    (|intersection| / |union| of the two tie-inclusive top sets; `NaN` only
    when both sets are empty), and `top_set_size` (size of the variant's own
    tie-inclusive top set).
    """
    base_top = _tie_inclusive_top(base_scores, top_n)
    variant_top = _tie_inclusive_top(variant_scores, top_n)
    union = base_top | variant_top
    inter = base_top & variant_top
    jaccard = (len(inter) / len(union)) if union else float("nan")

    if len(union) > 1:
        universe = sorted(union)
        a_ranks = _ranks_with_missing_lowest(base_scores, universe)
        b_ranks = _ranks_with_missing_lowest(variant_scores, universe)
        # Pearson correlation is undefined (0/0) when either side has zero
        # variance -- e.g. the whole union sits in one tie group in that run,
        # so every rank is the same constant. That's exactly the case a
        # same-tie-group comparison hits by construction, so treat "both
        # sides rank the union identically" as perfect agreement (1.0) rather
        # than let a real corrcoef 0/0 divide-by-zero warning through; any
        # other zero-variance mismatch has no defined slope, so NaN.
        if a_ranks.std(ddof=0) == 0 or b_ranks.std(ddof=0) == 0:
            rho = 1.0 if a_ranks.equals(b_ranks) else float("nan")
        else:
            rho = a_ranks.corr(b_ranks)
            rho = float(rho) if pd.notna(rho) else float("nan")
    else:
        rho = float("nan")

    return dict(spearman_union=rho, top_set_jaccard=jaccard, top_set_size=len(variant_top))


def sensitivity_table(records: pd.DataFrame, cfg: Config, meta: Optional[pd.DataFrame] = None,
                      top_n: int = 100) -> pd.DataFrame:
    """Rank-stability of the tie-inclusive top-`top_n` ratees between the
    base config and each variant in `VARIANTS`. `bootstrap_n` is forced to 0
    (point estimates only; sensitivity is about ranking stability, not CI
    width).

    Columns: `variant`, `spearman_union` (see `tie_aware_sensitivity`),
    `spearman_top` (an alias of `spearman_union`, kept for `fig_sensitivity`
    and older callers), `top_set_jaccard`, `top_set_size`, and
    `budget_limited` -- True when THAT variant's clustering hit a sybil pair
    budget, so its row may be measuring the budget rather than the parameter.
    """
    cfg = replace(cfg, bootstrap_n=0)
    base_scores, _ = _scored_series(records, cfg, meta)
    rows = []
    for name, kw in VARIANTS.items():
        variant_scores, budget_limited = _scored_series(records, replace(cfg, **kw), meta)
        result = tie_aware_sensitivity(base_scores, variant_scores, top_n)
        rows.append(dict(
            variant=name,
            spearman_union=result["spearman_union"],
            spearman_top=result["spearman_union"],
            top_set_jaccard=result["top_set_jaccard"],
            top_set_size=result["top_set_size"],
            budget_limited=budget_limited,
        ))
    return pd.DataFrame(rows)


def fig_sensitivity(table: pd.DataFrame, out: Path) -> None:
    """Fig 5: per-variant bar chart of `sensitivity_table`.

    Draws two bars per variant when `top_set_jaccard` is present (Spearman
    rho alongside the tie-inclusive top-set Jaccard, with a legend); falls
    back to a single Spearman-only bar (the pre-tie-aware shape) when it
    isn't, so an older/partial table still renders.

    A variant with no defined rho (`NaN` -- the union of the two tie-inclusive
    top sets has fewer than 2 members) is drawn as a zero-height bar annotated
    "insufficient overlap" rather than silently omitted or plotted as a
    fabricated value.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if table.empty:
        ax.text(0.5, 0.5, "no sensitivity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        spearman = table["spearman_top"] if "spearman_top" in table.columns else table["spearman_union"]
        has_jaccard = "top_set_jaccard" in table.columns
        x = np.arange(len(table))
        if has_jaccard:
            width = 0.35
            bars = ax.bar(x - width / 2, spearman.fillna(0.0), width, label="Spearman (union, avg rank)")
            ax.bar(x + width / 2, table["top_set_jaccard"].fillna(0.0), width, label="top-set Jaccard")
            ax.legend(loc="best", fontsize="small")
        else:
            bars = ax.bar(x, spearman.fillna(0.0))
        for bar, v in zip(bars, spearman):
            if pd.isna(v):
                ax.annotate("insufficient overlap", (bar.get_x() + bar.get_width() / 2, 0),
                           ha="center", va="bottom", rotation=90, fontsize=7)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("value")
        ax.set_xticks(x)
        ax.set_xticklabels(table["variant"], rotation=30, ha="right")
    ax.set_title("Fig 5. Ranking stability")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
