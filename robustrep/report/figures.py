"""Figures 1-4. Each function draws one figure and saves it to `out`.

Deterministic: no timestamps, wall-clock text, or random jitter in any figure
-- byte-identical output for identical input data. Every figure tolerates an
empty scored set (e.g. every ratee `insufficient`) by drawing an empty axes
with a note instead of raising.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _empty_note(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _short_label(x) -> str:
    """Truncate an address/id-like cluster label to `abcdef…wxyz`."""
    a = str(x)
    return f"{a[:6]}…{a[-4:]}"


def fig_mean_vs_robust(scores: pd.DataFrame, out: Path) -> None:
    """Fig 1: arithmetic mean vs robust score, one point per scored ratee.

    Color marks whether that ratee's largest single rater-cluster reaches the
    sybil-flag threshold (>= 50% of its records) -- a legend with proxy handles
    names the two colors, since a raw color-mapped scatter has no legend of its
    own. `.fillna("tab:grey")` on the color mapping is defensive: `sybil_flag` is
    always 0/1 for a scored (non-`insufficient`) ratee, but an unexpected value
    still renders (grey) instead of raising or vanishing.
    """
    s = scores.dropna(subset=["robust_score"])
    fig, ax = plt.subplots(figsize=(6, 6))
    if s.empty:
        _empty_note(ax, "no scored ratees (all insufficient)")
    else:
        colors = s["sybil_flag"].map({0: "tab:blue", 1: "tab:red"}).fillna("tab:grey")
        ax.scatter(s["naive_mean"], s["robust_score"], s=8, alpha=0.4, c=colors)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("arithmetic mean")
        ax.set_ylabel("robust score")
        handles = [
            Line2D([], [], marker="o", linestyle="", color="tab:blue", label="largest cluster < 50%"),
            Line2D([], [], marker="o", linestyle="", color="tab:red", label="largest cluster >= 50%"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize="small")
    ax.set_title("Fig 1. Mean vs robust score")
    _save(fig, out)


def fig_rank_shift(scores: pd.DataFrame, out: Path, top_n: int = 100, show: int = 20) -> None:
    """Fig 2: biggest rank drops among the mean's top-`top_n` once re-ranked by robust score.

    Tolerates fewer than `top_n` scored ratees (the whole scored set is then
    "top") and an empty scored set.
    """
    s = scores.dropna(subset=["robust_score"]).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if s.empty:
        _empty_note(ax, "no scored ratees (all insufficient)")
        ax.set_title(f"Fig 2. biggest rank drops among mean top-{top_n}")
        _save(fig, out)
        return
    s["rank_mean"] = s["naive_mean"].rank(ascending=False, method="first")
    s["rank_robust"] = s["robust_score"].rank(ascending=False, method="first")
    top = s[s["rank_mean"] <= top_n].copy()
    top["shift"] = top["rank_robust"] - top["rank_mean"]
    worst = top.sort_values("shift", ascending=False).head(show)
    if worst.empty:
        _empty_note(ax, "no rank shifts to show")
    else:
        ax.barh(worst["ratee"].astype(str), worst["shift"], color="tab:red")
        ax.invert_yaxis()
        ax.set_xlabel("rank drop (robust - mean)")
    ax.set_title(f"Fig 2. biggest rank drops among mean top-{top_n}")
    _save(fig, out)


def fig_evidence(records: pd.DataFrame, out: Path) -> None:
    """Fig 3: pie chart of evidence levels across all raw ratings."""
    fig, ax = plt.subplots(figsize=(5, 5))
    if records.empty:
        _empty_note(ax, "no records")
    else:
        counts = records["evidence_level"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
        if counts.sum() == 0:
            _empty_note(ax, "no records")
        else:
            ax.pie(counts, labels=[f"level {i}" for i in counts.index], autopct="%1.1f%%")
    ax.set_title("Fig 3. Evidence levels of all ratings")
    _save(fig, out)


def fig_sybil_clusters(records: pd.DataFrame, clusters: dict, out: Path, top: int = 10) -> None:
    """Fig 4: the `top` largest rater clusters by member count, each bar
    annotated with the number of distinct agents that cluster covers.

    Ranked by cluster SIZE (member count), not agent coverage -- a cluster's
    size is what drives its evidence-mass weight in scoring, while agent
    coverage (how many different ratees it touched) is the secondary,
    annotated number. Title never says "sybil cluster": a large cluster here is
    evidence of coordinated *rater* behavior, not a claim any of it is
    confirmed sybil activity.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if records.empty:
        _empty_note(ax, "no records")
    else:
        cluster_of = records["rater"].map(lambda r: clusters.get(r, r))
        df = records.assign(cluster=cluster_of)
        members = df.groupby("cluster")["rater"].nunique().sort_values(ascending=False).head(top)
        covered = df.groupby("cluster")["ratee"].nunique()
        if members.empty:
            _empty_note(ax, "no clusters")
        else:
            bars = ax.bar(range(len(members)), members.values)
            for bar, cluster_id in zip(bars, members.index):
                ax.annotate(str(int(covered.loc[cluster_id])),
                           (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha="center", va="bottom", fontsize=8)
            ax.set_xticks(range(len(members)))
            ax.set_xticklabels([_short_label(c) for c in members.index], rotation=45)
            ax.set_ylabel("members (agents covered annotated)")
    ax.set_title("Fig 4. Largest rater clusters (members; agents covered)")
    _save(fig, out)
