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
import pandas as pd  # noqa: E402

_SYBIL_LABEL = "red = largest single cluster >= 50% of records"


def _save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _empty_note(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_mean_vs_robust(scores: pd.DataFrame, out: Path) -> None:
    """Fig 1: arithmetic mean vs robust score, one point per scored ratee."""
    s = scores.dropna(subset=["robust_score"])
    fig, ax = plt.subplots(figsize=(6, 6))
    if s.empty:
        _empty_note(ax, "no scored ratees (all insufficient)")
    else:
        ax.scatter(s["naive_mean"], s["robust_score"], s=8, alpha=0.4,
                   c=s["sybil_flag"].map({0: "tab:blue", 1: "tab:red"}))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("arithmetic mean")
        ax.set_ylabel("robust score")
    ax.set_title(f"Fig 1. mean vs robust ({_SYBIL_LABEL})")
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
    ax.set_title("Fig 3. evidence levels of all ratings")
    _save(fig, out)


def fig_sybil_clusters(records: pd.DataFrame, clusters: dict, out: Path, top: int = 10) -> None:
    """Fig 4: agent coverage of the largest rater clusters ("largest single cluster"
    label, matching sybil_flag's definition -- never "sybil clusters" without qualification)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if records.empty:
        _empty_note(ax, "no records")
    else:
        c = records["rater"].map(lambda r: clusters.get(r, r))
        cover = records.assign(cluster=c).groupby("cluster")["ratee"].nunique().sort_values(
            ascending=False).head(top)
        if cover.empty:
            _empty_note(ax, "no clusters")
        else:
            ax.bar(range(len(cover)), cover.values)
            ax.set_xticks(range(len(cover)))
            ax.set_xticklabels([str(x)[:8] for x in cover.index], rotation=45)
            ax.set_ylabel("agents covered")
    ax.set_title(f"Fig 4. largest {top} rater clusters")
    _save(fig, out)
