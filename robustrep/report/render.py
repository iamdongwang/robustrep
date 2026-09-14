"""Markdown report from scores + records (spec Sec 5.5b: five figures, sensitivity,
adversarial evidence, limitations, provenance)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

# The ERC-8004 study's own headline finding, for scale: even the paper that
# introduced this feedback scheme found the overwhelming majority of ratings
# carried no interaction evidence at all.
_PAPER_CITATION = (
    "For reference, the ERC-8004 study (arXiv 2606.26028) reported 98.7-100% of "
    "ratings with no interaction evidence."
)

_BREAKEVEN_ROWS = ("G_smear_breakeven", "G2_boost_breakeven", "H_evasive_breakeven")


def _markdown_table(df: pd.DataFrame) -> str:
    """Render `df` as a Markdown table with NaN cells shown as a blank string,
    never the literal "nan" pandas/tabulate would otherwise print."""
    return df.astype(object).where(df.notna(), "").to_markdown(index=False)


def _flag_rate(scored: pd.DataFrame) -> tuple[int, float]:
    """Sybil-flag count/rate among `insufficient == 0` rows only -- a NaN'd,
    insufficient ratee was never even scored, so it has no meaningful flag."""
    n = len(scored)
    if n == 0:
        return 0, 0.0
    flagged = int(scored["sybil_flag"].sum())
    return flagged, flagged / n


def _non_revoked(records: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """(non-revoked records, revoked count). Tolerates a `records` frame with no
    `revoked` column (e.g. a raw pre-validation frame in a test) by treating it
    as entirely non-revoked."""
    if "revoked" not in records.columns:
        return records, 0
    return records[records["revoked"] == 0], int((records["revoked"] == 1).sum())


def _measured_boundaries_paragraph(adversarial: pd.DataFrame) -> Optional[str]:
    """The "Measured boundaries" paragraph, built from `scenario_table()`'s
    break-even rows (see `robustrep.report.adversarial`) -- every number here is
    read out of that table, never hand-typed into this string. Returns `None` if
    `adversarial` doesn't carry the expected rows (e.g. a caller passed a custom
    table without them). `break_even_k_boost` is part of that column guard too --
    when it's missing, the H row's boost-direction k is reported as `None`
    (rendered "n/a") rather than raising a KeyError."""
    if not {"scenario", "break_even_k"} <= set(adversarial.columns):
        return None
    t = adversarial.set_index("scenario")
    if not set(_BREAKEVEN_ROWS) <= set(t.index):
        return None
    g_smear_k = int(t.loc["G_smear_breakeven", "break_even_k"])
    g_boost_k = int(t.loc["G2_boost_breakeven", "break_even_k"])
    h_smear_k = int(t.loc["H_evasive_breakeven", "break_even_k"])
    if "break_even_k_boost" in t.columns and pd.notna(t.loc["H_evasive_breakeven", "break_even_k_boost"]):
        h_boost_k = int(t.loc["H_evasive_breakeven", "break_even_k_boost"])
    else:
        h_boost_k = None
    zr = t.loc[list(_BREAKEVEN_ROWS), "zero_evidence_ratio"]
    sybil_flags_zero = bool((t.loc[list(_BREAKEVEN_ROWS), "sybil_flag"] == 0).all())
    return (
        f"**Measured boundaries.** Against 3 level-3 honest votes (mass 3.0), a smear "
        f"campaign flips the score at k={g_smear_k} evidence-free raters (the exact "
        f"tie), a boost needs {g_boost_k}. Against 5 level-2 honest votes (mass 3.5) "
        f"under the evasive pattern, smear flips at {h_smear_k}, boost at "
        f"{h_boost_k if h_boost_k is not None else 'n/a'}; `sybil_flag` stays "
        f"{'0' if sybil_flags_zero else 'non-zero'} throughout while `zero_evidence_ratio` "
        f"is {zr.min():.2f}-{zr.max():.2f}."
    )


def render_markdown(scores: pd.DataFrame, records: pd.DataFrame, block: int, figures: dict,
                    sensitivity: pd.DataFrame, adversarial: Optional[pd.DataFrame] = None,
                    provenance: Optional[dict] = None) -> str:
    """Render the full Markdown report.

    `figures` maps a human figure title (e.g. "Fig 4. Largest rater clusters") ->
    relative file path (e.g. "fig4_sybil_clusters.png") for the Markdown image
    link -- the title is the heading text, never the raw filename. `provenance`,
    when given, may carry `rater_profile_mode`, `confirmations`, `config` (a dict
    of the scoring parameters actually used: bootstrap_n, bootstrap_seed,
    min_clusters, evidence_weights, and the sybil_* thresholds), and `versions`.
    """
    non_revoked, n_revoked = _non_revoked(records)
    scored = scores[scores["insufficient"] == 0]
    insufficient = scores[scores["insufficient"] == 1]
    n_flagged, flag_rate = _flag_rate(scored)
    zero_ev = (non_revoked["evidence_level"] == 0).mean() if len(non_revoked) else 0.0

    lines = [
        "# Robust reputation on ERC-8004 (Base)",
        "",
        f"Data cut at block {block}. Ratings: {len(records)} ({n_revoked} revoked, excluded below). "
        f"Agents rated: {len(scores)}. Agents scored (>= min_clusters independent clusters): "
        f"{len(scored)}; insufficient (too few clusters): {len(insufficient)}.",
        "",
        "## Headline numbers",
        f"- Zero-evidence ratings (non-revoked only): {zero_ev:.1%}. {_PAPER_CITATION}",
        "- Agents flagged (largest single cluster >= 50% of that agent's records, "
        f"among scored agents only): {n_flagged} ({flag_rate:.1%})",
        "- Read `sybil_flag` together with `n_clusters` and `zero_evidence_ratio`: in "
        "scenario F an attack split across two funders held 91% of the records while "
        "the largest single cluster was 45%, leaving the flag at 0.",
    ]
    if len(scored):
        med_gap = (scored["naive_mean"] - scored["robust_score"]).abs().median()
        lines.append(f"- Median |mean - robust| among scored agents: {med_gap:.3f}")
    lines += ["", "## Figures"]
    for title, path in figures.items():
        lines += [f"### {title}", f"![{title}]({path})", ""]

    lines += ["## Sensitivity",
              "Rank stability of the top-N robust-score ranking under reasonable parameter "
              "perturbations (evidence-weight shape, sybil Jaccard threshold, sybil time "
              "window), each vs the base configuration -- Spearman's rho, computed as the "
              "Pearson correlation of the two rank sequences (no scipy dependency). A blank "
              "cell means too few ratees overlapped with the base top-N to define a "
              "correlation at all.",
              "", _markdown_table(sensitivity), ""]

    lines += ["## Adversarial evidence",
              "Known-answer attack scenarios (see `robustrep.report.adversarial.scenario_table`), "
              "each computed with **bootstrap_n=0**: point estimates only -- no confidence-interval "
              "claims are made from this table.", ""]
    if adversarial is not None:
        lines += [_markdown_table(adversarial), ""]
        boundaries = _measured_boundaries_paragraph(adversarial)
        if boundaries:
            lines += [boundaries, ""]

    lines += [
        "## Method",
        "normalize per (tag, scale) -> evidence weights -> sybil collapse (2-of-3 signals: "
        "shared funder, first-seen within the sybil time window, Jaccard-similar ratee sets) "
        "-> one vote per (ratee, tag, cluster) -> per-tag weighted median, weighted across tags "
        "by summed evidence mass -> bootstrap CI. See the design spec in the parent repo.",
        "",
        "**Attacker break-even.** An attacker needs to command roughly half of the evidence "
        "mass behind a ratee's votes to move the weighted median at all -- not half the vote "
        "count. Exact ties resolve **downward**, toward the lower of the two tied values: a "
        "smearing attack that reaches exactly half the evidence mass succeeds at that tie, "
        "while a boosting attack needs to strictly exceed half (one more unit of mass) to flip "
        "the score, since at the tie the lower (honest) value is the one selected.",
        "",
        "**Flat-weight fallback.** With flat (all-equal) evidence weights, the aggregator "
        "reduces to the **lower weighted median** of the votes, not numpy's mean-of-two-middles "
        "convention: for an even split of evidence mass on either side, the smaller of the two "
        "middle values is reported, never their average.",
        "",
        "## Limitations",
        "- **Clustering evasion.** An attacker using a distinct funder per rater, spacing "
        "first-seen timestamps more than the sybil time window apart, and rating decoy ratees "
        "can still reach a Jaccard of 1.0 with another attacker who shares the same decoy -- "
        "but that pair still evades detection, because it fails both of the other two signals "
        "(distinct funders, and outside the time window), so no pair ever reaches 2 of the 3 "
        "required signals. Every attacker rater lands in its own singleton cluster. The "
        "residual defense is evidence weighting (an evasive farm is almost always "
        "evidence-free): `zero_evidence_ratio` is the signal that survives this evasion even "
        "when `sybil_flag` reads 0 and rater clustering sees nothing unusual.",
        "- **Large-ratee-group blind spot.** A ratee with more raters than `sybil_max_group` is "
        "skipped entirely for ratee-based blocking (see `robustrep.sybil`), so a farm "
        "concentrated on one very popular ratee can evade the ratee-sharing signal by sheer "
        "volume, independent of the evasion technique above.",
        "- **DNS rebinding in the evidence fetcher (not mitigated in v0.1).** The SSRF guard "
        "resolves and checks an evidence URI's host once, but the HTTP client resolves it again "
        "independently; a rebinding attacker timed between the two resolutions can route a "
        "blind GET to a private address. No response content is ever exposed or stored -- only "
        "a 0-3 evidence level -- bounding the blast radius. See "
        "`robustrep.sources.evidence_fetch` for the full writeup and its xfail regression test.",
        "- **Two-tag lower-median bias.** With exactly two tags, cross-tag combination is a "
        "step, not a blend: the reported score is whichever tag's median holds at least half "
        "the total evidence mass; the other tag is discarded. At an exact mass tie (within a "
        "1e-9 relative tolerance) the lower-scored tag wins by the same lower-median "
        "convention. The deviation from a blended estimate is bounded by the gap between the "
        "two tag medians and is downward only at the tie.",
        "",
        "## Provenance",
    ]
    prov = provenance or {}
    lines.append(f"- Rater profile mode: {prov.get('rater_profile_mode', 'unknown')}")
    lines.append(f"- Confirmations lag (data cut is final as of this block): "
                 f"{prov.get('confirmations', 'unknown')}")
    config = prov.get("config") or {}
    if config:
        lines.append("- Scoring configuration used:")
        for key in ("bootstrap_n", "bootstrap_seed", "min_clusters", "evidence_weights",
                    "sybil_jaccard", "sybil_window_s", "sybil_max_group", "sybil_flag_share"):
            if key in config:
                lines.append(f"  - `{key}`: {config[key]}")
    versions = prov.get("versions") or {}
    if versions:
        v_str = ", ".join(f"{k} {v}" for k, v in versions.items())
        lines.append(f"- Versions: {v_str}")
    return "\n".join(lines)
