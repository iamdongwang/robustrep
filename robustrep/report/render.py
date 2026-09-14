"""Markdown report from scores + records (spec Sec 5.5b: five figures, sensitivity,
adversarial evidence, limitations, provenance)."""
from __future__ import annotations

from typing import Optional

import pandas as pd


def _flag_rate(scored: pd.DataFrame) -> tuple[int, float]:
    """Sybil-flag count/rate among `insufficient == 0` rows only -- a NaN'd,
    insufficient ratee was never even scored, so it has no meaningful flag."""
    n = len(scored)
    if n == 0:
        return 0, 0.0
    flagged = int(scored["sybil_flag"].sum())
    return flagged, flagged / n


def render_markdown(scores: pd.DataFrame, records: pd.DataFrame, block: int, figures: dict,
                    sensitivity: pd.DataFrame, adversarial: Optional[pd.DataFrame] = None,
                    provenance: Optional[dict] = None) -> str:
    """Render the full Markdown report.

    `figures` maps a figure name -> relative path (e.g. "fig1.png") for the
    Markdown image link. `provenance`, when given, may carry
    `rater_profile_mode`, `confirmations`, and `versions` (a dict of package
    name -> version string).
    """
    scored = scores[scores["insufficient"] == 0]
    insufficient = scores[scores["insufficient"] == 1]
    n_flagged, flag_rate = _flag_rate(scored)
    zero_ev = (records["evidence_level"] == 0).mean() if len(records) else 0.0

    lines = [
        "# Robust reputation on ERC-8004 (Base)",
        "",
        f"Data cut at block {block}. Ratings: {len(records)}. Agents rated: {len(scores)}. "
        f"Agents scored (>= min_clusters independent clusters): {len(scored)}; "
        f"insufficient (too few clusters): {len(insufficient)}.",
        "",
        "## Headline numbers",
        f"- Zero-evidence ratings: {zero_ev:.1%}",
        "- Agents flagged (largest single cluster >= 50% of that agent's records, "
        f"among scored agents only): {n_flagged} ({flag_rate:.1%})",
    ]
    if len(scored):
        med_gap = (scored["naive_mean"] - scored["robust_score"]).abs().median()
        lines.append(f"- Median |mean - robust| among scored agents: {med_gap:.3f}")
    lines += ["", "## Figures"]
    for name, path in figures.items():
        lines += [f"### {name}", f"![{name}]({path})", ""]

    lines += ["## Sensitivity",
              "Spearman rank correlation of the top-N robust-score ranking under reasonable "
              "parameter perturbations (evidence-weight shape, sybil Jaccard threshold, sybil "
              "time window), each vs the base configuration.",
              "", sensitivity.to_markdown(index=False), ""]

    lines += ["## Adversarial evidence",
              "Known-answer attack scenarios (see `robustrep.report.adversarial.scenario_table`), "
              "each computed with **bootstrap_n=0**: point estimates only -- no confidence-interval "
              "claims are made from this table.", ""]
    if adversarial is not None:
        lines += [adversarial.to_markdown(index=False), ""]

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
        "so no two attacker ratee-sets are identical, evades sybil clustering entirely -- every "
        "such rater lands in its own singleton cluster. The residual defense is evidence "
        "weighting (an evasive farm is almost always evidence-free): `zero_evidence_ratio` is "
        "the signal that survives this evasion even when `sybil_flag` reads 0 and clustering "
        "sees nothing unusual.",
        "- **Large-ratee-group blind spot.** A ratee with more raters than `sybil_max_group` is "
        "skipped entirely for ratee-based sybil-clustering blocking (see `robustrep.sybil`), so "
        "a farm concentrated on one very popular ratee can evade the ratee-sharing signal by "
        "sheer volume, independent of the evasion technique above.",
        "- **DNS rebinding in the evidence fetcher (not mitigated in v0.1).** The SSRF guard "
        "resolves and checks an evidence URI's host once, but the HTTP client resolves it again "
        "independently; a rebinding attacker timed between the two resolutions can route a "
        "blind GET to a private address. No response content is ever exposed or stored -- only "
        "a 0-3 evidence level -- bounding the blast radius. See "
        "`robustrep.sources.evidence_fetch` for the full writeup and its xfail regression test.",
        "- **Two-tag lower-median bias.** Cross-tag combination is itself a weighted median: "
        "with exactly two tags of similar evidence mass, the lower-median tie-break above "
        "applies at the tag level too, so the lower-scored tag is preferred at an exact tie "
        "rather than averaging the two -- a small, systematic downward bias for ratees rated "
        "under precisely two evenly-weighted tags.",
        "",
        "## Provenance",
    ]
    prov = provenance or {}
    lines.append(f"- Rater profile mode: {prov.get('rater_profile_mode', 'unknown')}")
    lines.append(f"- Confirmations lag (data cut is final as of this block): "
                 f"{prov.get('confirmations', 'unknown')}")
    versions = prov.get("versions") or {}
    if versions:
        v_str = ", ".join(f"{k} {v}" for k, v in versions.items())
        lines.append(f"- Versions: {v_str}")
    return "\n".join(lines)
