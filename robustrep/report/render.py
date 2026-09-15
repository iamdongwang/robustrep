"""Markdown report from scores + records (spec Sec 5.5b: five figures, sensitivity,
adversarial evidence, limitations, provenance)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

# The ERC-8004 study's own headline finding, for scale: even the paper that
# introduced this feedback scheme found the overwhelming majority of ratings
# carried no interaction evidence at all.
_PAPER_CITATION = (
    "This is comparable to the study's (arXiv 2606.26028) reported 98.7-100% of "
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


def evidence_level_shares(non_revoked: pd.DataFrame) -> pd.Series:
    """Share of non-revoked records at each evidence level 0..3, reindexed so
    every level is present (0.0) even when absent from the data."""
    if len(non_revoked) == 0:
        return pd.Series(0.0, index=[0, 1, 2, 3])
    return non_revoked["evidence_level"].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0.0)


def _rater_concentration(records: pd.DataFrame) -> tuple[int, float, int]:
    """(distinct raters, median ratings-per-rater, max ratings-per-rater) over
    ALL records (revoked included -- a revoked rating still came from a real
    rater address)."""
    if len(records) == 0 or "rater" not in records.columns:
        return 0, 0.0, 0
    counts = records.groupby("rater").size()
    return int(counts.shape[0]), float(counts.median()), int(counts.max())


def _tag_hygiene(records: pd.DataFrame, min_count: int = 10) -> tuple[int, float]:
    """(distinct tag count, share of records whose tag has fewer than
    `min_count` records overall) over ALL records. `tag1` is free-text on
    ERC-8004, so a handful of records is a common outcome for a tag, not an
    error."""
    if len(records) == 0 or "tag" not in records.columns:
        return 0, 0.0
    counts = records["tag"].value_counts()
    rare_tags = counts[counts < min_count].index
    rare_share = records["tag"].isin(rare_tags).mean()
    return int(counts.shape[0]), float(rare_share)


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


# Share of the base top set sitting in the top-score tie block above which
# Spearman's rho is dominated by tie order rather than any ranking change.
_TIE_DOMINANCE_SHARE = 0.5


def _top_tie_block_lines(scored: pd.DataFrame, sensitivity: pd.DataFrame) -> list:
    """Diagnostic sentence(s) on the top-score tie block, or [] when the
    sensitivity table carries no base `top_set_size` (older callers)."""
    if scored.empty or "top_set_size" not in sensitivity.columns or "variant" not in sensitivity.columns:
        return []
    base = sensitivity[sensitivity["variant"] == "base"]
    if base.empty or pd.isna(base["top_set_size"].iloc[0]):
        return []
    base_size = int(base["top_set_size"].iloc[0])
    top_val = float(scored["robust_score"].max())
    n_tie = int((scored["robust_score"] == top_val).sum())
    text = (f"{n_tie} of the {len(scored)} scored agents tie at the top score ({top_val:.3f}), "
            f"against a base top set of {base_size}.")
    if base_size > 0 and n_tie / base_size >= _TIE_DOMINANCE_SHARE:
        text += (" The top set is essentially one tie block in which every member holds the "
                 "same rank, and Spearman's rho is therefore uninformative on this data: any "
                 "value is driven by the few agents outside the block, and a value near 0 "
                 "reflects tie order, not a ranking change. Read `top_set_jaccard`.")
    return [text]


def _budget_limited_variant_lines(sensitivity: pd.DataFrame) -> list:
    """One sentence explaining the sensitivity table's `budget_limited` column,
    or [] for a table that has no such column (an older//custom caller).

    Each variant re-clusters from scratch, so a variant can hit a sybil pair
    budget the base run did not; its row then measures the budget, not the
    parameter, and must not be read as a parameter effect.
    """
    if "budget_limited" not in sensitivity.columns:
        return []
    return ["A variant whose `budget_limited` is True hit a sybil pair budget while clustering "
            "(each variant re-clusters from scratch, and a wider sybil window generates more "
            "candidate pairs), so its row may reflect the budget rather than the parameter being "
            "varied; compare it against the base row's own `budget_limited`."]


def _budget_limit_lines(scores: pd.DataFrame) -> list:
    """The budget-limited-clustering bullet, or [] when nothing was skipped (or
    the caller attached no stats at all).

    Read off `scores.attrs["cluster_stats"]` -- a `robustrep.sybil.ClusterStats`
    as a dict, stamped by the CLI. Pair budgets degrade the clustering instead
    of aborting the run, and degrading can only UNDER-merge, so a budget-limited
    run has to say so and say which way its error points: never the flattering
    direction left unsaid.
    """
    stats = scores.attrs.get("cluster_stats") or {}
    size = int(stats.get("ratees_skipped_size", 0))
    per_ratee = int(stats.get("ratees_skipped_budget", 0))
    truncated = bool(stats.get("truncated", False))
    if not (size or per_ratee or truncated):
        return []
    return [
        f"- **Budget-limited clustering.** Sybil clustering was budget-limited: {size} ratee "
        f"block(s) skipped for size (> sybil_max_group), {per_ratee} for the per-ratee pair "
        f"budget, global pair budget reached: {'yes' if truncated else 'no'} "
        f"({int(stats.get('pairs_examined', 0))} candidate pairs examined, "
        f"{int(stats.get('pairs_tested', 0))} tested). Clusters among the affected raters "
        "may be under-merged, so their ratees' scores are less robust than reported, never more."
    ]


def render_markdown(scores: pd.DataFrame, records: pd.DataFrame, block: int, figures: dict,
                    sensitivity: pd.DataFrame, adversarial: Optional[pd.DataFrame] = None,
                    provenance: Optional[dict] = None) -> str:
    """Render the full Markdown report.

    `figures` maps a human figure title (e.g. "Fig 4. Largest rater clusters") ->
    relative file path (e.g. "fig4_sybil_clusters.png") for the Markdown image
    link -- the title is the heading text, never the raw filename. `provenance`,
    when given, may carry `rater_profile_mode`, `confirmations`, `config` (a dict
    of the scoring parameters actually used: bootstrap_n, bootstrap_seed,
    min_clusters, evidence_weights, and the sybil_* thresholds and pair budgets),
    and `versions`.

    `scores.attrs["cluster_stats"]`, when present (the CLI stamps it -- see
    `robustrep.sybil.ClusterStats`), adds a Limitations bullet whenever a sybil
    pair budget cut the clustering short.
    """
    non_revoked, n_revoked = _non_revoked(records)
    scored = scores[scores["insufficient"] == 0]
    insufficient = scores[scores["insufficient"] == 1]
    n_flagged, flag_rate = _flag_rate(scored)
    ev_shares = evidence_level_shares(non_revoked)
    zero_ev = float(ev_shares.loc[0])
    zero_one_ev = float(ev_shares.loc[0] + ev_shares.loc[1])
    level3_ev = float(ev_shares.loc[3])
    n_raters, rating_median, rating_max = _rater_concentration(records)
    n_tags, rare_tag_share = _tag_hygiene(records)

    lines = [
        "# Robust reputation on ERC-8004 (Base)",
        "",
        f"Data cut at block {block}. Ratings: {len(records)} ({n_revoked} revoked, excluded below). "
        f"Agents rated: {len(scores)}. Agents scored (>= min_clusters independent clusters): "
        f"{len(scored)}; insufficient (too few clusters): {len(insufficient)}.",
        "",
        "## Headline numbers",
        f"- No evidence URI (level 0): {zero_ev:.1%}",
        f"- No verifiable interaction evidence (levels 0-1): {zero_one_ev:.1%}. {_PAPER_CITATION}",
        f"- Verified on chain (level 3): {level3_ev:.1%}",
        "- Agents flagged (largest single cluster >= 50% of that agent's records, "
        f"among scored agents only): {n_flagged} ({flag_rate:.1%})",
        "- Read `sybil_flag` together with `n_clusters` and `zero_evidence_ratio`: in "
        "scenario F an attack split across two funders held 91% of the records while "
        "the largest single cluster was 45%, leaving the flag at 0.",
        f"- Rater concentration: {n_raters} distinct raters, {len(records)} ratings "
        f"({rating_median:.1f} median, {rating_max} max ratings per rater). With repeat raters "
        "this dense, the largest-single-cluster flag is mostly single-rater dominance (one "
        "address rating the same agent many times); read it with `n_raw` and `n_clusters`.",
        f"- Tag hygiene: {n_tags} distinct tags; {rare_tag_share:.1%} of records carry a tag "
        "with fewer than 10 records overall. tag1 is free text on ERC-8004; many values are "
        "sentences rather than categories. v0.1 keeps every tag as its own group; a rare-tag "
        "merge is a v0.2 item.",
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
              "window, normalization fit share), each vs the base configuration. Top sets are tie-inclusive (every "
              "agent scoring >= the N-th highest score). `top_set_jaccard` is the primary "
              "stability measure: the overlap of the base and variant top sets. Spearman's "
              "rho (Pearson correlation of average ranks over the union of the two top sets, "
              "no scipy dependency) is reported for completeness; a blank cell means one run "
              "ranked the whole union as a single tie while the other did not (zero variance "
              "on one side, so no correlation is defined).",
              *_top_tie_block_lines(scored, sensitivity),
              *_budget_limited_variant_lines(sensitivity),
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
        *_budget_limit_lines(scores),
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
        "- **Tag hygiene.** tag1 is free text on ERC-8004; many values are sentences rather "
        "than categories. v0.1 keeps every tag as its own group; a rare-tag merge is a v0.2 "
        "item.",
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
                    "sybil_jaccard", "sybil_window_s", "sybil_max_group", "sybil_max_pairs",
                    "sybil_max_pairs_per_ratee", "sybil_flag_share", "norm_fit_share",
                    "norm_rule_counts", "evidence_max_lookups_per_uri", "evidence_max_total_lookups",
                    "evidence_lookup_starved_uris"):
            if key in config:
                lines.append(f"  - `{key}`: {config[key]}")
    versions = prov.get("versions") or {}
    if versions:
        v_str = ", ".join(f"{k} {v}" for k, v in versions.items())
        lines.append(f"- Versions: {v_str}")
    return "\n".join(lines)
