import json
import re

import numpy as np
import pandas as pd
import pytest

from robustrep import Config, score
from robustrep.report.adversarial import scenario_table
from robustrep.report.export import export_json
from robustrep.report.figures import fig_evidence, fig_mean_vs_robust, fig_rank_shift, fig_sybil_clusters
from robustrep.report.render import render_markdown
from robustrep.report.sensitivity import fig_sensitivity, sensitivity_table, tie_aware_sensitivity

HOUR = 3600
WEEK = 7 * 24 * HOUR


def _data(records_factory):
    rows = []
    for a in range(6):
        rows += [dict(rater=f"r{a}{i}", ratee=str(a), value=50 + a * 8, evidence_level=i % 4) for i in range(4)]
    rec = records_factory(rows)
    return rec, score(rec, Config(bootstrap_n=5))


def _sensitivity_data(records_factory):
    """Distinct funders (`meta` frame) + widely spread `ts` (weeks apart) so every
    rater is its own singleton cluster under the base config -- except one
    deliberately close pair (30h apart) that only merges under `window_72h`.
    Ratee "mix" carries raters at every evidence level with DIFFERENT values, so a
    steeper weight shape changes which value the weighted median picks. Both are
    engineered to move the top-N ranking away from 1.0 for at least one variant,
    while every other rater pair stays a singleton cluster under every variant
    (distinct funder, and either no shared ratee or a multi-week gap), so no
    ratee ever becomes `insufficient` and the base-vs-base correlation is exactly
    1.0.
    """
    rows, meta_rows = [], []

    for a in range(5):
        ratee = f"plain{a}"
        for i in range(4):
            rater = f"r{a}_{i}"
            rows.append(dict(rater=rater, ratee=ratee, value=20 + a * 15, ts=i * WEEK, evidence_level=2))
            meta_rows.append(dict(rater=rater, first_seen_ts=i * WEEK, funder=f"f_{rater}"))

    for i, v in enumerate([10, 40, 70, 95]):
        rater = f"mix_{i}"
        rows.append(dict(rater=rater, ratee="mix", value=v, ts=i * WEEK, evidence_level=i))
        meta_rows.append(dict(rater=rater, first_seen_ts=i * WEEK, funder=f"f_{rater}"))

    win_ts = [0, 30 * HOUR, 20 * WEEK, 21 * WEEK]
    for i, (ts, v) in enumerate(zip(win_ts, [30, 90, 30, 90])):
        rater = f"win_{i}"
        rows.append(dict(rater=rater, ratee="window", value=v, ts=ts, evidence_level=2))
        meta_rows.append(dict(rater=rater, first_seen_ts=ts, funder=f"f_{rater}"))

    return records_factory(rows), pd.DataFrame(meta_rows)


def test_figures_save_png(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    clusters = {r: r for r in rec["rater"]}
    for fn, args in [(fig_mean_vs_robust, (sc,)), (fig_rank_shift, (sc,)), (fig_evidence, (rec,)),
                     (fig_sybil_clusters, (rec, clusters))]:
        p = tmp_path / f"{fn.__name__}.png"
        fn(*args, out=p)
        assert p.exists() and p.stat().st_size > 1000


def test_sensitivity_table_has_spearman(records_factory):
    records, meta = _sensitivity_data(records_factory)
    t = sensitivity_table(records, Config(bootstrap_n=0), meta=meta, top_n=5)
    assert {"variant", "spearman_top", "spearman_union", "top_set_jaccard", "top_set_size"} <= set(t.columns)
    assert len(t) >= 4
    assert t["spearman_top"].dropna().between(-1, 1).all()
    assert (t["spearman_top"] == t["spearman_union"]).all()
    assert t["top_set_jaccard"].dropna().between(0, 1).all()
    assert (t["top_set_size"] >= 0).all()


def test_sensitivity_table_reflects_real_perturbations(records_factory):
    """Item 3: a fixture with spread ts + distinct funders (via `meta`) so every
    ratee is scorable; steeper weights and a wider sybil window each actually
    move the top-N ranking away from the base configuration."""
    records, meta = _sensitivity_data(records_factory)
    t = sensitivity_table(records, Config(bootstrap_n=0), meta=meta, top_n=10)
    assert len(t) == 9
    # normalization fit share is a perturbation axis of its own (Rec 2)
    assert {"fit_0.8", "fit_1.0"} <= set(t["variant"])
    by_variant = t.set_index("variant")["spearman_top"]
    assert by_variant["base"] == pytest.approx(1.0)
    assert (by_variant < 1.0).any()
    assert by_variant["weights_steeper"] < 1.0
    assert by_variant["window_72h"] < 1.0


def test_sensitivity_table_defined_correlation_when_one_ratee_drops_out(records_factory):
    """A ratee that becomes `insufficient` under a variant no longer vanishes
    from the comparison (the old intersect-and-dropna behaviour, which read a
    fabricated-feeling NaN off a single overlapping point) -- it stays in the
    UNION of the two tie-inclusive top sets and gets that run's lowest rank
    (see `tie_aware_sensitivity`), so the correlation is still defined and
    reflects a real, measured ranking change. Two raters on ratee "only"
    (distinct funders, both rating only that ratee) sit 50h apart: outside the
    default/6h window (no merge, "only" scores fine with 3 singleton clusters)
    but inside the 72h window (merge -> 2 clusters, below min_clusters=3 ->
    "only" alone drops out under window_72h)."""
    rows = [
        dict(rater="p0", ratee="only", value=50, ts=0, evidence_level=2),
        dict(rater="p1", ratee="only", value=60, ts=50 * HOUR, evidence_level=2),
        dict(rater="p2", ratee="only", value=70, ts=2000 * HOUR, evidence_level=2),
        dict(rater="c0", ratee="control", value=20, ts=0, evidence_level=2),
        dict(rater="c1", ratee="control", value=30, ts=500 * HOUR, evidence_level=2),
        dict(rater="c2", ratee="control", value=40, ts=5000 * HOUR, evidence_level=2),
    ]
    meta = pd.DataFrame([
        dict(rater="p0", first_seen_ts=0, funder="fA"),
        dict(rater="p1", first_seen_ts=50 * HOUR, funder="fB"),
        dict(rater="p2", first_seen_ts=2000 * HOUR, funder="fC"),
        dict(rater="c0", first_seen_ts=0, funder="fD"),
        dict(rater="c1", first_seen_ts=500 * HOUR, funder="fE"),
        dict(rater="c2", first_seen_ts=5000 * HOUR, funder="fF"),
    ])
    records = records_factory(rows)
    t = sensitivity_table(records, Config(bootstrap_n=0), meta=meta, top_n=5)
    by_variant = t.set_index("variant")["spearman_top"]
    assert by_variant["base"] == pytest.approx(1.0)
    assert pd.notna(by_variant["window_72h"])
    assert by_variant["window_72h"] < 1.0
    jaccard = t.set_index("variant")["top_set_jaccard"]
    assert jaccard["window_72h"] == pytest.approx(0.5)  # "only" dropped, "control" kept: 1/2


def test_sensitivity_table_nan_when_union_has_fewer_than_two_ratees(records_factory):
    """The true degenerate case for NaN under the new tie-aware method: the
    union of the two tie-inclusive top sets itself has fewer than 2 members
    (here: a single ratee total, which then drops out entirely under
    window_72h, leaving the union at exactly that one ratee) -- there just
    isn't enough data to define a correlation, so NaN, never a fabricated
    value."""
    rows = [
        dict(rater="p0", ratee="only", value=50, ts=0, evidence_level=2),
        dict(rater="p1", ratee="only", value=60, ts=50 * HOUR, evidence_level=2),
        dict(rater="p2", ratee="only", value=70, ts=2000 * HOUR, evidence_level=2),
    ]
    meta = pd.DataFrame([
        dict(rater="p0", first_seen_ts=0, funder="fA"),
        dict(rater="p1", first_seen_ts=50 * HOUR, funder="fB"),
        dict(rater="p2", first_seen_ts=2000 * HOUR, funder="fC"),
    ])
    records = records_factory(rows)
    t = sensitivity_table(records, Config(bootstrap_n=0), meta=meta, top_n=5)
    by_variant = t.set_index("variant")["spearman_top"]
    # A single-ratee universe can never define a correlation (zero variance,
    # only one point) -- NaN even for the identical-config "base" row itself.
    assert pd.isna(by_variant["base"])
    assert pd.isna(by_variant["window_72h"])
    jaccard = t.set_index("variant")["top_set_jaccard"]
    assert jaccard["base"] == pytest.approx(1.0)  # base vs itself: same singleton top set
    assert jaccard["window_72h"] == pytest.approx(0.0)  # union non-empty ({"only"}), intersection empty


def test_tie_aware_sensitivity_handles_large_tie_group():
    """40 agents, 30 of which tie at 1.0 -- mirroring the real cut, where 1,067
    of 5,170 scored agents tie at exactly robust_score==1.0. With
    `rank(method="first")` on an arbitrary top-N cut through that tie group,
    any row-order difference between two otherwise-identical runs churns
    which N-of-30 get picked as "the" top set, reading a near-zero Spearman
    purely from tie order -- never from any real ranking change. The
    tie-aware top set (every score >= the N-th highest, ties included) fixes
    this: a no-op variant is perfectly stable, reordering within the tie
    changes nothing, and only an actual change in *which* agents are on top
    moves the numbers."""
    base = pd.Series({f"a{i}": 1.0 for i in range(30)} | {f"b{i}": 0.9 - i * 0.09 for i in range(10)})

    # No-op: identical scores -> perfect agreement.
    r = tie_aware_sensitivity(base, base.copy(), top_n=10)
    assert r["spearman_union"] == pytest.approx(1.0)
    assert r["top_set_jaccard"] == pytest.approx(1.0)
    assert r["top_set_size"] == 30  # the whole 30-way tie, not an arbitrary 10

    # Reordering the SAME values (only the Series' row order changes, no score
    # changes) must not move the tie-inclusive top set or its correlation at
    # all -- unlike rank(method="first"), which is order-sensitive.
    reordered = base.iloc[::-1]
    r_reordered = tie_aware_sensitivity(base, reordered, top_n=10)
    assert r_reordered["top_set_jaccard"] == pytest.approx(1.0)
    assert r_reordered["spearman_union"] == pytest.approx(1.0)

    # Swap 5 of the 30 tied agents out (drop a25-a29) for 5 that were outside
    # the tie group (raise b0-b4 to 1.0): the tie-inclusive top set changes by
    # exactly 5 members each way -- jaccard drops accordingly (25 kept / 35
    # union = 5/7), unlike the untouched no-op and reorder-only cases above.
    variant = base.copy()
    for i in range(25, 30):
        variant[f"a{i}"] = 0.05
    for i in range(5):
        variant[f"b{i}"] = 1.0
    r_swap = tie_aware_sensitivity(base, variant, top_n=10)
    assert r_swap["top_set_size"] == 30
    assert r_swap["top_set_jaccard"] == pytest.approx(25 / 35)
    assert r_swap["spearman_union"] < 1.0


def test_fig_sensitivity_annotates_nan_bars(tmp_path):
    table = pd.DataFrame([dict(variant="base", spearman_top=1.0),
                          dict(variant="window_72h", spearman_top=float("nan"))])
    p = tmp_path / "fig5_nan.png"
    fig_sensitivity(table, out=p)
    assert p.exists() and p.stat().st_size > 500


def test_render_and_export(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    md = render_markdown(sc, rec, block=123, figures={"Fig 1. Mean vs robust score": "fig1.png"},
                         sensitivity=pd.DataFrame([dict(variant="base", spearman_top=1.0)]))
    assert "block 123" in md and "fig1.png" in md
    p = export_json(sc, block=123, out=tmp_path / "scores.json")
    data = json.loads(p.read_text())
    assert data["block"] == 123 and len(data["scores"]) == 6 and "robust_score" in data["scores"][0]
    assert {"robustrep", "numpy", "pandas"} <= set(data["versions"])
    assert data["config"] == {}


# --- additional acceptance criteria (empty sets, wording, NaN-safe export) ------


def test_figures_handle_empty_scored_set(tmp_path, records_factory):
    rows = [dict(rater="r0", ratee="only", value=50, evidence_level=0)]
    rec = records_factory(rows)
    sc = score(rec, Config(bootstrap_n=0, min_clusters=3))
    assert sc["insufficient"].eq(1).all()
    for fn, args in [(fig_mean_vs_robust, (sc,)), (fig_rank_shift, (sc,))]:
        p = tmp_path / f"{fn.__name__}_empty.png"
        fn(*args, out=p)
        assert p.exists() and p.stat().st_size > 500


def test_fig_rank_shift_handles_fewer_than_top_n(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    p = tmp_path / "rank_shift_small.png"
    fig_rank_shift(sc, out=p, top_n=1000)
    assert p.exists() and p.stat().st_size > 500


def test_fig_sybil_clusters_ranks_by_member_count(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    # group raters r{a}0/r{a}1 of each ratee into one 2-member cluster, leaving
    # r{a}2/r{a}3 as singletons -- exercises the member-count ranking + the
    # agents-covered annotation on a non-trivial clustering.
    clusters = {}
    for r in rec["rater"]:
        clusters[r] = r[:-1] + "0" if r[-1] in "01" else r
    p = tmp_path / "fig4.png"
    fig_sybil_clusters(rec, clusters, out=p)
    assert p.exists() and p.stat().st_size > 1000


def test_fig_evidence_and_sybil_clusters_handle_empty_records(tmp_path):
    empty = pd.DataFrame(columns=["rater", "ratee", "evidence_level"])
    p1 = tmp_path / "fig3_empty.png"
    fig_evidence(empty, out=p1)
    assert p1.exists() and p1.stat().st_size > 500
    p2 = tmp_path / "fig4_empty.png"
    fig_sybil_clusters(empty, {}, out=p2)
    assert p2.exists() and p2.stat().st_size > 500


def test_fig_sensitivity_handles_empty_table(tmp_path):
    p = tmp_path / "fig5_empty.png"
    fig_sensitivity(pd.DataFrame(columns=["variant", "spearman_top"]), out=p)
    assert p.exists() and p.stat().st_size > 500


def test_render_markdown_tolerates_missing_revoked_column_and_bare_adversarial(records_factory):
    """`records` without a `revoked` column (e.g. a raw pre-validation frame) and an
    `adversarial` table that doesn't carry the break-even columns must both be
    handled gracefully (no KeyError), exercising the early-return branches of
    `_non_revoked` and `_measured_boundaries_paragraph`."""
    rec, sc = _data(records_factory)
    assert "revoked" not in rec.columns
    bare_adversarial = pd.DataFrame([dict(scenario="X", naive_mean=0.5, robust_score=0.5)])
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]), adversarial=bare_adversarial)
    assert "0 revoked" in md
    assert "Measured boundaries" not in md


def test_render_markdown_nan_cells_render_blank_not_literal_nan(records_factory):
    """Item 1: NaN cells in the sensitivity/adversarial tables must render as a
    blank Markdown cell, never the literal string "nan"."""
    rec, sc = _data(records_factory)
    adv = scenario_table()  # break_even_k/break_even_k_boost are NaN on most rows
    sensitivity = pd.DataFrame([dict(variant="base", spearman_top=1.0),
                                dict(variant="window_72h", spearman_top=float("nan"))])
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=sensitivity, adversarial=adv)
    assert not re.search(r"\bnan\b", md.lower())


def test_measured_boundaries_paragraph_handles_missing_break_even_k_boost(records_factory):
    """Item 3: `break_even_k_boost` missing entirely from the adversarial table
    (not just NaN on the H row) must not raise -- the boost side renders "n/a"."""
    rec, sc = _data(records_factory)
    adv = scenario_table().drop(columns=["break_even_k_boost"])
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]), adversarial=adv)
    assert "Measured boundaries" in md
    assert "boost at n/a" in md
    assert not re.search(r"\bnan\b", md.lower())


def test_export_json_config_key(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    config = dict(bootstrap_n=5, min_clusters=3, evidence_weights=[0.1, 0.3, 0.7, 1.0])
    p = export_json(sc, block=1, out=tmp_path / "scores.json", config=config)
    data = json.loads(p.read_text())
    assert data["config"] == config
    assert "config" in data and "versions" in data


def test_render_markdown_wording_constraints(records_factory):
    rec, sc = _data(records_factory)
    # inject one insufficient row so the flag-rate-among-scored logic is exercised.
    sc = sc.copy()
    sc.loc[0, "insufficient"] = 1
    sc.loc[0, "robust_score"] = np.nan
    adv = scenario_table()
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]), adversarial=adv, provenance=dict(
        rater_profile_mode="onchain", confirmations=20,
        config=dict(bootstrap_n=0, bootstrap_seed=0, min_clusters=3, evidence_weights=[0.1, 0.3, 0.7, 1.0],
                   sybil_jaccard=0.8, sybil_window_s=86400, sybil_max_group=2000, sybil_flag_share=0.5),
        versions={"robustrep": "0.1.0"}))
    for phrase in ["largest single cluster", "lower weighted median", "evidence mass",
                   "zero_evidence_ratio", "bootstrap_n=0", "Limitations", "Provenance",
                   "arXiv 2606.26028", "revoked", "bootstrap_n`: 0"]:
        assert phrase in md, phrase


def test_render_markdown_never_claims_sybil_detection_and_states_measured_boundaries(records_factory):
    rec, sc = _data(records_factory)
    adv = scenario_table()
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]), adversarial=adv, provenance=dict(
        rater_profile_mode="onchain", confirmations=20, config={}, versions={}))
    lowered = md.lower()
    for banned in ("sybil detected", "sybil detection", "detects sybils", "sybil cluster"):
        assert banned not in lowered, banned
    assert "flips the score at k=30" in md
    assert "smear flips at 35" in md


def test_render_markdown_headline_evidence_and_rater_concentration_lines(records_factory):
    """`_data()` gives 24 records over 6 ratees, 4 per ratee at evidence levels
    0/1/2/3 (6 records at each level), each from its own distinct rater (24
    raters, 1 rating each) -- clean round numbers for the new headline lines."""
    rec, sc = _data(records_factory)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]))
    assert "- No evidence URI (level 0): 25.0%" in md
    assert "- No verifiable interaction evidence (levels 0-1): 50.0%. " in md
    assert "comparable to the study's" in md and "98.7-100%" in md
    # the paper-contrast sentence is attached to the levels-0-1 line, not the level-0 line.
    level01_line = next(line for line in md.splitlines() if "levels 0-1" in line)
    assert "comparable to the study's" in level01_line and "98.7-100%" in level01_line
    assert "- Verified on chain (level 3): 25.0%" in md
    assert "- Rater concentration: 24 distinct raters, 24 ratings (1.0 median, 1 max ratings per rater)." in md
    assert ("With repeat raters this dense, the largest-single-cluster flag is mostly "
           "single-rater dominance (one address rating the same agent many times); read it "
           "with `n_raw` and `n_clusters`.") in md


def test_render_markdown_tag_hygiene_headline_and_limitations(records_factory):
    """15 records: 3 tagged with a rare, free-text tag (< 10 records overall) and
    12 tagged "quality" -- 2 distinct tags, 20% of records on a rare tag."""
    rows = []
    for a in range(3):
        for i in range(5):
            tag = "please_fix_the_thing_it_broke_again" if (a == 0 and i < 3) else "quality"
            rows.append(dict(rater=f"r{a}{i}", ratee=str(a), value=50, evidence_level=2, tag=tag))
    rec = records_factory(rows)
    sc = score(rec, Config(bootstrap_n=0))
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]))
    assert ("- Tag hygiene: 2 distinct tags; 20.0% of records carry a tag with fewer than 10 "
           "records overall.") in md
    sentence = ("tag1 is free text on ERC-8004; many values are sentences rather than "
               "categories. v0.1 keeps every tag as its own group; a rare-tag merge is a "
               "v0.2 item.")
    assert md.count(sentence) == 2  # once in Headline numbers, once in Limitations
    assert "- **Tag hygiene.** " + sentence in md


def test_export_json_nan_to_null_and_sorted(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    sc = sc.copy()
    sc.loc[0, "insufficient"] = 1
    sc.loc[0, "robust_score"] = np.nan
    sc.loc[0, "ci_low"] = np.nan
    sc.loc[0, "ci_high"] = np.nan
    p = export_json(sc, block=5, out=tmp_path / "scores2.json")
    text = p.read_text()
    data = json.loads(text)
    assert data["schema_version"] == 1
    scores = data["scores"]
    robust_scores = [row["robust_score"] for row in scores]
    assert robust_scores[-1] is None
    non_null = [v for v in robust_scores if v is not None]
    assert non_null == sorted(non_null, reverse=True)


# --- sensitivity prose: top-score tie block, Jaccard primary -------------------


def _sensitivity_with_base_size(size):
    return pd.DataFrame([dict(variant="base", spearman_union=1.0, spearman_top=1.0,
                              top_set_jaccard=1.0, top_set_size=size),
                         dict(variant="window_72h", spearman_union=float("nan"),
                              spearman_top=float("nan"), top_set_jaccard=0.99, top_set_size=size)])


def test_sensitivity_prose_names_jaccard_primary_and_explains_blank_rho(records_factory):
    rec, sc = _data(records_factory)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=_sensitivity_with_base_size(3))
    section = md.split("## Sensitivity")[1].split("## Adversarial")[0]
    assert "`top_set_jaccard` is the primary stability measure" in section
    assert "zero variance" in section
    assert "too few ratees overlapped" not in section


def test_sensitivity_prose_reports_top_tie_block_when_it_dominates_base_top_set(records_factory):
    rec, sc = _data(records_factory)
    scored_mask = sc["insufficient"] == 0
    n_scored = int(scored_mask.sum())
    assert n_scored >= 2
    sc = sc.copy()
    sc.loc[scored_mask, "robust_score"] = 1.0  # every scored agent ties at the top
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=_sensitivity_with_base_size(n_scored))
    section = md.split("## Sensitivity")[1].split("## Adversarial")[0]
    assert f"{n_scored} of the {n_scored} scored agents tie at the top score (1.000)" in section
    assert f"base top set of {n_scored}" in section
    assert "Spearman's rho is therefore uninformative on this data" in section


def test_sensitivity_prose_omits_tie_warning_when_top_set_not_dominated(records_factory):
    rec, sc = _data(records_factory)
    scored_mask = sc["insufficient"] == 0
    n_scored = int(scored_mask.sum())
    sc = sc.copy()
    # strictly distinct scores -> the top tie block is a single agent
    sc.loc[scored_mask, "robust_score"] = np.linspace(0.5, 1.0, n_scored)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=_sensitivity_with_base_size(n_scored))
    section = md.split("## Sensitivity")[1].split("## Adversarial")[0]
    assert f"1 of the {n_scored} scored agents tie at the top score" in section
    assert "uninformative" not in section


def test_sensitivity_prose_skips_tie_diagnostic_without_top_set_size(records_factory):
    rec, sc = _data(records_factory)
    md = render_markdown(sc, rec, block=1, figures={},
                         sensitivity=pd.DataFrame([dict(variant="base", spearman_top=1.0)]))
    section = md.split("## Sensitivity")[1].split("## Adversarial")[0]
    assert "tie at the top score" not in section
    assert "`top_set_jaccard` is the primary stability measure" in section


# --- budget-limited sybil clustering (H2) --------------------------------------


_BASE_SENS = pd.DataFrame([dict(variant="base", spearman_top=1.0)])


def _with_cluster_stats(sc, **stats):
    out = sc.copy()
    out.attrs["cluster_stats"] = dict(
        {"pairs_tested": 0, "pairs_examined": 0, "ratees_skipped_size": 0,
         "ratees_skipped_budget": 0, "truncated": False},
        **stats)
    return out


def test_render_markdown_budget_limited_bullet(records_factory):
    rec, sc = _data(records_factory)
    sc = _with_cluster_stats(sc, pairs_tested=5, pairs_examined=8, ratees_skipped_budget=2,
                             truncated=True)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=_BASE_SENS)
    assert ("Sybil clustering was budget-limited: 0 ratee block(s) skipped for size "
            "(> sybil_max_group), 2 for the per-ratee pair budget, global pair budget "
            "reached: yes (8 candidate pairs examined, 5 tested). Clusters among the "
            "affected raters may be "
            "under-merged, so their ratees' scores are less robust than reported, "
            "never more.") in md
    assert "## Limitations" in md.split("budget-limited")[0]


def test_render_markdown_budget_bullet_says_no_when_global_budget_untouched(records_factory):
    rec, sc = _data(records_factory)
    sc = _with_cluster_stats(sc, pairs_tested=7, pairs_examined=7, ratees_skipped_size=3)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=_BASE_SENS)
    assert ("3 ratee block(s) skipped for size (> sybil_max_group), 0 for the per-ratee "
            "pair budget, global pair budget reached: no "
            "(7 candidate pairs examined, 7 tested).") in md


def test_render_markdown_budget_bullet_on_truncation_alone(records_factory):
    """`truncated` on its own -- no ratee block skipped either way -- is still a
    budget-limited run and still has to be stated."""
    rec, sc = _data(records_factory)
    sc = _with_cluster_stats(sc, pairs_tested=9, pairs_examined=12, truncated=True)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=_BASE_SENS)
    assert ("Sybil clustering was budget-limited: 0 ratee block(s) skipped for size "
            "(> sybil_max_group), 0 for the per-ratee pair budget, global pair budget "
            "reached: yes (12 candidate pairs examined, 9 tested).") in md


def test_render_markdown_no_budget_bullet_when_every_counter_is_zero(records_factory):
    rec, sc = _data(records_factory)
    sc = _with_cluster_stats(sc, pairs_tested=12, pairs_examined=12)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=_BASE_SENS)
    assert "budget-limited" not in md


def test_render_markdown_without_cluster_stats_has_no_budget_bullet(records_factory):
    rec, sc = _data(records_factory)
    assert "cluster_stats" not in sc.attrs
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=_BASE_SENS)
    assert "budget-limited" not in md


# --- sensitivity budget flag and scores.json cluster_stats (I5) ---------------


def test_sensitivity_table_flags_budget_limited_variants(records_factory):
    """Every variant re-clusters, so each one can hit a pair budget on its own
    (window_72h widens the window and triples the candidate pairs). The table
    has to say which ones did."""
    records, meta = _sensitivity_data(records_factory)
    healthy = sensitivity_table(records, Config(bootstrap_n=0), meta=meta, top_n=5)
    assert "budget_limited" in healthy.columns
    assert not healthy["budget_limited"].any()

    starved = sensitivity_table(records, Config(bootstrap_n=0, sybil_max_pairs=1), meta=meta, top_n=5)
    assert starved["budget_limited"].all()


def test_render_markdown_explains_budget_limited_sensitivity_column(records_factory):
    rec, sc = _data(records_factory)
    sens = pd.DataFrame([dict(variant="base", spearman_top=1.0, budget_limited=False),
                         dict(variant="window_72h", spearman_top=0.9, budget_limited=True)])
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=sens)
    section = md.split("## Sensitivity")[1].split("## Adversarial")[0]
    assert "budget_limited" in section
    assert "rather than the parameter" in section


def test_render_markdown_omits_budget_column_prose_without_the_column(records_factory):
    rec, sc = _data(records_factory)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=_BASE_SENS)
    section = md.split("## Sensitivity")[1].split("## Adversarial")[0]
    assert "budget_limited" not in section


def test_export_json_carries_cluster_stats(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    stats = {"pairs_tested": 7, "pairs_examined": 9, "ratees_skipped_size": 1,
             "ratees_skipped_budget": 0, "truncated": False}
    p = export_json(sc, block=1, out=tmp_path / "scores.json", config={"bootstrap_n": 5},
                    cluster_stats=stats)
    data = json.loads(p.read_text())
    assert data["cluster_stats"] == stats
    assert data["schema_version"] == 1  # additive key: consumers reading v1 keep working


def test_export_json_cluster_stats_defaults_to_empty(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    p = export_json(sc, block=1, out=tmp_path / "scores.json")
    assert json.loads(p.read_text())["cluster_stats"] == {}


# --- published JSON: no address may ever leak; full provenance versions ------


def test_export_json_refuses_to_publish_an_address(tmp_path, records_factory):
    # scores.json is served publicly. Today no column carries a 0x address,
    # but a future explain-style field could; the guard fails the export
    # loudly rather than publishing one.
    rec, sc = _data(records_factory)
    sc = sc.copy()
    sc.loc[0, "ratee"] = "0x" + "ab" * 20
    with pytest.raises(ValueError) as e:
        export_json(sc, block=1, out=tmp_path / "scores.json")
    assert "address" in str(e.value)
    # The message must not itself repeat the address it refused.
    assert "0x" not in str(e.value)


def test_export_json_allows_a_tx_hash_length_hex_string(tmp_path, records_factory):
    # The guard targets 40-hex addresses specifically: a 64-hex tx hash is not
    # an address and must not trip it.
    rec, sc = _data(records_factory)
    sc = sc.copy()
    sc.loc[0, "ratee"] = "0x" + "cd" * 32
    p = export_json(sc, block=1, out=tmp_path / "scores.json")
    assert json.loads(p.read_text())["block"] == 1


def test_export_json_versions_cover_every_output_affecting_dependency(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    p = export_json(sc, block=1, out=tmp_path / "scores.json")
    versions = json.loads(p.read_text())["versions"]
    assert set(versions) == {"robustrep", "python", "numpy", "pandas", "matplotlib",
                             "requests", "urllib3", "eth_abi"}
    assert all(isinstance(v, str) and v for v in versions.values())


# --- Limitations wording (security review) -----------------------------------


def test_render_markdown_limitations_state_the_unique_tag_gaming_vector(records_factory):
    rec, sc = _data(records_factory)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]))
    bullet = next(line for line in md.splitlines() if "Unique-tag gaming" in line)
    for phrase in ("tag1", "normalization group", "1.0", "binary", "evidence mass"):
        assert phrase in bullet, phrase


def test_render_markdown_ssrf_bullet_states_the_oracle_not_a_blind_request(records_factory):
    # The old bullet claimed "no response content is ever exposed", which is
    # wrong: classify consumes the fetched text, so a bypass is a four-state
    # oracle. Keep this bullet consistent with evidence_fetch's docstring.
    rec, sc = _data(records_factory)
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]))
    bullet = next(line for line in md.splitlines() if "DNS rebinding" in line)
    for phrase in ("four-state oracle", "not a blind request", "parser", "allowlist",
                   "rebuilt", "robustrep.sources.evidence_fetch"):
        assert phrase in bullet, phrase
    assert "No response content is ever exposed" not in md


def test_versions_is_public_and_is_what_export_writes(tmp_path, records_factory):
    # cli._provenance renders the same mapping into report.md, so the two
    # published artifacts can never disagree about the stack that made them.
    from robustrep.report.export import versions

    rec, sc = _data(records_factory)
    p = export_json(sc, block=1, out=tmp_path / "scores.json")
    assert json.loads(p.read_text())["versions"] == versions()


def test_address_guard_scope_is_exactly_the_documented_one(tmp_path, records_factory):
    # The guard is a backstop against an accidental future leak, not a
    # sanitizer: these three shapes are documented as NOT caught, and the
    # docstring stays honest only if the behaviour is pinned.
    rec, sc = _data(records_factory)
    for uncaught in ("0X" + "AB" * 20,                  # uppercase 0X prefix
                     "0x" + "00" * 12 + "ab" * 20,      # ABI-padded to 32 bytes
                     "0x" + "ab" * 40):                 # two addresses concatenated
        frame = sc.copy()
        frame.loc[0, "ratee"] = uncaught
        export_json(frame, block=1, out=tmp_path / "scores.json")  # does not raise
    caught = sc.copy()
    caught.loc[0, "ratee"] = "0x" + "ab" * 20
    with pytest.raises(ValueError):
        export_json(caught, block=1, out=tmp_path / "scores.json")
