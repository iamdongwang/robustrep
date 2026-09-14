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
from robustrep.report.sensitivity import fig_sensitivity, sensitivity_table

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
    assert {"variant", "spearman_top"} <= set(t.columns) and len(t) >= 4
    assert t["spearman_top"].dropna().between(-1, 1).all()


def test_sensitivity_table_reflects_real_perturbations(records_factory):
    """Item 3: a fixture with spread ts + distinct funders (via `meta`) so every
    ratee is scorable; steeper weights and a wider sybil window each actually
    move the top-N ranking away from the base configuration."""
    records, meta = _sensitivity_data(records_factory)
    t = sensitivity_table(records, Config(bootstrap_n=0), meta=meta, top_n=10)
    assert len(t) == 7
    by_variant = t.set_index("variant")["spearman_top"]
    assert by_variant["base"] == pytest.approx(1.0)
    assert (by_variant < 1.0).any()
    assert by_variant["weights_steeper"] < 1.0
    assert by_variant["window_72h"] < 1.0


def test_sensitivity_table_nan_when_overlap_too_small(records_factory):
    """Item 1: fewer than 2 overlapping ratees between the base top-N and a
    variant's scored set must report NaN, never a fabricated 1.0. Two raters on
    ratee "only" (distinct funders, both rating only that ratee so their Jaccard
    is trivially 1.0) sit 50h apart: outside the default/6h window (no merge,
    "only" scores fine with 3 singleton clusters) but inside the 72h window
    (merge -> 2 clusters, below min_clusters=3 -> "only" alone drops out under
    window_72h). A second, always-stable "control" ratee keeps the base-vs-base
    overlap at 2 points (a real, defined correlation) so this isolates the
    window_72h-specific drop to exactly 1 overlapping point -- NaN, not 1.0."""
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
    assert pd.isna(by_variant["window_72h"])


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
