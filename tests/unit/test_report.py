import json

import numpy as np
import pandas as pd

from robustrep import Config, score
from robustrep.report.export import export_json
from robustrep.report.figures import fig_evidence, fig_mean_vs_robust, fig_rank_shift, fig_sybil_clusters
from robustrep.report.render import render_markdown
from robustrep.report.sensitivity import sensitivity_table


def _data(records_factory):
    rows = []
    for a in range(6):
        rows += [dict(rater=f"r{a}{i}", ratee=str(a), value=50 + a * 8, evidence_level=i % 4) for i in range(4)]
    rec = records_factory(rows)
    return rec, score(rec, Config(bootstrap_n=5))


def test_figures_save_png(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    clusters = {r: r for r in rec["rater"]}
    for fn, args in [(fig_mean_vs_robust, (sc,)), (fig_rank_shift, (sc,)), (fig_evidence, (rec,)),
                     (fig_sybil_clusters, (rec, clusters))]:
        p = tmp_path / f"{fn.__name__}.png"
        fn(*args, out=p)
        assert p.exists() and p.stat().st_size > 1000


def test_sensitivity_table_has_spearman(records_factory):
    rec, _ = _data(records_factory)
    t = sensitivity_table(rec, Config(bootstrap_n=0), top_n=5)
    assert {"variant", "spearman_top"} <= set(t.columns) and len(t) >= 4
    assert t["spearman_top"].between(-1, 1).all()


def test_render_and_export(tmp_path, records_factory):
    rec, sc = _data(records_factory)
    md = render_markdown(sc, rec, block=123, figures={"fig1": "fig1.png"}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]))
    assert "block 123" in md and "fig1.png" in md
    p = export_json(sc, block=123, out=tmp_path / "scores.json")
    data = json.loads(p.read_text())
    assert data["block"] == 123 and len(data["scores"]) == 6 and "robust_score" in data["scores"][0]
    assert {"robustrep", "numpy", "pandas"} <= set(data["versions"])


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


def test_render_markdown_wording_constraints(records_factory):
    rec, sc = _data(records_factory)
    # inject one insufficient row so the flag-rate-among-scored logic is exercised.
    sc = sc.copy()
    sc.loc[0, "insufficient"] = 1
    sc.loc[0, "robust_score"] = np.nan
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]), provenance=dict(
        rater_profile_mode="onchain", confirmations=20, versions={"robustrep": "0.1.0"}))
    for phrase in ["largest single cluster", "lower weighted median", "evidence mass",
                   "zero_evidence_ratio", "bootstrap_n=0", "Limitations", "Provenance"]:
        assert phrase in md, phrase


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
