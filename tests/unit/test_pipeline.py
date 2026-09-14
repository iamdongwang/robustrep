import math
import time

import numpy as np
import pandas as pd

from robustrep import Config, score
from robustrep.schema import RECORD_COLUMNS, RESULT_COLUMNS


def _cfg(**kw):
    return Config(**{"bootstrap_n": 50, **kw})


def test_output_columns_and_one_row_per_ratee(records_factory):
    df = records_factory([dict(rater=f"r{i}", ratee="A", value=80, evidence_level=3) for i in range(4)]
                         + [dict(rater="x", ratee="B", value=10)])
    out = score(df, _cfg())
    assert list(out.columns) == RESULT_COLUMNS and sorted(out["ratee"]) == ["A", "B"]


def test_insufficient_when_fewer_than_three_clusters(records_factory):
    out = score(records_factory([dict(rater="a", ratee="A", value=90), dict(rater="b", ratee="A", value=90)]), _cfg())
    row = out.set_index("ratee").loc["A"]
    assert row["insufficient"] == 1 and math.isnan(row["robust_score"]) and row["n_clusters"] == 2


def test_revoked_rows_are_ignored(records_factory):
    df = records_factory([dict(rater=f"r{i}", ratee="A", value=80) for i in range(3)]
                         + [dict(rater="z", ratee="A", value=0, revoked=1)])
    row = score(df, _cfg()).set_index("ratee").loc["A"]
    assert row["n_raw"] == 3 and row["robust_score"] == 0.8


def test_clusters_collapse_to_one_vote(records_factory):
    df = records_factory([dict(rater="h1", ratee="A", value=80), dict(rater="h2", ratee="A", value=80),
                          dict(rater="h3", ratee="A", value=80)]
                         + [dict(rater=f"s{i}", ratee="A", value=0) for i in range(20)])
    clusters = {f"s{i}": "s0" for i in range(20)} | {"h1": "h1", "h2": "h2", "h3": "h3"}
    row = score(df, _cfg(), clusters=clusters).set_index("ratee").loc["A"]
    assert row["robust_score"] == 0.8 and row["n_clusters"] == 4 and row["sybil_flag"] == 1
    assert row["naive_mean"] < 0.2


def test_zero_evidence_ratio_and_naive_mean(records_factory):
    df = records_factory([dict(rater="a", ratee="A", value=100, evidence_level=3),
                          dict(rater="b", ratee="A", value=0), dict(rater="c", ratee="A", value=0)])
    row = score(df, _cfg()).set_index("ratee").loc["A"]
    assert math.isclose(row["zero_evidence_ratio"], 2 / 3) and math.isclose(row["naive_mean"], 1 / 3)


def test_tags_combined_by_cluster_weighted_median(records_factory):
    # tag q: 3 clusters at 0.9; tag u: 3 clusters at 0.1 -> weighted median of {0.9 w3, 0.1 w3} = 0.1 (lower median)
    rows = [dict(rater=f"q{i}", ratee="A", value=90, tag="q") for i in range(3)]
    rows += [dict(rater=f"u{i}", ratee="A", value=10, tag="u") for i in range(3)]
    row = score(records_factory(rows), _cfg()).set_index("ratee").loc["A"]
    assert row["robust_score"] == 0.1 and row["n_clusters"] == 6


def test_empty_records_returns_empty_result_with_columns():
    df = pd.DataFrame(columns=RECORD_COLUMNS)
    out = score(df, _cfg())
    assert list(out.columns) == RESULT_COLUMNS
    assert len(out) == 0


def test_all_revoked_ratee_disappears(records_factory):
    df = records_factory([dict(rater="a", ratee="A", value=80, revoked=1),
                          dict(rater="b", ratee="B", value=80)])
    out = score(df, _cfg())
    assert "A" not in set(out["ratee"])
    assert "B" in set(out["ratee"])


def test_cluster_dict_unknown_rater_falls_back_to_itself(records_factory):
    df = records_factory([dict(rater=f"r{i}", ratee="A", value=80) for i in range(3)])
    # clusters dict present but doesn't mention any of these raters
    clusters = {"someone_else": "cluster0"}
    out = score(df, _cfg(), clusters=clusters).set_index("ratee").loc["A"]
    assert out["n_clusters"] == 3


def test_deterministic_across_runs(records_factory):
    df = records_factory([dict(rater=f"r{i}", ratee=f"A{i % 5}", value=(i * 7) % 100) for i in range(40)])
    cfg = _cfg()
    out1 = score(df, cfg)
    out2 = score(df, cfg)
    pd.testing.assert_frame_equal(out1.reset_index(drop=True), out2.reset_index(drop=True))


def test_ci_brackets_score_and_within_unit_interval(records_factory):
    values = [10, 20, 35, 50, 60, 75, 88, 95]
    rows = [dict(rater=f"c{i}", ratee="A", value=v) for i, v in enumerate(values)]
    row = score(records_factory(rows), _cfg()).set_index("ratee").loc["A"]
    assert row["ci_low"] <= row["robust_score"] <= row["ci_high"]
    assert 0 <= row["ci_low"] <= 1 and 0 <= row["ci_high"] <= 1


def test_bootstrap_zero_gives_degenerate_ci(records_factory):
    rows = [dict(rater=f"c{i}", ratee="A", value=v) for i, v in enumerate([10, 40, 70, 90])]
    row = score(records_factory(rows), _cfg(bootstrap_n=0)).set_index("ratee").loc["A"]
    assert row["ci_low"] == row["ci_high"] == row["robust_score"]


def test_input_not_mutated(records_factory):
    df = records_factory([dict(rater=f"r{i}", ratee="A", value=80) for i in range(3)])
    before = df.copy(deep=True)
    score(df, _cfg())
    pd.testing.assert_frame_equal(df, before)


def test_result_dtypes(records_factory):
    df = records_factory([dict(rater=f"r{i}", ratee="A", value=80) for i in range(3)]
                         + [dict(rater="x", ratee="B", value=10)])
    out = score(df, _cfg())
    assert pd.api.types.is_integer_dtype(out["n_clusters"])
    assert pd.api.types.is_integer_dtype(out["n_raw"])
    assert pd.api.types.is_integer_dtype(out["sybil_flag"])
    assert pd.api.types.is_integer_dtype(out["insufficient"])
    assert pd.api.types.is_float_dtype(out["robust_score"])


def test_performance_smoke(records_factory):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(2000):
        ratee = f"ratee{i}"
        for j in range(5):
            rows.append(dict(rater=f"r{i}_{j}", ratee=ratee, value=int(rng.integers(0, 101))))
    df = records_factory(rows)
    start = time.perf_counter()
    out = score(df, _cfg(bootstrap_n=50))
    elapsed = time.perf_counter() - start
    assert len(out) == 2000
    assert elapsed < 20.0
