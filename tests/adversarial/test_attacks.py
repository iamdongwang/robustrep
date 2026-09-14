"""Known-answer attacks. Naive mean must break; robust score must hold.

Every scenario here is evidence for the report: an attack that visibly moves
`naive_mean` (the unweighted arithmetic-mean baseline) while `robust_score`
(evidence-weighted, sybil-collapsed, weighted-median) holds at the honest
value. None of these tests may require changing library code to pass -- a
failure here means the pipeline's guarantees don't hold and must be reported
as BLOCKED, not patched around.
"""
import numpy as np
import pandas as pd

from robustrep import Config, score
from robustrep.sybil import cluster_raters, profiles_from_records
from robustrep.schema import validate_records

CFG = Config(bootstrap_n=0)
DAY = 86400


def honest(ratee, n, val=80, start_ts=0):
    # n independent raters, spread over months, each with its own funder
    return [dict(rater=f"h{i}", ratee=ratee, value=val, ts=start_ts + i * 7 * DAY, evidence_level=2)
            for i in range(n)]


def sybils(ratee, n, val, ts, funder="F"):
    return [dict(rater=f"s{i}", ratee=ratee, value=val, ts=ts + i * 60, evidence_level=0) for i in range(n)]


def meta_for(rows, sybil_funder="F"):
    raters = sorted({r["rater"] for r in rows})
    return pd.DataFrame([dict(rater=r, first_seen_ts=min(x["ts"] for x in rows if x["rater"] == r),
                              funder=(sybil_funder if r.startswith("s") else f"own-{r}")) for r in raters])


def build(rows):
    return validate_records(pd.DataFrame([{**dict(scale="d0", tag="q", evidence_uri=None, source="t"), **r}
                                          for r in rows]))


def run(rows):
    df = build(rows)
    clusters = cluster_raters(profiles_from_records(df, meta_for(rows)), CFG)
    return score(df, CFG, clusters=clusters).set_index("ratee").loc["A"]


def test_scenario_a_boosting_sybils_move_mean_not_robust():
    rows = honest("A", 5) + sybils("A", 50, val=100, ts=100 * DAY)
    row = run(rows)
    assert row["naive_mean"] > 0.95            # mean is captured
    assert row["robust_score"] == 0.8           # robust holds at honest value
    assert row["sybil_flag"] == 1 and row["n_clusters"] == 6


def test_scenario_b_smearing_sybils_do_not_sink_robust():
    rows = honest("A", 5) + sybils("A", 50, val=0, ts=100 * DAY)
    row = run(rows)
    assert row["naive_mean"] < 0.1
    assert row["robust_score"] == 0.8


def test_scenario_c_equal_evidence_weights_reduce_to_median():
    rows = honest("A", 5, val=80) + honest("A", 2, val=20, start_ts=400 * DAY)
    for r in rows[5:]:
        r["rater"] = r["rater"].replace("h", "g")
    flat = Config(bootstrap_n=0, evidence_weights=(1, 1, 1, 1))
    df = build(rows)
    out = score(df, flat).set_index("ratee").loc["A"]
    assert out["robust_score"] == np.median([0.8] * 5 + [0.2] * 2)


def test_scenario_d_evidence_free_flood_is_downweighted_even_without_clusters():
    # 5 honest with evidence vs 20 evidence-free raters with distinct funders: no cluster merge,
    # but weights 0.7 vs 0.1 keep the weighted median at the honest value.
    rows = honest("A", 5, val=80) + [dict(rater=f"z{i}", ratee="A", value=0, ts=i * 3 * DAY, evidence_level=0)
                                     for i in range(20)]
    df = build(rows)
    out = score(df, CFG).set_index("ratee").loc["A"]
    assert out["naive_mean"] < 0.2 and out["robust_score"] == 0.8


def test_scenario_e_fresh_tag_attack():
    # 3 honest, well-evidenced (level 3, weight 1.0) votes at 0.9 under tag "q" (mass 3.0)
    # vs 4 evidence-free (level 0, weight 0.1) votes at 0.0 stuffed under a brand-new tag
    # "junk" (mass 0.4). Cross-tag weighting by summed evidence mass means "junk" can't
    # outvote "q" no matter how many attacker votes it carries.
    rows = ([dict(rater=f"e{i}", ratee="A", value=90, ts=i * 7 * DAY, evidence_level=3, tag="q")
             for i in range(3)]
            + [dict(rater=f"j{i}", ratee="A", value=0, ts=i * 7 * DAY, evidence_level=0, tag="junk")
               for i in range(4)])
    df = build(rows)
    out = score(df, CFG).set_index("ratee").loc["A"]
    assert out["robust_score"] == 0.9
    assert out["naive_mean"] < 0.5   # the arithmetic mean over both tags is dragged down


def test_scenario_f_sybil_farm_split_across_two_funders():
    # 5 honest + two 25-rater sybil farms on distinct funders "F1"/"F2", each farm's raters
    # first-seen within the same hour (so each farm collapses to its own cluster), but the two
    # farms are ~200 days apart from each other and from the honest raters, so they don't merge
    # into one cluster: n_clusters = 5 honest singletons + 2 farm clusters = 7.
    honest_rows = honest("A", 5)

    def farm(prefix, n, val, ts0):
        return [dict(rater=f"{prefix}{i}", ratee="A", value=val, ts=ts0 + i * 100, evidence_level=0)
                for i in range(n)]

    f1 = farm("f1_", 25, 100, 100 * DAY)
    f2 = farm("f2_", 25, 100, 300 * DAY)
    rows = honest_rows + f1 + f2

    def funder(r):
        if r.startswith("f1_"):
            return "F1"
        if r.startswith("f2_"):
            return "F2"
        return f"own-{r}"

    raters = sorted({r["rater"] for r in rows})
    meta = pd.DataFrame([dict(rater=r, first_seen_ts=min(x["ts"] for x in rows if x["rater"] == r),
                              funder=funder(r)) for r in raters])
    df = build(rows)
    clusters = cluster_raters(profiles_from_records(df, meta), CFG)
    row = score(df, CFG, clusters=clusters).set_index("ratee").loc["A"]
    assert row["n_clusters"] == 7
    assert row["robust_score"] == 0.8
    # Each farm is only 25/55 = 45.5% of raw records: below the 50% single-cluster
    # threshold, so sybil_flag stays 0 even though the ratee is clearly under sybil
    # attack. sybil_flag is a single-cluster-dominance indicator, not a general sybil
    # detector -- splitting an attack across multiple funders/clusters can stay under
    # it while the per-cluster collapse (n_clusters, weighted median) still neutralizes
    # the attack on robust_score. This is a documented limitation, not a bug.
    assert row["sybil_flag"] == 0


def test_scenario_g_attack_breakeven_point():
    # 3 honest, level-3 (weight 1.0) votes at 0.9 (mass 3.0) vs k evidence-free (level 0,
    # weight 0.1) attackers at 0.0, each on a distinct funder and spread 3 days apart (so they
    # never cluster with each other or the honest raters -- see the n_clusters == 3 + k check
    # below). Find the smallest k that flips robust_score below 0.9.
    #
    # By weight: honest mass = 3.0, attacker mass = 0.1k. The weighted median (lower-median
    # convention) flips to 0.0 once the attacker mass reaches half the total weight, i.e.
    # 0.1k >= (3.0 + 0.1k) / 2, i.e. k >= 30. At k = 30 the two sides are EXACTLY tied
    # (0.1*30 == 3.0 == half), and the lower-weighted-median convention (the tie-breaking
    # epsilon in `_weighted_median_pos` nudges the half-weight threshold down) resolves the
    # tie to the lower value -- so the flip happens exactly at k = 30, not k = 31.
    def honest_l3(n):
        return [dict(rater=f"hl{i}", ratee="A", value=90, ts=i * 7 * DAY, evidence_level=3) for i in range(n)]

    def attackers(n, ts0):
        return [dict(rater=f"atk{i}", ratee="A", value=0, ts=ts0 + i * 3 * DAY, evidence_level=0)
                for i in range(n)]

    def meta_distinct(rows):
        raters = sorted({r["rater"] for r in rows})
        return pd.DataFrame([dict(rater=r, first_seen_ts=min(x["ts"] for x in rows if x["rater"] == r),
                                  funder=f"own-{r}") for r in raters])

    scores = {}
    for k in range(25, 36):
        rows = honest_l3(3) + attackers(k, ts0=1000 * DAY)
        df = build(rows)
        clusters = cluster_raters(profiles_from_records(df, meta_distinct(rows)), CFG)
        assert len(set(clusters.values())) == 3 + k  # sanity: attackers never cluster
        row = score(df, CFG, clusters=clusters).set_index("ratee").loc["A"]
        scores[k] = row["robust_score"]

    flips = [k for k, s in scores.items() if s < 0.9]
    first_flip = min(flips)
    assert first_flip >= 30
    assert first_flip == 30


def test_report_numbers_table():
    """The naive-vs-robust table the report cites for scenarios A, B, D, E."""
    a = run(honest("A", 5) + sybils("A", 50, val=100, ts=100 * DAY))
    b = run(honest("A", 5) + sybils("A", 50, val=0, ts=100 * DAY))
    d_rows = honest("A", 5, val=80) + [dict(rater=f"z{i}", ratee="A", value=0, ts=i * 3 * DAY, evidence_level=0)
                                       for i in range(20)]
    d = score(build(d_rows), CFG).set_index("ratee").loc["A"]
    e_rows = ([dict(rater=f"e{i}", ratee="A", value=90, ts=i * 7 * DAY, evidence_level=3, tag="q")
               for i in range(3)]
              + [dict(rater=f"j{i}", ratee="A", value=0, ts=i * 7 * DAY, evidence_level=0, tag="junk")
                 for i in range(4)])
    e = score(build(e_rows), CFG).set_index("ratee").loc["A"]

    table = pd.DataFrame(
        {"naive_mean": [a["naive_mean"], b["naive_mean"], d["naive_mean"], e["naive_mean"]],
         "robust_score": [a["robust_score"], b["robust_score"], d["robust_score"], e["robust_score"]]},
        index=["A_boosting", "B_smearing", "D_evidence_free_flood", "E_fresh_tag"],
    )
    assert table["robust_score"].tolist() == [0.8, 0.8, 0.8, 0.9]
