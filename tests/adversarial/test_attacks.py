"""Known-answer attacks. Naive mean must break; robust score must hold.

Every scenario here is evidence for the report: an attack that visibly moves
`naive_mean` (the unweighted arithmetic-mean baseline) while `robust_score`
(evidence-weighted, sybil-collapsed, weighted-median) holds at the honest
value. None of these tests may require changing library code to pass -- a
failure here means the pipeline's guarantees don't hold and must be reported
as BLOCKED, not patched around.

All scenarios use `Config(bootstrap_n=0)`: they check point estimates only,
never `ci_low`/`ci_high`, so no confidence-interval claims are made here.
"""
import math

import numpy as np
import pandas as pd

from robustrep import Config, score
from robustrep.sybil import cluster_raters, profiles_from_records
from robustrep.report.adversarial import (
    CFG,
    DAY,
    build,
    farm,
    farmless_attackers,
    funder_two_farms,
    honest,
    meta_distinct,
    meta_for,
    run,
    scenario_c_rows,
    scenario_table,
    sybils,
)


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


def test_scenario_c_flat_weights_reduce_to_lower_median():
    # 2 raters at 0.8 with evidence_level 3, 3 raters at 0.2 with evidence_level 0; distinct
    # funders/wide-apart timestamps (no clusters are computed for this test anyway -- no
    # `clusters` is passed to `score`, so every rater is trivially its own cluster).
    # Default weights (0.1,0.3,0.7,1.0): mass(0.8)=2*1.0=2.0 vs mass(0.2)=3*0.1=0.3 -> 0.8 wins.
    # Flat weights (1,1,1,1): mass(0.8)=2.0 vs mass(0.2)=3.0 -> 0.2 wins (lower weighted median).
    rows = ([dict(rater=f"c_hi{i}", ratee="A", value=80, ts=i * 30 * DAY, evidence_level=3) for i in range(2)]
            + [dict(rater=f"c_lo{i}", ratee="A", value=20, ts=1000 * DAY + i * 30 * DAY, evidence_level=0)
               for i in range(3)])
    df = build(rows)
    assert score(df, CFG).set_index("ratee").loc["A", "robust_score"] == 0.8
    flat = Config(bootstrap_n=0, evidence_weights=(1, 1, 1, 1))
    assert score(df, flat).set_index("ratee").loc["A", "robust_score"] == 0.2


def test_scenario_c2_even_count_lower_median_convention():
    # 4 votes at 0.8 + 4 votes at 0.2, flat weights (equal mass either side: 4.0 vs 4.0).
    # `robustrep.aggregate.weighted_median` is documented as the LOWER weighted median: for
    # an even count/tied mass it picks the smaller of the two middle values, NOT their
    # average. np.median (mean-of-two-middles) would give 0.5; the pipeline gives 0.2.
    rows = ([dict(rater=f"e_hi{i}", ratee="A", value=80, ts=i * 30 * DAY, evidence_level=0) for i in range(4)]
            + [dict(rater=f"e_lo{i}", ratee="A", value=20, ts=1000 * DAY + i * 30 * DAY, evidence_level=0)
               for i in range(4)])
    flat = Config(bootstrap_n=0, evidence_weights=(1, 1, 1, 1))
    df = build(rows)
    out = score(df, flat).set_index("ratee").loc["A"]
    assert np.median([0.8] * 4 + [0.2] * 4) == 0.5
    assert out["robust_score"] == 0.2


def test_scenario_c_value_poisoning_does_not_move_an_honest_ratee():
    # CONFIRMED finding C2. `value` is an attacker-chosen int128 and normalization
    # groups by (tag, scale) across the WHOLE dataset, so records aimed at one ratee
    # decide how every other ratee under that tag is scored. Two throwaway records of
    # +/-2**127 on a throwaway ratee used to drag the honest group into a min-max
    # fallback and flatten every honest score to exactly 0.5. They are now inside the
    # group's outlier tolerance, so they are clipped into the percent range and the
    # honest ratee's row is bit-identical to the clean run's.
    clean_rows, poison_rows = scenario_c_rows()
    clean = score(build(clean_rows), CFG).set_index("ratee").loc["A"]
    poisoned_all = score(build(clean_rows + poison_rows), CFG).set_index("ratee")
    pd.testing.assert_series_equal(clean, poisoned_all.loc["A"])
    # the attacker's own throwaway ratee is not scorable (2 clusters < min_clusters)
    assert poisoned_all.loc["Z", "insufficient"] == 1


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

    f1 = farm("f1_", 25, 100, 100 * DAY)
    f2 = farm("f2_", 25, 100, 300 * DAY)
    rows = honest_rows + f1 + f2

    raters = sorted({r["rater"] for r in rows})
    meta = pd.DataFrame([dict(rater=r, first_seen_ts=min(x["ts"] for x in rows if x["rater"] == r),
                              funder=funder_two_farms(r)) for r in raters])
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
    # By weight: honest mass = 3.0, attacker mass = 0.1k. The weighted median flips to 0.0
    # once attacker mass reaches half the total weight: 0.1k >= (3.0 + 0.1k) / 2 -> k >= 30.
    # 30 is the exact tie (3.0 vs 3.0); pins the lower-median convention and `_TIE_EPS`
    # scale-invariance -- without the epsilon a rescaled weight vector gives 31.
    def honest_l3(n):
        return [dict(rater=f"hl{i}", ratee="A", value=90, ts=i * 7 * DAY, evidence_level=3) for i in range(n)]

    def attackers(n, ts0):
        return [dict(rater=f"atk{i}", ratee="A", value=0, ts=ts0 + i * 3 * DAY, evidence_level=0)
                for i in range(n)]

    scores = {}
    for k in range(25, 36):
        rows = honest_l3(3) + attackers(k, ts0=1000 * DAY)
        df = build(rows)
        clusters = cluster_raters(profiles_from_records(df, meta_distinct(rows)), CFG)
        assert len(set(clusters.values())) == 3 + k  # sanity: attackers never cluster
        row = score(df, CFG, clusters=clusters).set_index("ratee").loc["A"]
        scores[k] = row["robust_score"]

    assert scores[25] == 0.9
    assert scores[35] < 0.9
    flips = [k for k, s in scores.items() if s < 0.9]
    assert min(flips) == 30


def test_scenario_g2_boost_direction():
    # Mirror of scenario G with attackers boosting (value 1.0, the higher side) instead of
    # smearing. At the same exact tie (k=30: honest mass 3.0 vs attacker mass 3.0), the lower
    # weighted-median convention picks the LOWER of the two tied values -- here that's the
    # honest 0.9, not the attacker's 1.0 -- so the tie favors the defender: robust_score stays
    # 0.9 at k=30 and only flips to 1.0 once the attacker mass strictly exceeds half, at k=31.
    def honest_l3(n):
        return [dict(rater=f"hl{i}", ratee="A", value=90, ts=i * 7 * DAY, evidence_level=3) for i in range(n)]

    def attackers_boost(n, ts0):
        return [dict(rater=f"atk{i}", ratee="A", value=100, ts=ts0 + i * 3 * DAY, evidence_level=0)
                for i in range(n)]

    def score_at(k):
        rows = honest_l3(3) + attackers_boost(k, ts0=1000 * DAY)
        df = build(rows)
        clusters = cluster_raters(profiles_from_records(df, meta_distinct(rows)), CFG)
        return score(df, CFG, clusters=clusters).set_index("ratee").loc["A", "robust_score"]

    assert score_at(30) == 0.9
    assert score_at(31) == 1.0


def test_scenario_h_evasive_attacker_defeats_sybil_signals_but_not_the_weighted_median():
    # The key limitation attack: k attackers, each on a DISTINCT funder, registered > 24h
    # apart (2 days), each ALSO rating a decoy ratee from a pool of 6 ("D1".."D6"). A pair
    # sharing the same decoy DOES reach Jaccard 1.0 (identical ratee sets {"A", "D_x"}) --
    # but it still fails to cluster: that pair is > 24h apart in first_seen_ts (decoys repeat
    # every 6 attackers, i.e. a >= 12-day gap) and each attacker has its own distinct funder,
    # so only 1 of the 3 signals (Jaccard) ever fires for it, short of the 2-of-3 threshold.
    # Every attacker evades sybil clustering entirely: cluster_raters gives it its own cluster.
    def clusters_and_row(k):
        rows = honest("A", 5) + farmless_attackers(k, start_ts=1000 * DAY)
        df = build(rows)
        clusters = cluster_raters(profiles_from_records(df, meta_distinct(rows)), CFG)
        row = score(df, CFG, clusters=clusters).set_index("ratee").loc["A"]
        return clusters, row

    # (a) no merging among attackers (each its own singleton cluster) at a representative k.
    clusters, _ = clusters_and_row(30)
    atk_clusters = {r for r, c in clusters.items() if r.startswith("atk")}
    assert len({clusters[r] for r in atk_clusters}) == len(atk_clusters) == 30

    # (b) honest mass 5*0.7=3.5 comfortably beats k*0.1 attacker mass at k=20 and k=30.
    for k in (20, 30):
        _, row = clusters_and_row(k)
        assert row["robust_score"] == 0.8
        assert row["sybil_flag"] == 0   # every attacker is its own cluster: no single cluster dominates

    # (c) break-even: honest mass 3.5 vs k*0.1 attacker mass flips once k*0.1 >= (3.5+k*0.1)/2,
    # i.e. k >= 35. 35 is again an exact tie (3.5 vs 3.5), and the lower-median convention
    # resolves exact ties to the lower (here: attacker) value -- so, as with scenario G, the
    # flip happens exactly AT the tie, at k=35, not k=36.
    robust_by_k = {}
    for k in range(30, 45):
        _, row = clusters_and_row(k)
        robust_by_k[k] = row["robust_score"]
        assert row["zero_evidence_ratio"] >= 0.8  # (d) see comment below
    flips = [k for k, s in robust_by_k.items() if s < 0.8]
    assert min(flips) == 35

    # (d) zero_evidence_ratio (fraction of raw A-votes at evidence_level 0) stays >= 0.8 at
    # every k tested above (k/(5+k) >= 0.8 once k >= 20) even while sybil_flag sits at 0 and
    # robust_score still reads the honest value. This is the signal that survives evasion --
    # sybil_flag is blind to a farm that never clusters, but the evidence-mass ratio is not.


def test_scenario_i_insufficient_clusters_refuses_to_score():
    # 1 honest rater + one 50-rater same-funder/same-hour farm: 2 clusters total, below
    # min_clusters=3 -> the pipeline refuses to produce a robust_score (NaN) rather than
    # publish an unreliable number. Refusing to score is the outcome an attacker most wants
    # to avoid -- it denies the very manipulation they were trying to buy.
    def farm50(ts0):
        return [dict(rater=f"far{i}", ratee="A", value=100, ts=ts0 + i * 30, evidence_level=0) for i in range(50)]

    def meta_farm(rows):
        raters = sorted({r["rater"] for r in rows})
        return pd.DataFrame([dict(rater=r, first_seen_ts=min(x["ts"] for x in rows if x["rater"] == r),
                                  funder=("FARM" if r.startswith("far") else f"own-{r}")) for r in raters])

    rows_2h = honest("A", 2) + farm50(500 * DAY)
    df = build(rows_2h)
    clusters = cluster_raters(profiles_from_records(df, meta_farm(rows_2h)), CFG)
    row = score(df, CFG, clusters=clusters).set_index("ratee").loc["A"]
    assert row["n_clusters"] == 3
    assert row["insufficient"] == 0
    assert row["robust_score"] == 0.8

    rows_1h = honest("A", 1) + farm50(500 * DAY)
    df = build(rows_1h)
    clusters = cluster_raters(profiles_from_records(df, meta_farm(rows_1h)), CFG)
    row = score(df, CFG, clusters=clusters).set_index("ratee").loc["A"]
    assert row["n_clusters"] == 2
    assert row["insufficient"] == 1
    assert math.isnan(row["robust_score"])


def test_report_numbers_table():
    """The naive-vs-robust table (robustrep.report.adversarial.scenario_table) the report
    cites for scenarios A, B, D, E, F, H(k=20), plus the measured break-even rows."""
    table = scenario_table().set_index("scenario")
    assert list(table.index) == [
        "A_boosting", "B_smearing", "C_value_poisoning", "D_evidence_free_flood",
        "E_fresh_tag", "F_split_funders", "H_evasive_k20",
        "G_smear_breakeven", "G2_boost_breakeven", "H_evasive_breakeven",
    ]
    # C's 0.5 is the lower weighted median of the 20 honest percent scores
    # (0.00 .. 1.00 in steps of 0.05), unmoved by the two +/-2**127 records.
    assert table.loc[
        ["A_boosting", "B_smearing", "C_value_poisoning", "D_evidence_free_flood",
         "E_fresh_tag", "F_split_funders", "H_evasive_k20"],
        "robust_score"].tolist() == [0.8, 0.8, 0.5, 0.8, 0.9, 0.8, 0.8]
    assert math.isclose(table.loc["A_boosting", "naive_mean"], 54 / 55, rel_tol=1e-3)
    assert math.isclose(table.loc["B_smearing", "naive_mean"], 4 / 55, rel_tol=1e-3)
    assert math.isclose(table.loc["D_evidence_free_flood", "naive_mean"], 0.16, rel_tol=1e-3)
    assert math.isclose(table.loc["E_fresh_tag", "naive_mean"], 2.7 / 7, rel_tol=1e-3)
    assert table.loc["F_split_funders", "n_clusters"] == 7
    assert table.loc["F_split_funders", "sybil_flag"] == 0
    assert table.loc["H_evasive_k20", "sybil_flag"] == 0
    assert table.loc["H_evasive_k20", "zero_evidence_ratio"] >= 0.8

    # Measured attacker break-even points (scanned by k in robustrep.report.adversarial,
    # never hand-typed): smear succeeds exactly at the mass tie, boost needs one more.
    assert table.loc["G_smear_breakeven", "break_even_k"] == 30
    assert table.loc["G2_boost_breakeven", "break_even_k"] == 31
    assert table.loc["H_evasive_breakeven", "break_even_k"] == 35
    assert table.loc["H_evasive_breakeven", "break_even_k_boost"] == 36
    assert table.loc[["G_smear_breakeven", "G2_boost_breakeven", "H_evasive_breakeven"], "sybil_flag"].eq(0).all()
