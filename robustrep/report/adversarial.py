"""Known-answer attack scenarios, shared by the adversarial test suite and the report.

This module is the single source of truth for the scenario builders (row
generators, funder/meta helpers, and the `run`/`build` glue) exercised by
``tests/adversarial/test_attacks.py``: the tests import from here rather than
defining their own copies, so the report's "adversarial evidence" table and
the test suite's assertions can never silently drift apart.

`scenario_table()` reports six representative scenarios for the report
(spec Sec 5.5b): A (boosting sybils), B (smearing sybils), D (evidence-free
flood without clustering), E (fresh-tag attack), F (sybil farm split across
two funders), and H at k=20 (the evasive-attacker limitation case: distinct
funders, >24h spacing, decoy ratees -- clustering never fires, but the
weighted median and evidence weighting still hold the honest value). All
computed with `Config(bootstrap_n=0)`: point estimates only, no CI claims.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config
from ..schema import validate_records
from ..sybil import cluster_raters, profiles_from_records
from .. import score

CFG = Config(bootstrap_n=0)
DAY = 86400


def honest(ratee, n, val=80, start_ts=0):
    """n independent raters, each with its own funder, spread weekly."""
    return [dict(rater=f"h{i}", ratee=ratee, value=val, ts=start_ts + i * 7 * DAY, evidence_level=2)
            for i in range(n)]


def sybils(ratee, n, val, ts):
    """n raters sharing funder "F" (wired via `meta_for`'s sybil_funder), 60s apart."""
    return [dict(rater=f"s{i}", ratee=ratee, value=val, ts=ts + i * 60, evidence_level=0) for i in range(n)]


def meta_for(rows, sybil_funder="F"):
    """Every "s*" rater shares `sybil_funder`; every other rater gets its own distinct funder."""
    raters = sorted({r["rater"] for r in rows})
    return pd.DataFrame([dict(rater=r, first_seen_ts=min(x["ts"] for x in rows if x["rater"] == r),
                              funder=(sybil_funder if r.startswith("s") else f"own-{r}")) for r in raters])


def meta_distinct(rows):
    """Every rater gets its own distinct funder ('own-<rater>') -- guarantees no
    funder-signal clustering regardless of timing/Jaccard."""
    raters = sorted({r["rater"] for r in rows})
    return pd.DataFrame([dict(rater=r, first_seen_ts=min(x["ts"] for x in rows if x["rater"] == r),
                              funder=f"own-{r}") for r in raters])


def build(rows):
    return validate_records(pd.DataFrame([{**dict(scale="d0", tag="q", evidence_uri=None, source="t"), **r}
                                          for r in rows]))


def run(rows):
    """Score scenario `rows` (clustered via `meta_for`'s default sybil funder), for ratee "A"."""
    df = build(rows)
    clusters = cluster_raters(profiles_from_records(df, meta_for(rows)), CFG)
    return score(df, CFG, clusters=clusters).set_index("ratee").loc["A"]


# --- scenario F: sybil farm split across two funders ----------------------------


def farm(prefix, n, val, ts0):
    """n raters named `prefix{i}`, first-seen within the same hour starting at `ts0`."""
    return [dict(rater=f"{prefix}{i}", ratee="A", value=val, ts=ts0 + i * 100, evidence_level=0)
            for i in range(n)]


def funder_two_farms(rater: str) -> str:
    """Funder assignment for scenario F: "f1_*" -> "F1", "f2_*" -> "F2", else its own."""
    if rater.startswith("f1_"):
        return "F1"
    if rater.startswith("f2_"):
        return "F2"
    return f"own-{rater}"


def scenario_f_rows():
    """5 honest + two 25-rater sybil farms on distinct funders, ~200 days apart from
    each other and from the honest raters (so the two farms don't merge into one
    cluster): n_clusters = 5 honest singletons + 2 farm clusters = 7."""
    return honest("A", 5) + farm("f1_", 25, 100, 100 * DAY) + farm("f2_", 25, 100, 300 * DAY)


def scenario_f_meta(rows):
    raters = sorted({r["rater"] for r in rows})
    return pd.DataFrame([dict(rater=r, first_seen_ts=min(x["ts"] for x in rows if x["rater"] == r),
                              funder=funder_two_farms(r)) for r in raters])


# --- scenario H: evasive attacker defeats sybil signals but not the weighted median --


def farmless_attackers(n, start_ts, ratee="A"):
    """k attackers, each on a DISTINCT funder, registered >24h apart (2 days), each
    ALSO rating a decoy ratee from a pool of 6 ("D1".."D6"). A pair sharing the same
    decoy has an IDENTICAL ratee set ({ratee, decoy}) and so a Jaccard of exactly
    1.0 -- but that pair is still >24h apart in first_seen_ts (a multiple of 6
    attackers' worth of 2-day steps) and each attacker has its own distinct funder,
    so it never reaches the 2-of-3 signal threshold (Jaccard alone is only 1
    signal). Every attacker therefore evades clustering entirely: `cluster_raters`
    gives it its own singleton cluster."""
    rows = []
    for i in range(n):
        ts = start_ts + i * 2 * DAY
        rows.append(dict(rater=f"atk{i}", ratee=ratee, value=0, ts=ts, evidence_level=0))
        rows.append(dict(rater=f"atk{i}", ratee=f"D{(i % 6) + 1}", value=0, ts=ts, evidence_level=0))
    return rows


def scenario_h_rows(k, start_ts=1000 * DAY):
    return honest("A", 5) + farmless_attackers(k, start_ts=start_ts)


def scenario_table() -> pd.DataFrame:
    """Naive-vs-robust numbers for scenarios A, B, D, E, F, H(k=20), for the report.

    Side-effect free: computes and returns a DataFrame, never writes a file.
    Columns: scenario, naive_mean, robust_score, n_clusters, sybil_flag,
    zero_evidence_ratio. All computed with `Config(bootstrap_n=0)` -- point
    estimates only, no CI claims are made from this table.
    """
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

    f_rows = scenario_f_rows()
    f_df = build(f_rows)
    f_clusters = cluster_raters(profiles_from_records(f_df, scenario_f_meta(f_rows)), CFG)
    f = score(f_df, CFG, clusters=f_clusters).set_index("ratee").loc["A"]

    h_rows = scenario_h_rows(20)
    h_df = build(h_rows)
    h_clusters = cluster_raters(profiles_from_records(h_df, meta_distinct(h_rows)), CFG)
    h = score(h_df, CFG, clusters=h_clusters).set_index("ratee").loc["A"]

    rows = []
    for name, row in (("A_boosting", a), ("B_smearing", b), ("D_evidence_free_flood", d),
                       ("E_fresh_tag", e), ("F_split_funders", f), ("H_evasive_k20", h)):
        rows.append(dict(scenario=name, naive_mean=row["naive_mean"], robust_score=row["robust_score"],
                          n_clusters=row["n_clusters"], sybil_flag=row["sybil_flag"],
                          zero_evidence_ratio=row["zero_evidence_ratio"]))
    return pd.DataFrame(rows)
