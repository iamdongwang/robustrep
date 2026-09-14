import time

import pandas as pd
import pytest

from robustrep.config import Config
from robustrep.sybil import RaterProfile, _candidate_pairs, _jaccard, cluster_raters, profiles_from_records
from robustrep.schema import validate_records

H = 3600


def P(r, ts, funder, ratees):
    return RaterProfile(rater=r, first_seen_ts=ts, funder=funder, ratees=frozenset(ratees))


def test_same_funder_and_time_window_cluster():
    c = cluster_raters([P("a", 0, "F", {"1"}), P("b", H, "F", {"2"})], Config())
    assert c["a"] == c["b"]


def test_same_funder_only_does_not_cluster():
    c = cluster_raters([P("a", 0, "F", {"1"}), P("b", 10 * 24 * H, "F", {"2"})], Config())
    assert c["a"] != c["b"]


def test_time_and_jaccard_cluster_without_funder():
    c = cluster_raters([P("a", 0, None, {"1", "2"}), P("b", H, None, {"1", "2"})], Config())
    assert c["a"] == c["b"]


def test_jaccard_below_threshold_does_not_cluster():
    c = cluster_raters([P("a", 0, None, {"1", "2", "3", "4", "5"}), P("b", H, None, {"1"})], Config())
    assert c["a"] != c["b"]


def test_transitive_union():
    c = cluster_raters([P("a", 0, "F", {"1"}), P("b", H, "F", {"9"}), P("c", 2 * H, None, {"9"})], Config())
    # a~b via funder+time; b~c via time+jaccard(1.0); so a,b,c share a cluster
    assert c["a"] == c["b"] == c["c"]


def test_profiles_from_records_uses_min_ts_and_ratee_sets(records_factory):
    df = validate_records(records_factory([
        dict(rater="a", ratee="1", value=1, ts=50), dict(rater="a", ratee="2", value=1, ts=10)]))
    profs = {p.rater: p for p in profiles_from_records(df)}
    assert profs["a"].first_seen_ts == 10 and profs["a"].ratees == frozenset({"1", "2"})
    assert profs["a"].funder is None


def test_profiles_from_records_takes_meta(records_factory):
    df = validate_records(records_factory([dict(rater="a", ratee="1", value=1, ts=50)]))
    meta = pd.DataFrame([dict(rater="a", first_seen_ts=5, funder="F")])
    p = profiles_from_records(df, meta)[0]
    assert p.first_seen_ts == 5 and p.funder == "F"


def test_cluster_ids_are_lexicographic_min():
    c = cluster_raters([P("z", 0, "F", {"1"}), P("m", H, "F", {"2"}), P("a", 2 * H, "F", {"3"})], Config())
    assert c["z"] == c["m"] == c["a"] == "a"


def test_empty_profiles():
    assert cluster_raters([], Config()) == {}


def test_singleton_profile_maps_to_itself():
    c = cluster_raters([P("solo", 0, None, {"1"})], Config())
    assert c == {"solo": "solo"}


def test_funder_case_insensitive():
    c = cluster_raters([P("a", 0, "0xAB", {"1"}), P("b", H, "0xab", {"2"})], Config())
    assert c["a"] == c["b"]


def test_large_group_blocking_skips_pairs():
    profiles = [P("a", 0, None, {"1"}), P("b", H, None, {"1"}), P("c", 2 * H, None, {"1"})]
    c_small = cluster_raters(profiles, Config(sybil_max_group=2))
    assert c_small["a"] != c_small["b"] and c_small["b"] != c_small["c"]

    c_default = cluster_raters(profiles, Config())
    assert c_default["a"] == c_default["b"] == c_default["c"]


def test_shared_ratee_duplicate_pairs_are_harmless():
    profiles = [
        P("a", 0, None, {"1", "2", "3"}),
        P("b", 0, None, {"1", "2", "3"}),
    ]
    pairs = list(_candidate_pairs(profiles, Config()))
    assert len(pairs) >= 1  # duplicates across shared ratees are allowed, not de-duplicated
    c = cluster_raters(profiles, Config())
    assert c["a"] == c["b"]


def test_profiles_from_records_meta_ignores_unknown_raters(records_factory):
    df = validate_records(records_factory([dict(rater="a", ratee="1", value=1, ts=50)]))
    meta = pd.DataFrame([
        dict(rater="a", first_seen_ts=5, funder="F"),
        dict(rater="ghost", first_seen_ts=1, funder="G"),
    ])
    profs = profiles_from_records(df, meta)
    assert [p.rater for p in profs] == ["a"]


def test_profiles_from_records_meta_null_fields_fall_back(records_factory):
    df = validate_records(records_factory([dict(rater="a", ratee="1", value=1, ts=50)]))
    meta = pd.DataFrame([dict(rater="a", first_seen_ts=float("nan"), funder=float("nan"))])
    p = profiles_from_records(df, meta)[0]
    assert p.first_seen_ts == 50 and p.funder is None


def test_duplicate_rater_in_profiles_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        cluster_raters([P("a", 0, None, {"1"}), P("a", 1, None, {"2"})], Config())


def test_performance_5000_raters_50_ratees():
    profiles = [P(f"r{i}", i, None, {str(i % 50)}) for i in range(5000)]
    start = time.perf_counter()
    c = cluster_raters(profiles, Config())
    elapsed = time.perf_counter() - start
    assert len(c) == 5000
    assert elapsed < 5.0


def test_large_single_funder_group_is_bounded():
    # 20,000 raters, one shared funder, distinct ratees, spread over ~2 years:
    # must not blow up into O(n^2) candidate pairs.
    n = 20_000
    span_s = 2 * 365 * 24 * H
    profiles = [P(f"r{i}", int(i * span_s / n), "F", {f"ratee{i}"}) for i in range(n)]
    start = time.perf_counter()
    c = cluster_raters(profiles, Config())
    elapsed = time.perf_counter() - start
    assert len(c) == n
    assert elapsed < 3.0


def test_sybil_farm_same_funder_within_window_one_cluster():
    n = 5000
    profiles = [P(f"s{i}", i, "F", {f"ratee{i}"}) for i in range(n)]  # all within the default 1-day window
    start = time.perf_counter()
    c = cluster_raters(profiles, Config())
    elapsed = time.perf_counter() - start
    assert len(set(c.values())) == 1
    assert elapsed < 3.0


def test_funder_pairs_outside_window_still_cluster_via_shared_ratee():
    c = cluster_raters([
        P("a", 0, "F", {"1", "2"}),
        P("b", 10 * 24 * H, "F", {"1", "2"}),
    ], Config())
    assert c["a"] == c["b"]


def test_ratee_group_out_of_window_pairs_skipped():
    c = cluster_raters([
        P("a", 0, None, {"1"}),
        P("b", 2 * 24 * H, None, {"1"}),
        P("c", 4 * 24 * H, None, {"1"}),
    ], Config())
    assert len({c["a"], c["b"], c["c"]}) == 3


def test_pair_budget_exceeded_raises():
    profiles = [P(f"r{i}", 0, None, {"1"}) for i in range(4)]  # C(4, 2) = 6 candidate pairs
    with pytest.raises(ValueError, match="sybil_max_pairs"):
        cluster_raters(profiles, Config(sybil_max_pairs=3))


def test_jaccard_both_empty_is_zero():
    assert _jaccard(frozenset(), frozenset()) == 0.0


def test_meta_duplicate_rater_rejected(records_factory):
    df = validate_records(records_factory([dict(rater="a", ratee="1", value=1, ts=1)]))
    meta = pd.DataFrame([
        dict(rater="a", first_seen_ts=1, funder="F"),
        dict(rater="a", first_seen_ts=2, funder="G"),
    ])
    with pytest.raises(ValueError, match="duplicate"):
        profiles_from_records(df, meta)


def test_meta_missing_columns_rejected(records_factory):
    df = validate_records(records_factory([dict(rater="a", ratee="1", value=1, ts=1)]))
    meta = pd.DataFrame([dict(rater="a", first_seen_ts=1)])
    with pytest.raises(ValueError, match="missing columns"):
        profiles_from_records(df, meta)


def test_meta_bad_first_seen_rejected(records_factory):
    df = validate_records(records_factory([dict(rater="a", ratee="1", value=1, ts=1)]))
    bad_type = pd.DataFrame([dict(rater="a", first_seen_ts="oops", funder="F")])
    with pytest.raises(ValueError, match="first_seen_ts"):
        profiles_from_records(df, bad_type)

    negative = pd.DataFrame([dict(rater="a", first_seen_ts=-5, funder="F")])
    with pytest.raises(ValueError):
        profiles_from_records(df, negative)
