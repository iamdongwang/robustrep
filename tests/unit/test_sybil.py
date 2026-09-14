import time

import pandas as pd
import pytest

from robustrep.config import Config
from robustrep.sybil import RaterProfile, _candidate_pairs, cluster_raters, profiles_from_records
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


def test_shared_ratee_pairs_only_once_across_ratees():
    profiles = [
        P("a", 0, None, {"1", "2", "3"}),
        P("b", 0, None, {"1", "2", "3"}),
    ]
    pairs = list(_candidate_pairs(profiles, Config()))
    assert pairs.count(("a", "b")) == 1


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
