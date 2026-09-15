import logging
import time

import pandas as pd
import pytest

from robustrep.config import Config
from robustrep.sybil import (
    ClusterStats, RaterProfile, _candidate_pairs, _jaccard, _ratee_group_pairs, cluster_raters,
    cluster_raters_with_stats, profiles_from_records,
)
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


def test_cluster_raters_no_longer_raises_on_budget_but_still_on_duplicates():
    # H2: a budget overrun must degrade (under-merge), never abort -- otherwise
    # ~6,000 cheap on-chain transactions permanently break scoring for everyone.
    profiles = [P(f"r{i}", 0, None, {"1"}) for i in range(4)]  # C(4, 2) = 6 candidate pairs
    c = cluster_raters(profiles, Config(sybil_max_pairs=3))
    assert set(c) == {"r0", "r1", "r2", "r3"}
    with pytest.raises(ValueError, match="duplicate"):
        cluster_raters([P("a", 0, None, {"1"}), P("a", 1, None, {"2"})], Config(sybil_max_pairs=3))


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


def test_meta_non_integer_first_seen_rejected(records_factory):
    df = validate_records(records_factory([dict(rater="a", ratee="1", value=1, ts=1)]))
    fractional = pd.DataFrame([dict(rater="a", first_seen_ts=5.9, funder="F")])
    with pytest.raises(ValueError, match="first_seen_ts"):
        profiles_from_records(df, fractional)

    whole_float = pd.DataFrame([dict(rater="a", first_seen_ts=5.0, funder="F")])
    p = profiles_from_records(df, whole_float)[0]
    assert p.first_seen_ts == 5


def test_farm_rating_two_agents_clusters_within_budget():
    # 2000 same-funder raters, all within 1h, each rating the same 2 ratees:
    # the same-funder sub-bucket must not regenerate every pair once per
    # shared ratee, or this blows past the default sybil_max_pairs budget.
    n = 2000
    profiles = [P(f"f{i}", i, "F", {"agent1", "agent2"}) for i in range(n)]
    start = time.perf_counter()
    c = cluster_raters(profiles, Config())
    elapsed = time.perf_counter() - start
    assert len(set(c.values())) == 1
    assert elapsed < 5.0


def test_same_funder_pair_emitted_once_across_shared_ratees():
    profiles = [
        P("a", 0, "F", {"1", "2", "3"}),
        P("b", 10 * 24 * H, "F", {"1", "2", "3"}),  # out of window, same funder, 3 shared ratees
    ]
    pairs = [p for p in _candidate_pairs(profiles, Config()) if set(p) == {"a", "b"}]
    assert pairs == [("a", "b")]


def test_same_funder_subbucket_mixed_window_skips_in_window_pair():
    # a-b and b-c are each in-window (already covered by _funder_group_pairs)
    # but the whole sub-bucket's span (a to c) exceeds the window, so the
    # per-pair in-window check inside the combinatorial loop must still fire
    # individually for a-b and b-c, leaving only the out-of-window a-c pair.
    profiles = [
        P("a", 0, "F", {"1"}),
        P("b", 50_000, "F", {"1"}),
        P("c", 100_000, "F", {"1"}),
    ]
    pairs = list(_ratee_group_pairs("1", profiles, Config(), {"1"}))
    assert pairs == [("a", "c")]


# --- pair budgets degrade instead of aborting (H2) ----------------------------


def _farm(n, ratees, funder=None, base_ts=1000):
    """`n` raters, all in-window of each other, all rating every one of `ratees`."""
    return [P(f"r{i:05d}", base_ts + i, funder, set(ratees)) for i in range(n)]


def test_global_budget_truncates_instead_of_raising():
    cfg = Config(sybil_max_pairs=100, sybil_max_pairs_per_ratee=10 ** 9)
    clusters, stats = cluster_raters_with_stats(_farm(60, ["a", "b", "c"]), cfg)
    assert len(clusters) == 60
    assert stats.truncated is True
    assert stats.pairs_tested == 100


def test_per_ratee_budget_skips_block_and_counts_it():
    cfg = Config(sybil_max_pairs_per_ratee=50, sybil_max_pairs=10 ** 9)
    clusters, stats = cluster_raters_with_stats(_farm(40, ["a"]), cfg)
    assert len(clusters) == 40
    assert stats.ratees_skipped_budget == 1
    assert stats.truncated is False
    assert stats.pairs_tested >= 50


def test_size_skip_is_counted():
    _, stats = cluster_raters_with_stats(_farm(11, ["a"]), Config(sybil_max_group=10))
    assert stats.ratees_skipped_size == 1
    assert stats.ratees_skipped_budget == 0 and stats.truncated is False


def test_stats_are_all_zero_within_budget():
    _, stats = cluster_raters_with_stats(_farm(5, ["a"]), Config())
    assert stats == ClusterStats(pairs_tested=10, ratees_skipped_size=0,
                                 ratees_skipped_budget=0, truncated=False)


def test_global_budget_can_truncate_during_funder_blocking():
    # The funder path runs first and can exhaust the budget on its own, before
    # any ratee block is generated at all.
    cfg = Config(sybil_max_pairs=3, sybil_max_pairs_per_ratee=10 ** 9)
    clusters, stats = cluster_raters_with_stats(_farm(10, ["x"], funder="F"), cfg)
    assert len(clusters) == 10
    assert stats.truncated is True and stats.pairs_tested == 3
    assert stats.ratees_skipped_size == 0 and stats.ratees_skipped_budget == 0


def test_clusters_found_before_truncation_are_kept():
    # The funder path is generated first, so its pair is emitted (and tested)
    # before the ratee farm below exhausts the global budget.
    pair = [P("aaa", 1000, "F", {"x"}), P("aab", 1001, "F", {"y"})]
    cfg = Config(sybil_max_pairs=5, sybil_max_pairs_per_ratee=10 ** 9)
    clusters, stats = cluster_raters_with_stats(pair + _farm(60, ["a", "b", "c"]), cfg)
    assert stats.truncated is True
    assert clusters["aaa"] == clusters["aab"]


def test_per_ratee_budget_logs_one_warning_with_repr(caplog):
    cfg = Config(sybil_max_pairs_per_ratee=5, sybil_max_pairs=10 ** 9)
    with caplog.at_level(logging.WARNING, logger="robustrep.sybil"):
        cluster_raters_with_stats(_farm(20, ["evil ratee"]), cfg)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "'evil ratee'" in caplog.text  # attacker-controlled string logged with %r


def test_global_budget_logs_one_warning_with_counts(caplog):
    cfg = Config(sybil_max_pairs=10, sybil_max_pairs_per_ratee=10 ** 9)
    with caplog.at_level(logging.WARNING, logger="robustrep.sybil"):
        cluster_raters_with_stats(_farm(60, ["a", "b", "c"]), cfg)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "sybil_max_pairs" in warnings[0].getMessage()


def test_cluster_raters_returns_only_the_mapping():
    c = cluster_raters(_farm(4, ["a"]), Config())
    assert isinstance(c, dict) and len(c) == 4
