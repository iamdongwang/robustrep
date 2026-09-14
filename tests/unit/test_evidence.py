import pandas as pd
import pytest

from robustrep.config import Config
from robustrep.evidence import classify, weights_for

TX = "0x" + "ab" * 32


def test_weights_map_levels():
    w = weights_for(pd.Series([0, 1, 2, 3]), Config())
    assert list(w) == [0.1, 0.3, 0.7, 1.0]


def test_level0_when_no_uri():
    assert classify(None, lambda u: "x", lambda h: set(), set()) == 0
    assert classify("", lambda u: "x", lambda h: set(), set()) == 0


def test_level1_when_unfetchable_or_no_hash():
    assert classify("https://x", lambda u: None, lambda h: set(), set()) == 1
    assert classify("https://x", lambda u: "no hash here", lambda h: set(), set()) == 1


def test_level2_when_hash_present_but_unverified():
    assert classify("https://x", lambda u: f"tx {TX}", lambda h: None, {"0xrater"}) == 2
    assert classify("https://x", lambda u: f"tx {TX}", lambda h: {"0xother"}, {"0xrater"}) == 2


def test_level2_when_task_id_key_present():
    assert classify("https://x", lambda u: '{"taskId": "t1"}', lambda h: None, set()) == 2


def test_level3_when_tx_party_matches():
    assert classify("https://x", lambda u: f"tx {TX}", lambda h: {"0xrater", "0xz"}, {"0xrater"}) == 3


def test_weights_for_empty_series():
    w = weights_for(pd.Series([], dtype=int), Config())
    assert isinstance(w, pd.Series)
    assert len(w) == 0
    assert w.dtype == float


def test_weights_use_config_values():
    # evidence_weights must be > 0 (see Config.__post_init__: zero weights
    # make bootstrap resamples degenerate), so level 0 uses a small non-zero weight.
    cfg = Config(evidence_weights=(0.05, 0.5, 0.5, 1))
    w = weights_for(pd.Series([0, 1, 2, 3]), cfg)
    assert list(w) == [0.05, 0.5, 0.5, 1.0]


def test_classify_case_insensitive_parties():
    result = classify(
        "https://x", lambda u: f"tx {TX}", lambda h: {"0xrater"}, {"0xRATER"}
    )
    assert result == 3


def test_classify_stops_at_first_verified_hash():
    tx2 = "0x" + "cd" * 32
    calls = []

    def tx_parties(h):
        calls.append(h)
        return {"0xrater"}

    result = classify(
        "https://x", lambda u: f"tx {TX} and {tx2}", tx_parties, {"0xrater"}
    )
    assert result == 3
    assert len(calls) == 1


def test_classify_hash_lookup_failure_is_level2():
    def tx_parties(h):
        raise RuntimeError("boom")

    result = classify("https://x", lambda u: f"tx {TX}", tx_parties, {"0xrater"})
    assert result == 2


def test_classify_mixed_case_hash_matches():
    tx_upper = "0x" + "AB" * 32
    result = classify(
        "https://x", lambda u: f"tx {tx_upper}", lambda h: None, {"0xrater"}
    )
    assert result == 2


def test_duplicate_hashes_looked_up_once():
    calls = []

    def tx_parties(h):
        calls.append(h)
        return None

    result = classify(
        "https://x", lambda u: f"tx {TX} {TX} {TX}", tx_parties, {"0xrater"}
    )
    assert result == 2
    assert calls == [TX]


def test_tx_parties_with_none_member_does_not_crash():
    result = classify(
        "https://x",
        lambda u: f"tx {TX}",
        lambda h: {"0xrater", None},
        {"0xrater"},
    )
    assert result == 3


def test_parties_with_none_ignored():
    result = classify(
        "https://x",
        lambda u: f"tx {TX}",
        lambda h: {"0xrater"},
        {None, "0xrater"},
    )
    assert result == 3


def test_weights_for_rejects_out_of_domain():
    with pytest.raises(ValueError, match="not in"):
        weights_for(pd.Series([0, 7]), Config())
    with pytest.raises(ValueError, match="null"):
        weights_for(pd.Series([1, None]), Config())


def test_weights_for_preserves_index():
    w = weights_for(pd.Series([0, 3], index=["x", "y"]), Config())
    assert list(w.index) == ["x", "y"]


def test_hash_prefix_of_longer_hex_not_matched():
    text = "0x" + "a" * 65
    result = classify("https://x", lambda u: text, lambda h: set(), set())
    assert result == 1


def test_whitespace_uri_is_level0():
    assert classify("   ", lambda u: "x", lambda h: set(), set()) == 0


def test_weights_for_rejects_non_integer():
    with pytest.raises(ValueError, match="not in"):
        weights_for(pd.Series([3.9]), Config())


def test_verified_non_string_member_does_not_crash():
    result = classify(
        "https://x", lambda u: f"tx {TX}", lambda h: {12345}, {"0xrater"}
    )
    assert result == 2


def test_dedupe_hashes_case_insensitive():
    calls = []

    def tx_parties(h):
        calls.append(h)
        return None

    tx_upper = "0x" + "AB" * 32
    text = f"tx {TX} and {tx_upper}"
    result = classify("https://x", lambda u: text, tx_parties, {"0xrater"})
    assert result == 2
    assert len(calls) == 1
