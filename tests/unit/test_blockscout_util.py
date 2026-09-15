"""Tests for the Blockscout-specific parsing helpers: ISO timestamps (M4) and
the pagination-parameter allowlist (L5)."""
import logging
from datetime import datetime, timezone

from robustrep.sources import blockscout_util as bu


def _epoch(*args):
    return int(datetime(*args, tzinfo=timezone.utc).timestamp())


# --- parse_timestamp ----------------------------------------------------------------


def test_parses_a_zulu_timestamp():
    assert bu.parse_timestamp("2026-02-22T21:16:19Z") == _epoch(2026, 2, 22, 21, 16, 19)


def test_parses_fractional_seconds():
    assert bu.parse_timestamp("2026-02-22T21:16:19.123456Z") == _epoch(2026, 2, 22, 21, 16, 19)


def test_treats_a_naive_timestamp_as_utc():
    # Without an explicit tzinfo, datetime.timestamp() would read the string in
    # whatever local zone the machine happens to be in -- a value that silently
    # differs by hours between a laptop and CI.
    assert bu.parse_timestamp("2026-02-22T21:16:19") == bu.parse_timestamp("2026-02-22T21:16:19Z")


def test_honors_an_explicit_offset():
    assert bu.parse_timestamp("2026-02-22T22:16:19+01:00") == _epoch(2026, 2, 22, 21, 16, 19)


def test_unparseable_timestamp_is_none():
    assert bu.parse_timestamp("garbage") is None
    assert bu.parse_timestamp("") is None


def test_non_string_timestamp_is_none():
    assert bu.parse_timestamp(None) is None
    assert bu.parse_timestamp(12345) is None


# --- safe_next_page_params ----------------------------------------------------------


def test_keeps_ordinary_pagination_keys():
    assert bu.safe_next_page_params({"block_number": 100, "index": 5}, "0xa") == {
        "block_number": 100, "index": 5}


def test_drops_keys_outside_the_allowlist(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.blockscout_util"):
        out = bu.safe_next_page_params(
            {"block_number": 1, "../../etc/passwd": "x", "Uppercase": "y", "a" * 33: 1}, "0xa")
    assert out == {"block_number": 1}
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 3


def test_drops_non_scalar_values():
    out = bu.safe_next_page_params({"index": 5, "nested": {"a": 1}, "listy": [1], "flt": 1.5}, "0xa")
    assert out == {"index": 5}


def test_keeps_bool_and_string_values():
    assert bu.safe_next_page_params({"more": True, "cursor": "abc"}, "0xa") == {
        "more": True, "cursor": "abc"}


def test_drops_an_over_long_value():
    out = bu.safe_next_page_params({"index": "x" * (bu.MAX_VALUE_CHARS + 1), "block_number": 2}, "0xa")
    assert out == {"block_number": 2}


def test_drops_an_over_long_numeric_value():
    out = bu.safe_next_page_params({"index": 10 ** 200, "block_number": 2}, "0xa")
    assert out == {"block_number": 2}


def _keys(n):
    # The allowlist is letters/underscore only, so names are built from letters.
    return {f"k{chr(ord('a') + i)}": i for i in range(n)}


def test_caps_the_number_of_forwarded_keys(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.blockscout_util"):
        out = bu.safe_next_page_params(_keys(20), "0xa")
    assert len(out) == bu.MAX_KEYS
    assert list(out) == list(_keys(bu.MAX_KEYS))  # first N, deterministic
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_non_mapping_params_yield_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.blockscout_util"):
        assert bu.safe_next_page_params(["not", "a", "dict"], "0xa") == {}
        assert bu.safe_next_page_params("cursor", "0xa") == {}
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_a_long_rejected_key_is_not_echoed_whole(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.blockscout_util"):
        assert bu.safe_next_page_params({"X" * 5000: 1}, "0xa") == {}
    assert len(caplog.text) < 1000
