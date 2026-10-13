"""Direct tests for the hygiene helpers every rater-profile HTTP client shares
(``robustrep.sources.profile_http``). The clients' own test modules exercise
these end to end; this file pins the helpers' contracts on their own."""
import logging

import pytest
import requests

from robustrep.sources import profile_http
from robustrep.sources.profile_http import (
    MAX_PLAUSIBLE_TS,
    MAX_PROFILE_RESPONSE_BYTES,
    RETRY_AFTER_CAP,
    ProfilePlanError,
    capped_json,
    implausible,
    is_transient_http,
    plausible_ts,
    raise_exhausted,
    retry_after_seconds,
    status_code,
)
from robustrep.sources.rater_profile import EtherscanPlanError


def _http_error(status, headers=None):
    err = requests.HTTPError(f"{status} error for url: https://x/v2/SECRET-KEY")
    resp = requests.Response()
    resp.status_code = status
    resp.headers.update(headers or {})
    err.response = resp
    return err


class _Raw:
    def __init__(self, data: bytes):
        self.data = data

    def read(self, amt, decode_content=True):
        chunk, self.data = self.data[:amt], self.data[amt:]
        return chunk


class _Resp:
    def __init__(self, body: bytes):
        self.raw = _Raw(body)


# --- constants -----------------------------------------------------------------------


def test_constants_match_the_values_the_clients_were_reviewed_with():
    assert MAX_PROFILE_RESPONSE_BYTES == 8 * 1024 * 1024
    assert MAX_PLAUSIBLE_TS == 4102444800  # 2100-01-01T00:00:00Z
    assert RETRY_AFTER_CAP == 60.0


# --- ProfilePlanError ----------------------------------------------------------------


def test_profile_plan_error_is_a_runtime_error_and_etherscan_plan_error_is_one():
    assert issubclass(ProfilePlanError, RuntimeError)
    assert issubclass(EtherscanPlanError, ProfilePlanError)


# --- M4: plausibility ----------------------------------------------------------------


@pytest.mark.parametrize("ts", [None, 0, -1, MAX_PLAUSIBLE_TS + 1])
def test_implausible_refuses_none_non_positive_and_far_future(ts):
    assert implausible(ts) is True


@pytest.mark.parametrize("ts", [1, 1_700_000_000, MAX_PLAUSIBLE_TS])
def test_implausible_accepts_a_real_epoch_second(ts):
    assert implausible(ts) is False


def test_plausible_ts_parses_numeric_strings():
    assert plausible_ts("99", "0xa") == 99
    assert plausible_ts(99, "0xa") == 99


@pytest.mark.parametrize("value", ["not-a-number", float("inf"), None, "-5", "0"])
def test_plausible_ts_returns_none_with_one_warning_naming_address_and_value(value, caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.profile_http"):
        assert plausible_ts(value, "0xabc") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                and r.name == "robustrep.sources.profile_http"]
    assert len(warnings) == 1
    assert "0xabc" in warnings[0].getMessage()


def test_warn_bad_ts_bounds_the_logged_value(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.profile_http"):
        profile_http.warn_bad_ts("0xa", "x" * 10_000)
    assert len(caplog.text) < 500


# --- M2: capped_json -----------------------------------------------------------------


def test_capped_json_decodes_a_small_body():
    assert capped_json(_Resp(b'{"ok": 1}'), "0xa", "Label") == {"ok": 1}


def test_capped_json_oversized_body_raises_runtime_error_naming_label_address_and_limit(monkeypatch):
    monkeypatch.setattr(profile_http, "MAX_PROFILE_RESPONSE_BYTES", 16)
    with pytest.raises(RuntimeError) as exc:
        capped_json(_Resp(b'{"leak": "' + b"SECRETBODY" * 100 + b'"}'), "0xa", "Label")
    msg = str(exc.value)
    assert "Label" in msg and "0xa" in msg and "16 byte limit" in msg
    assert "SECRETBODY" not in msg


def test_capped_json_non_json_body_raises_plain_value_error():
    with pytest.raises(ValueError) as exc:
        capped_json(_Resp(b"<html>"), "0xa", "Label")
    assert not isinstance(exc.value, RuntimeError)


# --- retry vocabulary ----------------------------------------------------------------


def test_raise_exhausted_json_kind_is_value_error_naming_label_and_address():
    with pytest.raises(ValueError) as exc:
        raise_exhausted("Label", "0xa", "json", None, 3)
    assert "non-JSON Label response for 0xa" == str(exc.value)


def test_raise_exhausted_http_kind_names_status_and_attempts():
    with pytest.raises(RuntimeError) as exc:
        raise_exhausted("Label", "0xa", "http", 503, 5)
    assert str(exc.value) == "Label HTTP error for 0xa after 5 attempts (status 503)"


def test_raise_exhausted_without_a_status_says_n_a():
    with pytest.raises(RuntimeError) as exc:
        raise_exhausted("Label", "0xa", "http", None, 2)
    assert "(status n/a)" in str(exc.value)


def test_status_code_reads_the_response_status_or_none():
    assert status_code(_http_error(429)) == 429
    assert status_code(requests.ConnectionError("boom https://x/v2/SECRET")) is None


@pytest.mark.parametrize("code,expected", [
    (None, True), (429, True), (500, True), (599, True),
    (400, False), (403, False), (404, False), (200, False), (600, False),
])
def test_is_transient_http(code, expected):
    assert is_transient_http(code) is expected


def test_retry_after_seconds_honors_numeric_header_and_caps_it():
    assert retry_after_seconds(_http_error(429, {"Retry-After": "5"})) == 5.0
    assert retry_after_seconds(_http_error(429, {"Retry-After": "3600"})) == RETRY_AFTER_CAP


def test_retry_after_seconds_none_without_response_header_or_numeric_value():
    assert retry_after_seconds(requests.ConnectionError("no response")) is None
    assert retry_after_seconds(_http_error(429)) is None
    assert retry_after_seconds(_http_error(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None
    assert retry_after_seconds(_http_error(429, {"Retry-After": "-3"})) is None
