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
    REJECTED,
    RETRY,
    RETRY_AFTER_CAP,
    ProfilePlanError,
    capped_json,
    fetch_with_retries,
    funder_address,
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


# --- L1: funder_address --------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("0x" + "a" * 40, "0x" + "a" * 40),
    ("0x" + "A" * 40, "0x" + "a" * 40),
    ("0xAbCdEf0123456789aBcDeF0123456789AbCdEf01", "0xabcdef0123456789abcdef0123456789abcdef01"),
])
def test_funder_address_accepts_a_40_hex_address_lowercased(value, expected, caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.profile_http"):
        assert funder_address(value, "0xabc") == expected
    assert caplog.records == []


@pytest.mark.parametrize("value", [
    None, 7, 0, 1.5, True, {"hash": "0x" + "a" * 40}, ["0x" + "a" * 40], b"0x" + b"a" * 40,
    "", "0x", "0x" + "Z" * 40, "0x" + "a" * 39, "0x" + "a" * 41, "a" * 40, "0X" + "a" * 40,
    " 0x" + "a" * 40, "0x" + "a" * 40 + "\n", "0x" + "a" * 40 + "; drop table raters",
])
def test_funder_address_refuses_anything_else_with_one_warning(value, caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.profile_http"):
        assert funder_address(value, "0xabc") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                and r.name == "robustrep.sources.profile_http"]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "0xabc" in msg and "funder" in msg
    assert repr(value)[:20] in msg or str(value)[:20] in msg


def test_funder_address_bounds_the_logged_value(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.profile_http"):
        assert funder_address("0x" + "a" * 100_000, "0xabc") is None
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


def test_raise_exhausted_body_kind_names_a_body_level_error_not_a_status():
    # HTTP 200 every time, but the body said "rate limited" every time: that is
    # not an "HTTP error (status n/a)", which would send an operator looking at
    # the transport.
    with pytest.raises(RuntimeError) as exc:
        raise_exhausted("Label", "0xa", "body", None, 4)
    assert str(exc.value) == "Label body-level error for 0xa after 4 attempts"
    assert not isinstance(exc.value, ValueError)


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


# --- fetch_with_retries ----------------------------------------------------------------


class _StreamedResp:
    """A streamed response as ``fetch_with_retries`` sees it: ``raise_for_status``,
    ``raw`` and ``close``. ``error`` (a requests exception) makes
    ``raise_for_status`` raise; the body then must never be read."""

    def __init__(self, body=b'{"ok": 1}', error=None):
        self.error, self.closed = error, False
        self.raw = _Raw(body) if error is None else _Exploding()

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


class _Exploding:
    def read(self, amt, decode_content=True):
        raise AssertionError("body must not be read when raise_for_status() raised")


class _Script:
    """``do_request`` over a list of items: a ``_StreamedResp``, an int status
    (raised through ``raise_for_status`` with a URL-bearing message), a
    ``(status, headers)`` tuple, raw ``bytes`` (a 200 with that body) or an
    exception instance (raised by the request itself)."""

    def __init__(self, items):
        self.items, self.made = list(items), []

    def __call__(self):
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, (int, tuple)):
            status, headers = item if isinstance(item, tuple) else (item, {})
            r = _StreamedResp(error=_http_error(status, headers))
        elif isinstance(item, bytes):
            r = _StreamedResp(body=item)
        else:
            r = item
        self.made.append(r)
        return r


def _run(items, on_response=lambda body: body, *, retries=3, gap=0.5, hook=None):
    script, sleeps = _Script(items), []
    try:
        value = fetch_with_retries("Label", "0xa", script, on_response, retries=retries, gap=gap,
                                   sleep=sleeps.append, on_http_error=hook)
    except BaseException as e:  # noqa: BLE001 - re-raised by the caller's pytest.raises
        e._script, e._sleeps = script, sleeps
        raise
    return value, script, sleeps


def test_fetch_success_first_try_returns_on_response_value_and_pays_close_and_gap():
    value, script, sleeps = _run([b'{"a": 1}'], lambda body: ("parsed", body))
    assert value == ("parsed", {"a": 1})
    assert sleeps == [0.5]
    assert script.made[0].closed is True


def test_fetch_non_dict_body_is_a_value_error_naming_label_and_address():
    with pytest.raises(ValueError) as exc:
        _run([b"[1, 2]"])
    assert str(exc.value) == "Label response is not a JSON object for 0xa"
    assert exc.value._script.made[0].closed and exc.value._sleeps == [0.5]


def test_fetch_transient_5xx_then_success_backs_off_exponentially():
    value, script, sleeps = _run([503, 502, b'{"ok": 1}'])
    assert value == {"ok": 1}
    assert sleeps == [0.5, 1, 0.5, 2, 0.5]  # gap, 2**0, gap, 2**1, gap
    assert all(r.closed for r in script.made)


def test_fetch_backoff_is_capped_at_30s():
    _, _, sleeps = _run([503] * 7 + [b'{"ok": 1}'], retries=8, gap=0.0)
    assert sleeps[1::2] == [1, 2, 4, 8, 16, 30, 30]


def test_fetch_exhausted_http_kind_names_last_status_and_attempts():
    with pytest.raises(RuntimeError) as exc:
        _run([502, 503, 500], retries=3)
    assert str(exc.value) == "Label HTTP error for 0xa after 3 attempts (status 500)"
    assert exc.value._sleeps == [0.5, 1, 0.5, 2, 0.5]  # no backoff after the last attempt


def test_fetch_exhausted_body_kind_after_an_http_failure_does_not_report_the_stale_status():
    with pytest.raises(RuntimeError) as exc:
        _run([503, b'{"busy": 1}'], lambda body: RETRY, retries=2)
    assert str(exc.value) == "Label body-level error for 0xa after 2 attempts"


def test_fetch_exhausted_json_kind_after_an_http_failure_is_a_value_error():
    with pytest.raises(ValueError) as exc:
        _run([503, b"<html>"], retries=2)
    assert str(exc.value) == "non-JSON Label response for 0xa"
    assert "html" not in str(exc.value)


def test_fetch_non_json_then_json_succeeds():
    value, _, _ = _run([b"<html>", b'{"ok": 1}'])
    assert value == {"ok": 1}


def test_fetch_request_exception_without_response_is_transient_and_pays_the_gap():
    err = requests.ConnectionError("boom https://x/v2/SECRET-KEY")
    with pytest.raises(RuntimeError) as exc:
        _run([err, err], retries=2)
    assert str(exc.value) == "Label HTTP error for 0xa after 2 attempts (status n/a)"
    assert "SECRET" not in str(exc.value) and "SECRET" not in repr(exc.value)
    assert exc.value.__context__ is None and exc.value.__cause__ is None
    assert exc.value._sleeps == [0.5, 1, 0.5]  # the gap is paid even with no response to close
    assert exc.value._script.made == []


def test_fetch_non_transient_4xx_without_hook_raises_immediately_without_the_requests_exception():
    with pytest.raises(RuntimeError) as exc:
        _run([404, b'{"ok": 1}'])
    assert str(exc.value) == "Label HTTP 404 for 0xa"
    assert exc.value.__suppress_context__ and exc.value.__cause__ is None
    assert "SECRET" not in repr(exc.value.__context__)
    assert len(exc.value._script.made) == 1 and exc.value._script.made[0].closed
    assert exc.value._sleeps == [0.5]


def test_fetch_hook_value_is_returned_for_a_non_transient_status():
    marker = object()
    seen = []

    def hook(code, r):
        seen.append((code, r))
        return marker

    value, script, sleeps = _run([404], hook=hook)
    assert value is marker
    assert seen == [(404, script.made[0])]
    assert script.made[0].closed and sleeps == [0.5]


def test_fetch_hook_rejected_falls_through_to_the_default_error():
    with pytest.raises(RuntimeError) as exc:
        _run([403], hook=lambda code, r: REJECTED)
    assert str(exc.value) == "Label HTTP 403 for 0xa"
    assert exc.value.__suppress_context__


def test_fetch_hook_retry_treats_the_status_as_transient():
    with pytest.raises(RuntimeError) as exc:
        _run([418, 418], retries=2, hook=lambda code, r: RETRY)
    assert str(exc.value) == "Label HTTP error for 0xa after 2 attempts (status 418)"
    assert exc.value._sleeps == [0.5, 1, 0.5]


def test_fetch_hook_is_not_consulted_for_transient_statuses_or_successes():
    calls = []
    value, _, _ = _run([429, 503, b'{"ok": 1}'], hook=lambda code, r: calls.append(code))
    assert value == {"ok": 1} and calls == []


def test_fetch_hook_exception_propagates_with_close_and_gap_paid_and_no_requests_context():
    class Boom(RuntimeError):
        pass

    def hook(code, r):
        raise Boom("fixed text")

    with pytest.raises(Boom) as exc:
        _run([401, 401], hook=hook)
    assert exc.value.__context__ is None
    assert len(exc.value._script.made) == 1 and exc.value._script.made[0].closed
    assert exc.value._sleeps == [0.5]


def test_fetch_on_response_exception_propagates_with_close_and_gap_paid():
    def parse(body):
        raise ValueError("malformed for 0xa")

    with pytest.raises(ValueError) as exc:
        _run([b'{"a": 1}', b'{"a": 1}'], parse)
    assert str(exc.value) == "malformed for 0xa"
    assert len(exc.value._script.made) == 1 and exc.value._script.made[0].closed
    assert exc.value._sleeps == [0.5]


def test_fetch_do_request_raising_a_non_requests_exception_propagates_and_pays_the_gap():
    with pytest.raises(OSError) as exc:
        _run([OSError("disk")])
    assert exc.value._sleeps == [0.5]


def test_fetch_429_retry_after_is_honored_and_capped():
    value, _, sleeps = _run([(429, {"Retry-After": "7"}), (429, {"Retry-After": "3600"}), b'{"ok": 1}'])
    assert value == {"ok": 1}
    assert sleeps == [0.5, 7.0, 0.5, RETRY_AFTER_CAP, 0.5]


def test_fetch_429_non_numeric_retry_after_falls_back_to_exponential_backoff():
    _, _, sleeps = _run([(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), 429, b'{"ok": 1}'])
    assert sleeps == [0.5, 1, 0.5, 2, 0.5]


def test_fetch_retry_after_on_a_5xx_is_ignored():
    _, _, sleeps = _run([(503, {"Retry-After": "7"}), b'{"ok": 1}'])
    assert sleeps == [0.5, 1, 0.5]


def test_fetch_oversized_body_is_a_runtime_error_not_retried(monkeypatch):
    monkeypatch.setattr(profile_http, "MAX_PROFILE_RESPONSE_BYTES", 16)
    with pytest.raises(RuntimeError) as exc:
        _run([b'{"leak": "' + b"SECRETBODY" * 10 + b'"}'] * 3)
    assert "16 byte limit" in str(exc.value) and "SECRETBODY" not in str(exc.value)
    assert len(exc.value._script.made) == 1 and exc.value._script.made[0].closed


def test_fetch_body_level_retry_then_success():
    outcomes = iter([RETRY, "done"])
    value, _, sleeps = _run([b"{}", b"{}"], lambda body: next(outcomes))
    assert value == "done"
    assert sleeps == [0.5, 1, 0.5]


def test_fetch_returns_none_when_on_response_does():
    value, _, _ = _run([b"{}"], lambda body: None)
    assert value is None


def test_fetch_zero_retries_raises_exhausted_without_a_request():
    with pytest.raises(RuntimeError) as exc:
        _run([b"{}"], retries=0)
    assert str(exc.value) == "Label HTTP error for 0xa after 0 attempts (status n/a)"
    assert exc.value._script.made == []


def test_sentinels_are_distinct_and_not_none():
    assert RETRY is not REJECTED and RETRY is not None and REJECTED is not None
