"""Tests for the live-suite's availability guard (``tests/e2e/test_live.py``'s
``rpc_unavailable_reason`` / ``rpc_available``).

The guard decides whether a weekly live run reports "the public endpoint was
down or throttling us" (a *labelled* failure, ``RPC unavailable: ...``) or
"the chain data no longer matches what we recorded" (the raw failure). Neither
is ever a skip -- the owner wants the email. Getting the label wrong misdirects
the reader of that email,
so the classifier is pinned here -- offline, unmarked, part of every run.

Two properties get the most attention, both of them ways the guard could go
quietly wrong:

* markers must not fire on ordinary chain data -- a block number containing
  ``429`` is not a rate limit;
* a failure label must carry *our* label and nothing else -- never a fragment of
  a server-supplied error message, which can echo a keyed request URL into a
  public CI log.

``tests/`` is not a package, so the live module is loaded by path under a name
of its own rather than imported; nothing in it touches the network at import
time.
"""
import importlib.util
from pathlib import Path

import pytest
import requests

from robustrep.sources.rpc import (RpcBatchRateLimitError, RpcBatchStructureError, RpcError,
                                   RpcRangeTooLargeError)
from robustrep.store import Store


def _load_live_module():
    path = Path(__file__).resolve().parents[1] / "e2e" / "test_live.py"
    spec = importlib.util.spec_from_file_location("robustrep_live_guard_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


live = _load_live_module()


def _http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} Server Error: for url: https://rpc.example/v2/opaque-path",
                              response=response)


# --- skippable: the endpoint was unusable -----------------------------------

@pytest.mark.parametrize("exc", [
    requests.ConnectionError("failed to establish a connection"),
    requests.ConnectTimeout("connect timed out"),
    requests.ReadTimeout("read timed out"),
    requests.Timeout("timed out"),
    requests.TooManyRedirects("exceeded 30 redirects"),
    requests.exceptions.ChunkedEncodingError("connection broken mid-body"),
    requests.exceptions.SSLError("certificate verify failed"),
    TimeoutError("socket timed out"),
])
def test_transport_failures_are_skippable(exc):
    reason = live.rpc_unavailable_reason(exc)
    assert reason is not None
    assert reason.startswith("transport failure: ")


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_side_http_errors_are_skippable(status):
    assert live.rpc_unavailable_reason(_http_error(status)) == f"transport failure: HTTP {status}"


def test_http_429_is_classified_as_rate_limiting():
    assert live.rpc_unavailable_reason(_http_error(429)) == "rate limited"


@pytest.mark.parametrize("status", [400, 403, 404, 418])
def test_client_side_http_errors_still_fail(status):
    """A 404/403 is the endpoint answering us, not refusing to: it means the
    request or the URL is wrong, which is a real problem worth failing on."""
    assert live.rpc_unavailable_reason(_http_error(status)) is None


@pytest.mark.parametrize("message", [
    "gave up after 5 attempts: HTTPError (HTTP 429) from https://mainnet.base.org",
    "gave up after 5 attempts: too many requests",
    "batch entry id=3: [-32016] over rate limit",
    "gave up after 5 attempts: [-32005] rate limit exceeded",
])
def test_rpc_errors_naming_throttling_are_skippable(message):
    assert live.rpc_unavailable_reason(RpcError(message)) == "rate limited"


@pytest.mark.parametrize("message, label", [
    ("gave up after 5 attempts: ConnectionError contacting https://mainnet.base.org", "connectionerror"),
    ("gave up after 5 attempts: ReadTimeout contacting https://mainnet.base.org", "readtimeout"),
    ("gave up after 5 attempts: ConnectTimeout contacting https://mainnet.base.org", "connecttimeout"),
    ("gave up after 5 attempts: SSLError contacting https://mainnet.base.org", "sslerror"),
    ("gave up after 5 attempts: HTTPError (HTTP 502) from https://mainnet.base.org", "http 502"),
    ("gave up after 5 attempts: HTTPError (HTTP 503) from https://mainnet.base.org", "http 503"),
])
def test_rpc_errors_naming_transport_failures_are_skippable(message, label):
    assert live.rpc_unavailable_reason(RpcError(message)) == f"transport failure: {label}"


def test_no_usable_endpoint_is_an_availability_failure():
    exc = RpcError("no usable endpoint: 4 of 4 endpoints rejected the minimum window")
    assert live.rpc_unavailable_reason(exc) == "transport failure: no usable endpoint"


def test_range_too_large_leaking_to_tests_is_a_real_failure():
    # LogFetcher must absorb this; if it reaches a test it is a bug, not an outage.
    assert live.rpc_unavailable_reason(RpcRangeTooLargeError("https://h", (50,))) is None


def test_rate_limited_batch_subclass_is_skippable():
    exc = RpcBatchRateLimitError("batch entry id=1: [-32016] over rate limit")
    assert live.rpc_unavailable_reason(exc) == "rate limited"


# --- the reason string leaks nothing ----------------------------------------

# A distinctive canary standing in for whatever an RPC URL might embed. Named
# so as not to trip the repo's own secret scanner -- it is a literal in a test,
# and the assertion below is that it never comes back out.
CANARY = "canary-9f3a-not-a-real-credential"


@pytest.mark.parametrize("exc", [
    requests.ConnectionError(f"HTTPSConnectionPool(host='rpc.example') url: /v2/{CANARY}"),
    RpcError(f"gave up after 5 attempts: ReadTimeout contacting https://rpc.example/v2/{CANARY}"),
    RpcError(f"batch entry id=1: [-32016] over rate limit ({CANARY})"),
    _http_error(503),
])
def test_failure_label_never_carries_server_supplied_text(exc):
    """The reason is our own label only. An exception message is partly
    server-supplied and routinely echoes the full request URL, which may embed
    a provider API key -- none of it may reach a CI log through here."""
    reason = live.rpc_unavailable_reason(exc)
    assert reason is not None
    assert CANARY not in reason
    assert str(exc) not in reason
    assert reason in {"rate limited"} or reason.startswith("transport failure: ")


# --- must still fail --------------------------------------------------------

@pytest.mark.parametrize("message", [
    "gave up after 5 attempts: [-32602] invalid params: block 429000 is out of range",
    "non-retryable error: limited to a 10000 block range starting at 0x429",
    "gave up after 5 attempts: eth_getLogs returned nothing for block 4290429",
])
def test_a_block_number_containing_429_is_not_a_rate_limit(message):
    """Regression: a bare ``"429"`` marker matched block numbers, quietly
    relabelling real failures as rate limits."""
    assert live.rpc_unavailable_reason(RpcError(message)) is None


@pytest.mark.parametrize("exc", [
    AssertionError("assert 41 == 42"),
    RpcError("gave up after 5 attempts: malformed response: expected an object"),
    RpcError("gave up after 5 attempts: malformed response: no 'result' or 'error' field"),
    RpcBatchStructureError("malformed batch response: expected 100 entries, got dict"),
    RpcBatchStructureError("batch entry id=7: missing result"),
    RpcError("non-retryable error: [-32602] invalid params"),
    RpcError("response from https://mainnet.base.org exceeds the 67108864 byte limit"),
    requests.exceptions.JSONDecodeError("Expecting value", "not json", 0),
    ValueError("Expecting value: line 1 column 1 (char 0)"),
    KeyError("result"),
])
def test_real_failures_are_not_skippable(exc):
    assert live.rpc_unavailable_reason(exc) is None


# --- the context manager ----------------------------------------------------

def test_guard_fails_loudly_on_an_unavailable_endpoint():
    with pytest.raises(pytest.fail.Exception, match="RPC unavailable: transport failure: ConnectionError"):
        with live.rpc_available():
            raise requests.ConnectionError("down")


def test_guard_fails_loudly_on_a_throttled_endpoint():
    with pytest.raises(pytest.fail.Exception, match="RPC unavailable: rate limited"):
        with live.rpc_available():
            raise RpcError("gave up after 5 attempts: HTTPError (HTTP 429) from https://mainnet.base.org")


def test_guard_reraises_a_data_mismatch():
    """The whole point: a decoded-data mismatch inside the guarded block still
    fails. (Assertions live outside the guard in the live tests; this pins the
    behaviour anyway, so a future refactor cannot quietly convert drift into a
    skip.)"""
    with pytest.raises(AssertionError, match="mismatch"):
        with live.rpc_available():
            raise AssertionError("decoded feedback keys mismatch")


def test_guard_reraises_a_malformed_response():
    with pytest.raises(RpcError, match="malformed"):
        with live.rpc_available():
            raise RpcError("gave up after 5 attempts: malformed response: expected an object")


def test_guard_passes_through_when_nothing_raises():
    with live.rpc_available():
        value = 1
    assert value == 1


# --- server-supplied text must not STEER the classifier (only our own prefixes count) ---


@pytest.mark.parametrize("message", [
    "gave up after 5 attempts: [-32000] bogus payload timeout",
    "gave up after 5 attempts: [-32000] garbage rate limit",
    "gave up after 5 attempts: [-32000] connectionerror in your face",
])
def test_server_text_after_a_json_rpc_code_cannot_force_an_availability_label(message):
    assert live.rpc_unavailable_reason(RpcError(message)) is None


def test_batch_structure_error_is_never_relabelled_even_with_transport_words():
    assert live.rpc_unavailable_reason(RpcBatchStructureError("batch entry id=1: missing result (timeout)")) is None


@pytest.mark.parametrize("message, expected", [
    ("gave up after 5 attempts: ReadTimeout contacting https://h", "transport failure: readtimeout"),
    ("gave up after 5 attempts: HTTPError (HTTP 503) from https://h", "transport failure: http 503"),
    ("gave up after 5 attempts: [-32016] over rate limit", "rate limited"),
    ("gave up after 5 attempts: HTTPError (HTTP 429) from https://h", "rate limited"),
])
def test_our_own_redacted_prefixes_still_classify(message, expected):
    assert live.rpc_unavailable_reason(RpcError(message)) == expected


# --- recorded fetch failures (the sync path that never raises) ------------------
#
# ``base.sync_feedback`` deliberately does not raise when a chunk's
# ``eth_getLogs`` fails: it records the range in ``failed_ranges`` and moves
# on. So ``rpc_available()`` around it sees nothing, and a throttled endpoint
# surfaced as a plain assertion failure on the checkpoint (2026-10-12 weekly
# run: both ranges HTTP 429, ``last_block`` never written). ``check_failed_ranges``
# closes that gap: it reads the recorded errors and applies the same classifier.


def _store_with_failures(tmp_path, errors):
    store = Store(tmp_path / "t.db")
    for i, err in enumerate(errors):
        store.add_failed_range(i * 2000, i * 2000 + 1999, err)
    return store


def test_check_failed_ranges_passes_through_when_nothing_failed(tmp_path):
    with Store(tmp_path / "t.db") as store:
        live.check_failed_ranges(store)  # no exception at all


def test_check_failed_ranges_fails_labelled_when_every_failure_is_throttling(tmp_path):
    with _store_with_failures(tmp_path, [
        "gave up after 5 attempts: HTTPError (HTTP 429) from https://mainnet.base.org",
        "gave up after 5 attempts: HTTPError (HTTP 429) from https://mainnet.base.org",
    ]) as store:
        with pytest.raises(pytest.fail.Exception, match="RPC unavailable: .*rate limited"):
            live.check_failed_ranges(store)


def test_check_failed_ranges_fails_labelled_on_transport_failures_too(tmp_path):
    with _store_with_failures(tmp_path, [
        "gave up after 5 attempts: ReadTimeout contacting https://mainnet.base.org",
        "gave up after 5 attempts: HTTPError (HTTP 503) from https://mainnet.base.org",
    ]) as store:
        with pytest.raises(pytest.fail.Exception, match="RPC unavailable: .*transport failure"):
            live.check_failed_ranges(store)


def test_check_failed_ranges_fails_on_a_real_failure(tmp_path):
    with _store_with_failures(tmp_path, [
        "gave up after 5 attempts: malformed response: expected an object",
    ]) as store:
        with pytest.raises(pytest.fail.Exception, match="0-1999"):
            live.check_failed_ranges(store)


def test_check_failed_ranges_does_not_let_throttled_siblings_hide_a_real_failure(tmp_path):
    with _store_with_failures(tmp_path, [
        "gave up after 5 attempts: HTTPError (HTTP 429) from https://mainnet.base.org",
        "gave up after 5 attempts: malformed response: expected an object",
    ]) as store:
        with pytest.raises(pytest.fail.Exception, match="2000-3999"):
            live.check_failed_ranges(store)


def test_check_failed_ranges_label_carries_no_server_text(tmp_path):
    with _store_with_failures(tmp_path, [
        f"gave up after 5 attempts: ReadTimeout contacting https://rpc.example/v2/{CANARY}",
    ]) as store:
        with pytest.raises(pytest.fail.Exception) as info:
            live.check_failed_ranges(store)
    assert CANARY not in str(info.value)


@pytest.mark.parametrize("err", [
    "gave up after 5 attempts: [-32016]",
    "gave up after 5 attempts: HTTPError (HTTP 429) from https://mainnet.base.org",
])
def test_check_failed_ranges_labels_the_scrubbed_sync_error(tmp_path, err):
    """``sync_feedback`` stores only our prefix plus the ``[code]`` token
    (``_scrub_rpc_error``); the classifier must still label that text."""
    with _store_with_failures(tmp_path, [err]) as store:
        with pytest.raises(pytest.fail.Exception, match="RPC unavailable: .*rate limited"):
            live.check_failed_ranges(store)
