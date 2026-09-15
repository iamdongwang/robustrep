"""Tests for the live-suite's availability guard (``tests/e2e/test_live.py``'s
``rpc_unavailable_reason`` / ``rpc_available``).

The guard decides whether a weekly live run reports "the public endpoint was
down or throttling us" (skip) or "the chain data no longer matches what we
recorded" (fail). Getting that backwards silently disables the drift detector,
so the classifier is pinned here -- offline, unmarked, part of every run.

``tests/`` is not a package, so the live module is loaded by path under a name
of its own rather than imported; nothing in it touches the network at import
time.
"""
import importlib.util
from pathlib import Path

import pytest
import requests

from robustrep.sources.rpc import RpcBatchRateLimitError, RpcError


def _load_live_module():
    path = Path(__file__).resolve().parents[1] / "e2e" / "test_live.py"
    spec = importlib.util.spec_from_file_location("robustrep_live_guard_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


live = _load_live_module()


@pytest.mark.parametrize("exc", [
    requests.ConnectionError("failed to establish a connection"),
    requests.ConnectTimeout("connect timed out"),
    requests.ReadTimeout("read timed out"),
    requests.Timeout("timed out"),
    TimeoutError("socket timed out"),
])
def test_transport_failures_are_skippable(exc):
    reason = live.rpc_unavailable_reason(exc)
    assert reason is not None
    assert type(exc).__name__ in reason


def test_transport_failure_reason_does_not_echo_the_url():
    """A raw ``requests`` exception's ``str()`` carries the full request URL,
    which may embed a provider API key -- the reason string must not."""
    exc = requests.ConnectionError("HTTPSConnectionPool(host='rpc.example') "
                                   "url: /v2/SECRET_KEY_9f3a (Caused by ...)")
    reason = live.rpc_unavailable_reason(exc)
    assert "SECRET_KEY_9f3a" not in reason


@pytest.mark.parametrize("message", [
    "gave up after 5 attempts: HTTPError (HTTP 429) from https://mainnet.base.org",
    "gave up after 5 attempts: too many requests",
    "batch entry id=3: [-32016] over rate limit",
    "gave up after 5 attempts: [-32005] rate limit exceeded",
    "gave up after 5 attempts: ConnectionError contacting https://mainnet.base.org",
    "gave up after 5 attempts: ReadTimeout contacting https://mainnet.base.org",
])
def test_rpc_errors_naming_throttling_or_transport_are_skippable(message):
    reason = live.rpc_unavailable_reason(RpcError(message))
    assert reason == message


def test_rate_limited_batch_subclass_is_skippable():
    exc = RpcBatchRateLimitError("batch entry id=1: [-32016] over rate limit")
    assert live.rpc_unavailable_reason(exc) is not None


@pytest.mark.parametrize("exc", [
    AssertionError("assert 41 == 42"),
    RpcError("gave up after 5 attempts: malformed response: expected an object"),
    RpcError("non-retryable error: [-32602] invalid params"),
    RpcError("response from https://mainnet.base.org exceeds the 67108864 byte limit"),
    ValueError("Expecting value: line 1 column 1 (char 0)"),
    KeyError("result"),
])
def test_real_failures_are_not_skippable(exc):
    assert live.rpc_unavailable_reason(exc) is None


def test_guard_skips_on_an_unavailable_endpoint():
    with pytest.raises(pytest.skip.Exception, match="RPC unavailable: ConnectionError"):
        with live.rpc_available():
            raise requests.ConnectionError("down")


def test_guard_reraises_a_data_mismatch():
    """The whole point: a decoded-data mismatch inside the guarded block still
    fails. (Assertions live outside the guard in the live tests; this pins the
    behaviour anyway, so a future refactor cannot quietly convert drift into a
    skip.)"""
    with pytest.raises(AssertionError, match="mismatch"):
        with live.rpc_available():
            raise AssertionError("decoded feedback keys mismatch")


def test_guard_passes_through_when_nothing_raises():
    with live.rpc_available():
        value = 1
    assert value == 1
