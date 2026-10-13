"""``RpcClient``: a reverting ``eth_call`` is non-retryable (``RpcRevertError``)
and ``batch(..., per_entry_errors=True)`` hands back one ``RpcEntryError``
per failed entry instead of failing the whole batch.

Measured 2026-10-13: Base public nodes answer a reverting ``eth_call`` with
JSON-RPC code 3 / ``execution reverted`` (some clients ``-32000``). Through
the client as it stood, that cost 5 POSTs and 15 s of backoff per call --
and, in a batch, per *chunk*. Split from ``test_rpc.py`` for file size.
"""
import pytest

from robustrep.sources.rpc import (
    NON_RETRYABLE,
    RpcBatchRateLimitError,
    RpcBatchStructureError,
    RpcClient,
    RpcEntryError,
    RpcError,
    RpcRevertError,
    scrub_rpc_error,
    scrub_rpc_text,
)
from test_rpc import FakeSession

REVERT_3 = {"code": 3, "message": "execution reverted"}
REVERT_32000 = {"code": -32000, "message": "execution reverted"}
OUT_OF_GAS = {"code": -32000, "message": "out of gas"}


def _client(session, sleeps, urls=("http://a", "http://b")):
    return RpcClient(list(urls), user_agent="ua", session=session, sleep=sleeps.append, retries=5)


# --- single call -------------------------------------------------------------------


def test_execution_reverted_is_in_the_non_retryable_set():
    assert "execution reverted" in NON_RETRYABLE


@pytest.mark.parametrize("error", [REVERT_3, REVERT_32000], ids=["code-3", "code--32000"])
def test_single_call_revert_is_one_post_no_sleep_and_a_revert_error(error):
    s = FakeSession([{"jsonrpc": "2.0", "id": 1, "error": error}])
    sleeps = []
    c = _client(s, sleeps)
    with pytest.raises(RpcRevertError, match="execution reverted") as e:
        c.call("eth_call", [{"to": "0x" + "a1" * 20, "data": "0x8ea69029"}, "latest"])
    assert isinstance(e.value, RpcError)
    assert len(s.calls) == 1 and sleeps == []
    assert c.current_url == "http://a"  # no rotation either


def test_other_json_rpc_errors_still_retry():
    s = FakeSession([{"error": OUT_OF_GAS}, {"result": "0x"}])
    sleeps = []
    assert _client(s, sleeps).call("eth_call", []) == "0x"
    assert len(s.calls) == 2 and sleeps == [1]


def test_revert_error_is_not_raised_for_text_that_merely_mentions_revert():
    # ``NON_RETRYABLE`` matches "execution reverted", not any "revert": a
    # provider's own rate-limit prose can say "reverted" and must still retry.
    s = FakeSession([{"error": {"message": "request reverted to queue, retry"}}, {"result": "ok"}])
    sleeps = []
    assert _client(s, sleeps).call("m", []) == "ok"
    assert len(s.calls) == 2


# --- batch, default contract -------------------------------------------------------


def test_batch_default_revert_entry_is_one_post_and_names_the_entry():
    s = FakeSession([[{"id": 1, "result": "a"}, {"id": 2, "error": REVERT_3}, {"id": 3, "result": "c"}]])
    sleeps = []
    with pytest.raises(RpcRevertError, match=r"batch entry id=2: \[3\] execution reverted"):
        _client(s, sleeps).batch([("m", [1]), ("m", [2]), ("m", [3])])
    assert len(s.calls) == 1 and sleeps == []


def test_batch_default_non_revert_entry_error_still_retries():
    s = FakeSession([[{"id": 1, "error": OUT_OF_GAS}], [{"id": 1, "result": "a"}]])
    sleeps = []
    assert _client(s, sleeps).batch([("m", [1])]) == ["a"]
    assert len(s.calls) == 2 and sleeps == [1]


# --- batch, per_entry_errors=True --------------------------------------------------


def test_per_entry_errors_returns_mixed_results_and_entry_errors_in_call_order():
    s = FakeSession([[
        {"id": 3, "result": "c"},
        {"id": 1, "result": "a"},
        {"id": 2, "error": REVERT_3},
        {"id": 4, "error": OUT_OF_GAS},
    ]])
    sleeps = []
    got = _client(s, sleeps).batch([("m", [1]), ("m", [2]), ("m", [3]), ("m", [4])], per_entry_errors=True)
    assert len(s.calls) == 1 and sleeps == []
    assert got[0] == "a" and got[2] == "c"
    assert got[1] == RpcEntryError(code=3, message="[3]", reverted=True)
    assert got[3] == RpcEntryError(code=-32000, message="[-32000]", reverted=False)
    assert all(isinstance(x, RpcEntryError) for x in (got[1], got[3]))


def test_per_entry_error_message_is_scrubbed_of_server_text():
    userinfo_url = "https://user:secret@host/path"  # pragma: allowlist secret
    s = FakeSession([[{"id": 1, "error": {"code": -32000, "message": f"boom {userinfo_url}\r\nmore"}}]])
    got = _client(s, []).batch([("m", [1])], per_entry_errors=True)
    assert got == [RpcEntryError(code=-32000, message="[-32000]", reverted=False)]
    assert "secret" not in got[0].message and "\r" not in got[0].message


def test_per_entry_error_without_a_code_has_none_code_and_scrubbed_message():
    s = FakeSession([[{"id": 1, "error": {"message": "execution reverted: nope"}}]])
    got = _client(s, []).batch([("m", [1])], per_entry_errors=True)
    assert got == [RpcEntryError(code=None, message="execution reverted: nope", reverted=True)]


def test_per_entry_error_ignores_a_non_integer_code():
    s = FakeSession([[{"id": 1, "error": {"code": "3", "message": "execution reverted"}}]])
    got = _client(s, []).batch([("m", [1])], per_entry_errors=True)
    assert got[0].code is None and got[0].reverted is True


def test_per_entry_errors_all_results_is_identical_to_the_default():
    body = [{"id": 1, "result": "a"}, {"id": 2, "result": "b"}]
    assert _client(FakeSession([body]), []).batch([("m", [1]), ("m", [2])], per_entry_errors=True) == ["a", "b"]


@pytest.mark.parametrize("body, exc", [
    ({"error": {"message": "single object"}}, RpcBatchStructureError),
    ([{"id": 1, "result": "a"}], RpcBatchStructureError),                         # short
    ([{"id": 1, "result": "a"}, {"id": 3, "result": "b"}], RpcBatchStructureError),  # id mismatch
    ([{"id": 1, "result": "a"}, {"id": 2, "result": None}], RpcBatchStructureError),  # null result, no error
    ([{"id": 1, "result": "a"}, {"id": 2, "error": {"code": -32016, "message": "over rate limit"}}],
     RpcBatchRateLimitError),
], ids=["dict", "short", "ids", "null-result", "rate-limit"])
def test_per_entry_errors_structural_and_rate_limit_failures_still_raise_at_once(body, exc):
    s = FakeSession([body])
    sleeps = []
    with pytest.raises(exc):
        _client(s, sleeps).batch([("m", [1]), ("m", [2])], per_entry_errors=True)
    assert len(s.calls) == 1 and sleeps == []


def test_per_entry_errors_transport_failure_still_retries_and_gives_up():
    import requests

    s = FakeSession([requests.ConnectionError()] * 5)
    sleeps = []
    with pytest.raises(RpcError, match="gave up after 5 attempts"):
        _client(s, sleeps).batch([("m", [1])], per_entry_errors=True)
    assert len(s.calls) == 5 and sleeps == [1, 2, 4, 8]


def test_rpc_entry_error_is_immutable_and_not_an_exception():
    e = RpcEntryError(code=3, message="[3]", reverted=True)
    assert not isinstance(e, BaseException)
    with pytest.raises(Exception):
        e.code = 4  # type: ignore[misc]


# --- scrub helpers live here now (base_erc8004 re-exports) -------------------------


def test_scrub_rpc_text_and_error_agree():
    raw = "gave up after 5 attempts: [-32016] https://user:secret@host/x"  # pragma: allowlist secret
    assert scrub_rpc_text(raw) == "gave up after 5 attempts: [-32016]"
    assert scrub_rpc_error(RpcError(raw)) == scrub_rpc_text(raw)
    from robustrep.sources import base_erc8004 as b

    assert b._scrub_rpc_error is scrub_rpc_error


# --- callers: owner_of / account_owner read a revert as "no owner" by type ----------


class _RevertingRpc:
    """``call`` raises ``exc`` for every eth_call."""

    def __init__(self, exc):
        self.exc, self.n = exc, 0

    def call(self, method, params):
        self.n += 1
        raise self.exc


@pytest.mark.parametrize("exc", [
    RpcRevertError("non-retryable error: [3] execution reverted"),
    RpcError("gave up after 5 attempts: execution reverted: ERC721: nonexistent"),  # legacy text shape
], ids=["typed", "substring"])
def test_owner_of_and_account_owner_treat_a_revert_as_none(exc):
    from robustrep.sources.account_owner import account_owner
    from robustrep.sources.base_erc8004 import owner_of
    from robustrep.sources.rpc import is_revert

    assert is_revert(exc)
    assert owner_of(_RevertingRpc(exc), "1") is None
    rpc = _RevertingRpc(exc)
    assert account_owner(rpc, "0x" + "a1" * 20) is None
    assert rpc.n == 2  # ownerAtIndex(0) reverted -> straight to owner(), which reverted too


def test_is_revert_is_false_for_other_errors():
    from robustrep.sources.rpc import is_revert

    assert not is_revert(RpcError("gave up after 5 attempts: HTTPError (HTTP 503) from https://h"))
    assert not is_revert(RuntimeError("execution reverted"))  # not an RpcError at all
