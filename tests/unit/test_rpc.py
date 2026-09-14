import pytest
import requests

from robustrep.sources.rpc import RpcClient, RpcError


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), [], []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json))
        self.headers.append(headers)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        class R:  # minimal response
            status_code = 200
            def json(self_inner):
                return r
            def raise_for_status(self_inner):
                pass
        return R()


def test_call_returns_result_and_sets_user_agent():
    s = FakeSession([{"jsonrpc": "2.0", "id": 1, "result": "0x10"}])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None)
    assert c.call("eth_blockNumber", []) == "0x10"
    assert s.calls[0][1]["method"] == "eth_blockNumber"


def test_retries_then_rotates_url():
    s = FakeSession([requests.ConnectionError(), {"error": {"message": "rate"}}, {"result": "ok"}])
    c = RpcClient(["http://a", "http://b"], user_agent="ua", session=s, sleep=lambda _: None, retries=3)
    assert c.call("m", []) == "ok"
    assert [u for u, _ in s.calls] == ["http://a", "http://b", "http://a"]


def test_gives_up_after_retries():
    s = FakeSession([{"error": {"message": "x"}}] * 3)
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None, retries=3)
    with pytest.raises(RpcError):
        c.call("m", [])


def test_batch_preserves_order():
    s = FakeSession([[{"id": 2, "result": "b"}, {"id": 1, "result": "a"}]])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None)
    assert c.batch([("m", [1]), ("m", [2])]) == ["a", "b"]


def test_call_raises_rpcerror_when_no_result_and_no_error():
    # response has neither "result" nor "error" -- malformed, must not KeyError.
    s = FakeSession([{"jsonrpc": "2.0", "id": 1}] * 2)
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None, retries=2)
    with pytest.raises(RpcError):
        c.call("m", [])


def test_batch_with_error_entry_raises_rpcerror():
    s = FakeSession([[{"id": 1, "result": "a"}, {"id": 2, "error": {"message": "bad"}}]])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None, retries=1)
    with pytest.raises(RpcError):
        c.batch([("m", [1]), ("m", [2])])


def test_backoff_sleeps_with_exponential_values():
    s = FakeSession([{"error": {"message": "x"}}, {"error": {"message": "x"}}, {"error": {"message": "x"}},
                      {"result": "ok"}])
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=4)
    assert c.call("m", []) == "ok"
    assert sleeps == [1, 2, 4]


def test_post_sends_user_agent_header():
    s = FakeSession([{"result": "ok"}])
    c = RpcClient(["http://a"], user_agent="my-ua/1.0", session=s, sleep=lambda _: None)
    c.call("m", [])
    assert s.headers[0]["user-agent"] == "my-ua/1.0"


def test_batch_short_response_raises():
    # 3 requested, only 2 returned
    s = FakeSession([[{"id": 1, "result": "a"}, {"id": 2, "result": "b"}]] * 2)
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None, retries=2)
    with pytest.raises(RpcError):
        c.batch([("m", [1]), ("m", [2]), ("m", [3])])


def test_batch_id_mismatch_raises():
    s = FakeSession([[{"id": 1, "result": "a"}, {"id": 2, "result": "b"}, {"id": 4, "result": "c"}]] * 2)
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None, retries=2)
    with pytest.raises(RpcError):
        c.batch([("m", [1]), ("m", [2]), ("m", [3])])


def test_batch_null_result_raises():
    s = FakeSession([[{"id": 1, "result": "a"}, {"id": 2, "result": None}]] * 2)
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None, retries=2)
    with pytest.raises(RpcError):
        c.batch([("m", [1]), ("m", [2])])


def test_batch_builds_result_by_id_not_position():
    # server returns entries out of order and mixed with extra whitespace-ish id ordering
    s = FakeSession([[{"id": 3, "result": "c"}, {"id": 1, "result": "a"}, {"id": 2, "result": "b"}]])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None)
    assert c.batch([("m", [1]), ("m", [2]), ("m", [3])]) == ["a", "b", "c"]


def test_error_null_is_not_treated_as_error():
    s = FakeSession([{"jsonrpc": "2.0", "id": 1, "result": "ok", "error": None}])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None)
    assert c.call("m", []) == "ok"


def test_batch_error_null_is_not_treated_as_error():
    s = FakeSession([[{"id": 1, "result": "a", "error": None}, {"id": 2, "result": "b", "error": None}]])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None)
    assert c.batch([("m", [1]), ("m", [2])]) == ["a", "b"]


def test_no_sleep_after_final_failed_attempt():
    s = FakeSession([{"error": {"message": "x"}}] * 3)
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=3)
    with pytest.raises(RpcError):
        c.call("m", [])
    assert sleeps == [1, 2]


def test_non_retryable_error_raises_immediately_without_rotation_or_sleep():
    s = FakeSession([{"error": {"code": -32602, "message": "invalid params"}}])
    sleeps = []
    c = RpcClient(["http://a", "http://b"], user_agent="ua", session=s, sleep=sleeps.append, retries=5)
    with pytest.raises(RpcError):
        c.call("m", [])
    assert len(s.calls) == 1
    assert sleeps == []


def test_non_retryable_error_by_message_substring():
    s = FakeSession([{"error": {"message": "eth_getLogs is limited to a 2000 range"}}])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None, retries=5)
    with pytest.raises(RpcError):
        c.call("m", [])
    assert len(s.calls) == 1
