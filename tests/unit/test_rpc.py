import json as jsonlib
import threading

import pytest
import requests

from robustrep.sources.rpc import RpcBatchRateLimitError, RpcBatchStructureError, RpcClient, RpcError


class FakeRaw:
    """Stand-in for ``urllib3.HTTPResponse``: the streamed byte source that
    ``read_json_capped`` consumes (bodies are capped, never ``r.json()``-ed --
    M2)."""

    def __init__(self, data: bytes):
        self.data = data

    def read(self, amt, decode_content=True):
        chunk, self.data = self.data[:amt], self.data[amt:]
        return chunk


class FakeResponse:
    status_code = 200

    def __init__(self, body: bytes, session=None):
        self.raw, self._session, self.closed = FakeRaw(body), session, False

    def raise_for_status(self):
        pass

    def close(self):
        self.closed = True
        if self._session is not None:
            self._session.closes += 1


class FakeSession:
    """Fake ``requests.Session`` whose responses are streamed byte bodies.

    ``responses`` items are decoded JSON objects (serialized here), raw
    ``bytes`` (for a malformed/oversized body), or an ``Exception`` to raise
    from ``post`` itself."""

    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), [], []
        self.streams, self.closes = [], 0

    def post(self, url, json=None, headers=None, timeout=None, stream=None):
        self.calls.append((url, json))
        self.headers.append(headers)
        self.streams.append(stream)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        body = r if isinstance(r, bytes) else jsonlib.dumps(r).encode()
        return FakeResponse(body, session=self)


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


def test_non_retryable_error_matches_via_code():
    # message alone ("bad") doesn't match NON_RETRYABLE; the numeric code must
    # be folded into the checked message so "-32602"/"-32600" entries work.
    s = FakeSession([{"error": {"code": -32602, "message": "bad"}}])
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=5)
    with pytest.raises(RpcError):
        c.call("m", [])
    assert len(s.calls) == 1
    assert sleeps == []


def test_gives_up_message_never_includes_raw_exception_text_or_url():
    # Many paid RPC providers embed an API key right in the URL path (e.g.
    # ".../v2/<key>"); requests/urllib3 exceptions stringify the *whole*
    # request URL, so surfacing str(exc) verbatim would leak it into logs,
    # CLI output, or a bug report. The RpcError message must carry only the
    # exception type and the endpoint's scheme+host -- never the raw text.
    secret_url = "https://rpc.example.com/v2/SUPER_SECRET"  # pragma: allowlist secret
    s = FakeSession([requests.ConnectionError(f"Failed to establish a connection to {secret_url}")] * 2)
    c = RpcClient([secret_url], user_agent="ua", session=s, sleep=lambda _: None, retries=2)
    with pytest.raises(RpcError) as exc_info:
        c.call("m", [])
    msg = str(exc_info.value)
    assert "SUPER_SECRET" not in msg
    assert secret_url not in msg
    assert "ConnectionError" in msg
    assert "rpc.example.com" in msg  # bare scheme+host is fine, not a secret


# --- structural batch failures: non-retryable, 1 attempt, no sleep ----------------

def test_batch_dict_response_is_non_retryable_one_attempt_no_sleep():
    # a dict "error object" instead of a list -- observed on mainnet.base.org
    # for a 100-call batch. Must not burn the full retry/backoff schedule.
    s = FakeSession([{"error": {"message": "something went wrong"}}])
    sleeps = []
    c = RpcClient(["http://a", "http://b"], user_agent="ua", session=s, sleep=sleeps.append, retries=5)
    with pytest.raises(RpcBatchStructureError, match="expected 2 entries, got dict"):
        c.batch([("m", [1]), ("m", [2])])
    assert len(s.calls) == 1
    assert sleeps == []


def test_batch_short_list_is_non_retryable_one_attempt_no_sleep():
    s = FakeSession([[{"id": 1, "result": "a"}]])  # 1 entry, 2 requested
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=5)
    with pytest.raises(RpcBatchStructureError):
        c.batch([("m", [1]), ("m", [2])])
    assert len(s.calls) == 1
    assert sleeps == []


def test_batch_id_mismatch_is_non_retryable_one_attempt():
    s = FakeSession([[{"id": 1, "result": "a"}, {"id": 4, "result": "b"}]])
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=5)
    with pytest.raises(RpcBatchStructureError):
        c.batch([("m", [1]), ("m", [2])])
    assert len(s.calls) == 1
    assert sleeps == []


def test_batch_missing_result_is_non_retryable_one_attempt():
    s = FakeSession([[{"id": 1, "result": "a"}, {"id": 2, "result": None}]])
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=5)
    with pytest.raises(RpcBatchStructureError):
        c.batch([("m", [1]), ("m", [2])])
    assert len(s.calls) == 1
    assert sleeps == []


def test_batch_entry_json_rpc_error_still_retries_normally():
    # a legitimate per-call JSON-RPC error (not a structural/rate-limit
    # failure) must keep the existing retry behavior.
    s = FakeSession([[{"id": 1, "result": "a"}, {"id": 2, "error": {"message": "bad"}}],
                      [{"id": 1, "result": "a"}, {"id": 2, "result": "b"}]])
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=3)
    assert c.batch([("m", [1]), ("m", [2])]) == ["a", "b"]
    assert len(s.calls) == 2
    assert sleeps == [1]


# --- batch rate-limit errors: non-retryable, 1 attempt, no sleep ------------------

def test_batch_rate_limit_error_by_code_is_non_retryable_one_attempt():
    s = FakeSession([[{"id": 1, "result": "a"}, {"id": 2, "error": {"code": -32016, "message": "over rate limit"}}]])
    sleeps = []
    c = RpcClient(["http://a", "http://b"], user_agent="ua", session=s, sleep=sleeps.append, retries=5)
    with pytest.raises(RpcBatchRateLimitError, match="over rate limit"):
        c.batch([("m", [1]), ("m", [2])])
    assert len(s.calls) == 1
    assert sleeps == []


def test_batch_rate_limit_error_by_alternate_code_is_non_retryable():
    s = FakeSession([[{"id": 1, "error": {"code": -32005, "message": "limited"}}]])
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=5)
    with pytest.raises(RpcBatchRateLimitError):
        c.batch([("m", [1])])
    assert len(s.calls) == 1
    assert sleeps == []


def test_single_call_rate_limit_error_still_retries_normally():
    # the same "over rate limit"/-32016 signal on a *single* call (not a
    # batch) must keep retrying as normal -- only batching is abandoned.
    s = FakeSession([{"error": {"code": -32016, "message": "over rate limit"}},
                      {"error": {"code": -32016, "message": "over rate limit"}},
                      {"result": "ok"}])
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=3)
    assert c.call("m", []) == "ok"
    assert len(s.calls) == 3
    assert sleeps == [1, 2]


def test_gives_up_message_includes_http_status_code():
    class FakeResponse:
        status_code = 503

    err = requests.exceptions.HTTPError("503 Server Error: url: https://rpc.example.com/v2/SUPER_SECRET")
    err.response = FakeResponse()
    s = FakeSession([err] * 2)
    c = RpcClient(["https://rpc.example.com/v2/SUPER_SECRET"], user_agent="ua", session=s,
                   sleep=lambda _: None, retries=2)
    with pytest.raises(RpcError) as exc_info:
        c.call("m", [])
    msg = str(exc_info.value)
    assert "SUPER_SECRET" not in msg
    assert "503" in msg
    assert "rpc.example.com" in msg


# --- thread-local session (no session injected) ------------------------------------

def test_session_is_thread_local_when_none_injected():
    # classify_all drives tx_parties (and so RpcClient.call) from a thread
    # pool; a bare requests.Session is not guaranteed safe to share across
    # threads, so each thread must get its own lazily-created session.
    c = RpcClient(["http://a"], user_agent="ua", sleep=lambda _: None)
    seen = {}
    lock = threading.Lock()
    # A barrier keeps both threads alive at the point they grab (and again
    # until both have recorded) their session, so the OS can't reuse a
    # terminated thread's ident for the other thread before both idents are
    # captured -- which would make this test flaky rather than a real check
    # of thread-local isolation.
    barrier = threading.Barrier(2)

    def grab():
        barrier.wait()
        sess = c.session
        with lock:
            seen[threading.get_ident()] = sess
        barrier.wait()

    threads = [threading.Thread(target=grab) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 2
    sessions = list(seen.values())
    assert sessions[0] is not sessions[1]
    assert all(isinstance(s, requests.Session) for s in sessions)


def test_session_is_stable_within_one_thread():
    c = RpcClient(["http://a"], user_agent="ua", sleep=lambda _: None)
    assert c.session is c.session


def test_injected_session_is_shared_across_all_threads():
    injected = FakeSession([])
    c = RpcClient(["http://a"], user_agent="ua", session=injected, sleep=lambda _: None)
    seen = []
    lock = threading.Lock()

    def grab():
        with lock:
            seen.append(c.session)

    threads = [threading.Thread(target=grab) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 3
    assert all(s is injected for s in seen)


# --- M2: bounded response bodies ----------------------------------------------------


def test_post_streams_and_closes_every_response():
    s = FakeSession([{"result": "ok"}])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None)
    assert c.call("m", []) == "ok"
    assert s.streams == [True]
    assert s.closes == 1


def test_default_max_response_bytes_is_64_mib():
    c = RpcClient(["http://a"], user_agent="ua", session=FakeSession([]), sleep=lambda _: None)
    assert c.max_response_bytes == 64 * 1024 * 1024


def test_oversized_response_raises_rpcerror():
    s = FakeSession([b'{"result": "' + b"x" * 5000 + b'"}'])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None,
                  max_response_bytes=64)
    with pytest.raises(RpcError):
        c.call("m", [])


def test_oversized_response_is_not_retried_and_does_not_rotate():
    # An oversized body is deterministic: retrying (and rotating) would only
    # burn every configured endpoint for a guaranteed repeat failure.
    s = FakeSession([b"x" * 5000] * 3)
    sleeps = []
    c = RpcClient(["http://a", "http://b"], user_agent="ua", session=s, sleep=sleeps.append,
                  retries=3, max_response_bytes=64)
    with pytest.raises(RpcError):
        c.call("m", [])
    assert len(s.calls) == 1
    assert [u for u, _ in s.calls] == ["http://a"]
    assert sleeps == []


def test_oversized_response_error_names_endpoint_not_body_or_url():
    s = FakeSession([b'{"leaked": "' + b"SUPERSECRETBODY" * 500 + b'"}'])
    c = RpcClient(["https://user:pw@rpc.example/v2/APIKEYINPATH"], user_agent="ua", session=s,
                  sleep=lambda _: None, max_response_bytes=64)
    with pytest.raises(RpcError) as exc:
        c.call("m", [])
    msg = str(exc.value)
    assert "SUPERSECRETBODY" not in msg
    assert "APIKEYINPATH" not in msg and "pw" not in msg
    assert "https://rpc.example" in msg


def test_oversized_response_is_still_closed():
    s = FakeSession([b"x" * 5000])
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=lambda _: None,
                  max_response_bytes=64)
    with pytest.raises(RpcError):
        c.call("m", [])
    assert s.closes == 1


def test_non_json_body_is_still_retried():
    # A truncated/garbage body stays transient (it was `r.json()` raising
    # ValueError before the cap existed) -- only an oversized one is terminal.
    s = FakeSession([b"<html>oops</html>", {"result": "ok"}])
    c = RpcClient(["http://a", "http://b"], user_agent="ua", session=s, sleep=lambda _: None, retries=3)
    assert c.call("m", []) == "ok"
    assert len(s.calls) == 2


def test_oversized_batch_response_raises_rpcerror_without_retry():
    s = FakeSession([b"x" * 5000] * 3)
    sleeps = []
    c = RpcClient(["http://a"], user_agent="ua", session=s, sleep=sleeps.append, retries=3,
                  max_response_bytes=64)
    with pytest.raises(RpcError):
        c.batch([("m", [1])])
    assert len(s.calls) == 1 and sleeps == []
