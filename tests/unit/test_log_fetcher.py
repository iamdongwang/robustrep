"""Unit tests for ``robustrep.sources.log_fetcher.LogFetcher``.

``RoutingSession`` stands in for ``requests.Session`` and routes each POST by
URL to a handler, so tests can give each RPC endpoint its own behaviour.
``FakeRaw``/``FakeResponse`` are copied from ``test_rpc.py`` because ``tests/``
is not a package.
"""
import json as jsonlib
from typing import Callable

import pytest

from robustrep.sources.log_fetcher import LogFetcher, Served
from robustrep.sources.rpc import RpcClient

FILTER = {"address": "0x8004", "topics": [["0xaa"]]}


class FakeRaw:
    """Stand-in for ``urllib3.HTTPResponse`` (see ``test_rpc.py``)."""

    def __init__(self, data: bytes):
        self.data = data

    def read(self, amt, decode_content=True):
        chunk, self.data = self.data[:amt], self.data[amt:]
        return chunk


class FakeResponse:
    status_code = 200

    def __init__(self, body: bytes):
        self.raw, self.closed = FakeRaw(body), False

    def raise_for_status(self):
        pass

    def close(self):
        self.closed = True


class RoutingSession:
    """Fake ``requests.Session`` that dispatches each POST to ``handlers[url]``.

    A handler takes the JSON-RPC payload and returns a JSON object (serialized
    as the response body) or returns/raises an ``Exception`` (raised from
    ``post``). Every call is recorded in ``calls`` as ``(url, payload)``."""

    def __init__(self, handlers: dict[str, Callable[[dict], object]]):
        self.handlers = handlers
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, json=None, headers=None, timeout=None, stream=None):
        self.calls.append((url, json))
        r = self.handlers[url](json)
        if isinstance(r, Exception):
            raise r
        return FakeResponse(jsonlib.dumps(r).encode())


def getlogs_ranges(session: RoutingSession, url: str) -> list[tuple[int, int]]:
    """The ``(fromBlock, toBlock)`` of every ``eth_getLogs`` sent to ``url``."""
    return [(int(p["params"][0]["fromBlock"], 16), int(p["params"][0]["toBlock"], 16))
            for u, p in session.calls if u == url and p["method"] == "eth_getLogs"]


def make_client(urls, session, retries=5) -> RpcClient:
    return RpcClient(urls, user_agent="ua", session=session, sleep=lambda _: None, retries=retries)


def test_single_window_success_records_served():
    s = RoutingSession({"https://a": lambda p: {"result": [{"blockNumber": "0x5"}]}})
    f = LogFetcher(make_client(["https://a"], s), max_window=2000)
    res = f.get_logs(0, 1999, FILTER)
    assert res.logs == [{"blockNumber": "0x5"}]
    assert res.served == [Served(0, 1999, "https://a", 2000, 1)]
    assert getlogs_ranges(s, "https://a") == [(0, 1999)]


def test_splits_range_into_windows_and_truncates_last():
    s = RoutingSession({"https://a": lambda p: {"result": []}})
    f = LogFetcher(make_client(["https://a"], s), max_window=2000, min_window=50, window_hints={"https://a": 700})
    res = f.get_logs(10, 1509, FILTER)
    assert getlogs_ranges(s, "https://a") == [(10, 709), (710, 1409), (1410, 1509)]
    assert [(x.from_block, x.to_block, x.window) for x in res.served] == [(10, 709, 700), (710, 1409, 700), (1410, 1509, 700)]


def test_logs_concatenated_in_window_order():
    s = RoutingSession({"https://a": lambda p: {"result": [{"blockNumber": p["params"][0]["fromBlock"]}]}})
    f = LogFetcher(make_client(["https://a"], s), max_window=100, min_window=50, window_hints={"https://a": 50})
    assert [l["blockNumber"] for l in f.get_logs(0, 149, FILTER).logs] == ["0x0", "0x32", "0x64"]


def test_request_shape_includes_filter():
    s = RoutingSession({"https://a": lambda p: {"result": []}})
    LogFetcher(make_client(["https://a"], s)).get_logs(3, 4, FILTER)
    assert s.calls[0][1]["params"] == [{"fromBlock": "0x3", "toBlock": "0x4", **FILTER}]


def test_from_greater_than_to_raises_value_error():
    with pytest.raises(ValueError):
        LogFetcher(make_client(["https://a"], RoutingSession({}))).get_logs(5, 4, FILTER)


@pytest.mark.parametrize("kw", [dict(min_window=0), dict(min_window=2001), dict(min_interval=-1),
                                dict(window_hints={"https://a": 49}), dict(window_hints={"https://a": 2001})])
def test_constructor_validation(kw):
    with pytest.raises(ValueError):
        LogFetcher(make_client(["https://a"], RoutingSession({})), max_window=2000, **{"min_window": 50, **kw})


def test_min_interval_is_start_to_start_and_skips_sleep_when_call_is_slow():
    now = [0.0]; slept = []
    def clock(): return now[0]
    def fast(p): return {"result": []}
    def slow(p): now[0] += 0.5; return {"result": []}
    s = RoutingSession({"https://a": fast})
    LogFetcher(make_client(["https://a"], s), max_window=50, min_window=50, min_interval=0.2,
               sleep=slept.append, clock=clock).get_logs(0, 149, FILTER)      # 3 calls, instant
    assert slept == pytest.approx([0.2, 0.2])
    slept.clear()
    LogFetcher(make_client(["https://a"], RoutingSession({"https://a": slow})), max_window=50, min_window=50,
               min_interval=0.2, sleep=slept.append, clock=clock).get_logs(0, 149, FILTER)
    assert slept == []


def test_default_min_interval_is_zero_and_never_sleeps():
    slept = []
    LogFetcher(make_client(["https://a"], RoutingSession({"https://a": lambda p: {"result": []}})),
               max_window=50, min_window=50, sleep=slept.append).get_logs(0, 149, FILTER)
    assert slept == []
