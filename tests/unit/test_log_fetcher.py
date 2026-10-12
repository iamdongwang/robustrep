"""Unit tests for ``robustrep.sources.log_fetcher.LogFetcher``.

``RoutingSession`` stands in for ``requests.Session`` and routes each POST by
URL to a handler, so tests can give each RPC endpoint its own behaviour.
``FakeRaw``/``FakeResponse`` are copied from ``test_rpc.py`` because ``tests/``
is not a package.
"""
import json as jsonlib
from typing import Callable

import pytest
import requests

from robustrep.sources.log_fetcher import LogFetcher, Served, shrink
from robustrep.sources.rpc import RpcClient, RpcError

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


def test_shrink_rules():
    assert shrink(2000, (32001, 50, 2), 50) == 50     # nodies：一步到 50（code 与 "v2" 被过滤）
    assert shrink(5000, (32614, 2000), 50) == 2000    # "2,000"：一步到 2000
    assert shrink(2000, (35, 10000), 50) == 1000      # drpc：没有合格数字，对半
    assert shrink(2000, (1999,), 50) == 1000          # 只给 w-1：仍减半
    assert shrink(2000, (), 50) == 1000
    assert shrink(60, (), 50) == 50                   # 对半低于下限取下限
    assert shrink(2000, (-5, 0, 10**30, 49), 50) == 1000

def _capped(limit):
    def h(p):
        a, z = int(p["params"][0]["fromBlock"], 16), int(p["params"][0]["toBlock"], 16)
        if z - a + 1 > limit:
            return {"error": {"code": -32001, "message": f"Block range too large: maximum allowed is {limit} blocks"}}
        return {"result": []}
    return h

def test_one_step_shrink_then_sticks():
    s = RoutingSession({"https://n": _capped(50)})
    f = LogFetcher(make_client(["https://n"], s), max_window=2000, min_window=50)
    res = f.get_logs(0, 99, FILTER)
    assert getlogs_ranges(s, "https://n") == [(0, 99), (0, 49), (50, 99)]
    assert f.windows["https://n"] == 50 and [x.window for x in res.served] == [50, 50]

def test_halving_sequence_for_drpc_style_message():
    def drpc(p):
        a, z = int(p["params"][0]["fromBlock"], 16), int(p["params"][0]["toBlock"], 16)
        return {"result": []} if z - a + 1 <= 100 else {"error": {"code": 35, "message": "ranges over 10000 blocks are not supported on free plan"}}
    s = RoutingSession({"https://d": drpc})
    LogFetcher(make_client(["https://d"], s), max_window=2000, min_window=50).get_logs(0, 1999, FILTER)
    sizes = [z - a + 1 for a, z in getlogs_ranges(s, "https://d")]
    assert sizes[:6] == [2000, 1000, 500, 250, 125, 62] and all(x <= 62 for x in sizes[6:])

def test_windows_only_shrink_never_grow():
    s = RoutingSession({"https://n": _capped(50)})
    f = LogFetcher(make_client(["https://n"], s), max_window=2000, min_window=50, window_hints={"https://n": 100})
    f.get_logs(0, 199, FILTER); f.get_logs(200, 399, FILTER)
    assert f.windows["https://n"] == 50 and all(z - a + 1 <= 50 for a, z in getlogs_ranges(s, "https://n")[1:])

def test_shrinks_the_raising_endpoint_not_the_starting_host():
    """drpc（提示 100）第一次 ConnectionError → 客户端轮到 nodies，仍是 100 块请求 → nodies 抛范围错误。
    必须对 nodies 缩窗到 50，不动 drpc 的 100，也不把 nodies 标 dead。"""
    seen = {"n": 0}
    def drpc(p):
        seen["n"] += 1
        if seen["n"] == 1: raise requests.ConnectionError()
        return {"result": []}
    s = RoutingSession({"https://d": drpc, "https://n": _capped(50)})
    f = LogFetcher(make_client(["https://d", "https://n"], s), max_window=2000, min_window=50, window_hints={"https://d": 100})
    f.get_logs(0, 99, FILTER)
    assert f.windows == {"https://d": 100, "https://n": 50} and f.dead == set()

def test_min_window_rejection_marks_dead_and_rotates():
    s = RoutingSession({"https://ten": _capped(10), "https://n": _capped(50)})
    f = LogFetcher(make_client(["https://ten", "https://n"], s), max_window=2000, min_window=50)
    res = f.get_logs(0, 49, FILTER)
    assert f.dead == {"https://ten"} and [x.host for x in res.served] == ["https://n"]

def test_all_endpoints_dead_raises_rpcerror_without_partial_result():
    s = RoutingSession({"https://a": _capped(10), "https://b": _capped(10)})
    f = LogFetcher(make_client(["https://a", "https://b"], s), max_window=2000, min_window=50)
    with pytest.raises(RpcError, match="no usable endpoint"):
        f.get_logs(0, 99, FILTER)

def test_request_count_bounded_under_hostile_endpoint():
    """每次只回 w-1：缩窗 ≤ log2(2000/50)+1 = 7 次，然后 dead。"""
    def hostile(p):
        a, z = int(p["params"][0]["fromBlock"], 16), int(p["params"][0]["toBlock"], 16)
        return {"error": {"code": -32001, "message": f"Block range too large: maximum allowed is {z - a} blocks"}}
    s = RoutingSession({"https://h": hostile, "https://n": _capped(50)})
    f = LogFetcher(make_client(["https://h", "https://n"], s), max_window=2000, min_window=50)
    f.get_logs(0, 1999, FILTER)
    assert len(getlogs_ranges(s, "https://h")) <= 7 and f.dead == {"https://h"}

def test_invalid_rounds_bound_terminates_when_live_host_keeps_failing():
    """nodies dead，drpc 永远 ConnectionError：客户端内部轮换不断转回 dead 的 nodies。必须在有限请求内抛 RpcError。"""
    def down(p): raise requests.ConnectionError()
    s = RoutingSession({"https://d": down, "https://n": _capped(10)})
    f = LogFetcher(make_client(["https://d", "https://n"], s), max_window=2000, min_window=50)
    with pytest.raises(RpcError):
        f.get_logs(0, 49, FILTER)
    assert len(s.calls) <= (2 + 1) * 5 + 7       # (len(urls)+1) 回合 × retries + 缩窗

def test_hints_seed_windows_and_learned_values_override_only_downwards():
    s = RoutingSession({"https://n": _capped(50)})
    f = LogFetcher(make_client(["https://n"], s), max_window=2000, min_window=50, window_hints={"https://n": 200})
    f.get_logs(0, 199, FILTER)
    assert getlogs_ranges(s, "https://n")[0] == (0, 199)      # 首个请求按提示 200
    assert f.windows["https://n"] == 50

def test_switching_to_a_larger_endpoint_returns_to_max_window():
    """nodies 学到 50 后挂掉（ConnectionError），客户端轮到 publicnode（未知 → 2000）。"""
    state = {"down": False}
    def nodies(p):
        if state["down"]: raise requests.ConnectionError()
        return _capped(50)(p)
    s = RoutingSession({"https://n": nodies, "https://p": lambda p: {"result": []}})
    f = LogFetcher(make_client(["https://n", "https://p"], s), max_window=2000, min_window=50)
    f.get_logs(0, 99, FILTER)                      # 100 块被拒 → nodies 学到 50
    assert f.windows["https://n"] == 50
    state["down"] = True
    res = f.get_logs(100, 2149, FILTER)            # (100,149) 由 p 以 50 块接手；之后 p 未知 → 2000
    assert res.served[0].host == "https://p" and res.served[0].window == 50
    assert [(x.from_block, x.to_block, x.window) for x in res.served][1:] == [(150, 2149, 2000)]

def test_other_rpcerrors_propagate_whole_range_discarded():
    s = RoutingSession({"https://a": lambda p: {"error": {"code": -32000, "message": "header not found"}}})
    with pytest.raises(RpcError, match="gave up"):
        LogFetcher(make_client(["https://a"], s, retries=2), max_window=50, min_window=50).get_logs(0, 99, FILTER)


def test_non_list_result_raises_rpcerror():
    for body in ({"result": None}, {"result": {"x": 1}}):
        s = RoutingSession({"https://a": lambda p, body=body: body})
        f = LogFetcher(make_client(["https://a"], s), max_window=50, min_window=50)
        res = None
        with pytest.raises(RpcError, match="non-list"):
            res = f.get_logs(0, 99, FILTER)
        assert res is None


def test_from_config_clamps_min_window_to_chunk():
    from robustrep.config import Config

    rpc = make_client(["https://a"], RoutingSession({}))
    assert LogFetcher.from_config(rpc, Config(chunk_blocks=20)).min_window == 20
    f = LogFetcher.from_config(rpc, Config())
    assert (f.max_window, f.min_window, f.min_interval) == (2000, 50, 0.2)
