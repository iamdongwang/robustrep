import logging

import pytest
from eth_abi import encode

from robustrep.config import Config
from robustrep.sources import base_erc8004 as b
from robustrep.sources.rpc import RpcBatchRateLimitError, RpcBatchStructureError, RpcError
from robustrep.store import Store


def test_topics_match_keccak_of_signatures():
    assert b.TOPIC_NEW_FEEDBACK == "0x6a4a61743519c9d648a14e6493f47dbe3ff1aa29e7785c96c8326a205e58febc"
    assert b.TOPIC_REVOKED.startswith("0x25156fd3288212246d")
    assert b.TOPIC_RESPONSE.startswith("0xb1c6be0b5b8aef6539")


def _log(topic0, topics, data, block=5, tx="0xt", idx=0):
    return {"topics": [topic0] + topics, "data": "0x" + data.hex(), "blockNumber": hex(block),
            "transactionHash": tx, "logIndex": hex(idx)}


def test_decode_new_feedback():
    data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                  [7, -5, 2, "quality", "t2", "https://ep", "ipfs://x", b"\x01" * 32])
    agent = "0x" + (25975).to_bytes(32, "big").hex()
    client = "0x" + "00" * 12 + "ab" * 20
    out = b.decode_log(_log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data))
    assert out["kind"] == "feedback" and out["agent_id"] == "25975" and out["client"] == "0x" + "ab" * 20
    assert out["feedback_index"] == 7 and out["value"] == "-5" and out["value_decimals"] == 2
    assert out["tag1"] == "quality" and out["feedback_uri"] == "ipfs://x" and out["block"] == 5


def test_decode_revoked_and_response():
    agent = "0x" + (1).to_bytes(32, "big").hex()
    client = "0x" + "00" * 12 + "ab" * 20
    idx = "0x" + (9).to_bytes(32, "big").hex()
    r = b.decode_log(_log(b.TOPIC_REVOKED, [agent, client, idx], b""))
    assert r["kind"] == "revoked" and r["feedback_index"] == 9
    data = encode(["uint64", "string", "bytes32"], [9, "https://resp", b"\x02" * 32])
    responder = "0x" + "00" * 12 + "cd" * 20
    s = b.decode_log(_log(b.TOPIC_RESPONSE, [agent, client, responder], data))
    assert s["kind"] == "response" and s["responder"] == "0x" + "cd" * 20 and s["response_uri"] == "https://resp"


def test_unknown_topic_is_none():
    assert b.decode_log(_log("0x" + "ff" * 32, [], b"")) is None


def test_decode_negative_value_and_max_decimals():
    data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                  [1, -12345, 255, "q", "", "", "", b"\x00" * 32])
    agent = "0x" + (2).to_bytes(32, "big").hex()
    client = "0x" + "00" * 12 + "ab" * 20
    out = b.decode_log(_log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data))
    assert out["value"] == "-12345" and out["value_decimals"] == 255


def test_decode_empty_feedback_uri():
    data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                  [1, 1, 0, "q", "", "", "", b"\x00" * 32])
    agent = "0x" + (2).to_bytes(32, "big").hex()
    client = "0x" + "00" * 12 + "ab" * 20
    out = b.decode_log(_log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data))
    assert out["feedback_uri"] == ""


def test_decode_log_missing_field_raises_valueerror():
    log = {"topics": [b.TOPIC_REVOKED], "data": "0x", "blockNumber": "0x5", "logIndex": "0x0"}
    with pytest.raises(ValueError, match="transactionHash"):
        b.decode_log(log)


class FakeRpc:
    def __init__(self, logs_by_range, head=10_000, fail_ranges=None):
        self.logs_by_range, self.head, self.calls = logs_by_range, head, []
        self.fail_ranges = set(fail_ranges or ())
        self.batch_calls = 0

    def call(self, method, params):
        self.calls.append((method, params))
        if method == "eth_blockNumber":
            return hex(self.head)
        if method == "eth_getLogs":
            f, t = int(params[0]["fromBlock"], 16), int(params[0]["toBlock"], 16)
            if (f, t) in self.fail_ranges:
                raise RuntimeError("boom")
            return self.logs_by_range.get((f, t), [])
        if method == "eth_call":
            return "0x" + "00" * 12 + "ee" * 20
        if method == "eth_getTransactionByHash":
            return {"from": "0xAA", "to": "0xBB"}
        raise AssertionError(method)

    def batch(self, calls):
        self.batch_calls += 1
        return [{"timestamp": hex(1000 + int(p[0], 16))} for _, p in calls]


def test_sync_chunks_and_checkpoints(tmp_path):
    data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                  [1, 80, 0, "q", "", "", "", b"\x00" * 32])
    agent = "0x" + (3).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
    log = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data, block=42)
    rpc = FakeRpc({(0, 1999): [log]}, head=3999)
    with Store(tmp_path / "t.db") as store:
        n = b.sync_feedback(store, rpc, chunk=2000, start_block=0, confirmations=0)
        assert n == 1 and store.get_sync("last_block") == "3999"
        ranges = [(int(p[0]["fromBlock"], 16), int(p[0]["toBlock"], 16)) for m, p in rpc.calls if m == "eth_getLogs"]
        assert ranges == [(0, 1999), (2000, 3999)]
        first_getlogs_params = next(p for m, p in rpc.calls if m == "eth_getLogs")[0]
        assert first_getlogs_params["topics"] == [[b.TOPIC_NEW_FEEDBACK, b.TOPIC_REVOKED, b.TOPIC_RESPONSE]]
        b.fill_block_timestamps(store, rpc)
        assert store.load_records()["ts"].iloc[0] == 1042
        assert b.owner_of(rpc, "3") == "0x" + "ee" * 20
        assert b.tx_parties(rpc, "0xh") == {"0xaa", "0xbb"}


def test_first_chunk_fails_last_block_still_advances(tmp_path):
    rpc = FakeRpc({}, head=3999, fail_ranges={(0, 1999)})
    with Store(tmp_path / "t1.db") as store:
        n = b.sync_feedback(store, rpc, chunk=2000, start_block=0, confirmations=0)
        assert n == 0
        assert store.get_sync("last_block") == "3999"
        assert store.pop_failed_ranges() == [(0, 1999)]


def test_failed_range_retried_and_does_not_rewind_checkpoint(tmp_path):
    data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                  [2, 10, 0, "q", "", "", "", b"\x00" * 32])
    agent = "0x" + (5).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
    log0 = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data, block=100)

    rpc = FakeRpc({}, head=3999, fail_ranges={(0, 1999)})
    with Store(tmp_path / "t2.db") as store:
        n1 = b.sync_feedback(store, rpc, chunk=2000, start_block=0, confirmations=0)
        assert n1 == 0
        assert store.get_sync("last_block") == "3999"

        # the failed range now succeeds and has a log to deliver
        rpc.fail_ranges.clear()
        rpc.logs_by_range[(0, 1999)] = [log0]
        n2 = b.sync_feedback(store, rpc, chunk=2000, start_block=0, confirmations=0)
        ranges_called = [(int(p[0]["fromBlock"], 16), int(p[0]["toBlock"], 16))
                          for m, p in rpc.calls if m == "eth_getLogs"]
        assert (0, 1999) in ranges_called
        assert n2 == 1
        # retrying an old, low range must never rewind the checkpoint backward
        assert store.get_sync("last_block") == "3999"


def test_sync_feedback_no_op_when_checkpoint_at_head(tmp_path):
    rpc = FakeRpc({}, head=100)
    with Store(tmp_path / "t3.db") as store:
        store.set_sync("last_block", "100")
        n = b.sync_feedback(store, rpc, chunk=2000, start_block=0, confirmations=0)
        assert n == 0
        assert not any(m == "eth_getLogs" for m, _ in rpc.calls)


def test_fill_block_timestamps_no_missing_no_batch_calls(tmp_path):
    rpc = FakeRpc({})
    with Store(tmp_path / "t4.db") as store:
        n = b.fill_block_timestamps(store, rpc)
        assert n == 0
        assert rpc.batch_calls == 0


def test_fill_block_timestamps_batches_of_100(tmp_path):
    rpc = FakeRpc({})
    with Store(tmp_path / "t5.db") as store:
        rows = []
        for i in range(250):
            data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                          [i, 1, 0, "q", "", "", "", b"\x00" * 32])
            agent = "0x" + (1).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
            log = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data, block=i)
            rows.append(b.decode_log(log))
        store.upsert_feedback([r for r in rows if r["kind"] == "feedback"])
        n = b.fill_block_timestamps(store, rpc)
        assert n == 250
        assert rpc.batch_calls == 3


def test_owner_of_zero_address_returns_none():
    rpc = FakeRpc({})
    rpc.call = lambda method, params: "0x" + "00" * 32
    assert b.owner_of(rpc, "1") is None


def test_tx_parties_contract_creation_only_from():
    rpc = FakeRpc({})
    rpc.call = lambda method, params: {"from": "0xAA", "to": None}
    assert b.tx_parties(rpc, "0xh") == {"0xaa"}


def test_tx_parties_not_found_returns_none():
    rpc = FakeRpc({})
    rpc.call = lambda method, params: None
    assert b.tx_parties(rpc, "0xh") is None


# --- confirmations lag -----------------------------------------------------------

def test_sync_feedback_confirmations_lag_behind_head(tmp_path):
    rpc = FakeRpc({}, head=10_019)
    with Store(tmp_path / "t_conf.db") as store:
        n = b.sync_feedback(store, rpc, chunk=20_000, start_block=0)  # default confirmations=20
        assert n == 0
        assert store.get_sync("last_block") == "9999"


def test_config_confirmations_default_and_validation():
    assert Config().confirmations == 20
    with pytest.raises(ValueError):
        Config(confirmations=-1)


# --- crash safety of failed ranges ------------------------------------------------

def test_sync_feedback_decode_failure_raises_and_records_range(tmp_path):
    bad_log = {"topics": [b.TOPIC_NEW_FEEDBACK, "0x" + (1).to_bytes(32, "big").hex()], "data": "0x",
               "blockNumber": hex(1500), "transactionHash": "0xbad", "logIndex": "0x0"}
    rpc = FakeRpc({(0, 999): [], (1000, 1999): [bad_log]}, head=1999)
    with Store(tmp_path / "t_crash1.db") as store:
        with pytest.raises(ValueError):
            b.sync_feedback(store, rpc, chunk=1000, start_block=0, confirmations=0)
        assert store.get_sync("last_block") == "999"
        assert store.pop_failed_ranges() == [(1000, 1999)]


def test_sync_feedback_interrupt_during_fetch_requeues_remaining(tmp_path):
    with Store(tmp_path / "t_crash2.db") as store:
        store.add_failed_range(100, 199, "prior failure")
        store.add_failed_range(200, 299, "prior failure")
        store.add_failed_range(300, 399, "prior failure")
        store.set_sync("last_block", "1000")

        class InterruptingRpc(FakeRpc):
            def call(self, method, params):
                if method == "eth_getLogs":
                    f, t = int(params[0]["fromBlock"], 16), int(params[0]["toBlock"], 16)
                    self.calls.append((method, params))
                    if (f, t) == (200, 299):
                        raise KeyboardInterrupt()
                    return []
                return super().call(method, params)

        rpc = InterruptingRpc({})
        with pytest.raises(KeyboardInterrupt):
            b.sync_feedback(store, rpc, chunk=100, start_block=0, end_block=1000, confirmations=0)
        assert store.pop_failed_ranges() == [(200, 299), (300, 399)]


def test_sync_feedback_applies_feedback_revoked_and_response(tmp_path):
    fb_data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                      [1, 5, 0, "q", "", "", "", b"\x00" * 32])
    agent = "0x" + (9).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
    fb_log = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], fb_data, block=10, tx="0xfb", idx=0)

    idx_topic = "0x" + (1).to_bytes(32, "big").hex()
    rv_log = _log(b.TOPIC_REVOKED, [agent, client, idx_topic], b"", block=10, tx="0xrv", idx=1)

    resp_data = encode(["uint64", "string", "bytes32"], [1, "https://r", b"\x03" * 32])
    responder = "0x" + "00" * 12 + "cd" * 20
    resp_log = _log(b.TOPIC_RESPONSE, [agent, client, responder], resp_data, block=10, tx="0xrp", idx=2)

    rpc = FakeRpc({(0, 1999): [fb_log, rv_log, resp_log]}, head=1999)
    with Store(tmp_path / "t_combo.db") as store:
        n = b.sync_feedback(store, rpc, chunk=2000, start_block=0, confirmations=0)
        assert n == 1
        recs = store.load_records()
        assert recs["revoked"].iloc[0] == 1
        row = store.conn.execute("SELECT responder, response_uri FROM responses").fetchone()
        assert row == ("0x" + "cd" * 20, "https://r")


# --- decode_log: _hexint and topic-count validation --------------------------------

def test_decode_log_rejects_int_block_and_log_index(caplog):
    # Previously tolerated as "some clients decode these for you". A node is
    # untrusted input, so the wire form (0x-hex) is now the only accepted one:
    # anything else is skipped rather than guessed at.
    data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                  [1, 1, 0, "q", "", "", "", b"\x00" * 32])
    agent = "0x" + (2).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
    log = {"topics": [b.TOPIC_NEW_FEEDBACK, agent, client, "0x" + "00" * 32], "data": "0x" + data.hex(),
           "blockNumber": 555, "transactionHash": "0xt", "logIndex": 3}
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(log) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_decode_log_short_topics_raises_valueerror():
    agent = "0x" + (2).to_bytes(32, "big").hex()
    log = _log(b.TOPIC_NEW_FEEDBACK, [agent], b"")  # missing client + tag1 topics
    with pytest.raises(ValueError, match="NewFeedback"):
        b.decode_log(log)


def test_decode_log_empty_topics_raises_valueerror():
    log = {"topics": [], "data": "0x", "blockNumber": "0x1", "transactionHash": "0xt", "logIndex": "0x0"}
    with pytest.raises(ValueError, match="empty topics"):
        b.decode_log(log)


# --- fill_block_timestamps: batch RpcError fallback, null block ---------------------

class FlakyBatchRpc(FakeRpc):
    """Batch endpoint always fails; per-block eth_getBlockByNumber works."""

    def batch(self, calls):
        self.batch_calls += 1
        raise RpcError("batch endpoint down")

    def call(self, method, params):
        if method == "eth_getBlockByNumber":
            self.calls.append((method, params))
            blk = int(params[0], 16)
            return {"timestamp": hex(1000 + blk)}
        return super().call(method, params)


def test_fill_block_timestamps_falls_back_to_per_block_on_rpcerror(tmp_path):
    with Store(tmp_path / "t_fallback.db") as store:
        data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                      [1, 1, 0, "q", "", "", "", b"\x00" * 32])
        agent = "0x" + (1).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
        log = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data, block=7)
        store.upsert_feedback([b.decode_log(log)])
        rpc = FlakyBatchRpc({})
        n = b.fill_block_timestamps(store, rpc)
        assert n == 1
        assert store.load_records()["ts"].iloc[0] == 1007


def test_fill_block_timestamps_skips_int_timestamp_via_batch(tmp_path):
    with Store(tmp_path / "t_int_ts_batch.db") as store:
        data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                      [1, 1, 0, "q", "", "", "", b"\x00" * 32])
        agent = "0x" + (1).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
        log = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data, block=7)
        store.upsert_feedback([b.decode_log(log)])

        class IntTsRpc(FakeRpc):
            def batch(self, calls):
                self.batch_calls += 1
                return [{"timestamp": 1000 + int(p[0], 16)} for _, p in calls]  # int, not "0x..." hex

        rpc = IntTsRpc({})
        n = b.fill_block_timestamps(store, rpc)
        assert n == 0  # skipped: not the 0x-hex wire form
        assert store.load_records()["ts"].iloc[0] == 0  # no timestamp cached
        assert store.missing_block_ts() == [7]  # still retryable on the next run


def test_fill_block_timestamps_skips_int_timestamp_via_fallback(tmp_path):
    with Store(tmp_path / "t_int_ts_fallback.db") as store:
        data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                      [1, 1, 0, "q", "", "", "", b"\x00" * 32])
        agent = "0x" + (1).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
        log = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data, block=9)
        store.upsert_feedback([b.decode_log(log)])

        class IntTsFallbackRpc(FakeRpc):
            def batch(self, calls):
                self.batch_calls += 1
                raise RpcError("batch endpoint down")

            def call(self, method, params):
                if method == "eth_getBlockByNumber":
                    self.calls.append((method, params))
                    return {"timestamp": 1000 + int(params[0], 16)}  # int, not "0x..." hex
                return super().call(method, params)

        rpc = IntTsFallbackRpc({})
        n = b.fill_block_timestamps(store, rpc)
        assert n == 0  # skipped: not the 0x-hex wire form
        assert store.missing_block_ts() == [9]


def test_fill_block_timestamps_null_block_raises_rpcerror(tmp_path):
    with Store(tmp_path / "t_nullblock.db") as store:
        data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                      [1, 1, 0, "q", "", "", "", b"\x00" * 32])
        agent = "0x" + (1).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
        log = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data, block=9)
        store.upsert_feedback([b.decode_log(log)])

        class NullBlockRpc(FakeRpc):
            def batch(self, calls):
                self.batch_calls += 1
                return [None for _ in calls]

        rpc = NullBlockRpc({})
        with pytest.raises(RpcError, match="9"):
            b.fill_block_timestamps(store, rpc)


def _feedback_rows(n: int) -> list:
    rows = []
    for i in range(n):
        data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                      [i, 1, 0, "q", "", "", "", b"\x00" * 32])
        agent = "0x" + (1).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
        log = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data, block=i)
        rows.append(b.decode_log(log))
    return rows


class _StructuralOnceRpc(FakeRpc):
    """``rpc.batch()`` raises a structural ``RpcBatchStructureError`` on its
    first call only; any later call (which must never happen once
    ``BatchState`` has learned batching is unsupported) would otherwise
    succeed -- proving via ``batch_calls == 1`` that a multi-chunk caller
    never retries the doomed batch call chunk after chunk."""

    def batch(self, calls):
        self.batch_calls += 1
        if self.batch_calls == 1:
            raise RpcBatchStructureError("malformed batch response: expected 100 entries, got dict")
        return [{"timestamp": hex(1000 + int(p[0], 16))} for _, p in calls]

    def call(self, method, params):
        if method == "eth_getBlockByNumber":
            self.calls.append((method, params))
            return {"timestamp": hex(1000 + int(params[0], 16))}
        return super().call(method, params)


def test_fill_block_timestamps_structural_batch_error_stops_batching_for_rest_of_run(tmp_path, caplog):
    with Store(tmp_path / "t_structural_ts.db") as store:
        store.upsert_feedback([r for r in _feedback_rows(250) if r["kind"] == "feedback"])
        rpc = _StructuralOnceRpc({})
        with caplog.at_level(logging.WARNING, logger=b.__name__):
            n = b.fill_block_timestamps(store, rpc)  # 250 missing -> chunks of 100, 100, 50
        assert n == 250
        assert rpc.batch_calls == 1  # only the first chunk ever attempted a batch call
        assert len([c for c in rpc.calls if c[0] == "eth_getBlockByNumber"]) == 250
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1  # logged once, not once per remaining chunk
        assert "base-rpc.publicnode.com" in warnings[0].message


def test_fill_block_timestamps_batch_size_le_1_never_calls_batch(tmp_path):
    class NoBatchRpc(FakeRpc):
        def batch(self, calls):
            raise AssertionError("rpc.batch must not be called when batch_size <= 1")

        def call(self, method, params):
            if method == "eth_getBlockByNumber":
                self.calls.append((method, params))
                return {"timestamp": hex(1000 + int(params[0], 16))}
            return super().call(method, params)

    with Store(tmp_path / "t_nobatch_ts.db") as store:
        store.upsert_feedback([r for r in _feedback_rows(3) if r["kind"] == "feedback"])
        rpc = NoBatchRpc({})
        n = b.fill_block_timestamps(store, rpc, batch_size=1)
        assert n == 3
        assert len([c for c in rpc.calls if c[0] == "eth_getBlockByNumber"]) == 3


def test_fill_block_timestamps_batch_rate_limit_error_stops_batching_for_rest_of_run(tmp_path, caplog):
    class _RateLimitedOnceRpc(FakeRpc):
        def batch(self, calls):
            self.batch_calls += 1
            if self.batch_calls == 1:
                raise RpcBatchRateLimitError("batch entry id=1: [-32016] over rate limit")
            return [{"timestamp": hex(1000 + int(p[0], 16))} for _, p in calls]

        def call(self, method, params):
            if method == "eth_getBlockByNumber":
                self.calls.append((method, params))
                return {"timestamp": hex(1000 + int(params[0], 16))}
            return super().call(method, params)

    with Store(tmp_path / "t_ratelimit_ts.db") as store:
        store.upsert_feedback([r for r in _feedback_rows(150) if r["kind"] == "feedback"])
        rpc = _RateLimitedOnceRpc({})
        with caplog.at_level(logging.WARNING, logger=b.__name__):
            n = b.fill_block_timestamps(store, rpc)  # 150 missing -> chunks of 100, 50
        assert n == 150
        assert rpc.batch_calls == 1
        assert len([c for c in rpc.calls if c[0] == "eth_getBlockByNumber"]) == 150
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


# --- owner_of: revert handling ------------------------------------------------------

def test_owner_of_revert_returns_none():
    class RevertingRpc(FakeRpc):
        def call(self, method, params):
            if method == "eth_call":
                raise RpcError("execution reverted: ERC721: owner query for nonexistent token")
            return super().call(method, params)

    rpc = RevertingRpc({})
    assert b.owner_of(rpc, "999999") is None


def test_owner_of_other_rpcerror_propagates():
    class FailingRpc(FakeRpc):
        def call(self, method, params):
            if method == "eth_call":
                raise RpcError("connection reset")
            return super().call(method, params)

    rpc = FailingRpc({})
    with pytest.raises(RpcError):
        b.owner_of(rpc, "1")


# --- owners_of: batched owner resolution ---------------------------------------

def _owner_hex(addr: str) -> str:
    """Encode a 20-byte ``0x``-address as a 32-byte-padded eth_call result hex string."""
    return "0x" + "00" * 12 + addr[2:]


class OwnerBatchRpc(FakeRpc):
    """FakeRpc extended with an eth_call-aware batch()/call() for owners_of.

    ``owners`` maps agent id -> owner address; an id absent from it (or mapped
    to ``None``) resolves to the zero address. ``revert_agents`` names ids
    whose ``eth_call`` reverts -- inside a batch, per RpcClient.batch's
    contract, one reverting entry fails the *whole* batch call, not just that
    entry. ``fail_batch`` makes every ``batch()`` call raise ``RpcError``
    outright (mimicking an endpoint that doesn't support batching, or returns
    a malformed response, e.g. a dict instead of a list).
    """

    def __init__(self, owners, revert_agents=(), fail_batch=False):
        super().__init__({})
        self.owners = owners
        self.revert_agents = set(revert_agents)
        self.fail_batch = fail_batch
        self.batch_sizes = []

    @staticmethod
    def _agent_id_from_data(data: str) -> str:
        return str(int(data[len(b._OWNER_OF_SELECTOR):], 16))

    def _result_for(self, agent_id: str) -> str:
        owner = self.owners.get(agent_id)
        return _owner_hex(owner) if owner else "0x" + "00" * 32

    def batch(self, calls):
        self.batch_calls += 1
        self.batch_sizes.append(len(calls))
        if self.fail_batch:
            raise RpcError("batch endpoint down")
        results = []
        for _method, params in calls:
            agent_id = self._agent_id_from_data(params[0]["data"])
            if agent_id in self.revert_agents:
                raise RpcError(f"execution reverted for agent {agent_id}")
            results.append(self._result_for(agent_id))
        return results

    def call(self, method, params):
        if method == "eth_call":
            self.calls.append((method, params))
            agent_id = self._agent_id_from_data(params[0]["data"])
            if agent_id in self.revert_agents:
                raise RpcError("execution reverted: nonexistent token")
            return self._result_for(agent_id)
        return super().call(method, params)


def test_owners_of_decodes_owner_and_zero_address_from_one_batch():
    rpc = OwnerBatchRpc({"1": "0x" + "11" * 20, "2": None})
    out = b.owners_of(rpc, ["1", "2"])
    assert out == {"1": "0x" + "11" * 20, "2": None}
    assert rpc.batch_calls == 1


def test_owners_of_reverting_entry_falls_back_to_per_agent(caplog):
    rpc = OwnerBatchRpc({"1": "0x" + "11" * 20, "3": None}, revert_agents={"2"})
    with caplog.at_level(logging.WARNING, logger=b.__name__):
        out = b.owners_of(rpc, ["1", "2", "3"])
    assert out == {"1": "0x" + "11" * 20, "2": None, "3": None}  # revert -> None via owner_of fallback
    assert rpc.batch_calls == 1  # one failed batch attempt covering the whole chunk
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # logged once, not once per fallen-back agent
    assert len([c for c in rpc.calls if c[0] == "eth_call"]) == 3  # fallback queried every agent in the chunk


def test_owners_of_batch_rpcerror_falls_back_and_returns_full_dict(caplog):
    rpc = OwnerBatchRpc({"1": "0x" + "11" * 20, "2": "0x" + "22" * 20}, fail_batch=True)
    with caplog.at_level(logging.WARNING, logger=b.__name__):
        out = b.owners_of(rpc, ["1", "2"])
    assert out == {"1": "0x" + "11" * 20, "2": "0x" + "22" * 20}
    assert rpc.batch_calls == 1
    assert len([c for c in rpc.calls if c[0] == "eth_call"]) == 2
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_owners_of_chunks_250_ids_into_3_batches():
    owners = {str(i): "0x" + f"{i % 100:02x}" * 20 for i in range(250)}
    rpc = OwnerBatchRpc(owners)
    out = b.owners_of(rpc, [str(i) for i in range(250)])
    assert len(out) == 250
    assert rpc.batch_calls == 3
    assert rpc.batch_sizes == [100, 100, 50]


# --- owners_of: batching permanently abandoned after one structural/rate-limit error --

class _StructuralOnceOwnerRpc(OwnerBatchRpc):
    """First ``rpc.batch()`` call raises a structural ``RpcBatchStructureError``;
    any later call (which must never happen once ``BatchState`` has learned
    batching is unsupported) would otherwise succeed -- proving via
    ``batch_calls == 1`` that ``owners_of`` never retries the doomed batch
    call chunk after chunk."""

    def batch(self, calls):
        self.batch_calls += 1
        if self.batch_calls == 1:
            raise RpcBatchStructureError("malformed batch response: expected 100 entries, got dict")
        return [self._result_for(self._agent_id_from_data(p[0]["data"])) for _, p in calls]


def test_owners_of_structural_batch_error_stops_batching_for_rest_of_run(caplog):
    owners = {str(i): "0x" + f"{i % 100:02x}" * 20 for i in range(250)}
    rpc = _StructuralOnceOwnerRpc(owners)
    with caplog.at_level(logging.WARNING, logger=b.__name__):
        out = b.owners_of(rpc, [str(i) for i in range(250)])  # chunks of 100, 100, 50
    assert len(out) == 250
    assert rpc.batch_calls == 1  # only the first chunk ever attempted a batch call
    assert len([c for c in rpc.calls if c[0] == "eth_call"]) == 250  # rest resolved per-agent
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # logged once, not once per remaining chunk
    assert "base-rpc.publicnode.com" in warnings[0].message


def test_owners_of_batch_rate_limit_error_stops_batching_for_rest_of_run(caplog):
    class _RateLimitedOnceRpc(OwnerBatchRpc):
        def batch(self, calls):
            self.batch_calls += 1
            if self.batch_calls == 1:
                raise RpcBatchRateLimitError("batch entry id=2: [-32016] over rate limit")
            return [self._result_for(self._agent_id_from_data(p[0]["data"])) for _, p in calls]

    owners = {"1": "0x" + "11" * 20, "2": "0x" + "22" * 20, "3": "0x" + "33" * 20}
    rpc = _RateLimitedOnceRpc(owners)
    with caplog.at_level(logging.WARNING, logger=b.__name__):
        out = b.owners_of(rpc, ["1", "2", "3"], batch_size=2)  # chunks: [1,2], [3]
    assert out == owners
    assert rpc.batch_calls == 1
    assert len([c for c in rpc.calls if c[0] == "eth_call"]) == 3
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_owners_of_ordinary_batch_error_does_not_disable_batching_for_later_chunks():
    # a per-call revert (an ordinary RpcError, not RpcBatchUnsupportedError)
    # must only fall back for the chunk it hit -- later chunks still try
    # rpc.batch normally.
    rpc = OwnerBatchRpc({"1": "0x" + "11" * 20, "3": "0x" + "33" * 20, "4": "0x" + "44" * 20},
                         revert_agents={"2"})
    out = b.owners_of(rpc, ["1", "2", "3", "4"], batch_size=2)
    assert out == {"1": "0x" + "11" * 20, "2": None, "3": "0x" + "33" * 20, "4": "0x" + "44" * 20}
    assert rpc.batch_calls == 2  # second chunk's batch was still attempted


def test_owners_of_batch_size_le_1_never_calls_batch():
    class NoBatchRpc(OwnerBatchRpc):
        def batch(self, calls):
            raise AssertionError("rpc.batch must not be called when batch_size <= 1")

    rpc = NoBatchRpc({"1": "0x" + "11" * 20, "2": None})
    out = b.owners_of(rpc, ["1", "2"], batch_size=1)
    assert out == {"1": "0x" + "11" * 20, "2": None}
    assert len([c for c in rpc.calls if c[0] == "eth_call"]) == 2


# --- L4: malformed topics / data / agent ids ----------------------------------------


def _agent_topic(n=1):
    return "0x" + (n).to_bytes(32, "big").hex()


_CLIENT_TOPIC = "0x" + "00" * 12 + "ab" * 20


def test_decode_log_short_topic_returns_none_with_warning(caplog):
    # A truncated indexed topic would silently yield a short "address" --
    # skip the log instead (and never crash the whole sync over it).
    log = _log(b.TOPIC_REVOKED, ["0xdead", _CLIENT_TOPIC, _agent_topic(9)], b"")
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(log) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "0xdead" in warnings[0].getMessage()


def test_decode_log_over_long_topic_returns_none():
    log = _log(b.TOPIC_REVOKED, ["0x" + "ff" * 40, _CLIENT_TOPIC, _agent_topic(9)], b"")
    assert b.decode_log(log) is None


def test_decode_log_non_hex_topic_returns_none():
    log = _log(b.TOPIC_REVOKED, ["0x" + "zz" * 32, _CLIENT_TOPIC, _agent_topic(9)], b"")
    assert b.decode_log(log) is None


def test_decode_log_non_string_topic_returns_none():
    log = _log(b.TOPIC_REVOKED, [12345, _CLIENT_TOPIC, _agent_topic(9)], b"")
    assert b.decode_log(log) is None


def test_decode_log_non_string_topic0_returns_none():
    log = {"topics": [12345], "data": "0x", "blockNumber": "0x1", "transactionHash": "0xt",
           "logIndex": "0x0"}
    assert b.decode_log(log) is None


def test_decode_log_non_hex_data_returns_none_with_warning(caplog):
    log = _log(b.TOPIC_RESPONSE, [_agent_topic(), _CLIENT_TOPIC, _CLIENT_TOPIC], b"")
    log["data"] = "0xnothexatall"
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(log) is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_decode_log_odd_length_data_returns_none():
    log = _log(b.TOPIC_RESPONSE, [_agent_topic(), _CLIENT_TOPIC, _CLIENT_TOPIC], b"")
    log["data"] = "0xabc"
    assert b.decode_log(log) is None


def test_decode_log_warning_truncates_a_huge_topic(caplog):
    log = _log(b.TOPIC_REVOKED, ["0x" + "a" * 100_000, _CLIENT_TOPIC, _agent_topic(9)], b"")
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(log) is None
    assert len(caplog.text) < 2000


def test_addr_rejects_a_topic_that_is_not_an_address():
    with pytest.raises(ValueError):
        b._addr("0xnothex")


def test_addr_accepts_a_valid_topic():
    assert b._addr(_CLIENT_TOPIC) == "0x" + "ab" * 20


def test_owner_call_data_rejects_an_over_range_agent_id():
    with pytest.raises(ValueError):
        b._owner_call_data(str(2 ** 256))


def test_owner_call_data_rejects_a_non_decimal_agent_id():
    with pytest.raises(ValueError):
        b._owner_call_data("0xdeadbeef")


def test_owner_call_data_rejects_a_negative_agent_id():
    with pytest.raises(ValueError):
        b._owner_call_data("-1")


def test_owner_call_data_accepts_the_max_uint256_id():
    data = b._owner_call_data(str(2 ** 256 - 1))
    assert data == b._OWNER_OF_SELECTOR + "ff" * 32


def test_owner_of_skips_a_malformed_agent_id_without_calling_rpc(caplog):
    class NoCallRpc:
        def call(self, method, params):
            raise AssertionError("must not reach the RPC for a malformed agent id")

    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.owner_of(NoCallRpc(), str(2 ** 256)) is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_owners_of_skips_malformed_ids_and_still_resolves_the_rest():
    class BatchRpc:
        def __init__(self):
            self.batched = []

        def batch(self, calls):
            self.batched.append(calls)
            return ["0x" + "00" * 12 + "ee" * 20] * len(calls)

        def call(self, method, params):
            raise AssertionError("batch path only")

    rpc = BatchRpc()
    out = b.owners_of(rpc, ["1", str(2 ** 256), "2"], batch_size=10)
    assert out["1"] == "0x" + "ee" * 20 and out["2"] == "0x" + "ee" * 20
    assert out[str(2 ** 256)] is None
    assert len(rpc.batched[0]) == 2  # the malformed id never entered the batch


def test_owners_of_all_ids_malformed_never_batches():
    class NeverRpc:
        def batch(self, calls):
            raise AssertionError("must not batch when every id is malformed")

        def call(self, method, params):
            raise AssertionError("must not call when every id is malformed")

    assert b.owners_of(NeverRpc(), ["0xzz", str(2 ** 256)], batch_size=10) == {"0xzz": None,
                                                                               str(2 ** 256): None}


# --- I1: a log whose data does not ABI-decode is skipped, not fatal -----------------


def test_new_feedback_with_empty_data_returns_none_with_one_warning(caplog):
    # eth_abi raises InsufficientDataBytes (not ValueError) here, and
    # sync_feedback treats a decode exception as a bug worth aborting the whole
    # run for -- so anyone able to emit a 2-byte log could stop the pipeline.
    log = _log(b.TOPIC_NEW_FEEDBACK, [_agent_topic(), _CLIENT_TOPIC, _agent_topic(0)], b"")
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(log) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "0xt" in warnings[0].getMessage()


def test_response_with_empty_data_returns_none_with_one_warning(caplog):
    log = _log(b.TOPIC_RESPONSE, [_agent_topic(), _CLIENT_TOPIC, _CLIENT_TOPIC], b"")
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(log) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_new_feedback_with_truncated_data_returns_none(caplog):
    log = _log(b.TOPIC_NEW_FEEDBACK, [_agent_topic(), _CLIENT_TOPIC, _agent_topic(0)], b"\x00" * 32)
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(log) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_revoked_with_empty_data_still_decodes():
    # FeedbackRevoked carries no ABI data at all (every field is an indexed
    # topic), so "0x" is its normal shape -- it must not be skipped.
    log = _log(b.TOPIC_REVOKED, [_agent_topic(), _CLIENT_TOPIC, _agent_topic(9)], b"")
    assert b.decode_log(log)["feedback_index"] == 9


def test_decode_log_never_raises_the_eth_abi_error():
    log = _log(b.TOPIC_NEW_FEEDBACK, [_agent_topic(), _CLIENT_TOPIC, _agent_topic(0)], b"\x01")
    assert b.decode_log(log) is None


def test_sync_feedback_continues_past_an_undecodable_log(tmp_path):
    good_data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                       [1, 5, 0, "q", "", "", "", b"\x00" * 32])
    bad = _log(b.TOPIC_NEW_FEEDBACK, [_agent_topic(1), _CLIENT_TOPIC, _agent_topic(0)], b"")
    good = _log(b.TOPIC_NEW_FEEDBACK, [_agent_topic(2), _CLIENT_TOPIC, _agent_topic(0)], good_data, idx=1)
    rpc = FakeRpc({(1, 10): [bad, good]}, head=10)
    with Store(tmp_path / "t.db") as s:
        n = b.sync_feedback(s, rpc, chunk=10, start_block=1, end_block=10)
        assert n == 1  # the good log landed; the undecodable one was skipped
        assert s.get_sync("last_block") == "10"  # checkpoint advanced, no abort
        assert s.pop_failed_ranges() == []


# --- I3: eth_call owner results are validated before being stored -------------------


def test_decode_owner_result_accepts_a_padded_word():
    assert b._decode_owner_result("0x" + "00" * 12 + "ee" * 20) == "0x" + "ee" * 20


def test_decode_owner_result_rejects_a_garbage_string(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b._decode_owner_result("garbage-from-a-hostile-node-xxxxxxxxxxxxxxxxxxx") is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_decode_owner_result_rejects_a_non_string(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b._decode_owner_result(12345) is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_decode_owner_result_keeps_none_and_empty_result_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b._decode_owner_result(None) is None
        assert b._decode_owner_result("0x") is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_owners_of_never_stores_a_garbage_owner():
    class GarbageRpc:
        def batch(self, calls):
            return ["not-an-address"] * len(calls)

    assert b.owners_of(GarbageRpc(), ["1", "2"], batch_size=10) == {"1": None, "2": None}


def test_owner_call_data_rejects_padded_or_separated_digits():
    for bad in (" 12 ", "1_0", "+7", "12.0", "0b11"):
        with pytest.raises(ValueError):
            b._owner_call_data(bad)


# --- strict hex parsing of node-supplied quantities ---------------------------------


def _feedback_log(agent=2, **overrides):
    """A well-formed NewFeedback log; ``agent`` varies the (agent_id, client,
    feedback_index) primary key so several can coexist in one store."""
    data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                  [1, 1, 0, "q", "", "", "", b"\x00" * 32])
    log = _log(b.TOPIC_NEW_FEEDBACK, [_agent_topic(agent), _CLIENT_TOPIC, _agent_topic(0)], data)
    log.update(overrides)
    return log


@pytest.mark.parametrize("bad_block", [
    float("inf"),  # what JSON's 1e400 decodes to: int(inf, 16) raises TypeError
    float("nan"),
    12,            # already-decoded int: not the wire form
    "12",          # decimal string
    "0xZZ",        # not hex
    "0x",          # no digits
    "1500",
    None,
    ["0x1"],
    "0x" + "f" * 65,  # wider than a uint256
])
def test_decode_log_rejects_a_non_hex_block_number(bad_block, caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(_feedback_log(blockNumber=bad_block)) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


@pytest.mark.parametrize("bad_index", [float("inf"), 3, "3", "0xZZ", None])
def test_decode_log_rejects_a_non_hex_log_index(bad_index, caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(_feedback_log(logIndex=bad_index)) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_decode_log_warning_does_not_echo_a_huge_block_number(caplog):
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        assert b.decode_log(_feedback_log(blockNumber="0x" + "9" * 100_000)) is None
    assert len(caplog.text) < 2000


def test_decode_log_accepts_an_uppercase_hex_quantity():
    assert b.decode_log(_feedback_log(blockNumber="0xABC"))["block"] == 0xABC


def test_hexint_accepts_the_wire_form():
    assert b._hexint("0x2dfccac") == 0x2dfccac
    assert b._hexint("0x0") == 0


@pytest.mark.parametrize("bad", [12, "12", "0xZZ", None, float("inf"), b"0x1", True])
def test_hexint_rejects_everything_else(bad):
    with pytest.raises(ValueError):
        b._hexint(bad)


def test_sync_feedback_continues_past_a_log_with_a_bad_block_number(tmp_path):
    bad = _feedback_log(blockNumber=float("inf"))
    good = _feedback_log(agent=3, logIndex="0x1")
    rpc = FakeRpc({(1, 10): [bad, good]}, head=10)
    with Store(tmp_path / "t.db") as s:
        n = b.sync_feedback(s, rpc, chunk=10, start_block=1, end_block=10)
        assert n == 1
        assert s.get_sync("last_block") == "10"
        assert s.pop_failed_ranges() == []


def test_sync_feedback_rejects_a_malformed_chain_head(tmp_path):
    class BadHeadRpc(FakeRpc):
        def call(self, method, params):
            if method == "eth_blockNumber":
                return 12345  # int, not the "0x..." wire form
            return super().call(method, params)

    with Store(tmp_path / "t.db") as s:
        with pytest.raises(RpcError):
            b.sync_feedback(s, BadHeadRpc({}), chunk=10, start_block=1)


def test_sync_feedback_rejects_a_non_hex_chain_head(tmp_path):
    class BadHeadRpc(FakeRpc):
        def call(self, method, params):
            if method == "eth_blockNumber":
                return "0xZZZZ"
            return super().call(method, params)

    with Store(tmp_path / "t.db") as s:
        with pytest.raises(RpcError):
            b.sync_feedback(s, BadHeadRpc({}), chunk=10, start_block=1)


def test_fill_block_timestamps_skips_a_missing_timestamp_field(tmp_path):
    with Store(tmp_path / "t.db") as store:
        store.upsert_feedback([b.decode_log(_feedback_log(blockNumber="0x5"))])

        class NoTsRpc(FakeRpc):
            def batch(self, calls):
                self.batch_calls += 1
                return [{"number": "0x5"} for _ in calls]

        assert b.fill_block_timestamps(store, NoTsRpc({})) == 0
        assert store.missing_block_ts() == [5]


def test_fill_block_timestamps_skips_a_non_dict_block(tmp_path):
    with Store(tmp_path / "t.db") as store:
        store.upsert_feedback([b.decode_log(_feedback_log(blockNumber="0x6"))])

        class StringBlockRpc(FakeRpc):
            def batch(self, calls):
                self.batch_calls += 1
                return ["not-a-block" for _ in calls]

        assert b.fill_block_timestamps(store, StringBlockRpc({})) == 0
        assert store.missing_block_ts() == [6]


def test_fill_block_timestamps_caches_the_good_blocks_in_a_mixed_batch(tmp_path, caplog):
    with Store(tmp_path / "t.db") as store:
        store.upsert_feedback([b.decode_log(_feedback_log(blockNumber="0x7")),
                               b.decode_log(_feedback_log(agent=3, blockNumber="0x8", logIndex="0x1"))])

        class MixedRpc(FakeRpc):
            def batch(self, calls):
                self.batch_calls += 1
                return [{"timestamp": hex(1000 + int(p[0], 16))} if int(p[0], 16) == 7
                        else {"timestamp": 1008} for _, p in calls]

        with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
            assert b.fill_block_timestamps(store, MixedRpc({})) == 1
        assert store.missing_block_ts() == [8]
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
