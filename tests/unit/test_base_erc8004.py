import pytest
from eth_abi import encode

from robustrep.sources import base_erc8004 as b
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
    store = Store(tmp_path / "t.db")
    n = b.sync_feedback(store, rpc, chunk=2000, start_block=0)
    assert n == 1 and store.get_sync("last_block") == "3999"
    ranges = [(int(p[0]["fromBlock"], 16), int(p[0]["toBlock"], 16)) for m, p in rpc.calls if m == "eth_getLogs"]
    assert ranges == [(0, 1999), (2000, 3999)]
    b.fill_block_timestamps(store, rpc)
    assert store.load_records()["ts"].iloc[0] == 1042
    assert b.owner_of(rpc, "3") == "0x" + "ee" * 20
    assert b.tx_parties(rpc, "0xh") == {"0xaa", "0xbb"}


def test_first_chunk_fails_last_block_still_advances(tmp_path):
    rpc = FakeRpc({}, head=3999, fail_ranges={(0, 1999)})
    store = Store(tmp_path / "t1.db")
    n = b.sync_feedback(store, rpc, chunk=2000, start_block=0)
    assert n == 0
    assert store.get_sync("last_block") == "3999"
    assert store.pop_failed_ranges() == [(0, 1999)]


def test_failed_range_retried_and_does_not_rewind_checkpoint(tmp_path):
    data = encode(["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"],
                  [2, 10, 0, "q", "", "", "", b"\x00" * 32])
    agent = "0x" + (5).to_bytes(32, "big").hex(); client = "0x" + "00" * 12 + "ab" * 20
    log0 = _log(b.TOPIC_NEW_FEEDBACK, [agent, client, "0x" + "00" * 32], data, block=100)

    rpc = FakeRpc({}, head=3999, fail_ranges={(0, 1999)})
    store = Store(tmp_path / "t2.db")
    n1 = b.sync_feedback(store, rpc, chunk=2000, start_block=0)
    assert n1 == 0
    assert store.get_sync("last_block") == "3999"

    # the failed range now succeeds and has a log to deliver
    rpc.fail_ranges.clear()
    rpc.logs_by_range[(0, 1999)] = [log0]
    n2 = b.sync_feedback(store, rpc, chunk=2000, start_block=0)
    ranges_called = [(int(p[0]["fromBlock"], 16), int(p[0]["toBlock"], 16))
                      for m, p in rpc.calls if m == "eth_getLogs"]
    assert (0, 1999) in ranges_called
    assert n2 == 1
    # retrying an old, low range must never rewind the checkpoint backward
    assert store.get_sync("last_block") == "3999"


def test_sync_feedback_no_op_when_checkpoint_at_head(tmp_path):
    rpc = FakeRpc({}, head=100)
    store = Store(tmp_path / "t3.db")
    store.set_sync("last_block", "100")
    n = b.sync_feedback(store, rpc, chunk=2000, start_block=0)
    assert n == 0
    assert not any(m == "eth_getLogs" for m, _ in rpc.calls)


def test_fill_block_timestamps_no_missing_no_batch_calls(tmp_path):
    rpc = FakeRpc({})
    store = Store(tmp_path / "t4.db")
    n = b.fill_block_timestamps(store, rpc)
    assert n == 0
    assert rpc.batch_calls == 0


def test_fill_block_timestamps_batches_of_100(tmp_path):
    rpc = FakeRpc({})
    store = Store(tmp_path / "t5.db")
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
