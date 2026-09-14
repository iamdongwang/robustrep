"""Record 2,000 blocks of real ReputationRegistry logs + their block timestamps into a fixture.

Usage: python scripts/record_fixture.py [to_block]

Fetches ``eth_getLogs`` for the ReputationRegistry's three known event topics
over ``[to_block - 1999, to_block]`` (default: ``head - 100``, i.e. a
comfortably final window) plus the timestamp of every block referenced by a
returned log, and writes both into ``tests/e2e/fixtures/base_logs_2000.json``
for the offline replay tests in ``tests/e2e/test_replay.py``.

``rpc.batch`` of ``eth_getBlockByNumber`` has, against the public
``mainnet.base.org`` node, sometimes returned a single dict instead of a list
of results (a malformed-batch response ``RpcClient.batch`` itself already
raises ``RpcError`` for) or otherwise failed; on any such failure this script
falls back to resolving each block's timestamp with an individual
``eth_getBlockByNumber`` ``rpc.call`` instead of aborting the recording.
"""
import json
import sys
from pathlib import Path

from robustrep.config import Config
from robustrep.sources import base_erc8004 as base
from robustrep.sources.rpc import RpcClient, RpcError

cfg = Config()
rpc = RpcClient(cfg.rpc_urls, user_agent=cfg.user_agent)
head = int(rpc.call("eth_blockNumber", []), 16)
to_block = int(sys.argv[1]) if len(sys.argv) > 1 else head - 100
from_block = to_block - 1999
logs = rpc.call("eth_getLogs", [{"fromBlock": hex(from_block), "toBlock": hex(to_block),
                                 "address": base.REPUTATION_REGISTRY,
                                 "topics": [[base.TOPIC_NEW_FEEDBACK, base.TOPIC_REVOKED, base.TOPIC_RESPONSE]]}])
blocks = sorted({int(l["blockNumber"], 16) for l in logs})
ts = {}
for i in range(0, len(blocks), 100):
    chunk = blocks[i:i + 100]
    try:
        results = rpc.batch([("eth_getBlockByNumber", [hex(b), False]) for b in chunk])
    except RpcError as e:
        print(f"batch fetch failed ({e}); falling back to per-block calls", file=sys.stderr)
        results = [rpc.call("eth_getBlockByNumber", [hex(b), False]) for b in chunk]
    for b, blk in zip(chunk, results):
        ts[b] = int(blk["timestamp"], 16)

out = Path(__file__).resolve().parents[1] / "tests/e2e/fixtures/base_logs_2000.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"from_block": from_block, "to_block": to_block, "logs": logs, "block_ts": ts}))

counts: dict[str, int] = {}
for l in logs:
    t0 = l["topics"][0].lower()
    kind = {base.TOPIC_NEW_FEEDBACK: "NewFeedback", base.TOPIC_REVOKED: "FeedbackRevoked",
            base.TOPIC_RESPONSE: "ResponseAppended"}.get(t0, "unknown")
    counts[kind] = counts.get(kind, 0) + 1

size = out.stat().st_size
print(f"recorded {len(logs)} logs over blocks {from_block}-{to_block} -> {out}")
print(f"by kind: {counts}")
print(f"file size: {size} bytes ({size / 1_000_000:.2f} MB)")
if size > 2_000_000:
    print("WARNING: fixture is >= 2 MB; re-record with an earlier/quieter to_block", file=sys.stderr)
