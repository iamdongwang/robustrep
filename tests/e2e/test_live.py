"""Live (network-touching) end-to-end checks. Marked ``live`` -- excluded from
the default test run (``pyproject.toml``'s ``addopts = "-m 'not live'"``);
run explicitly with ``pytest -m live tests/e2e``.
"""
import json
from pathlib import Path

import pytest

from robustrep.config import Config
from robustrep.sources import base_erc8004 as base
from robustrep.sources.rpc import RpcClient
from robustrep.store import Store

FIX = json.loads((Path(__file__).parent / "fixtures/base_logs_2000.json").read_text())


@pytest.mark.live
def test_live_recent_blocks(tmp_path):
    cfg = Config()
    rpc = RpcClient(cfg.rpc_urls, user_agent=cfg.user_agent)
    head = int(rpc.call("eth_blockNumber", []), 16)
    store = Store(tmp_path / "live.db")
    base.sync_feedback(store, rpc, chunk=2000, start_block=head - 3999, end_block=head)
    assert store.get_sync("last_block") == str(head)
    base.fill_block_timestamps(store, rpc)
    assert store.missing_block_ts() == []


@pytest.mark.live
def test_live_fixture_window_matches_recorded_fixture():
    """Drift detector: the fixture window (``tests/e2e/fixtures/base_logs_2000.json``)
    is deep in finalized history (well over ``confirmations`` blocks old), so
    replaying the exact same ``[from_block, to_block]`` against the live node
    should reproduce the identical set of ``NewFeedback`` events every time.

    If this ever fails, either the chain re-organized at a depth far beyond
    any realistic confirmation lag, or the RPC provider is serving bad data --
    either way, worth knowing about rather than silently trusting a stale
    fixture forever. The fixture is treated as final: no tolerance for
    newly-appeared ``ResponseAppended``/``FeedbackRevoked`` logs in the window,
    since ERC-8004 responses/revocations for already-final feedback would
    themselves be evidence of exactly the kind of drift this test is meant to
    catch.
    """
    cfg = Config()
    rpc = RpcClient(cfg.rpc_urls, user_agent=cfg.user_agent)
    logs = rpc.call("eth_getLogs", [{
        "fromBlock": hex(FIX["from_block"]), "toBlock": hex(FIX["to_block"]),
        "address": base.REPUTATION_REGISTRY,
        "topics": [[base.TOPIC_NEW_FEEDBACK, base.TOPIC_REVOKED, base.TOPIC_RESPONSE]],
    }])

    def feedback_keys(raw_logs):
        decoded = [base.decode_log(l) for l in raw_logs]
        return {(d["agent_id"], d["client"], d["feedback_index"])
                for d in decoded if d and d["kind"] == "feedback"}

    live_new_feedback = sum(1 for l in logs if l["topics"][0].lower() == base.TOPIC_NEW_FEEDBACK)
    fixture_new_feedback = sum(1 for l in FIX["logs"] if l["topics"][0].lower() == base.TOPIC_NEW_FEEDBACK)
    assert live_new_feedback == fixture_new_feedback

    assert feedback_keys(logs) == feedback_keys(FIX["logs"])
