"""ERC-8004 ReputationRegistry on Base: decode events, sync into Store, resolve helpers.

``sync_feedback`` walks ``eth_getLogs`` in fixed-size block-number chunks from the
last checkpoint (``Store.get_sync("last_block")``) to the chain head, decoding and
persisting ``NewFeedback``/``FeedbackRevoked``/``ResponseAppended`` events as it
goes. A chunk whose ``eth_getLogs`` call fails is recorded via
``Store.add_failed_range`` instead of aborting the whole sync, and does **not**
advance the checkpoint past it; chunks are otherwise independent, so later chunks
in the same run still advance the checkpoint even if an earlier one failed. Failed
ranges are retried (and popped) at the start of the *next* ``sync_feedback`` call.
Because retried ranges and freshly-computed ranges can be processed in either
order, the checkpoint is advanced with ``max(current, chunk_end)`` rather than a
plain overwrite, so retrying an old failed range can never rewind the cursor.
"""
from __future__ import annotations

import logging
from typing import Optional

from eth_abi import decode
from eth_hash.auto import keccak

from ..store import Store

logger = logging.getLogger(__name__)

CHAIN = "base"
REPUTATION_REGISTRY = "0x8004BAa17C55a88189AE136b182e5fdA19dE9b63"
IDENTITY_REGISTRY = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
DEPLOY_BLOCK = 41663784
ZERO_ADDRESS = "0x" + "00" * 20

_OWNER_OF_SELECTOR = "0x6352211e"


def _topic(sig: str) -> str:
    return "0x" + keccak(sig.encode()).hex()


TOPIC_NEW_FEEDBACK = _topic("NewFeedback(uint256,address,uint64,int128,uint8,string,string,string,string,string,bytes32)")
TOPIC_REVOKED = _topic("FeedbackRevoked(uint256,address,uint64)")
TOPIC_RESPONSE = _topic("ResponseAppended(uint256,address,uint64,address,string,bytes32)")
FEEDBACK_TYPES = ["uint64", "int128", "uint8", "string", "string", "string", "string", "bytes32"]
RESPONSE_TYPES = ["uint64", "string", "bytes32"]

_REQUIRED_LOG_FIELDS = ("topics", "blockNumber", "transactionHash", "logIndex")


def _addr(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def _uint(topic: str) -> int:
    return int(topic, 16)


def decode_log(log: dict) -> Optional[dict]:
    """Decode one ``eth_getLogs`` entry from the ReputationRegistry.

    Returns a dict with a ``kind`` of ``"feedback"``, ``"revoked"`` or
    ``"response"`` (plus the event's fields and the base ``chain``/``block``/
    ``tx_hash``/``log_index`` fields), or ``None`` if ``log``'s ``topics[0]``
    does not match any known event.

    Raises ``ValueError`` naming the missing field if ``log`` lacks one of the
    four fields every log is expected to carry: ``topics``, ``blockNumber``,
    ``transactionHash``, ``logIndex``.
    """
    for field in _REQUIRED_LOG_FIELDS:
        if field not in log:
            raise ValueError(f"decode_log: log missing {field!r}")
    t0, topics = log["topics"][0].lower(), log["topics"]
    base = dict(chain=CHAIN, block=int(log["blockNumber"], 16), tx_hash=log["transactionHash"],
                log_index=int(log["logIndex"], 16))
    data = bytes.fromhex(log["data"][2:]) if log.get("data", "0x") != "0x" else b""
    if t0 == TOPIC_NEW_FEEDBACK:
        fi, val, dec, tag1, tag2, endpoint, uri, h = decode(FEEDBACK_TYPES, data)
        return {**base, "kind": "feedback", "agent_id": str(_uint(topics[1])), "client": _addr(topics[2]),
                "feedback_index": fi, "value": str(val), "value_decimals": dec, "tag1": tag1, "tag2": tag2,
                "endpoint": endpoint, "feedback_uri": uri, "feedback_hash": "0x" + h.hex()}
    if t0 == TOPIC_REVOKED:
        return {**base, "kind": "revoked", "agent_id": str(_uint(topics[1])), "client": _addr(topics[2]),
                "feedback_index": _uint(topics[3])}
    if t0 == TOPIC_RESPONSE:
        fi, uri, h = decode(RESPONSE_TYPES, data)
        return {**base, "kind": "response", "agent_id": str(_uint(topics[1])), "client": _addr(topics[2]),
                "responder": _addr(topics[3]), "feedback_index": fi, "response_uri": uri,
                "response_hash": "0x" + h.hex()}
    return None


def _apply(store: Store, decoded: list[dict]) -> int:
    """Persist a batch of decoded events into ``store``; returns the number of
    ``feedback`` events in the batch (not the number actually inserted, since
    duplicates are silently ignored by ``Store.upsert_feedback``)."""
    fb = [d for d in decoded if d["kind"] == "feedback"]
    store.upsert_feedback(fb)
    for d in decoded:
        if d["kind"] == "revoked":
            store.mark_revoked(d["agent_id"], d["client"], d["feedback_index"], block=d["block"], tx_hash=d["tx_hash"])
        elif d["kind"] == "response":
            store.add_response(d)
    return len(fb)


def _advance_checkpoint(store: Store, chunk_end: int) -> None:
    cur = store.get_sync("last_block")
    new_val = chunk_end if cur is None else max(int(cur), chunk_end)
    store.set_sync("last_block", str(new_val))


def sync_feedback(store: Store, rpc, chunk: int = 2000, start_block: int = DEPLOY_BLOCK,
                   end_block: Optional[int] = None) -> int:
    """Pull NewFeedback/FeedbackRevoked/ResponseAppended from the last checkpoint
    (``store.get_sync("last_block")``) up to ``end_block`` (defaults to the chain
    head) and persist them into ``store``. Returns the number of feedback rows
    decoded across all chunks (see ``_apply`` for what "decoded" means).

    Block ranges are walked in ``chunk``-sized windows (Base's ``eth_getLogs`` is
    limited to 2,000 blocks per call). Any previously failed ranges (see
    ``Store.add_failed_range``) are retried first. A chunk whose ``eth_getLogs``
    call raises is recorded as a new failed range and skipped -- it does not stop
    the sync and does not advance the checkpoint -- but later chunks still do (see
    module docstring for the checkpoint-advance ordering guarantee). If the
    checkpoint is already at or past ``end_block``, no RPC calls are made and 0
    is returned.
    """
    head = end_block if end_block is not None else int(rpc.call("eth_blockNumber", []), 16)
    last = store.get_sync("last_block")
    frm = int(last) + 1 if last is not None else start_block
    ranges = store.pop_failed_ranges() + [(a, min(a + chunk - 1, head)) for a in range(frm, head + 1, chunk)]
    n = 0
    for a, z in ranges:
        try:
            logs = rpc.call("eth_getLogs", [{"fromBlock": hex(a), "toBlock": hex(z), "address": REPUTATION_REGISTRY}])
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            logger.warning("sync_feedback: range %d-%d failed: %s", a, z, e)
            store.add_failed_range(a, z, str(e)[:200])
            continue
        decoded = [d for d in map(decode_log, logs) if d]
        n_fb = _apply(store, decoded)
        n += n_fb
        _advance_checkpoint(store, z)
        logger.info("sync_feedback: range %d-%d: %d logs, %d feedback rows", a, z, len(logs), n_fb)
    return n


def fill_block_timestamps(store: Store, rpc, batch_size: int = 100) -> int:
    """Resolve and cache ``eth_getBlockByNumber`` timestamps for every block
    referenced by feedback rows that isn't already cached (``Store.missing_block_ts``).

    Blocks are resolved in JSON-RPC batches of ``batch_size``. Returns the number
    of blocks that were missing (and are now cached), 0 if none were missing (in
    which case no RPC calls are made at all).
    """
    missing = store.missing_block_ts()
    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        blocks = rpc.batch([("eth_getBlockByNumber", [hex(b), False]) for b in chunk])
        store.upsert_block_ts([(b, int(blk["timestamp"], 16)) for b, blk in zip(chunk, blocks)])
    return len(missing)


def owner_of(rpc, agent_id: str) -> Optional[str]:
    """Resolve the IdentityRegistry owner address of ``agent_id`` via ``eth_call``.

    Returns the lowercase ``0x``-prefixed owner address, or ``None`` if the call
    returns an empty result or the zero address (agent burned/never minted).
    """
    data = _OWNER_OF_SELECTOR + int(agent_id).to_bytes(32, "big").hex()
    out = rpc.call("eth_call", [{"to": IDENTITY_REGISTRY, "data": data}, "latest"])
    if out in (None, "0x"):
        return None
    addr = "0x" + out[-40:].lower()
    return None if addr == ZERO_ADDRESS else addr


def tx_parties(rpc, tx_hash: str) -> Optional[set]:
    """Return the lowercase {from, to} addresses involved in ``tx_hash`` via
    ``eth_getTransactionByHash``. ``to`` is omitted for contract-creation
    transactions (where it is ``None``). Returns ``None`` if the transaction is
    not found."""
    tx = rpc.call("eth_getTransactionByHash", [tx_hash])
    if not tx:
        return None
    return {a.lower() for a in (tx.get("from"), tx.get("to")) if a}
