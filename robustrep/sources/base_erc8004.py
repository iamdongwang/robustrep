"""ERC-8004 ReputationRegistry on Base: decode events, sync into Store, resolve helpers.

``sync_feedback`` walks ``eth_getLogs`` in fixed-size block-number chunks from the
last checkpoint (``Store.get_sync("last_block")``) up to ``head - confirmations``
(default 20 blocks), decoding and persisting ``NewFeedback``/``FeedbackRevoked``/
``ResponseAppended`` events as it goes. Rows are only ever ingested once "final"
(``confirmations`` blocks old) so a chain re-org near the tip cannot leave phantom
feedback rows behind that later vanish from the canonical chain.

A chunk whose ``eth_getLogs`` fetch raises is recorded via ``Store.add_failed_range``
and skipped -- a transient RPC hiccup does not abort the whole sync, and chunks are
otherwise independent, so later chunks in the same run still advance the checkpoint
even if an earlier one failed. Failed ranges are retried (and popped) at the start
of the *next* ``sync_feedback`` call, one at a time. Because retried ranges and
freshly-computed ranges can be processed in either order, the checkpoint is
advanced with ``max(current, chunk_end)`` rather than a plain overwrite, so
retrying an old failed range can never rewind the cursor.

Decoding/applying a fetched chunk's logs is a different story: a failure there
(e.g. a log that doesn't match the ABI we expect) is a *bug*, not a transient
condition, so it is logged at ERROR and re-raised rather than swallowed. To stay
crash-safe even then (or under an abrupt interruption such as Ctrl-C while
waiting on a fetch), the whole per-chunk loop runs under a ``try/finally``: on
*any* exception escaping the loop, every range that was popped off the queue but
not yet fully processed -- the one that was in flight plus everything still
pending -- is re-added to ``failed_ranges`` before the exception propagates, so
no range is ever silently lost to a crash.
"""
from __future__ import annotations

import logging
from typing import Optional

from eth_abi import decode
from eth_hash.auto import keccak

from ..config import DEFAULT_CONFIRMATIONS
from ..store import Store
from .rpc import RpcBatchUnsupportedError, RpcError

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
# (event name, expected len(topics) -- topic0 plus each indexed argument).
_EVENT_TOPIC_COUNT = {
    TOPIC_NEW_FEEDBACK: ("NewFeedback", 4),
    TOPIC_REVOKED: ("FeedbackRevoked", 4),
    TOPIC_RESPONSE: ("ResponseAppended", 4),
}


def _addr(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def _uint(topic: str) -> int:
    return int(topic, 16)


def _hexint(v) -> int:
    """Coerce a block number / log index field to ``int``, accepting either a
    already-decoded ``int`` or a ``0x``-prefixed hex string (different RPC
    clients/fixtures represent these differently)."""
    return v if isinstance(v, int) else int(v, 16)


def decode_log(log: dict) -> Optional[dict]:
    """Decode one ``eth_getLogs`` entry from the ReputationRegistry.

    Returns a dict with a ``kind`` of ``"feedback"``, ``"revoked"`` or
    ``"response"`` (plus the event's fields and the base ``chain``/``block``/
    ``tx_hash``/``log_index`` fields), or ``None`` if ``log``'s ``topics[0]``
    does not match any known event.

    Raises ``ValueError`` (naming the problem, never a bare ``IndexError`` or
    ABI-decode error) if: ``log`` lacks one of the four fields every log is
    expected to carry (``topics``, ``blockNumber``, ``transactionHash``,
    ``logIndex``); ``topics`` is empty (a registry log always has a topic0);
    or ``topics`` has the wrong length for a recognized event.
    """
    for field in _REQUIRED_LOG_FIELDS:
        if field not in log:
            raise ValueError(f"decode_log: log missing {field!r}")
    topics = log["topics"]
    if not topics:
        raise ValueError("decode_log: log has empty topics (expected topic0)")
    t0 = topics[0].lower()
    if t0 not in _EVENT_TOPIC_COUNT:
        return None
    name, want = _EVENT_TOPIC_COUNT[t0]
    if len(topics) != want:
        raise ValueError(f"decode_log: {name} log has {len(topics)} topics, expected {want}")
    base = dict(chain=CHAIN, block=_hexint(log["blockNumber"]), tx_hash=log["transactionHash"],
                log_index=_hexint(log["logIndex"]))
    data = bytes.fromhex(log["data"][2:]) if log.get("data", "0x") != "0x" else b""
    if t0 == TOPIC_NEW_FEEDBACK:
        fi, val, dec, tag1, tag2, endpoint, uri, h = decode(FEEDBACK_TYPES, data)
        return {**base, "kind": "feedback", "agent_id": str(_uint(topics[1])), "client": _addr(topics[2]),
                "feedback_index": fi, "value": str(val), "value_decimals": dec, "tag1": tag1, "tag2": tag2,
                "endpoint": endpoint, "feedback_uri": uri, "feedback_hash": "0x" + h.hex()}
    if t0 == TOPIC_REVOKED:
        return {**base, "kind": "revoked", "agent_id": str(_uint(topics[1])), "client": _addr(topics[2]),
                "feedback_index": _uint(topics[3])}
    fi, uri, h = decode(RESPONSE_TYPES, data)
    return {**base, "kind": "response", "agent_id": str(_uint(topics[1])), "client": _addr(topics[2]),
            "responder": _addr(topics[3]), "feedback_index": fi, "response_uri": uri,
            "response_hash": "0x" + h.hex()}


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
                   end_block: Optional[int] = None, confirmations: int = DEFAULT_CONFIRMATIONS) -> int:
    """Pull NewFeedback/FeedbackRevoked/ResponseAppended from the last checkpoint
    (``store.get_sync("last_block")``) up to ``end_block`` and persist them into
    ``store``. Returns the number of feedback rows decoded across all chunks
    (see ``_apply`` for what "decoded" means).

    When ``end_block`` is not given, it defaults to ``chain_head - confirmations``:
    rows are only ingested once "final" so a chain re-org near the tip cannot
    leave phantom feedback behind (``confirmations`` is ignored when ``end_block``
    is given explicitly -- the caller is asking for that exact block).

    Block ranges are walked in ``chunk``-sized windows (Base's ``eth_getLogs`` is
    limited to 2,000 blocks per call), filtered server-side to this registry's
    three known event topics. Any previously failed ranges (see
    ``Store.add_failed_range``) are retried first, one at a time. A chunk whose
    ``eth_getLogs`` call raises is recorded as a new failed range and skipped --
    it does not stop the sync and does not advance the checkpoint -- but later
    chunks still do (see module docstring for the checkpoint-advance ordering
    guarantee). A chunk that fetches fine but fails to *decode/apply* is a bug,
    not a transient condition: it is logged at ERROR and re-raised, and (per the
    module docstring's crash-safety guarantee) every not-yet-processed range,
    including the one that failed, is re-added to ``failed_ranges`` before the
    exception propagates. If the checkpoint is already at or past ``end_block``,
    no RPC calls are made and 0 is returned.
    """
    head = end_block if end_block is not None else int(rpc.call("eth_blockNumber", []), 16) - confirmations
    last = store.get_sync("last_block")
    frm = int(last) + 1 if last is not None else start_block
    pending = store.pop_failed_ranges() + [(a, min(a + chunk - 1, head)) for a in range(frm, head + 1, chunk)]
    n = 0
    current: Optional[tuple[int, int]] = None
    try:
        while pending:
            current = pending.pop(0)
            a, z = current
            try:
                logs = rpc.call("eth_getLogs", [{
                    "fromBlock": hex(a), "toBlock": hex(z), "address": REPUTATION_REGISTRY,
                    "topics": [[TOPIC_NEW_FEEDBACK, TOPIC_REVOKED, TOPIC_RESPONSE]],
                }])
            except Exception as e:  # noqa: BLE001 - transient fetch failure: recorded, not swallowed
                logger.warning("sync_feedback: range %d-%d fetch failed: %s", a, z, e)
                store.add_failed_range(a, z, str(e)[:200])
                current = None
                continue
            try:
                decoded = [d for d in map(decode_log, logs) if d]
                n_fb = _apply(store, decoded)
            except Exception as e:
                # A decode/apply failure is a bug, not a transient RPC condition:
                # log loudly and let it propagate (the outer `finally` records
                # this range -- still `current` -- before the exception exits).
                logger.error("sync_feedback: range %d-%d decode/apply failed: %s", a, z, e)
                raise
            n += n_fb
            _advance_checkpoint(store, z)
            logger.info("sync_feedback: range %d-%d: %d logs, %d feedback rows", a, z, len(logs), n_fb)
            current = None
    finally:
        leftover = ([current] if current is not None else []) + pending
        for fa, fz in leftover:
            store.add_failed_range(fa, fz, "sync_feedback: not processed (aborted mid-run)")
    return n


# Suggested in the fallback WARNING once an endpoint's batching is given up on
# for the rest of the run (see ``BatchState``/``_warn_batch_fallback``): a
# public Base endpoint known to handle JSON-RPC batches correctly.
_BATCH_FALLBACK_HINT = "consider --rpc-url https://base-rpc.publicnode.com"


class BatchState:
    """Remembers, for the duration of one call site's run (e.g. one
    ``robustrep fetch`` invocation), whether this RPC endpoint has already
    been found unable to batch (see ``RpcBatchUnsupportedError``) -- so
    ``fill_block_timestamps``/``owners_of`` stop re-attempting ``rpc.batch``
    on every remaining chunk once it's known to be futile, and go straight to
    per-item calls instead. Pass the *same* instance across multiple calls
    (e.g. once per chunk, as ``robustrep.cli._step_owners`` does) to share
    that knowledge across the whole run; a call given no ``state`` gets a
    private, call-scoped one, which still short-circuits correctly for any
    later chunks *within* that one call."""

    def __init__(self) -> None:
        self.batch_unsupported = False


def _warn_batch_fallback(op: str, exc: RpcError, state: BatchState) -> None:
    """Log a chunk's ``rpc.batch`` failure at WARNING and fall back to
    per-item calls for that chunk. If ``exc`` is an ``RpcBatchUnsupportedError``
    (batching itself is the problem -- a structurally wrong response or a
    batch-specific rate limit, not an ordinary transient/per-call failure),
    also flip ``state.batch_unsupported`` so every *remaining* chunk in this
    run skips ``rpc.batch`` entirely rather than re-attempting (and
    re-failing) an identical, doomed batch call."""
    if isinstance(exc, RpcBatchUnsupportedError):
        state.batch_unsupported = True
        logger.warning("%s: batch fetch failed (%s); batching unsupported by this endpoint, falling back "
                        "to per-item calls for the rest of this run (%s)", op, exc, _BATCH_FALLBACK_HINT)
    else:
        logger.warning("%s: batch fetch failed (%s), falling back to per-item calls", op, exc)


def fill_block_timestamps(store: Store, rpc, batch_size: int = 100, state: Optional[BatchState] = None) -> int:
    """Resolve and cache ``eth_getBlockByNumber`` timestamps for every block
    referenced by feedback rows that isn't already cached (``Store.missing_block_ts``).

    Blocks are resolved in JSON-RPC batches of ``batch_size`` (or one at a
    time, without ever calling ``rpc.batch``, when ``batch_size <= 1``). If a
    batch call itself fails (``RpcError`` -- e.g. the endpoint doesn't support
    batching, or rejects an oversized batch), that batch is retried one block
    at a time via individual ``eth_getBlockByNumber`` calls (logged once at
    WARNING; see ``_warn_batch_fallback``). When the failure is an
    ``RpcBatchUnsupportedError`` -- batching itself doesn't work against this
    endpoint, not just this one chunk -- every later chunk in this call also
    skips straight to per-item calls (see ``BatchState``) instead of
    re-attempting (and re-failing) ``rpc.batch`` for nothing; pass a shared
    ``state`` to carry that knowledge across multiple calls too. Raises
    ``RpcError`` naming the block if any resolved block comes back ``null``
    (should not happen for an already-mined block; treated as a hard error
    rather than silently caching a missing timestamp). Returns the number of
    blocks that were missing (and are now cached), 0 if none were missing (in
    which case no RPC calls are made at all).
    """
    state = state or BatchState()
    missing = store.missing_block_ts()
    step = max(batch_size, 1)
    for i in range(0, len(missing), step):
        block_nums = missing[i:i + step]
        if batch_size <= 1 or state.batch_unsupported:
            blocks = [rpc.call("eth_getBlockByNumber", [hex(b), False]) for b in block_nums]
        else:
            try:
                blocks = rpc.batch([("eth_getBlockByNumber", [hex(b), False]) for b in block_nums])
            except RpcError as e:
                _warn_batch_fallback("fill_block_timestamps", e, state)
                blocks = [rpc.call("eth_getBlockByNumber", [hex(b), False]) for b in block_nums]
        pairs = []
        for block_num, blk in zip(block_nums, blocks):
            if blk is None:
                raise RpcError(f"fill_block_timestamps: block {block_num} not found (null result)")
            pairs.append((block_num, _hexint(blk["timestamp"])))
        store.upsert_block_ts(pairs)
    return len(missing)


def _owner_call_data(agent_id: str) -> str:
    """``eth_call`` calldata for ``ownerOf(uint256 agent_id)`` on the
    IdentityRegistry -- the 4-byte selector plus the id left-padded to 32 bytes."""
    return _OWNER_OF_SELECTOR + int(agent_id).to_bytes(32, "big").hex()


def _decode_owner_result(out) -> Optional[str]:
    """Decode one ``eth_call`` result for ``ownerOf`` into a lowercase
    ``0x``-prefixed address, or ``None`` if it's empty or the zero address
    (agent burned/never minted). Shared by ``owner_of`` and ``owners_of`` so
    both apply exactly the same decoding rules."""
    if out in (None, "0x"):
        return None
    addr = "0x" + out[-40:].lower()
    return None if addr == ZERO_ADDRESS else addr


def owner_of(rpc, agent_id: str) -> Optional[str]:
    """Resolve the IdentityRegistry owner address of ``agent_id`` via ``eth_call``.

    Returns the lowercase ``0x``-prefixed owner address, or ``None`` if the call
    returns an empty result, the zero address (agent burned/never minted), or the
    call reverts (``RpcError`` whose message mentions "revert" -- ``ownerOf`` on
    a nonexistent token reverts rather than returning zero on most ERC-721
    implementations; logged at DEBUG). Any other ``RpcError`` (network failure,
    rate limit, ...) propagates rather than being mistaken for "no owner".
    """
    data = _owner_call_data(agent_id)
    try:
        out = rpc.call("eth_call", [{"to": IDENTITY_REGISTRY, "data": data}, "latest"])
    except RpcError as e:
        if "revert" in str(e).lower():
            logger.debug("owner_of: agent %s eth_call reverted (likely nonexistent): %s", agent_id, e)
            return None
        raise
    return _decode_owner_result(out)


def owners_of(rpc, agent_ids: list[str], batch_size: int = 100,
              state: Optional[BatchState] = None) -> dict[str, Optional[str]]:
    """Resolve the IdentityRegistry owner address of every id in ``agent_ids``,
    via JSON-RPC batches of ``batch_size`` ``eth_call``s instead of one request
    per agent (``owner_of`` does ~104ms/call; batching cuts a ~48-minute walk of
    28k agents down dramatically). ``batch_size <= 1`` resolves every agent via
    ``owner_of`` directly, without ever calling ``rpc.batch``.

    Each chunk is decoded with the same rules as ``owner_of`` (see
    ``_decode_owner_result``): empty result or the zero address map to ``None``.
    If a chunk's ``rpc.batch`` call raises ``RpcError`` -- e.g. the endpoint
    doesn't support batching, rejects an oversized batch, returns a malformed
    response (some public endpoints return a dict instead of a list for a
    100-call batch), rate-limits batches specifically, or one call in the
    chunk reverts (which fails the whole batch, not just that entry) -- that
    chunk is retried one agent at a time via ``owner_of`` (which already maps
    a revert to ``None``), logged once at WARNING (see ``_warn_batch_fallback``).
    When the failure is an ``RpcBatchUnsupportedError`` -- batching itself
    doesn't work against this endpoint, not just this one chunk -- every
    later chunk in this call also skips straight to per-agent calls (see
    ``BatchState``) instead of re-attempting (and re-failing) ``rpc.batch``
    for nothing; pass a shared ``state`` to carry that knowledge across
    multiple calls too (e.g. one per chunk, as ``robustrep.cli._step_owners``
    does). Returns a dict covering every id in ``agent_ids``, regardless of
    which chunks needed the fallback.
    """
    state = state or BatchState()
    result: dict[str, Optional[str]] = {}
    step = max(batch_size, 1)
    for i in range(0, len(agent_ids), step):
        chunk = agent_ids[i:i + step]
        if batch_size <= 1 or state.batch_unsupported:
            for a in chunk:
                result[a] = owner_of(rpc, a)
            continue
        calls = [("eth_call", [{"to": IDENTITY_REGISTRY, "data": _owner_call_data(a)}, "latest"]) for a in chunk]
        try:
            outs = rpc.batch(calls)
        except RpcError as e:
            _warn_batch_fallback("owners_of", e, state)
            for a in chunk:
                result[a] = owner_of(rpc, a)
            continue
        for a, out in zip(chunk, outs):
            result[a] = _decode_owner_result(out)
    return result


def tx_parties(rpc, tx_hash: str) -> Optional[set]:
    """Return the lowercase {from, to} addresses involved in ``tx_hash`` via
    ``eth_getTransactionByHash``. ``to`` is omitted for contract-creation
    transactions (where it is ``None``). Returns ``None`` if the transaction is
    not found."""
    tx = rpc.call("eth_getTransactionByHash", [tx_hash])
    if not tx:
        return None
    return {a.lower() for a in (tx.get("from"), tx.get("to")) if a}
