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

Every field of a log is attacker-influenced (anyone can emit an event from any
contract, and a node could serve a hand-written one), so ``decode_log`` validates
the shape of each topic and of the data blob before decoding (security review
finding L4) and *skips* a log whose shape is wrong -- returning ``None`` with one
WARNING, exactly like an unrecognized topic0 -- rather than raising. Raising would
be a new crash path reachable by anyone willing to emit a malformed log: see the
"a failure there is a *bug*" paragraph below, which would turn one hostile log
into an aborted sync. The pre-existing structural checks (missing log fields,
empty topics, wrong topic count for a recognized event) keep raising ``ValueError``
-- that contract predates this and callers rely on it.

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
import re
from typing import Optional

from eth_abi import decode
from eth_hash.auto import keccak

from ..config import DEFAULT_CONFIRMATIONS
from ..store import Store
from .http_util import short_for_log as _short
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


# A 32-byte log topic, and the address derived from one (L4).
_TOPIC_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")
_ADDR_RE = re.compile(r"^0x[0-9a-f]{40}$")
# An ABI data blob: "0x" plus a whole number of hex-encoded bytes.
_DATA_RE = re.compile(r"^0x(?:[0-9a-fA-F]{2})*$")


def _addr(topic: str) -> str:
    """The 20-byte address encoded in the low bytes of a 32-byte ``topic``.

    Raises ``ValueError`` if the result is not a canonical ``0x``-prefixed
    lowercase hex address (L4): slicing the last 40 characters of a too-short
    or non-hex topic would otherwise yield a plausible-looking string that is
    not an address at all, and it would be stored as a rater/responder identity.
    ``decode_log`` validates topics up front, so this is the second line of
    defense -- and the first one for any other caller."""
    addr = "0x" + str(topic)[-40:].lower()
    if not _ADDR_RE.match(addr):
        raise ValueError(f"not a valid address topic: {_short(topic)!r}")
    return addr


def _uint(topic: str) -> int:
    return int(topic, 16)


def _hexint(v) -> int:
    """Coerce a block number / log index field to ``int``, accepting either a
    already-decoded ``int`` or a ``0x``-prefixed hex string (different RPC
    clients/fixtures represent these differently)."""
    return v if isinstance(v, int) else int(v, 16)


def _valid_shape(name: str, topics: list, data_hex, tx_hash) -> bool:
    """True when every topic is a 32-byte hex string and ``data_hex`` is a hex
    byte string (L4). A mismatch logs one WARNING -- with every remote-chosen
    value rendered via ``%r`` and length-bounded by ``_short`` -- and the caller
    skips the log."""
    for i, topic in enumerate(topics):
        if not isinstance(topic, str) or not _TOPIC_RE.match(topic):
            logger.warning("decode_log: skipping %s log in tx %r: topic %d is not a 32-byte hex "
                            "string: %r", name, _short(tx_hash), i, _short(topic))
            return False
    if not isinstance(data_hex, str) or not _DATA_RE.match(data_hex):
        logger.warning("decode_log: skipping %s log in tx %r: data is not a hex byte string: %r",
                        name, _short(tx_hash), _short(data_hex))
        return False
    return True


def decode_log(log: dict) -> Optional[dict]:
    """Decode one ``eth_getLogs`` entry from the ReputationRegistry.

    Returns a dict with a ``kind`` of ``"feedback"``, ``"revoked"`` or
    ``"response"`` (plus the event's fields and the base ``chain``/``block``/
    ``tx_hash``/``log_index`` fields), or ``None`` if ``log``'s ``topics[0]``
    does not match any known event.

    Also returns ``None`` -- logging one WARNING -- for a recognized event whose
    *shape* is wrong: a topic that is not ``0x`` plus 64 hex characters, or a
    ``data`` field that is not ``0x`` plus whole hex bytes (L4). These fields
    come from whoever emitted the event, so a malformed one must be skipped like
    any uninteresting log, not turned into an exception that aborts the caller's
    whole sync (see the module docstring).

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
    t0 = topics[0].lower() if isinstance(topics[0], str) else None
    if t0 not in _EVENT_TOPIC_COUNT:
        return None
    name, want = _EVENT_TOPIC_COUNT[t0]
    if len(topics) != want:
        raise ValueError(f"decode_log: {name} log has {len(topics)} topics, expected {want}")
    data_hex = log.get("data", "0x")
    if not _valid_shape(name, topics, data_hex, log.get("transactionHash")):
        return None
    base = dict(chain=CHAIN, block=_hexint(log["blockNumber"]), tx_hash=log["transactionHash"],
                log_index=_hexint(log["logIndex"]))
    data = bytes.fromhex(data_hex[2:]) if data_hex != "0x" else b""
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


# uint256 upper bound: an agent id at or past this cannot be ABI-encoded.
_UINT256_LIMIT = 2 ** 256


def _owner_call_data(agent_id: str) -> str:
    """``eth_call`` calldata for ``ownerOf(uint256 agent_id)`` on the
    IdentityRegistry -- the 4-byte selector plus the id left-padded to 32 bytes.

    Raises ``ValueError`` if ``agent_id`` is not a decimal integer in
    ``[0, 2**256)`` (L4). Agent ids reach here from the store, which is filled
    from log topics, so a bad one is remote input: without this check
    ``int(agent_id)`` raises ``ValueError`` on a non-decimal id and
    ``.to_bytes(32, "big")`` raises ``OverflowError`` on an over-long one --
    neither of which the callers below (which handle only ``RpcError``) expect.
    """
    try:
        value = int(agent_id)
    except (TypeError, ValueError):
        raise ValueError(f"agent id is not a decimal integer: {_short(agent_id)!r}") from None
    if not 0 <= value < _UINT256_LIMIT:
        raise ValueError(f"agent id out of uint256 range: {_short(agent_id)!r}")
    return _OWNER_OF_SELECTOR + value.to_bytes(32, "big").hex()


def _owner_calls(agent_ids: list[str]) -> tuple[list[str], list, list[str]]:
    """Split ``agent_ids`` into ``(ok_ids, batch_calls, skipped_ids)``.

    An id ``_owner_call_data`` rejects is skipped with one WARNING rather than
    raising: it would fail identically on every retry, and one unusable id in a
    100-id chunk must not cost the other 99 their owner lookup (L4)."""
    ok_ids: list[str] = []
    calls: list = []
    skipped: list[str] = []
    for agent_id in agent_ids:
        try:
            data = _owner_call_data(agent_id)
        except ValueError as e:
            logger.warning("owners_of: skipping agent id %r: %s", _short(agent_id), e)
            skipped.append(agent_id)
            continue
        ok_ids.append(agent_id)
        calls.append(("eth_call", [{"to": IDENTITY_REGISTRY, "data": data}, "latest"]))
    return ok_ids, calls, skipped


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

    An ``agent_id`` that cannot be ABI-encoded at all (see ``_owner_call_data``)
    is also ``None``, logged once at WARNING: no RPC call is made for it (L4).
    """
    try:
        data = _owner_call_data(agent_id)
    except ValueError as e:
        logger.warning("owner_of: skipping agent id %r: %s", _short(agent_id), e)
        return None
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

    An id that cannot be ABI-encoded (see ``_owner_call_data``) maps to ``None``
    and never enters a batch, so one unusable id cannot cost its chunk-mates
    their lookup (L4).
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
        ok_ids, calls, skipped = _owner_calls(chunk)
        for a in skipped:
            result[a] = None
        if not calls:
            continue
        try:
            outs = rpc.batch(calls)
        except RpcError as e:
            _warn_batch_fallback("owners_of", e, state)
            for a in ok_ids:
                result[a] = owner_of(rpc, a)
            continue
        for a, out in zip(ok_ids, outs):
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
