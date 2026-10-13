"""The controlling key of a smart-contract rater that never received a transfer.

Sybil clustering (``robustrep.sybil``) links raters that share a ``funder``.
Through 0.2.1 that meant one thing: the ``from`` of the address's first
incoming transfer (``rater_profile``, ``alchemy_profile``). Measured 2026-10-13
on the live Base data, that rule has stopped describing the new raters: of 230
sampled unprofiled ERC-8004 raters, 230/230 were smart-contract accounts and
none had ever received a transfer -- the first thing each of them did on-chain
was post feedback -- so neither Blockscout nor Alchemy can name a funder for
them. 55 of 60 inspected are Coinbase Smart Wallet proxies (EIP-1967
implementation ``0x000100abaad02f1cfc8bbe32bd5a564817339e72``). Their first
feedback transactions reach the ERC-4337 EntryPoint v0.6, sent by ~10 bundlers
and sponsored by a single paymaster: shared infrastructure, used by everyone,
and therefore deliberately *not* read as a sybil signal here. What does
discriminate is the key that controls the account -- ``ownerAtIndex(0)`` on
those 55 wallets returned only **4 distinct owner addresses** (26/14/11/4).

This module reads that key. The result is stored in ``raters.funder`` with
``funder_kind = "owner"`` (``robustrep.store``), which widens the column's
meaning from "who funded the address" to "who controls or funded it"; the
comparison in ``robustrep.sybil`` (equality of the normalised funder) is
unchanged.

Two selectors, both over plain ``eth_call`` at ``latest`` (an address without
code answers ``"0x"`` to any ``eth_call``, so no ``eth_getCode`` is needed):

- ``ownerAtIndex(uint256) returns (bytes)`` (``OWNER_AT_INDEX_SELECTOR``), the
  Coinbase Smart Wallet owner list. Indices ``0..OWNER_INDEX_SCAN-1`` are
  scanned because owners can be removed, leaving index 0 empty. A 32-byte
  payload whose first 12 bytes are zero is an Ethereum address (the owner);
  a 64-byte payload is a passkey (WebAuthn) public key, which has no address
  form to compare against anything, so such a slot yields nothing.
- ``owner() returns (address)`` (``OWNER_SELECTOR``), the plain Ownable shape,
  tried only when no index holds an address.

A revert on either selector means the contract does not implement it (a
Solidity contract without a matching function and no fallback reverts) and is
treated as "no owner here" at DEBUG, exactly as ``base_erc8004.owner_of``
treats a reverting ``ownerOf`` (``rpc.is_revert``). ``account_owners`` asks a
whole chunk its ``ownerAtIndex(0)`` question in one ``rpc.batch(...,
per_entry_errors=True)``, so one contract's answer never decides its
neighbours': an entry that decodes to an address is done; an empty, passkey
or malformed entry gets the per-address scan of indices 1..3 and then
``owner()``; a reverting entry has no owner list, so its ``owner()`` is asked
once -- in a second lenient batch with the chunk's other reverters; an entry
that fails *without* reverting (out of gas, invalid opcode: the rater's own,
attacker-chosen contract misbehaving) is "no owner" at DEBUG and never a
failure (security review M2). Only a transport or structural ``RpcError``
escaping ``rpc.batch`` falls back to per-address calls (``_warn_batch_fallback``,
shared ``BatchState`` as ``owners_of``), and a per-address call that fails
with anything but a revert is recorded for *that* address alone: the result
dict holds the ``RpcError`` in its slot, so the caller fails one row and
keeps the other 49. Measured 2026-10-13 on the 5,985 live addresses pending
the owner pass: 5,359 are Coinbase Smart Wallet proxies (never revert on
``ownerAtIndex``), 87 another implementation and 534 have no proxy slot, so
nearly every 50-address batch holds a reverting entry -- treating that as a
batch failure cost 5 POSTs and 15 s of backoff per chunk before this.

Every result is node-supplied and decoded defensively (I3): a payload that is
not exactly address-shaped, a length or offset word pointing outside the data,
a non-string or non-hex result -- all of these are ``None``, never an
exception and never a sliced-on-faith string. The one address that comes out
passes through ``profile_http.funder_address`` (L1), the same gate every
transfer-derived funder passes, so the two rules store the same shape.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Union

from .base_erc8004 import ZERO_ADDRESS, BatchState, _decode_owner_result, _warn_batch_fallback
from .profile_http import funder_address
from .rpc import RpcEntryError, RpcError, is_revert, scrub_rpc_error

logger = logging.getLogger(__name__)

# keccak("ownerAtIndex(uint256)")[:4] -- Coinbase Smart Wallet's owner list,
# returning the raw ``bytes`` of the owner (an ABI-encoded address, or a
# 64-byte passkey public key).
OWNER_AT_INDEX_SELECTOR = "0x8ea69029"
# keccak("owner()")[:4] -- the plain Ownable getter.
OWNER_SELECTOR = "0x8da5cb5b"
# How many ``ownerAtIndex`` slots to scan (indices 0..OWNER_INDEX_SCAN-1): owners
# can be removed, so index 0 may be empty while a later one still holds a key.
OWNER_INDEX_SCAN = 4

# ABI layout constants for a ``bytes`` return value.
_WORD = 32
_ADDRESS_BYTES = 20
_ADDRESS_PAD = _WORD - _ADDRESS_BYTES
_PASSKEY_BYTES = 64


def _hex_bytes(result) -> Optional[bytes]:
    """The raw bytes of an ``eth_call`` result, or ``None`` when it is not a
    ``0x``-prefixed even-length hex string (``"0x"`` itself decodes to
    ``b""``, the "no code / empty return" answer)."""
    if not isinstance(result, str) or not result.startswith("0x") or len(result) % 2:
        return None
    try:
        return bytes.fromhex(result[2:])
    except ValueError:
        return None


def _abi_bytes_payload(raw: bytes) -> Optional[bytes]:
    """The payload of an ABI-encoded ``bytes`` return (offset word, length
    word, data), or ``None`` when the words do not describe a region inside
    ``raw`` -- a huge or misplaced offset/length word is a malformed or
    hostile answer, not a bigger one."""
    if len(raw) < 2 * _WORD:
        return None
    offset = int.from_bytes(raw[:_WORD], "big")
    if offset + _WORD > len(raw):
        return None
    length = int.from_bytes(raw[offset:offset + _WORD], "big")
    start = offset + _WORD
    if length > len(raw) - start:
        return None
    return raw[start:start + length]


def decode_owner_bytes(result, address: str = "") -> Optional[str]:
    """Decode an ``ownerAtIndex`` result -- an ABI-encoded ``bytes`` -- into a
    lowercase owner address, or ``None``.

    A 32-byte payload whose first 12 bytes are zero is an address (the zero
    address is ``None``: an empty slot). A 64-byte payload is a passkey public
    key: a real owner, but one with no address to compare, so ``None``. An
    empty payload (owner removed, or no such index), ``"0x"`` (no code at the
    address), a null or non-string result, non-hex text, a short or malformed
    encoding, or any other payload length is ``None`` too. Never raises on
    remote-shaped input. ``address`` only labels the one WARNING
    ``profile_http.funder_address`` would log, which cannot fire for an
    address-shaped payload -- it is the shared L1 gate, kept so both funder
    rules store exactly the same shape."""
    raw = _hex_bytes(result)
    if raw is None:
        return None
    payload = _abi_bytes_payload(raw)
    if payload is None or len(payload) == _PASSKEY_BYTES:
        return None
    if len(payload) != _WORD or payload[:_ADDRESS_PAD] != bytes(_ADDRESS_PAD):
        return None
    addr = "0x" + payload[_ADDRESS_PAD:].hex()
    if addr == ZERO_ADDRESS:
        return None
    return funder_address(addr, address)


def decode_owner_address(result) -> Optional[str]:
    """Decode an ``owner()`` result -- one ABI word holding an address -- with
    exactly ``base_erc8004._decode_owner_result``'s rules (empty, null and the
    zero address are ``None``; anything not address-shaped is ``None`` with
    one WARNING), so ``owner()`` here and ``ownerOf`` there never drift."""
    return _decode_owner_result(result)


def _index_call(address: str, index: int) -> list:
    """``eth_call`` params for ``ownerAtIndex(index)`` on ``address``: the
    selector plus the index as a zero-padded uint256, at ``latest``."""
    return [{"to": address, "data": OWNER_AT_INDEX_SELECTOR + index.to_bytes(_WORD, "big").hex()}, "latest"]


_REVERTED = object()

# What ``account_owners`` holds per address: the owner (lowercase address),
# ``None`` (no owner found), or the ``RpcError`` that stopped *this* address's
# lookup -- a transport failure the caller should record for it alone.
OwnerResult = Union[Optional[str], RpcError]


def _call(rpc, address: str, params: list, what: str):
    """One ``eth_call``; a revert (``rpc.is_revert``: an ``RpcRevertError``,
    or an ``RpcError`` whose message mentions "revert") is the ``_REVERTED``
    sentinel, logged at DEBUG -- the selector does not exist on this
    contract. Any other ``RpcError`` propagates."""
    try:
        return rpc.call("eth_call", params)
    except RpcError as e:
        if is_revert(e):
            logger.debug("account_owner: %s on %s reverted (selector not implemented)", what, address)
            return _REVERTED
        raise


def _owner_params(address: str) -> list:
    """``eth_call`` params for ``owner()`` on ``address``."""
    return [{"to": address, "data": OWNER_SELECTOR}, "latest"]


def _owner_call(rpc, address: str) -> Optional[str]:
    """``owner()`` on ``address`` as an address, or ``None`` (also on revert)."""
    out = _call(rpc, address, _owner_params(address), "owner()")
    return None if out is _REVERTED else decode_owner_address(out)


def _scan(rpc, address: str, start: int) -> Optional[str]:
    """``ownerAtIndex(start..OWNER_INDEX_SCAN-1)`` then ``owner()``: the first
    slot that decodes to an address, else ``owner()``'s answer. A revert on
    the index selector stops the scan (no owner list here). A non-revert
    ``RpcError`` propagates."""
    for index in range(start, OWNER_INDEX_SCAN):
        out = _call(rpc, address, _index_call(address, index), f"ownerAtIndex({index})")
        if out is _REVERTED:
            break
        owner = decode_owner_bytes(out, address)
        if owner is not None:
            return owner
    return _owner_call(rpc, address)


def account_owner(rpc, address: str) -> Optional[str]:
    """The controlling key of the smart account at ``address``, or ``None``.

    Scans ``ownerAtIndex(0..OWNER_INDEX_SCAN-1)`` and returns the first slot
    that decodes to an address (``decode_owner_bytes``); a revert on that
    selector stops the scan (the contract has no owner list). If no index held
    an address, tries ``owner()`` and returns its address, or ``None`` -- also
    ``None`` when it reverts. An address without code answers ``"0x"`` to every
    call and so is ``None`` without any ``eth_getCode``. A non-revert
    ``RpcError`` propagates from either selector.
    """
    return _scan(rpc, address, 0)


def _guarded(fn: Callable[[], Optional[str]], address: str) -> OwnerResult:
    """``fn()``, or the non-revert ``RpcError`` it raised -- recorded for
    ``address`` alone (DEBUG here; the caller decides how loudly to report a
    failed row). Reverts never reach this: ``_call`` has already turned them
    into "no owner"."""
    try:
        return fn()
    except RpcError as e:
        logger.debug("account_owner: lookup of %s failed: %s", address, scrub_rpc_error(e))
        return e


def _lenient_batch(rpc, calls: list, state: BatchState) -> Optional[list]:
    """``rpc.batch(calls, per_entry_errors=True)``, or ``None`` after the
    one WARNING of ``_warn_batch_fallback`` when the batch itself failed
    (transport, malformed shape, batch rate limit -- never a single entry's
    error, which comes back in its slot as an ``RpcEntryError``)."""
    try:
        return rpc.batch(calls, per_entry_errors=True)
    except RpcError as e:
        _warn_batch_fallback("account_owners", e, state)
        return None


def _owner_calls(rpc, addresses: list[str], state: BatchState) -> dict[str, OwnerResult]:
    """``owner()`` for every address whose ``ownerAtIndex`` reverted, in one
    lenient batch: a reverting entry (neither selector exists) or one that
    fails without reverting is ``None``. If the batch itself fails, or
    batching is already known unsupported, each ``owner()`` is asked singly
    under ``_guarded``."""
    if not addresses:
        return {}
    if not state.batch_unsupported:
        outs = _lenient_batch(rpc, [("eth_call", _owner_params(a)) for a in addresses], state)
        if outs is not None:
            return {a: None if isinstance(out, RpcEntryError) else decode_owner_address(out)
                    for a, out in zip(addresses, outs)}
    return {a: _guarded(lambda a=a: _owner_call(rpc, a), a) for a in addresses}


def _chunk_owners(rpc, chunk: list[str], state: BatchState) -> dict[str, OwnerResult]:
    """One chunk of ``account_owners``: the index-0 batch, then exactly the
    follow-up each entry calls for (see the module docstring)."""
    outs = _lenient_batch(rpc, [("eth_call", _index_call(a, 0)) for a in chunk], state)
    if outs is None:
        return {a: _guarded(lambda a=a: account_owner(rpc, a), a) for a in chunk}
    result: dict[str, OwnerResult] = {}
    scan, reverted = [], []
    for a, out in zip(chunk, outs):
        if isinstance(out, RpcEntryError):
            if out.reverted:
                reverted.append(a)
            else:
                logger.debug("account_owner: ownerAtIndex(0) on %s failed without reverting (%s): no owner",
                             a, out.message)
                result[a] = None
            continue
        owner = decode_owner_bytes(out, a)
        if owner is not None:
            result[a] = owner
        else:
            scan.append(a)
    for a in scan:
        result[a] = _guarded(lambda a=a: _scan(rpc, a, 1), a)
    result.update(_owner_calls(rpc, reverted, state))
    return result


def account_owners(rpc, addresses: list[str], batch_size: int = 50,
                   state: Optional[BatchState] = None) -> dict[str, OwnerResult]:
    """``account_owner`` for every address in ``addresses``, ``batch_size`` at a
    time, with every address's outcome independent of its neighbours'.

    Per chunk, one ``rpc.batch(..., per_entry_errors=True)`` of
    ``ownerAtIndex(0)`` calls -- index 0 answers for the vast majority of
    smart accounts, so one round trip settles most of a chunk. Then per
    entry: an address -> done; empty slot, passkey, no code or a malformed
    payload -> the per-address scan of indices 1..3 and ``owner()``; a
    reverting entry (no owner list) -> ``owner()`` once, batched with the
    chunk's other reverters; an entry that failed without reverting -> no
    owner (DEBUG). The rater's contract is attacker-chosen, so nothing it
    answers can fail the run or another address (security review M2).

    Returns ``{address: owner | None | RpcError}`` for every input address:
    the ``RpcError`` slot means a transport (non-revert) failure stopped
    *that* address's own lookup, and the caller should record it for that
    address alone. Only an ``RpcError`` from ``rpc.batch`` itself -- the
    endpoint cannot batch, a malformed batch response, a batch rate limit, a
    transport failure -- makes the chunk fall back to per-address calls, with
    one WARNING (``_warn_batch_fallback``); an ``RpcBatchUnsupportedError``
    also flips ``state.batch_unsupported`` so later chunks -- and later calls
    sharing the same ``state``, as ``owners_of`` does -- skip ``rpc.batch``.
    ``batch_size <= 1`` never calls ``rpc.batch``.
    """
    state = state or BatchState()
    result: dict[str, OwnerResult] = {}
    step = max(batch_size, 1)
    for i in range(0, len(addresses), step):
        chunk = addresses[i:i + step]
        if batch_size <= 1 or state.batch_unsupported:
            result.update({a: _guarded(lambda a=a: account_owner(rpc, a), a) for a in chunk})
        else:
            result.update(_chunk_owners(rpc, chunk, state))
    return result
