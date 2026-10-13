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
treats a reverting ``ownerOf``; any other ``RpcError`` propagates, so a network
failure is never mistaken for "no owner". ``account_owners`` batches the index-0
calls the way ``base_erc8004.owners_of`` batches ``ownerOf``, with the same
per-address fallback (and the same ``BatchState``) when a batch fails.

Every result is node-supplied and decoded defensively (I3): a payload that is
not exactly address-shaped, a length or offset word pointing outside the data,
a non-string or non-hex result -- all of these are ``None``, never an
exception and never a sliced-on-faith string. The one address that comes out
passes through ``profile_http.funder_address`` (L1), the same gate every
transfer-derived funder passes, so the two rules store the same shape.
"""
from __future__ import annotations

import logging
from typing import Optional

from .base_erc8004 import ZERO_ADDRESS, BatchState, _decode_owner_result, _warn_batch_fallback
from .profile_http import funder_address
from .rpc import RpcError

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


def _call(rpc, address: str, params: list, what: str):
    """One ``eth_call``; a revert (``RpcError`` whose message mentions
    "revert") is the ``_REVERTED`` sentinel, logged at DEBUG -- the selector
    does not exist on this contract. Any other ``RpcError`` propagates."""
    try:
        return rpc.call("eth_call", params)
    except RpcError as e:
        if "revert" in str(e).lower():
            logger.debug("account_owner: %s on %s reverted (selector not implemented): %s", what, address, e)
            return _REVERTED
        raise


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
    for index in range(OWNER_INDEX_SCAN):
        out = _call(rpc, address, _index_call(address, index), f"ownerAtIndex({index})")
        if out is _REVERTED:
            break
        owner = decode_owner_bytes(out, address)
        if owner is not None:
            return owner
    out = _call(rpc, address, [{"to": address, "data": OWNER_SELECTOR}, "latest"], "owner()")
    return None if out is _REVERTED else decode_owner_address(out)


def account_owners(rpc, addresses: list[str], batch_size: int = 50,
                   state: Optional[BatchState] = None) -> dict[str, Optional[str]]:
    """``account_owner`` for every address in ``addresses``, batching the
    ``ownerAtIndex(0)`` calls ``batch_size`` at a time via ``rpc.batch`` --
    index 0 answers for the vast majority of smart accounts, so one batch
    settles most of a chunk. An address whose index-0 result decodes to no
    address (empty slot, passkey, no code, ...) then gets the full per-address
    scan through ``account_owner``. Returns ``{address: owner-or-None}`` for
    every input address.

    Mirrors ``base_erc8004.owners_of``: if a chunk's ``rpc.batch`` raises
    ``RpcError`` (batching unsupported, a malformed batch response, a batch
    rate limit, or one reverting entry -- which fails the whole batch, not
    just that entry), the chunk is retried one address at a time via
    ``account_owner`` with one WARNING (``_warn_batch_fallback``); an
    ``RpcBatchUnsupportedError`` also flips ``state.batch_unsupported`` so
    later chunks -- and later calls sharing the same ``state`` -- skip
    ``rpc.batch`` entirely. ``batch_size <= 1`` never calls ``rpc.batch``.
    """
    state = state or BatchState()
    result: dict[str, Optional[str]] = {}
    step = max(batch_size, 1)
    for i in range(0, len(addresses), step):
        chunk = addresses[i:i + step]
        if batch_size <= 1 or state.batch_unsupported:
            result.update({a: account_owner(rpc, a) for a in chunk})
            continue
        try:
            outs = rpc.batch([("eth_call", _index_call(a, 0)) for a in chunk])
        except RpcError as e:
            _warn_batch_fallback("account_owners", e, state)
            result.update({a: account_owner(rpc, a) for a in chunk})
            continue
        for a, out in zip(chunk, outs):
            owner = decode_owner_bytes(out, a)
            result[a] = owner if owner is not None else account_owner(rpc, a)
    return result
