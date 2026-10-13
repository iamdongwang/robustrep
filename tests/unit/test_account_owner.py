"""``robustrep.sources.account_owner``: the controlling key of a smart-contract
rater that never received a transfer (ownerAtIndex / owner), driven through a
recording fake RPC. No network."""
import logging

import pytest

from robustrep.sources import account_owner as ao
from robustrep.sources.account_owner import (
    OWNER_AT_INDEX_SELECTOR,
    OWNER_INDEX_SCAN,
    OWNER_SELECTOR,
    account_owner,
    account_owners,
    decode_owner_address,
    decode_owner_bytes,
)
from robustrep.sources.rpc import RpcBatchStructureError, RpcClient, RpcEntryError, RpcError, RpcRevertError

WALLET = "0x" + "a1" * 20
WALLET_B = "0x" + "b2" * 20
WALLET_C = "0x" + "c3" * 20
KEY_UP = "0x" + "AB" * 20
KEY = KEY_UP.lower()
KEY2 = "0x" + "cd" * 20
ZERO = "0x" + "00" * 20


def word(n: int) -> str:
    return n.to_bytes(32, "big").hex()


def abi_bytes(payload: bytes, *, length=None, offset=32) -> str:
    """ABI-encode a ``bytes`` return value: offset word, length word, payload
    right-padded to a 32-byte boundary. ``length``/``offset`` can be forced to
    build malformed encodings."""
    padded = payload + b"\0" * (-len(payload) % 32)
    return "0x" + word(offset) + word(len(payload) if length is None else length) + padded.hex()


def addr_payload(addr: str) -> bytes:
    return b"\0" * 12 + bytes.fromhex(addr[2:])


ADDRESS_RETURN = abi_bytes(addr_payload(KEY_UP))
PASSKEY_RETURN = abi_bytes(b"\x11" * 64)
EMPTY_RETURN = abi_bytes(b"")
OWNER_RETURN = "0x" + "00" * 12 + KEY[2:]


def index_data(index: int) -> str:
    return OWNER_AT_INDEX_SELECTOR + word(index)


TRANSPORT_ERROR = "gave up after 5 attempts: HTTPError (HTTP 503) from https://rpc.example"


class FakeRpc:
    """Records every call/batch. ``answers`` maps ``(to, data)`` -> result; a
    pair not listed answers ``"0x"`` (what an address without code returns to
    any eth_call). ``revert`` names ``(to, selector)`` pairs whose call
    reverts (``RpcRevertError`` from ``call``; an ``RpcEntryError`` with
    ``reverted=True`` inside a lenient batch); ``fail`` names ``(to,
    selector)`` pairs whose *single* call raises a transport-shaped
    ``RpcError`` (a transport failure cannot be per-entry, so inside a batch
    those pairs answer ``"0x"``); ``entry_fail`` names ``(to, selector)``
    pairs that fail without reverting inside a lenient batch (out of gas) and
    raise a plain ``RpcError`` as a single call. ``batch_error`` makes
    ``batch`` raise it."""

    def __init__(self, answers=None, *, revert=(), fail=(), entry_fail=(), batch_error=None):
        self.answers = dict(answers or {})
        self.revert, self.fail, self.entry_fail = set(revert), set(fail), set(entry_fail)
        self.batch_error = batch_error
        self.calls, self.batches = [], []

    def _answer(self, params):
        to, data = params[0]["to"], params[0]["data"]
        if (to, data[:10]) in self.revert:
            raise RpcRevertError("non-retryable error: [3] execution reverted")
        if (to, data[:10]) in self.fail:
            raise RpcError(TRANSPORT_ERROR)
        if (to, data[:10]) in self.entry_fail:
            raise RpcError("gave up after 5 attempts: [-32000] out of gas")
        return self.answers.get((to, data), "0x")

    def _entry(self, params):
        to, data = params[0]["to"], params[0]["data"]
        if (to, data[:10]) in self.revert:
            return RpcEntryError(code=3, message="[3]", reverted=True)
        if (to, data[:10]) in self.entry_fail:
            return RpcEntryError(code=-32000, message="[-32000]", reverted=False)
        return self.answers.get((to, data), "0x")

    def call(self, method, params):
        self.calls.append((method, params))
        return self._answer(params)

    def batch(self, calls, *, per_entry_errors=False):
        self.batches.append(calls)
        if self.batch_error is not None:
            raise self.batch_error
        assert per_entry_errors, "account_owners must ask for per-entry errors"
        return [self._entry(p) for _, p in calls]


def _eth_calls(rpc):
    assert all(m == "eth_call" for m, _ in rpc.calls)
    return [(p[0]["to"], p[0]["data"], p[1]) for _, p in rpc.calls]


# --- constants ---------------------------------------------------------------------


def test_selectors_are_the_keccak_prefixes_of_the_signatures():
    from eth_hash.auto import keccak

    assert OWNER_AT_INDEX_SELECTOR == "0x" + keccak(b"ownerAtIndex(uint256)")[:4].hex()
    assert OWNER_SELECTOR == "0x" + keccak(b"owner()")[:4].hex()
    assert OWNER_AT_INDEX_SELECTOR == "0x8ea69029" and OWNER_SELECTOR == "0x8da5cb5b"
    assert OWNER_INDEX_SCAN == 4


# --- decode_owner_bytes ------------------------------------------------------------


@pytest.mark.parametrize("result,expected", [
    (ADDRESS_RETURN, KEY),                                              # 32-byte address payload, lowercased
    (abi_bytes(addr_payload(ZERO)), None),                              # zero address
    (PASSKEY_RETURN, None),                                             # 64-byte passkey public key
    (EMPTY_RETURN, None),                                               # length 0: owner slot empty/removed
    ("0x", None),                                                       # no code at the address
    (None, None),                                                       # null result
    (12, None),                                                         # non-string
    ("garbage-from-a-node", None),                                      # not hex
    ("0x" + "ab" * 63 + "a", None),                                     # odd hex length
    ("0x" + "zz" * 64, None),                                           # 0x-prefixed but not hex
    ("0x" + word(32), None),                                            # short: offset word only
    (abi_bytes(b"\0" * 32, length=2 ** 255), None),                     # huge length word
    (abi_bytes(b"\0" * 32, length=40), None),                           # length past the payload
    (abi_bytes(addr_payload(KEY), offset=2 ** 200), None),              # huge offset word
    (abi_bytes(addr_payload(KEY), offset=64), None),                    # offset past the data
    (abi_bytes(b"\x01" + b"\0" * 11 + bytes.fromhex(KEY[2:])), None),  # 32 bytes, not address-shaped
    (abi_bytes(b"\x11" * 20), None),                                    # 20-byte payload (not ABI word)
    (abi_bytes(b"\x11" * 96), None),                                    # some other length
], ids=["address", "zero-address", "passkey", "empty", "0x", "null", "non-string", "garbage", "odd-hex",
        "not-hex", "short", "huge-length", "length-mismatch", "huge-offset", "offset-past-end", "not-address-shaped",
        "20-bytes", "96-bytes"])
def test_decode_owner_bytes(result, expected):
    assert decode_owner_bytes(result) == expected


def test_decode_owner_bytes_never_logs_for_the_ordinary_answers(caplog):
    with caplog.at_level(logging.DEBUG):
        for result in (ADDRESS_RETURN, PASSKEY_RETURN, EMPTY_RETURN, "0x", None):
            decode_owner_bytes(result)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_decode_owner_address_reuses_the_owner_of_rules():
    assert decode_owner_address(OWNER_RETURN) == KEY
    assert decode_owner_address("0x" + "00" * 32) is None
    assert decode_owner_address("0x") is None
    assert decode_owner_address(None) is None
    assert decode_owner_address("not-hex") is None


# --- account_owner: the per-address scan -------------------------------------------


def test_account_owner_calldata_is_selector_plus_zero_padded_uint256_index():
    rpc = FakeRpc({(WALLET, index_data(0)): ADDRESS_RETURN})
    assert account_owner(rpc, WALLET) == KEY
    assert _eth_calls(rpc) == [(WALLET, "0x8ea69029" + "0" * 64, "latest")]
    assert len(rpc.calls[0][1][0]["data"]) == 10 + 64


def test_account_owner_stops_at_the_first_index_that_holds_an_address():
    rpc = FakeRpc({(WALLET, index_data(0)): ADDRESS_RETURN, (WALLET, index_data(1)): abi_bytes(addr_payload(KEY2))})
    assert account_owner(rpc, WALLET) == KEY
    assert len(rpc.calls) == 1


def test_account_owner_skips_an_empty_index_zero_and_finds_index_two():
    rpc = FakeRpc({(WALLET, index_data(0)): EMPTY_RETURN, (WALLET, index_data(1)): EMPTY_RETURN,
                   (WALLET, index_data(2)): abi_bytes(addr_payload(KEY2))})
    assert account_owner(rpc, WALLET) == KEY2
    assert [d for _, d, _ in _eth_calls(rpc)] == [index_data(0), index_data(1), index_data(2)]


def test_account_owner_skips_passkey_slots_and_takes_a_later_address():
    rpc = FakeRpc({(WALLET, index_data(0)): PASSKEY_RETURN, (WALLET, index_data(1)): ADDRESS_RETURN})
    assert account_owner(rpc, WALLET) == KEY
    assert len(rpc.calls) == 2


def test_account_owner_falls_back_to_owner_when_every_index_is_empty():
    rpc = FakeRpc({(WALLET, index_data(i)): EMPTY_RETURN for i in range(OWNER_INDEX_SCAN)}
                  | {(WALLET, OWNER_SELECTOR): OWNER_RETURN})
    assert account_owner(rpc, WALLET) == KEY
    datas = [d for _, d, _ in _eth_calls(rpc)]
    assert datas == [index_data(i) for i in range(OWNER_INDEX_SCAN)] + [OWNER_SELECTOR]


def test_account_owner_is_none_when_every_index_and_owner_are_empty():
    rpc = FakeRpc({(WALLET, index_data(i)): EMPTY_RETURN for i in range(OWNER_INDEX_SCAN)}
                  | {(WALLET, OWNER_SELECTOR): "0x" + "00" * 32})
    assert account_owner(rpc, WALLET) is None
    assert len(rpc.calls) == OWNER_INDEX_SCAN + 1


def test_account_owner_is_none_for_an_address_without_code():
    # No code: every eth_call answers "0x", no eth_getCode needed.
    rpc = FakeRpc()
    assert account_owner(rpc, WALLET) is None
    assert [m for m, _ in rpc.calls] == ["eth_call"] * (OWNER_INDEX_SCAN + 1)


def test_account_owner_revert_on_owner_at_index_falls_through_to_owner(caplog):
    rpc = FakeRpc({(WALLET, OWNER_SELECTOR): OWNER_RETURN}, revert=[(WALLET, OWNER_AT_INDEX_SELECTOR)])
    with caplog.at_level(logging.DEBUG, logger="robustrep.sources.account_owner"):
        assert account_owner(rpc, WALLET) == KEY
    # The selector does not exist here, so the remaining indices are not tried.
    assert [d[:10] for _, d, _ in _eth_calls(rpc)] == [OWNER_AT_INDEX_SELECTOR, OWNER_SELECTOR]
    assert any(r.levelno == logging.DEBUG and "revert" in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_account_owner_revert_on_both_selectors_is_none():
    rpc = FakeRpc(revert=[(WALLET, OWNER_AT_INDEX_SELECTOR), (WALLET, OWNER_SELECTOR)])
    assert account_owner(rpc, WALLET) is None
    assert len(rpc.calls) == 2


def test_account_owner_non_revert_rpc_error_propagates():
    rpc = FakeRpc(fail=[(WALLET, OWNER_AT_INDEX_SELECTOR)])
    with pytest.raises(RpcError, match="503"):
        account_owner(rpc, WALLET)
    rpc = FakeRpc({(WALLET, index_data(i)): EMPTY_RETURN for i in range(OWNER_INDEX_SCAN)},
                  fail=[(WALLET, OWNER_SELECTOR)])
    with pytest.raises(RpcError, match="503"):
        account_owner(rpc, WALLET)


# --- account_owners: batched index-0 calls with per-address fallback ---------------


def test_account_owners_batches_index_zero_and_scans_only_the_misses():
    rpc = FakeRpc({(WALLET, index_data(0)): ADDRESS_RETURN,
                   (WALLET_B, index_data(0)): EMPTY_RETURN, (WALLET_B, index_data(1)): abi_bytes(addr_payload(KEY2)),
                   (WALLET_C, index_data(0)): PASSKEY_RETURN})
    got = account_owners(rpc, [WALLET, WALLET_B, WALLET_C])
    assert got == {WALLET: KEY, WALLET_B: KEY2, WALLET_C: None}
    assert len(rpc.batches) == 1
    assert rpc.batches[0] == [("eth_call", [{"to": a, "data": index_data(0)}, "latest"])
                              for a in (WALLET, WALLET_B, WALLET_C)]
    # WALLET was settled by the batch; B and C got the per-address scan from
    # index 1 on (the batch already answered index 0 for them).
    scanned = [(to, d[:10] + d[10:].lstrip("0")) for to, d, _ in _eth_calls(rpc)]
    assert WALLET not in {to for to, _ in scanned}
    assert [d for to, d in scanned if to == WALLET_B] == [OWNER_AT_INDEX_SELECTOR + "1"]
    assert [d for to, d in scanned if to == WALLET_C] == (
        [OWNER_AT_INDEX_SELECTOR + str(i) for i in range(1, OWNER_INDEX_SCAN)] + [OWNER_SELECTOR])


def test_account_owners_chunks_by_batch_size():
    addrs = ["0x" + f"{i:040x}" for i in range(7)]
    rpc = FakeRpc({(a, index_data(0)): ADDRESS_RETURN for a in addrs})
    got = account_owners(rpc, addrs, batch_size=3)
    assert got == {a: KEY for a in addrs}
    assert [len(b) for b in rpc.batches] == [3, 3, 1]
    assert rpc.calls == []


def test_account_owners_empty_input_makes_no_calls():
    rpc = FakeRpc()
    assert account_owners(rpc, []) == {}
    assert rpc.batches == [] and rpc.calls == []


def test_account_owners_batch_size_one_never_batches():
    rpc = FakeRpc({(WALLET, index_data(0)): ADDRESS_RETURN})
    assert account_owners(rpc, [WALLET], batch_size=1) == {WALLET: KEY}
    assert rpc.batches == [] and len(rpc.calls) == 1


def test_account_owners_batch_failure_falls_back_per_address_with_one_warning(caplog):
    rpc = FakeRpc({(WALLET, index_data(0)): ADDRESS_RETURN, (WALLET_B, index_data(0)): abi_bytes(addr_payload(KEY2))},
                  batch_error=RpcError(TRANSPORT_ERROR))
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        got = account_owners(rpc, [WALLET, WALLET_B])
    assert got == {WALLET: KEY, WALLET_B: KEY2}
    assert len(rpc.batches) == 1
    assert [to for to, _, _ in _eth_calls(rpc)] == [WALLET, WALLET_B]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "account_owners" in warnings[0].getMessage()


def test_account_owners_batch_unsupported_skips_batching_for_later_chunks(caplog):
    addrs = ["0x" + f"{i:040x}" for i in range(4)]
    rpc = FakeRpc({(a, index_data(0)): ADDRESS_RETURN for a in addrs},
                  batch_error=RpcBatchStructureError("batch response is not a list"))
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        got = account_owners(rpc, addrs, batch_size=2)
    assert got == {a: KEY for a in addrs}
    assert len(rpc.batches) == 1                     # second chunk never attempted a batch
    assert len(rpc.calls) == 4
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_account_owners_shares_batch_state_across_calls():
    from robustrep.sources.base_erc8004 import BatchState

    state = BatchState()
    rpc = FakeRpc({(WALLET, index_data(0)): ADDRESS_RETURN},
                  batch_error=RpcBatchStructureError("batch response is not a list"))
    assert account_owners(rpc, [WALLET], state=state) == {WALLET: KEY}
    assert state.batch_unsupported is True
    assert account_owners(rpc, [WALLET], state=state) == {WALLET: KEY}
    assert len(rpc.batches) == 1


def test_account_owners_non_revert_error_in_fallback_is_recorded_for_that_address_only():
    rpc = FakeRpc({(WALLET_B, index_data(0)): ADDRESS_RETURN},
                  fail=[(WALLET, OWNER_AT_INDEX_SELECTOR)], batch_error=RpcError("batch down"))
    got = account_owners(rpc, [WALLET, WALLET_B])
    assert got[WALLET_B] == KEY
    assert isinstance(got[WALLET], RpcError) and "503" in str(got[WALLET])


# --- account_owners: per-entry errors, isolation ------------------------------------

W = ["0x" + f"{i:040x}" for i in range(1, 7)]   # [address, empty, passkey, reverted, out-of-gas, address]


def _mixed_rpc(**kw):
    return FakeRpc({(W[0], index_data(0)): ADDRESS_RETURN,
                    (W[1], index_data(0)): EMPTY_RETURN, (W[1], index_data(2)): abi_bytes(addr_payload(KEY2)),
                    (W[2], index_data(0)): PASSKEY_RETURN, (W[2], OWNER_SELECTOR): OWNER_RETURN,
                    (W[3], OWNER_SELECTOR): OWNER_RETURN,
                    (W[5], index_data(0)): abi_bytes(addr_payload(KEY2))},
                   revert=[(W[3], OWNER_AT_INDEX_SELECTOR)], entry_fail=[(W[4], OWNER_AT_INDEX_SELECTOR)], **kw)


def test_account_owners_mixed_chunk_does_exactly_the_right_follow_ups(caplog):
    rpc = _mixed_rpc()
    with caplog.at_level(logging.DEBUG, logger="robustrep.sources.account_owner"):
        got = account_owners(rpc, W)
    assert got == {W[0]: KEY, W[1]: KEY2, W[2]: KEY, W[3]: KEY, W[4]: None, W[5]: KEY2}
    # One index-0 batch for the chunk, then one owner() batch for the reverted entry.
    assert [[p[0]["to"] for _, p in b] for b in rpc.batches] == [W, [W[3]]]
    assert rpc.batches[1][0][1][0]["data"] == OWNER_SELECTOR
    # Per-address calls: the empty slot scans 1, 2 (found); the passkey scans
    # 1..3 then owner(); nothing else is asked anything.
    scanned = [(to, d[:10] + d[10:].lstrip("0")) for to, d, _ in _eth_calls(rpc)]
    assert scanned == [(W[1], OWNER_AT_INDEX_SELECTOR + "1"), (W[1], OWNER_AT_INDEX_SELECTOR + "2"),
                       (W[2], OWNER_AT_INDEX_SELECTOR + "1"), (W[2], OWNER_AT_INDEX_SELECTOR + "2"),
                       (W[2], OWNER_AT_INDEX_SELECTOR + "3"), (W[2], OWNER_SELECTOR)]
    # The out-of-gas contract is "no owner" at DEBUG, never a failure.
    assert any(r.levelno == logging.DEBUG and W[4] in r.getMessage() and "without reverting" in r.getMessage()
               for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_account_owners_owner_batch_entry_errors_are_none():
    # Both selectors absent (reverts twice) and an owner() that fails without
    # reverting: each is "no owner", none fails the chunk or its neighbours.
    rpc = FakeRpc({(WALLET_C, index_data(0)): ADDRESS_RETURN},
                  revert=[(WALLET, OWNER_AT_INDEX_SELECTOR), (WALLET, OWNER_SELECTOR),
                          (WALLET_B, OWNER_AT_INDEX_SELECTOR)],
                  entry_fail=[(WALLET_B, OWNER_SELECTOR)])
    assert account_owners(rpc, [WALLET, WALLET_B, WALLET_C]) == {WALLET: None, WALLET_B: None, WALLET_C: KEY}
    assert len(rpc.batches) == 2 and rpc.calls == []


def test_account_owners_owner_batch_transport_failure_falls_back_per_address(caplog):
    class _SecondBatchFails(FakeRpc):
        def batch(self, calls, *, per_entry_errors=False):
            if self.batches:
                self.batches.append(calls)
                raise RpcError(TRANSPORT_ERROR)
            return super().batch(calls, per_entry_errors=per_entry_errors)

    rpc = _SecondBatchFails({(WALLET, OWNER_SELECTOR): OWNER_RETURN},
                            revert=[(WALLET, OWNER_AT_INDEX_SELECTOR), (WALLET_B, OWNER_AT_INDEX_SELECTOR)],
                            fail=[(WALLET_B, OWNER_SELECTOR)])
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.base_erc8004"):
        got = account_owners(rpc, [WALLET, WALLET_B])
    assert got[WALLET] == KEY
    assert isinstance(got[WALLET_B], RpcError)
    assert [d[:10] for _, d, _ in _eth_calls(rpc)] == [OWNER_SELECTOR, OWNER_SELECTOR]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_account_owners_one_transport_error_fails_only_that_address():
    addrs = ["0x" + f"{i:040x}" for i in range(50)]
    answers = {(a, index_data(0)): ADDRESS_RETURN for a in addrs[1:]}
    # addrs[0]: index 0 empty, so it gets the per-address scan, whose first call dies.
    answers[(addrs[0], index_data(0))] = EMPTY_RETURN
    rpc = FakeRpc(answers, fail=[(addrs[0], OWNER_AT_INDEX_SELECTOR)])
    got = account_owners(rpc, addrs)
    assert isinstance(got[addrs[0]], RpcError)
    assert {a: got[a] for a in addrs[1:]} == {a: KEY for a in addrs[1:]}
    assert len(rpc.batches) == 1 and len(rpc.calls) == 1


def test_account_owners_per_address_mode_isolates_failures_too():
    rpc = FakeRpc({(WALLET_B, index_data(0)): ADDRESS_RETURN}, fail=[(WALLET, OWNER_AT_INDEX_SELECTOR)])
    got = account_owners(rpc, [WALLET, WALLET_B], batch_size=1)
    assert got[WALLET_B] == KEY and isinstance(got[WALLET], RpcError)
    assert rpc.batches == []


def test_account_owners_scan_revert_after_an_empty_index_zero_goes_to_owner():
    # index 0 empty (via the batch), index 1 reverts (a contract whose owner
    # list is shorter than it says) -> owner() once, no more indices.
    class _RevertFromIndexOne(FakeRpc):
        def _answer(self, params):
            if params[0]["data"] == index_data(1):
                raise RpcRevertError("non-retryable error: [3] execution reverted")
            return super()._answer(params)

    rpc = _RevertFromIndexOne({(WALLET, index_data(0)): EMPTY_RETURN, (WALLET, OWNER_SELECTOR): OWNER_RETURN})
    assert account_owners(rpc, [WALLET]) == {WALLET: KEY}
    assert [d[:10] + d[10:].lstrip("0") for _, d, _ in _eth_calls(rpc)] == [OWNER_AT_INDEX_SELECTOR + "1",
                                                                              OWNER_SELECTOR]


# --- regression through a real RpcClient: a 50-chunk with code-3 reverts ----------


class _NodeSession:
    """A fake ``requests.Session`` playing a Base public node: every
    ``ownerAtIndex``/``owner()`` eth_call on an address in ``reverting``
    answers ``{"code": 3, "message": "execution reverted"}``; every other
    ``ownerAtIndex(0)`` answers ``KEY``."""

    def __init__(self, reverting):
        self.reverting, self.posts = set(reverting), []

    def _result(self, call):
        to = call["params"][0]["to"]
        if to in self.reverting:
            return {"jsonrpc": "2.0", "id": call["id"], "error": {"code": 3, "message": "execution reverted"}}
        return {"jsonrpc": "2.0", "id": call["id"], "result": ADDRESS_RETURN}

    def post(self, url, json=None, headers=None, timeout=None, stream=None):
        import json as jsonlib

        from test_rpc import FakeResponse

        self.posts.append(json)
        body = [self._result(c) for c in json] if isinstance(json, list) else self._result(json)
        return FakeResponse(jsonlib.dumps(body).encode())


def test_fifty_chunk_with_reverting_entries_costs_two_posts_and_no_sleep():
    addrs = ["0x" + f"{i:040x}" for i in range(50)]
    reverting = addrs[::10]  # 5 of 50 do not implement ownerAtIndex
    session, sleeps = _NodeSession(reverting), []
    rpc = RpcClient(["http://a", "http://b"], user_agent="ua", session=session, sleep=sleeps.append)
    got = account_owners(rpc, addrs, batch_size=50)
    assert got == {a: (None if a in reverting else KEY) for a in addrs}
    assert len(session.posts) == 2 and sleeps == []
    assert len(session.posts[0]) == 50 and len(session.posts[1]) == 5
    assert all(c["params"][0]["data"] == OWNER_SELECTOR for c in session.posts[1])


def test_module_logger_is_named_after_the_module():
    assert ao.logger.name == "robustrep.sources.account_owner"
