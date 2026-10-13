"""``enrich_raters`` pass 2: the owner lookup and internal-transfer retry for
raters whose first-transfer lookup found nothing (``funder_kind`` NULL).

Split out of ``test_rater_profile.py`` (file size). Everything here runs
against fakes -- a duck-typed transfer client, an ``AlchemyClient`` subclass
that records the categories it was asked for, and a recording RPC -- so no
test touches the network.
"""
import logging

import pytest

from robustrep.sources.account_owner import OWNER_AT_INDEX_SELECTOR
from robustrep.sources.alchemy_profile import AlchemyClient
from robustrep.sources.profile_http import ProfilePlanError
from robustrep.sources.rater_profile import enrich_raters
from robustrep.sources.base_erc8004 import BatchState
from robustrep.sources.rpc import RpcBatchStructureError, RpcError
from robustrep.store import Store

FB = dict(chain="base", tx_hash="0x", value="1", value_decimals=0, tag1="", tag2="", endpoint="",
          feedback_uri="", feedback_hash="")
A, B, C, D = ("0x" + "a1" * 20, "0x" + "b2" * 20, "0x" + "c3" * 20, "0x" + "d4" * 20)
FUNDER = "0x" + "f0" * 20
KEY = "0x" + "ab" * 20
KEY2 = "0x" + "cd" * 20
ALCHEMY_KEY = "ALCHEMY-TEST-KEY-VALUE-0123456789"


def fb(client, log_index=0):
    return dict(FB, block=1, log_index=log_index, agent_id="1", client=client, feedback_index=0)


def seed(store, *clients):
    store.upsert_feedback([fb(c, i) for i, c in enumerate(clients)])


def rows(store):
    out = store.conn.execute("SELECT address, first_seen_ts, funder, funder_kind FROM raters").fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in out}


def abi_address(addr):
    return "0x" + (32).to_bytes(32, "big").hex() + (32).to_bytes(32, "big").hex() + "00" * 12 + addr[2:]


class TransferClient:
    """Duck-typed (non-Alchemy) first-transfer client: ``answers`` maps an
    address to a ``(block, ts, funder)`` tuple or ``None``."""
    source = "blockscout"

    def __init__(self, answers=None, raise_for=()):
        self.answers, self.raise_for, self.calls = dict(answers or {}), set(raise_for), []

    def first_tx(self, address):
        self.calls.append(address)
        if address in self.raise_for:
            raise RuntimeError("boom")
        return self.answers.get(address)


class RecordingAlchemy(AlchemyClient):
    """A real ``AlchemyClient`` (so ``isinstance`` holds) whose ``first_tx``
    answers from dicts instead of HTTP: ``answers`` for the default external
    lookup, ``retry_answers`` for the ``("external", "internal")`` retry.
    ``raise_on_retry`` maps an address to an exception the retry raises."""

    def __init__(self, answers=None, retry_answers=None, raise_on_retry=None):
        super().__init__(ALCHEMY_KEY, session=object(), sleep=lambda _: None)
        self.answers, self.retry_answers = dict(answers or {}), dict(retry_answers or {})
        self.raise_on_retry = dict(raise_on_retry or {})
        self.calls = []

    def first_tx(self, address, categories=("external",)):
        self.calls.append((address, tuple(categories)))
        if tuple(categories) == ("external",):
            return self.answers.get(address)
        if address in self.raise_on_retry:
            raise self.raise_on_retry[address]
        return self.retry_answers.get(address)


class OwnerRpc:
    """Recording RPC: ``owners`` maps an address to its ``ownerAtIndex(0)``
    owner; every other eth_call answers ``"0x"`` (no code / empty). ``fail``
    names addresses whose *single* calls raise a transport-shaped
    ``RpcError`` (a transport failure cannot be per-entry, so inside a batch
    they answer ``"0x"`` and get the per-address scan, whose first call then
    dies). ``batch_error`` makes every ``batch`` raise it."""

    def __init__(self, owners=None, fail=(), batch_error=None):
        self.owners, self.fail, self.batch_error = dict(owners or {}), set(fail), batch_error
        self.calls, self.batches = [], []

    def _answer(self, params):
        to, data = params[0]["to"], params[0]["data"]
        if data == OWNER_AT_INDEX_SELECTOR + "0" * 64 and self.owners.get(to):
            return abi_address(self.owners[to])
        return "0x"

    def call(self, method, params):
        self.calls.append(params[0]["to"])
        if params[0]["to"] in self.fail:
            raise RpcError("gave up after 5 attempts: HTTPError (HTTP 503) from https://rpc.example")
        return self._answer(params)

    def batch(self, calls, *, per_entry_errors=False):
        self.batches.append([p[0]["to"] for _, p in calls])
        if self.batch_error is not None:
            raise self.batch_error
        return [self._answer(p) for _, p in calls]

    def queried(self):
        return set(self.calls) | {a for b in self.batches for a in b}


# --- pass 1 writes funder_kind -----------------------------------------------------


def test_pass_one_marks_a_found_funder_transfer_and_leaves_no_transfer_null(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B)
        assert enrich_raters(s, TransferClient({A: (1, 100, FUNDER)})) == 2
        assert rows(s) == {A: (100, FUNDER, "transfer"), B: (None, None, None)}
        assert s.raters_needing_owner_lookup() == [B]
        assert s.get_sync("rater_profile_mode") == "blockscout"


def test_no_rpc_means_no_pass_two(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A)
        assert enrich_raters(s, TransferClient(), rpc=None) == 1
        assert rows(s) == {A: (None, None, None)}


# --- pass 2: owner found -----------------------------------------------------------


def test_owner_found_writes_kind_owner_with_first_seen_left_null(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B)
        rpc = OwnerRpc({A: KEY, B: KEY})
        n = enrich_raters(s, TransferClient(), rpc=rpc)
        assert n == 4  # 2 processed in pass 1 + 2 in pass 2
        assert rows(s) == {A: (None, KEY, "owner"), B: (None, KEY, "owner")}
        assert s.raters_needing_owner_lookup() == []
        assert s.get_sync("rater_profile_mode") == "blockscout"
        # Clustering sees the shared key through the ordinary funder column.
        meta = s.load_rater_meta()
        assert set(meta["funder"]) == {KEY} and meta["first_seen_ts"].isna().all()


def test_pass_two_batches_index_zero_calls(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B)
        rpc = OwnerRpc({A: KEY, B: KEY2})
        enrich_raters(s, TransferClient(), rpc=rpc)
        assert rpc.batches == [[A, B]]
        assert rpc.calls == []


# --- pass 2: internal-transfer retry -----------------------------------------------


def test_alchemy_retry_with_internal_category_writes_transfer(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A)
        client = RecordingAlchemy(retry_answers={A: (7, 777, FUNDER)})
        assert enrich_raters(s, client, rpc=OwnerRpc()) == 2
        assert rows(s) == {A: (777, FUNDER, "transfer")}
        assert client.calls == [(A, ("external",)), (A, ("external", "internal"))]
        assert s.get_sync("rater_profile_mode") == "alchemy"


def test_alchemy_retry_is_skipped_when_an_owner_was_found(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A)
        client = RecordingAlchemy(retry_answers={A: (7, 777, FUNDER)})
        enrich_raters(s, client, rpc=OwnerRpc({A: KEY}))
        assert rows(s) == {A: (None, KEY, "owner")}
        assert client.calls == [(A, ("external",))]


def test_non_alchemy_client_gets_no_internal_retry(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A)
        client = TransferClient()
        assert enrich_raters(s, client, rpc=OwnerRpc()) == 2
        assert client.calls == [A]
        assert rows(s) == {A: (None, None, "none")}


# --- pass 2: nothing found -> none -------------------------------------------------


def test_nothing_found_anywhere_writes_kind_none(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A)
        client = RecordingAlchemy()
        assert enrich_raters(s, client, rpc=OwnerRpc()) == 2
        assert rows(s) == {A: (None, None, "none")}
        assert client.calls == [(A, ("external",)), (A, ("external", "internal"))]


def test_pass_two_selects_only_null_kind_rows_without_a_funder(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_rater(A, 100, FUNDER, "transfer")   # has a funder
        s.upsert_rater(B, None, None, "none")        # already settled
        s.upsert_rater(C, None, KEY2, "owner")       # has an owner
        s.upsert_rater(D, None, None)                # legacy NULL -> the one to look up
        rpc = OwnerRpc({D: KEY})
        # Every client address is already in raters, so pass 1 has nothing to do.
        assert enrich_raters(s, TransferClient(), rpc=rpc) == 1
        assert rpc.queried() == {D}
        assert rows(s)[D] == (None, KEY, "owner")
        assert rows(s)[A] == (100, FUNDER, "transfer") and rows(s)[B] == (None, None, "none")


def test_later_run_does_not_requery_settled_rows(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B)
        first = OwnerRpc({A: KEY})
        assert enrich_raters(s, TransferClient(), rpc=first) == 4
        assert rows(s) == {A: (None, KEY, "owner"), B: (None, None, "none")}
        again = OwnerRpc({A: KEY, B: KEY})
        client = TransferClient()
        assert enrich_raters(s, client, rpc=again) == 0
        assert again.queried() == set() and client.calls == []
        assert rows(s)[B] == (None, None, "none")
        assert s.get_sync("rater_profile_mode") == "blockscout"  # unchanged by a no-op run


def test_pass_two_runs_for_legacy_rows_even_when_pass_one_is_idle(tmp_path):
    # The upgrade path for a store profiled before the owner pass existed:
    # every rater already has a row, so pass 1 has no addresses, yet the
    # NULL-kind rows must still be looked up.
    with Store(tmp_path / "t.db") as s:
        seed(s, A)
        s.upsert_rater(A, None, None)
        rpc = OwnerRpc({A: KEY})
        assert enrich_raters(s, TransferClient(), rpc=rpc) == 1
        assert rows(s) == {A: (None, KEY, "owner")}
        assert s.get_sync("rater_profile_mode") == "blockscout"


# --- pass 2: failures --------------------------------------------------------------


def test_rpc_failure_in_pass_two_fails_only_that_address(tmp_path, caplog):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B, C)
        rpc = OwnerRpc({A: KEY}, fail=[B])
        with caplog.at_level(logging.ERROR, logger="robustrep.sources.rater_profile"):
            with pytest.raises(RuntimeError, match="1 address") as e:
                enrich_raters(s, TransferClient(), rpc=rpc)
        assert B in str(e.value) and A not in str(e.value) and C not in str(e.value)
        # B's own per-address scan died, so B alone stays NULL for the next
        # run; A (owner) and C (nothing anywhere) are settled.
        assert rows(s) == {A: (None, KEY, "owner"), B: (None, None, None), C: (None, None, "none")}
        assert s.get_sync("rater_profile_mode") == "blockscout-partial"
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1 and B in errors[0] and "owner lookup failed" in errors[0]


def test_whole_batch_failure_in_pass_two_still_settles_every_address_it_can(tmp_path, caplog):
    # The batch endpoint is down (transport); per-address calls still work
    # for everyone but B -> 49 of 50 settled, one WARNING for the fallback.
    addrs = ["0x" + f"{i:040x}" for i in range(50)]
    with Store(tmp_path / "t.db") as s:
        seed(s, *addrs)
        rpc = OwnerRpc({a: KEY for a in addrs}, fail=[addrs[7]], batch_error=RpcError("batch down"))
        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="1 address"):
                enrich_raters(s, TransferClient(), rpc=rpc)
        got = rows(s)
        assert got[addrs[7]] == (None, None, None)
        assert all(got[a] == (None, KEY, "owner") for a in addrs if a != addrs[7])
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_owner_pass_uses_the_callers_batch_state(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B)
        state = BatchState()
        rpc = OwnerRpc({A: KEY, B: KEY2}, batch_error=RpcBatchStructureError("not a list"))
        enrich_raters(s, TransferClient(), rpc=rpc, batch_state=state)
        assert state.batch_unsupported is True
        assert rows(s) == {A: (None, KEY, "owner"), B: (None, KEY2, "owner")}
        # A state that already knows batching is unsupported skips rpc.batch entirely.
        s.upsert_rater(C, None, None)
        again = OwnerRpc({C: KEY}, batch_error=RpcBatchStructureError("not a list"))
        enrich_raters(s, TransferClient(), rpc=again, batch_state=state)
        assert again.batches == [] and again.calls == [C]


def test_retry_failure_in_pass_two_is_collected_and_the_rest_still_settle(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B)
        client = RecordingAlchemy(raise_on_retry={A: RuntimeError("boom")})
        with pytest.raises(RuntimeError) as e:
            enrich_raters(s, client, rpc=OwnerRpc())
        assert "1 address(es)" in str(e.value) and A in str(e.value)
        assert rows(s) == {A: (None, None, None), B: (None, None, "none")}
        assert s.get_sync("rater_profile_mode") == "alchemy-partial"


def test_plan_error_on_the_very_first_client_call_aborts_even_in_pass_two(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B)
        s.upsert_rater(A, None, None)
        s.upsert_rater(B, None, None)
        client = RecordingAlchemy(raise_on_retry={A: ProfilePlanError("plan"), B: ProfilePlanError("plan")})
        with pytest.raises(ProfilePlanError):
            enrich_raters(s, client, rpc=OwnerRpc())
        assert len(client.calls) == 1
        assert rows(s) == {A: (None, None, None), B: (None, None, None)}
        assert s.get_sync("rater_profile_mode") is None


def test_plan_error_after_an_earlier_client_call_is_a_regular_failure(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A)
        client = RecordingAlchemy(raise_on_retry={A: ProfilePlanError("plan")})
        with pytest.raises(RuntimeError, match="1 address"):
            enrich_raters(s, client, rpc=OwnerRpc())
        assert rows(s) == {A: (None, None, None)}
        assert s.get_sync("rater_profile_mode") == "alchemy-partial"


def test_pass_one_failure_still_lets_pass_two_run(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B)
        s.upsert_rater(C, None, None)
        with pytest.raises(RuntimeError, match=B):
            enrich_raters(s, TransferClient(raise_for=[B]), rpc=OwnerRpc({A: KEY, C: KEY2}))
        assert rows(s) == {A: (None, KEY, "owner"), C: (None, KEY2, "owner")}


# --- logging -----------------------------------------------------------------------


def test_one_info_line_per_pass_with_counts(tmp_path, caplog):
    with Store(tmp_path / "t.db") as s:
        seed(s, A, B, C, D)
        client = RecordingAlchemy(answers={A: (1, 100, FUNDER)}, retry_answers={C: (2, 200, FUNDER)})
        with caplog.at_level(logging.INFO, logger="robustrep.sources.rater_profile"):
            assert enrich_raters(s, client, rpc=OwnerRpc({B: KEY})) == 7
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 2
        assert "pass 1" in infos[0] and "4 address(es)" in infos[0] and "1 funder(s) found" in infos[0]
        assert "pass 2" in infos[1]
        assert "owner: 1, transfer-internal: 1, none: 1" in infos[1]
        assert rows(s) == {A: (100, FUNDER, "transfer"), B: (None, KEY, "owner"),
                           C: (200, FUNDER, "transfer"), D: (None, None, "none")}


def test_fallback_mode_skips_pass_two_and_writes_nothing(tmp_path):
    with Store(tmp_path / "t.db") as s:
        seed(s, A)
        s.upsert_rater(B, None, None)
        rpc = OwnerRpc({A: KEY, B: KEY})
        assert enrich_raters(s, None, rpc=rpc) == 1
        assert rpc.queried() == set()
        assert rows(s) == {B: (None, None, None)}
        assert s.get_sync("rater_profile_mode") == "fallback"
