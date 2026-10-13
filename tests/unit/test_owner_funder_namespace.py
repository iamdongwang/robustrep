"""Owner-kind funders live in their own namespace (security review M1).

``ownerAtIndex`` is self-reported by a contract the rater deployed, so a
controlling key must never compare equal to a transfer-kind funder: it is
stored as ``owner:<0x40hex lowercase>`` and, since ``sybil._norm_funder`` only
lowercases, equals only another ``owner:...`` value. No network anywhere here.
"""
import pandas as pd
import pytest

from robustrep.config import Config
from robustrep.sources.account_owner import OWNER_FUNDER_PREFIX, is_owner_funder, owner_funder
from robustrep.sources.profile_http import funder_address
from robustrep.store import Store
from robustrep.sybil import RaterProfile, _signals, cluster_raters, profiles_from_records

KEY_UP = "0x" + "AB" * 20
KEY = KEY_UP.lower()
OWNER = "owner:" + KEY
A, B = "0x" + "a1" * 20, "0x" + "b2" * 20


# --- helpers ----------------------------------------------------------------------


def test_owner_funder_prefixes_and_lowercases_a_valid_address():
    assert OWNER_FUNDER_PREFIX == "owner:"
    assert owner_funder(KEY_UP) == OWNER
    assert owner_funder(KEY) == OWNER
    assert is_owner_funder(OWNER) and is_owner_funder("owner:" + "0x" + "00" * 20)


@pytest.mark.parametrize("bad", ["", "0x", "0x" + "ab" * 19, "ab" * 20, KEY + "0", OWNER, None, 12, "0xzz" + "a" * 38])
def test_owner_funder_rejects_anything_but_an_address(bad):
    with pytest.raises(ValueError, match="owner"):
        owner_funder(bad)


@pytest.mark.parametrize("value", [KEY, KEY_UP, "", None, 7, "owner:", "owner:0x12", "owner:" + KEY_UP,
                                   "Owner:" + KEY, " owner:" + KEY, "owner:" + KEY + "0"])
def test_is_owner_funder_is_strict(value):
    assert is_owner_funder(value) is False


def test_funder_address_still_rejects_the_prefixed_form_where_a_raw_address_is_required(caplog):
    assert funder_address(OWNER, A) is None
    assert funder_address(KEY_UP, A) == KEY


# --- store round-trip -------------------------------------------------------------


def test_store_round_trips_the_prefixed_owner_funder(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_rater(A, None, OWNER, "owner")
        meta = s.load_rater_meta()
        assert meta.iloc[0]["funder"] == OWNER
        assert s.funder_kind_counts()["owner"] == 1
        assert s.raters_needing_owner_lookup() == []


# --- sybil: profiles keep it as given; the two kinds never share a funder signal ---


def test_profiles_from_records_keeps_the_prefixed_funder_as_given():
    records = pd.DataFrame([dict(rater=A, ratee="1", ts=10), dict(rater=B, ratee="1", ts=20)])
    meta = pd.DataFrame([dict(rater=A, first_seen_ts=None, funder=OWNER),
                         dict(rater=B, first_seen_ts=5, funder=KEY_UP)])
    profiles = {p.rater: p for p in profiles_from_records(records, meta)}
    assert profiles[A].funder == OWNER and profiles[A].first_seen_ts == 10   # NULL -> earliest feedback
    assert profiles[B].funder == KEY_UP and profiles[B].first_seen_ts == 5


def _p(r, ts, funder, ratees=("1",)):
    return RaterProfile(rater=r, first_seen_ts=ts, funder=funder, ratees=frozenset(ratees))


def test_transfer_funder_x_and_owner_x_do_not_share_the_funder_signal():
    cfg = Config()
    transfer, owner = _p(A, 0, KEY), _p(B, 0, OWNER)
    # Same window, same ratee (Jaccard 1.0): two signals already. Funder would
    # be a third -- and the pairwise count must stop at 2, not 3.
    assert _signals(transfer, owner, cfg) == 2
    # Out of window and disjoint ratees: nothing left but funder, which must not fire.
    far = _p(B, 10 * 24 * 3600, OWNER, ("2",))
    assert _signals(transfer, far, cfg) == 0
    assert cluster_raters([transfer, far], cfg) == {A: A, B: B}


def test_two_owner_kind_raters_with_the_same_key_do_cluster():
    cfg = Config()
    assert cluster_raters([_p(A, 0, OWNER), _p(B, 3600, "OWNER:" + KEY_UP)], cfg) == {A: A, B: A}
    assert _signals(_p(A, 0, OWNER), _p(B, 0, OWNER, ("2",)), cfg) == 2  # funder + window
