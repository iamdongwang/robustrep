import time

import numpy as np
import pytest

from robustrep.store import Store

FB = dict(chain="base", block=100, tx_hash="0xt", log_index=0, agent_id="7", client="0xc",
          feedback_index=1, value="87", value_decimals=0, tag1="quality", tag2="", endpoint="",
          feedback_uri="https://e", feedback_hash="0xh")


def test_roundtrip_feedback_and_records(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([FB, {**FB, "feedback_index": 2, "value": "50"}])
    s.upsert_block_ts([(100, 1_700_000_000)])
    s.upsert_evidence("https://e", 2)
    df = s.load_records()
    assert len(df) == 2 and set(df.columns) >= {"rater", "ratee", "value", "scale", "tag", "ts",
                                                 "evidence_uri", "source", "evidence_level", "revoked"}
    assert df["ts"].iloc[0] == 1_700_000_000 and df["evidence_level"].iloc[0] == 2
    assert df["scale"].iloc[0] == "d0" and df["source"].iloc[0] == "base-erc8004"


def test_upsert_is_idempotent(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([FB]); s.upsert_feedback([FB])
    assert len(s.load_records()) == 1


def test_revoke_and_sync_state(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([FB]); s.upsert_block_ts([(100, 1)])
    s.mark_revoked("7", "0xc", 1)
    assert s.load_records()["revoked"].iloc[0] == 1
    assert s.get_sync("last_block") is None
    s.set_sync("last_block", "123")
    assert s.get_sync("last_block") == "123"


def test_failed_ranges_and_missing_blocks(tmp_path):
    s = Store(tmp_path / "t.db")
    s.add_failed_range(1, 2000, "boom")
    assert s.pop_failed_ranges() == [(1, 2000)] and s.pop_failed_ranges() == []
    s.upsert_feedback([FB, {**FB, "block": 101, "feedback_index": 3}])
    assert s.missing_block_ts() == [100, 101]


def test_raters_and_agents(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_rater("0xc", first_seen_ts=5, funder="0xf")
    s.upsert_agent_owner("7", "0xowner")
    assert s.load_rater_meta().iloc[0]["funder"] == "0xf"
    assert s.agent_owner("7") == "0xowner" and s.agent_owner("8") is None


def test_load_records_value_kept_as_string_precision(tmp_path):
    s = Store(tmp_path / "t.db")
    big = "123456789012345678901234567890"
    s.upsert_feedback([
        {**FB, "value": big},
        {**FB, "feedback_index": 2, "value": "abc"},
    ])
    df = s.load_records()
    # int128-scale value is stored as TEXT and comes back as a float (precision loss
    # accepted downstream; validate_records coerces to float64 anyway).
    assert np.isclose(df["value"].iloc[0], float(big))
    assert df["value"].isna().any()


def test_missing_ts_is_zero_and_flagged(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([FB])
    df = s.load_records()
    assert df["ts"].iloc[0] == 0
    assert s.n_missing_block_ts() == 1
    s.upsert_block_ts([(100, 1)])
    assert s.n_missing_block_ts() == 0


def test_context_manager_and_close(tmp_path):
    path = tmp_path / "t.db"
    with Store(path) as s:
        s.upsert_feedback([FB])
    with pytest.raises(Exception):
        s.conn.execute("SELECT 1")

    s2 = Store(path)
    assert len(s2.load_records()) == 1
    s2.close()
    with pytest.raises(Exception):
        s2.conn.execute("SELECT 1")


def test_wal_and_indexes(tmp_path):
    s = Store(tmp_path / "t.db")
    mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    idx_sql = " ".join(r[0] or "" for r in
                        s.conn.execute("SELECT sql FROM sqlite_master WHERE type='index'").fetchall())
    assert "feedback" in idx_sql
    names = [r[0] for r in s.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='feedback'").fetchall()]
    assert len(names) >= 3


def test_upsert_feedback_rejects_missing_column(tmp_path):
    s = Store(tmp_path / "t.db")
    bad = {k: v for k, v in FB.items() if k != "tx_hash"}
    with pytest.raises(ValueError, match="tx_hash"):
        s.upsert_feedback([bad])


def test_bulk_insert_performance(tmp_path):
    s = Store(tmp_path / "t.db")
    rows = [{**FB, "feedback_index": i} for i in range(50_000)]
    start = time.monotonic()
    s.upsert_feedback(rows)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
    assert len(s.load_records()) == 50_000


def test_distinct_helpers(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([
        FB,
        {**FB, "feedback_index": 2, "client": "0xd", "agent_id": "8", "feedback_uri": "https://e2"},
        {**FB, "feedback_index": 3, "client": "0xe", "agent_id": "9", "feedback_uri": ""},
    ])
    s.upsert_evidence("https://e", 1)
    s.upsert_rater("0xc", first_seen_ts=1, funder=None)
    s.upsert_agent_owner("7", "0xowner")
    assert s.distinct_uris() == ["https://e2"]
    assert set(s.distinct_clients()) == {"0xd", "0xe"}
    assert set(s.distinct_agents()) == {"8", "9"}


def test_add_response_idempotent(tmp_path):
    s = Store(tmp_path / "t.db")
    resp = dict(agent_id="7", client="0xc", feedback_index=1, responder="0xr",
                response_uri="https://r", response_hash="0xrh", block=101, tx_hash="0xrt", log_index=0)
    s.add_response(resp)
    s.add_response(resp)
    n = s.conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    assert n == 1


def test_load_records_dtypes(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([FB])
    s.upsert_block_ts([(100, 1_700_000_000)])
    s.upsert_evidence("https://e", 2)
    df = s.load_records()
    assert df["evidence_level"].iloc[0] == 2 and str(df["evidence_level"].dtype).startswith("int")
    assert df["revoked"].iloc[0] == 0 and str(df["revoked"].dtype).startswith("int")
    assert str(df["ts"].dtype).startswith("int")
    assert str(df["value"].dtype) == "float64"
    assert df["scale"].dtype == object and isinstance(df["scale"].iloc[0], str)
