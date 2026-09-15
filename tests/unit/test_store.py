import pandas as pd
import logging
import sqlite3
import time

import numpy as np
import pytest

from robustrep.store import Store

FB = dict(chain="base", block=100, tx_hash="0xt", log_index=0, agent_id="7", client="0xc",
          feedback_index=1, value="87", value_decimals=0, tag1="quality", tag2="", endpoint="",
          feedback_uri="https://e", feedback_hash="0xh")


def test_roundtrip_feedback_and_records(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([FB, {**FB, "feedback_index": 2, "value": "50"}])
        s.upsert_block_ts([(100, 1_700_000_000)])
        s.upsert_evidence("https://e", 2)
        df = s.load_records()
        assert len(df) == 2 and set(df.columns) >= {"rater", "ratee", "value", "scale", "tag", "ts",
                                                     "evidence_uri", "source", "evidence_level", "revoked"}
        assert df["ts"].iloc[0] == 1_700_000_000 and df["evidence_level"].iloc[0] == 2
        assert df["scale"].iloc[0] == "d0" and df["source"].iloc[0] == "base-erc8004"


def test_upsert_is_idempotent(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([FB]); s.upsert_feedback([FB])
        assert len(s.load_records()) == 1


def test_revoke_and_sync_state(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([FB]); s.upsert_block_ts([(100, 1)])
        s.mark_revoked("7", "0xc", 1)
        assert s.load_records()["revoked"].iloc[0] == 1
        assert s.get_sync("last_block") is None
        s.set_sync("last_block", "123")
        assert s.get_sync("last_block") == "123"


def test_failed_ranges_and_missing_blocks(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.add_failed_range(1, 2000, "boom")
        assert s.pop_failed_ranges() == [(1, 2000)] and s.pop_failed_ranges() == []
        s.upsert_feedback([FB, {**FB, "block": 101, "feedback_index": 3}])
        assert s.missing_block_ts() == [100, 101]


def test_raters_and_agents(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_rater("0xc", first_seen_ts=5, funder="0xf")
        s.upsert_agent_owner("7", "0xowner")
        assert s.load_rater_meta().iloc[0]["funder"] == "0xf"
        assert s.agent_owner("7") == "0xowner" and s.agent_owner("8") is None


def test_load_records_value_kept_as_string_precision(tmp_path):
    with Store(tmp_path / "t.db") as s:
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
    with Store(tmp_path / "t.db") as s:
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
    with Store(tmp_path / "t.db") as s:
        mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        idx_sql = " ".join(r[0] or "" for r in
                            s.conn.execute("SELECT sql FROM sqlite_master WHERE type='index'").fetchall())
        assert "feedback" in idx_sql
        names = [r[0] for r in s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='feedback'").fetchall()]
        assert len(names) >= 3


def test_upsert_feedback_rejects_missing_column(tmp_path):
    with Store(tmp_path / "t.db") as s:
        bad = {k: v for k, v in FB.items() if k != "tx_hash"}
        with pytest.raises(ValueError, match="tx_hash"):
            s.upsert_feedback([bad])


def test_bulk_insert_performance(tmp_path):
    with Store(tmp_path / "t.db") as s:
        rows = [{**FB, "feedback_index": i} for i in range(50_000)]
        start = time.monotonic()
        s.upsert_feedback(rows)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0
        assert len(s.load_records()) == 50_000


def test_distinct_helpers(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([
            FB,
            {**FB, "feedback_index": 2, "client": "0xd", "agent_id": "8", "feedback_uri": "https://e2"},
            {**FB, "feedback_index": 3, "client": "0xe", "agent_id": "9", "feedback_uri": ""},
        ])
        s.upsert_evidence("https://e", 1)
        s.upsert_rater("0xc", first_seen_ts=1, funder=None)
        s.upsert_agent_owner("7", "0xowner")
        assert set(s.distinct_clients()) == {"0xd", "0xe"}
        assert set(s.distinct_agents()) == {"8", "9"}


def test_n_lookup_budget_starved_counts_only_lookup_budget_notes(tmp_path):
    with Store(tmp_path / "tstarved.db") as s:
        s.upsert_evidence("https://a", 2, "lookup-budget:1")
        s.upsert_evidence("https://b", 2, "lookup-budget:2")
        s.upsert_evidence("https://c", 3, "")
        s.upsert_evidence("https://d", 1, "unfetchable")
        assert s.n_lookup_budget_starved() == 2


def test_upsert_evidence_coerces_none_note_to_empty_string(tmp_path):
    # A caller passing note=None explicitly (rather than relying on the ""
    # default) must not be able to write a SQL NULL note -- pending_uris'
    # retry filter treats NULL specially (see its NULL-safety test below),
    # and a NULL note must never be reachable through the normal write path.
    with Store(tmp_path / "tnone.db") as s:
        s.upsert_evidence("https://n", 1, None)
        row = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://n'").fetchone()
        assert row[0] == ""


def test_pending_uris_default_excludes_anything_cached(tmp_path):
    with Store(tmp_path / "tpending.db") as s:
        s.upsert_feedback([
            FB,  # feedback_uri "https://e"
            {**FB, "feedback_index": 2, "feedback_uri": "https://never"},
            {**FB, "feedback_index": 3, "feedback_uri": "https://budgeted"},
            {**FB, "feedback_index": 4, "feedback_uri": "https://miss"},
            {**FB, "feedback_index": 5, "feedback_uri": ""},
        ])
        s.upsert_evidence("https://e", 3)  # ordinary cached result -- note ""
        s.upsert_evidence("https://budgeted", 2, "lookup-budget:1")
        s.upsert_evidence("https://miss", 1, "unfetchable")
        # With no include_notes, only the never-cached URI is pending -- matches
        # the original "not yet cached at all" NOT EXISTS behavior exactly.
        assert s.pending_uris() == [("https://never", "0xc", None, None)]


def test_pending_uris_retries_only_the_given_notes(tmp_path):
    with Store(tmp_path / "tpending2.db") as s:
        s.upsert_feedback([
            FB,
            {**FB, "feedback_index": 2, "feedback_uri": "https://never"},
            {**FB, "feedback_index": 3, "feedback_uri": "https://budgeted"},
            {**FB, "feedback_index": 4, "feedback_uri": "https://miss"},
        ])
        s.upsert_evidence("https://e", 3)
        s.upsert_evidence("https://budgeted", 2, "lookup-budget:1")
        s.upsert_evidence("https://miss", 1, "unfetchable")
        pending = {row[0] for row in s.pending_uris(include_notes=("lookup-budget:1",))}
        # The never-cached URI and the retryable "lookup-budget:1" one are
        # pending; the ordinary result and the (not included) "unfetchable" one
        # are not.
        assert pending == {"https://never", "https://budgeted"}


def test_pending_uris_returns_prior_note_for_a_retryable_uri(tmp_path):
    with Store(tmp_path / "tpending3.db") as s:
        s.upsert_feedback([{**FB, "feedback_uri": "https://retry"}])
        s.upsert_evidence("https://retry", 2, "lookup-budget:1")
        rows = s.pending_uris(include_notes=("lookup-budget:1",))
        assert rows == [("https://retry", "0xc", None, "lookup-budget:1")]


def test_pending_uris_null_note_is_not_pending_forever_on_retry_path(tmp_path):
    # A row with a SQL NULL note (e.g. written before upsert_evidence started
    # coercing None -> "", or by a direct INSERT bypassing it) must not
    # compare unequal to every literal in the retry IN-clause forever: a bare
    # `e.note NOT IN (...)` evaluates to SQL NULL (neither true nor false)
    # against a NULL note, which would make NOT EXISTS see nothing blocking
    # and keep the URI "pending" no matter what include_notes says.
    with Store(tmp_path / "tpendingnull.db") as s:
        s.upsert_feedback([{**FB, "feedback_uri": "https://nullnote"}])
        with s.conn:
            s.conn.execute("INSERT INTO evidence_cache VALUES(?,?,?)", ("https://nullnote", 1, None))
        pending = {row[0] for row in s.pending_uris(include_notes=("lookup-budget:1",))}
        assert "https://nullnote" not in pending


def test_add_response_idempotent(tmp_path):
    with Store(tmp_path / "t.db") as s:
        resp = dict(agent_id="7", client="0xc", feedback_index=1, responder="0xr",
                    response_uri="https://r", response_hash="0xrh", block=101, tx_hash="0xrt", log_index=0)
        s.add_response(resp)
        s.add_response(resp)
        n = s.conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        assert n == 1


def test_load_records_dtypes(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([FB])
        s.upsert_block_ts([(100, 1_700_000_000)])
        s.upsert_evidence("https://e", 2)
        df = s.load_records()
        assert df["evidence_level"].iloc[0] == 2 and str(df["evidence_level"].dtype).startswith("int")
        assert df["revoked"].iloc[0] == 0 and str(df["revoked"].dtype).startswith("int")
        assert str(df["ts"].dtype).startswith("int")
        assert str(df["value"].dtype) == "float64"
        assert pd.api.types.is_string_dtype(df["scale"]) and isinstance(df["scale"].iloc[0], str)


# --- fixes: order-independent revocations, NULL-safe distinct, atomic pop, conflict visibility ---

def test_revocation_before_insert_survives(tmp_path):
    """A FeedbackRevoked event replayed (e.g. from a retried failed range) before its
    FeedbackGiven row exists must not be lost when the feedback row is later inserted."""
    with Store(tmp_path / "t.db") as s:
        s.mark_revoked("7", "0xc", 1, block=99, tx_hash="0xrev")
        s.upsert_feedback([FB])
        assert s.load_records()["revoked"].iloc[0] == 1


def test_revocation_after_insert_still_works(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([FB])
        s.mark_revoked("7", "0xc", 1)
        assert s.load_records()["revoked"].iloc[0] == 1


def test_mark_revoked_idempotent(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.mark_revoked("7", "0xc", 1)
        s.mark_revoked("7", "0xc", 1)
        n = s.conn.execute("SELECT COUNT(*) FROM revocations").fetchone()[0]
        assert n == 1


@pytest.mark.parametrize("table,columns,values", [
    ("evidence_cache", "uri,level,note", (None, 1, "")),
    ("raters", "address,first_seen_ts,funder", (None, 1, "f")),
    ("agents", "agent_id,owner", (None, "0xo")),
    ("blocks", "number,ts", (None, 1)),
])
def test_null_primary_key_rejected(tmp_path, table, columns, values):
    with Store(tmp_path / "t.db") as s:
        placeholders = ",".join("?" * len(values))
        with pytest.raises(sqlite3.IntegrityError):
            s.conn.execute(f"INSERT INTO {table}({columns}) VALUES({placeholders})", values)


def test_upsert_evidence_rejects_none_uri(tmp_path):
    with Store(tmp_path / "t.db") as s:
        with pytest.raises(ValueError, match="evidence_cache"):
            s.upsert_evidence(None, 1)


def test_upsert_rater_rejects_none_address(tmp_path):
    with Store(tmp_path / "t.db") as s:
        with pytest.raises(ValueError, match="raters"):
            s.upsert_rater(None, first_seen_ts=1, funder="f")


def test_upsert_agent_owner_rejects_none_agent_id(tmp_path):
    with Store(tmp_path / "t.db") as s:
        with pytest.raises(ValueError, match="agents"):
            s.upsert_agent_owner(None, "0xo")


def test_upsert_block_ts_rejects_none_number(tmp_path):
    with Store(tmp_path / "t.db") as s:
        with pytest.raises(ValueError, match="blocks"):
            s.upsert_block_ts([(None, 1)])


def test_pop_failed_ranges_clears_and_repeats(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.add_failed_range(1, 100, "boom")
        assert s.pop_failed_ranges() == [(1, 100)]
        # a range added after the pop must not be lost or duplicated on the next pop
        s.add_failed_range(200, 300, "again")
        assert s.pop_failed_ranges() == [(200, 300)]
        assert s.pop_failed_ranges() == []


def test_upsert_feedback_returns_counts(tmp_path):
    with Store(tmp_path / "t.db") as s:
        counts = s.upsert_feedback([FB, {**FB, "feedback_index": 2}])
        assert counts == (2, 0)
        counts = s.upsert_feedback([FB, {**FB, "feedback_index": 2}])
        assert counts == (0, 2)


def test_upsert_feedback_logs_on_ignored(tmp_path, caplog):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([FB])
        with caplog.at_level(logging.INFO, logger="robustrep.store"):
            s.upsert_feedback([FB])
        assert any("ignored" in r.message.lower() for r in caplog.records)


def test_evidence_level_lookup(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_evidence("https://e", 2)
        assert s.evidence_level("https://e") == 2
        assert s.evidence_level("https://unknown") is None


def test_add_response_rejects_missing_column(tmp_path):
    with Store(tmp_path / "t.db") as s:
        resp = dict(agent_id="7", client="0xc", feedback_index=1, responder="0xr",
                    response_uri="https://r", response_hash="0xrh", block=101, tx_hash="0xrt", log_index=0)
        bad = {k: v for k, v in resp.items() if k != "responder"}
        with pytest.raises(ValueError, match="responder"):
            s.add_response(bad)


def test_responses_index_exists(tmp_path):
    with Store(tmp_path / "t.db") as s:
        names = [r[0] for r in s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='responses'").fetchall()]
        assert len(names) >= 1


def test_store_creates_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "c.db"
    with Store(nested) as s:
        s.upsert_feedback([FB])
        assert nested.exists()


def test_store_in_memory_still_works():
    with Store(":memory:") as s:
        s.upsert_feedback([FB])
        assert len(s.load_records()) == 1


def test_stale_schema_guard(tmp_path):
    """A pre-revocations database with a legacy feedback.revoked column must fail
    loudly rather than silently losing revocations (see fix in the previous commit)."""
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(str(path))
    legacy.execute("""CREATE TABLE feedback(
        chain TEXT, block INTEGER, tx_hash TEXT, log_index INTEGER, agent_id TEXT, client TEXT,
        feedback_index INTEGER, value TEXT, value_decimals INTEGER, tag1 TEXT, tag2 TEXT, endpoint TEXT,
        feedback_uri TEXT, feedback_hash TEXT, revoked INTEGER DEFAULT 0,
        PRIMARY KEY(agent_id, client, feedback_index))""")
    legacy.commit()
    legacy.close()
    with pytest.raises(ValueError, match="stale"):
        Store(path)


# --- M4: clearing the rater-profile cache -------------------------------------------


def test_clear_raters_empties_the_table_and_returns_the_row_count(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([FB])
        s.upsert_rater("0xc", 1_700_000_000, "0xf")
        s.upsert_rater("0xd", None, None)
        assert s.clear_raters() == 2
        assert len(s.load_rater_meta()) == 0
        # ...and the addresses are up for profiling again.
        assert s.distinct_clients() == ["0xc"]


def test_clear_raters_resets_the_profile_mode(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_rater("0xc", 1, None)
        s.set_sync("rater_profile_mode", "blockscout-partial")
        s.set_sync("last_block", "123")
        s.clear_raters()
        assert s.get_sync("rater_profile_mode") is None
        assert s.get_sync("last_block") == "123"  # other sync state is untouched


def test_clear_raters_on_an_empty_table_returns_zero(tmp_path):
    with Store(tmp_path / "t.db") as s:
        assert s.clear_raters() == 0


def test_clear_raters_lets_a_wedging_timestamp_be_reprofiled(tmp_path):
    # A single implausible cached first_seen_ts wedges score/report (sybil
    # ._validate_meta raises on it); clear_raters is the escape hatch.
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([FB])
        s.upsert_rater("0xc", -1, "0xf")
        assert s.distinct_clients() == []
        s.clear_raters()
        assert s.distinct_clients() == ["0xc"]


def test_n_lookup_budget_starved_counts_the_legacy_bare_note(tmp_path):
    # 7f391ef briefly wrote the bare note "lookup-budget" (no ":N"); a store
    # written by that build must still be counted as starved, or a report
    # would silently under-state how many cached levels are under-checked.
    with Store(tmp_path / "tlegacy.db") as s:
        s.upsert_evidence("https://legacy", 2, "lookup-budget")
        s.upsert_evidence("https://a", 2, "lookup-budget:1")
        s.upsert_evidence("https://c", 3, "")
        s.upsert_evidence("https://d", 1, "unfetchable")
        assert s.n_lookup_budget_starved() == 2
