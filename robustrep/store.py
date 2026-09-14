"""SQLite persistence for raw chain events, enrichment caches and sync state.

One SQLite file holds everything the ``sources/base_erc8004.py`` adapter (and its
sibling enrichment tasks) need: raw ``FeedbackGiven``/``ResponseAppended`` events,
a block-number -> timestamp cache, an evidence-URI enrichment cache, rater/agent
metadata caches, sync-cursor bookkeeping and a log of failed block ranges to retry.

``load_records`` projects the raw tables into the eight-field record frame that
``robustrep.schema.validate_records`` expects (plus ``evidence_level`` and
``revoked``). Values are returned raw-but-well-typed: ``value`` is numeric
(``NaN`` for anything non-numeric — left for ``validate_records`` to reject),
``ts`` is an ``int`` that is ``0`` when the block timestamp is not yet known
(``validate_records`` accepts 0; ``n_missing_block_ts`` lets callers warn).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

SOURCE = "base-erc8004"

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback(
  chain TEXT, block INTEGER, tx_hash TEXT, log_index INTEGER, agent_id TEXT, client TEXT,
  feedback_index INTEGER, value TEXT, value_decimals INTEGER, tag1 TEXT, tag2 TEXT, endpoint TEXT,
  feedback_uri TEXT, feedback_hash TEXT, revoked INTEGER DEFAULT 0,
  PRIMARY KEY(agent_id, client, feedback_index));
CREATE TABLE IF NOT EXISTS responses(
  agent_id TEXT, client TEXT, feedback_index INTEGER, responder TEXT, response_uri TEXT,
  response_hash TEXT, block INTEGER, tx_hash TEXT, log_index INTEGER,
  PRIMARY KEY(tx_hash, log_index));
CREATE TABLE IF NOT EXISTS blocks(number INTEGER PRIMARY KEY, ts INTEGER);
CREATE TABLE IF NOT EXISTS evidence_cache(uri TEXT PRIMARY KEY, level INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS raters(address TEXT PRIMARY KEY, first_seen_ts INTEGER, funder TEXT);
CREATE TABLE IF NOT EXISTS agents(agent_id TEXT PRIMARY KEY, owner TEXT);
CREATE TABLE IF NOT EXISTS sync_state(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS failed_ranges(from_block INTEGER, to_block INTEGER, error TEXT,
  PRIMARY KEY(from_block, to_block));
CREATE INDEX IF NOT EXISTS idx_feedback_block ON feedback(block);
CREATE INDEX IF NOT EXISTS idx_feedback_client ON feedback(client);
CREATE INDEX IF NOT EXISTS idx_feedback_uri ON feedback(feedback_uri);
"""

FB_COLS = ["chain", "block", "tx_hash", "log_index", "agent_id", "client", "feedback_index", "value",
           "value_decimals", "tag1", "tag2", "endpoint", "feedback_uri", "feedback_hash"]

RESPONSE_COLS = ["agent_id", "client", "feedback_index", "responder", "response_uri", "response_hash",
                  "block", "tx_hash", "log_index"]


class Store:
    """SQLite-backed persistence for one chain adapter's raw events and caches.

    Opens (creating if needed) a single SQLite file at ``path``, applies the
    schema (idempotent ``CREATE TABLE/INDEX IF NOT EXISTS``), and enables WAL
    journaling for concurrent-friendly reads. Supports use as a context manager.
    """

    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.conn.close()

    # --- feedback -----------------------------------------------------------------
    def upsert_feedback(self, rows: Iterable[dict]) -> None:
        """Insert feedback rows, ignoring duplicates on (agent_id, client, feedback_index).

        Raises ``ValueError`` naming the missing column if a row lacks one of the
        required ``FB_COLS`` fields.
        """
        rows = list(rows)
        values = []
        for r in rows:
            row_values = []
            for c in FB_COLS:
                if c not in r:
                    raise ValueError(f"feedback row missing {c!r}")
                row_values.append(r[c])
            values.append(row_values)
        q = f"INSERT OR IGNORE INTO feedback({','.join(FB_COLS)}) VALUES({','.join('?' * len(FB_COLS))})"
        with self.conn:
            self.conn.executemany(q, values)

    def mark_revoked(self, agent_id: str, client: str, feedback_index: int) -> None:
        """Mark a specific feedback row as revoked."""
        with self.conn:
            self.conn.execute(
                "UPDATE feedback SET revoked=1 WHERE agent_id=? AND client=? AND feedback_index=?",
                (agent_id, client, feedback_index))

    def add_response(self, row: dict) -> None:
        """Insert a response row, ignoring duplicates on (tx_hash, log_index)."""
        with self.conn:
            self.conn.execute(
                f"INSERT OR IGNORE INTO responses({','.join(RESPONSE_COLS)}) "
                f"VALUES({','.join('?' * len(RESPONSE_COLS))})",
                [row[c] for c in RESPONSE_COLS])

    # --- caches (block timestamps, evidence, rater/agent metadata) ----------------
    def upsert_block_ts(self, pairs: Iterable[tuple[int, int]]) -> None:
        """Cache (block_number, unix_ts) pairs, replacing existing entries."""
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO blocks VALUES(?,?)", list(pairs))

    def missing_block_ts(self) -> list[int]:
        """Block numbers referenced by feedback rows with no cached timestamp yet."""
        rows = self.conn.execute(
            "SELECT DISTINCT f.block FROM feedback f LEFT JOIN blocks b ON b.number=f.block "
            "WHERE b.ts IS NULL ORDER BY f.block").fetchall()
        return [r[0] for r in rows]

    def n_missing_block_ts(self) -> int:
        """Count of distinct blocks referenced by feedback with no cached timestamp."""
        return len(self.missing_block_ts())

    def upsert_evidence(self, uri: str, level: int, note: str = "") -> None:
        """Cache an evidence level (and optional note) for a URI."""
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO evidence_cache VALUES(?,?,?)", (uri, level, note))

    def evidence_level(self, uri: str) -> Optional[int]:
        """Cached evidence level for a URI, or ``None`` if not yet enriched."""
        r = self.conn.execute("SELECT level FROM evidence_cache WHERE uri=?", (uri,)).fetchone()
        return None if r is None else int(r[0])

    def upsert_rater(self, address: str, first_seen_ts: Optional[int], funder: Optional[str]) -> None:
        """Cache rater profile metadata (first-seen timestamp, funder address)."""
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO raters VALUES(?,?,?)", (address, first_seen_ts, funder))

    def upsert_agent_owner(self, agent_id: str, owner: str) -> None:
        """Cache the owner address for an agent id."""
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO agents VALUES(?,?)", (agent_id, owner))

    def agent_owner(self, agent_id: str) -> Optional[str]:
        """Cached owner address for an agent id, or ``None`` if unknown."""
        r = self.conn.execute("SELECT owner FROM agents WHERE agent_id=?", (agent_id,)).fetchone()
        return None if r is None else r[0]

    # --- sync bookkeeping -----------------------------------------------------------
    def get_sync(self, key: str) -> Optional[str]:
        """Read a sync-cursor value (e.g. last synced block), or ``None`` if unset."""
        r = self.conn.execute("SELECT value FROM sync_state WHERE key=?", (key,)).fetchone()
        return None if r is None else r[0]

    def set_sync(self, key: str, value: str) -> None:
        """Persist a sync-cursor value."""
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO sync_state VALUES(?,?)", (key, value))

    def add_failed_range(self, from_block: int, to_block: int, error: str) -> None:
        """Record a block range that failed to sync, for later retry."""
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO failed_ranges VALUES(?,?,?)", (from_block, to_block, error))

    def pop_failed_ranges(self) -> list[tuple[int, int]]:
        """Return and clear all recorded failed block ranges, ordered by start block."""
        rows = self.conn.execute(
            "SELECT from_block,to_block FROM failed_ranges ORDER BY from_block").fetchall()
        with self.conn:
            self.conn.execute("DELETE FROM failed_ranges")
        return [(r[0], r[1]) for r in rows]

    # --- loaders (produce the pipeline-facing record frame) --------------------------
    def load_records(self) -> pd.DataFrame:
        """Project raw feedback + caches into the eight-field pipeline record frame.

        ``value`` is coerced to numeric (NaN on non-numeric strings; downstream
        ``validate_records`` rejects NaN). Large int128-scale values stored as TEXT
        lose float precision on this coercion — accepted, since the pipeline only
        needs relative magnitude, not exact integer values. ``ts`` is 0 when the
        block timestamp is not yet cached (see ``missing_block_ts``/``n_missing_block_ts``).
        """
        q = """SELECT f.client AS rater, f.agent_id AS ratee, f.value AS value,
                      ('d' || f.value_decimals) AS scale, f.tag1 AS tag, b.ts AS ts,
                      f.feedback_uri AS evidence_uri, ? AS source,
                      COALESCE(e.level, 0) AS evidence_level, f.revoked AS revoked
               FROM feedback f LEFT JOIN blocks b ON b.number=f.block
               LEFT JOIN evidence_cache e ON e.uri=f.feedback_uri"""
        df = pd.read_sql_query(q, self.conn, params=(SOURCE,))
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0).astype("int64")
        df["evidence_level"] = pd.to_numeric(df["evidence_level"], errors="coerce").fillna(0).astype("int64")
        df["revoked"] = pd.to_numeric(df["revoked"], errors="coerce").fillna(0).astype("int64")
        return df

    def load_rater_meta(self) -> pd.DataFrame:
        """Load cached rater metadata (rater address, first-seen ts, funder)."""
        return pd.read_sql_query("SELECT address AS rater, first_seen_ts, funder FROM raters", self.conn)

    def distinct_uris(self) -> list[str]:
        """Feedback URIs not yet present (empty ones excluded) in the evidence cache."""
        rows = self.conn.execute(
            "SELECT DISTINCT feedback_uri FROM feedback WHERE feedback_uri<>'' "
            "AND feedback_uri NOT IN (SELECT uri FROM evidence_cache)").fetchall()
        return [r[0] for r in rows]

    def distinct_clients(self) -> list[str]:
        """Client (rater) addresses not yet profiled in the raters cache."""
        rows = self.conn.execute(
            "SELECT DISTINCT client FROM feedback WHERE client NOT IN "
            "(SELECT address FROM raters)").fetchall()
        return [r[0] for r in rows]

    def distinct_agents(self) -> list[str]:
        """Agent ids not yet resolved in the agents (owner) cache."""
        rows = self.conn.execute(
            "SELECT DISTINCT agent_id FROM feedback WHERE agent_id NOT IN "
            "(SELECT agent_id FROM agents)").fetchall()
        return [r[0] for r in rows]
