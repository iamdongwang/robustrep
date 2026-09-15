"""SQLite persistence for raw chain events, enrichment caches and sync state.

One SQLite file holds everything the ``sources/base_erc8004.py`` adapter (and its
sibling enrichment tasks) need: raw ``NewFeedback``/``FeedbackRevoked``/
``ResponseAppended`` chain events, a block-number -> timestamp cache, an
evidence-URI enrichment cache, rater/agent metadata caches, sync-cursor
bookkeeping and a log of failed block ranges to retry.

``load_records`` projects the raw tables into the eight-field record frame that
``robustrep.schema.validate_records`` expects (plus ``evidence_level`` and
``revoked``). Values are returned raw-but-well-typed: ``value`` is numeric
(``NaN`` for anything non-numeric — left for ``validate_records`` to reject),
``ts`` is an ``int`` that is ``0`` when the block timestamp is not yet known
(``validate_records`` accepts 0; ``n_missing_block_ts`` lets callers warn).

A ``Store`` wraps one ``sqlite3.Connection`` and is a single-writer object: it is
not safe to share across threads (``sqlite3`` connections default to
``check_same_thread=True``, and no locking is added here) — open one ``Store``
per thread/process if you need concurrent writers. ``load_records`` materializes
the whole feedback table into memory as a ``pandas.DataFrame`` (at roughly
500k rows this is on the order of ~200 MB); callers processing much larger
histories should page or filter at the SQL level instead.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SOURCE = "base-erc8004"

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback(
  chain TEXT, block INTEGER, tx_hash TEXT, log_index INTEGER, agent_id TEXT, client TEXT,
  feedback_index INTEGER, value TEXT, value_decimals INTEGER, tag1 TEXT, tag2 TEXT, endpoint TEXT,
  feedback_uri TEXT, feedback_hash TEXT,
  PRIMARY KEY(agent_id, client, feedback_index));
CREATE TABLE IF NOT EXISTS revocations(
  agent_id TEXT NOT NULL, client TEXT NOT NULL, feedback_index INTEGER NOT NULL,
  block INTEGER, tx_hash TEXT,
  PRIMARY KEY(agent_id, client, feedback_index));
CREATE TABLE IF NOT EXISTS responses(
  agent_id TEXT, client TEXT, feedback_index INTEGER, responder TEXT, response_uri TEXT,
  response_hash TEXT, block INTEGER, tx_hash TEXT, log_index INTEGER,
  PRIMARY KEY(tx_hash, log_index));
-- number uses "INT" (not "INTEGER") PRIMARY KEY: a bare INTEGER PRIMARY KEY column
-- is a rowid alias in SQLite, and inserting NULL into it silently auto-assigns a
-- new rowid instead of enforcing NOT NULL. "INT" avoids that rowid-alias special
-- case so NULL block numbers are correctly rejected.
CREATE TABLE IF NOT EXISTS blocks(number INT PRIMARY KEY NOT NULL, ts INTEGER);
CREATE TABLE IF NOT EXISTS evidence_cache(uri TEXT PRIMARY KEY NOT NULL, level INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS raters(address TEXT PRIMARY KEY NOT NULL, first_seen_ts INTEGER, funder TEXT);
CREATE TABLE IF NOT EXISTS agents(agent_id TEXT PRIMARY KEY NOT NULL, owner TEXT);
CREATE TABLE IF NOT EXISTS sync_state(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS failed_ranges(from_block INTEGER, to_block INTEGER, error TEXT,
  PRIMARY KEY(from_block, to_block));
CREATE INDEX IF NOT EXISTS idx_feedback_block ON feedback(block);
CREATE INDEX IF NOT EXISTS idx_feedback_client ON feedback(client);
CREATE INDEX IF NOT EXISTS idx_feedback_uri ON feedback(feedback_uri);
CREATE INDEX IF NOT EXISTS idx_responses_key ON responses(agent_id, client, feedback_index);
"""

FB_COLS = ["chain", "block", "tx_hash", "log_index", "agent_id", "client", "feedback_index", "value",
           "value_decimals", "tag1", "tag2", "endpoint", "feedback_uri", "feedback_hash"]

RESPONSE_COLS = ["agent_id", "client", "feedback_index", "responder", "response_uri", "response_hash",
                  "block", "tx_hash", "log_index"]


def _row_values(row: dict, cols: list[str], what: str) -> list:
    """Pull ``cols`` out of ``row`` in order, raising ``ValueError`` naming the
    first missing column instead of letting a bare ``KeyError`` escape."""
    values = []
    for c in cols:
        if c not in row:
            raise ValueError(f"{what} row missing {c!r}")
        values.append(row[c])
    return values


class Store:
    """SQLite-backed persistence for one chain adapter's raw events and caches.

    Opens (creating if needed) a single SQLite file at ``path``, applies the
    schema (idempotent ``CREATE TABLE/INDEX IF NOT EXISTS``), and enables WAL
    journaling for concurrent-friendly reads. Supports use as a context manager.

    Not thread-shareable: this wraps one ``sqlite3.Connection`` opened with the
    default ``check_same_thread=True`` and does no internal locking — treat a
    ``Store`` as owned by a single writer (thread/process) at a time.
    """

    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            parent = Path(path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(feedback)").fetchall()}
        if "revoked" in cols:
            self.conn.close()
            raise ValueError(
                "stale robustrep schema (feedback.revoked column present): "
                "delete the database file and re-run fetch")

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.conn.close()

    # --- feedback -----------------------------------------------------------------
    def upsert_feedback(self, rows: Iterable[dict]) -> tuple[int, int]:
        """Insert feedback rows, ignoring duplicates on (agent_id, client, feedback_index).

        Returns ``(inserted, ignored)`` counts and logs at INFO when any rows were
        ignored as duplicates. Raises ``ValueError`` naming the missing column if a
        row lacks one of the required ``FB_COLS`` fields.
        """
        rows = list(rows)
        values = [_row_values(r, FB_COLS, "feedback") for r in rows]
        q = f"INSERT OR IGNORE INTO feedback({','.join(FB_COLS)}) VALUES({','.join('?' * len(FB_COLS))})"
        with self.conn:
            cur = self.conn.executemany(q, values)
            inserted = cur.rowcount if cur.rowcount >= 0 else 0
        ignored = len(rows) - inserted
        if ignored > 0:
            logger.info("upsert_feedback: %d inserted, %d ignored (already present)", inserted, ignored)
        return inserted, ignored

    def mark_revoked(self, agent_id: str, client: str, feedback_index: int,
                      block: Optional[int] = None, tx_hash: Optional[str] = None) -> None:
        """Record a revocation for (agent_id, client, feedback_index).

        Written to a standalone ``revocations`` table (not a column on ``feedback``)
        so that a ``FeedbackRevoked`` event replayed before its matching
        ``NewFeedback`` row has been inserted (e.g. a retried failed block range
        landing out of order) is never lost: ``load_records`` computes ``revoked``
        by joining against this table, so the revocation is visible as soon as the
        feedback row eventually arrives, regardless of insertion order.
        """
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO revocations(agent_id, client, feedback_index, block, tx_hash) "
                "VALUES(?,?,?,?,?)",
                (agent_id, client, feedback_index, block, tx_hash))

    def add_response(self, row: dict) -> None:
        """Insert a response row, ignoring duplicates on (tx_hash, log_index)."""
        values = _row_values(row, RESPONSE_COLS, "response")
        with self.conn:
            self.conn.execute(
                f"INSERT OR IGNORE INTO responses({','.join(RESPONSE_COLS)}) "
                f"VALUES({','.join('?' * len(RESPONSE_COLS))})",
                values)

    # --- caches (block timestamps, evidence, rater/agent metadata) ----------------
    def upsert_block_ts(self, pairs: Iterable[tuple[int, int]]) -> None:
        """Cache (block_number, unix_ts) pairs, replacing existing entries."""
        try:
            with self.conn:
                self.conn.executemany("INSERT OR REPLACE INTO blocks VALUES(?,?)", list(pairs))
        except sqlite3.IntegrityError as e:
            raise ValueError(f"blocks: {e}") from e

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
        """Cache an evidence level (and optional note) for a URI.

        ``note=None`` is coerced to ``""`` rather than stored as SQL NULL: a
        NULL note would make ``COALESCE(e.note, '')`` in
        ``pending_uris``'s retry check compare equal to a real ``""`` note --
        harmless there -- but would also compare unequal to every literal
        note string in an ``IN (...)`` clause without that COALESCE, which is
        exactly the bug a caller passing ``None`` (instead of the default)
        must not be able to reintroduce.
        """
        note = "" if note is None else note
        try:
            with self.conn:
                self.conn.execute("INSERT OR REPLACE INTO evidence_cache VALUES(?,?,?)", (uri, level, note))
        except sqlite3.IntegrityError as e:
            raise ValueError(f"evidence_cache: {e}") from e

    def n_lookup_budget_starved(self) -> int:
        """Count of ``evidence_cache`` rows still marked ``"lookup-budget:N"``
        (H3, see ``evidence_batch``) -- a URI whose tx-hash verification was
        cut short by a shared per-run lookup budget, and so may hold a false
        negative. Published in report provenance (``cli._provenance``'s
        ``evidence_lookup_starved_uris``) so a report is auditable against
        how many of its cached evidence levels are still, as of that run,
        possibly under-checked -- whether because a later ``fetch`` run
        hasn't retried them yet, or because they hit the retry cap
        (``evidence_batch.MAX_LOOKUP_RETRIES``) and are no longer retried at
        all.
        """
        r = self.conn.execute(
            "SELECT COUNT(*) FROM evidence_cache WHERE note LIKE 'lookup-budget:%'").fetchone()
        return int(r[0])

    def evidence_level(self, uri: str) -> Optional[int]:
        """Cached evidence level for a URI, or ``None`` if not yet enriched."""
        r = self.conn.execute("SELECT level FROM evidence_cache WHERE uri=?", (uri,)).fetchone()
        return None if r is None else int(r[0])

    def upsert_rater(self, address: str, first_seen_ts: Optional[int], funder: Optional[str]) -> None:
        """Cache rater profile metadata (first-seen timestamp, funder address)."""
        try:
            with self.conn:
                self.conn.execute("INSERT OR REPLACE INTO raters VALUES(?,?,?)", (address, first_seen_ts, funder))
        except sqlite3.IntegrityError as e:
            raise ValueError(f"raters: {e}") from e

    def clear_raters(self) -> int:
        """Drop every cached rater profile and forget ``rater_profile_mode``;
        returns the number of rows deleted.

        The escape hatch for a poisoned cache (M4). ``raters`` rows are written
        once and then never revisited -- ``distinct_clients`` only returns
        addresses with no row -- so a single bad ``first_seen_ts`` cached from a
        third-party API is permanent: ``robustrep.sybil._validate_meta`` raises
        on it, and ``score``/``report`` exit 1 on every later run with no way to
        recover short of deleting the whole store. Clearing the table puts every
        address back in ``distinct_clients``, so the next ``fetch`` re-profiles
        them from scratch. ``rater_profile_mode`` goes with it (the rows it
        described are gone); no other sync state is touched, so the block
        checkpoint and failed ranges survive.
        """
        with self.conn:
            deleted = self.conn.execute("DELETE FROM raters").rowcount
            self.conn.execute("DELETE FROM sync_state WHERE key='rater_profile_mode'")
        return max(int(deleted), 0)

    def upsert_agent_owner(self, agent_id: str, owner: str) -> None:
        """Cache the owner address for an agent id."""
        try:
            with self.conn:
                self.conn.execute("INSERT OR REPLACE INTO agents VALUES(?,?)", (agent_id, owner))
        except sqlite3.IntegrityError as e:
            raise ValueError(f"agents: {e}") from e

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
        """Atomically return and clear all recorded failed block ranges, ordered by
        start block. SELECT and DELETE run inside a single transaction so a range
        added concurrently between the two statements is never silently dropped."""
        with self.conn:
            rows = self.conn.execute(
                "SELECT from_block,to_block FROM failed_ranges ORDER BY from_block").fetchall()
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
        ``revoked`` is computed by joining against the standalone ``revocations``
        table (see ``mark_revoked``), so it reflects revocations recorded either
        before or after the matching feedback row was inserted.

        Materializes the whole feedback table as a DataFrame in memory — at
        roughly 500k rows this is on the order of ~200 MB; callers with much
        larger histories should filter/page at the SQL level instead.
        """
        q = """SELECT f.client AS rater, f.agent_id AS ratee, f.value AS value,
                      ('d' || f.value_decimals) AS scale, f.tag1 AS tag, b.ts AS ts,
                      f.feedback_uri AS evidence_uri, ? AS source,
                      COALESCE(e.level, 0) AS evidence_level,
                      CASE WHEN r.agent_id IS NULL THEN 0 ELSE 1 END AS revoked
               FROM feedback f LEFT JOIN blocks b ON b.number=f.block
               LEFT JOIN evidence_cache e ON e.uri=f.feedback_uri
               LEFT JOIN revocations r ON r.agent_id=f.agent_id AND r.client=f.client
                                       AND r.feedback_index=f.feedback_index"""
        df = pd.read_sql_query(q, self.conn, params=(SOURCE,))
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0).astype("int64")
        df["evidence_level"] = pd.to_numeric(df["evidence_level"], errors="coerce").fillna(0).astype("int64")
        df["revoked"] = pd.to_numeric(df["revoked"], errors="coerce").fillna(0).astype("int64")
        return df

    def load_rater_meta(self) -> pd.DataFrame:
        """Load cached rater metadata (rater address, first-seen ts, funder)."""
        return pd.read_sql_query("SELECT address AS rater, first_seen_ts, funder FROM raters", self.conn)

    def pending_uris(self, include_notes: tuple[str, ...] = ()
                      ) -> list[tuple[str, Optional[str], Optional[str], Optional[str]]]:
        """Feedback URIs (empty ones excluded) that still need evidence
        classification, each with its rater/owner parties GROUP_CONCAT-joined
        exactly as ``evidence_batch.classify_all`` consumes them, plus its
        existing cache note (if any): ``(uri, clients_csv, owners_csv,
        prior_note)``. ``clients_csv``/``owners_csv`` are ``None`` when there
        is nothing to join (e.g. no agent-owner row yet); ``prior_note`` is
        ``None`` when the URI has never been cached at all.

        A URI is pending when it has never been cached, OR when it *is*
        cached but with a note in ``include_notes`` -- e.g. a
        ``"lookup-budget:N"`` set (see ``evidence_batch.MAX_LOOKUP_RETRIES``,
        ``_RETRYABLE_LOOKUP_BUDGET_NOTES``) to retry URIs whose tx-hash
        verification was cut short by ``classify_all``'s shared per-run
        lookup budget (H3): such a cached level may be a false negative (a
        verifying hash sat past the cap), and a fresh run gets a fresh
        budget. A retry is not free, though -- it re-fetches the URI's whole
        text (the actually expensive part; the caller bounds how many times
        this happens per URI, since retrying forever would let a
        persistently-starved backlog inflate every future run's fetch
        volume). ``include_notes=()`` (the default) reproduces the original,
        simple "not yet cached at all" behavior -- an ``unfetchable`` or
        ``fetch-error`` cache entry is *not* retried by default; only notes
        explicitly listed are.

        The note comparison is NULL-safe (``COALESCE(e.note, '') NOT IN
        (...)``): ``evidence_cache.note`` can be SQL NULL for a row written
        before ``upsert_evidence`` started coercing ``None`` to ``""``, or by
        a caller bypassing ``upsert_evidence``. Bare ``e.note NOT IN (...)``
        would evaluate to SQL NULL (neither true nor false) for such a row,
        so ``NOT EXISTS`` would never see it as blocking -- the row would
        read as pending on *every* call, including ``include_notes=()``,
        regardless of what ``include_notes`` actually says.

        feedback.client and agents.owner are 0x-hex addresses, which never
        contain a comma, so GROUP_CONCAT's default "," separator can't
        collide with an address value and corrupt a caller's split-back-apart.

        Read up front (not streamed): the correlated NOT EXISTS sub-select
        re-reads evidence_cache on every row, which is only safe to interleave
        with writes when nothing is written back until the whole pending set
        has been captured -- a caller writing results back mid-scan (e.g.
        concurrent workers completing out of order) could otherwise race the
        cursor's own re-evaluation of NOT EXISTS.
        """
        retry_clause = (f"AND COALESCE(e2.note, '') NOT IN ({','.join('?' * len(include_notes))})"
                         if include_notes else "")
        q = f"""SELECT f.feedback_uri, GROUP_CONCAT(DISTINCT f.client), GROUP_CONCAT(DISTINCT a.owner),
                       MAX(e.note)
                FROM feedback f LEFT JOIN agents a ON a.agent_id=f.agent_id
                                LEFT JOIN evidence_cache e ON e.uri=f.feedback_uri
                WHERE f.feedback_uri<>'' AND NOT EXISTS (
                    SELECT 1 FROM evidence_cache e2 WHERE e2.uri=f.feedback_uri {retry_clause}
                )
                GROUP BY f.feedback_uri"""
        return self.conn.execute(q, include_notes).fetchall()

    def distinct_clients(self) -> list[str]:
        """Client (rater) addresses not yet profiled in the raters cache."""
        rows = self.conn.execute(
            "SELECT DISTINCT client FROM feedback f WHERE "
            "NOT EXISTS (SELECT 1 FROM raters ra WHERE ra.address=f.client)").fetchall()
        return [r[0] for r in rows]

    def distinct_agents(self) -> list[str]:
        """Agent ids not yet resolved in the agents (owner) cache."""
        rows = self.conn.execute(
            "SELECT DISTINCT agent_id FROM feedback f WHERE "
            "NOT EXISTS (SELECT 1 FROM agents a WHERE a.agent_id=f.agent_id)").fetchall()
        return [r[0] for r in rows]
