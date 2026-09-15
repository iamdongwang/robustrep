"""Batch classification of evidence URIs (`robustrep.sources.evidence_batch`).

Covers `classify_all`'s contract, its per-URI error isolation, its worker-thread
/ session behaviour, and the H3 run-wide tx-hash lookup budget. The single-URI
SSRF guard and fetch tests live in `test_evidence_fetch.py`, next to the module
they exercise.
"""
import logging
import threading

import pytest

import robustrep.sources.evidence_batch as eb
from robustrep.sources.evidence_batch import classify_all
from robustrep.store import Store

TX = "0x" + "cd" * 32


# --- classify_all (task's own contract test) -------------------------------------

def _fb(uri, client="0xaa", agent="7"):
    return dict(chain="base", block=1, tx_hash="0x", log_index=0, agent_id=agent, client=client,
                feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
                feedback_uri=uri, feedback_hash="")


def test_classify_all_uses_cache_and_parties(tmp_path):
    s = Store(tmp_path / "t.db")
    # Note: the third row must use a feedback_index distinct from the first
    # (both default to agent="7", client="0xaa") -- otherwise it collides with
    # row 1 on feedback's (agent_id, client, feedback_index) primary key and is
    # silently dropped by INSERT OR IGNORE, which would starve "https://bad" of
    # any feedback row at all.
    s.upsert_feedback([_fb("https://good"), {**_fb("https://good"), "feedback_index": 1},
                        {**_fb("https://bad"), "feedback_index": 2}])
    s.upsert_agent_owner("7", "0xowner")
    fetched = []

    def fetch(uri, session=None):
        fetched.append(uri)
        return f"tx {TX}" if uri == "https://good" else None

    def parties(h):
        return {"0xowner", "0xz"}

    n = classify_all(s, fetch_text=fetch, tx_parties=parties)
    assert n == 2 and sorted(fetched) == ["https://bad", "https://good"]
    assert s.evidence_level("https://good") == 3 and s.evidence_level("https://bad") == 1
    # "good" was fetched successfully -> note ""; "bad"'s fetch returned None
    # (per `fetch` above) -> note "unfetchable" (see the dedicated test below
    # for the level-1-but-fetched-successfully case, which also gets note "").
    good_note = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://good'").fetchone()[0]
    bad_note = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://bad'").fetchone()[0]
    assert good_note == "" and bad_note == "unfetchable"
    assert classify_all(s, fetch_text=fetch, tx_parties=parties) == 0  # cached


def test_classify_all_notes_unfetchable_vs_empty_vs_fetch_error(tmp_path):
    s = Store(tmp_path / "tnotes.db")
    s.upsert_feedback([
        _fb("https://never-fetched"),
        {**_fb("https://boom"), "feedback_index": 1},
        {**_fb("https://no-evidence"), "feedback_index": 2},
    ])

    def fetch(uri, session=None):
        if uri == "https://never-fetched":
            return None  # genuinely unfetchable
        if uri == "https://boom":
            raise RuntimeError("boom")  # fetch_text violating its no-raise contract
        return "just some prose, no tx hash or taskId here"  # fetched fine, level 1

    n = classify_all(s, fetch_text=fetch)
    assert n == 3
    note = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://never-fetched'").fetchone()[0]
    assert note == "unfetchable"
    note2 = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://boom'").fetchone()[0]
    assert note2 == "fetch-error"
    note3 = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://no-evidence'").fetchone()[0]
    assert s.evidence_level("https://no-evidence") == 1 and note3 == ""


# --- classify_all: error isolation, progress logging, session reuse ------------------

def test_classify_all_isolates_per_uri_errors_and_continues(tmp_path, caplog):
    s = Store(tmp_path / "t2.db")
    s.upsert_feedback([_fb("https://ok"), {**_fb("https://boom"), "feedback_index": 1}])

    def fetch(uri, session=None):
        if uri == "https://boom":
            raise RuntimeError("boom")
        return f"tx {TX}"

    def parties(h):
        return {"0xaa"}

    with caplog.at_level(logging.ERROR):
        n = classify_all(s, fetch_text=fetch, tx_parties=parties)
    assert n == 2
    assert s.evidence_level("https://boom") == 1
    r = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://boom'").fetchone()
    assert r[0] == "fetch-error"
    # The URI is attacker-controlled, so it is logged with %r (quoted/escaped),
    # never bare %s -- a URI carrying newlines must not forge log lines.
    assert any("'https://boom'" in rec.message for rec in caplog.records)


def test_classify_all_logs_progress(tmp_path, caplog):
    s = Store(tmp_path / "t3.db")
    rows = [{**_fb(f"https://u{i}"), "feedback_index": i} for i in range(5)]
    s.upsert_feedback(rows)

    def fetch(uri, session=None):
        return None

    with caplog.at_level(logging.INFO):
        n = classify_all(s, fetch_text=fetch, log_every=2)
    assert n == 5
    progress_msgs = [rec.message for rec in caplog.records if "processed" in rec.message]
    assert len(progress_msgs) == 2  # at i=2 and i=4


def test_classify_all_log_every_zero_disables_progress_logging(tmp_path, caplog):
    s = Store(tmp_path / "t3b.db")
    rows = [{**_fb(f"https://v{i}"), "feedback_index": i} for i in range(3)]
    s.upsert_feedback(rows)

    def fetch(uri, session=None):
        return None

    with caplog.at_level(logging.INFO):
        n = classify_all(s, fetch_text=fetch, log_every=0)  # must not raise ZeroDivisionError
    assert n == 3
    assert not [rec for rec in caplog.records if "processed" in rec.message]


def test_classify_all_reuses_one_session_across_uris(tmp_path):
    # With a single worker thread (workers=1), classify_all behaves like the
    # old sequential implementation: one thread -> one lazily-created,
    # thread-local requests.Session, reused for every URI.
    s = Store(tmp_path / "t4.db")
    rows = [{**_fb(f"https://s{i}"), "feedback_index": i} for i in range(3)]
    s.upsert_feedback(rows)
    seen_sessions = []

    def fetch(uri, session=None):
        seen_sessions.append(session)
        return None

    classify_all(s, fetch_text=fetch, workers=1)
    assert len(seen_sessions) == 3
    assert len({id(x) for x in seen_sessions}) == 1
    assert all(x is not None for x in seen_sessions)


# --- classify_all: concurrent workers -------------------------------------------------

def _seed_many(store: Store, n: int, prefix: str = "u") -> None:
    rows = [{**_fb(f"https://{prefix}{i}"), "feedback_index": i} for i in range(n)]
    store.upsert_feedback(rows)


def _mixed_fetch(uri, session=None):
    """Deterministic per-URI outcome (by trailing int in the uri) covering all
    three ``classify_all`` note branches: unfetchable, fetched-with-evidence,
    fetched-with-no-evidence."""
    i = int("".join(ch for ch in uri.rsplit("/", 1)[-1] if ch.isdigit()))
    if i % 7 == 0:
        return None  # unfetchable
    if i % 5 == 0:
        return f"tx {TX}"  # fetched, evidence found -> level 3
    return "just some prose, no evidence markers"  # fetched, no evidence -> level 1


def test_classify_all_workers_4_matches_workers_1(tmp_path):
    s1 = Store(tmp_path / "seq.db")
    _seed_many(s1, 50, prefix="p")
    n1 = classify_all(s1, fetch_text=_mixed_fetch, workers=1)

    s2 = Store(tmp_path / "par.db")
    _seed_many(s2, 50, prefix="p")
    n2 = classify_all(s2, fetch_text=_mixed_fetch, workers=4)

    assert n1 == n2 == 50
    rows1 = s1.conn.execute("SELECT uri, level, note FROM evidence_cache ORDER BY uri").fetchall()
    rows2 = s2.conn.execute("SELECT uri, level, note FROM evidence_cache ORDER BY uri").fetchall()
    assert rows1 == rows2
    assert len(rows1) == 50


def test_classify_all_worker_exception_still_yields_fetch_error(tmp_path):
    s = Store(tmp_path / "werr.db")
    _seed_many(s, 10, prefix="e")

    def fetch(uri, session=None):
        if uri == "https://e3":
            raise RuntimeError("boom")
        return None

    n = classify_all(s, fetch_text=fetch, workers=4)
    assert n == 10
    level, note = s.conn.execute(
        "SELECT level, note FROM evidence_cache WHERE uri='https://e3'").fetchone()
    assert level == 1 and note == "fetch-error"


def test_classify_all_writes_store_only_from_main_thread(tmp_path):
    s = Store(tmp_path / "tthread.db")
    _seed_many(s, 50, prefix="w")
    main_thread_id = threading.get_ident()
    write_thread_ids = []
    orig_upsert = s.upsert_evidence

    def recording_upsert(uri, level, note=""):
        write_thread_ids.append(threading.get_ident())
        return orig_upsert(uri, level, note)

    s.upsert_evidence = recording_upsert

    def fetch(uri, session=None):
        return None

    n = classify_all(s, fetch_text=fetch, workers=4)
    assert n == 50
    assert len(write_thread_ids) == 50
    assert all(tid == main_thread_id for tid in write_thread_ids)


# --- H3: total tx-hash lookup budget shared across a whole classify_all run --------

def _many_hashes_text(n: int, seed: int) -> str:
    """``n`` distinct 64-hex tx hashes, offset by ``seed`` so different URIs
    contribute disjoint hash sets (nothing to verify against)."""
    return " ".join("0x" + f"{seed * n + i:064x}" for i in range(n))


def test_classify_all_total_lookup_budget_is_shared_across_uris(tmp_path, caplog):
    s = Store(tmp_path / "tbudget.db")
    rows = [{**_fb(f"https://b{i}"), "feedback_index": i} for i in range(5)]
    s.upsert_feedback(rows)

    def fetch(uri, session=None):
        i = int(uri.rsplit("b", 1)[-1])
        return _many_hashes_text(8, i)

    calls = []

    def tx_parties(h):
        calls.append(h)
        return None  # nothing ever verifies

    with caplog.at_level(logging.WARNING):
        n = classify_all(s, fetch_text=fetch, tx_parties=tx_parties,
                          max_total_lookups=20, workers=2)
    assert n == 5
    assert len(calls) == 20
    levels = s.conn.execute("SELECT level FROM evidence_cache").fetchall()
    assert all(level == 2 for (level,) in levels)
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "budget" in warnings[0].message


def test_classify_all_budget_not_exhausted_logs_no_warning(tmp_path, caplog):
    s = Store(tmp_path / "tbudget2.db")
    rows = [{**_fb(f"https://c{i}"), "feedback_index": i} for i in range(5)]
    s.upsert_feedback(rows)

    def fetch(uri, session=None):
        i = int(uri.rsplit("c", 1)[-1])
        return _many_hashes_text(8, i)

    def tx_parties(h):
        return None

    with caplog.at_level(logging.WARNING):
        n = classify_all(s, fetch_text=fetch, tx_parties=tx_parties,
                          max_total_lookups=20_000, workers=2)
    assert n == 5
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warnings == []


@pytest.mark.parametrize("max_total_lookups", [0, -5])
def test_classify_all_zero_or_negative_budget_logs_exactly_one_warning(tmp_path, caplog, max_total_lookups):
    # Regression: the first cut only warned from the branch that *decrements*
    # `_remaining` across zero, so a budget that starts at (or below) zero --
    # nothing to spend from the very first lookup -- never took that branch
    # and silently warned never. It must still warn exactly once, from
    # whichever thread's first refused lookup discovers the budget is empty.
    s = Store(tmp_path / f"tzero{max_total_lookups}.db")
    rows = [{**_fb(f"https://z{i}"), "feedback_index": i} for i in range(3)]
    s.upsert_feedback(rows)

    def fetch(uri, session=None):
        return f"tx {TX}"

    def tx_parties(h):
        return None

    with caplog.at_level(logging.WARNING):
        n = classify_all(s, fetch_text=fetch, tx_parties=tx_parties,
                          max_total_lookups=max_total_lookups, workers=3)
    assert n == 3
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_classify_all_per_uri_cap_still_applies_within_ample_total_budget(tmp_path):
    # 2 URIs x 50 distinct hashes each, but MAX_TX_LOOKUPS_PER_URI (8) bounds
    # each URI's own lookups regardless of how large max_total_lookups is --
    # the per-URI cap and the shared run budget are independent controls.
    s = Store(tmp_path / "tpassthrough.db")
    rows = [{**_fb(f"https://pu{i}"), "feedback_index": i} for i in range(2)]
    s.upsert_feedback(rows)

    def fetch(uri, session=None):
        i = int(uri.rsplit("pu", 1)[-1])
        return _many_hashes_text(50, i)

    calls = []

    def tx_parties(h):
        calls.append(h)
        return None

    n = classify_all(s, fetch_text=fetch, tx_parties=tx_parties,
                      max_total_lookups=10_000, workers=2)
    assert n == 2
    assert len(calls) == 16  # 2 URIs * MAX_TX_LOOKUPS_PER_URI (8)


def test_classify_all_starved_uri_is_retryable_and_upgrades_on_retry(tmp_path):
    # A URI whose verifying hash sits past the point the run's shared budget
    # cut off at (H3) must be persisted as retryable, not as an ordinary
    # (possibly wrong) level 2 -- and a later run, with its own fresh budget,
    # must pick it back up and upgrade it once it can actually reach that hash.
    s = Store(tmp_path / "tretry.db")
    hashes = ["0x" + f"{i:064x}" for i in range(5)]
    text = " ".join(hashes)
    verifying_hash = hashes[4]  # the 5th (last) hash -- past a budget of 3

    def fetch(uri, session=None):
        return text

    def tx_parties(h):
        return {"0xowner"} if h == verifying_hash else None

    s.upsert_feedback([_fb("https://retry1")])
    s.upsert_agent_owner("7", "0xowner")

    # First run: the shared budget (3) is spent on hashes 1-3 before the
    # verifying 5th hash is ever reached.
    n1 = classify_all(s, fetch_text=fetch, tx_parties=tx_parties,
                       max_total_lookups=3, workers=1)
    assert n1 == 1
    level1, note1 = s.conn.execute(
        "SELECT level, note FROM evidence_cache WHERE uri='https://retry1'").fetchone()
    assert (level1, note1) == (2, "lookup-budget:1")

    # Second run: the "lookup-budget:1" note makes it pending again, and this
    # run's fresh (ample, default) budget reaches the verifying hash.
    n2 = classify_all(s, fetch_text=fetch, tx_parties=tx_parties, workers=1)
    assert n2 == 1
    level2, note2 = s.conn.execute(
        "SELECT level, note FROM evidence_cache WHERE uri='https://retry1'").fetchone()
    assert (level2, note2) == (3, "")

    # Third run: the upgraded, non-"lookup-budget:N" result is final --
    # nothing left pending.
    n3 = classify_all(s, fetch_text=fetch, tx_parties=tx_parties, workers=1)
    assert n3 == 0


def test_classify_all_lookup_budget_retries_are_bounded(tmp_path):
    # A URI that is starved on every attempt must stop being retried once it
    # has been starved MAX_LOOKUP_ATTEMPTS times -- otherwise a persistently
    # starved backlog would re-fetch its (expensive) text forever for no
    # further progress.
    s = Store(tmp_path / "tretrycap.db")
    text = " ".join("0x" + f"{i:064x}" for i in range(20))  # always starved: budget=1 < 20 hashes

    def fetch(uri, session=None):
        return text

    def tx_parties(h):
        return None  # never verifies, regardless of budget

    s.upsert_feedback([_fb("https://always-starved")])

    notes = []
    for _ in range(eb.MAX_LOOKUP_ATTEMPTS + 2):
        n = classify_all(s, fetch_text=fetch, tx_parties=tx_parties,
                          max_total_lookups=1, workers=1)
        if n == 0:
            break
        note = s.conn.execute(
            "SELECT note FROM evidence_cache WHERE uri='https://always-starved'").fetchone()[0]
        notes.append(note)

    # It was classified under a starved budget MAX_LOOKUP_ATTEMPTS times
    # (attempt counts 1 through MAX_LOOKUP_ATTEMPTS), then stopped being
    # pending -- classify_all found nothing left to do on the run right after
    # the last attempt.
    assert notes == [f"lookup-budget:{n}" for n in range(1, eb.MAX_LOOKUP_ATTEMPTS + 1)]
    assert classify_all(s, fetch_text=fetch, tx_parties=tx_parties, max_total_lookups=1, workers=1) == 0


def test_classify_all_each_worker_thread_gets_its_own_session(tmp_path):
    s = Store(tmp_path / "tsess.db")
    _seed_many(s, 40, prefix="s")
    seen = []  # (thread_ident, session_id)
    lock = threading.Lock()

    def fetch(uri, session=None):
        with lock:
            seen.append((threading.get_ident(), id(session)))
        return None

    classify_all(s, fetch_text=fetch, workers=4)

    by_thread: dict = {}
    for tid, sid in seen:
        by_thread.setdefault(tid, set()).add(sid)
    # Each worker thread reused exactly one session for every URI it handled
    # (thread-local caching), never a fresh one per call.
    assert all(len(sids) == 1 for sids in by_thread.values())
    # Distinct worker threads (if more than one actually ran) never share a
    # session object.
    if len(by_thread) > 1:
        session_ids_by_thread = [next(iter(sids)) for sids in by_thread.values()]
        assert len(set(session_ids_by_thread)) == len(by_thread)


# --- legacy bare "lookup-budget" note, per-URI cap threading ------------------


def test_lookup_budget_attempts_reads_the_note_or_falls_back_to_zero():
    assert eb._lookup_budget_attempts("lookup-budget:2") == 2
    # 7f391ef's bare note counts as the first starved attempt.
    assert eb._lookup_budget_attempts("lookup-budget") == 1
    # Anything else -- never cached, an ordinary result, another note, or a
    # malformed count -- starts the next attempt at 1.
    assert eb._lookup_budget_attempts(None) == 0
    assert eb._lookup_budget_attempts("") == 0
    assert eb._lookup_budget_attempts("unfetchable") == 0
    assert eb._lookup_budget_attempts("lookup-budget:not-a-number") == 0


def test_classify_all_retries_a_legacy_bare_lookup_budget_note(tmp_path):
    s = Store(tmp_path / "tlegacy.db")
    s.upsert_feedback([_fb("https://legacy")])
    s.upsert_evidence("https://legacy", 2, "lookup-budget")

    def fetch(uri, session=None):
        return f"see {TX}"

    # Starved again (budget 0): the bare note counted as attempt 1, so this
    # attempt is recorded as 2 rather than restarting the count.
    n = classify_all(s, fetch_text=fetch, tx_parties=lambda h: None,
                     max_total_lookups=0, workers=1)
    assert n == 1
    note = s.conn.execute(
        "SELECT note FROM evidence_cache WHERE uri='https://legacy'").fetchone()[0]
    assert note == "lookup-budget:2"


def test_classify_all_max_lookups_per_uri_reaches_classify(tmp_path):
    # The per-URI cap is a classify_all parameter, not just classify's own
    # default, so the CLI can pass (and then publish) the value it used.
    s = Store(tmp_path / "tperuri.db")
    s.upsert_feedback([_fb("https://many")])
    hashes = ["0x" + f"{i:064x}" for i in range(5)]
    looked_up = []

    def fetch(uri, session=None):
        return " ".join(hashes)

    def tx_parties(h):
        looked_up.append(h)
        return None

    classify_all(s, fetch_text=fetch, tx_parties=tx_parties, workers=1,
                 max_lookups_per_uri=2)
    assert len(looked_up) == 2


def test_classify_all_max_lookups_per_uri_defaults_to_the_module_constant(tmp_path):
    from robustrep.evidence import MAX_TX_LOOKUPS_PER_URI

    s = Store(tmp_path / "tperuridef.db")
    s.upsert_feedback([_fb("https://many")])
    hashes = ["0x" + f"{i:064x}" for i in range(MAX_TX_LOOKUPS_PER_URI + 3)]
    looked_up = []

    def fetch(uri, session=None):
        return " ".join(hashes)

    def tx_parties(h):
        looked_up.append(h)
        return None

    classify_all(s, fetch_text=fetch, tx_parties=tx_parties, workers=1)
    assert len(looked_up) == MAX_TX_LOOKUPS_PER_URI
