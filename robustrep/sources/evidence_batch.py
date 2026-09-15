"""Classify every distinct evidence URI into the cache, with a bounded lookup budget.

This is the batch layer above `evidence_fetch`, which owns the single-URI
security boundary (SSRF guard, bounded fetch). Split out from it so the URL
vetting story and the run-level budgeting story can each be read on their own;
the dependency runs one way only (`evidence_batch` imports `http_fetch_text`,
never the reverse).

``classify_all`` walks every URI referenced by ``feedback`` rows that is not yet
in the evidence cache, classifies it, and persists the level -- one bad URI (an
unexpected exception out of ``classify``) is caught, logged, and recorded as
level 1 rather than aborting the whole batch.

**H3 (security review): tx-hash lookup budget.** ``robustrep.evidence.classify``
already caps lookups *per URI* at ``MAX_TX_LOOKUPS_PER_URI``, but that alone
does not bound a whole ``fetch`` run: an attacker can post many distinct
evidence URIs, each packed with hashes up to that per-URI cap, and still
multiply out to a large number of RPC calls across the batch. ``classify_all``
additionally shares one ``_LookupBudget`` across every URI and every worker
thread for the run (``max_total_lookups``, default 20,000): once it is spent,
the wrapped ``tx_parties`` stops calling the underlying function at all and
returns ``None`` for every further hash, which ``classify`` treats as
"unverified" (its documented, conservative degrade path) rather than raising.
Exactly one WARNING is logged for the whole run, the moment the budget is
exhausted -- including a ``max_total_lookups <= 0`` run, which has nothing to
spend from its very first lookup. A URI classified while the budget was
exhausted may therefore hold a false negative (level 2 instead of the 3 a
later, unchecked hash would have verified) -- such a URI is persisted with
note ``"lookup-budget:N"`` (``N`` = how many times, including this one, it
has been starved) instead of an ordinary result, and
``Store.pending_uris(include_notes=_RETRYABLE_LOOKUP_BUDGET_NOTES)`` (used by
``classify_all`` itself) treats it as still pending -- but only while
``N < MAX_LOOKUP_RETRIES`` (default 3): a retry re-fetches the URI's whole
text, so retrying forever would let a persistently-starved backlog amplify
every future run's fetch volume without bound. After that many starved
attempts, the last cached (possibly false-negative) level is accepted as
final.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import requests

from ..evidence import classify
from ..store import Store
from .evidence_fetch import http_fetch_text

# H3: total eth_getTransactionByHash calls allowed across one classify_all
# run, shared by every URI and every worker thread -- see the module
# docstring and `_LookupBudget`.
DEFAULT_MAX_TOTAL_LOOKUPS = 20_000
# H3: a budget-starved URI's classification may be a false negative (see
# `_classify_uri`'s "lookup-budget:N" note), so it is retried -- but a retry
# re-fetches the URI's whole text (the expensive part) to resolve at most a
# few more lookups, so retrying forever would let one persistently-starved
# backlog amplify every future run's fetch volume without bound. A URI is
# retried at most `MAX_LOOKUP_RETRIES - 1` times (i.e. while its starved
# attempt count is < this) before its last cached (possibly false-negative)
# result is accepted as final.
MAX_LOOKUP_RETRIES = 3
# The exact "lookup-budget:N" note values `classify_all` treats as pending --
# passed to `Store.pending_uris(include_notes=...)` as literal, parameterized
# values (no SQL LIKE/prefix matching needed).
_RETRYABLE_LOOKUP_BUDGET_NOTES = tuple(f"lookup-budget:{n}" for n in range(1, MAX_LOOKUP_RETRIES))

_log = logging.getLogger(__name__)


def _lookup_budget_attempts(prior_note: Optional[str]) -> int:
    """The starved-attempt count ``N`` encoded in a ``"lookup-budget:N"``
    note, or 0 for anything else (never cached, a normal result, or another
    note entirely) -- the starting point for the next attempt's count."""
    if prior_note and prior_note.startswith("lookup-budget:"):
        try:
            return int(prior_note.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def _classify_uri(uri: str, parties: set, fetch_text: Callable, tx_parties: Callable, session,
                   was_starved: Optional[Callable[[], bool]] = None,
                   prior_note: Optional[str] = None) -> tuple[int, str]:
    """Classify one URI, returning ``(level, note)``.

    ``note`` is ``"unfetchable"`` when ``fetch_text`` returned ``None`` (the
    URI could not be retrieved at all -- a later re-run can target these
    specifically); ``"fetch-error"`` if ``classify`` itself raised (defensive:
    ``fetch_text``'s contract is to never raise, but a bug elsewhere in
    ``classify``, e.g. in ``tx_parties`` handling, should still degrade
    gracefully rather than abort the batch); ``"lookup-budget:N"`` (H3) when
    the URI *was* fetched and classified but ``was_starved()`` reports that
    the shared per-run tx lookup budget ran out partway through its hash
    list -- the returned level may be a false negative (a verifying hash
    could have sat past the point the budget cut off at), so it is persisted
    as retryable rather than as an ordinary final result (see
    ``Store.pending_uris``, ``MAX_LOOKUP_RETRIES``). ``N`` is
    ``_lookup_budget_attempts(prior_note) + 1`` -- how many times, including
    this one, the URI has now been starved; ``""`` for a normal, fully-
    checked result.

    ``was_starved``, when given, must be a callable that reports (and resets)
    whether the *this-thread* budget-wrapped ``tx_parties`` was refused for
    lack of budget since it was last checked -- see ``_LookupBudget.wrap``.
    It is called once before ``classify`` (discarding any stale flag left by
    a previous URI this same worker thread handled) and once after.

    ``prior_note`` is this URI's note from its last cache entry, if any (as
    returned by ``Store.pending_uris``), used only to compute ``N`` above.
    """
    fetched_none = False

    def _fetch(u):
        nonlocal fetched_none
        text = fetch_text(u, session=session)
        if text is None:
            fetched_none = True
        return text

    if was_starved is not None:
        was_starved()

    try:
        level = classify(uri, _fetch, tx_parties, parties)
    except Exception:
        _log.error("classify_all: classify failed for %r", uri, exc_info=True)
        if was_starved is not None:
            was_starved()
        return 1, "fetch-error"
    if fetched_none:
        return level, "unfetchable"
    if was_starved is not None and was_starved():
        return level, f"lookup-budget:{_lookup_budget_attempts(prior_note) + 1}"
    return level, ""


class _LookupBudget:
    """Thread-safe counter that wraps ``tx_parties`` to cap the total number
    of RPC lookups spent across one ``classify_all`` run (H3, see module
    docstring).

    ``spend()`` is called once per candidate hash, from whichever worker
    thread ``robustrep.evidence.classify`` happens to be running in, so the
    decrement-and-check has to be atomic -- a plain ``if self._remaining > 0``
    followed by a decrement would let two threads both pass the check for the
    last unit of budget. ``_exhausted`` is a lock-free fast path checked
    before taking the lock at all: it only ever flips ``False`` -> ``True``
    (never back), so every call after the one that actually exhausts the
    budget can skip the lock entirely, at the cost of nothing worse than a
    handful of calls racing the exhausting one itself still taking the slow,
    always-correct locked path. Once exhausted, ``wrap``'s callable stops
    calling the underlying ``tx_parties`` entirely and returns ``None`` (the
    same "unverified" signal a real RPC miss would give), and exactly one
    WARNING is logged for the whole run -- logged *outside* the lock (holding
    a lock across a logging call would serialize every other thread's
    lookups behind a slow handler for no benefit). This includes a
    ``total <= 0`` budget (nothing to spend at all): the very first call to
    ``spend()`` finds ``_remaining <= 0`` immediately and must still be the
    one that flips ``_exhausted`` and logs -- a bug in an earlier version
    only ever warned from the decrement branch, so a zero/negative budget
    silently warned never. Both branches therefore guard the flip with the
    same ``if not self._exhausted`` check under the lock, so exactly one
    caller -- whichever branch and whichever thread first observes the
    budget as spent -- logs, no matter which path exhausted it.
    """

    def __init__(self, total: int):
        self._total = max(total, 0)
        self._remaining = self._total
        self._lock = threading.Lock()
        self._exhausted = False
        self.spent = 0

    @property
    def total(self) -> int:
        """The clamped (never-negative) budget this run started with."""
        return self._total

    def spend(self) -> bool:
        """Atomically consume one unit of budget; True if one was available
        (and was just consumed), False if the budget is (now, or already)
        exhausted -- including a ``total <= 0`` budget, which has nothing to
        spend from the very first call."""
        if self._exhausted:
            return False
        success = False
        just_exhausted = False
        with self._lock:
            if self._remaining <= 0:
                if not self._exhausted:
                    self._exhausted = True
                    just_exhausted = True
            else:
                self._remaining -= 1
                self.spent += 1
                success = True
                if self._remaining <= 0:
                    self._exhausted = True
                    just_exhausted = True
        if just_exhausted:
            _log.warning(
                "evidence: tx lookup budget of %d exhausted; remaining hashes treated as unverified",
                self._total)
        return success

    def wrap(self, tx_parties: Callable[[str], Optional[set]]
             ) -> tuple[Callable[[str], Optional[set]], Callable[[], bool]]:
        """Return ``(budgeted_tx_parties, was_starved)``.

        ``budgeted_tx_parties`` is a ``tx_parties``-shaped callable that
        spends one unit of this budget per call and, once exhausted, calls
        ``tx_parties`` no further (returning ``None`` instead).

        ``was_starved()`` reports, and resets, whether *this calling thread's*
        most recent run of ``budgeted_tx_parties`` calls included one refused
        for lack of budget. It is thread-local rather than a single shared
        flag because ``classify_all`` gives each worker thread one URI to
        classify at a time (never two interleaved ``classify()`` calls on the
        same thread), so "since this thread last checked" is exactly "during
        the URI this thread is classifying right now" -- which is precisely
        what ``_classify_uri`` needs to decide whether *that* URI's result is
        retryable.
        """
        local = threading.local()

        def _budgeted(h: str) -> Optional[set]:
            if not self.spend():
                local.starved = True
                return None
            return tx_parties(h)

        def _was_starved() -> bool:
            starved = getattr(local, "starved", False)
            local.starved = False
            return starved

        return _budgeted, _was_starved


def classify_all(store: Store, fetch_text: Callable = http_fetch_text,
                  tx_parties: Callable[[str], Optional[set]] = lambda h: None,
                  log_every: int = 500, workers: int = 8,
                  max_total_lookups: int = DEFAULT_MAX_TOTAL_LOOKUPS) -> int:
    """Classify every distinct URI referenced by ``feedback`` that is not yet in
    the evidence cache, and persist each result via ``store.upsert_evidence``.

    ``parties`` passed to ``classify`` for each URI is the union of every
    rater address (``feedback.client``) and agent-owner address that used it.

    The pending ``(uri, parties)`` rows are read into a list up front (a
    ~34k-row batch is fine in memory), then classified concurrently across
    ``workers`` threads (``concurrent.futures.ThreadPoolExecutor``) -- each
    URI's fetch+classify (``_classify_uri``, including its own per-URI
    exception guard) runs in a worker thread. ``store.upsert_evidence`` is
    called ONLY from this (the main) thread, as each future completes via
    ``as_completed``: ``Store`` wraps a single ``sqlite3`` connection opened
    with the default ``check_same_thread=True`` and is not safe to write from
    multiple threads. ``workers=1`` behaves like the previous sequential
    implementation (results identical; the order of store writes is not
    guaranteed either way).

    Each worker thread lazily creates and reuses its own ``requests.Session``
    (via thread-local storage) for every ``fetch_text`` call it makes --
    sessions are never shared across threads, since a single
    ``requests.Session`` is not guaranteed safe for concurrent use. Every
    created session is closed once all URIs have been processed.

    Logs progress at INFO every ``log_every`` URIs *completed* (regardless of
    completion order); ``log_every <= 0`` disables progress logging entirely
    (also guards against a ``ZeroDivisionError`` from ``% log_every``).

    ``max_total_lookups`` (H3, see module docstring) bounds the total number
    of ``tx_parties`` calls across the *whole run*, shared by every URI and
    every worker thread via a single ``_LookupBudget``; the number actually
    spent (out of the clamped total, never negative even if a caller passes
    one) is logged at INFO once the run completes, in a ``finally`` so it is
    logged even if the run itself raised. A URI whose classification was cut
    short by the budget is persisted with note ``"lookup-budget:N"`` rather
    than ``""``/an ordinary level (see ``_classify_uri``), and is picked back
    up as pending on the *next* call via ``store.pending_uris(include_notes=
    _RETRYABLE_LOOKUP_BUDGET_NOTES)`` below -- a fresh run gets a fresh
    budget, so a URI that was starved is not starved forever. A retry is not
    free, though: it re-fetches the URI's whole text (the actually expensive
    part; the RPC lookups it unlocks are comparatively cheap), so a URI is
    retried at most ``MAX_LOOKUP_RETRIES - 1`` times before its last cached
    result -- a possible false negative -- is accepted as final, bounding how
    much a persistently-starved backlog can inflate a run's fetch volume.

    Returns the number of URIs processed.
    """
    # A "lookup-budget:N" note with N < MAX_LOOKUP_RETRIES is retried (see
    # Store.pending_uris, MAX_LOOKUP_RETRIES); a URI cached with any other
    # note (a normal result, "unfetchable", "fetch-error", or a
    # "lookup-budget:N" that has hit the retry cap) is not retried here.
    rows = store.pending_uris(include_notes=_RETRYABLE_LOOKUP_BUDGET_NOTES)

    budget = _LookupBudget(max_total_lookups)
    budgeted_tx_parties, was_starved = budget.wrap(tx_parties)

    thread_local = threading.local()
    sessions: list = []
    sessions_lock = threading.Lock()

    def _thread_session():
        sess = getattr(thread_local, "session", None)
        if sess is None:
            sess = requests.Session()
            thread_local.session = sess
            with sessions_lock:
                sessions.append(sess)
        return sess

    def _classify_one(uri: str, parties: set, prior_note: Optional[str]):
        session = _thread_session()
        return _classify_uri(uri, parties, fetch_text, budgeted_tx_parties, session,
                              was_starved, prior_note)

    processed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_uri = {}
            for uri, clients, owners, prior_note in rows:
                parties = {p for p in (clients or "").split(",") + (owners or "").split(",") if p}
                future = executor.submit(_classify_one, uri, parties, prior_note)
                future_to_uri[future] = uri
            for future in as_completed(future_to_uri):
                uri = future_to_uri[future]
                level, note = future.result()
                store.upsert_evidence(uri, level, note)
                processed += 1
                if log_every > 0 and processed % log_every == 0:
                    _log.info("classify_all: processed %d URIs", processed)
    finally:
        for sess in sessions:
            sess.close()
        _log.info("classify_all: tx lookup budget spent %d/%d", budget.spent, budget.total)
    return processed
