"""Live (network-touching) end-to-end checks. Marked ``live`` -- excluded from
the default test run (``pyproject.toml``'s ``addopts = "-m 'not live'"``);
run explicitly with ``pytest -m live tests/e2e``.

These run against free public RPC endpoints on a weekly cron. Every network
call is wrapped in ``rpc_available()``, which classifies "the endpoint was
down, slow, or throttling us" and reports it as a *labelled* failure (``RPC
unavailable: rate limited`` / ``transport failure: ...``) rather than letting
the raw exception through. It is still a failure, on purpose: a run that
cannot reach the chain has not checked anything, and the owner wants the
email (decided 2026-10-12, when mainnet.base.org began refusing every
``eth_getLogs`` and a skip would have hidden that for weeks). The label is
the whole value: the email says at a glance whether to look at the endpoint
list or at the data. A genuine data failure -- live data that does not match
what we recorded, a malformed response -- propagates unchanged. Assertions
are deliberately kept *outside* the guarded blocks so they can never be
reclassified.
"""
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest
import requests

from robustrep.config import Config
from robustrep.sources import base_erc8004 as base
from robustrep.sources.log_fetcher import LogFetcher
from robustrep.sources.rpc import RpcBatchStructureError, RpcClient, RpcError
from robustrep.store import Store

FIX = json.loads((Path(__file__).parent / "fixtures/base_logs_2000.json").read_text())

# Substrings (case-insensitive) of an ``RpcError`` message meaning the endpoint
# would not serve us -- not that the data it served is wrong. ``RpcClient``
# swallows the underlying ``requests`` exception and re-raises a *redacted*
# ``RpcError`` (see ``rpc._redact``: exception type, HTTP status, scheme+host
# only), so the transport names below are what a connection failure or timeout
# actually looks like by the time it reaches a test.
#
# Every marker is anchored enough not to fire on ordinary chain data: ``429``
# alone would match a block number, so the HTTP statuses are matched as
# ``"http <code>"``, which is exactly how ``_redact`` renders them.
RATE_LIMIT_MARKERS = (
    "http 429", "too many requests", "over rate limit", "rate limit",
    "rate-limited",
)
# JSON-RPC error codes for rate limiting, as our own "[code] " formatting
# renders them; matched anywhere in the message (see rpc_unavailable_reason).
RATE_LIMIT_CODES = ("[-32016]", "[-32005]")

# Transport-level failures and the server-side 5xx family: the endpoint is
# broken or overloaded, which says nothing about the chain.
TRANSPORT_MARKERS = (
    "connectionerror", "connecttimeout", "readtimeout", "timeout", "sslerror",
    "chunkedencodingerror", "toomanyredirects",
    "http 500", "http 502", "http 503", "http 504",
    # LogFetcher's own give-up error: every endpoint refused even the minimum window.
    "no usable endpoint",
)

# HTTP statuses that mean "try again later" rather than "the data is wrong".
SKIPPABLE_STATUSES = (429, 500, 502, 503, 504)

# Raised straight out of ``requests`` when a call bypasses ``RpcClient``'s
# wrapping. ``SSLError`` is a ``ConnectionError`` subclass, so it is covered.
SKIPPABLE_REQUESTS_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.TooManyRedirects,
    requests.exceptions.ChunkedEncodingError,
    TimeoutError,
)


def rpc_unavailable_reason(exc: BaseException) -> Optional[str]:
    """Return a short reason string when ``exc`` means *the RPC endpoint was
    unusable*, or ``None`` when it is a genuine failure that must not be
    papered over.

    Skippable: a transport-level ``requests`` failure (connection, timeout,
    TLS, chunked-encoding, redirect loop), an HTTP 429/5xx, and an ``RpcError``
    whose message names any of those. Everything else -- above all an
    ``AssertionError`` from live data disagreeing with the recorded fixture,
    but equally a malformed/structurally-wrong response, a JSON decode failure
    or a non-retryable protocol error -- returns ``None`` and is left to fail.

    The reason string is **only ever our own classification label**. No part of
    the exception message reaches it, because that message is in part
    server-supplied: a hostile or merely sloppy provider that echoes a keyed
    request URL back in an error body would otherwise land it in a public CI
    log. (``rpc._redact`` already scrubs what it can; this is the second
    layer, on the consuming side.)
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(exc, requests.HTTPError) and status in SKIPPABLE_STATUSES:
        return "rate limited" if status == 429 else f"transport failure: HTTP {status}"
    if isinstance(exc, SKIPPABLE_REQUESTS_ERRORS):
        return f"transport failure: {type(exc).__name__}"
    if isinstance(exc, RpcBatchStructureError):
        # A structurally wrong batch answer is a data failure whatever words
        # the server put in it.
        return None
    if isinstance(exc, RpcError):
        low = str(exc).lower()
        # Server-supplied text only ever appears AFTER a "[code] " prefix
        # (``rpc._error_message``); our own redacted transport rendering never
        # contains "[". Word markers are therefore matched only in the head,
        # so a provider cannot steer the classifier by echoing "timeout" or
        # "rate limit" in an error body. The numeric rate-limit codes are our
        # own bracketed formatting and may be matched anywhere.
        head = low.split("[", 1)[0]
        if any(code in low for code in RATE_LIMIT_CODES):
            return "rate limited"
        if any(marker in head for marker in RATE_LIMIT_MARKERS):
            return "rate limited"
        for marker in TRANSPORT_MARKERS:
            if marker in head:
                return f"transport failure: {marker}"
    return None


@contextmanager
def rpc_available():
    """Wrap live RPC calls: an unavailable/throttled endpoint becomes a
    *labelled* failure (``RPC unavailable: <our label>``, server text never
    included), anything else propagates unchanged.

    Keep assertions *outside* this block -- a mismatch between live data and
    the recorded fixture is precisely what these tests are for and must fail
    as itself, not as an availability problem.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 -- re-raised unless classified as availability
        reason = rpc_unavailable_reason(exc)
        if reason is None:
            raise
        pytest.fail(f"RPC unavailable: {reason}", pytrace=False)


def check_failed_ranges(store: Store) -> None:
    """Apply the availability guard to the sync path that never raises.

    ``base.sync_feedback`` records a chunk whose ``eth_getLogs`` failed in
    ``failed_ranges`` and carries on (so a real fetch can resume later) instead
    of raising -- which means ``rpc_available()`` around it sees nothing, and a
    throttled endpoint would surface as a bare checkpoint assertion failure
    (as it did on the 2026-10-12 weekly run: both ranges HTTP 429). Call this
    after ``sync_feedback``, before asserting on the checkpoint:

    * no recorded failures -> return, let the assertions run;
    * every recorded failure classifies as "endpoint unusable" (via
      ``rpc_unavailable_reason``, so the same markers and the same no-leak
      rule apply) -> fail with the ``RPC unavailable`` label, like
      ``rpc_available()``;
    * any failure that does *not* classify -> fail naming the range and its
      recorded error, exactly as the exception would have if it had
      propagated. A throttled sibling range never hides a real one.
    """
    rows = store.failed_ranges()
    if not rows:
        return
    reasons = [rpc_unavailable_reason(RpcError(err)) for _, _, err in rows]
    real = [(a, z, err) for (a, z, err), r in zip(rows, reasons) if r is None]
    if real:
        detail = "; ".join(f"{a}-{z}: {err}" for a, z, err in real)
        pytest.fail(f"sync_feedback recorded {len(real)} failed range(s) that are not an RPC-availability "
                    f"problem: {detail}")
    labels = ", ".join(sorted(set(reasons)))
    pytest.fail(f"RPC unavailable: {len(rows)} range(s) failed to fetch ({labels})", pytrace=False)


@pytest.mark.live
def test_live_recent_blocks(tmp_path):
    cfg = Config()
    rpc = RpcClient(cfg.rpc_urls, user_agent=cfg.user_agent)
    with Store(tmp_path / "live.db") as store:
        with rpc_available():
            head = int(rpc.call("eth_blockNumber", []), 16)
            fetcher = LogFetcher.from_config(rpc, cfg)
            base.sync_feedback(store, rpc, chunk=2000, start_block=head - 3999, end_block=head,
                               fetcher=fetcher)
        check_failed_ranges(store)
        assert store.get_sync("last_block") == str(head)
        assert store.sync_sources().sources
        with rpc_available():
            base.fill_block_timestamps(store, rpc)
        assert store.missing_block_ts() == []


@pytest.mark.live
def test_live_fixture_window_matches_recorded_fixture():
    """Drift detector: the fixture window (``tests/e2e/fixtures/base_logs_2000.json``)
    is deep in finalized history (well over ``confirmations`` blocks old), so
    replaying the exact same ``[from_block, to_block]`` against the live node
    should reproduce the identical set of ``NewFeedback`` events every time.

    If this ever fails, either the chain re-organized at a depth far beyond
    any realistic confirmation lag, or the RPC provider is serving bad data --
    either way, worth knowing about rather than silently trusting a stale
    fixture forever. The fixture is treated as final: no tolerance for
    newly-appeared ``ResponseAppended``/``FeedbackRevoked`` logs in the window,
    since ERC-8004 responses/revocations for already-final feedback would
    themselves be evidence of exactly the kind of drift this test is meant to
    catch.

    Only the ``LogFetcher.get_logs`` call (the same adaptive path the CLI
    uses) is guarded by ``rpc_available()``: an unreachable or throttling
    endpoint tells us nothing, but every comparison below runs unguarded so
    drift fails loudly.
    """
    cfg = Config()
    rpc = RpcClient(cfg.rpc_urls, user_agent=cfg.user_agent)
    with rpc_available():
        logs = LogFetcher.from_config(rpc, cfg).get_logs(
            FIX["from_block"], FIX["to_block"], base.LOG_FILTER).logs

    def feedback_keys(raw_logs):
        decoded = [base.decode_log(l) for l in raw_logs]
        return {(d["agent_id"], d["client"], d["feedback_index"])
                for d in decoded if d and d["kind"] == "feedback"}

    live_new_feedback = sum(1 for l in logs if l["topics"][0].lower() == base.TOPIC_NEW_FEEDBACK)
    fixture_new_feedback = sum(1 for l in FIX["logs"] if l["topics"][0].lower() == base.TOPIC_NEW_FEEDBACK)
    assert live_new_feedback == fixture_new_feedback

    assert feedback_keys(logs) == feedback_keys(FIX["logs"])
