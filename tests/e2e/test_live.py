"""Live (network-touching) end-to-end checks. Marked ``live`` -- excluded from
the default test run (``pyproject.toml``'s ``addopts = "-m 'not live'"``);
run explicitly with ``pytest -m live tests/e2e``.

These run against free public RPC endpoints on a weekly cron, so "the endpoint
was down, slow, or throttling us" is a routine, uninteresting outcome that must
not page anyone: every network call is wrapped in ``rpc_available()``, which
turns exactly those failures into a skip. What stays a *failure* is the thing
these tests exist to detect -- live data that does not match what we recorded.
Assertions are therefore deliberately kept *outside* the guarded blocks.
"""
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest
import requests

from robustrep.config import Config
from robustrep.sources import base_erc8004 as base
from robustrep.sources.rpc import RpcClient, RpcError
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
    "rate-limited", "-32016", "-32005",
)

# Transport-level failures and the server-side 5xx family: the endpoint is
# broken or overloaded, which says nothing about the chain.
TRANSPORT_MARKERS = (
    "connectionerror", "connecttimeout", "readtimeout", "timeout", "sslerror",
    "chunkedencodingerror", "toomanyredirects",
    "http 500", "http 502", "http 503", "http 504",
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
    if isinstance(exc, RpcError):
        low = str(exc).lower()
        if any(marker in low for marker in RATE_LIMIT_MARKERS):
            return "rate limited"
        for marker in TRANSPORT_MARKERS:
            if marker in low:
                return f"transport failure: {marker}"
    return None


@contextmanager
def rpc_available():
    """Wrap live RPC calls: an unavailable/throttled endpoint becomes a skip,
    anything else propagates unchanged.

    Keep assertions *outside* this block -- a mismatch between live data and
    the recorded fixture is precisely what these tests are for and must fail.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 -- re-raised unless classified skippable
        reason = rpc_unavailable_reason(exc)
        if reason is None:
            raise
        pytest.skip(f"RPC unavailable: {reason}")


@pytest.mark.live
def test_live_recent_blocks(tmp_path):
    cfg = Config()
    rpc = RpcClient(cfg.rpc_urls, user_agent=cfg.user_agent)
    store = Store(tmp_path / "live.db")
    with rpc_available():
        head = int(rpc.call("eth_blockNumber", []), 16)
        base.sync_feedback(store, rpc, chunk=2000, start_block=head - 3999, end_block=head)
    assert store.get_sync("last_block") == str(head)
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

    Only the ``eth_getLogs`` call is guarded by ``rpc_available()``: an
    unreachable or throttling endpoint tells us nothing, but every comparison
    below runs unguarded so drift fails loudly.
    """
    cfg = Config()
    rpc = RpcClient(cfg.rpc_urls, user_agent=cfg.user_agent)
    with rpc_available():
        logs = rpc.call("eth_getLogs", [{
            "fromBlock": hex(FIX["from_block"]), "toBlock": hex(FIX["to_block"]),
            "address": base.REPUTATION_REGISTRY,
            "topics": [[base.TOPIC_NEW_FEEDBACK, base.TOPIC_REVOKED, base.TOPIC_RESPONSE]],
        }])

    def feedback_keys(raw_logs):
        decoded = [base.decode_log(l) for l in raw_logs]
        return {(d["agent_id"], d["client"], d["feedback_index"])
                for d in decoded if d and d["kind"] == "feedback"}

    live_new_feedback = sum(1 for l in logs if l["topics"][0].lower() == base.TOPIC_NEW_FEEDBACK)
    fixture_new_feedback = sum(1 for l in FIX["logs"] if l["topics"][0].lower() == base.TOPIC_NEW_FEEDBACK)
    assert live_new_feedback == fixture_new_feedback

    assert feedback_keys(logs) == feedback_keys(FIX["logs"])
