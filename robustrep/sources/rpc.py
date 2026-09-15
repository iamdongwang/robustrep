"""Tiny JSON-RPC client: retries with backoff, rotates across URLs, batch support.

Designed for public/free RPC endpoints (e.g. Base's ``https://mainnet.base.org``)
that are flaky, rate-limit aggressively, and sometimes reject requests without a
recognizable ``User-Agent``. ``RpcClient`` retries each request up to ``retries``
times with exponential backoff (1s, 2s, 4s, ...capped at 30s), rotating to the
next configured URL on every attempt (including the first retry) so a single bad
endpoint does not stall the whole sync.

Response bodies are read through ``http_util.read_json_capped`` with a
``max_response_bytes`` cap (security review finding M2): a public endpoint is an
untrusted third party, and ``requests``' ``Response.json()`` would happily buffer
an unbounded -- or deliberately endless -- body straight into this process's
memory.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional
from urllib.parse import urlsplit

import requests

from .http_util import ResponseTooLarge, read_json_capped

logger = logging.getLogger(__name__)

# Largest JSON-RPC response body this client will buffer (M2). A single
# ``eth_getLogs`` chunk over a busy 2,000-block range is comfortably inside
# this; a body past it is either a broken endpoint or an attempt to exhaust
# this process's memory, and either way is not worth parsing. See
# ``http_util.read_json_capped``.
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class RpcError(RuntimeError):
    """Raised when a JSON-RPC call/batch fails on every configured retry, or
    immediately for a non-retryable error (see ``NON_RETRYABLE``)."""


class RpcBatchUnsupportedError(RpcError):
    """Base class for a batch-level failure that means *batching itself* is
    unusable against this endpoint -- as opposed to an ordinary transient
    failure or a single call's own error. Raised immediately (no retry, no
    URL rotation, no backoff): identically re-attempting the same batch
    cannot succeed, so retrying would only burn the full backoff schedule
    for a guaranteed repeat failure. Callers that batch across many chunks
    (see ``base_erc8004.fill_block_timestamps``/``owners_of``) catch this
    base class specifically to permanently stop attempting ``batch()`` for
    the rest of the run and fall back to per-item calls (each of which keeps
    retrying normally -- only *batching* is abandoned, not the RPC calls
    themselves)."""


class RpcBatchStructureError(RpcBatchUnsupportedError):
    """A batch response that is structurally wrong: not a list, the wrong
    number of entries, mismatched ids, or an entry with a missing/null
    ``result``. Unlike a JSON-RPC error body (a legitimate per-call failure,
    e.g. a revert) or a transport error (plausibly transient), this means the
    endpoint does not support -- or mishandles -- JSON-RPC batching at all
    (observed in the wild: a public node returning a single error *object*
    instead of a *list* for a 100-call batch)."""


class RpcBatchRateLimitError(RpcBatchUnsupportedError):
    """A batch request whose response reports an "over rate limit" JSON-RPC
    error (see ``BATCH_RATE_LIMIT``) on any entry. Observed in the wild:
    ``mainnet.base.org`` throttles *batched* requests far more aggressively
    than the equivalent single calls, returning ``[-32016] over rate limit``
    reliably on every 100-call batch while individual calls succeed fine --
    so, unlike an ordinary rate-limited single call (which should keep
    retrying/rotating as normal), a batch hitting this is a signal to stop
    batching entirely, not to keep re-attempting the same doomed batch."""


def _endpoint(url: str) -> str:
    """``url`` reduced to ``scheme://host`` -- strips path, query string,
    port and userinfo, any of which may embed a provider API key (many paid
    RPC providers put it right in the URL path, e.g. ``.../v2/<key>``)."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname}"


def _redact(last: Optional[BaseException], url: str) -> str:
    """Render the last retry failure for an ``RpcError`` message without ever
    including the raw ``requests``/``urllib3`` exception text or the full
    request URL -- both routinely echo the whole URL (query string and all)
    in their ``str()``, which would leak an API key embedded in it straight
    into logs, CLI output or a bug report. Only the exception type, an HTTP
    status code when available, and the endpoint's bare ``scheme://host``
    survive into the message; full detail still reaches DEBUG-level logs.

    ``last`` being our own ``RpcError`` (raised from ``check()`` on a
    malformed/JSON-RPC-level error) is not a raw transport exception -- its
    message is text we constructed ourselves -- so it is passed through
    unredacted.
    """
    endpoint = _endpoint(url)
    if isinstance(last, requests.exceptions.HTTPError):
        status = getattr(getattr(last, "response", None), "status_code", None)
        status_part = f" (HTTP {status})" if status is not None else ""
        return f"HTTPError{status_part} from {endpoint}"
    if isinstance(last, requests.RequestException):
        return f"{type(last).__name__} contacting {endpoint}"
    return str(last)


# Substrings (case-insensitive) of a JSON-RPC error message/code that indicate the
# request itself is malformed or permanently rejected -- retrying identically (or
# rotating URL) cannot help, so these are raised immediately instead of consuming
# the retry budget.
NON_RETRYABLE = ("limited to a", "invalid params", "-32602", "-32600")

# Substrings (case-insensitive) of a JSON-RPC error message/code that mean "you
# are being rate-limited" -- checked only against a *batch* entry's error (see
# ``_check_batch``). A single call hitting one of these should still retry/rotate
# normally (rate limits are transient); a *batch* hitting one is instead treated
# as evidence the endpoint throttles batching specifically, see
# ``RpcBatchRateLimitError``.
BATCH_RATE_LIMIT = ("over rate limit", "-32016", "-32005")


def _is_non_retryable(message: str) -> bool:
    low = message.lower()
    return any(s.lower() in low for s in NON_RETRYABLE)


def _is_batch_rate_limited(message: str) -> bool:
    low = message.lower()
    return any(s.lower() in low for s in BATCH_RATE_LIMIT)


class RpcClient:
    """Minimal JSON-RPC 2.0 HTTP client with retry/backoff and URL rotation.

    Parameters mirror what a chain adapter needs: a list of candidate RPC
    ``urls`` (rotated through on failure), a mandatory ``user_agent`` (some
    public endpoints 403 the default ``python-requests`` UA), an injectable
    ``session`` (for tests) and ``sleep`` function (so backoff is instant in
    tests), and ``retries``/``timeout`` knobs.

    ``max_response_bytes`` caps how much of a response body is ever buffered
    (M2); a body past it fails the call immediately and non-retryably (see
    ``_post``).

    ``session`` is thread-local when not injected: each calling thread gets
    its own lazily-created ``requests.Session`` (via the ``session`` property
    below), since a single ``requests.Session`` is not guaranteed safe to
    share across threads and ``classify_all`` may drive one ``RpcClient``
    from a thread pool. An explicitly injected ``session`` (the test-only
    path) is used as-is, shared by every thread -- tests rely on this to
    observe every call through one fake session.
    """

    def __init__(self, urls, user_agent: str, session=None, retries: int = 5, timeout: int = 30,
                 sleep: Callable[[float], None] = time.sleep,
                 max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES):
        self.urls, self.i = list(urls), 0
        self.headers = {"content-type": "application/json", "user-agent": user_agent}
        self._injected_session = session
        self._local = threading.local()
        self.retries, self.timeout, self.sleep = retries, timeout, sleep
        self.max_response_bytes = max_response_bytes

    @property
    def session(self):
        """The ``requests.Session`` to use on the calling thread.

        Returns the injected session as-is when one was given to ``__init__``
        (shared across every thread -- this is what tests inject and assert
        against). Otherwise returns this thread's own lazily-created session,
        creating one on first access from a given thread.
        """
        if self._injected_session is not None:
            return self._injected_session
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            self._local.session = sess
        return sess

    def _post(self, payload):
        """POST ``payload`` and return the decoded body, reading at most
        ``max_response_bytes`` of it (M2 -- see ``http_util.read_json_capped``).

        The request is streamed so the cap is enforced while reading rather
        than after ``requests`` has already buffered the whole body, and the
        response is always closed. An oversized body raises ``RpcError``
        *immediately*: unlike a timeout or a rate limit, "this endpoint
        answered with more bytes than we will parse" is deterministic, so
        ``_with_retry`` must not retry it (``RpcError`` is not one of the
        exception types it catches) -- re-asking would burn the whole retry
        budget, and rotating would drag every other configured endpoint into
        a failure that is not theirs. The message names only the scrubbed
        endpoint and the limit: never the body, never the full URL (which may
        embed a provider API key).
        """
        url = self.urls[self.i % len(self.urls)]
        r = self.session.post(url, json=payload, headers=self.headers, timeout=self.timeout, stream=True)
        try:
            r.raise_for_status()
            return read_json_capped(r, self.max_response_bytes)
        except ResponseTooLarge:
            raise RpcError(f"response from {_endpoint(url)} exceeds the "
                           f"{self.max_response_bytes} byte limit") from None
        finally:
            r.close()

    def _with_retry(self, payload, check: Callable[[object], Optional[str]]):
        """Post ``payload``, retrying (rotating URL, sleeping with backoff) while
        ``check(body)`` returns a non-``None`` error message. Raises ``RpcError``
        naming the last failure once ``retries`` attempts are exhausted, or
        immediately (no rotation, no sleep, no further attempts) if ``check``
        reports a ``NON_RETRYABLE`` error, or if ``check`` itself raises (as
        ``_check_batch`` does for ``RpcBatchStructureError``/``RpcBatchRateLimitError``
        -- neither of these is caught here, so they propagate straight through
        on the very first attempt). The final failed attempt is never followed
        by a sleep, since nothing more will be tried afterwards.

        The final ``RpcError``'s message is redacted (see ``_redact``): it
        never contains the raw exception text or full request URL, only the
        exception type/status code and the endpoint's scheme+host. Full,
        unredacted detail (exception + traceback, against the actual URL
        attempted) is logged at DEBUG on every failed attempt instead."""
        last: Optional[Exception] = None
        last_url = self.urls[self.i % len(self.urls)]
        for attempt in range(self.retries):
            last_url = self.urls[self.i % len(self.urls)]
            try:
                body = self._post(payload)
                err = check(body)
                if err is None:
                    return body
                if _is_non_retryable(err):
                    raise RpcError(f"non-retryable error: {err}")
                last = RpcError(err)
            except (requests.RequestException, ValueError) as e:
                last = e
                logger.debug("rpc: attempt %d against %s failed", attempt + 1, _endpoint(last_url), exc_info=True)
            self.i += 1
            if attempt < self.retries - 1:
                self.sleep(min(2 ** attempt, 30))
        raise RpcError(f"gave up after {self.retries} attempts: {_redact(last, last_url)}")

    @staticmethod
    def _error_message(err) -> str:
        """Render a JSON-RPC ``error`` object as a single string that folds in
        its numeric ``code`` (as ``"[code] message"``) so ``NON_RETRYABLE``
        entries that match on a code (e.g. ``"-32602"``) can match regardless
        of whether the server's ``message`` text happens to repeat it."""
        if not isinstance(err, dict):
            return str(err)
        msg = err.get("message", "error")
        code = err.get("code")
        return f"[{code}] {msg}" if code is not None else msg

    @classmethod
    def _check_single(cls, body) -> Optional[str]:
        if not isinstance(body, dict):
            return "malformed response: expected an object"
        err = body.get("error")
        if err is not None:
            return cls._error_message(err)
        if "result" not in body:
            return "malformed response: no 'result' or 'error' field"
        return None

    @classmethod
    def _check_batch(cls, n: int, body) -> Optional[str]:
        """Validate a batch response body, either returning a retryable error
        message (an ordinary per-entry JSON-RPC error) or raising directly
        for a failure that retrying cannot fix -- see ``RpcBatchStructureError``
        (malformed shape: not a list, wrong count, id mismatch, missing/null
        result) and ``RpcBatchRateLimitError`` (an entry's error reports the
        endpoint is rate-limiting this batch specifically). Raising instead
        of returning skips ``_with_retry``'s retry/backoff/rotation entirely,
        the same way a ``NON_RETRYABLE`` single-call error does."""
        if not isinstance(body, list) or len(body) != n:
            got = len(body) if isinstance(body, list) else type(body).__name__
            raise RpcBatchStructureError(f"malformed batch response: expected {n} entries, got {got}")
        ids = sorted(x.get("id") for x in body if isinstance(x, dict))
        if ids != list(range(1, n + 1)):
            raise RpcBatchStructureError(f"malformed batch response: ids {ids} != 1..{n}")
        for item in body:
            err = item.get("error")
            if err is not None:
                msg = cls._error_message(err)
                if _is_batch_rate_limited(msg):
                    raise RpcBatchRateLimitError(f"batch entry id={item.get('id')}: {msg}")
                return f"batch entry id={item.get('id')}: {msg}"
            if item.get("result") is None:
                raise RpcBatchStructureError(f"batch entry id={item.get('id')}: missing result")
        return None

    def call(self, method: str, params: list):
        """Make a single JSON-RPC call and return its ``result``.

        Raises ``RpcError`` if every retry either transport-fails, returns a
        JSON-RPC ``error`` (a null ``error`` is not treated as one), or returns
        a body with neither ``result`` nor ``error`` (a malformed response). A
        non-retryable error (see ``NON_RETRYABLE``) raises immediately.
        """
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        body = self._with_retry(payload, self._check_single)
        return body["result"]

    def batch(self, calls: list[tuple[str, list]]) -> list:
        """Make a JSON-RPC batch call and return results in the same order as
        ``calls``, matched by response ``id`` (not response position -- servers
        are free to return batch entries in any order).

        Raises ``RpcError`` if every retry transport-fails or an entry has a
        non-null JSON-RPC ``error`` (retried like a single-call error, subject
        to the same ``NON_RETRYABLE`` immediate-raise). Raises immediately --
        no retry, no rotation, no backoff -- via a more specific subclass when
        the failure means retrying the batch itself is pointless:
        ``RpcBatchStructureError`` (the response isn't a list of exactly
        ``len(calls)`` entries, entry ids aren't exactly ``{1..len(calls)}``,
        or an entry has a missing/null ``result``) or ``RpcBatchRateLimitError``
        (an entry's error reports the batch itself is rate-limited).
        """
        n = len(calls)
        payload = [{"jsonrpc": "2.0", "id": i + 1, "method": m, "params": p} for i, (m, p) in enumerate(calls)]
        body = self._with_retry(payload, lambda b: self._check_batch(n, b))
        by_id = {x["id"]: x for x in body}
        return [by_id[i + 1]["result"] for i in range(n)]
