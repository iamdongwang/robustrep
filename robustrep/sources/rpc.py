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

Two kinds of JSON-RPC error are not retried: a request the endpoint rejects
as malformed or over its limits (``NON_RETRYABLE``), and an ``eth_call`` the
node executed and that reverted (``RpcRevertError``) -- the second is an
answer about the contract, not the endpoint, and asking again cannot change
it. ``batch(..., per_entry_errors=True)`` goes one step further for callers
that ask many independent contracts one question each: a failed entry comes
back as an ``RpcEntryError`` in its slot instead of failing the batch.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
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

# The one non-2xx status whose JSON-RPC error body is classified like a 200's
# (see ``RpcClient._post``): base.drpc.org reports its range cap as HTTP 400.
HTTP_BAD_REQUEST = 400


class RpcError(RuntimeError):
    """Raised when a JSON-RPC call/batch fails on every configured retry, or
    immediately for a non-retryable error (see ``NON_RETRYABLE``)."""


class RpcRevertError(RpcError):
    """The node executed the call and it reverted (``execution reverted``,
    JSON-RPC code 3 on Base's public nodes, ``-32000`` on some clients).
    Raised immediately -- no retry, rotation or sleep -- for a single call
    and for a ``batch()`` entry alike (the batch message names the entry
    id): the answer is a property of the contract, not of the endpoint, so
    re-asking cannot change it. Measured 2026-10-13, treating it as
    retryable cost 5 POSTs and 15 s of backoff per reverting call. Callers
    that read a revert as "selector not implemented" (``base_erc8004.
    owner_of``, ``account_owner``) test ``isinstance`` on this class via
    ``is_revert`` instead of substring-matching the message."""


@dataclass(frozen=True)
class RpcEntryError:
    """One failed entry of a ``batch(..., per_entry_errors=True)`` call, in
    the slot its call occupied. ``message`` has already been through
    ``scrub_rpc_text`` -- it is our ``[code]`` token at most, never server
    text, so it can be logged and stored as is. ``reverted`` is True for an
    ``execution reverted`` entry (the contract has no such function or
    refused the call); any other entry error is a node-side execution
    failure for that call alone (out of gas, invalid opcode, ...)."""
    code: Optional[int]
    message: str
    reverted: bool


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


class RpcRangeTooLargeError(RpcError):
    """The endpoint rejected an ``eth_getLogs`` block range as too large.
    Raised immediately (no retry, rotation or sleep) so the caller can shrink
    its window. Only integers leave this module: ``numbers`` holds the integers
    found in the server's message (in order of appearance) and the exception
    text is fixed -- never server text, which may echo a provider API key."""

    def __init__(self, endpoint: str, numbers: tuple[int, ...] = ()):
        super().__init__(f"range too large for {endpoint}")
        self.endpoint = endpoint
        self.numbers = numbers


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
NON_RETRYABLE = ("limited to a", "invalid params", "-32602", "-32600", "execution reverted")

# The ``NON_RETRYABLE`` entry that means "the call executed and reverted" --
# raised as ``RpcRevertError`` rather than a plain ``RpcError`` so callers
# can tell "this contract does not implement the selector" from "the request
# was malformed" by type.
REVERT_MARKER = "execution reverted"

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


def _is_range_too_large(message: str) -> bool:
    """True if ``message`` is a provider's "eth_getLogs block range too large" rejection."""
    low = message.lower()
    return ("block range too large" in low
            or "blocks range" in low
            or "ranges over" in low
            or ("limited to a" in low and "range" in low)
            or ("up to a" in low and "block range" in low)
            or ("block range" in low and any(k in low for k in ("limited", "maximum", "up to"))))


def _numbers_in(message: str, limit: int = 8) -> tuple[int, ...]:
    """The first ``limit`` unsigned integers in ``message``, in order, thousands separators removed."""
    return tuple(int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*", message)[:limit])


def _is_revert(message: str) -> bool:
    return REVERT_MARKER in message.lower()


def is_revert(e: BaseException) -> bool:
    """True when ``e`` says the call reverted -- an ``RpcRevertError``, which
    only ``_raise_if_fatal`` builds from a JSON-RPC error carrying the revert
    marker. Deliberately a type check and nothing else: an ``RpcError`` whose
    *text* merely mentions "revert" (a provider's prose after the retries
    are exhausted, an endpoint hostname) is a transport failure, and reading
    it as the contract's answer would turn a transient outage into a sticky
    "no owner" verdict for that address (re-review I-3)."""
    return isinstance(e, RpcRevertError)


# ``[<int>]`` for a numeric JSON-RPC code, or the ``[?]`` placeholder
# ``_error_message`` emits when the server sent none -- so a message always
# carries a token and ``scrub_rpc_text`` can always drop the server text.
_RPC_CODE_TOKEN = re.compile(r"\[(?:-?\d{1,10}|\?)\]")
NO_CODE_TOKEN = "[?]"
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_ERROR_TEXT_CAP = 200


def scrub_rpc_text(text: str) -> str:
    """The part of an RPC error message that is safe to log and store.

    Server text in this module's messages only ever follows a ``[code]``
    token (``"gave up after 5 attempts: [-32016] <server text>"``; ``[?]``
    when the server sent no code), so keep our own prefix before the first
    ``[`` plus that token, and drop everything after it. Control characters
    become spaces (no log-line or CI workflow-command injection), whitespace
    runs collapse, and the result is capped at 200 characters."""
    head, bracket, rest = text.partition("[")
    if bracket:
        code = _RPC_CODE_TOKEN.match("[" + rest)
        head = head + code.group(0) if code else head
    return " ".join(_CONTROL_CHARS.sub(" ", head).split())[:_ERROR_TEXT_CAP]


def scrub_rpc_error(e: BaseException) -> str:
    """``scrub_rpc_text`` of ``str(e)``."""
    return scrub_rpc_text(str(e))


def _raise_if_fatal(err: str, url: str) -> None:
    """Raise (no further retries) if ``err`` is a range-too-large rejection
    (``RpcRangeTooLargeError``), an execution revert (``RpcRevertError``) or
    names any other ``NON_RETRYABLE`` error (``RpcError``)."""
    if _is_range_too_large(err):
        logger.debug("rpc: range too large from %s: %s", _endpoint(url), err)
        raise RpcRangeTooLargeError(_endpoint(url), _numbers_in(err))
    if _is_revert(err):
        raise RpcRevertError(f"non-retryable error: {err}")
    if _is_non_retryable(err):
        raise RpcError(f"non-retryable error: {err}")


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
    def current_url(self) -> str:
        """The URL the next request will go to."""
        return self.urls[self.i % len(self.urls)]

    def rotate(self) -> None:
        """Advance to the next configured URL (wraps around)."""
        self.i += 1

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

        For a single call (``payload`` is a dict, not a batch list), an HTTP
        **400** whose body decodes to a JSON object carrying an ``error`` key
        is returned like a 200 body, so ``_with_retry``'s
        ``check`` classifies it as usual (range-too-large, ``NON_RETRYABLE``,
        or retry/rotate): base.drpc.org reports its ``eth_getLogs`` range cap
        this way. A 400 whose body is not such an object (not JSON, oversized,
        or JSON without ``error``) raises ``HTTPError`` as before, and so does
        any 400 to a batch POST: a lone error object there would reach
        ``_check_batch`` as a structure error, which is non-retryable and turns
        batching off for the rest of the run. Every other
        non-2xx status -- 403, 429, 5xx -- raises ``HTTPError`` without its
        body being read: a parsed 403 archive refusal (``-32602``) would be
        ``NON_RETRYABLE`` and lose rotation, and a 429 must stay recognisable
        as ``HTTPError (HTTP 429)``.
        """
        url = self.current_url
        r = self.session.post(url, json=payload, headers=self.headers, timeout=self.timeout, stream=True)
        try:
            if r.status_code == HTTP_BAD_REQUEST and isinstance(payload, dict):
                body = self._json_error_body(r)
                if body is not None:
                    return body
            r.raise_for_status()
            return read_json_capped(r, self.max_response_bytes)
        except ResponseTooLarge:
            raise RpcError(f"response from {_endpoint(url)} exceeds the "
                           f"{self.max_response_bytes} byte limit") from None
        finally:
            r.close()

    def _json_error_body(self, r) -> Optional[dict]:
        """The body of the non-2xx response ``r`` if it is a JSON object with an
        ``error`` key, else ``None`` (also when it is not JSON or is over
        ``max_response_bytes`` -- the caller then raises the HTTP error)."""
        try:
            body = read_json_capped(r, self.max_response_bytes)
        except ValueError:      # includes ResponseTooLarge
            return None
        return body if isinstance(body, dict) and body.get("error") is not None else None

    def _with_retry(self, payload, check: Callable[[object], Optional[str]]):
        """Post ``payload``, retrying (rotating URL, sleeping with backoff) while
        ``check(body)`` returns a non-``None`` error message. ``body`` is a
        2xx response's decoded JSON, or, for a single call, an HTTP 400's
        JSON-RPC error object (see ``_post``); any other non-2xx status reaches this loop as
        ``requests.HTTPError`` instead and is retried as a transport failure. Raises ``RpcError``
        naming the last failure once ``retries`` attempts are exhausted, or
        immediately (no rotation, no sleep, no further attempts) if ``check``
        reports a ``NON_RETRYABLE`` error (a revert raises ``RpcRevertError``,
        a range-too-large one ``RpcRangeTooLargeError``), or if ``check`` itself raises (as
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
        last_url = self.current_url
        for attempt in range(self.retries):
            last_url = self.current_url
            try:
                body = self._post(payload)
                err = check(body)
                if err is None:
                    return body
                _raise_if_fatal(err, last_url)
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
        # Always a token, so ``scrub_rpc_text`` has something to keep and a
        # code-less server message can never survive scrubbing (I-2).
        token = f"[{code}]" if code is not None else NO_CODE_TOKEN
        return f"{token} {msg}"

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

    @staticmethod
    def _check_batch_shape(n: int, body) -> None:
        """Raise ``RpcBatchStructureError`` unless ``body`` is a list of
        exactly ``n`` entries whose ids are exactly ``1..n``."""
        if not isinstance(body, list) or len(body) != n:
            got = len(body) if isinstance(body, list) else type(body).__name__
            raise RpcBatchStructureError(f"malformed batch response: expected {n} entries, got {got}")
        ids = sorted(x.get("id") for x in body if isinstance(x, dict))
        if ids != list(range(1, n + 1)):
            raise RpcBatchStructureError(f"malformed batch response: ids {ids} != 1..{n}")

    @classmethod
    def _check_batch(cls, n: int, body, per_entry_errors: bool = False) -> Optional[str]:
        """Validate a batch response body, either returning a retryable error
        message (an ordinary per-entry JSON-RPC error) or raising directly
        for a failure that retrying cannot fix -- see ``RpcBatchStructureError``
        (malformed shape: not a list, wrong count, id mismatch, an entry with
        neither error nor result) and ``RpcBatchRateLimitError`` (an entry's
        error reports the endpoint is rate-limiting this batch specifically).
        Raising instead of returning skips ``_with_retry``'s
        retry/backoff/rotation entirely, the same way a ``NON_RETRYABLE``
        single-call error does.

        With ``per_entry_errors`` an entry's own JSON-RPC error is not a
        failure of the batch at all: it is returned to the caller in that
        entry's slot (see ``batch``), so only the shape and rate-limit checks
        apply and the method never returns a retryable message."""
        cls._check_batch_shape(n, body)
        for item in body:
            err = item.get("error")
            if err is not None:
                msg = cls._error_message(err)
                # A reverted entry is the contract's own answer (its revert
                # reason is attacker-chosen text), never the endpoint saying
                # the batch is rate-limited -- so it is not allowed to switch
                # batching off for the rest of the run (re-review I-1).
                if not _is_revert(msg) and _is_batch_rate_limited(msg):
                    raise RpcBatchRateLimitError(f"batch entry id={item.get('id')}: {msg}")
                if not per_entry_errors:
                    return f"batch entry id={item.get('id')}: {msg}"
            elif item.get("result") is None:
                raise RpcBatchStructureError(f"batch entry id={item.get('id')}: missing result")
        return None

    @classmethod
    def _entry_error(cls, err) -> RpcEntryError:
        """An entry's JSON-RPC ``error`` object as an ``RpcEntryError``: the
        integer ``code`` (``None`` when absent or not an int), the scrubbed
        ``[code] message`` text, and whether it is a revert."""
        code = err.get("code") if isinstance(err, dict) else None
        if isinstance(code, bool) or not isinstance(code, int):
            code = None
        msg = cls._error_message(err)
        return RpcEntryError(code=code, message=scrub_rpc_text(msg), reverted=_is_revert(msg))

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

    def batch(self, calls: list[tuple[str, list]], *, per_entry_errors: bool = False) -> list:
        """Make a JSON-RPC batch call and return results in the same order as
        ``calls``, matched by response ``id`` (not response position -- servers
        are free to return batch entries in any order).

        By default (``per_entry_errors=False``) raises ``RpcError`` if every
        retry transport-fails or an entry has a non-null JSON-RPC ``error``
        (retried like a single-call error, subject to the same
        ``NON_RETRYABLE`` immediate-raise -- a reverting entry raises
        ``RpcRevertError`` naming its id on the first attempt). Raises
        immediately -- no retry, no rotation, no backoff -- via a more
        specific subclass when the failure means retrying the batch itself is
        pointless: ``RpcBatchStructureError`` (the response isn't a list of
        exactly ``len(calls)`` entries, entry ids aren't exactly
        ``{1..len(calls)}``, or an entry has neither ``result`` nor ``error``)
        or ``RpcBatchRateLimitError`` (an entry's error reports the batch
        itself is rate-limited).

        With ``per_entry_errors=True`` an entry's own JSON-RPC error does
        not fail the batch: its slot in the returned list holds an
        ``RpcEntryError`` (scrubbed message, ``reverted`` flag) and every
        other slot holds its ``result`` as usual, so a caller asking fifty
        independent contracts one question each learns every answer in one
        round trip even when some of them revert. Structural failures,
        batch-level rate limits and transport failures behave exactly as
        in the default mode.
        """
        n = len(calls)
        payload = [{"jsonrpc": "2.0", "id": i + 1, "method": m, "params": p} for i, (m, p) in enumerate(calls)]
        body = self._with_retry(payload, lambda b: self._check_batch(n, b, per_entry_errors))
        by_id = {x["id"]: x for x in body}
        out = []
        for i in range(n):
            item = by_id[i + 1]
            err = item.get("error")
            out.append(self._entry_error(err) if per_entry_errors and err is not None else item["result"])
        return out
