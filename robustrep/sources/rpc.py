"""Tiny JSON-RPC client: retries with backoff, rotates across URLs, batch support.

Designed for public/free RPC endpoints (e.g. Base's ``https://mainnet.base.org``)
that are flaky, rate-limit aggressively, and sometimes reject requests without a
recognizable ``User-Agent``. ``RpcClient`` retries each request up to ``retries``
times with exponential backoff (1s, 2s, 4s, ...capped at 30s), rotating to the
next configured URL on every attempt (including the first retry) so a single bad
endpoint does not stall the whole sync.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import requests


class RpcError(RuntimeError):
    """Raised when a JSON-RPC call/batch fails on every configured retry, or
    immediately for a non-retryable error (see ``NON_RETRYABLE``)."""


# Substrings (case-insensitive) of a JSON-RPC error message/code that indicate the
# request itself is malformed or permanently rejected -- retrying identically (or
# rotating URL) cannot help, so these are raised immediately instead of consuming
# the retry budget.
NON_RETRYABLE = ("limited to a", "invalid params", "-32602", "-32600")


def _is_non_retryable(message: str) -> bool:
    low = message.lower()
    return any(s.lower() in low for s in NON_RETRYABLE)


class RpcClient:
    """Minimal JSON-RPC 2.0 HTTP client with retry/backoff and URL rotation.

    Parameters mirror what a chain adapter needs: a list of candidate RPC
    ``urls`` (rotated through on failure), a mandatory ``user_agent`` (some
    public endpoints 403 the default ``python-requests`` UA), an injectable
    ``session`` (for tests) and ``sleep`` function (so backoff is instant in
    tests), and ``retries``/``timeout`` knobs.
    """

    def __init__(self, urls, user_agent: str, session=None, retries: int = 5, timeout: int = 30,
                 sleep: Callable[[float], None] = time.sleep):
        self.urls, self.i = list(urls), 0
        self.headers = {"content-type": "application/json", "user-agent": user_agent}
        self.session = session or requests.Session()
        self.retries, self.timeout, self.sleep = retries, timeout, sleep

    def _post(self, payload):
        url = self.urls[self.i % len(self.urls)]
        r = self.session.post(url, json=payload, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _with_retry(self, payload, check: Callable[[object], Optional[str]]):
        """Post ``payload``, retrying (rotating URL, sleeping with backoff) while
        ``check(body)`` returns a non-``None`` error message. Raises ``RpcError``
        naming the last failure once ``retries`` attempts are exhausted, or
        immediately (no rotation, no sleep, no further attempts) if ``check``
        reports a ``NON_RETRYABLE`` error. The final failed attempt is never
        followed by a sleep, since nothing more will be tried afterwards."""
        last: Optional[Exception] = None
        for attempt in range(self.retries):
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
            self.i += 1
            if attempt < self.retries - 1:
                self.sleep(min(2 ** attempt, 30))
        raise RpcError(f"gave up after {self.retries} attempts: {last}")

    @staticmethod
    def _error_message(err) -> str:
        return err.get("message", "error") if isinstance(err, dict) else str(err)

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
        if not isinstance(body, list) or len(body) != n:
            got = len(body) if isinstance(body, list) else type(body).__name__
            return f"malformed batch response: expected {n} entries, got {got}"
        ids = sorted(x.get("id") for x in body if isinstance(x, dict))
        if ids != list(range(1, n + 1)):
            return f"malformed batch response: ids {ids} != 1..{n}"
        for item in body:
            err = item.get("error")
            if err is not None:
                return f"batch entry id={item.get('id')}: {cls._error_message(err)}"
            if item.get("result") is None:
                return f"batch entry id={item.get('id')}: missing result"
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

        Raises ``RpcError`` if every retry either transport-fails, returns a
        response that isn't a list of exactly ``len(calls)`` entries, returns
        entries whose ids aren't exactly ``{1..len(calls)}``, or contains an
        entry with a non-null ``error`` or a missing/null ``result``. A
        non-retryable error (see ``NON_RETRYABLE``) raises immediately.
        """
        n = len(calls)
        payload = [{"jsonrpc": "2.0", "id": i + 1, "method": m, "params": p} for i, (m, p) in enumerate(calls)]
        body = self._with_retry(payload, lambda b: self._check_batch(n, b))
        by_id = {x["id"]: x for x in body}
        return [by_id[i + 1]["result"] for i in range(n)]
