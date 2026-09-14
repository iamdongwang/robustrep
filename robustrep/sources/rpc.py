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
    """Raised when a JSON-RPC call/batch fails on every configured retry."""


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
        naming the last failure once ``retries`` attempts are exhausted."""
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                body = self._post(payload)
                err = check(body)
                if err is None:
                    return body
                last = RpcError(err)
            except (requests.RequestException, ValueError) as e:
                last = e
            self.i += 1
            self.sleep(min(2 ** attempt, 30))
        raise RpcError(f"gave up after {self.retries} attempts: {last}")

    @staticmethod
    def _check_single(body) -> Optional[str]:
        if not isinstance(body, dict):
            return "malformed response: expected an object"
        if "error" in body:
            return body["error"].get("message", "error") if isinstance(body["error"], dict) else str(body["error"])
        if "result" not in body:
            return "malformed response: no 'result' or 'error' field"
        return None

    @staticmethod
    def _check_batch(body) -> Optional[str]:
        if not isinstance(body, list):
            return "malformed batch response: expected a list"
        for item in body:
            if not isinstance(item, dict) or "result" not in item:
                return f"batch entry error: {item}"
        return None

    def call(self, method: str, params: list):
        """Make a single JSON-RPC call and return its ``result``.

        Raises ``RpcError`` if every retry either transport-fails, returns a
        JSON-RPC ``error``, or returns a body with neither ``result`` nor
        ``error`` (a malformed response).
        """
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        body = self._with_retry(payload, self._check_single)
        return body["result"]

    def batch(self, calls: list[tuple[str, list]]) -> list:
        """Make a JSON-RPC batch call and return results in the same order as
        ``calls`` (regardless of the order the server returns them in).

        Raises ``RpcError`` if every retry either transport-fails or returns a
        batch containing an entry without a ``result`` (i.e. an error entry).
        """
        payload = [{"jsonrpc": "2.0", "id": i + 1, "method": m, "params": p} for i, (m, p) in enumerate(calls)]
        body = self._with_retry(payload, self._check_batch)
        return [x["result"] for x in sorted(body, key=lambda x: x["id"])]
