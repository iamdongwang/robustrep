"""Fetch every ``eth_getLogs`` entry in a block range, across capped endpoints.

Free public RPC endpoints cap the block range an ``eth_getLogs`` call may span,
each at its own size (2000, 100, 50, 10 blocks...). ``LogFetcher`` sits between
the sync code and ``RpcClient`` and has one job: given a block range, return
ALL its logs, splitting the range into windows sized for whichever endpoint
is currently in use and concatenating the results in block order. Each served
window is recorded (``Served``) so callers can log provenance. Requests are
optionally paced (``min_interval`` between call starts). Any ``RpcError`` the
fetcher cannot handle propagates unchanged -- there is no partial result.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple, Optional

from .rpc import RpcClient, _endpoint

logger = logging.getLogger(__name__)

DEFAULT_MAX_WINDOW = 2000
DEFAULT_MIN_WINDOW = 50


class Served(NamedTuple):
    """One successful ``eth_getLogs`` sub-request: the inclusive block range,
    the endpoint host (``scheme://hostname``) that served it, the window size
    in force for that host, and how many logs it returned."""

    from_block: int
    to_block: int
    host: str
    window: int
    n_logs: int


@dataclass(frozen=True)
class FetchResult:
    """All logs of a range, in block order, plus the sub-requests that served them."""

    logs: list[dict]
    served: list[Served]


class LogFetcher:
    """Split a block range into per-endpoint windows and fetch all its logs.

    ``windows`` maps endpoint host -> window size currently used for it
    (initially a copy of ``window_hints``; hosts not in it use
    ``max_window``). ``dead`` holds hosts no longer to be used.
    """

    def __init__(self, rpc: RpcClient, *, max_window: int = DEFAULT_MAX_WINDOW,
                 min_window: int = DEFAULT_MIN_WINDOW,
                 window_hints: Optional[Mapping[str, int]] = None,
                 min_interval: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        if min_window < 1:
            raise ValueError(f"min_window must be >= 1, got {min_window}")
        if max_window < min_window:
            raise ValueError(f"max_window ({max_window}) must be >= min_window ({min_window})")
        if min_interval < 0:
            raise ValueError(f"min_interval must be >= 0, got {min_interval}")
        hints = dict(window_hints or {})
        for host, w in hints.items():
            if not min_window <= w <= max_window:
                raise ValueError(f"window hint for {host} ({w}) outside [{min_window}, {max_window}]")
        self.rpc = rpc
        self.max_window, self.min_window, self.min_interval = max_window, min_window, min_interval
        self.windows: dict[str, int] = hints
        self.dead: set[str] = set()
        self._sleep, self._clock = sleep, clock
        self._last_start: Optional[float] = None

    def get_logs(self, from_block: int, to_block: int, filter: dict) -> FetchResult:
        """Return every log in ``[from_block, to_block]`` (inclusive) matching
        ``filter``. Raises ``ValueError`` if ``from_block > to_block``; any
        ``RpcError`` propagates and no partial result is returned."""
        if from_block > to_block:
            raise ValueError(f"from_block ({from_block}) > to_block ({to_block})")
        logs: list[dict] = []
        served: list[Served] = []
        cursor = from_block
        while cursor <= to_block:
            host = _endpoint(self.rpc.current_url)
            # Task 4: if host in self.dead, rpc.rotate() until it is not; all dead -> RpcError.
            w = self.windows.get(host, self.max_window)
            sub_to = min(cursor + w - 1, to_block)
            # Task 4: on RpcRangeTooLargeError, shrink this host's window and retry.
            # Every other RpcError propagates unchanged.
            sub_logs = self._call_get_logs(cursor, sub_to, filter)
            logs.extend(sub_logs)
            served.append(Served(cursor, sub_to, _endpoint(self.rpc.current_url), w, len(sub_logs)))
            cursor = sub_to + 1
        return FetchResult(logs=logs, served=served)

    def _call_get_logs(self, a: int, z: int, filter: dict) -> list[dict]:
        self._pace()
        self._last_start = self._clock()
        return self.rpc.call("eth_getLogs", [{"fromBlock": hex(a), "toBlock": hex(z), **filter}])

    def _pace(self) -> None:
        """Sleep so consecutive call STARTS are >= ``min_interval`` apart; a
        call that itself took longer than the interval needs no sleep."""
        if self._last_start is None:
            return
        wait = self._last_start + self.min_interval - self._clock()
        if wait > 0:
            self._sleep(wait)
