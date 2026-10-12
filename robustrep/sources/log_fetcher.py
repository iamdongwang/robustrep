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
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, NamedTuple, Optional

from .rpc import RpcClient, RpcError, RpcRangeTooLargeError, _endpoint

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

DEFAULT_MAX_WINDOW = 2000
DEFAULT_MIN_WINDOW = 50


def shrink(window: int, numbers: Iterable[int], min_window: int) -> int:
    """The next, smaller window after ``window`` blocks were rejected as too
    large. ``numbers`` are the integers in the server's message; one in
    ``[min_window, window)`` is taken as the server's stated limit, so the
    window jumps straight to it (but always at least halves). With no such
    number the window halves. Never below ``min_window``."""
    cands = [n for n in numbers if min_window <= n < window]
    new = min(max(cands), window // 2) if cands else window // 2
    return max(new, min_window)


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
    ``max_window``). Keys of both ``window_hints`` and ``windows`` are hosts
    in ``scheme://hostname`` form (as ``_endpoint`` produces), NOT full URLs.
    A host's window only ever shrinks: when it rejects a range as too large
    the window is cut with ``shrink``; when it rejects even ``min_window``
    blocks it is added to ``dead`` and skipped for the rest of this
    fetcher's life. Rounds that end on a dead host count as invalid; more
    than ``len(rpc.urls)`` of them in one ``get_logs`` raise ``RpcError``.
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

    @classmethod
    def from_config(cls, rpc: RpcClient, cfg: "Config") -> "LogFetcher":
        """Build from a ``Config``; ``min_window`` is clamped to ``chunk_blocks``
        so a small ``--chunk`` never makes ``min_window > max_window``."""
        return cls(rpc, max_window=cfg.chunk_blocks,
                   min_window=min(cfg.rpc_min_window, cfg.chunk_blocks),
                   window_hints=dict(cfg.rpc_window_hints),
                   min_interval=cfg.rpc_min_interval)

    def get_logs(self, from_block: int, to_block: int, filter: dict) -> FetchResult:
        """Return every log in ``[from_block, to_block]`` (inclusive) matching
        ``filter``. Raises ``ValueError`` if ``from_block > to_block``; any
        ``RpcError`` propagates and no partial result is returned."""
        if from_block > to_block:
            raise ValueError(f"from_block ({from_block}) > to_block ({to_block})")
        logs: list[dict] = []
        served: list[Served] = []
        cursor = from_block
        invalid_rounds = 0
        while cursor <= to_block:
            host = self._live_host()
            window = self.windows.get(host, self.max_window)
            sub_to = min(cursor + window - 1, to_block)
            w = sub_to - cursor + 1  # the size actually requested (after truncation)
            try:
                sub_logs = self._call_get_logs(cursor, sub_to, filter)
            except RpcRangeTooLargeError as e:
                # e.endpoint is the host that raised, which may differ from
                # ``host`` if the client rotated within this call.
                if w <= self.min_window:
                    self.dead.add(e.endpoint)
                    logger.warning("log_fetcher: %s rejected the minimum window (%d); skipping it for this run",
                                   e.endpoint, self.min_window)
                    invalid_rounds += 1
                    self.rpc.rotate()
                elif e.endpoint in self.dead:
                    invalid_rounds += 1
                    self.rpc.rotate()
                else:
                    current = self.windows.get(e.endpoint, self.max_window)
                    self.windows[e.endpoint] = min(current, shrink(w, e.numbers, self.min_window))
                if invalid_rounds > len(self.rpc.urls):
                    raise self._no_usable_endpoint()
                continue
            logs.extend(sub_logs)
            served.append(Served(cursor, sub_to, _endpoint(self.rpc.current_url), window, len(sub_logs)))
            cursor = sub_to + 1
        return FetchResult(logs=logs, served=served)

    def _live_host(self) -> str:
        """Host of ``rpc.current_url``, rotating past dead hosts first.
        Raises ``RpcError`` if every configured URL's host is dead."""
        if all(_endpoint(u) in self.dead for u in self.rpc.urls):
            raise self._no_usable_endpoint()
        host = _endpoint(self.rpc.current_url)
        while host in self.dead:
            self.rpc.rotate()
            host = _endpoint(self.rpc.current_url)
        return host

    def _no_usable_endpoint(self) -> RpcError:
        return RpcError(f"no usable endpoint: {len(self.dead)} of {len(self.rpc.urls)} "
                        "endpoints rejected the minimum window")

    def _call_get_logs(self, a: int, z: int, filter: dict) -> list[dict]:
        self._pace()
        self._last_start = self._clock()
        result = self.rpc.call("eth_getLogs", [{"fromBlock": hex(a), "toBlock": hex(z), **filter}])
        if not isinstance(result, list):
            raise RpcError(f"eth_getLogs returned a non-list result from {_endpoint(self.rpc.current_url)}")
        return result

    def _pace(self) -> None:
        """Sleep so consecutive call STARTS are >= ``min_interval`` apart; a
        call that itself took longer than the interval needs no sleep."""
        if self._last_start is None:
            return
        wait = self._last_start + self.min_interval - self._clock()
        if wait > 0:
            self._sleep(wait)
