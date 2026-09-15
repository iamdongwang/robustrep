import functools
import sqlite3
import traceback

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_records(rows):
    """rows: list of dicts with at least rater, ratee, value. Fills defaults."""
    defaults = dict(scale="d0", tag="quality", ts=0, evidence_uri=None, source="test")
    return pd.DataFrame([{**defaults, **r} for r in rows])


@pytest.fixture
def records_factory():
    return make_records


# --- sqlite connection-leak guard --------------------------------------------
#
# Python 3.13's sqlite3 emits ``ResourceWarning: unclosed database`` when a
# ``Connection`` is garbage-collected without ``close()``. ``filterwarnings =
# ["error"]`` in pyproject.toml turns that into a ``PytestUnraisableException-
# Warning`` that fails whichever *unrelated* test happens to be running when the
# collector runs -- so a leak in one test breaks a different one, at random.
#
# Rather than mask that with ``Store.__del__``, this autouse fixture makes leaks
# impossible to land: every ``sqlite3.connect`` made during a test is tracked and
# the test fails at teardown if any connection was left open.


class TrackedConnection(sqlite3.Connection):
    """``sqlite3.Connection`` that remembers whether ``close()`` was called.

    Note ``__exit__`` is deliberately NOT treated as closing: ``with conn:`` on a
    sqlite3 connection is a *transaction* scope (commit/rollback), not a
    lifetime scope, and ``Store`` uses it that way internally.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.robustrep_closed = False
        self.robustrep_opened_at = "".join(traceback.format_stack()[:-1])

    def close(self) -> None:
        self.robustrep_closed = True
        super().close()


@functools.lru_cache(maxsize=None)
def _tracked_subclass(factory: type) -> type:
    """A ``TrackedConnection``-flavoured subclass of a caller-supplied factory."""
    if issubclass(factory, TrackedConnection):
        return factory
    return type("Tracked" + factory.__name__, (TrackedConnection, factory), {})


@pytest.fixture(autouse=True)
def _no_leaked_sqlite_connections(monkeypatch):
    """Fail any test that leaves a ``sqlite3`` connection open.

    Keeps a strong reference to every connection opened during the test so the
    garbage collector cannot quietly dispose of one (and emit the 3.13
    ResourceWarning) before teardown gets to look at it.
    """
    opened: list[TrackedConnection] = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        factory = kwargs.get("factory", sqlite3.Connection)
        kwargs["factory"] = _tracked_subclass(factory)
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    yield
    monkeypatch.undo()

    leaked = [c for c in opened if not c.robustrep_closed]
    for conn in leaked:  # never let the guard itself leak into the next test
        conn.close()
    if leaked:
        where = "\n".join(f"--- connection {i + 1} opened at ---\n{c.robustrep_opened_at}"
                          for i, c in enumerate(leaked))
        pytest.fail(f"{len(leaked)} sqlite3 connection(s) left open by this test\n{where}",
                    pytrace=False)
