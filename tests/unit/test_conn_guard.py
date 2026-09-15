"""The sqlite connection-leak guard (``tests/conftest.py``) must actually fail a
leaking test.

Without this, the guard is decoration: it would silently stop catching leaks and
the Python 3.13 ``ResourceWarning: unclosed database`` failures -- which land on
whichever unrelated test happens to be running when the collector fires -- would
come back. ``pytester`` runs a throwaway pytest session (in a subprocess, so the
leak it deliberately creates is not also seen by *this* test's own guard) over a
copy of the real root conftest.
"""
from pathlib import Path

pytest_plugins = ["pytester"]

ROOT_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def test_guard_fails_a_test_that_leaks_a_connection(pytester):
    pytester.makeconftest(ROOT_CONFTEST.read_text())
    pytester.makepyfile("""
        import sqlite3

        def test_leaks(tmp_path):
            sqlite3.connect(str(tmp_path / "leak.db"))
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*1 sqlite3 connection(s) left open by this test*"])


def test_guard_passes_a_test_that_closes_its_connections(pytester):
    pytester.makeconftest(ROOT_CONFTEST.read_text())
    pytester.makepyfile("""
        import sqlite3

        def test_closes(tmp_path):
            conn = sqlite3.connect(str(tmp_path / "ok.db"))
            with conn:  # a transaction scope, not a lifetime scope
                conn.execute("CREATE TABLE t(x)")
            conn.close()
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
