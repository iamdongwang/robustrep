import pandas as pd
from typer.testing import CliRunner

from robustrep import cli
from robustrep.cli import app
from robustrep.schema import RESULT_COLUMNS
from robustrep.store import Store

runner = CliRunner()


def _seed(db, n=4):
    """Seed a store with ``n`` feedback rows for ratee "1", one rater each, all
    with a cached block timestamp and a profiled rater."""
    s = Store(db)
    rows = []
    for i in range(n):
        rows.append(dict(chain="base", block=1, tx_hash="0x", log_index=i, agent_id="1", client=f"0x{i:040x}",
                          feedback_index=0, value="80", value_decimals=0, tag1="q", tag2="", endpoint="",
                          feedback_uri="", feedback_hash=""))
    s.upsert_feedback(rows)
    s.upsert_block_ts([(1, 10)])
    for i in range(n):
        s.upsert_rater(f"0x{i:040x}", 10 + i * 100000, None)
    s.close()


class _FakeRpc:
    """Stand-in for RpcClient: fetch/score tests never touch the network."""

    def __init__(self, *args, **kwargs):
        pass


# --- score ------------------------------------------------------------------


def test_score_command_writes_csv(tmp_path):
    db, out = tmp_path / "t.db", tmp_path / "scores.csv"
    _seed(db)
    r = runner.invoke(app, ["score", "--db", str(db), "--out", str(out), "--bootstrap-n", "10"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(out)
    assert df.loc[0, "ratee"] == 1 and df.loc[0, "robust_score"] == 0.8
    assert list(df.columns) == RESULT_COLUMNS


def test_score_default_out_is_scores_csv(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["score", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "scores.csv").exists()


def test_score_no_records_exits_1(tmp_path):
    db = tmp_path / "empty.db"
    Store(db).close()
    r = runner.invoke(app, ["score", "--db", str(db)])
    assert r.exit_code == 1
    assert "no records" in r.output


def test_score_min_clusters_reaches_config(tmp_path, monkeypatch):
    db, out = tmp_path / "t.db", tmp_path / "scores.csv"
    _seed(db)
    seen = {}
    orig_score_fn = cli.score_fn

    def _capture(records, cfg, clusters=None):
        seen["min_clusters"] = cfg.min_clusters
        return orig_score_fn(records, cfg, clusters=clusters)

    monkeypatch.setattr(cli, "score_fn", _capture)
    r = runner.invoke(app, ["score", "--db", str(db), "--out", str(out), "--min-clusters", "7"])
    assert r.exit_code == 0, r.output
    assert seen["min_clusters"] == 7


def test_score_prints_summary_line(tmp_path):
    db, out = tmp_path / "t.db", tmp_path / "scores.csv"
    _seed(db)
    r = runner.invoke(app, ["score", "--db", str(db), "--out", str(out), "--bootstrap-n", "10"])
    assert r.exit_code == 0, r.output
    assert "scored" in r.output and "insufficient" in r.output and "sybil-flagged" in r.output
    assert "rater profile mode: unknown" in r.output


# --- fetch: dry-run -----------------------------------------------------------


def test_fetch_command_requires_network_flag_free_dry_run(tmp_path):
    r = runner.invoke(app, ["fetch", "--db", str(tmp_path / "t.db"), "--dry-run"])
    assert r.exit_code == 0 and "would sync" in r.output


# --- fetch: end-to-end with every step stubbed --------------------------------


def test_fetch_end_to_end_with_stubs(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=2)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 3)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 2)
    monkeypatch.setattr(cli.base, "owner_of", lambda *a, **k: "0xowner")
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 1)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 5)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "3" in r.output  # feedback rows added
    assert "2" in r.output  # block timestamps filled
    assert "1" in r.output  # raters profiled
    assert "5" in r.output  # evidence URIs classified
    assert "rater profile mode" in r.output


def test_fetch_skip_evidence_skips_classify_all(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    calls = []
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owner_of", lambda *a, **k: None)
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: calls.append(1))

    r = runner.invoke(app, ["fetch", "--db", str(db), "--skip-evidence"])
    assert r.exit_code == 0, r.output
    assert calls == []
    assert "evidence" in r.output


def test_fetch_skip_raters_skips_enrich_raters(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    calls = []
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owner_of", lambda *a, **k: None)
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--skip-raters"])
    assert r.exit_code == 0, r.output
    assert calls == []
    assert "raters" in r.output


def test_fetch_etherscan_key_overrides_env_and_is_not_echoed(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setenv("ETHERSCAN_API_KEY", "env-secret-key")
    seen_keys = []

    class _FakeEtherscanClient:
        def __init__(self, api_key):
            seen_keys.append(api_key)

    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli, "EtherscanClient", _FakeEtherscanClient)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owner_of", lambda *a, **k: None)
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--etherscan-key", "cli-override-key"])
    assert r.exit_code == 0, r.output
    assert seen_keys == ["cli-override-key"]
    assert "cli-override-key" not in r.output
    assert "env-secret-key" not in r.output


def test_fetch_rpc_url_and_confirmations_reach_config(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    seen = {}

    class _CapturingRpc:
        def __init__(self, urls, user_agent=None):
            seen["urls"] = list(urls)

    monkeypatch.setattr(cli, "RpcClient", _CapturingRpc)

    def _capture_sync(store, rpc, chunk=None, end_block=None, confirmations=None):
        seen["confirmations"] = confirmations
        return 0

    monkeypatch.setattr(cli.base, "sync_feedback", _capture_sync)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owner_of", lambda *a, **k: None)
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--rpc-url", "https://a.example",
                             "--rpc-url", "https://b.example", "--confirmations", "5"])
    assert r.exit_code == 0, r.output
    assert seen["urls"] == ["https://a.example", "https://b.example"]
    assert seen["confirmations"] == 5


def test_fetch_warns_when_block_timestamps_still_missing(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owner_of", lambda *a, **k: None)
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)
    monkeypatch.setattr(Store, "n_missing_block_ts", lambda self: 3)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "WARNING" in r.output


def test_fetch_partial_rater_failure_continues_and_exits_2(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    evidence_calls = []
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owner_of", lambda *a, **k: None)

    def _raise(*a, **k):
        raise RuntimeError("2 rater(s) failed: 0xdead, 0xbeef")

    monkeypatch.setattr(cli, "enrich_raters", _raise)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: evidence_calls.append(1))

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 2, r.output
    assert evidence_calls == [1]  # evidence step still ran after the partial rater failure
    assert "WARNING" in r.output


def test_fetch_keyboard_interrupt_propagates(tmp_path, monkeypatch):
    # fetch() only catches RuntimeError (the partial-rater-failure case), so a
    # KeyboardInterrupt from any step is never swallowed by our own code -- it
    # escapes fetch() unhandled. Typer's own top-level wrapper (not ours) then
    # converts *any* uncaught KeyboardInterrupt reaching a command's main() into
    # the conventional Unix SIGINT exit code 130, which is the "propagates
    # normally" behavior for a CLI (see typer.core: `except KeyboardInterrupt as
    # e: raise Exit(130)`) -- so this asserts our code never intercepts it
    # first, not that a bare KeyboardInterrupt crosses the CliRunner boundary.
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)

    def _interrupt(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli.base, "sync_feedback", _interrupt)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 130


# --- global options / help -----------------------------------------------------


def test_verbose_flag_sets_info_logging(tmp_path, monkeypatch):
    import logging

    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owner_of", lambda *a, **k: None)
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["--verbose", "fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert logging.getLogger().level == logging.INFO


def test_help_lists_fetch_and_score():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "fetch" in r.output and "score" in r.output
