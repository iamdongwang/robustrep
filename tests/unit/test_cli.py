import json
import logging

import pandas as pd
import pytest
from typer.testing import CliRunner

from robustrep import cli
from robustrep.cli import app
from robustrep.schema import RESULT_COLUMNS
from robustrep.sources.rpc import RpcError
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


def _seed_agents(db, agent_ids):
    """Seed a store with one feedback row per agent id in ``agent_ids``
    (distinct raters, all rater-profiled), for exercising ``_step_owners``
    with more than one agent."""
    s = Store(db)
    rows = []
    for i, agent_id in enumerate(agent_ids):
        rows.append(dict(chain="base", block=1, tx_hash="0x", log_index=i, agent_id=agent_id,
                          client=f"0x{i:040x}", feedback_index=0, value="80", value_decimals=0, tag1="q",
                          tag2="", endpoint="", feedback_uri="", feedback_hash=""))
    s.upsert_feedback(rows)
    s.upsert_block_ts([(1, 10)])
    for i in range(len(agent_ids)):
        s.upsert_rater(f"0x{i:040x}", 10 + i * 100000, None)
    s.close()


def _seed_unprofiled(db, n=1):
    """Seed a store with ``n`` feedback rows whose raters are NOT yet
    profiled (``distinct_clients()`` returns them) -- for exercising the
    Etherscan ETA branch of ``_step_raters``."""
    s = Store(db)
    rows = []
    for i in range(n):
        rows.append(dict(chain="base", block=1, tx_hash="0x", log_index=i, agent_id="1", client=f"0x{i:040x}",
                          feedback_index=0, value="80", value_decimals=0, tag1="q", tag2="", endpoint="",
                          feedback_uri="", feedback_hash=""))
    s.upsert_feedback(rows)
    s.upsert_block_ts([(1, 10)])
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
    _seed_agents(db, ["1", "2"])
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 3)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 2)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: "0xowner" for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 1)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 5)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "feedback rows added: 3" in r.output
    assert "block timestamps filled: 2" in r.output
    assert "agent owners resolved: 2/2" in r.output
    assert "raters profiled: 1" in r.output
    assert "evidence URIs classified: 5" in r.output
    assert "rater profile mode" in r.output


def test_fetch_evidence_line_reports_lookup_caps(tmp_path, monkeypatch):
    # H3 (security review): the caps that shaped every cached evidence level
    # this run produced must be visible in the fetch step's own summary line,
    # not just buried in a later report's provenance.
    from robustrep.evidence import MAX_TX_LOOKUPS_PER_URI
    from robustrep.sources.evidence_batch import DEFAULT_MAX_TOTAL_LOOKUPS

    db = tmp_path / "t.db"
    _seed_agents(db, ["1", "2"])
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 3)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 2)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: "0xowner" for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 1)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 5)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert f"max_lookups_per_uri={MAX_TX_LOOKUPS_PER_URI}" in r.output
    assert f"max_total_lookups={DEFAULT_MAX_TOTAL_LOOKUPS}" in r.output


def test_fetch_skip_evidence_skips_classify_all(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    calls = []
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
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
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
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
    seen = []

    class _FakeEtherscanClient:
        source = "etherscan"

        def __init__(self, api_key):
            pass

    def _fake_default_client(source, key, rps=None, retries=None):
        seen.append((source, key))
        return _FakeEtherscanClient(key)

    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli, "default_client", _fake_default_client)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--etherscan-key", "cli-override-key"])
    assert r.exit_code == 0, r.output
    assert seen == [("auto", "cli-override-key")]
    assert "cli-override-key" not in r.output
    assert "env-secret-key" not in r.output


# --- --profile-source propagation -----------------------------------------------


def _patch_fetch_steps_except_raters(monkeypatch, capture):
    """Stub every fetch step except raters, and capture default_client's
    (source, key) args -- shared setup for the --profile-source tests."""
    class _FakeClient:
        source = "etherscan"

        def __init__(self, key):
            pass

    def _fake_default_client(source, key, rps=None, retries=None):
        capture.append((source, key))
        return _FakeClient(key) if source != "none" else None

    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli, "default_client", _fake_default_client)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)


def test_fetch_profile_source_defaults_to_auto(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    seen = []
    _patch_fetch_steps_except_raters(monkeypatch, seen)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert seen == [("auto", None)]


def test_fetch_profile_source_blockscout_flag_reaches_default_client(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setenv("ETHERSCAN_API_KEY", "some-key")  # even with a key set...
    seen = []
    _patch_fetch_steps_except_raters(monkeypatch, seen)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--profile-source", "blockscout"])
    assert r.exit_code == 0, r.output
    assert seen == [("blockscout", "some-key")]  # ...source is exactly what was requested


def test_fetch_profile_source_none_reaches_default_client_and_skips_client(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    seen = []
    _patch_fetch_steps_except_raters(monkeypatch, seen)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--profile-source", "none"])
    assert r.exit_code == 0, r.output
    assert seen == [("none", None)]
    assert "fallback" in r.output


def test_fetch_profile_source_etherscan_without_key_exits_1_with_error(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--profile-source", "etherscan"])
    assert r.exit_code == 1, r.output
    assert "ERROR" in r.output
    assert "etherscan" in r.output.lower()


def test_fetch_profile_source_rejects_invalid_choice(tmp_path):
    r = runner.invoke(app, ["fetch", "--db", str(tmp_path / "t.db"), "--profile-source", "bogus", "--dry-run"])
    assert r.exit_code != 0


def test_step_raters_profile_source_blockscout_builds_real_client_without_network(tmp_path):
    # No feedback rows -> distinct_clients() is empty -> enrich_raters short-
    # circuits before any HTTP call, so this exercises the real
    # default_client("blockscout", ...) -> EtherscanClient.blockscout() path
    # (no monkeypatching) without ever touching the network.
    db = tmp_path / "t.db"
    with Store(db) as store:
        summary = cli._step_raters(store, None, profile_source="blockscout")
    assert "raters profiled: 0" in summary
    assert "rater profile mode: unknown" in summary  # nothing to do -> mode never set


def test_step_raters_profile_source_string_default_is_auto():
    import inspect

    sig = inspect.signature(cli._step_raters)
    assert sig.parameters["profile_source"].default == cli.ProfileSource.auto


# --- --profile-rps/--profile-retries propagation ---------------------------------


class _RpsCapturingClient:
    source = "etherscan"

    def __init__(self, key, rps=None, retries=None):
        self.gap = 1.0 / (rps if rps is not None else 4.0)
        self.retries = retries


def test_fetch_profile_rps_and_retries_reach_default_client(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    seen = []

    def _fake_default_client(source, key, rps=None, retries=None):
        seen.append((source, key, rps, retries))
        return _RpsCapturingClient(key, rps=rps, retries=retries)

    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli, "default_client", _fake_default_client)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--profile-rps", "0.5", "--profile-retries", "9"])
    assert r.exit_code == 0, r.output
    assert seen == [("auto", None, 0.5, 9)]


def test_fetch_profile_rps_and_retries_default_to_none(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    seen = []

    def _fake_default_client(source, key, rps=None, retries=None):
        seen.append((source, key, rps, retries))
        return _RpsCapturingClient(key, rps=rps, retries=retries)

    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli, "default_client", _fake_default_client)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert seen == [("auto", None, None, None)]


def test_fetch_profile_rps_rejects_below_minimum(tmp_path):
    r = runner.invoke(app, ["fetch", "--db", str(tmp_path / "t.db"), "--profile-rps", "0.05", "--dry-run"])
    assert r.exit_code != 0


def test_fetch_profile_retries_rejects_below_minimum(tmp_path):
    r = runner.invoke(app, ["fetch", "--db", str(tmp_path / "t.db"), "--profile-retries", "0", "--dry-run"])
    assert r.exit_code != 0


def test_fetch_eta_uses_effective_rps_from_profile_rps_flag(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed_unprofiled(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 1)
    monkeypatch.setattr(
        cli, "default_client",
        lambda source, key, rps=None, retries=None: _RpsCapturingClient(key, rps=rps, retries=retries))

    seen_rps = []
    orig_estimate = cli.estimate_seconds

    def _capture_estimate(n, rps):
        seen_rps.append(rps)
        return orig_estimate(n, rps)

    monkeypatch.setattr(cli, "estimate_seconds", _capture_estimate)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--profile-rps", "0.25"])
    assert r.exit_code == 0, r.output
    assert seen_rps == [0.25]


def test_fetch_eta_uses_client_default_rps_when_flag_omitted(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed_unprofiled(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 1)
    monkeypatch.setattr(
        cli, "default_client",
        lambda source, key, rps=None, retries=None: _RpsCapturingClient(key, rps=rps, retries=retries))

    seen_rps = []
    orig_estimate = cli.estimate_seconds

    def _capture_estimate(n, rps):
        seen_rps.append(rps)
        return orig_estimate(n, rps)

    monkeypatch.setattr(cli, "estimate_seconds", _capture_estimate)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    # _RpsCapturingClient's own default (matching EtherscanClient's) is 4.0.
    assert seen_rps == [4.0]


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
        seen["chunk"] = chunk
        seen["end_block"] = end_block
        return 0

    monkeypatch.setattr(cli.base, "sync_feedback", _capture_sync)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--rpc-url", "https://a.example",
                             "--rpc-url", "https://b.example", "--confirmations", "5"])
    assert r.exit_code == 0, r.output
    assert seen["urls"] == ["https://a.example", "https://b.example"]
    assert seen["confirmations"] == 5


def test_fetch_to_block_and_chunk_reach_sync_feedback(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    seen = {}

    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)

    def _capture_sync(store, rpc, chunk=None, end_block=None, confirmations=None):
        seen["chunk"] = chunk
        seen["end_block"] = end_block
        return 0

    monkeypatch.setattr(cli.base, "sync_feedback", _capture_sync)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--to-block", "123", "--chunk", "500"])
    assert r.exit_code == 0, r.output
    assert seen["end_block"] == 123
    assert seen["chunk"] == 500


def test_fetch_warns_when_block_timestamps_still_missing(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
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
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})

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


def test_verbose_flag_sets_debug_logging(tmp_path, monkeypatch):
    # DEBUG, not INFO: _rpc_guard's "re-run with --verbose for detail" message
    # promises that --verbose actually surfaces the DEBUG-level detail rpc.py
    # and _rpc_guard log on an RpcError -- INFO would not.
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["--verbose", "fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert logging.getLogger("robustrep").level == logging.DEBUG


def test_help_lists_fetch_and_score():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "fetch" in r.output and "score" in r.output


def test_repeated_invoke_never_binds_handler_to_a_closed_stream(tmp_path, capsys):
    # Regression test: main() used to call logging.basicConfig(force=True),
    # which rebuilds the root logger's handlers on every invocation, binding a
    # fresh StreamHandler to whatever sys.stderr CliRunner had redirected to
    # *at that call*. That redirected stream is closed once invoke() returns,
    # so a later direct log call reusing the (now-closed) handler used to
    # raise "I/O operation on closed file" (caught internally by logging's
    # Handler.handleError, which prints a "--- Logging error ---" diagnostic
    # to stderr instead of propagating -- so the regression must be checked
    # via captured stderr, not just "did calling .warning() raise").
    db = tmp_path / "t.db"
    r1 = runner.invoke(app, ["--verbose", "fetch", "--db", str(db), "--dry-run"])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["--verbose", "fetch", "--db", str(db), "--dry-run"])
    assert r2.exit_code == 0, r2.output

    capsys.readouterr()  # discard anything captured so far
    logging.getLogger("robustrep").warning("x")  # must not raise / print a logging error
    captured = capsys.readouterr()
    assert "Logging error" not in captured.err
    assert "closed file" not in captured.err


# --- RPC error handling (redacted, exit 3) -------------------------------------


def test_fetch_rpc_error_from_sync_step_exits_3_with_clean_message(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)

    def _raise(*a, **k):
        # Already-redacted, as rpc.py itself guarantees (see test_rpc.py) --
        # this test only proves the CLI's own plumbing, not rpc.py's redaction.
        raise RpcError("gave up after 5 attempts: ConnectionError contacting https://rpc.example.com")

    monkeypatch.setattr(cli.base, "sync_feedback", _raise)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 3, r.output
    assert "ERROR: RPC failed" in r.output
    assert "re-run with --verbose" in r.output
    assert "SUPER_SECRET" not in r.output
    assert "Traceback" not in r.output


def test_fetch_rpc_error_from_evidence_step_exits_3(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)

    def _raise(*a, **k):
        raise RpcError("gave up after 5 attempts: ConnectionError contacting https://rpc.example.com")

    monkeypatch.setattr(cli, "classify_all", _raise)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 3, r.output
    assert "ERROR: RPC failed" in r.output


def test_fetch_closes_store_on_rpc_error(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)

    def _raise(*a, **k):
        raise RpcError("gave up: ConnectionError contacting https://rpc.example.com")

    monkeypatch.setattr(cli.base, "sync_feedback", _raise)

    closed = []
    real_store_cls = cli.Store

    class _TrackingStore(real_store_cls):
        def close(self):
            closed.append(1)
            super().close()

    monkeypatch.setattr(cli, "Store", _TrackingStore)
    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 3, r.output
    assert closed == [1]


# --- _step_owners: always records, progress log, --skip-owners ----------------


def test_step_owners_records_unresolved_and_skips_on_rerun(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed_agents(db, ["1"])
    calls = []
    monkeypatch.setattr(cli.base, "owners_of",
                         lambda rpc, agent_ids, **k: calls.append(1) or {a: None for a in agent_ids})

    with Store(db) as store:
        summary = cli._step_owners(store, _FakeRpc())
        assert summary == "agent owners resolved: 0/1"
        assert store.agent_owner("1") == ""  # recorded, even though unresolved

        # a second run must not re-query an agent already recorded (with or
        # without an owner) -- distinct_agents() excludes it now.
        summary2 = cli._step_owners(store, _FakeRpc())
        assert summary2 == "agent owners resolved: 0/0"
    assert len(calls) == 1


def test_step_owners_progress_logged_every_n(tmp_path, monkeypatch, caplog):
    db = tmp_path / "t.db"
    _seed_agents(db, ["1", "2", "3", "4"])
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})

    with Store(db) as store:
        with caplog.at_level(logging.INFO, logger=cli.__name__):
            summary = cli._step_owners(store, _FakeRpc(), log_every=2)
    assert summary == "agent owners resolved: 0/4"
    # caplog's records already have `.message` fully formatted (its handler
    # calls record.getMessage() as it captures each record).
    progress = [r.message for r in caplog.records if "agent owners resolved" in r.message]
    assert progress == ["fetch: agent owners resolved 2/4", "fetch: agent owners resolved 4/4"]


def test_fetch_skip_owners_skips_owners_of(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    calls = []
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: calls.append(1))
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--skip-owners"])
    assert r.exit_code == 0, r.output
    assert calls == []
    assert "agent owner resolution skipped" in r.output


def test_fetch_owner_batch_controls_batch_call_count(tmp_path, monkeypatch):
    # base.owners_of is deliberately NOT mocked here -- this exercises the real
    # chunking/batching path end to end, through a fake RpcClient whose
    # .batch() records how many calls it received and how big each was.
    db = tmp_path / "t.db"
    _seed_agents(db, [str(i) for i in range(5)])
    created = []

    class _BatchTrackingRpc:
        def __init__(self, *a, **k):
            self.batch_calls = 0
            self.batch_sizes = []
            created.append(self)

        def batch(self, calls):
            self.batch_calls += 1
            self.batch_sizes.append(len(calls))
            return ["0x" + "00" * 32 for _ in calls]  # zero address -> unresolved

    monkeypatch.setattr(cli, "RpcClient", _BatchTrackingRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--owner-batch", "2"])
    assert r.exit_code == 0, r.output
    assert "agent owners resolved: 0/5" in r.output  # summary line format unchanged
    assert created[0].batch_calls == 3  # ceil(5/2)
    assert created[0].batch_sizes == [2, 2, 1]


def test_fetch_rejects_non_positive_owner_batch(tmp_path):
    r = runner.invoke(app, ["fetch", "--db", str(tmp_path / "t.db"), "--owner-batch", "0", "--dry-run"])
    assert r.exit_code == 2


# --- fetch: --evidence-workers ------------------------------------------------


def test_fetch_evidence_workers_reaches_classify_all(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)
    seen = {}

    def _capture(store, tx_parties=None, workers=8, **kwargs):
        seen["workers"] = workers
        return 0

    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", _capture)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--evidence-workers", "3"])
    assert r.exit_code == 0, r.output
    assert seen["workers"] == 3


def test_fetch_evidence_workers_defaults_to_eight(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)
    seen = {}

    def _capture(store, tx_parties=None, workers=8, **kwargs):
        seen["workers"] = workers
        return 0

    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", _capture)

    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert seen["workers"] == 8


def test_fetch_rejects_non_positive_evidence_workers(tmp_path):
    r = runner.invoke(app, ["fetch", "--db", str(tmp_path / "t.db"), "--evidence-workers", "0", "--dry-run"])
    assert r.exit_code == 2


# --- ValueError handling (Config / cluster_raters / validate_records) ---------


def test_fetch_config_value_error_exits_1(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)

    def _raise_cfg(*a, **k):
        raise ValueError("bad config")

    monkeypatch.setattr(cli, "Config", _raise_cfg)
    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 1, r.output
    assert "ERROR: bad config" in r.output


def test_score_config_value_error_exits_1(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)

    def _raise_cfg(*a, **k):
        raise ValueError("bad cfg")

    monkeypatch.setattr(cli, "Config", _raise_cfg)
    r = runner.invoke(app, ["score", "--db", str(db)])
    assert r.exit_code == 1, r.output
    assert "ERROR: bad cfg" in r.output


def test_score_cluster_raters_value_error_exits_1(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)

    def _raise(*a, **k):
        raise ValueError("too many candidate pairs")

    monkeypatch.setattr(cli, "cluster_raters_with_stats", _raise)
    r = runner.invoke(app, ["score", "--db", str(db)])
    assert r.exit_code == 1, r.output
    assert "ERROR: too many candidate pairs" in r.output


# --- click-level option validation (exit 2) ------------------------------------


def test_fetch_rejects_non_positive_chunk(tmp_path):
    r = runner.invoke(app, ["fetch", "--db", str(tmp_path / "t.db"), "--chunk", "0", "--dry-run"])
    assert r.exit_code == 2


def test_fetch_rejects_negative_confirmations(tmp_path):
    r = runner.invoke(app, ["fetch", "--db", str(tmp_path / "t.db"), "--confirmations", "-1", "--dry-run"])
    assert r.exit_code == 2


def test_score_rejects_negative_bootstrap_n(tmp_path):
    r = runner.invoke(app, ["score", "--db", str(tmp_path / "t.db"), "--bootstrap-n", "-1"])
    assert r.exit_code == 2


def test_score_rejects_non_positive_min_clusters(tmp_path):
    r = runner.invoke(app, ["score", "--db", str(tmp_path / "t.db"), "--min-clusters", "0"])
    assert r.exit_code == 2


def test_fetch_rejects_blank_etherscan_key(tmp_path):
    r = runner.invoke(app, ["fetch", "--db", str(tmp_path / "t.db"), "--etherscan-key", "", "--dry-run"])
    assert r.exit_code == 2


# --- dry-run must not create the DB file ---------------------------------------


def test_fetch_dry_run_does_not_create_db_file(tmp_path):
    db = tmp_path / "new.db"
    assert not db.exists()
    r = runner.invoke(app, ["fetch", "--db", str(db), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert not db.exists()


def test_fetch_dry_run_uses_existing_checkpoint(tmp_path):
    db = tmp_path / "t.db"
    with Store(db) as store:
        store.set_sync("last_block", "1000")
    r = runner.invoke(app, ["fetch", "--db", str(db), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "would sync from block 1001" in r.output


# --- score: mkdir parent for --out ----------------------------------------------


def test_score_creates_out_parent_dir(tmp_path):
    db = tmp_path / "t.db"
    _seed(db)
    out = tmp_path / "nested" / "dir" / "scores.csv"
    r = runner.invoke(app, ["score", "--db", str(db), "--out", str(out), "--bootstrap-n", "10"])
    assert r.exit_code == 0, r.output
    assert out.exists()


# --- Store used as a context manager (closed even on early exit) --------------


def test_fetch_closes_store_on_success(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    closed = []
    real_store_cls = cli.Store

    class _TrackingStore(real_store_cls):
        def close(self):
            closed.append(1)
            super().close()

    monkeypatch.setattr(cli, "Store", _TrackingStore)
    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert closed == [1]


def test_score_closes_store_on_success(tmp_path):
    db, out = tmp_path / "t.db", tmp_path / "scores.csv"
    _seed(db)

    closed = []
    real_store_cls = cli.Store

    class _TrackingStore(real_store_cls):
        def close(self):
            closed.append(1)
            super().close()

    orig = cli.Store
    cli.Store = _TrackingStore
    try:
        r = runner.invoke(app, ["score", "--db", str(db), "--out", str(out), "--bootstrap-n", "10"])
    finally:
        cli.Store = orig
    assert r.exit_code == 0, r.output
    assert closed == [1]


# --- _step_raters: ETA branch, single distinct_clients() call, DEFAULT_RPS ----


def test_fetch_eta_branch_prints_estimate(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed_unprofiled(db, n=1)
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)

    class _FakeEtherscanClient:
        source = "etherscan"

        def __init__(self, api_key, rps=None, retries=None):
            self.gap = 1.0 / (rps if rps is not None else cli.DEFAULT_RPS)

    monkeypatch.setattr(
        cli, "default_client",
        lambda source, key, rps=None, retries=None: _FakeEtherscanClient(key, rps=rps, retries=retries))
    monkeypatch.setattr(cli, "enrich_raters", lambda *a, **k: 1)

    r = runner.invoke(app, ["fetch", "--db", str(db), "--etherscan-key", "k"])
    assert r.exit_code == 0, r.output
    assert "profiling 1 rater(s) via Etherscan" in r.output
    assert "ETA" in r.output


def test_step_raters_with_real_enrich_raters_queries_distinct_clients_once(tmp_path, monkeypatch):
    # Uses the REAL enrich_raters (not a stub) to prove _step_raters's
    # `addresses=targets` actually reaches it and is honored: enrich_raters
    # only calls store.distinct_clients() itself when NOT given `addresses`
    # (see robustrep.sources.rater_profile.enrich_raters), so if the CLI's
    # own single call plus enrich_raters's were both happening, this would
    # count 2, not 1.
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    db = tmp_path / "t.db"
    _seed_unprofiled(db, n=1)

    with Store(db) as store:
        calls = []
        orig = store.distinct_clients

        def _counting():
            calls.append(1)
            return orig()

        store.distinct_clients = _counting
        # profile_source="none" -> offline fallback, no HTTP client at all.
        summary = cli._step_raters(store, None, profile_source="none")
    assert len(calls) == 1
    assert "raters profiled: 1" in summary


def test_default_rps_exported_and_used_by_cli():
    from robustrep.sources.rater_profile import DEFAULT_RPS

    assert DEFAULT_RPS == 4.0
    assert cli.DEFAULT_RPS == 4.0


# --- help text: etherscan key recommends the env var ---------------------------


def test_fetch_help_recommends_env_var_for_etherscan_key():
    r = runner.invoke(app, ["fetch", "--help"])
    assert r.exit_code == 0
    assert "ETHERSCAN_API_KEY" in r.output


# --- report ---------------------------------------------------------------------


def test_report_writes_figures_markdown_and_json_under_block_and_latest(tmp_path):
    db = tmp_path / "t.db"
    _seed(db)
    out_dir = tmp_path / "reports"
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(out_dir), "--bootstrap-n", "5"])
    assert r.exit_code == 0, r.output
    assert "report written to" in r.output

    with Store(db) as store:
        last_block = store.get_sync("last_block")
    block = int(last_block) if last_block is not None else 0

    for target in (out_dir / str(block), out_dir / "latest"):
        assert target.is_dir()
        pngs = sorted(target.glob("*.png"))
        assert len(pngs) == 5
        for p in pngs:
            assert p.stat().st_size > 500
        md_path = target / "report.md"
        json_path = target / "scores.json"
        assert md_path.exists() and json_path.exists()
        md = md_path.read_text()
        assert f"block {block}" in md and "Limitations" in md and "Provenance" in md
        data = json.loads(json_path.read_text())
        assert data["block"] == block and data["schema_version"] == 1


def test_report_no_records_exits_1(tmp_path):
    db = tmp_path / "empty.db"
    Store(db).close()
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(tmp_path / "reports")])
    assert r.exit_code == 1
    assert "no records" in r.output


def test_report_config_value_error_exits_1(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)

    def _raise_cfg(*a, **k):
        raise ValueError("bad cfg")

    monkeypatch.setattr(cli, "Config", _raise_cfg)
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(tmp_path / "reports")])
    assert r.exit_code == 1, r.output
    assert "ERROR: bad cfg" in r.output


def test_report_cluster_raters_value_error_exits_1(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)

    def _raise(*a, **k):
        raise ValueError("too many candidate pairs")

    monkeypatch.setattr(cli, "cluster_raters_with_stats", _raise)
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(tmp_path / "reports")])
    assert r.exit_code == 1, r.output
    assert "ERROR: too many candidate pairs" in r.output


def test_report_figure_headings_are_human_titles_not_filenames(tmp_path):
    db = tmp_path / "t.db"
    _seed(db)
    out_dir = tmp_path / "reports"
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(out_dir), "--bootstrap-n", "5"])
    assert r.exit_code == 0, r.output
    md = (out_dir / "latest" / "report.md").read_text()
    assert "### Fig 4. Largest rater clusters" in md
    assert "### fig4_sybil_clusters.png" not in md
    assert "fig4_sybil_clusters.png" in md  # still referenced as the image path


def test_report_provenance_includes_config_in_markdown_and_json(tmp_path):
    db = tmp_path / "t.db"
    _seed(db)
    out_dir = tmp_path / "reports"
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(out_dir), "--bootstrap-n", "5"])
    assert r.exit_code == 0, r.output

    md = (out_dir / "latest" / "report.md").read_text()
    for phrase in ("bootstrap_n", "bootstrap_seed", "min_clusters", "evidence_weights",
                  "sybil_jaccard", "sybil_window_s", "sybil_max_group", "sybil_max_pairs",
                  "sybil_max_pairs_per_ratee", "norm_fit_share", "norm_rule_counts"):
        assert phrase in md, phrase

    data = json.loads((out_dir / "latest" / "scores.json").read_text())
    config = data["config"]
    assert config["bootstrap_n"] == 5
    # norm_fit_share decides which normalization rule each (tag, scale) group is
    # scored under, so a report is not reproducible without it on the record.
    assert config["norm_fit_share"] == cli.Config().norm_fit_share
    # norm_rule_counts makes a rung flip between runs visible in a report diff.
    counts = config["norm_rule_counts"]
    assert isinstance(counts, dict) and counts and sum(counts.values()) > 0
    for key in ("bootstrap_seed", "min_clusters", "evidence_weights", "sybil_jaccard",
                "sybil_window_s", "sybil_max_group", "sybil_max_pairs", "sybil_max_pairs_per_ratee",
                "sybil_flag_share", "norm_fit_share", "norm_rule_counts"):
        assert key in config
    assert config["sybil_max_pairs_per_ratee"] == cli.Config().sybil_max_pairs_per_ratee


def test_report_provenance_includes_evidence_lookup_caps(tmp_path):
    # H3 (security review): evidence_max_lookups_per_uri/evidence_max_total_lookups
    # bound which cached evidence levels might be false negatives (see
    # evidence_batch's "lookup-budget:N" note) -- publish them next to
    # evidence_level_shares so a report is auditable against the caps that
    # actually produced it.
    from robustrep.evidence import MAX_TX_LOOKUPS_PER_URI
    from robustrep.sources.evidence_batch import DEFAULT_MAX_TOTAL_LOOKUPS

    db = tmp_path / "t.db"
    _seed(db)
    out_dir = tmp_path / "reports"
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(out_dir), "--bootstrap-n", "5"])
    assert r.exit_code == 0, r.output

    md = (out_dir / "latest" / "report.md").read_text()
    for phrase in ("evidence_max_lookups_per_uri", "evidence_max_total_lookups"):
        assert phrase in md, phrase

    data = json.loads((out_dir / "latest" / "scores.json").read_text())
    config = data["config"]
    assert config["evidence_max_lookups_per_uri"] == MAX_TX_LOOKUPS_PER_URI
    assert config["evidence_max_total_lookups"] == DEFAULT_MAX_TOTAL_LOOKUPS


def test_report_provenance_includes_evidence_lookup_starved_uris(tmp_path):
    # evidence_lookup_starved_uris (Store.n_lookup_budget_starved) must
    # reflect this run's own store snapshot, taken while the store is still
    # open -- not a stale default -- so a report is auditable against how
    # many cached evidence levels might still be under-checked (H3).
    db = tmp_path / "t.db"
    _seed(db, n=4)
    s = Store(db)
    s.upsert_evidence("https://starved1", 2, "lookup-budget:1")
    s.upsert_evidence("https://starved2", 2, "lookup-budget:2")
    s.upsert_evidence("https://done", 3, "")
    s.close()

    out_dir = tmp_path / "reports"
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(out_dir), "--bootstrap-n", "5"])
    assert r.exit_code == 0, r.output

    md = (out_dir / "latest" / "report.md").read_text()
    assert "evidence_lookup_starved_uris" in md

    data = json.loads((out_dir / "latest" / "scores.json").read_text())
    assert data["config"]["evidence_lookup_starved_uris"] == 2


def test_provenance_n_lookup_budget_starved_defaults_to_zero_without_a_store():
    # cli._provenance is also called directly (e.g. from tests) on an
    # in-memory records frame with no store to count from -- it must not
    # require one.
    from robustrep import Config
    from robustrep.schema import validate_records

    records = validate_records(pd.DataFrame([dict(
        rater="r", ratee="A", value=50, scale="d0", tag="q", ts=0,
        evidence_uri=None, source="test", evidence_level=0)]))
    prov = cli._provenance("onchain", Config(), records)
    assert prov["config"]["evidence_lookup_starved_uris"] == 0


def test_report_provenance_includes_evidence_level_shares(tmp_path):
    """`_seed` writes feedback rows with no evidence_uri -- every row classifies
    to evidence level 0 -- so the shares should read 100% level 0, and land in
    both the CLI provenance dict's `config` and the same-named key of
    scores.json's `config` block (they're the same dict)."""
    db = tmp_path / "t.db"
    _seed(db, n=4)
    out_dir = tmp_path / "reports"
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(out_dir), "--bootstrap-n", "5"])
    assert r.exit_code == 0, r.output

    data = json.loads((out_dir / "latest" / "scores.json").read_text())
    shares = data["config"]["evidence_level_shares"]
    assert shares == {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0}


def test_provenance_evidence_level_shares_from_records(tmp_path):
    """`cli._provenance` computes `evidence_level_shares` directly from the
    `records` frame it's given (non-revoked rows only), independent of the
    `report` command's wiring."""
    from robustrep import Config
    from robustrep.schema import validate_records

    rows = []
    for lvl, n in ((0, 2), (1, 6), (2, 1), (3, 1)):
        for i in range(n):
            rows.append(dict(rater=f"r{lvl}_{i}", ratee="A", value=50, scale="d0", tag="q", ts=0,
                             evidence_uri=None, source="test", evidence_level=lvl))
    records = validate_records(pd.DataFrame(rows))
    prov = cli._provenance("onchain", Config(), records)
    shares = prov["config"]["evidence_level_shares"]
    assert shares[0] == pytest.approx(0.2)
    assert shares[1] == pytest.approx(0.6)
    assert shares[2] == pytest.approx(0.1)
    assert shares[3] == pytest.approx(0.1)


def test_report_latest_is_atomically_replaced_and_stale_files_removed(tmp_path):
    db = tmp_path / "t.db"
    _seed(db)
    out_dir = tmp_path / "reports"
    r1 = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(out_dir), "--bootstrap-n", "5"])
    assert r1.exit_code == 0, r1.output

    stale = out_dir / "latest" / "stale_from_a_previous_run.txt"
    stale.write_text("old")
    assert stale.exists()

    r2 = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(out_dir), "--bootstrap-n", "5"])
    assert r2.exit_code == 0, r2.output

    assert not stale.exists()
    assert (out_dir / "latest" / "report.md").exists()
    assert (out_dir / "latest" / "scores.json").exists()
    assert len(list((out_dir / "latest").glob("*.png"))) == 5


def test_provenance_norm_rule_counts_follow_the_run_config():
    """`norm_rule_counts` must describe the run that was actually scored, not a
    default-config re-run: the same frame is `rank` at norm_fit_share 0.9 and
    `percent` at 0.75."""
    from robustrep import Config, score
    from robustrep.schema import validate_records

    honest = [10, 50, 90, 100, 0, 75, 25, 60]
    rows = [dict(rater=f"r{i}", ratee="A", value=v, scale="d0", tag="q", ts=0,
                 evidence_uri=None, source="test") for i, v in enumerate(honest)]
    rows += [dict(rater="x1", ratee="Z", value=2**127 - 1, scale="d0", tag="q", ts=0,
                  evidence_uri=None, source="test"),
             dict(rater="x2", ratee="Z", value=-(2**127), scale="d0", tag="q", ts=0,
                  evidence_uri=None, source="test")]
    records = validate_records(pd.DataFrame(rows))
    for fit_share, rule in ((0.9, "rank"), (0.75, "percent")):
        cfg = Config(bootstrap_n=0, norm_fit_share=fit_share)
        prov = cli._provenance("onchain", cfg, records, score(records, cfg))
        assert prov["config"]["norm_fit_share"] == fit_share
        assert prov["config"]["norm_rule_counts"] == {rule: 10}


# --- sybil pair-budget options and cluster stats (H2) --------------------------


_SYBIL_OPTS = ["--sybil-max-group", "--sybil-max-pairs", "--sybil-max-pairs-per-ratee"]


@pytest.mark.parametrize("command", ["score", "report"])
def test_help_lists_sybil_budget_options(command):
    # Inspect the click command's parameters rather than the rendered help:
    # typer's rich help wraps/colours option names differently on a CI
    # (non-TTY) console, splitting long names with escape codes.
    import typer.main
    cmd = typer.main.get_command(app).commands[command]
    declared = {opt for p in cmd.params for opt in p.opts}
    for opt in _SYBIL_OPTS:
        assert opt in declared, opt
    per_ratee = next(p for p in cmd.params if "--sybil-max-pairs-per-ratee" in p.opts)
    assert per_ratee.default == cli.Config().sybil_max_pairs_per_ratee
    r = runner.invoke(app, [command, "--help"])
    assert r.exit_code == 0, r.output


def _capture_cfg(monkeypatch):
    """Wrap ``cli.score_fn`` so a test can read the Config and the scored frame
    the command actually used."""
    seen = {}
    orig = cli.score_fn

    def _capture(records, cfg, clusters=None):
        result = orig(records, cfg, clusters=clusters)
        seen["cfg"], seen["result"] = cfg, result
        return result

    monkeypatch.setattr(cli, "score_fn", _capture)
    return seen


def test_score_sybil_budget_options_reach_config(tmp_path, monkeypatch):
    db, out = tmp_path / "t.db", tmp_path / "scores.csv"
    _seed(db)
    seen = _capture_cfg(monkeypatch)
    r = runner.invoke(app, ["score", "--db", str(db), "--out", str(out), "--bootstrap-n", "5",
                            "--sybil-max-group", "5", "--sybil-max-pairs", "10",
                            "--sybil-max-pairs-per-ratee", "5"])
    assert r.exit_code == 0, r.output
    cfg = seen["cfg"]
    assert (cfg.sybil_max_group, cfg.sybil_max_pairs, cfg.sybil_max_pairs_per_ratee) == (5, 10, 5)


def test_score_attaches_cluster_stats_to_scores_frame(tmp_path, monkeypatch):
    db, out = tmp_path / "t.db", tmp_path / "scores.csv"
    _seed(db)
    seen = _capture_cfg(monkeypatch)
    r = runner.invoke(app, ["score", "--db", str(db), "--out", str(out), "--bootstrap-n", "5"])
    assert r.exit_code == 0, r.output
    stats = seen["result"].attrs["cluster_stats"]
    assert set(stats) == {"pairs_tested", "pairs_examined", "ratees_skipped_size",
                          "ratees_skipped_budget", "truncated"}
    assert stats["truncated"] is False


def test_report_sybil_budget_options_reach_config_and_stats(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)
    seen = _capture_cfg(monkeypatch)
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(tmp_path / "reports"),
                            "--bootstrap-n", "5", "--sybil-max-group", "3",
                            "--sybil-max-pairs", "7", "--sybil-max-pairs-per-ratee", "2"])
    assert r.exit_code == 0, r.output
    cfg = seen["cfg"]
    assert (cfg.sybil_max_group, cfg.sybil_max_pairs, cfg.sybil_max_pairs_per_ratee) == (3, 7, 2)
    assert "cluster_stats" in seen["result"].attrs


def test_report_sensitivity_value_error_exits_1_not_traceback(tmp_path, monkeypatch):
    """L2: a ValueError out of `sensitivity_table` must exit 1 with the same
    ERROR line as any other scoring rejection, never a raw traceback."""
    db = tmp_path / "t.db"
    _seed(db)

    def _raise(*a, **k):
        raise ValueError("sensitivity blew up")

    monkeypatch.setattr(cli, "sensitivity_table", _raise)
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(tmp_path / "reports"),
                            "--bootstrap-n", "5"])
    assert r.exit_code == 1, r.output
    assert "ERROR: sensitivity blew up" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


# --- cluster stats in provenance / scores.json, and the score summary note ----


def test_report_scores_json_carries_cluster_stats(tmp_path):
    db = tmp_path / "t.db"
    _seed(db)
    out_dir = tmp_path / "reports"
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(out_dir), "--bootstrap-n", "5"])
    assert r.exit_code == 0, r.output
    data = json.loads((out_dir / "latest" / "scores.json").read_text())
    stats = data["cluster_stats"]
    assert set(stats) == {"pairs_tested", "pairs_examined", "ratees_skipped_size",
                          "ratees_skipped_budget", "truncated"}
    assert stats["truncated"] is False


def test_provenance_cluster_stats_empty_without_a_scored_frame(tmp_path):
    from robustrep import Config

    db = tmp_path / "t.db"
    _seed(db)
    with Store(db) as store:
        records = store.load_records()
    assert cli._provenance("onchain", Config(), records)["cluster_stats"] == {}


def test_score_summary_notes_budget_limited_clustering(tmp_path):
    db, out = tmp_path / "t.db", tmp_path / "scores.csv"
    _seed(db)
    r = runner.invoke(app, ["score", "--db", str(db), "--out", str(out), "--bootstrap-n", "5",
                            "--sybil-max-pairs", "1"])
    assert r.exit_code == 0, r.output
    assert "clustering budget-limited" in r.output


def test_score_summary_has_no_budget_note_on_a_healthy_run(tmp_path):
    db, out = tmp_path / "t.db", tmp_path / "scores.csv"
    _seed(db)
    r = runner.invoke(app, ["score", "--db", str(db), "--out", str(out), "--bootstrap-n", "5"])
    assert r.exit_code == 0, r.output
    assert "budget-limited" not in r.output


def test_report_min_clusters_reaches_config(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db)
    seen = _capture_cfg(monkeypatch)
    r = runner.invoke(app, ["report", "--db", str(db), "--out-dir", str(tmp_path / "reports"),
                            "--bootstrap-n", "5", "--min-clusters", "2"])
    assert r.exit_code == 0, r.output
    assert seen["cfg"].min_clusters == 2


def test_report_rejects_non_positive_min_clusters(tmp_path):
    r = runner.invoke(app, ["report", "--db", str(tmp_path / "t.db"), "--min-clusters", "0"])
    assert r.exit_code == 2


# --- fetch --reprofile-raters (M4 escape hatch) -------------------------------------


def _stub_fetch_steps(monkeypatch, on_enrich=None):
    """Stub every fetch step so only the raters step does real work."""
    monkeypatch.setattr(cli, "RpcClient", _FakeRpc)
    monkeypatch.setattr(cli.base, "sync_feedback", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "fill_block_timestamps", lambda *a, **k: 0)
    monkeypatch.setattr(cli.base, "owners_of", lambda rpc, agent_ids, **k: {a: None for a in agent_ids})
    monkeypatch.setattr(cli, "classify_all", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "enrich_raters", on_enrich or (lambda *a, **k: 0))


def test_fetch_reprofile_raters_clears_the_cache_before_profiling(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=2)  # both raters already profiled
    seen = {}

    def _enrich(store, client, addresses=None):
        seen["targets"] = list(addresses or [])
        return len(seen["targets"])

    _stub_fetch_steps(monkeypatch, on_enrich=_enrich)
    r = runner.invoke(app, ["fetch", "--db", str(db), "--reprofile-raters"])
    assert r.exit_code == 0, r.output
    assert "rater profiles cleared: 2" in r.output
    assert len(seen["targets"]) == 2  # cleared first, so both are targets again
    with Store(db) as s:
        assert len(s.load_rater_meta()) == 0


def test_fetch_without_reprofile_raters_keeps_the_cache(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=2)
    seen = {}

    def _enrich(store, client, addresses=None):
        seen["targets"] = list(addresses or [])
        return 0

    _stub_fetch_steps(monkeypatch, on_enrich=_enrich)
    r = runner.invoke(app, ["fetch", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "cleared" not in r.output
    assert seen["targets"] == []
    with Store(db) as s:
        assert len(s.load_rater_meta()) == 2


def test_fetch_reprofile_raters_is_ignored_when_raters_are_skipped(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed(db, n=2)
    _stub_fetch_steps(monkeypatch, on_enrich=lambda *a, **k: 0)
    r = runner.invoke(app, ["fetch", "--db", str(db), "--reprofile-raters", "--skip-raters"])
    assert r.exit_code == 0, r.output
    with Store(db) as s:
        assert len(s.load_rater_meta()) == 2


def test_fetch_declares_the_reprofile_raters_option():
    import typer.main
    cmd = typer.main.get_command(app).commands["fetch"]
    declared = {opt for p in cmd.params for opt in p.opts}
    assert "--reprofile-raters" in declared
    flag = next(p for p in cmd.params if "--reprofile-raters" in p.opts)
    assert flag.default is False
