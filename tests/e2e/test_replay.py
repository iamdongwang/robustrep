"""End-to-end replay: a recorded fixture of real ReputationRegistry logs
(``tests/e2e/fixtures/base_logs_2000.json``, see ``scripts/record_fixture.py``)
replayed through ``sync_feedback`` / ``fill_block_timestamps`` / ``score`` /
the CLI ``report`` command, entirely offline via the ``ReplayRpc`` fake below.
No test in this module touches the network -- see ``tests/e2e/test_live.py``
for the ``@pytest.mark.live`` counterparts.
"""
import json
from pathlib import Path

from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from robustrep import Config, score
from robustrep.cli import app
from robustrep.sources import base_erc8004 as base
from robustrep.store import Store
from robustrep.sybil import cluster_raters, profiles_from_records

FIX = json.loads((Path(__file__).parent / "fixtures/base_logs_2000.json").read_text())

runner = CliRunner()


class ReplayRpc:
    """Offline stand-in for ``RpcClient``: answers every call/batch from the
    recorded fixture, and raises on any method the fixture doesn't cover
    (so an accidental new network call fails loudly instead of hanging)."""

    def call(self, method, params):
        if method == "eth_blockNumber":
            return hex(FIX["to_block"])
        if method == "eth_getLogs":
            f, t = int(params[0]["fromBlock"], 16), int(params[0]["toBlock"], 16)
            return [l for l in FIX["logs"] if f <= int(l["blockNumber"], 16) <= t]
        raise AssertionError(method)

    def batch(self, calls):
        return [{"timestamp": hex(FIX["block_ts"][str(int(p[0], 16))])} for _, p in calls]


def _sync_and_score(db_path, end_block=None):
    """Sync the fixture into a fresh store at `db_path`, fill timestamps,
    cluster and score. Returns (store, records, result)."""
    store = Store(db_path)
    base.sync_feedback(store, ReplayRpc(), chunk=2000, start_block=FIX["from_block"],
                       end_block=end_block if end_block is not None else FIX["to_block"])
    base.fill_block_timestamps(store, ReplayRpc())
    records = store.load_records()
    cfg = Config(bootstrap_n=20)
    clusters = cluster_raters(profiles_from_records(records, store.load_rater_meta()), cfg)
    result = score(records, cfg, clusters=clusters)
    return store, records, result


def test_replay_full_pipeline(tmp_path):
    store = Store(tmp_path / "replay.db")
    n = base.sync_feedback(store, ReplayRpc(), chunk=2000, start_block=FIX["from_block"], end_block=FIX["to_block"])
    assert n == sum(1 for l in FIX["logs"] if l["topics"][0].lower() == base.TOPIC_NEW_FEEDBACK)
    base.fill_block_timestamps(store, ReplayRpc())
    records = store.load_records()
    assert (records["ts"] > 0).all()
    cfg = Config(bootstrap_n=20)
    clusters = cluster_raters(profiles_from_records(records, store.load_rater_meta()), cfg)
    result = score(records, cfg, clusters=clusters)
    assert len(result) == records["ratee"].nunique()
    assert result["naive_mean"].between(0, 1).all()

    assert store.get_sync("last_block") == str(FIX["to_block"])
    store.close()

    # --- CLI report on the same replayed store -----------------------------
    out_dir = tmp_path / "reports"
    r = runner.invoke(app, ["report", "--db", str(tmp_path / "replay.db"), "--out-dir", str(out_dir),
                            "--bootstrap-n", "20"])
    assert r.exit_code == 0, r.output

    block = FIX["to_block"]
    for target in (out_dir / str(block), out_dir / "latest"):
        assert target.is_dir()
        pngs = sorted(target.glob("*.png"))
        assert len(pngs) == 5
        assert (target / "report.md").exists()
        json_path = target / "scores.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["block"] == block

    n_scored = int((result["insufficient"] == 0).sum())
    n_insufficient = int((result["insufficient"] == 1).sum())
    assert n_scored + n_insufficient == len(result)
    if n_scored == 0:
        # Note: the recorded window happens to have only insufficient ratees
        # (fewer than cfg.min_clusters distinct rater clusters each) -- the
        # pipeline still ran end-to-end, just with nothing to rank yet.
        assert n_insufficient == len(result) > 0
    else:
        assert n_scored > 0


def test_replay_is_deterministic(tmp_path):
    _, _, result_a = _sync_and_score(tmp_path / "a.db")
    _, _, result_b = _sync_and_score(tmp_path / "b.db")
    assert_frame_equal(result_a, result_b)


def test_replay_resume(tmp_path):
    single_store = Store(tmp_path / "single.db")
    n_single = base.sync_feedback(single_store, ReplayRpc(), chunk=2000, start_block=FIX["from_block"],
                                  end_block=FIX["to_block"])
    single_store.close()

    resume_store = Store(tmp_path / "resume.db")
    midpoint = FIX["from_block"] + 999
    n1 = base.sync_feedback(resume_store, ReplayRpc(), chunk=2000, start_block=FIX["from_block"],
                            end_block=midpoint)
    n2 = base.sync_feedback(resume_store, ReplayRpc(), chunk=2000, start_block=FIX["from_block"],
                            end_block=FIX["to_block"])
    assert n1 + n2 == n_single
    assert resume_store.get_sync("last_block") == str(FIX["to_block"])
    resume_store.close()
