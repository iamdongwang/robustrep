"""Report assembly and publishing (`robustrep.report.publish`).

Unit tests for the functions that decide what a report contains
(`build_provenance` and its helpers) and which of its files reach
`reports/latest/` (`publish_latest`, `_published_files`). The end-to-end
`report` command tests -- everything that drives these through a CliRunner
invocation, including the L1/L9 behaviour as the CLI exposes it -- stay in
`test_cli.py`, next to the command wiring they exercise.
"""
import logging

import pandas as pd
import pytest

from robustrep.report import publish
from robustrep.store import Store


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


def test_provenance_n_lookup_budget_starved_defaults_to_zero_without_a_store():
    # build_provenance is also called directly (e.g. from tests) on an
    # in-memory records frame with no store to count from -- it must not
    # require one.
    from robustrep import Config
    from robustrep.schema import validate_records

    records = validate_records(pd.DataFrame([dict(
        rater="r", ratee="A", value=50, scale="d0", tag="q", ts=0,
        evidence_uri=None, source="test", evidence_level=0)]))
    prov = publish.build_provenance("onchain", Config(), records)
    assert prov["config"]["evidence_lookup_starved_uris"] == 0


def test_provenance_evidence_level_shares_from_records(tmp_path):
    """`build_provenance` computes `evidence_level_shares` directly from the
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
    prov = publish.build_provenance("onchain", Config(), records)
    shares = prov["config"]["evidence_level_shares"]
    assert shares[0] == pytest.approx(0.2)
    assert shares[1] == pytest.approx(0.6)
    assert shares[2] == pytest.approx(0.1)
    assert shares[3] == pytest.approx(0.1)


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
        prov = publish.build_provenance("onchain", cfg, records, score(records, cfg))
        assert prov["config"]["norm_fit_share"] == fit_share
        assert prov["config"]["norm_rule_counts"] == {rule: 10}


def test_provenance_cluster_stats_empty_without_a_scored_frame(tmp_path):
    from robustrep import Config

    db = tmp_path / "t.db"
    _seed(db)
    with Store(db) as store:
        records = store.load_records()
    assert publish.build_provenance("onchain", Config(), records)["cluster_stats"] == {}


def test_provenance_versions_come_from_the_export_module(tmp_path):
    from robustrep import Config
    from robustrep.report.export import versions as export_versions
    from robustrep.schema import validate_records

    records = validate_records(pd.DataFrame([dict(
        rater="r", ratee="A", value=50, scale="d0", tag="q", ts=0,
        evidence_uri=None, source="test", evidence_level=0)]))
    assert publish.build_provenance("onchain", Config(), records)["versions"] == export_versions()


def test_evidence_caps_for_provenance_falls_back_to_constants(tmp_path):
    from robustrep.evidence import MAX_TX_LOOKUPS_PER_URI
    from robustrep.sources.evidence_batch import DEFAULT_MAX_TOTAL_LOOKUPS

    expected = {"evidence_max_lookups_per_uri": MAX_TX_LOOKUPS_PER_URI,
                "evidence_max_total_lookups": DEFAULT_MAX_TOTAL_LOOKUPS}
    # No store at all (a caller scoring an in-memory frame).
    assert publish.evidence_caps_for_provenance(None) == expected
    # A store that has never run an evidence step -- neither key is set.
    with Store(tmp_path / "fresh.db") as store:
        assert publish.evidence_caps_for_provenance(store) == expected
    # A malformed value (hand-edited sync_state) falls back rather than
    # crashing the whole report.
    with Store(tmp_path / "bad.db") as store:
        store.set_sync("evidence_max_total_lookups", "not-a-number")
        assert publish.evidence_caps_for_provenance(store) == expected


def test_evidence_caps_for_provenance_rejects_non_positive_values(tmp_path, caplog):
    from robustrep.evidence import MAX_TX_LOOKUPS_PER_URI
    from robustrep.sources.evidence_batch import DEFAULT_MAX_TOTAL_LOOKUPS

    expected = {"evidence_max_lookups_per_uri": MAX_TX_LOOKUPS_PER_URI,
                "evidence_max_total_lookups": DEFAULT_MAX_TOTAL_LOOKUPS}
    with Store(tmp_path / "nonpos.db") as store:
        store.set_sync("evidence_max_lookups_per_uri", "0")
        store.set_sync("evidence_max_total_lookups", "-5")
        with caplog.at_level(logging.WARNING, logger="robustrep.report.publish"):
            assert publish.evidence_caps_for_provenance(store) == expected
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "evidence_max_lookups_per_uri" in warned and "evidence_max_total_lookups" in warned


def test_publish_latest_copies_only_the_allowlisted_files(tmp_path):
    # L1 (security review): `latest/` is served by GitHub Pages, so only the
    # three artifacts the report actually consists of may be published -- a
    # stray debug dump left in the block directory must never ride along onto
    # a public CDN.
    out_dir = tmp_path / "reports"
    block_dir = out_dir / "42"
    block_dir.mkdir(parents=True)
    (block_dir / "report.md").write_text("# report")
    (block_dir / "scores.json").write_text("{}")
    (block_dir / "fig1_mean_vs_robust.png").write_bytes(b"png")
    (block_dir / "debug.csv").write_text("rater,secret\n")
    (block_dir / "notes.txt").write_text("scratch")
    (block_dir / "raw").mkdir()

    publish.publish_latest(out_dir, block_dir)

    latest = out_dir / "latest"
    assert {p.name for p in latest.iterdir()} == {
        "report.md", "scores.json", "fig1_mean_vs_robust.png"}


def test_publish_latest_replaces_a_symlinked_latest_with_a_real_directory(tmp_path):
    # L9: `shutil.rmtree` raises OSError on a symlink, so a `latest` symlink
    # (however it got there) used to wedge every later report run.
    out_dir = tmp_path / "reports"
    block_dir = out_dir / "7"
    block_dir.mkdir(parents=True)
    (block_dir / "report.md").write_text("# report")
    (block_dir / "scores.json").write_text("{}")
    (block_dir / "fig1.png").write_bytes(b"png")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("not ours")
    (out_dir / "latest").symlink_to(elsewhere, target_is_directory=True)

    publish.publish_latest(out_dir, block_dir)

    latest = out_dir / "latest"
    assert not latest.is_symlink() and latest.is_dir()
    assert (latest / "report.md").exists()
    # The symlink target itself is untouched -- only the link was removed.
    assert (elsewhere / "keep.txt").exists()


def test_publish_latest_logs_the_published_files_at_debug(tmp_path, caplog):
    out_dir = tmp_path / "reports"
    block_dir = out_dir / "3"
    block_dir.mkdir(parents=True)
    (block_dir / "report.md").write_text("# report")
    (block_dir / "scores.json").write_text("{}")
    (block_dir / "fig1.png").write_bytes(b"png")

    with caplog.at_level(logging.DEBUG, logger="robustrep.report.publish"):
        publish.publish_latest(out_dir, block_dir)

    published = " ".join(r.getMessage() for r in caplog.records)
    for name in ("report.md", "scores.json", "fig1.png"):
        assert name in published, name


def test_publish_latest_logs_only_after_latest_is_in_place(tmp_path, caplog):
    # The DEBUG line says what *was* published, so it must not be emitted
    # while the staged copy could still fail to move into place.
    out_dir = tmp_path / "reports"
    block_dir = out_dir / "3"
    block_dir.mkdir(parents=True)
    (block_dir / "report.md").write_text("# report")
    (block_dir / "scores.json").write_text("{}")
    (block_dir / "fig1.png").write_bytes(b"png")

    published_at_log_time = []

    class _Probe(logging.Handler):
        def emit(self, record):
            published_at_log_time.append((out_dir / "latest" / "report.md").exists())

    probe = _Probe(level=logging.DEBUG)
    logger = logging.getLogger("robustrep.report.publish")
    logger.addHandler(probe)
    try:
        with caplog.at_level(logging.DEBUG, logger="robustrep.report.publish"):
            publish.publish_latest(out_dir, block_dir)
    finally:
        logger.removeHandler(probe)

    assert published_at_log_time and all(published_at_log_time)


@pytest.mark.parametrize("missing", ["report.md", "scores.json"])
def test_publish_latest_refuses_when_a_core_file_is_missing(tmp_path, missing):
    # Publishing an empty (or figure-only) `latest/` would take the live
    # report offline; refuse and let report() turn it into an ERROR line.
    out_dir = tmp_path / "reports"
    block_dir = out_dir / "9"
    block_dir.mkdir(parents=True)
    for name in ("report.md", "scores.json"):
        if name != missing:
            (block_dir / name).write_text("x")
    (block_dir / "fig1.png").write_bytes(b"png")

    with pytest.raises(ValueError) as e:
        publish.publish_latest(out_dir, block_dir)
    assert missing in str(e.value)
    assert not (out_dir / "latest").exists()


def test_published_files_never_follows_a_symlink(tmp_path):
    # A figure-shaped symlink in the block directory must not publish whatever
    # it points at -- the allowlist is about bytes leaving the machine, not
    # about names.
    block_dir = tmp_path / "3"
    block_dir.mkdir()
    (block_dir / "report.md").write_text("# report")
    (block_dir / "scores.json").write_text("{}")
    (block_dir / "fig1.png").write_bytes(b"png")
    secret = tmp_path / "secret"
    secret.write_text("private key material")
    (block_dir / "fig3.png").symlink_to(secret)

    names = {p.name for p in publish._published_files(block_dir)}
    assert names == {"report.md", "scores.json", "fig1.png"}

    out_dir = tmp_path / "reports"
    publish.publish_latest(out_dir, block_dir)
    assert not (out_dir / "latest" / "fig3.png").exists()
