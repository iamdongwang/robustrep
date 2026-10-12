"""Report wording and the 0.2.0 "Log sources" Provenance lines.

Split out of ``test_report.py`` (file size). The wording guard runs against
both log-source paths -- not recorded and listed -- so neither can introduce
an overclaim or a second "cross-checked against".
"""
import pandas as pd
import pytest

from robustrep import Config, score
from robustrep.report.adversarial import scenario_table
from robustrep.report.render import render_markdown


def _data(records_factory):
    rows = []
    for a in range(6):
        rows += [dict(rater=f"r{a}{i}", ratee=str(a), value=50 + a * 8, evidence_level=i % 4) for i in range(4)]
    rec = records_factory(rows)
    return rec, score(rec, Config(bootstrap_n=5))


def _render_with_config(records_factory, config):
    rec, sc = _data(records_factory)
    return render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]), provenance=dict(
        rater_profile_mode="onchain", confirmations=20, config=config, versions={}))


_TWO_SOURCES = {
    "sources": [
        dict(host="https://base-pokt.nodies.app", blocks_min=51_322_404, blocks_max=52_507_435,
             n_ranges=23_703, n_blocks=1_185_032, n_logs=9_870,
             first_fetched_at=1, last_fetched_at=2),
        dict(host="https://mainnet.base.org:8545/rpc", blocks_min=52_507_436, blocks_max=52_510_000,
             n_ranges=3, n_blocks=2_565, n_logs=12, first_fetched_at=3, last_fetched_at=4),
    ],
    "recorded_from_block": 51_322_404,
}


@pytest.mark.parametrize("config", [{}, {"rpc_sources": _TWO_SOURCES}], ids=["not-recorded", "listed"])
def test_render_markdown_never_claims_sybil_detection_and_states_measured_boundaries(records_factory, config):
    rec, sc = _data(records_factory)
    adv = scenario_table()
    md = render_markdown(sc, rec, block=1, figures={}, sensitivity=pd.DataFrame(
        [dict(variant="base", spearman_top=1.0)]), adversarial=adv, provenance=dict(
        rater_profile_mode="onchain", confirmations=20, config=config, versions={}))
    lowered = md.lower()
    for banned in ("sybil detected", "sybil detection", "detects sybils", "sybil cluster",
                   "corroborated", "independent sources confirm", "verified by multiple"):
        assert banned not in lowered, banned
    assert md.count("cross-checked against") == 1
    limitation = next(line for line in md.splitlines() if "cross-checked against" in line)
    assert limitation.startswith("- **Log sources not cross-checked.**")
    assert "flips the score at k=30" in md
    assert "smear flips at 35" in md


def test_render_log_sources_not_recorded(records_factory):
    md = _render_with_config(records_factory, {"rpc_sources": "unavailable"})
    assert "- Log sources: not recorded (fetched before source recording existed)." in md
    assert "Log sources not cross-checked" in md


def test_render_log_sources_missing_key_reads_not_recorded(records_factory):
    md = _render_with_config(records_factory, {"bootstrap_n": 20})
    assert "- Log sources: not recorded (fetched before source recording existed)." in md
    assert "Log sources not cross-checked" in md


def test_render_log_sources_listed(records_factory):
    md = _render_with_config(records_factory, {"rpc_sources": _TWO_SOURCES})
    assert ("- Log sources: base-pokt.nodies.app (blocks 51,322,404\u201352,507,435, "
            "23,703 ranges, 1,185,032 blocks, 9,870 logs); mainnet.base.org (blocks "
            "52,507,436\u201352,510,000, 3 ranges, 2,565 blocks, 12 logs)") in md
    assert ("- Source recording starts at block 51,322,404; any blocks before it were fetched "
            "before source recording existed.") in md
    assert "Log sources not cross-checked" in md
    assert "not recorded (fetched before" not in md
    # the source lines sit in Provenance, before the scoring-configuration dump.
    lines = md.splitlines()
    assert (lines.index("## Provenance")
            < next(i for i, l in enumerate(lines) if l.startswith("- Log sources:"))
            < lines.index("- Scoring configuration used:"))


def test_rpc_sources_not_dumped_raw(records_factory):
    md = _render_with_config(records_factory, {"bootstrap_n": 20, "rpc_sources": _TWO_SOURCES})
    assert "{'host'" not in md
    assert "`rpc_sources`" not in md
    assert "  - `bootstrap_n`: 20" in md


def test_log_source_limitation_is_last_limitation(records_factory):
    md = _render_with_config(records_factory, {})
    lines = md.splitlines()
    end = lines.index("## Provenance")
    bullets = [l for l in lines[lines.index("## Limitations"):end] if l.startswith("- ")]
    assert bullets[-1] == (
        "- **Log sources not cross-checked.** Logs from different endpoints are not "
        "cross-checked against each other; the weekly drift test covers only the recorded "
        "fixture window.")


def test_render_log_sources_host_without_hostname_reads_unknown_host(records_factory):
    bad = dict(_TWO_SOURCES, sources=[dict(_TWO_SOURCES["sources"][0], host="not-a-url")])
    md = _render_with_config(records_factory, {"rpc_sources": bad})
    assert "- Log sources: unknown host (blocks 51,322,404\u201352,507,435," in md
