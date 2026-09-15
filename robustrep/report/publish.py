"""Assemble one report directory and publish it as ``reports/latest/``.

Split out of ``robustrep.cli``, which had grown past this project's file-size
ceiling carrying two unrelated jobs: command wiring (parsing options, opening
the store, sequencing steps, turning failures into exit codes) and report
assembly (what a report *is*, what provenance it carries, and which of its
files may reach a public CDN). This module owns the second, so the publishing
rules can be read -- and reviewed -- without wading through typer option
declarations. The dependency runs one way only: ``cli`` imports this module,
never the reverse.

Two security-review rules live here:

- **L1, the publish allowlist** (``PUBLISHED_FILES`` / ``PUBLISHED_FIGURE_GLOB``
  / ``_published_files``): ``latest/`` is committed and served by GitHub Pages,
  so ``publish_latest`` copies named artifacts, never a whole directory.
- **L9, symlink-safe replacement** (``publish_latest``): a ``latest`` that is a
  symlink is unlinked rather than ``rmtree``d, and its target is left alone.

``build_provenance`` belongs here for the same reason: what a published report
says about the run that made it is part of the artifact, not part of the CLI.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas

from ..config import Config
from ..evidence import MAX_TX_LOOKUPS_PER_URI
from ..sources.evidence_batch import DEFAULT_MAX_TOTAL_LOOKUPS
from ..store import Store
from .export import export_json, versions
from .figures import fig_evidence, fig_mean_vs_robust, fig_rank_shift, fig_sybil_clusters
from .render import evidence_level_shares, render_markdown
from .sensitivity import fig_sensitivity

logger = logging.getLogger(__name__)


# L1 (security review): with `PUBLISHED_FIGURE_GLOB`, everything
# `publish_latest` copies out of a block directory into `latest/`, which is
# committed and served by GitHub Pages. An allowlist, never "everything in the
# directory": anything else that ever lands next to the report (a debug dump,
# a scratch CSV, an operator's notes) stays local, not on a public CDN.
PUBLISHED_FILES = ("report.md", "scores.json")
PUBLISHED_FIGURE_GLOB = "fig*.png"

# sync_state keys recording the two H3 evidence lookup caps a `fetch` run
# actually used: written by `cli._step_evidence`, read by
# `evidence_caps_for_provenance`. Named once because a writer and a reader
# that spell a key differently fail silently (the reader just falls back).
EVIDENCE_CAP_PER_URI_KEY = "evidence_max_lookups_per_uri"
EVIDENCE_CAP_TOTAL_KEY = "evidence_max_total_lookups"
EVIDENCE_CAP_KEYS = (EVIDENCE_CAP_PER_URI_KEY, EVIDENCE_CAP_TOTAL_KEY)


# Figure titles (heading text) -> file names, in report order. Titles are what
# `render_markdown` uses as the Markdown heading/alt text (never a raw
# filename); the mapping keeps the two in one place.
_FIGURES = {
    "Fig 1. Mean vs robust score": "fig1_mean_vs_robust.png",
    "Fig 2. Biggest rank drops": "fig2_rank_shift.png",
    "Fig 3. Evidence levels": "fig3_evidence.png",
    "Fig 4. Largest rater clusters": "fig4_sybil_clusters.png",
    "Fig 5. Ranking stability": "fig5_sensitivity.png",
}


def _evidence_shares_for_provenance(records: pandas.DataFrame) -> dict:
    """Per-record evidence-level shares (0..3) among non-revoked records, as a
    JSON-friendly ``{level: share}`` dict -- the same figure the report's
    headline numbers are built from (see ``robustrep.report.render``)."""
    non_revoked = records[records["revoked"] == 0] if "revoked" in records.columns else records
    shares = evidence_level_shares(non_revoked)
    return {int(level): float(share) for level, share in shares.items()}


def _norm_rule_counts_for_provenance(result: Optional[pandas.DataFrame]) -> dict:
    """Records per normalization rule, read off the scored frame's ``attrs``.

    ``pipeline.score`` stamps these from the frame it already prepared, so they
    describe exactly the run being reported (revoked rows dropped, that run's
    ``norm_fit_share`` applied) and cost nothing -- recomputing them here would
    re-run validate/normalize over the whole dataset. Normalization picks a rule
    per (tag, scale) group from the group's own contents, so a rung flip is the
    visible symptom of a value-poisoning attempt: publishing the counts makes one
    show up as a plain diff between two runs' reports. ``{}`` when no scored
    frame is supplied.
    """
    if result is None:
        return {}
    return {str(rule): int(n) for rule, n in result.attrs.get("norm_rule_counts", {}).items()}


def _cluster_stats_for_provenance(result: Optional[pandas.DataFrame]) -> dict:
    """The run's `ClusterStats` as a plain dict, read off the scored frame's
    ``attrs`` (stamped by `score`/`report`).

    Published next to `config` so a consumer diffing two exports can tell a
    real change from one caused by a budget-limited clustering pass. ``{}``
    when no scored frame is supplied.
    """
    if result is None:
        return {}
    return dict(result.attrs.get("cluster_stats", {}))


def evidence_caps_for_provenance(store: Optional[Store]) -> dict:
    """The two H3 lookup caps the store's last ``fetch`` run actually used
    (``EVIDENCE_CAP_KEYS`` in ``sync_state``, written by ``cli._step_evidence``),
    falling back to today's module constants per key.

    Provenance has to describe the run that produced the cached evidence
    levels, not the build that happens to be rendering the report: the levels
    outlive the constants, so a store fetched under an older cap must keep
    reporting that cap. The fallback covers a store that predates this
    recording, one whose evidence step was skipped, and a hand-edited value
    that is not a usable cap -- a bad sync_state row degrades one provenance
    field rather than failing the whole report.

    "Usable" means a positive integer: a recorded ``0`` or negative is a
    corrupted row (``cli._step_evidence`` only writes module constants), and
    publishing it would claim the run did no lookups at all.
    """
    caps = {EVIDENCE_CAP_PER_URI_KEY: MAX_TX_LOOKUPS_PER_URI,
            EVIDENCE_CAP_TOTAL_KEY: DEFAULT_MAX_TOTAL_LOOKUPS}
    if store is None:
        return caps
    for key in EVIDENCE_CAP_KEYS:
        raw = store.get_sync(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            caps[key] = value
        else:
            logger.warning("provenance: ignoring unusable %s=%r in sync_state", key, raw)
    return caps


def build_provenance(mode: str, cfg: Config, records: pandas.DataFrame,
                result: Optional[pandas.DataFrame] = None,
                n_lookup_budget_starved: int = 0,
                evidence_caps: Optional[dict] = None) -> dict:
    caps = evidence_caps if evidence_caps is not None else evidence_caps_for_provenance(None)
    config = dict(
        bootstrap_n=cfg.bootstrap_n,
        bootstrap_seed=cfg.bootstrap_seed,
        min_clusters=cfg.min_clusters,
        evidence_weights=list(cfg.evidence_weights),
        sybil_jaccard=cfg.sybil_jaccard,
        sybil_window_s=cfg.sybil_window_s,
        sybil_max_group=cfg.sybil_max_group,
        sybil_max_pairs=cfg.sybil_max_pairs,
        sybil_max_pairs_per_ratee=cfg.sybil_max_pairs_per_ratee,
        sybil_flag_share=cfg.sybil_flag_share,
        norm_fit_share=cfg.norm_fit_share,
        norm_rule_counts=_norm_rule_counts_for_provenance(result),
        evidence_level_shares=_evidence_shares_for_provenance(records),
        # H3 (security review): not Config fields (they're module constants,
        # not user-tunable per this task's scope) but they directly shaped
        # which cached evidence levels are false negatives, so they belong
        # next to evidence_level_shares for a report to be auditable. The
        # values are the ones the *fetch run* used, read back out of the store.
        **caps,
        # How many evidence_cache rows are still "lookup-budget:N" as of this
        # report's store snapshot -- see Store.n_lookup_budget_starved.
        # `n_lookup_budget_starved` defaults to 0 (not "unknown") for a
        # caller with no store to count from (e.g. `build_provenance` called
        # directly on an in-memory `records` frame, as several tests do).
        evidence_lookup_starved_uris=n_lookup_budget_starved,
    )
    return dict(
        rater_profile_mode=mode,
        confirmations=cfg.confirmations,
        config=config,
        cluster_stats=_cluster_stats_for_provenance(result),
        versions=versions(),
    )


def write_report(target: Path, result, records, clusters, sens, adv, block: int, provenance: dict,
                  top_n: int) -> None:
    """Write scores.json, then draw all 5 figures and write report.md, into
    `target`.

    scores.json goes FIRST because `export_json` is the only step here that
    can refuse its input (the L1 address guard raises `ValueError`, which
    `cli.report` turns into an ERROR line) and it writes nothing when it does.
    Exporting first means a refusal leaves the block directory empty rather
    than holding five figures and a report.md with no scores.json beside them
    -- a shape an operator could mistake for a finished run.
    """
    target.mkdir(parents=True, exist_ok=True)
    export_json(result, block=block, out=target / "scores.json", config=provenance.get("config"),
                cluster_stats=provenance.get("cluster_stats"))
    fig_mean_vs_robust(result, out=target / _FIGURES["Fig 1. Mean vs robust score"])
    fig_rank_shift(result, out=target / _FIGURES["Fig 2. Biggest rank drops"], top_n=top_n)
    fig_evidence(records, out=target / _FIGURES["Fig 3. Evidence levels"])
    fig_sybil_clusters(records, clusters, out=target / _FIGURES["Fig 4. Largest rater clusters"])
    fig_sensitivity(sens, out=target / _FIGURES["Fig 5. Ranking stability"])
    md = render_markdown(result, records, block=block, figures=_FIGURES, sensitivity=sens,
                         adversarial=adv, provenance=provenance)
    (target / "report.md").write_text(md)


def _published_files(block_dir: Path) -> List[Path]:
    """The files in `block_dir` that may be published (L1): the
    `PUBLISHED_FILES` allowlist plus every `PUBLISHED_FIGURE_GLOB` match, in
    that order, skipping any that do not exist.

    An allowlist rather than a denylist, and names rather than a tree walk:
    `latest/` goes to a public CDN, so the decision has to be "these three
    kinds of artifact", not "whatever the last run left lying around".
    Directories are never published (`is_file`), so a subdirectory matching
    the glob cannot smuggle its contents through either.

    Symlinks are excluded too (`not is_symlink()`): `is_file()` follows them,
    so an allowlisted *name* pointing anywhere on disk would otherwise publish
    that target's bytes -- the allowlist is about bytes, not names.
    """
    named = [block_dir / name for name in PUBLISHED_FILES]
    figures = sorted(block_dir.glob(PUBLISHED_FIGURE_GLOB))
    return [p for p in named + figures if p.is_file() and not p.is_symlink()]


def publish_latest(out_dir: Path, block_dir: Path) -> None:
    """Publish the allowlisted files of `block_dir` (see `_published_files`)
    as `out_dir/latest/`.

    Near-atomic: the old `latest/` is removed then the staged copy is moved in
    -- a reader may briefly see no `latest/` at all, in that gap. Contents are
    built in a temp directory first, so the gap is one `os.replace` (no partial
    writes ever visible) rather than however long the figures/report.md/
    scores.json take to generate; stale files never linger beside the new ones.

    L9: an existing `latest` that is a *symlink* is unlinked, not `rmtree`d
    (which refuses a symlink with `OSError` and would wedge every later run);
    only the link goes, never its target. The check is `is_symlink()` and
    comes first, since `exists()` follows symlinks and reports a dangling one
    as False.

    Raises `ValueError` (which `cli.report` turns into an ERROR line) if
    `block_dir` is missing either of `PUBLISHED_FILES`: publishing is
    destructive, so a block directory with no report in it must not replace a
    good `latest/` with a figures-only or empty one and take the live report
    offline. Missing *figures* are not an error -- refusing on the recoverable
    half of the report would only be a new way to break publishing.
    """
    missing = [name for name in PUBLISHED_FILES if not (block_dir / name).is_file()]
    if missing:
        raise ValueError(f"cannot publish {block_dir}: missing {', '.join(missing)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest"
    sources = _published_files(block_dir)
    tmp_parent = Path(tempfile.mkdtemp(dir=out_dir))
    try:
        staged = tmp_parent / "latest"
        staged.mkdir()
        for src in sources:
            shutil.copy2(src, staged / src.name)
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            shutil.rmtree(latest)
        os.replace(staged, latest)
        # After the move: the line reports what is published, not what staged.
        logger.debug("published %d file(s) to %s: %s", len(sources), latest,
                     [p.name for p in sources])
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)
