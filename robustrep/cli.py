"""``robustrep fetch`` / ``robustrep score`` command-line entry points.

Kept intentionally thin: this module only orchestrates calls into the
library (``robustrep.sources.*``, ``robustrep.pipeline``, ``robustrep.sybil``,
``robustrep.store``) and never re-implements their business logic. ``fetch``'s
pipeline is broken into small private ``_step_*`` helpers (one per stage:
sync, block timestamps, agent owners, rater profiles, evidence) that each
return a one-line summary string -- this lets tests monkeypatch the
``base``/``enrich_raters``/``classify_all`` symbols in this module and drive
the whole command end-to-end without any network access.

``report`` generates figures 1-5, ``report.md`` and ``scores.json`` under
``out_dir/<block>/``, then publishes an atomic copy to ``out_dir/latest/``.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy
import pandas
import typer

from . import __version__
from .config import Config
from .pipeline import score as score_fn
from .report.adversarial import scenario_table
from .report.export import export_json
from .report.figures import fig_evidence, fig_mean_vs_robust, fig_rank_shift, fig_sybil_clusters
from .report.render import render_markdown
from .report.sensitivity import fig_sensitivity, sensitivity_table
from .sources import base_erc8004 as base
from .sources.evidence_fetch import classify_all
from .sources.rater_profile import DEFAULT_RPS, EtherscanClient, enrich_raters, estimate_seconds
from .sources.rpc import RpcClient, RpcError
from .store import Store
from .sybil import cluster_raters, profiles_from_records

app = typer.Typer(add_completion=False, help="Robust, transport-agnostic reputation scoring for AI agents.")

logger = logging.getLogger(__name__)

# Agent-owner resolution progress is logged (INFO) every this many agents.
OWNER_LOG_EVERY = 500

# Default number of agents resolved per JSON-RPC batch in _step_owners (see
# --owner-batch).
OWNER_BATCH_DEFAULT = 100


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", help="Enable DEBUG-level logging.")) -> None:
    """Robust, transport-agnostic reputation scoring for AI agents."""
    # Only ever *adjust the level* of our own package logger -- never call
    # logging.basicConfig(..., force=True), which tears down and replaces every
    # handler on the root logger (including one belonging to a caller embedding
    # this CLI, or pytest's caplog) with a fresh StreamHandler bound to
    # whatever sys.stderr happens to be *at this exact call*. Under repeated
    # `typer.testing.CliRunner.invoke()` calls in one process, that snapshot is
    # a redirected stream that gets closed once the invocation ends, so a
    # later log call reusing that handler raises "I/O operation on closed
    # file." Adding a handler only when the root logger has none yet avoids
    # ever rebinding a handler to a stream that may later be closed.
    #
    # --verbose sets DEBUG (not INFO): _rpc_guard's "re-run with --verbose for
    # detail" message promises that --verbose actually surfaces the
    # DEBUG-level detail rpc.py and _rpc_guard log on failure; INFO wouldn't.
    #
    # Non-verbose resets to NOTSET (defer to the root logger's own level,
    # WARNING by default) rather than pinning WARNING explicitly on this
    # named logger: an explicit level here would otherwise outlive this call
    # (loggers are process-global singletons) and, unlike NOTSET, would block
    # a later `caplog.at_level(logging.INFO)` (which only raises the ROOT
    # logger's level) from ever reaching child loggers under "robustrep" --
    # Python's level lookup stops at the first ancestor with an explicit
    # level, so an explicit WARNING here would shadow root's override.
    logging.getLogger("robustrep").setLevel(logging.DEBUG if verbose else logging.NOTSET)
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler(sys.stderr))


def _no_blank(value: Optional[str]) -> Optional[str]:
    """Typer/click callback: reject an explicitly-empty/whitespace-only option
    value (e.g. ``--etherscan-key ""``) as a usage error rather than silently
    treating it as "not given"."""
    if value is not None and value.strip() == "":
        raise typer.BadParameter("must not be empty")
    return value


@contextmanager
def _rpc_guard():
    """Convert an ``RpcError`` escaping an RPC-touching step into a clean CLI
    failure: exit 3, echoing only the already-redacted message (see
    ``robustrep.sources.rpc._redact`` -- never the raw exception, which is
    what ``str(e)`` on an ``RpcError`` already guarantees) plus a pointer to
    ``--verbose``. The full exception (traceback included) still reaches the
    DEBUG log, which ``--verbose`` turns on.
    """
    try:
        yield
    except RpcError as e:
        typer.echo(f"ERROR: RPC failed: {e} (re-run with --verbose for detail)")
        logger.debug("fetch: RPC failure", exc_info=True)
        raise typer.Exit(3)


def _step_sync(store: Store, rpc: RpcClient, cfg: Config, to_block: Optional[int]) -> str:
    """Sync feedback/revocation/response events up to ``to_block`` (or head)."""
    n = base.sync_feedback(store, rpc, chunk=cfg.chunk_blocks, end_block=to_block, confirmations=cfg.confirmations)
    return f"feedback rows added: {n}"


def _step_timestamps(store: Store, rpc: RpcClient) -> str:
    """Resolve block timestamps for every block referenced by feedback rows."""
    n = base.fill_block_timestamps(store, rpc)
    still_missing = store.n_missing_block_ts()
    if still_missing > 0:
        typer.echo(f"WARNING: {still_missing} block(s) still missing a cached timestamp")
    return f"block timestamps filled: {n}"


def _step_owners(store: Store, rpc: RpcClient, log_every: int = OWNER_LOG_EVERY,
                  batch_size: int = OWNER_BATCH_DEFAULT) -> str:
    """Resolve the IdentityRegistry owner for every not-yet-resolved agent id.

    Owners are resolved ``batch_size`` at a time via ``base.owners_of`` (one
    JSON-RPC batch call per chunk instead of one ``eth_call`` per agent --
    ``owners_of`` itself falls back to per-agent calls if a chunk's batch
    fails). Every agent id is recorded, even when no owner resolves -- as an
    empty string (``Store.upsert_agent_owner`` accepts one; the evidence
    classification join already filters falsy owners out of its parties set)
    -- so an unresolved agent is never re-queried on a later run. Logs
    progress at INFO every ``log_every`` agents (not every chunk).
    """
    agents = store.distinct_agents()
    resolved = 0
    for i in range(0, len(agents), batch_size):
        chunk = agents[i:i + batch_size]
        owners = base.owners_of(rpc, chunk, batch_size=batch_size)
        for j, agent_id in enumerate(chunk):
            owner = owners.get(agent_id)
            store.upsert_agent_owner(agent_id, owner or "")
            if owner:
                resolved += 1
            n_done = i + j + 1
            if log_every > 0 and n_done % log_every == 0:
                logger.info("fetch: agent owners resolved %d/%d", n_done, len(agents))
    return f"agent owners resolved: {resolved}/{len(agents)}"


def _step_raters(store: Store, etherscan_key: Optional[str]) -> str:
    """Profile not-yet-profiled raters via Etherscan, or fall back offline.

    ``etherscan_key``, when given, overrides ``ETHERSCAN_API_KEY`` (the key
    itself is never echoed). Raises ``RuntimeError`` (from ``enrich_raters``)
    when some -- but not all -- addresses failed; the caller decides how to
    handle a partial failure.
    """
    key = etherscan_key if etherscan_key is not None else os.environ.get("ETHERSCAN_API_KEY")
    client = EtherscanClient(key) if key else None
    # store.distinct_clients() computed exactly once here and handed to
    # enrich_raters(addresses=...) below, instead of letting it re-query the
    # store itself.
    targets = store.distinct_clients()
    if client is not None and targets:
        eta_h = estimate_seconds(len(targets), DEFAULT_RPS) / 3600
        typer.echo(f"profiling {len(targets)} rater(s) via Etherscan, ETA ~{eta_h:.1f}h")
    n = enrich_raters(store, client, addresses=targets)
    mode = store.get_sync("rater_profile_mode") or "unknown"
    fallback_note = "" if client is not None else " (fallback: no ETHERSCAN_API_KEY; rows not written)"
    return f"raters profiled: {n}{fallback_note}; rater profile mode: {mode}"


def _step_evidence(store: Store, rpc: RpcClient) -> str:
    """Classify every not-yet-cached evidence URI referenced by feedback rows."""
    n = classify_all(store, tx_parties=lambda h: base.tx_parties(rpc, h))
    return f"evidence URIs classified: {n}"


@app.command()
def fetch(
    db: Path = typer.Option(..., help="SQLite store path."),
    to_block: Optional[int] = typer.Option(
        None, help="Sync feedback up to this block (default: chain head - confirmations)."),
    chunk: int = typer.Option(Config().chunk_blocks, min=1, help="Blocks per eth_getLogs call."),
    confirmations: int = typer.Option(
        Config().confirmations, min=0, help="Blocks to lag behind the chain head before ingesting."),
    rpc_url: List[str] = typer.Option(
        [], "--rpc-url", help="RPC endpoint (repeatable); overrides the default endpoint list."),
    etherscan_key: Optional[str] = typer.Option(
        None, "--etherscan-key", callback=_no_blank,
        help="Etherscan API key. Prefer the ETHERSCAN_API_KEY environment variable instead: "
             "a command-line argument is visible to other local users (e.g. via `ps`)."),
    skip_owners: bool = typer.Option(False, "--skip-owners", help="Skip agent owner resolution."),
    owner_batch: int = typer.Option(
        OWNER_BATCH_DEFAULT, "--owner-batch", min=1,
        help="Agents resolved per JSON-RPC batch call for owner resolution (1 = one call per agent)."),
    skip_raters: bool = typer.Option(False, "--skip-raters", help="Skip rater profile enrichment."),
    skip_evidence: bool = typer.Option(False, "--skip-evidence", help="Skip evidence URI classification."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be synced and exit, without any network calls."),
) -> None:
    """Sync Base ERC-8004 feedback, block timestamps, agent owners, rater
    profiles and evidence levels into the SQLite store at --db.

    Exits 0 on full success; 1 for a bad-input error (an invalid option
    combination that Config itself rejects); 2 when rater profiling partially
    failed (some addresses errored, the rest -- and every other step -- still
    completed; see _step_raters); 3 when an RPC call failed on every retry.
    --dry-run never opens or creates the store file when it does not already
    exist.
    """
    if dry_run:
        if db.exists():
            with Store(db) as store:
                last = store.get_sync("last_block")
            start = int(last) + 1 if last is not None else base.DEPLOY_BLOCK
        else:
            start = base.DEPLOY_BLOCK
        end = to_block if to_block is not None else "head"
        typer.echo(f"would sync from block {start} to {end}")
        raise typer.Exit(0)

    try:
        cfg = Config(chunk_blocks=chunk, confirmations=confirmations,
                     rpc_urls=tuple(rpc_url) if rpc_url else Config().rpc_urls)
    except ValueError as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

    rpc = RpcClient(cfg.rpc_urls, user_agent=cfg.user_agent)
    partial_failure = False

    with Store(db) as store:
        with _rpc_guard():
            typer.echo(_step_sync(store, rpc, cfg, to_block))
            typer.echo(_step_timestamps(store, rpc))
            if skip_owners:
                typer.echo("agent owner resolution skipped")
            else:
                typer.echo(_step_owners(store, rpc, batch_size=owner_batch))

        if skip_raters:
            mode = store.get_sync("rater_profile_mode") or "unknown"
            typer.echo(f"raters profiling skipped; rater profile mode: {mode}")
        else:
            try:
                typer.echo(_step_raters(store, etherscan_key))
            except RuntimeError as e:
                typer.echo(f"WARNING: {e}")
                partial_failure = True

        with _rpc_guard():
            if skip_evidence:
                typer.echo("evidence classification skipped")
            else:
                typer.echo(_step_evidence(store, rpc))

    if partial_failure:
        raise typer.Exit(2)


@app.command()
def score(
    db: Path = typer.Option(..., help="SQLite store path."),
    out: Path = typer.Option(Path("scores.csv"), help="CSV path to write results to."),
    bootstrap_n: int = typer.Option(
        Config().bootstrap_n, min=0, help="Bootstrap resamples for the confidence interval."),
    min_clusters: int = typer.Option(
        Config().min_clusters, min=1, help="Minimum distinct rater clusters required to score a ratee."),
) -> None:
    """Compute robust reputation scores for every ratee in the store and
    write them to --out as a CSV with RESULT_COLUMNS columns.

    Exits 1 with "no records" if the store has no feedback rows yet (run
    fetch first), or with an ERROR line if the given options or the data
    itself is rejected (e.g. too many candidate sybil pairs, or malformed
    records).
    """
    try:
        cfg = Config(bootstrap_n=bootstrap_n, min_clusters=min_clusters)
    except ValueError as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

    with Store(db) as store:
        records = store.load_records()
        if records.empty:
            typer.echo("no records")
            raise typer.Exit(1)

        try:
            clusters = cluster_raters(profiles_from_records(records, store.load_rater_meta()), cfg)
            result = score_fn(records, cfg, clusters=clusters)
        except ValueError as e:
            typer.echo(f"ERROR: {e}")
            raise typer.Exit(1)

        mode = store.get_sync("rater_profile_mode") or "unknown"

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    n_scored = int((result["insufficient"] == 0).sum())
    n_insufficient = int((result["insufficient"] == 1).sum())
    n_sybil = int(result["sybil_flag"].sum())
    typer.echo(
        f"scored {n_scored} ratee(s), {n_insufficient} insufficient, {n_sybil} sybil-flagged "
        f"-> {out} (rater profile mode: {mode})")


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


def _provenance(mode: str, cfg: Config) -> dict:
    config = dict(
        bootstrap_n=cfg.bootstrap_n,
        bootstrap_seed=cfg.bootstrap_seed,
        min_clusters=cfg.min_clusters,
        evidence_weights=list(cfg.evidence_weights),
        sybil_jaccard=cfg.sybil_jaccard,
        sybil_window_s=cfg.sybil_window_s,
        sybil_max_group=cfg.sybil_max_group,
        sybil_flag_share=cfg.sybil_flag_share,
    )
    return dict(
        rater_profile_mode=mode,
        confirmations=cfg.confirmations,
        config=config,
        versions={"robustrep": __version__, "numpy": numpy.__version__, "pandas": pandas.__version__},
    )


def _write_report(target: Path, result, records, clusters, sens, adv, block: int, provenance: dict,
                  top_n: int) -> None:
    """Draw all 5 figures and write report.md + scores.json into `target`."""
    target.mkdir(parents=True, exist_ok=True)
    fig_mean_vs_robust(result, out=target / _FIGURES["Fig 1. Mean vs robust score"])
    fig_rank_shift(result, out=target / _FIGURES["Fig 2. Biggest rank drops"], top_n=top_n)
    fig_evidence(records, out=target / _FIGURES["Fig 3. Evidence levels"])
    fig_sybil_clusters(records, clusters, out=target / _FIGURES["Fig 4. Largest rater clusters"])
    fig_sensitivity(sens, out=target / _FIGURES["Fig 5. Ranking stability"])
    md = render_markdown(result, records, block=block, figures=_FIGURES, sensitivity=sens,
                         adversarial=adv, provenance=provenance)
    (target / "report.md").write_text(md)
    export_json(result, block=block, out=target / "scores.json", config=provenance.get("config"))


def _publish_latest(out_dir: Path, block_dir: Path) -> None:
    """Publish a copy of `block_dir` as `out_dir/latest/`.

    Near-atomic: the old `latest/` is removed then the staged copy is moved in
    -- a reader may briefly see no `latest/` at all, in the gap between the
    removal and the move. Builds the new contents in a temp directory first, so
    that gap is as short as a single `os.replace` (moving the fully-staged
    directory into place, no partial writes ever visible) rather than however
    long the figures/report.md/scores.json themselves take to generate; stale
    files from an earlier run never linger alongside the new ones.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest"
    tmp_parent = Path(tempfile.mkdtemp(dir=out_dir))
    try:
        staged = tmp_parent / "latest"
        shutil.copytree(block_dir, staged)
        if latest.exists():
            shutil.rmtree(latest)
        os.replace(staged, latest)
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)


@app.command()
def report(
    db: Path = typer.Option(..., help="SQLite store path."),
    out_dir: Path = typer.Option(Path("reports"), help="Output directory."),
    bootstrap_n: int = typer.Option(
        1000, min=0, help="Bootstrap resamples for the confidence interval."),
    top_n: int = typer.Option(
        100, min=1, help="Top-N ratees (by naive mean / robust score) considered for the "
                         "rank-shift and sensitivity figures."),
) -> None:
    """Generate figures 1-5, report.md and scores.json under out_dir/<block>/,
    then atomically publish a copy to out_dir/latest/ (a fixed link always has
    the latest report while every past run stays addressable by block number).

    Exits 1 with "no records" if the store has no feedback rows yet (run
    fetch first), or with an ERROR line if the given options or the data
    itself is rejected (e.g. too many candidate sybil pairs, or malformed
    records).
    """
    try:
        cfg = Config(bootstrap_n=bootstrap_n)
    except ValueError as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)

    try:
        with Store(db) as store:
            records = store.load_records()
            if records.empty:
                typer.echo("no records")
                raise typer.Exit(1)
            meta = store.load_rater_meta()
            clusters = cluster_raters(profiles_from_records(records, meta), cfg)
            result = score_fn(records, cfg, clusters=clusters)
            mode = store.get_sync("rater_profile_mode") or "unknown"
            last_block = store.get_sync("last_block")
    except ValueError as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)
    block = int(last_block) if last_block is not None else 0

    sens = sensitivity_table(records, cfg, meta=meta, top_n=top_n)
    adv = scenario_table()
    provenance = _provenance(mode, cfg)

    block_dir = out_dir / str(block)
    _write_report(block_dir, result, records, clusters, sens, adv, block, provenance, top_n)
    _publish_latest(out_dir, block_dir)

    typer.echo(f"report written to {block_dir} and {out_dir / 'latest'} "
              f"(rater profile mode: {mode})")


if __name__ == "__main__":
    app()
