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

import enum
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
from .report.render import evidence_level_shares, render_markdown
from .report.sensitivity import fig_sensitivity, sensitivity_table
from .sources import base_erc8004 as base
from .sources.evidence_fetch import classify_all
from .sources.rater_profile import DEFAULT_RPS, EtherscanClient, default_client, enrich_raters, estimate_seconds
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


class ProfileSource(str, enum.Enum):
    """``--profile-source`` choices -- see ``rater_profile.default_client``
    for what each one builds. ``str`` mixin so typer renders/parses plain
    strings (and so a raw ``"auto"``/etc. from a test compares equal)."""

    auto = "auto"
    blockscout = "blockscout"
    etherscan = "etherscan"
    none = "none"


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


def _step_timestamps(store: Store, rpc: RpcClient, batch_state: Optional[base.BatchState] = None) -> str:
    """Resolve block timestamps for every block referenced by feedback rows."""
    n = base.fill_block_timestamps(store, rpc, state=batch_state)
    still_missing = store.n_missing_block_ts()
    if still_missing > 0:
        typer.echo(f"WARNING: {still_missing} block(s) still missing a cached timestamp")
    return f"block timestamps filled: {n}"


def _step_owners(store: Store, rpc: RpcClient, log_every: int = OWNER_LOG_EVERY,
                  batch_size: int = OWNER_BATCH_DEFAULT, batch_state: Optional[base.BatchState] = None) -> str:
    """Resolve the IdentityRegistry owner for every not-yet-resolved agent id.

    Owners are resolved ``batch_size`` at a time via ``base.owners_of`` (one
    JSON-RPC batch call per chunk instead of one ``eth_call`` per agent --
    ``owners_of`` itself falls back to per-agent calls if a chunk's batch
    fails). Every chunk shares one ``base.BatchState`` (``batch_state``, or a
    fresh one if not given) so that once any chunk's batch fails with an
    ``RpcBatchUnsupportedError`` (batching itself doesn't work against this
    endpoint), every later chunk skips straight to per-agent calls instead of
    re-attempting an identical, doomed batch call. Every agent id is
    recorded, even when no owner resolves -- as an empty string
    (``Store.upsert_agent_owner`` accepts one; the evidence classification
    join already filters falsy owners out of its parties set) -- so an
    unresolved agent is never re-queried on a later run. Logs progress at
    INFO every ``log_every`` agents (not every chunk).
    """
    agents = store.distinct_agents()
    resolved = 0
    batch_state = batch_state or base.BatchState()
    for i in range(0, len(agents), batch_size):
        chunk = agents[i:i + batch_size]
        owners = base.owners_of(rpc, chunk, batch_size=batch_size, state=batch_state)
        for j, agent_id in enumerate(chunk):
            owner = owners.get(agent_id)
            store.upsert_agent_owner(agent_id, owner or "")
            if owner:
                resolved += 1
            n_done = i + j + 1
            if log_every > 0 and n_done % log_every == 0:
                logger.info("fetch: agent owners resolved %d/%d", n_done, len(agents))
    return f"agent owners resolved: {resolved}/{len(agents)}"


def _step_raters(store: Store, etherscan_key: Optional[str], profile_source: str = ProfileSource.auto,
                  profile_rps: Optional[float] = None, profile_retries: Optional[int] = None) -> str:
    """Profile not-yet-profiled raters via ``profile_source``, or fall back
    offline (``profile_source="none"``).

    ``etherscan_key``, when given, overrides ``ETHERSCAN_API_KEY`` (the key
    itself is never echoed). ``profile_source`` is one of ``auto`` (default:
    Blockscout -- free, no key -- unless a key is available, in which case
    Etherscan V2), ``blockscout``, ``etherscan`` (requires a key) or ``none``
    (offline fallback). See ``rater_profile.default_client``. Raises
    ``RuntimeError`` (from ``enrich_raters``) when some -- but not all --
    addresses failed, or when the very first one hit a non-retryable
    plan-configuration error; the caller decides how to handle it. Raises
    ``ValueError`` when ``profile_source="etherscan"`` but no key is
    available.

    ``profile_rps``/``profile_retries``, when given (not ``None``), override
    the chosen client's own default request rate / retry count -- see
    ``rater_profile.default_client``. The ETA estimate printed before
    profiling starts uses the *effective* rps -- the client's actual
    configured rate (``1 / client.gap``), which is ``profile_rps`` when
    given, else whatever the client itself defaulted to.
    """
    key = etherscan_key if etherscan_key is not None else os.environ.get("ETHERSCAN_API_KEY")
    # `profile_source` may be a ProfileSource (from the `fetch` CLI option) or
    # a plain string (direct callers, tests) -- normalize via `.value` rather
    # than `str(...)`, which on a `(str, Enum)` member yields "ProfileSource.
    # auto" rather than "auto".
    source = profile_source.value if isinstance(profile_source, ProfileSource) else profile_source
    client = default_client(source, key, rps=profile_rps, retries=profile_retries)
    # store.distinct_clients() computed exactly once here and handed to
    # enrich_raters(addresses=...) below, instead of letting it re-query the
    # store itself.
    targets = store.distinct_clients()
    if client is not None and targets:
        effective_rps = 1.0 / client.gap
        eta_h = estimate_seconds(len(targets), effective_rps) / 3600
        source_label = getattr(client, "source", "etherscan").capitalize()
        typer.echo(f"profiling {len(targets)} rater(s) via {source_label}, ETA ~{eta_h:.1f}h")
    n = enrich_raters(store, client, addresses=targets)
    mode = store.get_sync("rater_profile_mode") or "unknown"
    fallback_note = "" if client is not None else " (fallback: --profile-source none; rows not written)"
    return f"raters profiled: {n}{fallback_note}; rater profile mode: {mode}"


def _step_evidence(store: Store, rpc: RpcClient, workers: int) -> str:
    """Classify every not-yet-cached evidence URI referenced by feedback rows,
    fetching+classifying up to ``workers`` URIs concurrently."""
    n = classify_all(store, tx_parties=lambda h: base.tx_parties(rpc, h), workers=workers)
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
    profile_source: ProfileSource = typer.Option(
        ProfileSource.auto, "--profile-source",
        help="Rater-profile HTTP source: 'auto' (default) uses Blockscout (free, no key) unless an "
             "Etherscan key is available, in which case Etherscan V2; 'blockscout' always uses Base's "
             "free Blockscout instance; 'etherscan' always uses Etherscan V2 (requires a key -- note "
             "Etherscan's free plan does not cover Base); 'none' skips profiling (offline fallback)."),
    profile_rps: Optional[float] = typer.Option(
        None, "--profile-rps", min=0.1,
        help="Requests/sec against the rater-profile HTTP source (default: the chosen client's own "
             "default -- 1.0 for Blockscout, 4.0 for Etherscan)."),
    profile_retries: Optional[int] = typer.Option(
        None, "--profile-retries", min=1,
        help="Retry attempts for a transient rater-profile HTTP failure, per address (default: the "
             "chosen client's own default -- 5 for Blockscout, 3 for Etherscan)."),
    skip_evidence: bool = typer.Option(False, "--skip-evidence", help="Skip evidence URI classification."),
    evidence_workers: int = typer.Option(
        8, "--evidence-workers", min=1,
        help="Concurrent worker threads used to fetch+classify evidence URIs."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be synced and exit, without any network calls."),
) -> None:
    """Sync Base ERC-8004 feedback, block timestamps, agent owners, rater
    profiles and evidence levels into the SQLite store at --db.

    Exits 0 on full success; 1 for a bad-input error (an invalid option
    combination that Config itself rejects, or --profile-source etherscan
    with no key available); 2 when rater profiling partially failed (some
    addresses errored, the rest -- and every other step -- still completed;
    see _step_raters); 3 when an RPC call failed on every retry.
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
    # Shared across every batch-capable step of this run: once one step's
    # batch calls are found to fail structurally (or get batch-rate-limited),
    # every later step also skips straight to per-item calls instead of
    # re-discovering the same doomed batch failure from scratch.
    batch_state = base.BatchState()

    with Store(db) as store:
        with _rpc_guard():
            typer.echo(_step_sync(store, rpc, cfg, to_block))
            typer.echo(_step_timestamps(store, rpc, batch_state=batch_state))
            if skip_owners:
                typer.echo("agent owner resolution skipped")
            else:
                typer.echo(_step_owners(store, rpc, batch_size=owner_batch, batch_state=batch_state))

        if skip_raters:
            mode = store.get_sync("rater_profile_mode") or "unknown"
            typer.echo(f"raters profiling skipped; rater profile mode: {mode}")
        else:
            try:
                typer.echo(_step_raters(store, etherscan_key, profile_source=profile_source,
                                        profile_rps=profile_rps, profile_retries=profile_retries))
            except ValueError as e:
                typer.echo(f"ERROR: {e}")
                raise typer.Exit(1)
            except RuntimeError as e:
                typer.echo(f"WARNING: {e}")
                partial_failure = True

        with _rpc_guard():
            if skip_evidence:
                typer.echo("evidence classification skipped")
            else:
                typer.echo(_step_evidence(store, rpc, workers=evidence_workers))

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


def _provenance(mode: str, cfg: Config, records: pandas.DataFrame,
                result: Optional[pandas.DataFrame] = None) -> dict:
    config = dict(
        bootstrap_n=cfg.bootstrap_n,
        bootstrap_seed=cfg.bootstrap_seed,
        min_clusters=cfg.min_clusters,
        evidence_weights=list(cfg.evidence_weights),
        sybil_jaccard=cfg.sybil_jaccard,
        sybil_window_s=cfg.sybil_window_s,
        sybil_max_group=cfg.sybil_max_group,
        sybil_flag_share=cfg.sybil_flag_share,
        norm_fit_share=cfg.norm_fit_share,
        norm_rule_counts=_norm_rule_counts_for_provenance(result),
        evidence_level_shares=_evidence_shares_for_provenance(records),
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
    provenance = _provenance(mode, cfg, records, result)

    block_dir = out_dir / str(block)
    _write_report(block_dir, result, records, clusters, sens, adv, block, provenance, top_n)
    _publish_latest(out_dir, block_dir)

    typer.echo(f"report written to {block_dir} and {out_dir / 'latest'} "
              f"(rater profile mode: {mode})")


if __name__ == "__main__":
    app()
