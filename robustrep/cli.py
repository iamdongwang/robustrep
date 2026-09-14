"""``robustrep fetch`` / ``robustrep score`` command-line entry points.

Kept intentionally thin: this module only orchestrates calls into the
library (``robustrep.sources.*``, ``robustrep.pipeline``, ``robustrep.sybil``,
``robustrep.store``) and never re-implements their business logic. ``fetch``'s
pipeline is broken into small private ``_step_*`` helpers (one per stage:
sync, block timestamps, agent owners, rater profiles, evidence) that each
return a one-line summary string -- this lets tests monkeypatch the
``base``/``enrich_raters``/``classify_all`` symbols in this module and drive
the whole command end-to-end without any network access.

``report`` (the third subcommand) is added in a later task.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import typer

from .config import Config
from .pipeline import score as score_fn
from .sources import base_erc8004 as base
from .sources.evidence_fetch import classify_all
from .sources.rater_profile import EtherscanClient, enrich_raters, estimate_seconds
from .sources.rpc import RpcClient
from .store import Store
from .sybil import cluster_raters, profiles_from_records

app = typer.Typer(add_completion=False, help="Robust, transport-agnostic reputation scoring for AI agents.")

logger = logging.getLogger(__name__)

_ETHERSCAN_RPS = 4.0


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", help="Enable INFO-level logging.")) -> None:
    """Robust, transport-agnostic reputation scoring for AI agents."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, force=True)


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


def _step_owners(store: Store, rpc: RpcClient) -> str:
    """Resolve the IdentityRegistry owner for every not-yet-resolved agent id."""
    agents = store.distinct_agents()
    resolved = 0
    for agent_id in agents:
        owner = base.owner_of(rpc, agent_id)
        if owner:
            store.upsert_agent_owner(agent_id, owner)
            resolved += 1
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
    if client is not None:
        n_targets = len(store.distinct_clients())
        if n_targets:
            eta_h = estimate_seconds(n_targets, _ETHERSCAN_RPS) / 3600
            typer.echo(f"profiling {n_targets} rater(s) via Etherscan, ETA ~{eta_h:.1f}h")
    n = enrich_raters(store, client)
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
    chunk: int = typer.Option(Config().chunk_blocks, help="Blocks per eth_getLogs call."),
    confirmations: int = typer.Option(
        Config().confirmations, help="Blocks to lag behind the chain head before ingesting."),
    rpc_url: List[str] = typer.Option(
        [], "--rpc-url", help="RPC endpoint (repeatable); overrides the default endpoint list."),
    etherscan_key: Optional[str] = typer.Option(
        None, "--etherscan-key", help="Etherscan API key; overrides ETHERSCAN_API_KEY."),
    skip_raters: bool = typer.Option(False, "--skip-raters", help="Skip rater profile enrichment."),
    skip_evidence: bool = typer.Option(False, "--skip-evidence", help="Skip evidence URI classification."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be synced and exit, without any network calls."),
) -> None:
    """Sync Base ERC-8004 feedback, block timestamps, agent owners, rater
    profiles and evidence levels into the SQLite store at ``--db``.

    Exits 0 on full success, 1 for a bad-input/no-op error and 2 when rater
    profiling partially failed (some addresses errored, the rest -- and every
    other step -- still completed; see ``_step_raters``)."""
    cfg = Config(chunk_blocks=chunk, confirmations=confirmations,
                 rpc_urls=tuple(rpc_url) if rpc_url else Config().rpc_urls)
    store = Store(db)
    if dry_run:
        last = store.get_sync("last_block")
        start = int(last) + 1 if last is not None else base.DEPLOY_BLOCK
        end = to_block if to_block is not None else "head"
        typer.echo(f"would sync from block {start} to {end}")
        raise typer.Exit(0)

    rpc = RpcClient(cfg.rpc_urls, user_agent=cfg.user_agent)
    partial_failure = False

    typer.echo(_step_sync(store, rpc, cfg, to_block))
    typer.echo(_step_timestamps(store, rpc))
    typer.echo(_step_owners(store, rpc))

    if skip_raters:
        mode = store.get_sync("rater_profile_mode") or "unknown"
        typer.echo(f"raters profiling skipped; rater profile mode: {mode}")
    else:
        try:
            typer.echo(_step_raters(store, etherscan_key))
        except RuntimeError as e:
            typer.echo(f"WARNING: {e}")
            partial_failure = True

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
    bootstrap_n: int = typer.Option(Config().bootstrap_n, help="Bootstrap resamples for the confidence interval."),
    min_clusters: int = typer.Option(
        Config().min_clusters, help="Minimum distinct rater clusters required to score a ratee."),
) -> None:
    """Compute robust reputation scores for every ratee in the store and
    write them to ``--out`` as a CSV with ``RESULT_COLUMNS`` columns.

    Exits 1 with "no records" if the store has no feedback rows yet (run
    ``fetch`` first)."""
    cfg = Config(bootstrap_n=bootstrap_n, min_clusters=min_clusters)
    store = Store(db)
    records = store.load_records()
    if records.empty:
        typer.echo("no records")
        raise typer.Exit(1)

    clusters = cluster_raters(profiles_from_records(records, store.load_rater_meta()), cfg)
    result = score_fn(records, cfg, clusters=clusters)
    result.to_csv(out, index=False)

    n_scored = int((result["insufficient"] == 0).sum())
    n_insufficient = int((result["insufficient"] == 1).sum())
    n_sybil = int(result["sybil_flag"].sum())
    mode = store.get_sync("rater_profile_mode") or "unknown"
    typer.echo(
        f"scored {n_scored} ratee(s), {n_insufficient} insufficient, {n_sybil} sybil-flagged "
        f"-> {out} (rater profile mode: {mode})")


if __name__ == "__main__":
    app()
