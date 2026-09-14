"""Cluster raters that look like one actor. Two of three signals => same cluster.

Signals, compared pairwise: (1) same non-null funder (case-insensitive), (2)
first-seen within `Config.sybil_window_s` seconds, (3) Jaccard(ratees) >=
`Config.sybil_jaccard` (requires `sybil_jaccard > 0`, enforced by Config, so a
shared ratee is necessary for signal 3 to ever fire).

Candidate pairs come from blocking (same funder; or sharing a ratee with at
most `Config.sybil_max_group` raters), each pruned to avoid an O(n^2) blowup
-- see `_candidate_pairs`. Pairs may repeat across blocks and are not
de-duplicated (union-find unions are idempotent; de-duplicating would need an
unbounded `seen` set). `Config.sybil_max_pairs` is a global budget:
`cluster_raters` fails fast with a ValueError instead of grinding through an
adversarial input.

Limitations: a ratee with more than `sybil_max_group` raters is not used for
ratee-blocking at all, so a large sybil farm that only rates one popular
agent, while varying funders and spreading out first-seen timestamps, will
not be clustered. The report must state this limitation explicitly.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

_META_COLUMNS = {"rater", "first_seen_ts", "funder"}


@dataclass(frozen=True)
class RaterProfile:
    """A rater's identity fingerprint: earliest `first_seen_ts` (seconds), `funder`
    address (None if unknown; compared case-insensitively), and `ratees` rated."""

    rater: str
    first_seen_ts: int
    funder: Optional[str]
    ratees: frozenset


class _UnionFind:
    """Union-find (disjoint set) over an arbitrary hashable universe of items."""

    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two sets; 0.0 when both are empty."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _norm_funder(funder: Optional[str]) -> Optional[str]:
    return funder.lower() if funder is not None else None


def _signals(p: RaterProfile, q: RaterProfile, cfg: Config) -> int:
    """Count of the 3 sybil signals holding between two profiles (0-3).

    Short-circuits to 2 once funder+window both match, skipping Jaccard --
    callers only care whether the count is >= 2.
    """
    pf, qf = _norm_funder(p.funder), _norm_funder(q.funder)
    same_funder = pf is not None and pf == qf
    in_window = abs(p.first_seen_ts - q.first_seen_ts) <= cfg.sybil_window_s
    if same_funder and in_window:
        return 2
    similar = _jaccard(p.ratees, q.ratees) >= cfg.sybil_jaccard
    return int(same_funder) + int(in_window) + int(similar)


def _pair_key(p: RaterProfile, q: RaterProfile) -> tuple[str, str]:
    a, b = p.rater, q.rater
    return (a, b) if a < b else (b, a)


def _funder_group_pairs(group: list[RaterProfile], cfg: Config):
    # same_funder + in_window is already 2 signals, so every in-window pair
    # unions. Unioning consecutive in-window neighbours (sorted by
    # first_seen_ts) yields identical connected components by transitivity:
    # if a and c are within the window of each other, every b between them
    # in sorted order is within the window of both. Same-funder pairs
    # OUTSIDE the window can only reach 2 signals via Jaccard, which
    # requires a shared ratee -- covered by `_ratee_group_pairs`. O(n) per
    # group, no blowup regardless of group size.
    ordered = sorted(group, key=lambda p: p.first_seen_ts)
    for p, q in zip(ordered, ordered[1:]):
        if q.first_seen_ts - p.first_seen_ts <= cfg.sybil_window_s:
            yield _pair_key(p, q)


def _ratee_group_pairs(ratee: str, group: list[RaterProfile], cfg: Config, small: set):
    # Skip ratees with more than sybil_max_group raters entirely (see module
    # "Limitations"). Without a shared funder, a pair needs in-window +
    # Jaccard, so out-of-window pairs are pruned via a sorted sliding window
    # (break at the first out-of-window j; every later j is further still).
    # A pair that DOES share a funder only needs Jaccard (funder + Jaccard
    # is already 2 signals, no window requirement), so those come from a
    # separate same-funder sub-bucket below, still bounded by the same cap.
    if len(group) > cfg.sybil_max_group:
        return
    ordered = sorted(group, key=lambda p: p.first_seen_ts)
    n = len(ordered)
    funders = {_norm_funder(p.funder) for p in group}
    # If every rater sharing this ratee has the SAME funder, no window-sliding
    # pair below could ever pass the funder-exclusion check (every pair is
    # same-funder), so skip the O(n^2)-worst-case double loop entirely rather
    # than iterate it just to discard everything.
    if len(funders) > 1 or None in funders:
        for i in range(n):
            for j in range(i + 1, n):
                if ordered[j].first_seen_ts - ordered[i].first_seen_ts > cfg.sybil_window_s:
                    break
                pi, pj = ordered[i], ordered[j]
                # Same-funder + in-window pairs are already fully covered
                # (and correctly transitively closed) by `_funder_group_pairs`,
                # so skip them here rather than re-emitting once per ratee.
                if _norm_funder(pi.funder) is None or _norm_funder(pi.funder) != _norm_funder(pj.funder):
                    yield _pair_key(pi, pj)
    by_funder: dict = defaultdict(list)
    for p in group:
        if p.funder is not None:
            by_funder[_norm_funder(p.funder)].append(p)
    for sub in by_funder.values():
        sub_by_ts = sorted(sub, key=lambda p: p.first_seen_ts)
        # If the whole sub-bucket's timestamps fit in one window, every pair
        # in it is in-window and thus already covered by `_funder_group_pairs`
        # (same argument as above) -- skip the O(k^2) loop below entirely.
        if sub_by_ts[-1].first_seen_ts - sub_by_ts[0].first_seen_ts <= cfg.sybil_window_s:
            continue
        for p, q in combinations(sub, 2):
            if abs(p.first_seen_ts - q.first_seen_ts) <= cfg.sybil_window_s:
                continue
            # A same-funder, out-of-window pair may share several small
            # ratees, and this sub-bucket runs once per ratee -- naively
            # yielding here would regenerate the same pair once per shared
            # ratee (a farm sharing k small ratees inflates candidate pairs
            # ~k-fold, enough to blow the sybil_max_pairs budget). Emit it
            # exactly once, at the lexicographically smallest of its shared
            # small ratees.
            shared = p.ratees & q.ratees & small
            if shared and min(shared) == ratee:
                yield _pair_key(p, q)


def _candidate_pairs(profiles: list[RaterProfile], cfg: Config):
    """Candidate (rater, rater) pairs to test, via funder- and ratee-blocking.

    Pairs may repeat across blocks (e.g. same funder and multiple shared
    ratees); callers must tolerate duplicates (union-find unions are idempotent).
    """
    by_funder: dict = defaultdict(list)
    by_ratee: dict = defaultdict(list)
    for p in profiles:
        if p.funder is not None:
            by_funder[_norm_funder(p.funder)].append(p)
        for r in p.ratees:
            by_ratee[r].append(p)
    small = {ratee for ratee, group in by_ratee.items() if len(group) <= cfg.sybil_max_group}
    for group in by_funder.values():
        yield from _funder_group_pairs(group, cfg)
    for ratee, group in by_ratee.items():
        yield from _ratee_group_pairs(ratee, group, cfg, small)


def cluster_raters(profiles: list[RaterProfile], cfg: Config) -> dict[str, str]:
    """Cluster raters whose profiles share at least 2 of 3 sybil signals.

    Returns rater -> cluster id, the lexicographically smallest rater in the
    cluster. Raises ValueError on duplicate rater names, or once candidate-pair
    generation exceeds `cfg.sybil_max_pairs` (fail fast on adversarial input).
    """
    by_id: dict[str, RaterProfile] = {}
    for p in profiles:
        if p.rater in by_id:
            raise ValueError(f"duplicate rater in profiles: {p.rater!r}")
        by_id[p.rater] = p
    uf = _UnionFind(by_id.keys())
    pair_count = 0
    for a, b in _candidate_pairs(profiles, cfg):
        pair_count += 1
        if pair_count > cfg.sybil_max_pairs:
            raise ValueError(
                f"sybil candidate pairs exceed sybil_max_pairs={cfg.sybil_max_pairs}; "
                "raise sybil_max_pairs first; lowering sybil_max_group also helps but "
                "trades sybil-detection recall for speed"
            )
        if _signals(by_id[a], by_id[b], cfg) >= 2:
            uf.union(a, b)
    logger.info("sybil candidate pairs: %d", pair_count)
    return {r: uf.find(r) for r in by_id}


def _validate_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Validate a `meta` frame (columns rater, first_seen_ts, funder); returns a new copy."""
    missing_cols = _META_COLUMNS - set(meta.columns)
    if missing_cols:
        raise ValueError(f"meta missing columns: {sorted(missing_cols)}")
    dup = sorted(meta.loc[meta["rater"].duplicated(), "rater"].unique().tolist())
    if dup:
        raise ValueError(f"meta has duplicate raters: {dup}")
    out = meta.copy()
    try:
        out["first_seen_ts"] = pd.to_numeric(out["first_seen_ts"], errors="raise")
    except (ValueError, TypeError) as e:
        raise ValueError(f"meta.first_seen_ts: {e}") from e
    ts = out["first_seen_ts"].dropna()
    if (ts < 0).any():
        raise ValueError("meta.first_seen_ts: negative values not allowed")
    if (ts % 1 != 0).any():
        raise ValueError("meta.first_seen_ts: non-integer values not allowed")
    return out


def profiles_from_records(records: pd.DataFrame, meta: Optional[pd.DataFrame] = None) -> list[RaterProfile]:
    """Build one RaterProfile per rater from a validated record frame.

    first_seen_ts defaults to the minimum `ts` observed for that rater; funder
    defaults to None. `meta` (columns rater, first_seen_ts, funder), when
    given, is validated (see `_validate_meta`) and its non-null values override
    the defaults for raters it covers; raters absent from `meta`, and `meta`
    rows for raters absent from `records`, are ignored. funder is stored as
    given; `cluster_raters` compares it case-insensitively.
    """
    g = records.groupby("rater")
    first = g["ts"].min()
    ratees = g["ratee"].agg(lambda s: frozenset(s))
    m = _validate_meta(meta).set_index("rater") if meta is not None else None
    out = []
    for r in first.index:
        ts, funder = int(first[r]), None
        if m is not None and r in m.index:
            row = m.loc[r]
            if pd.notna(row["first_seen_ts"]):
                ts = int(row["first_seen_ts"])
            if pd.notna(row["funder"]):
                funder = str(row["funder"])
        out.append(RaterProfile(rater=r, first_seen_ts=ts, funder=funder, ratees=ratees[r]))
    return out
