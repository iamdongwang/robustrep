"""Cluster raters that look like one actor. Two of three signals => same cluster.

Signals, compared pairwise: (1) same non-null funder (case-insensitive), (2)
first-seen within `Config.sybil_window_s` seconds, (3) Jaccard(ratees) >=
`Config.sybil_jaccard` (requires `sybil_jaccard > 0`, enforced by Config, so a
shared ratee is necessary for signal 3 to ever fire).

Candidate pairs come from blocking (same funder; or sharing a ratee with at
most `Config.sybil_max_group` raters), each pruned to avoid an O(n^2) blowup
-- see `_candidate_pairs`. Pairs may repeat across blocks and are not
de-duplicated (union-find unions are idempotent; de-duplicating would need an
unbounded `seen` set). `Config.sybil_max_pairs` (global) and
`Config.sybil_max_pairs_per_ratee` (one ratee block) bound that work. Both
count candidate-pair EXAMINATIONS, not just the pairs that survive to be
emitted: a block can scan O(k^2) pairs and emit none of them (the same-funder
de-dup emits a pair only at the lexicographically smallest of its shared
ratees, so m-1 of m identical blocks scan in full and yield nothing), so
counting emitted pairs would have bounded the output while leaving the CPU
cost of the scan unbounded in m, the number of ratees a farm rates. Charging
examinations bounds the whole of candidate generation at `sybil_max_pairs`
examinations, whatever shape the input takes; each examination is O(1) except
for the shared-ratee de-dup, which scans a rater's sorted blockable ratees
only until the answer is settled (see `_emits_at`) -- one lookup for the farm
shapes that matter, O(m) only if the shared ratee sorts last.

Blocks are visited in a fixed order (funders by name; then ratee blocks
largest first, ties by name), never in `dict` insertion order, which for ratee
blocks follows the iteration order of a `frozenset` of ratee names and so
varies with PYTHONHASHSEED. Order is irrelevant to a run that tests every
pair, but under truncation it decides which clusters are found at all, so an
unstable order would mean the same data clustering differently in different
processes.

Exceeding a budget DEGRADES the clustering -- generation stops there and the
run continues with the clusters found so far -- it never aborts. Failing fast
was the original design and it was a denial of service: the budget is reached
on attacker-chosen input (H2 -- ~2,000 rater addresses x 3 feedback events on
the same 3 ratees inside one 24h window, roughly 6,000 cheap transactions),
and ERC-8004 feedback is on-chain forever, so a ValueError there would have
broken `score` and `report` for the WHOLE dataset, permanently, for everyone,
with no option to raise the budget. Degrading instead can only UNDER-merge
(a pair never tested is never collapsed, and no merge is ever invented), so
the affected ratees read as better-supported than they are -- which is why
every skip is counted in `ClusterStats` and stated in the report rather than
passed over in silence.

Limitations: a ratee with more than `sybil_max_group` raters is not used for
ratee-blocking at all, so a large sybil farm that only rates one popular
agent, while varying funders and spreading out first-seen timestamps, will
not be clustered. A ratee block cut short by `sybil_max_pairs_per_ratee`, or
any block left ungenerated once `sybil_max_pairs` is reached, is under-merged
the same way. The report must state these limitations explicitly, with the
counts from `ClusterStats`.
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


@dataclass(frozen=True)
class ClusterStats:
    """What one `cluster_raters_with_stats` run actually managed to test.

    `pairs_examined` candidate pairs were charged against the budgets and
    `pairs_tested` of them survived blocking to reach the signal test (emitted
    pairs are a subset of examined ones -- a block can examine a pair and drop
    it, e.g. as out-of-window, without ever emitting it); `ratees_skipped_size`
    ratee blocks were skipped whole for holding more than
    `Config.sybil_max_group` raters; `ratees_skipped_budget` blocks were
    abandoned part-way for exceeding `Config.sybil_max_pairs_per_ratee`; and
    `truncated` says the global `Config.sybil_max_pairs` budget stopped
    generation early. Any non-zero SKIP count, or `truncated`, means some
    raters may be under-merged (`pairs_tested` is non-zero on a healthy run too,
    so it says nothing on its own) -- see the module docstring, and
    `robustrep.report.render`, which prints these numbers as a report
    limitation.
    """

    pairs_tested: int
    pairs_examined: int
    ratees_skipped_size: int
    ratees_skipped_budget: int
    truncated: bool

    @property
    def budget_limited(self) -> bool:
        """True when a budget or the group cap cost this run some merges.

        `robustrep.report.render` mirrors this test on the `dataclasses.asdict`
        form the CLI stamps on the scored frame; keep the two in step.
        """
        return bool(self.ratees_skipped_size or self.ratees_skipped_budget or self.truncated)


@dataclass
class _Budget:
    """Run-wide budget of candidate-pair EXAMINATIONS, plus what was skipped.

    Mutable and passed in from outside (rather than counted inside the
    generators) so the caller can read back what was spent once generation
    ends, without the generators having to yield anything but pairs. There is
    deliberately no per-block counter here: it belongs to `_BlockCharge`, one
    per block, so one block's spend can never leak into the next.
    """

    max_total: int
    max_per_ratee: int
    total: int = 0
    truncated: bool = False
    ratees_skipped_size: int = 0
    ratees_skipped_budget: int = 0

    @classmethod
    def for_config(cls, cfg: Config) -> "_Budget":
        return cls(max_total=cfg.sybil_max_pairs, max_per_ratee=cfg.sybil_max_pairs_per_ratee)


class _BlockCharge:
    """One block's charge against a run `_Budget`. `examined` is local to it.

    `spend()` is called once per candidate pair EXAMINED -- not once per pair
    yielded, and including the comparison that ends a sliding-window scan --
    because the examination is what costs CPU, and a block can examine O(k^2)
    pairs while yielding none of them: the same-funder de-dup below emits a
    pair only at the lexicographically smallest of its shared ratees, so m-1
    of m identical blocks scan in full and emit nothing. Charging emitted
    pairs alone left exactly that scan unbounded (measured: 2,000 same-funder
    raters over 50 shared ratees took 105s with every ClusterStats counter
    still reading zero). Charging examinations bounds the whole of candidate
    generation at `sybil_max_pairs` examinations, whatever the input shape.

    `spend()` returns False once this block must stop, and records WHY exactly
    once -- `stopped` keeps the count honest when a block's second inner
    generator asks again after the first one gave up.
    """

    def __init__(self, budget: _Budget, ratee: Optional[str] = None) -> None:
        self.budget = budget
        self.ratee = ratee
        self.examined = 0
        self.stopped = False

    def spend(self) -> bool:
        if self.stopped:
            return False
        if self.budget.total >= self.budget.max_total:
            self.budget.truncated = True
            self.stopped = True
            return False
        if self.ratee is not None and self.examined >= self.budget.max_per_ratee:
            self.budget.ratees_skipped_budget += 1
            self.stopped = True
            logger.warning(
                "sybil: ratee %r abandoned after %d candidate pairs examined "
                "(sybil_max_pairs_per_ratee=%d); its raters may be under-merged",
                self.ratee, self.examined, self.budget.max_per_ratee)
            return False
        self.budget.total += 1
        self.examined += 1
        return True


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


def _small_ratees_by_rater(profiles: list[RaterProfile], small: set) -> dict:
    """rater -> its blockable ("small") ratees, sorted ascending.

    Built once per run so the same-funder de-dup below can answer "is this
    block's ratee the smallest ratee the pair shares?" by scanning until the
    answer is settled, instead of intersecting two ratee sets and taking a
    `min` over the result for every examined pair. Profiled at 2,000 raters x
    500 shared ratees, those two lines were 60s + 31.5s of a 95.6s run.
    """
    return {p.rater: sorted(p.ratees & small) for p in profiles}


def _emits_at(ordered_small: list, other: frozenset, ratee: str) -> bool:
    """Exactly `min(p.ratees & q.ratees & small) == ratee`, without building
    the intersection.

    `ordered_small` is `sorted(p.ratees & small)` (see `_small_ratees_by_rater`)
    and `other` is `q.ratees`. Scanning ascending settles it at the first entry
    the two share -- that entry IS the minimum -- or at `ratee` itself, since
    nothing after it can sort before it. Both raters of a block rate its ratee
    by construction, so the scan normally stops there; `r in other` is still
    checked on that entry so the result stays exact for any caller.
    """
    for r in ordered_small:
        if r == ratee:
            return r in other
        if r in other:
            return False
    return False


def _funder_group_pairs(group: list[RaterProfile], cfg: Config, charge: _BlockCharge):
    # same_funder + in_window is already 2 signals, so every in-window pair
    # unions. Unioning consecutive in-window neighbours (sorted by
    # first_seen_ts) yields identical connected components by transitivity:
    # if a and c are within the window of each other, every b between them
    # in sorted order is within the window of both. Same-funder pairs
    # OUTSIDE the window can only reach 2 signals via Jaccard, which
    # requires a shared ratee -- covered by `_ratee_group_pairs`. O(n) per
    # group, no blowup regardless of group size; `charge` bounds it anyway,
    # one spend per neighbour pair examined (in-window or not).
    ordered = sorted(group, key=lambda p: p.first_seen_ts)
    for p, q in zip(ordered, ordered[1:]):
        if not charge.spend():
            return
        if q.first_seen_ts - p.first_seen_ts <= cfg.sybil_window_s:
            yield _pair_key(p, q)


def _ratee_window_pairs(group: list[RaterProfile], cfg: Config, charge: _BlockCharge):
    # Without a shared funder, a pair needs in-window + Jaccard, so
    # out-of-window pairs are pruned via a sorted sliding window (break at the
    # first out-of-window j; every later j is further still). A pair that DOES
    # share a funder only needs Jaccard (funder + Jaccard is already 2 signals,
    # no window requirement), so those come from `_ratee_same_funder_pairs`.
    ordered = sorted(group, key=lambda p: p.first_seen_ts)
    funders = {_norm_funder(p.funder) for p in group}
    # If every rater sharing this ratee has the SAME funder, no window-sliding
    # pair below could ever pass the funder-exclusion check (every pair is
    # same-funder), so skip the O(n^2)-worst-case double loop entirely rather
    # than iterate it just to discard everything.
    if not (len(funders) > 1 or None in funders):
        return
    for i in range(len(ordered)):
        pi = ordered[i]
        fi = _norm_funder(pi.funder)  # hoisted: constant for the whole j loop
        for j in range(i + 1, len(ordered)):
            pj = ordered[j]
            # Charged before the window test, not after: the comparison that
            # ends this scan is itself an examination, and a block of entirely
            # out-of-window raters does one per i. Budgets bound examinations,
            # so every one of them has to be paid for.
            if not charge.spend():
                return
            if pj.first_seen_ts - pi.first_seen_ts > cfg.sybil_window_s:
                break
            # Same-funder + in-window pairs are already fully covered (and
            # correctly transitively closed) by `_funder_group_pairs`, so skip
            # them here rather than re-emitting once per ratee.
            if fi is None or fi != _norm_funder(pj.funder):
                yield _pair_key(pi, pj)


def _ratee_same_funder_pairs(ratee: str, group: list[RaterProfile], cfg: Config, small_sorted: dict,
                             charge: _BlockCharge):
    # Same-funder, out-of-window pairs sharing this ratee: funder + Jaccard is
    # already 2 signals, so the window does not apply to them.
    by_funder: dict = defaultdict(list)
    for p in group:
        if p.funder is not None:
            by_funder[_norm_funder(p.funder)].append(p)
    for funder in sorted(by_funder):  # sorted: block order must not vary per run
        sub = by_funder[funder]
        sub_by_ts = sorted(sub, key=lambda p: p.first_seen_ts)
        # If the whole sub-bucket's timestamps fit in one window, every pair
        # in it is in-window and thus already covered by `_funder_group_pairs`
        # (same argument as above) -- skip the O(k^2) loop below entirely.
        if sub_by_ts[-1].first_seen_ts - sub_by_ts[0].first_seen_ts <= cfg.sybil_window_s:
            continue
        for p, q in combinations(sub, 2):
            if not charge.spend():
                return
            if abs(p.first_seen_ts - q.first_seen_ts) <= cfg.sybil_window_s:
                continue
            # A same-funder, out-of-window pair may share several small
            # ratees, and this sub-bucket runs once per ratee -- naively
            # yielding here would regenerate the same pair once per shared
            # ratee (a farm sharing k small ratees inflates candidate pairs
            # ~k-fold, enough to blow the sybil_max_pairs budget). Emit it
            # exactly once, at the lexicographically smallest of its shared
            # small ratees. The m-1 blocks that emit nothing still SCAN this
            # loop in full, which is why `charge` is spent per examination.
            if _emits_at(small_sorted[p.rater], q.ratees, ratee):
                yield _pair_key(p, q)


def _ratee_group_pairs(ratee: str, group: list[RaterProfile], cfg: Config, small_sorted: dict,
                       charge: _BlockCharge):
    """Candidate pairs from one ratee block, bounded by `charge`.

    Callers screen out over-sized groups (see `_candidate_pairs`, which owns
    the `sybil_max_group` check so it can count what it skipped) and supply
    `small_sorted` (see `_small_ratees_by_rater`). Both inner
    generators share one `charge`, so the block's per-ratee budget covers the
    two of them together and the second stops immediately if the first used
    the block up.
    """
    yield from _ratee_window_pairs(group, cfg, charge)
    yield from _ratee_same_funder_pairs(ratee, group, cfg, small_sorted, charge)


def _candidate_pairs(profiles: list[RaterProfile], cfg: Config, budget: Optional[_Budget] = None):
    """Candidate (rater, rater) pairs to test, via funder- and ratee-blocking.

    Pairs may repeat across blocks (e.g. same funder and multiple shared
    ratees); callers must tolerate duplicates (union-find unions are idempotent).

    Bounded by `budget` (a throwaway `_Budget` when not given), which also
    records every block this skips: a ratee block with more than
    `cfg.sybil_max_group` raters is skipped whole (see module "Limitations"),
    one over `cfg.sybil_max_pairs_per_ratee` EXAMINATIONS is abandoned
    part-way, and reaching `cfg.sybil_max_pairs` ends generation. None of them
    raises.

    Blocks are visited in a fixed order -- funders by name, then ratee blocks
    largest first and by name within a size -- never in `dict` insertion order,
    which for ratee blocks derives from iterating a `frozenset` of ratee names
    and so varies with PYTHONHASHSEED. Under truncation the visit order decides
    which clusters get found, so an unstable order would make the same input
    produce different clusters in different processes.
    """
    budget = _Budget.for_config(cfg) if budget is None else budget
    by_funder: dict = defaultdict(list)
    by_ratee: dict = defaultdict(list)
    for p in profiles:
        if p.funder is not None:
            by_funder[_norm_funder(p.funder)].append(p)
        for r in p.ratees:
            by_ratee[r].append(p)
    small = {ratee for ratee, group in by_ratee.items() if len(group) <= cfg.sybil_max_group}
    small_sorted = _small_ratees_by_rater(profiles, small)
    for funder in sorted(by_funder):
        yield from _funder_group_pairs(by_funder[funder], cfg, _BlockCharge(budget))
        if budget.truncated:
            return
    for ratee in sorted(by_ratee, key=lambda r: (-len(by_ratee[r]), r)):
        group = by_ratee[ratee]
        if len(group) > cfg.sybil_max_group:
            budget.ratees_skipped_size += 1
            continue
        yield from _ratee_group_pairs(ratee, group, cfg, small_sorted,
                                      _BlockCharge(budget, ratee))
        if budget.truncated:
            return


def _index_profiles(profiles: list[RaterProfile]) -> dict[str, RaterProfile]:
    """rater -> profile. Raises ValueError on a duplicate rater name: a caller
    holding two profiles for one rater has a data bug, and silently keeping
    either one would quietly change who clusters with whom."""
    by_id: dict[str, RaterProfile] = {}
    for p in profiles:
        if p.rater in by_id:
            raise ValueError(f"duplicate rater in profiles: {p.rater!r}")
        by_id[p.rater] = p
    return by_id


def _log_budget(cfg: Config, stats: ClusterStats) -> None:
    """One INFO summary per run, plus one WARNING when the global budget cut
    generation short (per-ratee abandonments log their own, in `_emit`)."""
    logger.info("sybil candidate pairs: %d examined, %d tested (ratee blocks skipped: %d for size, "
                "%d for budget)", stats.pairs_examined, stats.pairs_tested,
                stats.ratees_skipped_size, stats.ratees_skipped_budget)
    if stats.truncated:
        logger.warning(
            "sybil: candidate-pair generation stopped at sybil_max_pairs=%d (%d pairs examined, "
            "%d tested, %d ratee block(s) skipped for size, %d for the per-ratee budget); the "
            "clusters found so far stand, but some raters may be under-merged",
            cfg.sybil_max_pairs, stats.pairs_examined, stats.pairs_tested,
            stats.ratees_skipped_size, stats.ratees_skipped_budget)


def cluster_raters_with_stats(profiles: list[RaterProfile],
                              cfg: Config) -> tuple[dict[str, str], ClusterStats]:
    """`cluster_raters`'s mapping, plus the `ClusterStats` describing the run.

    Callers that publish results (the CLI, the report) use this one: a run that
    hit a pair budget is still a valid run, but its clusters may be
    under-merged, and that has to be stated rather than hidden. Raises
    ValueError only on duplicate rater names -- never on a budget.
    """
    by_id = _index_profiles(profiles)
    uf = _UnionFind(by_id.keys())
    budget = _Budget.for_config(cfg)
    tested = 0
    for a, b in _candidate_pairs(profiles, cfg, budget):
        tested += 1
        if _signals(by_id[a], by_id[b], cfg) >= 2:
            uf.union(a, b)
    stats = ClusterStats(pairs_tested=tested, pairs_examined=budget.total,
                         ratees_skipped_size=budget.ratees_skipped_size,
                         ratees_skipped_budget=budget.ratees_skipped_budget,
                         truncated=budget.truncated)
    _log_budget(cfg, stats)
    return {r: uf.find(r) for r in by_id}, stats


def cluster_raters(profiles: list[RaterProfile], cfg: Config) -> dict[str, str]:
    """Cluster raters whose profiles share at least 2 of 3 sybil signals.

    Returns rater -> cluster id, the lexicographically smallest rater in the
    cluster. Raises ValueError on duplicate rater names -- and on nothing else.
    Exceeding `cfg.sybil_max_pairs` or `cfg.sybil_max_pairs_per_ratee` (both
    count candidate-pair EXAMINATIONS) stops candidate generation and degrades
    the clustering (some raters may be under-merged) instead of failing the
    run, because the budget is reachable
    by ~6,000 cheap on-chain transactions and an abort there is permanent for
    everyone -- see the module docstring. Use `cluster_raters_with_stats` when
    the caller must know whether that happened; anything that publishes a score
    must.
    """
    return cluster_raters_with_stats(profiles, cfg)[0]


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
