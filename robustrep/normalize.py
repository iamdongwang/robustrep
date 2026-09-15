"""Map raw values to [0, 1] per (tag, scale) group. Groups never mix.

Why rule selection tolerates a bounded number of out-of-range records, clips
them, and never rescales by a group's own min/max (CONFIRMED security finding
C2): ERC-8004's `value` is an attacker-chosen int128. When rule selection keyed
off a group's min and max, ONE record of 2**127-1 pushed an honest `percent`
group into a min-max fallback and squashed every honest score to ~0; two
records (+max and -min) flattened the group to exactly 0.5. That is a breakdown
point of 1/n in the normalizer, which wastes the ~0.5 breakdown of the weighted
median downstream -- and two Base addresses have already posted such values.

Two properties bound that:

* A rule is chosen when the records falling outside its range are within the
  group's outlier tolerance (`_fits`), and those records are CLIPPED into the
  range they are scored against, so an outlier's damage is confined to its own
  record. The tolerance is a count, never a bare share, because real groups are
  small: 489 of 654 (tag, scale) groups observed on Base hold <= 8 records, and
  at n <= 9 a 0.9 share test still lets a single record decide the rule.
* Every rung's own map is ABSOLUTE -- binary, percent, unit and constant score
  a record from its own value alone, with no reference to the rest of the group.

What this does NOT buy, stated plainly because it is security-load-bearing:

* `rank` is group-relative by construction. Its scores are positions within the
  group, so anyone who adds records to a `rank` group re-levels every honest
  score in it. This is the designed limit of the scheme, not an oversight.
* Which rung applies is still a group-level decision, so an attacker who posts
  MORE than the tolerance allows can still re-level honest records by forcing a
  rule change. At d0 a binary <-> percent switch rescales by 100x: tag
  `execution_success` on Base is 8 records all of value 1, and under a bare
  share test one added record of value 50 turned those honest 1.0s into 0.01.
  The count-based tolerance raises that floor -- it now takes at least two
  records at every group size >= 2 -- but it does not remove the effect.

Cost of the `rank` fallback beyond the above: it preserves only within-group
ORDER, so a 0.8 under one (tag, scale) group means something different from a
0.8 under another. tag1 is free text on ERC-8004, so groups are not comparable
categories anyway -- the report's Limitations already flags tag hygiene.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import DEFAULT_NORM_FIT_SHARE, validate_norm_fit_share
from .schema import scale_decimals

RULE_BINARY = "binary"
RULE_PERCENT = "percent"
RULE_UNIT = "unit"
RULE_CONSTANT = "constant"
RULE_RANK = "rank"

# Guards the floor below against float error. `n - n * fit_share` is used rather
# than the algebraically equal `n * (1 - fit_share)` because the latter is short
# by one at every multiple of 10 for fit_share=0.9 (1 - 0.9 is not exact in
# binary, so n=20 gives 1.9999999999999996 and floors to 1, not 2).
_FLOOR_EPS = 1e-9


def _fits(mask: np.ndarray, fit_share: float) -> bool:
    """True when the records failing `mask` are within the group's tolerance.

    The single place the tolerance is computed. It is a COUNT, not a share:

        allowed_outside = 0            if n < 2
                          max(1, floor(n - n * fit_share))   otherwise

    For n >= 10 at the default fit_share this is identical to "share in range
    >= fit_share". For 2 <= n <= 9 it is deliberately looser: exactly one
    out-of-range record is tolerated where a share test would tolerate none.
    That closes single-record poisoning at every group size >= 2 -- an attacker
    always needs at least two records -- which matters because most real
    (tag, scale) groups are that small. At n == 1 nothing is tolerated, so a
    lone record only ever matches a rule it genuinely falls inside.
    """
    n = mask.size
    allowed_outside = 0 if n < 2 else max(1, int(math.floor(n - n * fit_share + _FLOOR_EPS)))
    return int((~mask).sum()) <= allowed_outside


def _binary_scores(vals: np.ndarray, decimals: int) -> np.ndarray:
    """Binary map for values in {0, 1}; the next absolute rung for the rest.

    Values outside {0, 1} are deliberately NOT clipped into {0, 1}. Clipping
    them there would make the binary map group-relative: an attacker who posts
    enough in-range 0/1 rows to carry a percent group onto this rung would turn
    every honest 80 into 1.0 without ever rating those agents. Instead an
    out-of-set value takes the next ABSOLUTE rung for its scale -- `percent`
    when decimals == 0, `unit` otherwise -- so the honest 80 still scores 0.8
    and a stray 7 in a genuine binary group scores 0.07 rather than 1.0.

    The price is a deliberate discontinuity inside a d0 binary-fitting group:
    1 -> 1.0 but 2 -> 0.02. Only genuinely mixed groups can see it; a true
    binary group has nothing outside {0, 1} to map.
    """
    in_set = np.isin(vals, (0.0, 1.0))
    other = np.clip(vals, 0.0, 100.0) / 100.0 if decimals == 0 else np.clip(vals, 0.0, 1.0)
    return np.where(in_set, vals, other)


def _group_scores(real: pd.Series, decimals: int, fit_share: float) -> tuple[np.ndarray, str]:
    """Map one (tag, scale) group's real values to [0, 1].

    A rule *fits* when the records outside its range are within the group's
    outlier tolerance (`_fits`). Rules are tried in order, first fit wins:

    1. binary   - values in {0, 1}, mapped straight through. Tried first so a
       {0, 1} group is never divided by 100 by "percent"; values outside {0, 1}
       take the next absolute rung instead (see `_binary_scores`).
    2. percent (d0 only) - integer scale, values in [0, 100]: clipped, / 100.
       Tried before "unit" so an integer-scaled group that happens to sit
       inside [0, 1] is read as a percentage, not as already-normalized.
    3. unit     - values in [0, 1]: clipped, passed through.
    4. percent (any scale) - values in [0, 100]: clipped, / 100. Genuine
       percent data is also carried at d2 and up (real 99.77 under d2), which
       would otherwise have no absolute rung at all. After "unit" so a decimal
       group already confined to [0, 1] is not divided by 100 a second time.
    5. constant - every value identical: a neutral 0.5. After "percent" so a
       constant all-100 d0 group scores 1.0 there, while a constant all-500 d0
       group has no natural anchor.
    6. rank     - fallback: percentile rank, `(rank - 1) / (n - 1)` over
       average ranks (ties share a rank). n >= 2 here, since a 1-value group is
       caught by "constant".

    Rungs 1-5 map each record from its own value alone. Rung 6 does not: a rank
    is a position within the group, so records added to a `rank` group re-level
    every honest score in it. Pushing a group past its tolerance and onto
    `rank`, or across the d0 binary/percent boundary (a 100x rescale), is the
    residual way to move honest records -- see the module docstring.
    """
    vals = real.to_numpy(dtype=float)
    percent_fits = _fits((vals >= 0) & (vals <= 100), fit_share)   # rungs 2 and 4 share it
    if _fits(np.isin(vals, (0.0, 1.0)), fit_share):
        return _binary_scores(vals, decimals), RULE_BINARY
    if decimals == 0 and percent_fits:
        return np.clip(vals, 0.0, 100.0) / 100.0, RULE_PERCENT
    if _fits((vals >= 0) & (vals <= 1), fit_share):
        return np.clip(vals, 0.0, 1.0), RULE_UNIT
    if percent_fits:
        return np.clip(vals, 0.0, 100.0) / 100.0, RULE_PERCENT
    if vals.min() == vals.max():
        return np.full_like(vals, 0.5), RULE_CONSTANT
    ranks = pd.Series(vals).rank(method="average").to_numpy()
    return (ranks - 1.0) / (len(vals) - 1.0), RULE_RANK


def normalize(df: pd.DataFrame, fit_share: float = DEFAULT_NORM_FIT_SHARE) -> pd.DataFrame:
    """Add `score` (float, in [0, 1]) and `norm_rule` (str) columns.

    Input must have passed validate_records. Groups are formed strictly by
    (tag, scale) and never mixed - rows are selected by position, not by
    index label, so frames with duplicate index labels (e.g. the result of
    `pd.concat`) are handled correctly. Resets the index; any existing
    `score`/`norm_rule` columns are overwritten.

    `fit_share` sets each group's outlier tolerance (see `_fits`); callers going
    through the pipeline pass `Config.norm_fit_share`. It is range-checked here
    too, by the same validator Config uses, since `normalize` is also called
    directly.
    """
    validate_norm_fit_share(fit_share)
    out = df.reset_index(drop=True)
    decimals = out["scale"].map(scale_decimals)
    real = out["value"] / (10.0 ** decimals)
    scores = np.full(len(out), np.nan)
    rules = np.full(len(out), None, dtype=object)
    for (_, scale), pos in out.groupby(["tag", "scale"], sort=False).indices.items():
        group_scores, rule = _group_scores(real.iloc[pos], scale_decimals(scale), fit_share)
        scores[pos] = group_scores
        rules[pos] = rule
    out["score"] = scores
    out["norm_rule"] = rules
    return out
