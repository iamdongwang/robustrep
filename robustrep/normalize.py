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
  rule change. Two rung pairs differ by exactly 100x and are the ones to watch:
  binary <-> percent at d0 (tag `execution_success` on Base is 8 records all of
  value 1; under a bare share test one added record of value 50 turned those
  honest 1.0s into 0.01), and unit <-> percent at any decimal scale (a d2 group
  of 0.20..0.95 would collapse to 0.002..0.0095). The count-based tolerance
  raises the floor to at least two records at every group size >= 2 (at any
  norm_fit_share < 1.0; at exactly 1.0 nothing is tolerated, by design), and the
  percent rung additionally demands positive evidence of percent shape before it
  can take a unit-shaped group, but neither removes the effect.
* `value` is float64 by the time it reaches here (validate_records coerces it),
  so two distinct int128s above 2**53 can land on the same float and therefore
  tie under `rank`. Ties share an average rank, so this costs resolution between
  two enormous values, never an honest record's position.

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

# The tolerance below floors a float, so it needs both of these. `n - n *
# fit_share` is used rather than the algebraically equal `n * (1 - fit_share)`,
# which is systematically short by one wherever the true answer is a whole
# number -- at every multiple of 1/(1 - fit_share), e.g. every 10th n at 0.9,
# where n=20 gives 1.9999999999999996 and floors to 1 instead of 2. The epsilon
# then absorbs the residual error that survives at other fit_share values
# (0.55 at n=100 lands just under 45 either way).
_FLOOR_EPS = 1e-9


def _fits(mask: np.ndarray, fit_share: float) -> bool:
    """True when the records failing `mask` are within the group's tolerance.

    The single place the tolerance is computed. It is a COUNT, not a share:

        allowed_outside = 0                                  if n < 2
                          0                                  if fit_share == 1.0
                          max(1, floor(n - n * fit_share))   otherwise

    For n >= 10 at the default fit_share this is identical to "share in range
    >= fit_share". For 2 <= n <= 9 it is deliberately looser: exactly one
    out-of-range record is tolerated where a share test would tolerate none.
    That closes single-record poisoning for any group holding at least 2 honest
    records -- an attacker always needs at least two of its own -- which matters
    because most real (tag, scale) groups are that small. A group of 1 honest
    record is not protectable and is not claimed to be: at n == 1 nothing is
    tolerated, so a lone record only ever matches a rule it genuinely falls
    inside, and a 1-honest-plus-1-attacker group is simply a 2-record group with
    no majority to appeal to.

    `fit_share == 1.0` means strictly all-or-nothing, with no small-group floor:
    every value must fall in the rule's range. That is the only way to ask for
    the pre-tolerance behaviour, so the floor must not quietly override it.
    """
    n = mask.size
    if n < 2 or fit_share >= 1.0:
        allowed_outside = 0
    else:
        allowed_outside = max(1, int(math.floor(n - n * fit_share + _FLOOR_EPS)))
    return int((~mask).sum()) <= allowed_outside


def _percent_scores(vals: np.ndarray) -> np.ndarray:
    """Percent map: clip into [0, 100], then divide by 100."""
    return np.clip(vals, 0.0, 100.0) / 100.0


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
    other = _percent_scores(vals) if decimals == 0 else np.clip(vals, 0.0, 1.0)
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
    4. percent (any scale) - values in [0, 100] AND a fit share strictly above
       1: clipped, / 100. Genuine percent data is also carried at d2 and up
       (real 99.77 under d2), which would otherwise have no absolute rung at
       all. It sits below "unit" and spans a 100x wider range, so it demands
       positive evidence of percent shape before it may claim a group: without
       the "above 1" test, two records of real 1.01 would carry a d2 unit group
       of 0.20..0.95 onto this rung and collapse it to 0.002..0.0095. The cost is
       a false negative: a genuine decimal percent group whose values are mostly
       sub-1% (error rates, say) has no evidence of percent shape either and
       falls to `rank`, keeping its order but losing absolute meaning -- which
       beats the alternative, a 100x squash of every honest unit group.
    5. constant - all values identical, within the same outlier tolerance: a
       neutral 0.5. Tolerance matters here too, or one record would move an
       otherwise-constant group onto the group-relative "rank" rung. The test is
       against the group's MEDIAN, which is an honest value whenever
       allowed_outside < n/2 -- guaranteed by the `fit_share > 0.5` validator
       for every n >= 3 (at n <= 2 the small-group floor of 1 is not below
       n/2, and a 2-record group has no majority to appeal to anyway).
       After "percent" so a constant all-100 d0 group scores 1.0 there, while a
       constant all-500 d0 group has no natural anchor.
    6. rank     - fallback: percentile rank, `(rank - 1) / (n - 1)` over
       average ranks (ties share a rank). n >= 2 here, since a 1-value group is
       caught by "constant".

    Rungs 1-5 map each record from its own value alone. Rung 6 does not: a rank
    is a position within the group, so records added to a `rank` group re-level
    every honest score in it. Pushing a group past its tolerance and onto
    `rank`, or across one of the two 100x rung boundaries (binary/percent at d0,
    unit/percent at any scale), is the residual way to move honest records --
    see the module docstring.
    """
    vals = real.to_numpy(dtype=float)
    if _fits(np.isin(vals, (0.0, 1.0)), fit_share):
        return _binary_scores(vals, decimals), RULE_BINARY
    percent_fits = _fits((vals >= 0) & (vals <= 100), fit_share)   # rungs 2 and 4 share it
    if decimals == 0 and percent_fits:
        return _percent_scores(vals), RULE_PERCENT
    if _fits((vals >= 0) & (vals <= 1), fit_share):
        return np.clip(vals, 0.0, 1.0), RULE_UNIT
    if percent_fits and _fits((vals > 1) & (vals <= 100), fit_share):
        return _percent_scores(vals), RULE_PERCENT
    if _fits(vals == np.median(vals), fit_share):
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
