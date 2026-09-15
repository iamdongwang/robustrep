"""Map raw values to [0, 1] per (tag, scale) group. Groups never mix.

Why the rule ladder below is share-based, clips, and never rescales by a
group's own min/max (CONFIRMED security finding C2): ERC-8004's `value` is an
attacker-chosen int128. When rule selection keyed off a group's min and max,
ONE record of 2**127-1 pushed an honest `percent` group into a min-max fallback
and squashed every honest score to ~0; two records (+max and -min) flattened
the group to exactly 0.5. That is a breakdown point of 1/n in the normalizer,
which wastes the ~0.5 breakdown of the weighted median downstream -- and two
Base addresses have already posted such values.

Two properties close that hole:

* A rule is chosen when at least `fit_share` of the group's values land in its
  range, so selection needs a share of the group rather than one outlier, and
  out-of-range values are CLIPPED into the range they are scored against, so an
  outlier's damage is confined to its own record.
* Every map is ABSOLUTE - only the choice of WHICH rule applies is
  group-relative. A flood can therefore change a group's rule name but can
  never re-level the honest records inside it (see `_binary_scores`).

Cost of the `rank` fallback: it preserves only within-group ORDER, so a 0.8
under one (tag, scale) group means something different from a 0.8 under
another. tag1 is free text on ERC-8004, so groups are not comparable categories
anyway -- the report's Limitations already flags tag hygiene.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_NORM_FIT_SHARE, validate_norm_fit_share
from .schema import scale_decimals

RULE_BINARY = "binary"
RULE_PERCENT = "percent"
RULE_UNIT = "unit"
RULE_CONSTANT = "constant"
RULE_RANK = "rank"


def _fits(mask: np.ndarray, fit_share: float) -> bool:
    """True when at least `fit_share` of a group's values satisfy `mask`."""
    return bool(mask.mean() >= fit_share)


def _binary_scores(vals: np.ndarray, decimals: int) -> np.ndarray:
    """Binary map for values in {0, 1}; the next absolute rule for the rest.

    Values outside {0, 1} are deliberately NOT clipped into {0, 1}. Clipping
    them there would be a group-relative re-levelling: an attacker who floods a
    percent group with enough 0/1 rows to clear the binary fit share would turn
    every honest 80 into 1.0 without ever rating those agents. Instead an
    out-of-set value takes the next ABSOLUTE rule for its scale - `percent`
    when decimals == 0, `unit` otherwise - so the honest 80 still scores 0.8
    and a stray 7 in a genuine binary group scores 0.07 rather than 1.0.

    This buys the invariant at the price of a deliberate discontinuity inside a
    d0 binary-fitting group: 1 -> 1.0 but 2 -> 0.02. Only genuinely mixed
    groups ever see it; a true binary group has nothing outside {0, 1} to map.
    """
    in_set = np.isin(vals, (0.0, 1.0))
    other = np.clip(vals, 0.0, 100.0) / 100.0 if decimals == 0 else np.clip(vals, 0.0, 1.0)
    return np.where(in_set, vals, other)


def _group_scores(real: pd.Series, decimals: int, fit_share: float) -> tuple[np.ndarray, str]:
    """Map one (tag, scale) group's real values to [0, 1].

    A rule *fits* when at least `fit_share` of the group's values land in its
    range (see the module docstring for why). Rules are tried in order and the
    first that fits wins:

    1. binary   - values in {0, 1}, mapped straight through. Tried first so a
       {0, 1} group is never divided by 100 by "percent" below; values outside
       {0, 1} take the next absolute rule instead (see `_binary_scores`).
    2. percent  - integer scale (decimals == 0), values in [0, 100]: clipped,
       then divided by 100. Tried before "unit" so an integer-scaled group that
       happens to sit inside [0, 1] is still read as a percentage rather than
       assumed already-normalized.
    3. unit     - values in [0, 1]: clipped, passed through.
    4. constant - every value identical: a neutral 0.5. After "percent" so a
       constant all-100 d0 group scores 1.0 there, while a constant all-500 d0
       group has no natural anchor.
    5. rank     - fallback: percentile rank, `(rank - 1) / (n - 1)` over average
       ranks (ties share a rank). n >= 2 here, since a 1-value group is caught
       by "constant".
    """
    vals = real.to_numpy(dtype=float)
    if _fits(np.isin(vals, (0.0, 1.0)), fit_share):
        return _binary_scores(vals, decimals), RULE_BINARY
    if decimals == 0 and _fits((vals >= 0) & (vals <= 100), fit_share):
        return np.clip(vals, 0.0, 100.0) / 100.0, RULE_PERCENT
    if _fits((vals >= 0) & (vals <= 1), fit_share):
        return np.clip(vals, 0.0, 1.0), RULE_UNIT
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

    `fit_share` is the share of a group's values that must land in a rule's
    range for that rule to be chosen (see `_group_scores`); callers going
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
