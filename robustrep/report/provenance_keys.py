"""The provenance ``config`` keys, in published order -- named once.

Two modules have to agree on this list: ``report.publish.build_provenance``
builds the dict, and ``report.render`` writes it into report.md. While each
carried its own copy they drifted, and the renderer's copy silently dropped
``evidence_level_shares`` -- a key that was in every ``scores.json`` and in no
report.md. A published number that exists in one artifact and not the other is
worse than one that exists in neither, so the order lives here, imported by
both, and the renderer additionally appends any key it does not recognise
(sorted) so a future addition can be out of order but never missing.

Its own module rather than a constant in ``publish``: ``publish`` imports
``render``, so ``render`` cannot import back from it.
"""
from __future__ import annotations

PROVENANCE_KEYS = (
    "bootstrap_n",
    "bootstrap_seed",
    "min_clusters",
    "evidence_weights",
    "sybil_jaccard",
    "sybil_window_s",
    "sybil_max_group",
    "sybil_max_pairs",
    "sybil_max_pairs_per_ratee",
    "sybil_flag_share",
    "norm_fit_share",
    "norm_rule_counts",
    "evidence_level_shares",
    "evidence_max_lookups_per_uri",
    "evidence_max_total_lookups",
    "evidence_lookup_starved_uris",
)
