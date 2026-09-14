# robustrep

Robust, transport-agnostic reputation scoring for AI agents.

Feed it any table of `(rater, ratee, value, scale, tag, ts, evidence_uri, source)` and it returns a
robust score per ratee that resists Sybil raters and evidence-free ratings. Ships with a Base-chain
ERC-8004 adapter and a reproducible report.

Status: v0.1 in development. See `docs/` in the parent repo for the design.
