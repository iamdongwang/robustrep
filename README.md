# robustrep

Robust, transport-agnostic reputation scoring for AI agents.

Feed it any table of `(rater, ratee, value, scale, tag, ts, evidence_uri, source)` and it returns a
robust score per ratee that resists Sybil raters and evidence-free ratings. Ships with a Base-chain
ERC-8004 adapter and a reproducible report.

Status: v0.1.1 released on PyPI. Design spec and plan live in the parent workspace; a summary is in `reports/latest/report.md`.
Release notes, including the security fixes in 0.1.1: [CHANGELOG.md](CHANGELOG.md).

## Install

    pip install robustrep

## Library

```python
import pandas as pd
from robustrep import Config, score

records = pd.DataFrame([...])  # columns: rater, ratee, value, scale, tag, ts, evidence_uri, source
result = score(records, Config())
```

## Base ERC-8004 report

    robustrep fetch  --db data/base.db          # pull events (resumable)
    robustrep score  --db data/base.db --out scores.csv
    robustrep report --db data/base.db --out-dir reports

Rater profiles: Blockscout (free, no key) by default; Etherscan V2 requires a paid plan for Base.
Pick a source explicitly with `fetch --profile-source {auto,blockscout,etherscan,none}`.

Latest scores: `reports/latest/scores.json`, also served at
https://iamdongwang.github.io/robustrep/reports/latest/scores.json

## Security considerations

Everything robustrep reads is written by untrusted parties: on-chain values, tags, evidence URIs and
addresses, plus whatever the RPC/Blockscout endpoints and evidence hosts return.

`robustrep fetch` makes outbound HTTP from *your* machine to evidence URIs chosen by the rated party.
Those go through an SSRF guard (scheme/host allowlist, parser cross-check, canonical rebuild, refusal
of private/reserved addresses, manual redirects, a size cap and per-socket-operation timeouts — not a
wall-clock bound). Its one known SSRF bypass, DNS rebinding, is unmitigated in v0.1 — so run `fetch`
on a host with no privileged reach into an internal network. API keys should come from the
environment (`ETHERSCAN_API_KEY`), never the source tree; `--etherscan-key` is an escape hatch that
takes precedence over the environment and is visible to anyone who can run `ps`.

Full threat model and private reporting instructions: [SECURITY.md](SECURITY.md).
