# robustrep

Robust, transport-agnostic reputation scoring for AI agents.

Feed it any table of `(rater, ratee, value, scale, tag, ts, evidence_uri, source)` and it returns a
robust score per ratee that resists Sybil raters and evidence-free ratings. Ships with a Base-chain
ERC-8004 adapter and a reproducible report.

Status: v0.1.0 released on PyPI. Design spec and plan live in the parent workspace; a summary is in `reports/latest/report.md`.

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

Latest scores: `reports/latest/scores.json`, also served at
https://iamdongwang.github.io/robustrep/reports/latest/scores.json
