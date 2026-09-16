# Changelog

All notable changes to robustrep are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is pre-1.0,
so a `0.x` bump may change published numbers without changing the API.

## [0.1.1] — 2026-09-15

A security release: it closes every finding of an external security review of
0.1.0. There are no API changes — the library entry points and CLI commands are
the same, and existing calls keep working — but **scores will differ from 0.1.0
for any (tag, scale) group that held a poisoning value**, because normalization
no longer rescales a group by its own extremes.

### Security — Fixed

- **SSRF guard bypass via a URL-parser differential.** The guard vetted what
  `urlsplit` saw, but two transforms ran between it and the socket. `urllib3`'s
  parser treats a backslash as an authority terminator, so
  `http://127.0.0.1:6379\@example.com/` vetted as `example.com` and connected to
  `127.0.0.1:6379`; and `requests`' `requote_uri()` percent-*decodes* unreserved
  characters in the authority after the guard has run, so
  `http://169.254.169.25%34/` vetted as an unresolvable-looking host and
  connected to `169.254.169.254`. The request URL is now rebuilt from the vetted
  components rather than the attacker's string, hostnames must be LDH labels
  (nothing requoting would rewrite), and `urlsplit` and `urllib3` must agree on
  scheme/host/port before anything is fetched.
  **Operators who ran `fetch` with 0.1.0 on a host with reachability into an
  internal network should treat it as a possible internal-GET / IMDS-touch
  event.**
- **Normalization no longer min-max rescales by group extremes.** `value` is an
  attacker-chosen int128 on ERC-8004, and rule selection used to key off a
  group's own min and max: one record of `2**127-1` pushed an honest `percent`
  group into a min-max fallback and squashed every honest score in it to ~0, and
  two records (+max and −min) flattened the group to exactly 0.5 — a breakdown
  point of 1/n in the normalizer, ahead of the weighted median's ~0.5. Rule
  selection now tolerates a bounded *count* of out-of-range records, clips those
  records into the range they are scored against, and falls back to a percentile
  rank; every rung but `rank` maps a record from its own value alone.
- **Sybil clustering degrades under pair budgets instead of aborting.**
  `sybil_max_group`, `sybil_max_pairs` and `sybil_max_pairs_per_ratee` bound the
  candidate pairs a run examines. A run that hits a budget finishes with an
  under-merged clustering rather than failing, and says so: what was skipped is
  disclosed in the report's Limitations and in `cluster_stats` in `scores.json`,
  including which way the error points (under-merged, never over-merged).
- **Evidence tx-hash lookups are capped per URI and per run.** A single 200 KB
  evidence document can pack ~2,985 distinct 64-hex strings, each of which used
  to cost an `eth_getTransactionByHash` call. At most 8 distinct hashes per URI
  and 20,000 per run are now looked up. A URI cut short by the run budget is
  cached with a `lookup-budget:N` note and retried by later `fetch` runs up to 3
  attempts, and the number of still-starved URIs is published in provenance.
- **Etherscan key redacted from urllib3 DEBUG logs.** urllib3 logs every request
  line — path and query string included — at DEBUG, which `robustrep --verbose`
  turns on, so a key passed as a query parameter could reach any log sink. A
  logging filter now substitutes the key value in records passing urllib3's
  loggers. **Rotate `ETHERSCAN_API_KEY` if DEBUG logs were ever shared or
  stored.**
- **Further input bounds.** Response bodies are size-capped as they are read;
  RPC-supplied block timestamps and log topics are validated, and an undecodable
  log is skipped instead of aborting a sync; rater-profile pagination parameters
  echoed back by the API are allowlisted rather than followed as given;
  multi-tag bootstrap draws are chunked so peak memory stays bounded; and
  `urllib3 >= 2.6` is now required — the decoded-read cap that bounds a
  decompression bomb first exists in 2.6.0, and 2.0–2.5 decompress the whole
  body before truncating.

### Added

- `--sybil-max-group`, `--sybil-max-pairs` and `--sybil-max-pairs-per-ratee` on
  `score` and `report`.
- `--min-clusters` on `report`.
- `fetch --reprofile-raters`, to re-profile raters already cached.
- `SECURITY.md` (threat model, supported versions, private reporting) and
  `constraints.txt` (the fully pinned environment CI installs and audits).
- Provenance now carries `norm_fit_share`, `norm_rule_counts`, `cluster_stats`,
  the evidence lookup caps the *fetch* run actually used,
  `evidence_lookup_starved_uris`, and the full `versions` map — the same mapping
  in `report.md` and `scores.json`, so the two artifacts cannot disagree about
  the stack that produced them.
- Python 3.13 in the CI matrix (now 3.10–3.13).

### Changed

- `reports/latest/` publishes an allowlist of files (`report.md`,
  `scores.json`, `fig*.png`), never whatever the last run left in the block
  directory; it is served by GitHub Pages.
- `export_json` refuses to write a payload containing anything shaped like an
  EVM address — a backstop against de-anonymising a rater through a future
  column, not a sanitizer.
- IPFS gateways are `ipfs.io` and `dweb.link` (`cloudflare-ipfs.com` was
  retired).
- `__version__` is derived from the installed distribution metadata instead of a
  literal that could go stale after a version bump.

### Known limitations

- **DNS rebinding is unmitigated.** The guard resolves an evidence URI's host
  and checks those addresses, but `requests`/`urllib3` resolve it again to make
  the connection. The persisted `(level, note)` pair makes a bypass a narrow
  oracle rather than a blind request, which is why it is tracked rather than
  shrugged off; a pinned-IP transport adapter is planned for v0.2. Until then,
  run `fetch` on a host with no privileged reach into an internal network.
- **Normalization precedes sybil collapse.** Rule selection runs per (tag,
  scale) group over raw records, before clustering collapses a farm into one
  vote, so a perfectly collapsed farm can still have forced a rung change that
  re-levels every honest record in its group. `norm_rule_counts` in provenance
  is the signal — diff it run over run.
- **`rank`-rung scores are group-relative.** When a group falls back to
  percentile rank, a 0.8 there means a position within that group, not the same
  thing as a 0.8 in another group.

## [0.1.0] — 2026-09-14

Initial release on PyPI: the scoring library, the Base ERC-8004 adapter, and the
reproducible report.

[0.1.1]: https://github.com/iamdongwang/robustrep/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/iamdongwang/robustrep/releases/tag/v0.1.0
