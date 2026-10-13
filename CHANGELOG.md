# Changelog

All notable changes to robustrep are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is pre-1.0,
so a `0.x` bump may change published numbers without changing the API.

## [Unreleased]

Since October 2026 the new ERC-8004 raters on Base are almost all
smart-contract accounts: of 230 sampled unprofiled raters (2026-10-13),
230/230 had code and none had ever received a transfer, so no profile
source can name a first funder for them. 55 of 60 inspected are Coinbase
Smart Wallet proxies whose first transactions go through the ERC-4337
EntryPoint, sent by ~10 bundlers and sponsored by one paymaster -- shared
infrastructure, not a signal. `ownerAtIndex(0)` on those 55 wallets returns
only 4 distinct controlling keys (26/14/11/4). Scoring is unchanged; what
changes is which raters the funder signal can link.

### Added

- **Owner lookup (`robustrep/sources/account_owner.py`).** For a rater
  whose first-transfer lookup found nothing, `fetch` reads the smart
  account's controlling key over plain RPC -- `ownerAtIndex(0..3)` (Coinbase
  Smart Wallet; a 32-byte address payload is the owner, a 64-byte payload is
  a passkey and yields nothing, an empty slot moves to the next index), then
  `owner()` -- batched 50 addresses per JSON-RPC batch with the same
  per-address fallback as agent-owner resolution. A revert means the
  selector is absent; any other RPC error fails the chunk and leaves its
  rows for the next run. The key is stored in `raters.funder`, so sybil
  clustering links wallets sharing a controlling key exactly as it links
  wallets sharing a first funder; `first_seen_ts` stays NULL for such rows
  (clustering falls back to the earliest feedback time; caching that
  approximation would break the M4 rule).
- **`raters.funder_kind`:** `transfer` (first incoming transfer), `owner`
  (controlling key), `none` (every lookup tried, nothing found; never
  retried) or NULL (legacy row: only the transfer lookup ran). Opening a
  pre-0.2.2 store adds the column and marks existing funders `transfer`
  once; `Store.upsert_rater` takes the kind, `raters_needing_owner_lookup`
  and `funder_kind_counts` are new.
- **Internal-transfer retry.** An address with no owner is retried through
  Alchemy with `category: ["external", "internal"]`
  (`AlchemyClient.first_tx(categories=...)`) before being marked `none`.
- **Provenance `funder_kind_counts`** (in `scores.json` and as the
  `Funder kinds` line of `report.md`), a `Controlling-key linking`
  Limitations bullet, and the `fetch` summary line's per-run funder-kind
  counts.

### Changed

- **Meaning of `funder`:** "who funded the address" becomes "who controls
  or funded it". The sybil comparison (equality of the normalised funder)
  is unchanged, so an existing store keeps its clusters and gains the
  owner-linked ones on the next `fetch`.

## [0.2.1] — 2026-10-13

Blockscout's free tier stopped serving Base: its developer portal states
"Starting October 1, 2026, Base, Polygon and ZkSync will no longer be
available on Free tier", and since 2026-10-09 `base.blockscout.com/api` has
answered with a Cloudflare challenge page. Etherscan's free plan never covered
Base via the V2 API. Rater profiling (first-seen time and funder per rater,
the input to sybil clustering) therefore had no free source left. Alchemy's
free plan does cover Base, and on 12 raters already profiled through
Blockscout its earliest incoming external transfer gave the identical funder
and timestamp (12/12, 2026-10-13). Scoring is unchanged.

### Added

- **`AlchemyClient` (`robustrep/sources/alchemy_profile.py`).** Looks up
  the first incoming external transfer per address with one
  `alchemy_getAssetTransfers` call (`toAddress`, `category: ["external"]`,
  `order: asc`, `maxCount: 1`, `withMetadata`). Default 2.5 requests/sec
  (free plan: 300 CU/s, 120 CU per call), 5 retries. Same duck-typed
  interface and the same security contract as the other clients: streamed
  and capped body, implausible timestamps refused and warned once, the key
  redacted from urllib3 logs. Alchemy puts the key in the URL *path*, so no
  `requests` exception, URL or response text is ever interpolated into a
  raised message or log record; a JSON-RPC error's server text is logged at
  DEBUG only, length-bounded. A rejected key or a Base network not enabled
  for the app raises `AlchemyConfigError` (a `ProfilePlanError`, fixed
  message), which aborts profiling on the first address.
- **`fetch --profile-source alchemy`** and the **`ALCHEMY_API_KEY`**
  environment variable (environment only -- there is deliberately no flag).
  `--profile-rps`/`--profile-retries` apply as for the other sources.
- **`auto` precedence:** Etherscan key -> Alchemy key -> keyless Blockscout.
  `rater_profile_mode` records `alchemy` / `alchemy-partial` for a run
  profiled through Alchemy.

### Changed

- **`--profile-source auto` now picks Alchemy** when `ALCHEMY_API_KEY` is
  set and no Etherscan key is. A run that previously fell through to
  keyless Blockscout profiles through Alchemy instead once the variable is
  exported; with neither key it still uses Blockscout.
- **One retry loop for all three profile clients**
  (`profile_http.fetch_with_retries`). The response close, throttle sleep,
  transient/permanent classification, `Retry-After` handling and terminal
  error now exist once. A run of HTTP 200s whose bodies were all transient
  (a JSON-RPC rate limit, Etherscan's "Max rate limit reached") is reported
  as `<source> body-level error for <address> after N attempts` instead of
  the misleading `HTTP error ... (status n/a)`.
- **Alchemy HTTP 401/403 abort on the first address.** A rejected key
  (HTTP 401, `Must be authenticated!`) or a network not enabled for the
  app (HTTP 403, `BASE_MAINNET is not enabled for this app`) now raises
  `AlchemyConfigError` instead of a per-address `RuntimeError` that would
  have been repeated for every rater. `AlchemyConfigError`'s fixed message
  now says to check that Base Mainnet is enabled for the app.
- **Alchemy error classification:** a JSON-RPC error with code 429 (or a
  rate-limit message) is transient even when its text mentions the API
  key; the rate-limit substrings are `rate limit` / `rate-limit` /
  `throughput` / `capacity` (the bare `rate` matched "enumerate" and
  retried permanent errors); `"error": null` beside a result is a success.
- **Funder addresses are validated from every profile source.** A funder
  that is not a `0x` + 40-hex string is refused with one WARNING and the
  address is left unprofiled, as for an implausible timestamp, instead of
  being cached verbatim (security review L1).
- **The API key is redacted from robustrep's own loggers too**
  (`robustrep.sources.alchemy_profile`, `robustrep.sources.profile_http`),
  which echo bounded server-chosen text at DEBUG/WARNING; previously only
  urllib3's loggers were filtered (L2).
- **`Retry-After: nan`/`inf` is ignored** (falls back to exponential
  backoff) instead of failing the address on `time.sleep(nan)` (L4).
- **CLI tracebacks never print local variables:** `pretty_exceptions_enable`
  is off on the typer app, so an unexpected exception prints Python's
  standard traceback rather than relying on typer's `show_locals` default
  (I4). Expected errors still print one line and exit 1/2/3 as before.
- **Shared profile-HTTP helpers moved to `robustrep/sources/profile_http.py`**
  (`capped_json`, `plausible_ts`/`implausible`/`warn_bad_ts`,
  `status_code`, `is_transient_http`, `retry_after_seconds`,
  `raise_exhausted`, and the `MAX_PROFILE_RESPONSE_BYTES`,
  `MAX_PLAUSIBLE_TS`, `RETRY_AFTER_CAP` constants) as public names, so every
  rater-profile client applies one copy of the security review's rules.
  `MAX_PLAUSIBLE_TS`, `MAX_PROFILE_RESPONSE_BYTES` and `EtherscanPlanError`
  remain importable from `rater_profile`. The implausible-timestamp WARNING
  is now emitted by the `robustrep.sources.profile_http` logger.
- **`ProfilePlanError`** is the new base of `EtherscanPlanError` (and
  `AlchemyConfigError`); `enrich_raters` aborts on the first address for any
  of them.
- A store profiled partly through Blockscout and partly through Alchemy
  holds rows from both, while `rater_profile_mode` names only the last run's
  source. Acceptable because the two definitions were verified equivalent
  (above); the report's mode line describes the method, not the host per row.
- **Re-review follow-ups.** HTTP 401 from Alchemy aborts before the body
  is read, so an oversized body can no longer demote the abort into a
  per-address limit error; `short_for_log` scrubs every installed API key
  before truncating, so a key echoed at any offset of a server-chosen value
  is redacted rather than surviving as a prefix; the Etherscan and
  Blockscout malformed-entry errors no longer interpolate the parser's own
  message, which could carry server text.

## [0.2.0] — 2026-10-12

Adaptive `eth_getLogs` fetching. On 2026-10-12 mainnet.base.org began answering
every `eth_getLogs` with HTTP 429 (`-32011 request limit reached`), even for a
100-block window on a first request. The remaining free endpoints each cap the
window differently (publicnode serves roughly the last 5,000 blocks, nodies
50 blocks per call but deep history, drpc about 100), so fixed 2,000-block
chunks could not back-fill the roughly 1.2M blocks since the last report. `fetch`
now learns each endpoint's window at runtime and records which host served
what. Scoring is unchanged.

### Added

- **`LogFetcher` (`robustrep/sources/log_fetcher.py`).** Sits between
  `sync_feedback` and `RpcClient`: given a block range it returns all logs or
  raises `RpcError`. It learns each endpoint's window cap (shrinks by at least
  half on a range-too-large error, in one step to the cap when the server
  names one), marks an endpoint dead for the run when it rejects the minimum
  window (default 50 blocks), rotates past dead endpoints, and bounds invalid
  rounds so it always terminates. `LogFetcher.from_config(rpc, cfg)` is the
  construction path shared by the CLI and the live tests.
- **`sync_log` table and `Store.sync_sources()`.** One row per synced
  sub-window of a chunk (`from_block, to_block, host, window, n_logs, fetched_at`), written by
  delete-intersecting-then-insert so reruns are idempotent;
  `sync_sources()` aggregates per host.
- **`rpc_sources` report provenance** (last in `PROVENANCE_KEYS`). `report.md`
  gets a `Log sources:` line and a `Source recording starts at block ...`
  line under Provenance (`Log sources: not recorded ...` for data fetched
  before this version), and an always-present Limitations bullet, `Log sources
  not cross-checked.` RPC endpoint hostnames appear in the public report and
  `scores.json`; do not use an endpoint whose hostname embeds an account id or
  key.
- **`Config` fields:** `rpc_window_hints` (tuple of `(host, cap)` pairs,
  default empty), `rpc_min_window` (50) and `rpc_min_interval` (0.5 s between
  call starts; this is the CLI default). The library default (`LogFetcher`'s
  `min_interval`) is 0; `RpcClient` has no pacing.
- **`RpcRangeTooLargeError(endpoint, numbers)`**, raised immediately (no
  rotation or backoff) on provider range-cap errors. Its message is fixed
  text; only parsed integers leave the module. `RpcClient` also gains
  `current_url` and `rotate()`.
- **`Store.failed_ranges()`**, a read-only view of the recorded failed
  ranges, used by the live suite.

### Changed

- **Default `rpc_urls` is now four endpoints, in rotation order:**
  `https://mainnet.base.org`, `https://base-rpc.publicnode.com`,
  `https://base-pokt.nodies.app`, `https://base.drpc.org`. `--rpc-url`
  (repeatable) still replaces the whole list. Report provenance does not
  record this list; it records `rpc_sources`, the hosts that actually served
  each block range.
- **`sync_feedback(..., fetcher=None)`.** With a fetcher it records
  `sync_log`; without one it behaves as before and records no sources. The
  CLI always passes one.
- **Live suite: an unusable endpoint is a labelled failure, never a skip.**
  `rpc_available()` and the new `check_failed_ranges()` report
  `RPC unavailable: rate limited` / `transport failure: ...` with `pytest.fail`
  instead of `pytest.skip`. A run that cannot reach the chain has checked
  nothing, and a skip would have hidden the mainnet.base.org change for weeks;
  the owner prefers the email. The label (never server-supplied text) says
  whether to look at the endpoint list or at the data. Genuine data failures
  propagate unchanged. Both live tests now go through `LogFetcher.from_config`.
- **`--chunk` smaller than the minimum window still works**; it caps the
  window and is not raised to 50.

### Fixed

- **Live suite saw a throttled `eth_getLogs` as a bare checkpoint failure.**
  `sync_feedback` records a chunk whose fetch failed in `failed_ranges` and
  carries on rather than raising, so the availability guard never saw the 429
  and the test failed on `last_block` with no hint why (weekly run of
  2026-10-12). The test now reads the recorded errors through
  `Store.failed_ranges()` and classifies them with the same markers; a
  throttled sibling range can never hide a real one.
- **JSON-RPC error bodies carried by HTTP 400 are now classified.**
  base.drpc.org answers an over-cap `eth_getLogs` with HTTP 400 and a JSON-RPC
  error (`ranges over 10000 blocks are not supported on free plan`).
  `RpcClient` raised on the status before reading the body, so the range error
  was never recognised and drpc's window was never learned; every rotation to
  it re-sent a full-width request that failed. A 400 whose body is a JSON
  object with an `error` now goes through the same classification as a 200
  with an error (range too large, non-retryable, or retry and rotate). A 400
  without such a body, and every 403, 429 and 5xx, is still an HTTP error as
  before.
- **Default CLI pacing is 0.5 s between `eth_getLogs` calls (was 0.2 s).** The
  2026-10-12 back-fill showed nodies throttling at about 5 calls per second
  sustained (104 HTTP 429s). `LogFetcher`'s library default stays 0.

### Rollback

To go back to 0.1.2 (no source recording, no automatic rotation), install the
old release first: `pip install 'robustrep==0.1.2'`, then
`robustrep fetch --db data/base.db --chunk 50 --rpc-url https://base-pokt.nodies.app`;
the database is compatible in both directions (0.1.2 ignores `sync_log`;
blocks fetched under 0.1.2 have no source rows).

## [0.1.2] — 2026-10-03

A dependency security release. No API, CLI or scoring changes.

### Security — Fixed

- **urllib3 floor raised to 2.8.** urllib3 2.6–2.7 can enter an infinite loop
  on a chunked deflate stream (PYSEC-2026-4176) and buffer an unbounded
  chunk-size line into memory (PYSEC-2026-4177). Both are reachable from
  `fetch`, which reads responses from attacker-chosen evidence hosts, so a
  hostile URI could hang or exhaust the fetcher. 2.8 also fixes HTTPS-proxy TLS
  settings being ignored (PYSEC-2026-4175). The CI pin moves to 2.8.0 as well.

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

[0.2.0]: https://github.com/iamdongwang/robustrep/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/iamdongwang/robustrep/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/iamdongwang/robustrep/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/iamdongwang/robustrep/releases/tag/v0.1.0
