# Security Policy

## Supported versions

robustrep is pre-1.0. Only the latest published `0.x` release receives security
fixes; older `0.x` releases are not patched.

| Version        | Supported |
| -------------- | --------- |
| latest `0.x`   | yes       |
| any earlier    | no        |

## Reporting a vulnerability

Please report privately, not in a public issue. Use GitHub's **Report a
vulnerability** button on the repository's
[Security tab](https://github.com/iamdongwang/robustrep/security/advisories)
(private vulnerability reporting is enabled). Include what you did, what
happened, and what you expected; a reproducer helps.

Response target: an acknowledgement within **7 days**, with an assessment and
a plan (fix, mitigation, or a reasoned "not a vulnerability") in the same
thread. Please give us a chance to ship a fix before disclosing publicly.

## Threat model

robustrep reads data written by arbitrary, unauthenticated third parties and
must never be trusted to be reading anything benign.

**Untrusted inputs.** On-chain feedback is attacker-controlled in every field:
values, tags, evidence URIs and addresses. So are the responses of the RPC
endpoints and the Blockscout API the adapters talk to, and so is every byte
served by an evidence host. Nothing from these sources is treated as
authoritative; all of it is validated, bounded and normalized before use.

**Outbound fetches are the sharp edge.** `robustrep fetch` performs outbound
HTTP from the operator's own machine to evidence URIs chosen by the party being
rated. Those requests go through an SSRF guard
(`robustrep/sources/evidence_fetch.py`): scheme/host allowlist (http/https on
ports 80/443, LDH hostnames only, no bare IP literals, no non-ASCII hosts),
a cross-check that `urllib3` and `urlsplit` agree on scheme/host/port, a
request URL rebuilt from the vetted components rather than the attacker's raw
string, refusal of any address that resolves to a private/loopback/link-local/
reserved range, manually followed redirects (re-vetted at every hop, capped in
count), a hard cap on response size, and per-socket-operation (connect and
read) timeouts -- which bound each socket operation, not the request as a
whole: there is no wall-clock cap on a slow-drip response.

**Residual limitation: DNS rebinding.** The guard resolves the hostname and
checks those addresses, but `requests`/`urllib3` resolves it again to make the
connection. An attacker who can answer differently between the two lookups can
still cause a GET to a private address. The response body is never persisted
or reported -- but what *is* persisted, the `(level, note)` pair recorded per
evidence URI, makes the bypass at least a four-state oracle about the target:
no response at all, a response with no markers in it, a response containing a
transaction hash or a task-id key, or a hash involving the rater/owner
addresses -- plus a state per further note the cache can hold (`fetch-error`,
`lookup-budget:N`). A narrow leak rather than a blind request, which is why the
gap is tracked instead of shrugged off. It is documented in full in
`robustrep/sources/evidence_fetch.py` and pinned by a strict-xfail test; a
pinned-IP transport adapter is scheduled for v0.2. Until then: **run `fetch`
on a host that has no privileged reachability into an internal network** (a
laptop on a plain internet connection, or a throwaway cloud box), not on a
machine inside a private network or with access to a cloud metadata service.

**Secrets.** API keys (e.g. an Etherscan key) should come from the environment
(`ETHERSCAN_API_KEY`). `--etherscan-key` exists as an escape hatch and takes
precedence over the environment, but a command line is world-readable via `ps`
on most systems and lands in shell history -- prefer the environment variable
and treat the flag as a last resort. Never put a key in a config file in the
repository, and never commit a `.env` file (`.gitignore` refuses them).

Do not embed a key in an RPC URL either. `RpcError` messages are redacted to
`scheme://host` precisely so a key in a URL cannot leak into CLI output or a
bug report -- but `--verbose` turns on DEBUG logging, and the DEBUG line for a
failed RPC attempt carries the full transport exception (`exc_info`), whose
text can include the request URL the exception was raised against. The
urllib3 log filter only scrubs an exact Etherscan key out of string log
records, not out of an exception's own repr. So: keep keys out of RPC URLs,
and do not hand `--verbose` logs to anyone you would not hand the keys to.
