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
count) and hard caps on response size and time.

**Residual limitation: DNS rebinding.** The guard resolves the hostname and
checks those addresses, but `requests`/`urllib3` resolves it again to make the
connection. An attacker who can answer differently between the two lookups can
still cause a GET to a private address; the response body is never returned to
the caller, but the recorded evidence level leaks a few bits about the target.
This gap is documented in detail in `robustrep/sources/evidence_fetch.py` and
pinned by a strict-xfail test; a pinned-IP transport adapter is scheduled for
v0.2. Until then: **run `fetch` on a host that has no privileged reachability
into an internal network** (a laptop on a plain internet connection, or a
throwaway cloud box), not on a machine inside a private network or with access
to a cloud metadata service.

**Secrets.** API keys (e.g. an Etherscan key) are read from environment
variables only. Do not put them on the command line, in a config file in the
repository, or in an RPC URL that gets logged: RPC errors are redacted to
`scheme://host` precisely so a key embedded in a URL cannot leak into logs or a
bug report. Never commit a `.env` file (`.gitignore` refuses them).
