import logging
from urllib.parse import urlsplit

import pytest

from robustrep.sources.evidence_fetch import (
    IPFS_GATEWAYS,
    MAX_BYTES,
    _is_disallowed_ip,
    _is_safe_url,
    classify_all,
    http_fetch_text,
    resolve_uri,
)
from robustrep.store import Store

TX = "0x" + "cd" * 32


# --- fakes ----------------------------------------------------------------------

class FakeRaw:
    def __init__(self, body: bytes):
        self._body = body

    def read(self, n, decode_content=True):
        return self._body[:n]


class FakeResp:
    def __init__(self, status_code=200, headers=None, body=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self.raw = FakeRaw(body)
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def close(self):
        self.closed = True


class FakeSession:
    """Serves canned responses per URL; records every .get() call. Raises
    loudly (not silently) on a URL it wasn't told to expect, so a guard that
    fails open is caught by the assertion failure, not masked as "no call"."""

    def __init__(self, by_url):
        self.by_url = dict(by_url)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        resp = self.by_url.get(url)
        if resp is None:
            raise RuntimeError(f"unexpected fetch of {url}")
        if isinstance(resp, Exception):
            raise resp
        return resp


class NeverCalledSession:
    """A session that must never be used. Records the call (so a refusal test
    can assert the exact -- here, empty -- set of URLs that were fetched) and
    then raises, so a guard that fails open surfaces as a loud RuntimeError
    rather than silently returning None from a session that quietly no-ops.

    Regression check for this fixture itself: temporarily making
    `_is_safe_url` always return True turns every refusal test in this file
    red (either on the RuntimeError propagating, or on the `sess.calls == []`
    assertion), confirming these tests actually detect a fail-open guard
    rather than passing vacuously.
    """

    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        raise RuntimeError("session.get must not be called for a refused URL")


# --- resolve_uri / basic http_fetch_text (task's own contract tests) ------------

def test_resolve_uri_ipfs_and_data():
    assert resolve_uri("ipfs://Qm1/x.json") == "https://ipfs.io/ipfs/Qm1/x.json"
    assert resolve_uri("https://a/b") == "https://a/b"
    assert resolve_uri("data:application/json,{\"a\":1}") is None


def test_http_fetch_text_handles_data_uri_and_failure():
    assert http_fetch_text("data:application/json,{\"taskId\":\"1\"}", session=None) == '{"taskId":"1"}'

    class Boom:
        def get(self, *a, **k):
            raise RuntimeError("x")

    assert http_fetch_text("https://a", session=Boom()) is None


def test_http_fetch_text_rejects_non_str_and_empty_uri():
    assert http_fetch_text(None) is None
    assert http_fetch_text("") is None
    assert http_fetch_text(123) is None


# --- classify_all (task's own contract test) -------------------------------------

def _fb(uri, client="0xaa", agent="7"):
    return dict(chain="base", block=1, tx_hash="0x", log_index=0, agent_id=agent, client=client,
                feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
                feedback_uri=uri, feedback_hash="")


def test_classify_all_uses_cache_and_parties(tmp_path):
    s = Store(tmp_path / "t.db")
    # Note: the third row must use a feedback_index distinct from the first
    # (both default to agent="7", client="0xaa") -- otherwise it collides with
    # row 1 on feedback's (agent_id, client, feedback_index) primary key and is
    # silently dropped by INSERT OR IGNORE, which would starve "https://bad" of
    # any feedback row at all.
    s.upsert_feedback([_fb("https://good"), {**_fb("https://good"), "feedback_index": 1},
                        {**_fb("https://bad"), "feedback_index": 2}])
    s.upsert_agent_owner("7", "0xowner")
    fetched = []

    def fetch(uri, session=None):
        fetched.append(uri)
        return f"tx {TX}" if uri == "https://good" else None

    def parties(h):
        return {"0xowner", "0xz"}

    n = classify_all(s, fetch_text=fetch, tx_parties=parties)
    assert n == 2 and sorted(fetched) == ["https://bad", "https://good"]
    assert s.evidence_level("https://good") == 3 and s.evidence_level("https://bad") == 1
    # "good" was fetched successfully -> note ""; "bad"'s fetch returned None
    # (per `fetch` above) -> note "unfetchable" (see the dedicated test below
    # for the level-1-but-fetched-successfully case, which also gets note "").
    good_note = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://good'").fetchone()[0]
    bad_note = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://bad'").fetchone()[0]
    assert good_note == "" and bad_note == "unfetchable"
    assert classify_all(s, fetch_text=fetch, tx_parties=parties) == 0  # cached


def test_classify_all_notes_unfetchable_vs_empty_vs_fetch_error(tmp_path):
    s = Store(tmp_path / "tnotes.db")
    s.upsert_feedback([
        _fb("https://never-fetched"),
        {**_fb("https://boom"), "feedback_index": 1},
        {**_fb("https://no-evidence"), "feedback_index": 2},
    ])

    def fetch(uri, session=None):
        if uri == "https://never-fetched":
            return None  # genuinely unfetchable
        if uri == "https://boom":
            raise RuntimeError("boom")  # fetch_text violating its no-raise contract
        return "just some prose, no tx hash or taskId here"  # fetched fine, level 1

    n = classify_all(s, fetch_text=fetch)
    assert n == 3
    note = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://never-fetched'").fetchone()[0]
    assert note == "unfetchable"
    note2 = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://boom'").fetchone()[0]
    assert note2 == "fetch-error"
    note3 = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://no-evidence'").fetchone()[0]
    assert s.evidence_level("https://no-evidence") == 1 and note3 == ""


# --- SSRF guard -------------------------------------------------------------------

@pytest.mark.parametrize("uri", [
    "http://127.0.0.1/x",
    "http://localhost/x",
    "http://10.0.0.1/x",
    "http://169.254.169.254/latest/meta-data",
    "http://[::1]/",
    "http://localhost./x",  # trailing FQDN dot must not bypass the localhost check
])
def test_ssrf_guard_refuses_local_and_link_local_hosts(uri):
    sess = NeverCalledSession()
    assert http_fetch_text(uri, session=sess) is None
    assert sess.calls == []


def test_ssrf_guard_refuses_non_standard_port():
    sess = NeverCalledSession()
    assert http_fetch_text("https://evil.example:8443/", session=sess) is None
    assert sess.calls == []


def test_ssrf_guard_refuses_malformed_port():
    sess = NeverCalledSession()
    assert http_fetch_text("http://evil.example:abc/", session=sess) is None
    assert http_fetch_text("http://evil.example:99999/", session=sess) is None
    assert sess.calls == []


def test_ssrf_guard_refuses_unparseable_ipv6_url():
    sess = NeverCalledSession()
    assert http_fetch_text("http://[::1", session=sess) is None
    assert sess.calls == []


def test_ssrf_guard_refuses_ip_literal_outright_even_when_public():
    # A bare IP literal is refused regardless of whether it's public --
    # evidence URIs must name a resolvable host so every fetch goes through
    # the resolver-based check, with no "it's just an IP" bypass.
    sess = NeverCalledSession()
    assert http_fetch_text("http://8.8.8.8/x", session=sess) is None
    assert sess.calls == []


def test_ssrf_guard_refuses_non_ascii_host():
    sess = NeverCalledSession()
    assert http_fetch_text("http://faß.example/x", session=sess) is None  # "faß.example"
    assert sess.calls == []


# _is_safe_url / _is_disallowed_ip: internal units, covering branches
# http_fetch_text's own scheme dispatch short-circuits before ever reaching
# _is_safe_url (a non-http(s) scheme never even gets resolve_uri'd to a URL).

def test_is_safe_url_rejects_non_http_scheme_directly():
    assert _is_safe_url("ftp://public.example/x") is False


def test_is_safe_url_rejects_missing_host_directly():
    assert _is_safe_url("http:///path") is False


def test_is_disallowed_ip_true_for_non_ip_string():
    assert _is_disallowed_ip("not-an-ip") is True


def test_is_disallowed_ip_refuses_cgnat_and_6to4_loopback():
    assert _is_disallowed_ip("100.64.0.1") is True  # CGNAT shared address space
    assert _is_disallowed_ip("2002:7f00:1::") is True  # 6to4 embedding 127.0.0.1
    assert _is_disallowed_ip("93.184.216.34") is False  # sanity: a real public IP is allowed


def test_is_disallowed_ip_refuses_nat64_reserved_ranges():
    # 64:ff9b::/96 (well-known NAT64) and 64:ff9b:1::/48 (RFC 8215 local-use
    # NAT64) both embed an IPv4 address -- and are marked `is_reserved` by
    # Python's ipaddress module (verified: is_global is True for these, so it
    # takes `is_reserved` specifically to catch them).
    assert _is_disallowed_ip("64:ff9b::7f00:1") is True  # embeds 127.0.0.1
    assert _is_disallowed_ip("64:ff9b::a00:1") is True  # embeds 10.0.0.1
    assert _is_disallowed_ip("64:ff9b:1::7f00:1") is True  # RFC 8215 local-use variant
    assert _is_disallowed_ip("8.8.8.8") is False  # sanity: a real public IP is allowed


def test_ssrf_guard_refuses_when_resolver_returns_garbage():
    sess = NeverCalledSession()

    def resolver(host):
        return ["not-an-ip"]

    assert http_fetch_text("http://public.example/x", session=sess, resolver=resolver) is None
    assert sess.calls == []


def test_ssrf_guard_refuses_when_resolver_returns_private_ip():
    sess = NeverCalledSession()

    def resolver(host):
        return ["10.1.2.3"]

    assert http_fetch_text("http://internal.example/x", session=sess, resolver=resolver) is None
    assert sess.calls == []


def test_ssrf_guard_refuses_mixed_public_and_private_resolved_addresses():
    # Any disallowed address in the resolver's answer refuses the whole host,
    # even if a public address is also present.
    sess = NeverCalledSession()

    def resolver(host):
        return ["93.184.216.34", "10.0.0.5"]

    assert http_fetch_text("http://multi.example/x", session=sess, resolver=resolver) is None
    assert sess.calls == []


def test_ssrf_guard_allows_public_resolved_host():
    def resolver(host):
        return ["93.184.216.34"]

    sess = FakeSession({"http://public.example/x": FakeResp(200, body=b"hello")})
    assert http_fetch_text("http://public.example/x", session=sess, resolver=resolver) == "hello"
    assert sess.calls == ["http://public.example/x"]


def test_ssrf_guard_allows_host_resolving_to_8_8_8_8():
    def resolver(host):
        return ["8.8.8.8"]

    sess = FakeSession({"http://dns.example/x": FakeResp(200, body=b"hello")})
    assert http_fetch_text("http://dns.example/x", session=sess, resolver=resolver) == "hello"
    assert sess.calls == ["http://dns.example/x"]


# --- redirects ----------------------------------------------------------------------

def test_redirect_to_private_host_refused():
    def resolver(host):
        return ["10.1.2.3"] if host == "internal.example" else ["93.184.216.34"]

    sess = FakeSession({
        "https://public.example/a": FakeResp(302, headers={"location": "https://internal.example/b"}),
    })
    assert http_fetch_text("https://public.example/a", session=sess, resolver=resolver) is None
    # Only the first (public) hop is ever fetched -- the internal redirect
    # target must never reach session.get.
    assert sess.calls == ["https://public.example/a"]


def test_redirect_to_public_host_followed():
    def resolver(host):
        return ["93.184.216.34"]

    sess = FakeSession({
        "https://public.example/a": FakeResp(302, headers={"location": "https://public.example/b"}),
        "https://public.example/b": FakeResp(200, body=b"final body"),
    })
    assert http_fetch_text("https://public.example/a", session=sess, resolver=resolver) == "final body"
    assert sess.calls == ["https://public.example/a", "https://public.example/b"]


def test_transport_error_returns_none():
    def resolver(host):
        return ["93.184.216.34"]

    class RaisingSession:
        def get(self, *a, **k):
            raise ConnectionError("network unreachable")

    assert http_fetch_text("https://public.example/x", session=RaisingSession(), resolver=resolver) is None


def test_unrecognized_scheme_returns_none():
    sess = NeverCalledSession()
    assert http_fetch_text("ftp://public.example/x", session=sess) is None
    assert sess.calls == []


def test_redirect_chain_longer_than_max_is_refused():
    def resolver(host):
        return ["93.184.216.34"]

    by_url = {}
    for i in range(6):
        by_url[f"https://public.example/{i}"] = FakeResp(
            302, headers={"location": f"https://public.example/{i + 1}"})
    sess = FakeSession(by_url)
    assert http_fetch_text("https://public.example/0", session=sess, resolver=resolver) is None
    # MAX_REDIRECTS=3: the initial fetch plus 3 hops = 4 calls, then refused.
    assert sess.calls == [f"https://public.example/{i}" for i in range(4)]


# --- response lifecycle: closed on every path ----------------------------------------

def test_response_closed_on_success_redirect_and_error_paths():
    def resolver(host):
        return ["93.184.216.34"]

    ok = FakeResp(200, body=b"hi")
    redirect = FakeResp(302, headers={"location": "https://public.example/final"})
    final = FakeResp(200, body=b"done")
    err = FakeResp(500)

    sess = FakeSession({
        "https://public.example/ok": ok,
        "https://public.example/redirect": redirect,
        "https://public.example/final": final,
        "https://public.example/err": err,
    })

    assert http_fetch_text("https://public.example/ok", session=sess, resolver=resolver) == "hi"
    assert ok.closed is True

    assert http_fetch_text("https://public.example/redirect", session=sess, resolver=resolver) == "done"
    assert redirect.closed is True
    assert final.closed is True

    assert http_fetch_text("https://public.example/err", session=sess, resolver=resolver) is None
    assert err.closed is True


def test_response_closed_when_body_read_raises():
    def resolver(host):
        return ["93.184.216.34"]

    class ExplodingRaw:
        def read(self, n, decode_content=True):
            raise RuntimeError("stream error")

    class ExplodingResp(FakeResp):
        def __init__(self):
            super().__init__(200)
            self.raw = ExplodingRaw()

    resp = ExplodingResp()
    sess = FakeSession({"https://public.example/explode": resp})
    assert http_fetch_text("https://public.example/explode", session=sess, resolver=resolver) is None
    assert resp.closed is True


# --- size / content -------------------------------------------------------------------

def test_body_is_capped_at_max_bytes():
    def resolver(host):
        return ["93.184.216.34"]

    big = b"x" * (1024 * 1024)
    sess = FakeSession({"https://public.example/big": FakeResp(200, body=big)})
    text = http_fetch_text("https://public.example/big", session=sess, resolver=resolver)
    assert text is not None and len(text) <= MAX_BYTES


def test_content_length_header_lying_does_not_matter():
    def resolver(host):
        return ["93.184.216.34"]

    body = b"y" * 1000
    sess = FakeSession({
        "https://public.example/big2": FakeResp(200, headers={"content-length": "999999999"}, body=body),
    })
    text = http_fetch_text("https://public.example/big2", session=sess, resolver=resolver)
    assert text == "y" * 1000


def test_binary_content_is_still_returned():
    def resolver(host):
        return ["93.184.216.34"]

    sess = FakeSession({"https://public.example/pdf": FakeResp(200, body=b"%PDF-1.4\xff\xfe")})
    text = http_fetch_text("https://public.example/pdf", session=sess, resolver=resolver)
    assert text is not None and text.startswith("%PDF")


# --- data: URIs -----------------------------------------------------------------------

def test_data_uri_base64_decodes():
    import base64
    payload = base64.b64encode(b'{"taskId":"1"}').decode()
    assert http_fetch_text(f"data:application/json;base64,{payload}") == '{"taskId":"1"}'


def test_data_uri_base64_marker_case_insensitive():
    import base64
    payload = base64.b64encode(b'{"taskId":"1"}').decode()
    assert http_fetch_text(f"data:application/json;BASE64,{payload}") == '{"taskId":"1"}'
    assert http_fetch_text(f"data:application/json;Base64,{payload}") == '{"taskId":"1"}'


def test_data_uri_invalid_base64_returns_none():
    assert http_fetch_text("data:application/json;base64,not-valid-base64!!!") is None


def test_data_uri_missing_comma_returns_none():
    assert http_fetch_text("data:application/json") is None


# --- ipfs gateways ----------------------------------------------------------------------

def test_ipfs_falls_through_to_next_gateway_on_failure():
    def resolver(host):
        return ["93.184.216.34"]

    first_url = IPFS_GATEWAYS[0] + "Qm1/x.json"
    second_url = IPFS_GATEWAYS[1] + "Qm1/x.json"
    sess = FakeSession({
        first_url: FakeResp(404),
        second_url: FakeResp(200, body=b'{"taskId":"1"}'),
    })
    assert http_fetch_text("ipfs://Qm1/x.json", session=sess, resolver=resolver) == '{"taskId":"1"}'


def test_ipfs_returns_none_if_all_gateways_fail():
    def resolver(host):
        return ["93.184.216.34"]

    by_url = {gw + "Qm2/x.json": FakeResp(500) for gw in IPFS_GATEWAYS}
    sess = FakeSession(by_url)
    assert http_fetch_text("ipfs://Qm2/x.json", session=sess, resolver=resolver) is None


# --- classify_all: error isolation, progress logging, session reuse ------------------

def test_classify_all_isolates_per_uri_errors_and_continues(tmp_path, caplog):
    s = Store(tmp_path / "t2.db")
    s.upsert_feedback([_fb("https://ok"), {**_fb("https://boom"), "feedback_index": 1}])

    def fetch(uri, session=None):
        if uri == "https://boom":
            raise RuntimeError("boom")
        return f"tx {TX}"

    def parties(h):
        return {"0xaa"}

    with caplog.at_level(logging.ERROR):
        n = classify_all(s, fetch_text=fetch, tx_parties=parties)
    assert n == 2
    assert s.evidence_level("https://boom") == 1
    r = s.conn.execute("SELECT note FROM evidence_cache WHERE uri='https://boom'").fetchone()
    assert r[0] == "fetch-error"
    assert any("https://boom" in rec.message for rec in caplog.records)


def test_classify_all_logs_progress(tmp_path, caplog):
    s = Store(tmp_path / "t3.db")
    rows = [{**_fb(f"https://u{i}"), "feedback_index": i} for i in range(5)]
    s.upsert_feedback(rows)

    def fetch(uri, session=None):
        return None

    with caplog.at_level(logging.INFO):
        n = classify_all(s, fetch_text=fetch, log_every=2)
    assert n == 5
    progress_msgs = [rec.message for rec in caplog.records if "processed" in rec.message]
    assert len(progress_msgs) == 2  # at i=2 and i=4


def test_classify_all_log_every_zero_disables_progress_logging(tmp_path, caplog):
    s = Store(tmp_path / "t3b.db")
    rows = [{**_fb(f"https://v{i}"), "feedback_index": i} for i in range(3)]
    s.upsert_feedback(rows)

    def fetch(uri, session=None):
        return None

    with caplog.at_level(logging.INFO):
        n = classify_all(s, fetch_text=fetch, log_every=0)  # must not raise ZeroDivisionError
    assert n == 3
    assert not [rec for rec in caplog.records if "processed" in rec.message]


def test_classify_all_reuses_one_session_across_uris(tmp_path):
    s = Store(tmp_path / "t4.db")
    rows = [{**_fb(f"https://s{i}"), "feedback_index": i} for i in range(3)]
    s.upsert_feedback(rows)
    seen_sessions = []

    def fetch(uri, session=None):
        seen_sessions.append(session)
        return None

    classify_all(s, fetch_text=fetch)
    assert len(seen_sessions) == 3
    assert len({id(x) for x in seen_sessions}) == 1
    assert all(x is not None for x in seen_sessions)


# --- known limitation: DNS rebinding (documented, not mitigated in v0.1) -------------

@pytest.mark.xfail(strict=True, reason="DNS rebinding not mitigated in v0.1 (see module docstring)")
def test_dns_rebinding_not_mitigated_in_v0_1():
    """Documents the vetted-address invariant a real fix must satisfy: the
    address actually connected to must be the *first* (guard-vetted) answer,
    never a later, independent re-resolution of the same host.

    ``_is_safe_url`` calls ``resolver(host)`` once to vet the host. A real
    HTTP client (requests/urllib3) resolves the host *again*, independently,
    when it actually opens the connection. ``RebindingSession`` stands in for
    that second resolution by calling the same stateful ``resolver`` a second
    time from ``.get()`` -- mimicking a DNS server that answers differently on
    consecutive queries (a rebinding attack): public on the first (the
    guard's) lookup, then the link-local metadata address afterwards.

    Nothing today pins the connection to the address the guard vetted, so
    ``connected_to`` ends up holding the *second* (rebound, private) answer
    instead of the first (public, vetted) one, and this assertion fails --
    hence the strict xfail. Once v0.2 adds a pinned-IP transport adapter
    (resolve once, connect to that literal address, keep the original
    hostname only for TLS SNI/Host), the connection will be pinned to the
    first answer, ``connected_to`` will equal ``["93.184.216.34"]``, and this
    test will XPASS -- turning the strict xfail into a hard failure that
    forces the module docstring and this test's xfail marker to be
    updated/removed.
    """
    answers = iter(["93.184.216.34", "169.254.169.254"])

    def resolver(host):
        return [next(answers)]

    class RebindingSession:
        def __init__(self):
            self.connected_to = []

        def get(self, url, **kwargs):
            host = urlsplit(url).hostname
            self.connected_to.append(resolver(host)[0])
            return FakeResp(200, body=b"ok")

    sess = RebindingSession()
    http_fetch_text("http://rebind.example/x", session=sess, resolver=resolver)
    assert sess.connected_to == ["93.184.216.34"]
