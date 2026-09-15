import logging
import threading
from urllib.parse import urlsplit

import pytest

import robustrep.sources.evidence_fetch as ef
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
        self.call_kwargs = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        self.call_kwargs.append(kwargs)
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
    # The URI is attacker-controlled, so it is logged with %r (quoted/escaped),
    # never bare %s -- a URI carrying newlines must not forge log lines.
    assert any("'https://boom'" in rec.message for rec in caplog.records)


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
    # With a single worker thread (workers=1), classify_all behaves like the
    # old sequential implementation: one thread -> one lazily-created,
    # thread-local requests.Session, reused for every URI.
    s = Store(tmp_path / "t4.db")
    rows = [{**_fb(f"https://s{i}"), "feedback_index": i} for i in range(3)]
    s.upsert_feedback(rows)
    seen_sessions = []

    def fetch(uri, session=None):
        seen_sessions.append(session)
        return None

    classify_all(s, fetch_text=fetch, workers=1)
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


# --- classify_all: concurrent workers -------------------------------------------------

def _seed_many(store: Store, n: int, prefix: str = "u") -> None:
    rows = [{**_fb(f"https://{prefix}{i}"), "feedback_index": i} for i in range(n)]
    store.upsert_feedback(rows)


def _mixed_fetch(uri, session=None):
    """Deterministic per-URI outcome (by trailing int in the uri) covering all
    three ``classify_all`` note branches: unfetchable, fetched-with-evidence,
    fetched-with-no-evidence."""
    i = int("".join(ch for ch in uri.rsplit("/", 1)[-1] if ch.isdigit()))
    if i % 7 == 0:
        return None  # unfetchable
    if i % 5 == 0:
        return f"tx {TX}"  # fetched, evidence found -> level 3
    return "just some prose, no evidence markers"  # fetched, no evidence -> level 1


def test_classify_all_workers_4_matches_workers_1(tmp_path):
    s1 = Store(tmp_path / "seq.db")
    _seed_many(s1, 50, prefix="p")
    n1 = classify_all(s1, fetch_text=_mixed_fetch, workers=1)

    s2 = Store(tmp_path / "par.db")
    _seed_many(s2, 50, prefix="p")
    n2 = classify_all(s2, fetch_text=_mixed_fetch, workers=4)

    assert n1 == n2 == 50
    rows1 = s1.conn.execute("SELECT uri, level, note FROM evidence_cache ORDER BY uri").fetchall()
    rows2 = s2.conn.execute("SELECT uri, level, note FROM evidence_cache ORDER BY uri").fetchall()
    assert rows1 == rows2
    assert len(rows1) == 50


def test_classify_all_worker_exception_still_yields_fetch_error(tmp_path):
    s = Store(tmp_path / "werr.db")
    _seed_many(s, 10, prefix="e")

    def fetch(uri, session=None):
        if uri == "https://e3":
            raise RuntimeError("boom")
        return None

    n = classify_all(s, fetch_text=fetch, workers=4)
    assert n == 10
    level, note = s.conn.execute(
        "SELECT level, note FROM evidence_cache WHERE uri='https://e3'").fetchone()
    assert level == 1 and note == "fetch-error"


def test_classify_all_writes_store_only_from_main_thread(tmp_path):
    s = Store(tmp_path / "tthread.db")
    _seed_many(s, 50, prefix="w")
    main_thread_id = threading.get_ident()
    write_thread_ids = []
    orig_upsert = s.upsert_evidence

    def recording_upsert(uri, level, note=""):
        write_thread_ids.append(threading.get_ident())
        return orig_upsert(uri, level, note)

    s.upsert_evidence = recording_upsert

    def fetch(uri, session=None):
        return None

    n = classify_all(s, fetch_text=fetch, workers=4)
    assert n == 50
    assert len(write_thread_ids) == 50
    assert all(tid == main_thread_id for tid in write_thread_ids)


def test_classify_all_each_worker_thread_gets_its_own_session(tmp_path):
    s = Store(tmp_path / "tsess.db")
    _seed_many(s, 40, prefix="s")
    seen = []  # (thread_ident, session_id)
    lock = threading.Lock()

    def fetch(uri, session=None):
        with lock:
            seen.append((threading.get_ident(), id(session)))
        return None

    classify_all(s, fetch_text=fetch, workers=4)

    by_thread: dict = {}
    for tid, sid in seen:
        by_thread.setdefault(tid, set()).add(sid)
    # Each worker thread reused exactly one session for every URI it handled
    # (thread-local caching), never a fresh one per call.
    assert all(len(sids) == 1 for sids in by_thread.values())
    # Distinct worker threads (if more than one actually ran) never share a
    # session object.
    if len(by_thread) > 1:
        session_ids_by_thread = [next(iter(sids)) for sids in by_thread.values()]
        assert len(set(session_ids_by_thread)) == len(by_thread)


# --- parser differential: urlsplit vs urllib3 (C1) -------------------------------------

def _public_resolver(host):
    """Resolver stub answering one public address for every host, so these
    tests exercise the parsing/authority checks rather than the DNS check."""
    return ["93.184.216.34"]


@pytest.mark.parametrize("url", [
    # urlsplit sees host "example.com" (port 80) while urllib3 -- which is what
    # requests actually connects with -- terminates the authority at the
    # backslash and sees 127.0.0.1:6379.
    "http://127.0.0.1:6379\\@example.com/",
    "http://169.254.169.254\\@example.com/latest/meta-data/",
    "http://example.com\\@127.0.0.1/",
    "http://user@example.com/",  # any userinfo is refused outright
    "http://exa mple.com/",
    "http://example.com\x00/",
    "http://example.com\r\n/",  # urlsplit strips CR/LF; the raw URL must not
    "http://example.com\x85/",
])
def test_is_safe_url_refuses_authority_with_backslash_userinfo_or_control_chars(url):
    assert ef._is_safe_url(url, resolver=_public_resolver) is False


def test_is_safe_url_cross_checks_urllib3_parse(monkeypatch):
    # Stand in for any future parser differential: if urllib3 disagrees with
    # urlsplit about scheme/host/port, the URL is refused rather than fetched.
    from urllib3.util import Url

    monkeypatch.setattr(
        ef, "_urllib3_parse",
        lambda u: Url(scheme="http", host="127.0.0.1", port=6379, path="/"))
    assert ef._is_safe_url("http://example.com/", resolver=_public_resolver) is False


def test_is_safe_url_still_allows_ordinary_public_url():
    assert ef._is_safe_url("https://example.com/path?x=1", resolver=_public_resolver) is True


def test_is_safe_url_still_refuses_bracketed_ipv6_literal():
    # A public IPv6 literal is still refused as a bare IP literal, as before --
    # the bracket stripping in the parser cross-check must not open a bypass.
    assert ef._is_safe_url("http://[2606:2800:220:1:248:1893:25c8:1946]/",
                           resolver=_public_resolver) is False


def test_canonical_url_is_rebuilt_from_vetted_parts():
    assert ef._canonical_url("HTTP://Example.COM:80/a/b?x=1#frag") == "http://example.com/a/b?x=1"
    assert ef._canonical_url("https://example.com/") == "https://example.com/"
    assert ef._canonical_url("https://example.com") == "https://example.com/"
    assert ef._canonical_url("https://example.com.:8443/x") == "https://example.com:8443/x"


def test_fetch_url_requests_canonical_url_not_raw():
    resp = FakeResp(200, body=b"ok")
    sess = FakeSession({"http://example.com/p?q=1": resp})
    # Raw string differs from the canonical form in scheme case, host case,
    # redundant default port and fragment -- the request must use the form
    # rebuilt from the components the guard actually vetted.
    text = ef._fetch_url("HTTP://Example.com:80/p?q=1#f", sess, _public_resolver,
                         ef.DEFAULT_TIMEOUT)
    assert text == "ok"
    assert sess.calls == ["http://example.com/p?q=1"]
    headers = {k.lower(): v for k, v in sess.call_kwargs[0]["headers"].items()}
    assert headers["accept-encoding"] == "identity"
    assert headers["user-agent"] == ef.USER_AGENT
    assert resp.closed is True


def test_redirect_target_is_canonicalized_and_guarded():
    first = FakeResp(302, headers={"location": "HTTP://Other.example:80/next#x"})
    final = FakeResp(200, body=b"done")
    sess = FakeSession({
        "http://example.com/a": first,
        "http://other.example/next": final,
    })
    assert ef._fetch_url("http://example.com/a", sess, _public_resolver,
                         ef.DEFAULT_TIMEOUT) == "done"
    assert sess.calls == ["http://example.com/a", "http://other.example/next"]


def test_redirect_location_with_backslash_authority_is_not_requested():
    first = FakeResp(302, headers={"location": "http://127.0.0.1\\@other.example/"})
    sess = FakeSession({"http://example.com/a": first})
    assert ef._fetch_url("http://example.com/a", sess, _public_resolver,
                         ef.DEFAULT_TIMEOUT) is None
    # Recording the calls (rather than relying on FakeSession's raise) is what
    # proves the redirect target never reached the transport.
    assert sess.calls == ["http://example.com/a"]


def test_resolve_uri_ipfs_gateway_is_not_cloudflare():
    # cloudflare-ipfs.com was retired; a dead gateway is a wasted hop.
    assert "cloudflare" not in resolve_uri("ipfs://bafy123")
    assert not any("cloudflare" in gw for gw in IPFS_GATEWAYS)


def test_is_safe_url_refuses_when_urllib3_cannot_parse(monkeypatch):
    # urllib3 rejecting a URL that urlsplit happily parses is itself a
    # differential: refuse rather than fetch a URL only one parser understands.
    def boom(url):
        raise ValueError("cannot parse")

    monkeypatch.setattr(ef, "_urllib3_parse", boom)
    assert ef._is_safe_url("http://example.com/", resolver=_public_resolver) is False


@pytest.mark.parametrize("loc", [
    "http://ot\rher.example/x",
    "http://ot\ther.example/x",
    "http://other.example/x\n",
])
def test_redirect_location_with_control_characters_is_refused(loc):
    # urljoin (via urlsplit) strips TAB/CR/LF *before* the guard would ever see
    # them, silently normalizing a hostile Location into a different, allowed
    # URL -- so the raw header value is screened first.
    first = FakeResp(302, headers={"location": loc})
    sess = FakeSession({"http://example.com/a": first})
    assert ef._fetch_url("http://example.com/a", sess, _public_resolver,
                         ef.DEFAULT_TIMEOUT) is None
    assert sess.calls == ["http://example.com/a"]
    assert first.closed is True


# --- host allowlist: the vetted host must survive requests' requoting --------------

@pytest.mark.parametrize("url", [
    # requests' prepare_url runs requote_uri(), which percent-DECODES unreserved
    # characters in the authority: these vet as one host and connect to another.
    "http://169.254.169.25%34/x",
    "http://127.0.0.%31/",
    "http://ev%69l.example/",
    # ...and percent-ENCODES others, again changing the connected host.
    'http://exa"mple.com/',
    "http://ex`ample.com/",
    "http://exa^mple.com/",
    "http://host_name.example/",  # underscore is not an LDH label character
    "http://-lead.example/",
    "http://.example.com/",
])
def test_is_safe_url_refuses_host_outside_ldh_allowlist(url):
    assert ef._is_safe_url(url, resolver=_public_resolver) is False


def test_redirect_location_with_percent_encoded_host_is_refused():
    first = FakeResp(302, headers={"location": "http://169.254.169.25%34/"})
    sess = FakeSession({"http://example.com/a": first})
    assert ef._fetch_url("http://example.com/a", sess, _public_resolver,
                         ef.DEFAULT_TIMEOUT) is None
    assert sess.calls == ["http://example.com/a"]


@pytest.mark.parametrize("url", [
    "https://example.com/path?x=1",
    "http://sub.domain.example/a/b%20c?q=%2F#frag",
    "https://xn--bcher-kva.example./x",  # punycode label + trailing root dot
    "http://a-b-c.example/p@th",  # an @ in the *path* is not userinfo
    "https://example.com/ünicode",  # non-ASCII path, ASCII host
    "http://example.com:80/a?b=c%20d&e=f",
])
def test_vetted_host_is_the_host_requests_would_connect_to(url):
    """The invariant the whole guard rests on: whatever transforms the URL
    between here and the socket (``requests.PreparedRequest.prepare_url`` ->
    ``requote_uri`` -> urllib3), the host/port actually connected to is the
    host/port ``_is_safe_url`` vetted and resolved."""
    import requests
    from urllib3.util import parse_url

    assert ef._is_safe_url(url, resolver=_public_resolver) is True
    parts = urlsplit(url)
    prepared = requests.Request("GET", ef._canonical_url(url)).prepare()
    connected = parse_url(prepared.url)
    default_port = 443 if parts.scheme == "https" else 80
    assert connected.host == parts.hostname.rstrip(".").lower()
    assert (connected.port or default_port) == (parts.port or default_port)


# --- ipfs tails must stay under the gateway's /ipfs/ prefix -------------------------

@pytest.mark.parametrize("uri", [
    "ipfs://../api/v0/id",  # would address an arbitrary gateway API path
    "ipfs:///etc/passwd",
    "ipfs://Qm1/../../api/v0/id",
    "ipfs://",
    # requests' requote_uri decodes %2e back into "." on the wire, so a raw
    # ".." check alone is not enough -- any % in the tail is refused.
    "ipfs://%2e%2e/%2e%2e/api/v0/id",
    "ipfs://a/%2E%2E/x",
])
def test_ipfs_uri_with_traversal_or_absolute_tail_is_refused(uri):
    sess = NeverCalledSession()
    assert resolve_uri(uri) is None
    assert http_fetch_text(uri, session=sess) is None
    assert sess.calls == []


# --- session lifetime ----------------------------------------------------------------

def test_fetch_url_closes_a_session_it_created(monkeypatch):
    created = []

    class OwnedSession:
        def __init__(self):
            self.closed = False
            self.calls = []
            created.append(self)

        def get(self, url, **kwargs):
            self.calls.append(url)
            return FakeResp(200, body=b"hi")

        def close(self):
            self.closed = True

    monkeypatch.setattr(ef.requests, "Session", OwnedSession)
    assert ef._fetch_url("http://example.com/x", None, _public_resolver,
                         ef.DEFAULT_TIMEOUT) == "hi"
    assert len(created) == 1 and created[0].closed is True


def test_fetch_url_closes_its_own_session_when_the_request_raises(monkeypatch):
    created = []

    class ExplodingSession:
        def __init__(self):
            self.closed = False
            created.append(self)

        def get(self, url, **kwargs):
            raise ConnectionError("network unreachable")

        def close(self):
            self.closed = True

    monkeypatch.setattr(ef.requests, "Session", ExplodingSession)
    assert ef._fetch_url("http://example.com/x", None, _public_resolver,
                         ef.DEFAULT_TIMEOUT) is None
    assert len(created) == 1 and created[0].closed is True


def test_fetch_url_does_not_close_a_caller_supplied_session():
    # classify_all reuses one thread-local session across many URIs; closing a
    # session this function did not create would break that reuse.
    sess = FakeSession({"http://example.com/x": FakeResp(200, body=b"hi")})
    sess.closed = False
    sess.close = lambda: setattr(sess, "closed", True)
    assert ef._fetch_url("http://example.com/x", sess, _public_resolver,
                         ef.DEFAULT_TIMEOUT) == "hi"
    assert sess.closed is False
