import logging

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

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeSession:
    """Serves canned responses per URL; records every .get() call."""

    def __init__(self, by_url):
        self.by_url = dict(by_url)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        resp = self.by_url.get(url)
        if resp is None:
            raise AssertionError(f"unexpected fetch of {url}")
        if isinstance(resp, Exception):
            raise resp
        return resp


class NeverCalledSession:
    def get(self, *a, **k):
        raise AssertionError("session.get must not be called")


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
    assert classify_all(s, fetch_text=fetch, tx_parties=parties) == 0  # cached


# --- SSRF guard -------------------------------------------------------------------

@pytest.mark.parametrize("uri", [
    "http://127.0.0.1/x",
    "http://localhost/x",
    "http://10.0.0.1/x",
    "http://169.254.169.254/latest/meta-data",
    "http://[::1]/",
])
def test_ssrf_guard_refuses_local_and_link_local_hosts(uri):
    assert http_fetch_text(uri, session=NeverCalledSession()) is None


def test_ssrf_guard_refuses_non_standard_port():
    assert http_fetch_text("https://evil.example:8443/", session=NeverCalledSession()) is None


def test_ssrf_guard_refuses_malformed_port():
    assert http_fetch_text("http://evil.example:abc/", session=NeverCalledSession()) is None
    assert http_fetch_text("http://evil.example:99999/", session=NeverCalledSession()) is None


def test_ssrf_guard_refuses_unparseable_ipv6_url():
    assert http_fetch_text("http://[::1", session=NeverCalledSession()) is None


def test_ssrf_guard_allows_public_ip_literal():
    sess = FakeSession({"http://8.8.8.8/x": FakeResp(200, body=b"ok")})
    assert http_fetch_text("http://8.8.8.8/x", session=sess) == "ok"


# _is_safe_url / _is_disallowed_ip: internal units, covering branches
# http_fetch_text's own scheme dispatch short-circuits before ever reaching
# _is_safe_url (a non-http(s) scheme never even gets resolve_uri'd to a URL).

def test_is_safe_url_rejects_non_http_scheme_directly():
    assert _is_safe_url("ftp://public.example/x") is False


def test_is_safe_url_rejects_missing_host_directly():
    assert _is_safe_url("http:///path") is False


def test_is_disallowed_ip_true_for_non_ip_string():
    assert _is_disallowed_ip("not-an-ip") is True


def test_ssrf_guard_refuses_when_resolver_returns_garbage():
    def resolver(host):
        return ["not-an-ip"]

    assert http_fetch_text("http://public.example/x", session=NeverCalledSession(), resolver=resolver) is None


def test_ssrf_guard_refuses_when_resolver_returns_private_ip():
    def resolver(host):
        return ["10.1.2.3"]

    assert http_fetch_text("http://internal.example/x", session=NeverCalledSession(), resolver=resolver) is None


def test_ssrf_guard_allows_public_resolved_host():
    def resolver(host):
        return ["93.184.216.34"]

    sess = FakeSession({"http://public.example/x": FakeResp(200, body=b"hello")})
    assert http_fetch_text("http://public.example/x", session=sess, resolver=resolver) == "hello"
    assert sess.calls[0][1]["allow_redirects"] is False
    assert sess.calls[0][1]["timeout"] == (5, 10)


# --- redirects ----------------------------------------------------------------------

def test_redirect_to_private_host_refused():
    def resolver(host):
        return ["10.1.2.3"] if host == "internal.example" else ["93.184.216.34"]

    sess = FakeSession({
        "https://public.example/a": FakeResp(302, headers={"location": "https://internal.example/b"}),
    })
    assert http_fetch_text("https://public.example/a", session=sess, resolver=resolver) is None


def test_redirect_to_public_host_followed():
    def resolver(host):
        return ["93.184.216.34"]

    sess = FakeSession({
        "https://public.example/a": FakeResp(302, headers={"location": "https://public.example/b"}),
        "https://public.example/b": FakeResp(200, body=b"final body"),
    })
    assert http_fetch_text("https://public.example/a", session=sess, resolver=resolver) == "final body"


def test_transport_error_returns_none():
    def resolver(host):
        return ["93.184.216.34"]

    class RaisingSession:
        def get(self, *a, **k):
            raise ConnectionError("network unreachable")

    assert http_fetch_text("https://public.example/x", session=RaisingSession(), resolver=resolver) is None


def test_unrecognized_scheme_returns_none():
    assert http_fetch_text("ftp://public.example/x") is None


def test_redirect_chain_longer_than_max_is_refused():
    def resolver(host):
        return ["93.184.216.34"]

    by_url = {}
    for i in range(6):
        by_url[f"https://public.example/{i}"] = FakeResp(
            302, headers={"location": f"https://public.example/{i + 1}"})
    sess = FakeSession(by_url)
    assert http_fetch_text("https://public.example/0", session=sess, resolver=resolver) is None


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
