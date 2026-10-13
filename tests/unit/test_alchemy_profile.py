"""``AlchemyClient`` (robustrep.sources.alchemy_profile) and its wiring into
``rater_profile.default_client``/``enrich_raters``.

The fakes mirror ``test_rater_profile``'s ``FakeV2Http``/``_FakeResponse``:
a streamed response (``raw``, ``raise_for_status``, ``close``) handed back by
a session whose ``post`` records what it was asked. Every error path also
asserts the API key -- which Alchemy puts in the URL *path* -- appears in no
raised message and no log record (M1)."""
import json
import logging
from datetime import datetime, timezone

import pytest
import requests

from robustrep.sources import alchemy_profile, http_util, profile_http
from robustrep.sources.alchemy_profile import (
    ALCHEMY_BASE_URL,
    CONFIG_REJECTED_MESSAGE,
    DEFAULT_ALCHEMY_RPS,
    AlchemyClient,
    AlchemyConfigError,
)
from robustrep.sources.profile_http import ProfilePlanError
from robustrep.sources.rater_profile import (
    BlockscoutV2Client,
    EtherscanClient,
    default_client,
    enrich_raters,
)
from robustrep.store import Store

KEY = "ALCHEMY-SECRET-KEY-VALUE-0123456789"
ADDR = "0xAbCdEf0000000000000000000000000000000001"
# A funder as the server spells it (checksum-style mixed case) and as it is cached.
FUNDER_UP = "0xF00D00000000000000000000000000000000BEEF"
FUNDER = FUNDER_UP.lower()

FB = dict(chain="base", tx_hash="0x", value="1", value_decimals=0, tag1="", tag2="", endpoint="",
          feedback_uri="", feedback_hash="")


def fb(client, block=1, log_index=0, agent_id="1", feedback_index=0):
    return dict(FB, block=block, log_index=log_index, agent_id=agent_id, client=client,
                feedback_index=feedback_index)


# --- fakes -----------------------------------------------------------------------------


class _FakeRaw:
    def __init__(self, data: bytes):
        self.data = data

    def read(self, amt, decode_content=True):
        chunk, self.data = self.data[:amt], self.data[amt:]
        return chunk


class _ExplodingRaw:
    """A body that must never be read: a transient HTTP error (429/5xx) is
    classified from its status alone, and a 4xx the test gave no body to."""

    def read(self, amt, decode_content=True):
        raise AssertionError("body must not be read when raise_for_status() raises")


class _FakeResponse:
    def __init__(self, payload=None, *, body=None, status_code=200, error=None):
        self.status_code, self._error, self.closed = status_code, error, False
        if error is not None and body is None:
            self.raw = _ExplodingRaw()
        else:
            self.raw = _FakeRaw(body if body is not None else json.dumps(payload).encode())

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def close(self):
        self.closed = True


def _http_error(status_code, url, headers=None):
    # The real requests.HTTPError text embeds the URL -- and the URL embeds the key.
    err = requests.HTTPError(f"{status_code} error for url: {url}")
    resp = requests.Response()
    resp.status_code = status_code
    resp.headers.update(headers or {})
    err.response = resp
    return err


def _as_bytes(body) -> bytes:
    return body if isinstance(body, bytes) else json.dumps(body).encode()


class FakeAlchemyHttp:
    """Each item in ``responses`` is a dict body (200 OK), raw ``bytes`` (200
    with that body), an int HTTP status (raised via ``raise_for_status``; the
    body then must not be read), a ``(status, headers)`` tuple, a ``(status,
    headers, body)`` tuple (an error status whose body -- ``bytes`` or a
    dict -- the client may read, as it does for a non-transient 4xx), or an
    exception instance (raised by ``post`` itself -- a connection error)."""

    def __init__(self, responses):
        self.responses, self.calls, self.responses_made = list(responses), [], []

    def post(self, url, json=None, timeout=None, stream=None):
        self.calls.append(dict(url=url, json=json, timeout=timeout, stream=stream))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, (int, tuple)):
            status, headers, body = (item, {}, None) if isinstance(item, int) else (tuple(item) + (None,))[:3]
            body = None if body is None else _as_bytes(body)  # `json` here is post()'s kwarg
            r = _FakeResponse(status_code=status, body=body, error=_http_error(status, url, headers))
        elif isinstance(item, bytes):
            r = _FakeResponse(body=item)
        else:
            r = _FakeResponse(item)
        self.responses_made.append(r)
        return r


def transfer(block_hex="0x1a2b", ts="2026-02-22T21:16:19.000Z", frm=FUNDER_UP, to=ADDR.lower()):
    return {"from": frm, "to": to, "blockNum": block_hex, "hash": "0xdeadbeef",
            "metadata": {"blockTimestamp": ts}}


def ok(transfers):
    return {"jsonrpc": "2.0", "id": 1, "result": {"transfers": transfers}}


# Bodies the live service answered with on 2026-10-13: HTTP 401 for a bad key,
# HTTP 403 for a network the app has not enabled.
BODY_401 = (b'{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Must be authenticated!"}}')
BODY_403_NOT_ENABLED = (
    b'{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"BASE_MAINNET is not enabled for this '
    b'app. Visit this page to enable the network: https://dashboard.alchemy.com/apps/<id>/networks"}}')


def rpc_error(code, message):
    return {"jsonrpc": "2.0", "id": 1, "error": {"code": code, "message": message}}


def client(http, **kw):
    kw.setdefault("sleep", lambda _: None)
    return AlchemyClient(KEY, session=http, **kw)


def _assert_key_hidden(exc, caplog):
    assert KEY not in str(exc)
    assert KEY not in repr(exc)
    assert KEY not in caplog.text


@pytest.fixture
def debug_log(caplog):
    with caplog.at_level(logging.DEBUG):
        yield caplog


# --- construction ------------------------------------------------------------------


def test_constants():
    assert ALCHEMY_BASE_URL == "https://base-mainnet.g.alchemy.com/v2"
    assert DEFAULT_ALCHEMY_RPS == 2.5  # 300 CU/s free plan / 120 CU per getAssetTransfers


def test_client_attributes_and_defaults():
    c = client(FakeAlchemyHttp([]))
    assert c.source == "alchemy"
    assert c.gap == 1 / 2.5
    assert c.retries == 5


def test_client_rps_sets_gap():
    assert client(FakeAlchemyHttp([]), rps=4.0).gap == 0.25


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_blank_key_raises_value_error(bad):
    with pytest.raises(ValueError) as exc:
        AlchemyClient(bad, session=FakeAlchemyHttp([]))
    assert "ALCHEMY_API_KEY" in str(exc.value)


def test_alchemy_config_error_is_a_profile_plan_error():
    assert issubclass(AlchemyConfigError, ProfilePlanError)


def test_constructor_installs_urllib3_key_redaction():
    # The fakes never go through urllib3, so the redactor cannot be observed
    # in action here; assert it was registered for this key (M1 -- the key
    # sits in the URL path, which urllib3 logs at DEBUG).
    c = client(FakeAlchemyHttp([]))
    assert KEY in http_util._installed_redactors
    assert isinstance(http_util._installed_redactors[KEY], http_util.ApiKeyRedactor)
    for name in ("urllib3", "urllib3.connectionpool"):
        assert any(isinstance(f, http_util.ApiKeyRedactor) and f.key == KEY
                   for f in logging.getLogger(name).filters)
    assert c is not None


def test_redactor_scrubs_a_path_embedded_key(caplog):
    client(FakeAlchemyHttp([]))
    with caplog.at_level(logging.DEBUG, logger="urllib3.connectionpool"):
        logging.getLogger("urllib3.connectionpool").debug(
            '%s://%s:%s "%s %s %s" %s %s', "https", "base-mainnet.g.alchemy.com", 443,
            "POST", f"/v2/{KEY}", "HTTP/1.1", 200, 512)
    assert KEY not in caplog.text
    assert "[redacted]" in caplog.text


def test_server_text_starting_with_the_key_is_redacted_from_the_debug_log(debug_log):
    # L2: the JSON-RPC message is server-chosen and logged (bounded) at DEBUG
    # on robustrep's own logger, which urllib3's redactors do not cover. The
    # key is the first 34 characters of the message so that it survives the
    # 40-character bound whole.
    http = FakeAlchemyHttp([rpc_error(-32602, f"{KEY} was rejected")])
    with pytest.raises(RuntimeError):
        client(http).first_tx(ADDR)
    records = [r for r in debug_log.records if r.name == "robustrep.sources.alchemy_profile"]
    assert records and any("[redacted]" in r.getMessage() for r in records)
    assert KEY not in debug_log.text


def test_server_chosen_timestamp_equal_to_the_key_is_redacted_from_the_warning(debug_log):
    http = FakeAlchemyHttp([ok([transfer(ts=KEY)])])
    assert client(http).first_tx(ADDR) is None
    warnings = [r for r in debug_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "[redacted]" in warnings[0].getMessage()
    assert KEY not in debug_log.text


def test_server_chosen_funder_equal_to_the_key_is_redacted_from_the_warning(debug_log):
    http = FakeAlchemyHttp([ok([transfer(frm=KEY)])])
    assert client(http).first_tx(ADDR) is None
    warnings = [r for r in debug_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "[redacted]" in warnings[0].getMessage()
    assert KEY not in debug_log.text


def test_constructor_installs_redaction_once_per_key_on_the_robustrep_loggers():
    names = ("robustrep.sources.alchemy_profile", "robustrep.sources.profile_http")
    before = {n: list(logging.getLogger(n).filters) for n in names}
    saved = dict(http_util._installed_redactors)
    for n in names:
        logging.getLogger(n).filters = []
    http_util._installed_redactors.clear()
    try:
        client(FakeAlchemyHttp([]))
        client(FakeAlchemyHttp([]))
        for n in names:
            redactors = [f for f in logging.getLogger(n).filters
                         if isinstance(f, http_util.ApiKeyRedactor) and f.key == KEY]
            assert len(redactors) == 1, n
    finally:
        for n, filters in before.items():
            logging.getLogger(n).filters = filters
        http_util._installed_redactors.clear()
        http_util._installed_redactors.update(saved)


# --- happy path --------------------------------------------------------------------------


def test_first_tx_parses_hex_block_iso_timestamp_and_lowercases_funder():
    http = FakeAlchemyHttp([ok([transfer()])])
    got = client(http).first_tx(ADDR)
    expected_ts = int(datetime(2026, 2, 22, 21, 16, 19, tzinfo=timezone.utc).timestamp())
    assert got == (0x1a2b, expected_ts, FUNDER)


def test_first_tx_request_shape():
    http = FakeAlchemyHttp([ok([transfer()])])
    client(http).first_tx(ADDR)
    assert len(http.calls) == 1
    call = http.calls[0]
    assert call["url"] == f"{ALCHEMY_BASE_URL}/{KEY}"
    assert call["stream"] is True
    assert call["timeout"] == 30
    body = call["json"]
    assert body["jsonrpc"] == "2.0" and body["id"] == 1
    assert body["method"] == "alchemy_getAssetTransfers"
    assert body["params"] == [{
        "fromBlock": "0x0", "toBlock": "latest", "toAddress": ADDR.lower(),
        "category": ["external"], "order": "asc", "maxCount": "0x1",
        "withMetadata": True, "excludeZeroValue": False,
    }]
    assert http.responses_made[0].closed is True


def test_first_tx_custom_base_url_is_used():
    http = FakeAlchemyHttp([ok([transfer()])])
    AlchemyClient(KEY, base_url="https://example.test/v2", session=http,
                  sleep=lambda _: None).first_tx(ADDR)
    assert http.calls[0]["url"] == f"https://example.test/v2/{KEY}"


def test_first_tx_throttles_after_every_attempt():
    sleeps = []
    http = FakeAlchemyHttp([503, ok([transfer()])])
    c = AlchemyClient(KEY, session=http, rps=2.0, sleep=sleeps.append)
    c.first_tx(ADDR)
    # gap after attempt 1, backoff 2**0, gap after attempt 2
    assert sleeps == [0.5, 1, 0.5]


def test_first_tx_empty_transfers_returns_none():
    http = FakeAlchemyHttp([ok([])])
    assert client(http).first_tx(ADDR) is None
    assert len(http.calls) == 1


# --- M4: implausible timestamps ----------------------------------------------------------


@pytest.mark.parametrize("ts", ["2200-01-01T00:00:00.000Z", "1900-01-01T00:00:00.000Z",
                                "not-a-timestamp", 12345, None])
def test_first_tx_implausible_or_unparseable_timestamp_returns_none_with_one_warning(ts, debug_log):
    http = FakeAlchemyHttp([ok([transfer(ts=ts)])])
    assert client(http).first_tx(ADDR) is None
    warnings = [r for r in debug_log.records if r.levelno == logging.WARNING
                and r.name == "robustrep.sources.profile_http"]
    assert len(warnings) == 1
    assert ADDR.lower() in warnings[0].getMessage()
    assert KEY not in debug_log.text


# --- L1: funder must be an address -------------------------------------------------------


@pytest.mark.parametrize("frm", [
    None, 7, {"hash": FUNDER}, ["x"], "0xZZ" + "Z" * 38, "0x" + "a" * 39, "0x" + "a" * 41,
    "", "0xFUNDER", FUNDER + "<script>",
])
def test_first_tx_non_address_funder_returns_none_with_one_warning(frm, debug_log):
    http = FakeAlchemyHttp([ok([transfer(frm=frm)])])
    assert client(http).first_tx(ADDR) is None
    warnings = [r for r in debug_log.records if r.levelno == logging.WARNING
                and r.name == "robustrep.sources.profile_http"]
    assert len(warnings) == 1
    assert ADDR.lower() in warnings[0].getMessage() and "funder" in warnings[0].getMessage()
    assert KEY not in debug_log.text
    assert len(http.calls) == 1


def test_first_tx_uppercase_funder_is_accepted_and_lowercased():
    http = FakeAlchemyHttp([ok([transfer(frm="0x" + "A" * 40)])])
    assert client(http).first_tx(ADDR)[2] == "0x" + "a" * 40


def test_first_tx_implausible_timestamp_takes_precedence_over_a_bad_funder(debug_log):
    # One WARNING, not two: the timestamp is checked first and the entry is
    # already refused.
    http = FakeAlchemyHttp([ok([transfer(ts="2200-01-01T00:00:00.000Z", frm=None)])])
    assert client(http).first_tx(ADDR) is None
    warnings = [r for r in debug_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "timestamp" in warnings[0].getMessage()


def test_first_tx_non_address_funder_never_reaches_the_store(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([fb("0xa")])
        http = FakeAlchemyHttp([ok([transfer(frm="0x" + "a" * 40 + "; drop table raters")])])
        assert enrich_raters(s, client(http)) == 1
        meta = s.load_rater_meta()
        # Same outcome as an implausible timestamp (M4): the address is
        # recorded as unprofiled -- no funder value and no timestamp cached.
        assert meta["funder"].isna().all() and meta["first_seen_ts"].isna().all()
        assert s.get_sync("rater_profile_mode") == "alchemy"


def test_first_tx_implausible_timestamp_never_reaches_the_store(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([fb("0xa")])
        http = FakeAlchemyHttp([ok([transfer(ts="2200-01-01T00:00:00.000Z")])])
        assert enrich_raters(s, client(http)) == 1
        assert s.load_rater_meta()["first_seen_ts"].isna().all()


# --- malformed bodies --------------------------------------------------------------------


@pytest.mark.parametrize("body", [
    ok([{"from": FUNDER, "metadata": {"blockTimestamp": "2026-01-01T00:00:00Z"}}]),  # no blockNum
    ok([{"blockNum": "0x1", "metadata": {"blockTimestamp": "2026-01-01T00:00:00Z"}}]),  # no from
    ok([{"blockNum": "0x1", "from": FUNDER}]),  # no metadata
    ok([{"blockNum": "0x1", "from": FUNDER, "metadata": "nope"}]),  # metadata not an object
    ok([{"blockNum": "zz", "from": FUNDER, "metadata": {"blockTimestamp": "2026-01-01T00:00:00Z"}}]),
    ok([{"blockNum": 5, "from": FUNDER, "metadata": {"blockTimestamp": "2026-01-01T00:00:00Z"}}]),
    ok(["not-an-object"]),
    {"jsonrpc": "2.0", "id": 1, "result": {"transfers": "not-a-list"}},
    {"jsonrpc": "2.0", "id": 1, "result": {}},
    {"jsonrpc": "2.0", "id": 1, "result": []},
    {"jsonrpc": "2.0", "id": 1},
])
def test_first_tx_malformed_entry_raises_value_error_naming_only_the_address(body, debug_log):
    http = FakeAlchemyHttp([body])
    with pytest.raises(ValueError) as exc:
        client(http).first_tx(ADDR)
    msg = str(exc.value)
    assert ADDR.lower() in msg
    assert "zz" not in msg and "nope" not in msg and "not-a-list" not in msg
    _assert_key_hidden(exc.value, debug_log)
    assert len(http.calls) == 1  # malformed is not transient


def test_first_tx_non_object_body_raises_value_error(debug_log):
    http = FakeAlchemyHttp([b"[1, 2, 3]"])
    with pytest.raises(ValueError) as exc:
        client(http).first_tx(ADDR)
    assert ADDR.lower() in str(exc.value)
    _assert_key_hidden(exc.value, debug_log)


# --- JSON-RPC error bodies ---------------------------------------------------------------


@pytest.mark.parametrize("message", [
    "Must be authenticated!",
    "Network Base is not enabled on this app (ref 12345)",
    "Invalid API key",
    "UNAUTHENTICATED: bad API KEY",
])
def test_first_tx_jsonrpc_auth_error_raises_config_error_immediately(message, debug_log):
    http = FakeAlchemyHttp([rpc_error(-32600, message)] * 3)
    with pytest.raises(AlchemyConfigError) as exc:
        client(http, retries=3).first_tx(ADDR)
    assert len(http.calls) == 1
    msg = str(exc.value)
    assert message not in msg and message.lower() not in msg.lower()
    assert "Alchemy" in msg and "ALCHEMY_API_KEY" in msg
    _assert_key_hidden(exc.value, debug_log)


def test_alchemy_config_error_message_is_fixed():
    http = FakeAlchemyHttp([rpc_error(1, "Must be authenticated!"), rpc_error(2, "api key bad")])
    c = client(http)
    with pytest.raises(AlchemyConfigError) as e1:
        c.first_tx(ADDR)
    with pytest.raises(AlchemyConfigError) as e2:
        c.first_tx("0xB")
    assert str(e1.value) == str(e2.value)


@pytest.mark.parametrize("err", [
    rpc_error(429, "Your app has exceeded its compute units per second capacity"),
    rpc_error(-32005, "rate limited"),
    rpc_error(-32000, "Throughput limit reached"),
    rpc_error("429", "over capacity"),
])
def test_first_tx_jsonrpc_rate_error_retried_then_exhausted(err, debug_log):
    http = FakeAlchemyHttp([err] * 3)
    with pytest.raises(RuntimeError) as exc:
        client(http, retries=3).first_tx(ADDR)
    assert not isinstance(exc.value, ProfilePlanError)
    assert len(http.calls) == 3
    # Exact: the fixed exhausted message -- no server text, no stale "status n/a".
    assert str(exc.value) == f"Alchemy body-level error for {ADDR.lower()} after 3 attempts"
    _assert_key_hidden(exc.value, debug_log)


def test_first_tx_jsonrpc_rate_error_then_success():
    http = FakeAlchemyHttp([rpc_error(429, "rate limited"), ok([transfer()])])
    assert client(http).first_tx(ADDR)[0] == 0x1a2b
    assert len(http.calls) == 2


@pytest.mark.parametrize("err", [
    rpc_error(429, "API key rate limit exceeded"),
    rpc_error(429, "Must be authenticated!"),  # code 429 outranks any config-looking text
    rpc_error(-32000, "API key throughput limit reached"),
])
def test_first_tx_rate_limit_outranks_config_rejected_text(err):
    # "API key rate limit exceeded" names the key *and* a rate limit; it is a
    # transient condition, not a rejected key, so it must be retried rather
    # than aborting the whole profiling step.
    http = FakeAlchemyHttp([err, ok([transfer()])])
    assert client(http).first_tx(ADDR) is not None
    assert len(http.calls) == 2


@pytest.mark.parametrize("message", [
    "rate limit exceeded", "Rate-Limit hit", "RATE LIMITED", "Throughput limit reached", "over capacity",
])
def test_first_tx_rate_limit_substrings_are_transient(message):
    http = FakeAlchemyHttp([rpc_error(-32000, message), ok([transfer()])])
    assert client(http).first_tx(ADDR) is not None
    assert len(http.calls) == 2


@pytest.mark.parametrize("message", [
    "could not enumerate transfers", "cannot operate on this address", "moderate load", "integrate first",
])
def test_first_tx_message_merely_containing_rate_fails_fast(message, debug_log):
    # A bare "rate" substring matched "enumerate"/"operate"/"moderate"; those
    # are permanent errors that must not burn the retry budget.
    http = FakeAlchemyHttp([rpc_error(-32000, message)] * 5)
    with pytest.raises(RuntimeError) as exc:
        client(http, retries=5).first_tx(ADDR)
    assert not isinstance(exc.value, ProfilePlanError)
    assert len(http.calls) == 1
    assert str(exc.value) == f"Alchemy JSON-RPC error for {ADDR.lower()}"
    _assert_key_hidden(exc.value, debug_log)


def test_first_tx_null_error_member_beside_a_result_is_a_success():
    body = {"jsonrpc": "2.0", "id": 1, "result": {"transfers": [transfer()]}, "error": None}
    assert client(FakeAlchemyHttp([body])).first_tx(ADDR)[0] == 0x1a2b


def test_first_tx_null_error_member_without_a_result_is_a_value_error():
    body = {"jsonrpc": "2.0", "id": 1, "error": None}
    with pytest.raises(ValueError) as exc:
        client(FakeAlchemyHttp([body])).first_tx(ADDR)
    assert ADDR.lower() in str(exc.value)


def test_first_tx_other_jsonrpc_error_raises_runtime_error_without_server_text(debug_log):
    http = FakeAlchemyHttp([rpc_error(-32602, "SERVER-SAID-SOMETHING-SPECIFIC")] * 2)
    with pytest.raises(RuntimeError) as exc:
        client(http).first_tx(ADDR)
    assert not isinstance(exc.value, ProfilePlanError)
    assert len(http.calls) == 1
    msg = str(exc.value)
    assert msg == f"Alchemy JSON-RPC error for {ADDR.lower()}"
    _assert_key_hidden(exc.value, debug_log)
    # The server text is available at DEBUG only, length-bounded.
    debug_records = [r for r in debug_log.records if r.levelno == logging.DEBUG
                     and r.name == "robustrep.sources.alchemy_profile"]
    assert any("SERVER-SAID-SOMETHING-SPECIFIC" in r.getMessage() for r in debug_records)
    assert not any("SERVER-SAID-SOMETHING-SPECIFIC" in r.getMessage()
                   for r in debug_log.records if r.levelno > logging.DEBUG)


def test_first_tx_jsonrpc_error_server_text_is_length_bounded_in_debug_log(debug_log):
    http = FakeAlchemyHttp([rpc_error(-32602, "x" * 100_000)])
    with pytest.raises(RuntimeError):
        client(http).first_tx(ADDR)
    records = [r for r in debug_log.records if r.name == "robustrep.sources.alchemy_profile"]
    assert records  # the server text was logged ...
    assert all(len(r.getMessage()) < 200 for r in records)  # ... but bounded (short_for_log)
    assert len(debug_log.text) < 2_000


@pytest.mark.parametrize("error", ["just a string", 42, 0, False, ["list"], {}])
def test_first_tx_jsonrpc_error_of_odd_shape_is_a_runtime_error(error, debug_log):
    http = FakeAlchemyHttp([{"jsonrpc": "2.0", "id": 1, "error": error}])
    with pytest.raises(RuntimeError) as exc:
        client(http).first_tx(ADDR)
    assert not isinstance(exc.value, ProfilePlanError)
    assert ADDR.lower() in str(exc.value)
    _assert_key_hidden(exc.value, debug_log)


def test_first_tx_jsonrpc_error_with_null_message_but_429_code_is_transient():
    http = FakeAlchemyHttp([{"jsonrpc": "2.0", "id": 1, "error": {"code": 429}}, ok([transfer()])])
    assert client(http).first_tx(ADDR) is not None
    assert len(http.calls) == 2


# --- HTTP-level errors -------------------------------------------------------------------


def test_first_tx_429_honors_retry_after_and_caps_it():
    sleeps = []
    http = FakeAlchemyHttp([(429, {"Retry-After": "7"}), (429, {"Retry-After": "3600"}), ok([transfer()])])
    c = AlchemyClient(KEY, session=http, rps=1.0, sleep=sleeps.append)
    assert c.first_tx(ADDR) is not None
    assert sleeps == [1.0, 7.0, 1.0, profile_http.RETRY_AFTER_CAP, 1.0]


def test_first_tx_429_without_retry_after_uses_exponential_backoff():
    sleeps = []
    http = FakeAlchemyHttp([429, 429, 429, ok([transfer()])])
    c = AlchemyClient(KEY, session=http, rps=1.0, sleep=sleeps.append)
    c.first_tx(ADDR)
    assert sleeps == [1.0, 1, 1.0, 2, 1.0, 4, 1.0]


def test_first_tx_backoff_is_capped_at_30s():
    sleeps = []
    http = FakeAlchemyHttp([503] * 7 + [ok([transfer()])])
    c = AlchemyClient(KEY, session=http, rps=1.0, retries=8, sleep=sleeps.append)
    c.first_tx(ADDR)
    backoffs = sleeps[1::2]
    assert backoffs == [1, 2, 4, 8, 16, 30, 30]


def test_first_tx_5xx_retried_then_runtime_error_with_status(debug_log):
    http = FakeAlchemyHttp([502, 503, 500])
    with pytest.raises(RuntimeError) as exc:
        client(http, retries=3).first_tx(ADDR)
    assert len(http.calls) == 3
    msg = str(exc.value)
    assert "status 500" in msg and ADDR.lower() in msg and "3 attempts" in msg
    _assert_key_hidden(exc.value, debug_log)


def test_first_tx_429_exhausted_names_status_429(debug_log):
    http = FakeAlchemyHttp([429, 429])
    with pytest.raises(RuntimeError) as exc:
        client(http, retries=2).first_tx(ADDR)
    assert "status 429" in str(exc.value)
    _assert_key_hidden(exc.value, debug_log)


@pytest.mark.parametrize("status,body", [
    (400, b""), (404, b""), (400, b"<html>bad request</html>"), (404, rpc_error(-32601, "method not found")),
    (403, b""), (403, b"<html>forbidden</html>"), (403, b"[1, 2]"), (403, b"null"),
    (403, rpc_error(-32602, "SERVER-SAID-SOMETHING-SPECIFIC")),
    (403, {"jsonrpc": "2.0", "id": 1, "error": "just a string"}),
    (403, {"jsonrpc": "2.0", "id": 1, "error": None, "result": None}),
    (403, rpc_error(429, "rate limited")),  # an HTTP 403 is permanent whatever the body claims
])
def test_first_tx_non_transient_4xx_raises_immediately_with_status(status, body, debug_log):
    http = FakeAlchemyHttp([(status, {}, body)] * 3)
    with pytest.raises(RuntimeError) as exc:
        client(http, retries=3).first_tx(ADDR)
    assert not isinstance(exc.value, ProfilePlanError)
    assert len(http.calls) == 1
    assert str(exc.value) == f"Alchemy HTTP {status} for {ADDR.lower()}"
    assert http.responses_made[0].closed is True
    _assert_key_hidden(exc.value, debug_log)
    assert exc.value.__suppress_context__  # the requests exception (URL + key) is not chained visibly
    for text in ("forbidden", "bad request", "SERVER-SAID", "just a string"):
        assert text not in str(exc.value)
        assert not any(text in r.getMessage() for r in debug_log.records if r.levelno > logging.DEBUG)


def test_first_tx_http_401_raises_config_error_on_the_first_attempt(debug_log):
    http = FakeAlchemyHttp([(401, {}, BODY_401)] * 3)
    with pytest.raises(AlchemyConfigError) as exc:
        client(http, retries=3).first_tx(ADDR)
    assert len(http.calls) == 1
    assert str(exc.value) == CONFIG_REJECTED_MESSAGE
    assert "authenticated" not in str(exc.value).lower()
    assert exc.value.__context__ is None and exc.value.__cause__ is None
    assert http.responses_made[0].closed is True
    _assert_key_hidden(exc.value, debug_log)


@pytest.mark.parametrize("body", [b"", b"<html>unauthorized</html>", b"[1]", b"null",
                                  rpc_error(-32602, "unrelated"), {"jsonrpc": "2.0", "id": 1}])
def test_first_tx_http_401_is_a_config_error_regardless_of_body(body, debug_log):
    http = FakeAlchemyHttp([(401, {}, body)])
    with pytest.raises(AlchemyConfigError):
        client(http).first_tx(ADDR)
    assert len(http.calls) == 1
    assert "unauthorized" not in debug_log.text
    assert KEY not in debug_log.text


def test_first_tx_http_401_without_a_readable_response_is_still_a_config_error():
    # The session itself raised the HTTPError (a response hook, say): there is
    # no response object to read a body from, only the status.
    err = _http_error(401, f"{ALCHEMY_BASE_URL}/{KEY}")
    http = FakeAlchemyHttp([err])
    with pytest.raises(AlchemyConfigError) as exc:
        client(http).first_tx(ADDR)
    assert KEY not in str(exc.value) and exc.value.__context__ is None


def test_first_tx_http_403_not_enabled_body_raises_config_error(debug_log):
    http = FakeAlchemyHttp([(403, {}, BODY_403_NOT_ENABLED)] * 3)
    with pytest.raises(AlchemyConfigError) as exc:
        client(http, retries=3).first_tx(ADDR)
    assert len(http.calls) == 1
    assert str(exc.value) == CONFIG_REJECTED_MESSAGE
    # The server's text (which names a dashboard URL) is DEBUG-only.
    assert "dashboard.alchemy.com" not in str(exc.value)
    assert not any("dashboard" in r.getMessage() for r in debug_log.records if r.levelno > logging.DEBUG)
    assert any("not enabled" in r.getMessage() for r in debug_log.records
               if r.levelno == logging.DEBUG and r.name == "robustrep.sources.alchemy_profile")
    _assert_key_hidden(exc.value, debug_log)


@pytest.mark.parametrize("status", [400, 403, 404, 418])
@pytest.mark.parametrize("message", ["Must be authenticated!", "Invalid API key",
                                     "BASE_MAINNET is not enabled for this app."])
def test_first_tx_any_non_transient_4xx_with_a_config_rejected_body_is_a_config_error(status, message):
    http = FakeAlchemyHttp([(status, {}, rpc_error(-32600, message))])
    with pytest.raises(AlchemyConfigError):
        client(http).first_tx(ADDR)
    assert len(http.calls) == 1


def test_first_tx_http_403_oversized_body_raises_the_limit_error(monkeypatch, debug_log):
    monkeypatch.setattr(profile_http, "MAX_PROFILE_RESPONSE_BYTES", 1024)
    http = FakeAlchemyHttp([(403, {}, b'{"error": {"message": "' + b"SECRETBODY" * 2000 + b'"}}')] * 2)
    with pytest.raises(RuntimeError) as exc:
        client(http).first_tx(ADDR)
    assert not isinstance(exc.value, ProfilePlanError)
    assert "1024 byte limit" in str(exc.value) and "SECRETBODY" not in str(exc.value)
    assert "SECRETBODY" not in debug_log.text
    assert len(http.calls) == 1


def test_first_tx_transient_statuses_never_read_the_body():
    # 429/5xx are classified from the status alone; the fakes' _ExplodingRaw
    # turns any read into an AssertionError.
    http = FakeAlchemyHttp([429, 503, ok([transfer()])])
    assert client(http).first_tx(ADDR) is not None


def test_first_tx_connection_error_retried_then_success(debug_log):
    http = FakeAlchemyHttp([requests.ConnectionError(f"boom {ALCHEMY_BASE_URL}/{KEY}"),
                            requests.Timeout(f"slow {ALCHEMY_BASE_URL}/{KEY}"),
                            ok([transfer()])])
    assert client(http).first_tx(ADDR) is not None
    assert len(http.calls) == 3
    assert KEY not in debug_log.text


def test_first_tx_connection_error_exhausted_says_status_n_a(debug_log):
    http = FakeAlchemyHttp([requests.ConnectionError(f"boom {ALCHEMY_BASE_URL}/{KEY}")] * 2)
    with pytest.raises(RuntimeError) as exc:
        client(http, retries=2).first_tx(ADDR)
    assert "status n/a" in str(exc.value)
    _assert_key_hidden(exc.value, debug_log)


def test_first_tx_non_json_body_retried_then_value_error(debug_log):
    http = FakeAlchemyHttp([b"<html>cloudflare</html>", b"<html>cloudflare</html>"])
    with pytest.raises(ValueError) as exc:
        client(http, retries=2).first_tx(ADDR)
    assert len(http.calls) == 2
    msg = str(exc.value)
    assert msg == f"non-JSON Alchemy response for {ADDR.lower()}"
    assert "cloudflare" not in msg
    _assert_key_hidden(exc.value, debug_log)


def test_first_tx_non_json_then_json_succeeds():
    http = FakeAlchemyHttp([b"<html>", ok([transfer()])])
    assert client(http).first_tx(ADDR) is not None


def test_first_tx_oversized_body_raises_runtime_error_naming_limit_not_retried(monkeypatch, debug_log):
    monkeypatch.setattr(profile_http, "MAX_PROFILE_RESPONSE_BYTES", 1024)
    http = FakeAlchemyHttp([b'{"result": {"transfers": [], "leak": "' + b"SECRETBODY" * 2000 + b'"}}'] * 3)
    with pytest.raises(RuntimeError) as exc:
        client(http, retries=3).first_tx(ADDR)
    msg = str(exc.value)
    assert "1024 byte limit" in msg and ADDR.lower() in msg and "Alchemy" in msg
    assert "SECRETBODY" not in msg and "SECRETBODY" not in debug_log.text
    assert len(http.calls) == 1
    _assert_key_hidden(exc.value, debug_log)


def test_first_tx_closes_response_on_every_path():
    http = FakeAlchemyHttp([b"<html>", rpc_error(429, "rate"), ok([transfer()])])
    client(http).first_tx(ADDR)
    assert all(r.closed for r in http.responses_made)


# --- enrich_raters -----------------------------------------------------------------------


def test_enrich_records_alchemy_mode(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([fb("0xa")])
        http = FakeAlchemyHttp([ok([transfer(block_hex="0x3", ts="2026-01-01T00:00:00.000Z", frm=FUNDER_UP)])])
        assert enrich_raters(s, client(http)) == 1
        m = s.load_rater_meta().iloc[0]
        assert m["funder"] == FUNDER
        assert m["first_seen_ts"] == int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
        assert s.get_sync("rater_profile_mode") == "alchemy"


def test_enrich_aborts_on_first_address_with_config_error(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([fb("0xa", log_index=0), fb("0xb", log_index=1), fb("0xc", log_index=2)])
        http = FakeAlchemyHttp([rpc_error(1, "Must be authenticated!")] * 3)
        with pytest.raises(AlchemyConfigError):
            enrich_raters(s, client(http))
        assert len(http.calls) == 1
        assert len(s.load_rater_meta()) == 0
        assert s.get_sync("rater_profile_mode") is None


@pytest.mark.parametrize("item", [(401, {}, BODY_401), (403, {}, BODY_403_NOT_ENABLED)])
def test_enrich_aborts_on_first_address_with_http_config_rejection(tmp_path, item, caplog):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([fb("0xa", log_index=0), fb("0xb", log_index=1), fb("0xc", log_index=2)])
        http = FakeAlchemyHttp([item] * 3)
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(AlchemyConfigError) as exc:
                enrich_raters(s, client(http))
        assert str(exc.value) == CONFIG_REJECTED_MESSAGE
        assert len(http.calls) == 1
        assert len(s.load_rater_meta()) == 0
        assert s.get_sync("rater_profile_mode") is None
        assert KEY not in caplog.text
        assert not any("failed to enrich" in r.getMessage() for r in caplog.records)


def test_enrich_config_error_on_later_address_is_a_regular_failure(tmp_path, caplog):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([fb("0xa", log_index=0), fb("0xb", log_index=1)])
        http = FakeAlchemyHttp([ok([transfer()]), rpc_error(1, "Must be authenticated!")])
        with caplog.at_level(logging.ERROR, logger="robustrep.sources.rater_profile"):
            with pytest.raises(RuntimeError) as exc:
                enrich_raters(s, client(http))
        assert not isinstance(exc.value, ProfilePlanError)
        assert "0xb" in str(exc.value)
        assert set(s.load_rater_meta()["rater"]) == {"0xa"}
        assert s.get_sync("rater_profile_mode") == "alchemy-partial"
        assert KEY not in caplog.text


def test_enrich_partial_failure_log_never_carries_the_key(tmp_path, caplog):
    with Store(tmp_path / "t.db") as s:
        s.upsert_feedback([fb("0xa", log_index=0), fb("0xb", log_index=1)])
        http = FakeAlchemyHttp([(403, {}, b"<html>forbidden</html>"), ok([transfer()])])
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(RuntimeError):
                enrich_raters(s, client(http))
        assert KEY not in caplog.text
        assert "forbidden" not in caplog.text
        assert any("failed to enrich" in r.getMessage() for r in caplog.records)


# --- default_client ----------------------------------------------------------------------


def test_default_client_alchemy_with_key():
    c = default_client("alchemy", None, alchemy_key=KEY)
    assert isinstance(c, AlchemyClient)
    assert c.source == "alchemy"
    assert c.gap == 1 / DEFAULT_ALCHEMY_RPS and c.retries == 5


@pytest.mark.parametrize("bad", [None, "", "  "])
def test_default_client_alchemy_without_key_raises_value_error(bad):
    with pytest.raises(ValueError) as exc:
        default_client("alchemy", "ETHERSCAN-KEY", alchemy_key=bad)
    assert str(exc.value) == "--profile-source alchemy requires ALCHEMY_API_KEY"


def test_default_client_alchemy_ignores_etherscan_key():
    c = default_client("alchemy", "ETHERSCAN-KEY", alchemy_key=KEY)
    assert isinstance(c, AlchemyClient)


def test_default_client_alchemy_passes_through_rps_and_retries():
    c = default_client("alchemy", None, rps=1.0, retries=2, alchemy_key=KEY)
    assert isinstance(c, AlchemyClient)
    assert c.gap == 1.0 and c.retries == 2


def test_default_client_auto_etherscan_key_wins_over_alchemy():
    c = default_client("auto", "ETHERSCAN-KEY", alchemy_key=KEY)
    assert isinstance(c, EtherscanClient) and c.source == "etherscan"


def test_default_client_auto_alchemy_key_wins_over_blockscout():
    c = default_client("auto", None, alchemy_key=KEY)
    assert isinstance(c, AlchemyClient)


def test_default_client_auto_alchemy_passes_through_rps_and_retries():
    c = default_client("auto", None, rps=0.5, retries=9, alchemy_key=KEY)
    assert isinstance(c, AlchemyClient)
    assert c.gap == 2.0 and c.retries == 9


def test_default_client_auto_without_any_key_uses_blockscout():
    c = default_client("auto", None, alchemy_key=None)
    assert isinstance(c, BlockscoutV2Client)
    assert isinstance(default_client("auto", None, alchemy_key="  "), BlockscoutV2Client)


def test_default_client_positional_signature_still_works():
    assert isinstance(default_client("blockscout", None, 2.0, 3), BlockscoutV2Client)
    assert default_client("none", None, None, None, KEY) is None


def test_default_client_blockscout_and_etherscan_ignore_alchemy_key():
    assert isinstance(default_client("blockscout", None, alchemy_key=KEY), BlockscoutV2Client)
    assert isinstance(default_client("etherscan", "E", alchemy_key=KEY), EtherscanClient)


def test_module_exposes_source_label_used_by_enrich_raters():
    assert alchemy_profile.AlchemyClient.source == "alchemy"


def test_first_tx_http_401_oversized_body_is_still_a_config_error(monkeypatch, debug_log):
    # I-1: 401 means "bad key" whatever the body says, so the body must not be
    # read at all -- an oversized one must not turn the abort into a per-address
    # limit error (which would make enrich_raters fail ~13,000 addresses one by
    # one, reading up to 8 MiB each).
    monkeypatch.setattr(profile_http, "MAX_PROFILE_RESPONSE_BYTES", 1024)
    http = FakeAlchemyHttp([(401, {}, b'{"error": {"message": "' + b"SECRETBODY" * 2000 + b'"}}')] * 2)
    with pytest.raises(AlchemyConfigError) as exc:
        client(http).first_tx(ADDR)
    assert str(exc.value) == CONFIG_REJECTED_MESSAGE
    assert len(http.calls) == 1
    assert "SECRETBODY" not in debug_log.text
    assert http.responses_made[0].closed is True


# --- no categories parameter: the internal-transfer retry was dropped --------------


def test_first_tx_has_no_categories_parameter_and_asks_for_external_only():
    import inspect

    assert "categories" not in inspect.signature(AlchemyClient.first_tx).parameters
    http = FakeAlchemyHttp([ok([transfer()])])
    assert client(http).first_tx(ADDR) is not None
    assert http.calls[0]["json"]["params"][0]["category"] == ["external"]
