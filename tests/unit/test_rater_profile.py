import json
import logging
from datetime import datetime, timezone

import pytest
import requests

from robustrep.sources import http_util, rater_profile
from robustrep.sources.rater_profile import (
    BLOCKSCOUT_BASE,
    BLOCKSCOUT_V2_BASE,
    BlockscoutV2Client,
    EtherscanClient,
    EtherscanPlanError,
    client_from_env,
    default_client,
    enrich_raters,
    estimate_seconds,
)
from robustrep.store import Store

FB = dict(chain="base", tx_hash="0x", value="1", value_decimals=0, tag1="", tag2="", endpoint="",
          feedback_uri="", feedback_hash="")


def fb(client, block=1, log_index=0, agent_id="1", feedback_index=0):
    return dict(FB, block=block, log_index=log_index, agent_id=agent_id, client=client,
                feedback_index=feedback_index)


class _FakeRaw:
    """Stand-in for ``urllib3.HTTPResponse``: the streamed byte source
    ``read_json_capped`` consumes (bodies are capped, never ``r.json()``-ed
    -- M2)."""

    def __init__(self, data: bytes):
        self.data = data

    def read(self, amt, decode_content=True):
        chunk, self.data = self.data[:amt], self.data[amt:]
        return chunk


class _ExplodingRaw:
    """A body that must never be read (the response already errored)."""

    def read(self, amt, decode_content=True):
        raise AssertionError("body must not be read when raise_for_status() raises")


class _FakeResponse:
    """Streamed stand-in for ``requests.Response``: exposes ``raw`` (what
    ``read_json_capped`` reads), ``raise_for_status`` and ``close``."""

    def __init__(self, payload=None, *, body=None, status_code=200, error=None):
        self.status_code, self._error, self.closed = status_code, error, False
        if error is not None:
            self.raw = _ExplodingRaw()
        else:
            self.raw = _FakeRaw(body if body is not None else json.dumps(payload).encode())

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def close(self):
        self.closed = True


def _http_error(status_code, message, headers=None):
    err = requests.HTTPError(message)
    resp = requests.Response()
    resp.status_code = status_code
    resp.headers.update(headers or {})
    err.response = resp
    return err


class FakeHttp:
    """Simulates a JSON body response (success or Etherscan-level error body).

    Each item in ``payloads`` is a JSON-able object, or raw ``bytes`` for a
    body that is malformed or deliberately oversized."""

    def __init__(self, payloads):
        self.payloads, self.calls, self.streams, self.responses = list(payloads), [], [], []

    def get(self, url, params=None, timeout=None, stream=None):
        self.calls.append(params)
        self.streams.append(stream)
        p = self.payloads.pop(0)
        r = _FakeResponse(body=p) if isinstance(p, bytes) else _FakeResponse(p)
        self.responses.append(r)
        return r


class FakeHttpRaisingOnGet:
    """Simulates a transport-level failure (e.g. ConnectionError) raised by ``get`` itself."""

    def __init__(self, exc):
        self.exc, self.calls = exc, []

    def get(self, url, params=None, timeout=None, stream=None):
        self.calls.append(params)
        raise self.exc


class FakeHttpRaisingOnStatus:
    """Simulates an HTTP error surfaced via ``raise_for_status`` (e.g. 429/404).

    ``headers``, when given, is attached to the simulated response (e.g.
    ``{"Retry-After": "5"}``) so tests can exercise the Retry-After path."""

    def __init__(self, status_code, message, headers=None):
        self.status_code, self.message, self.headers, self.calls = status_code, message, headers or {}, []

    def get(self, url, params=None, timeout=None, stream=None):
        self.calls.append(params)
        return _FakeResponse(status_code=self.status_code,
                             error=_http_error(self.status_code, self.message, self.headers))


class FakeHttpBadJson:
    """Simulates a 200 response whose body is not valid JSON."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None, stream=None):
        self.calls.append(params)
        return _FakeResponse(body=b"<html>not json</html>")


class FakeV2Http:
    """Simulates a sequence of Blockscout v2 page responses. Each item in
    ``responses`` is either a dict body (200 OK), an int HTTP status code
    (429/5xx/404) simulated via ``raise_for_status``, or a ``(status,
    headers)`` tuple (same, with response headers attached -- e.g.
    ``(429, {"Retry-After": "5"})``)."""

    def __init__(self, responses):
        self.responses, self.calls, self.streams = list(responses), [], []

    def get(self, url, params=None, timeout=None, stream=None):
        self.calls.append(dict(params or {}))
        self.streams.append(stream)
        item = self.responses.pop(0)
        if isinstance(item, (int, tuple)):
            status, headers = item if isinstance(item, tuple) else (item, {})
            return _FakeResponse(
                status_code=status,
                error=_http_error(status, f"{status} error for url: {url}?apikey=SECRET", headers))
        if isinstance(item, bytes):
            return _FakeResponse(body=item)
        return _FakeResponse(item)


class FakeV2HttpConnErr:
    """Simulates a transport-level failure (e.g. ConnectionError) raised by
    ``get`` itself, then a successful page."""

    def __init__(self, exc, then_body):
        self.exc, self.then_body, self.calls = exc, then_body, []
        self._raised = False

    def get(self, url, params=None, timeout=None, stream=None):
        self.calls.append(dict(params or {}))
        if not self._raised:
            self._raised = True
            raise self.exc
        return _FakeResponse(self.then_body)


class FakeV2HttpBadJson:
    """Simulates a 200 response whose body is not valid JSON."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None, stream=None):
        self.calls.append(dict(params or {}))
        return _FakeResponse(body=b"<html>not json</html>")


def bs_tx(hash_, block, ts, from_hash, to_hash="0x6974"):
    return {"hash": hash_, "block_number": block, "timestamp": ts,
            "from": {"hash": from_hash}, "to": {"hash": to_hash}}


# --- EtherscanClient.first_tx: happy paths -----------------------------------------


def test_first_tx_parses_etherscan_v2():
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "10", "timeStamp": "1700", "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    assert c.first_tx("0xA") == (10, 1700, "0xf")
    assert http.calls[0]["chainid"] == 8453 and http.calls[0]["apikey"] == "KEY"  # pragma: allowlist secret


def test_first_tx_none_when_empty():
    c = EtherscanClient("KEY", session=FakeHttp([{"status": "0", "result": []}]), sleep=lambda _: None)
    assert c.first_tx("0xA") is None


def test_first_tx_none_when_status_ok_but_result_empty():
    c = EtherscanClient("KEY", session=FakeHttp([{"status": "1", "result": []}]), sleep=lambda _: None)
    assert c.first_tx("0xA") is None


def test_first_tx_none_when_no_transactions_found_message():
    c = EtherscanClient("KEY", session=FakeHttp([{"status": "0", "message": "NOTOK",
                                                    "result": "No transactions found"}]), sleep=lambda _: None)
    assert c.first_tx("0xA") is None


def test_first_tx_lowercases_request_address():
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "1", "timeStamp": "1", "from": "0xf"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    c.first_tx("0xABCDEF")
    assert http.calls[0]["address"] == "0xabcdef"


def test_first_tx_never_logs_params(caplog):
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "1", "timeStamp": "1", "from": "0xf"}]}])
    c = EtherscanClient("SECRET-KEY", session=http, sleep=lambda _: None)
    with caplog.at_level(logging.DEBUG):
        c.first_tx("0xA")
    assert "SECRET-KEY" not in caplog.text


# --- EtherscanClient.blockscout(): keyless Base Blockscout endpoint ---------------


def test_blockscout_request_has_no_chainid_or_apikey_and_hits_blockscout_url():
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "42502816", "timeStamp": "1771794979",
                                                   "from": "0x6463"}]}])
    c = EtherscanClient.blockscout(session=http, sleep=lambda _: None)
    assert c.first_tx("0xA") == (42502816, 1771794979, "0x6463")
    assert http.calls[0] == {"module": "account", "action": "txlist", "address": "0xa",
                              "page": 1, "offset": 1, "sort": "asc"}
    assert "chainid" not in http.calls[0]
    assert "apikey" not in http.calls[0]


def test_blockscout_client_has_no_key_and_targets_blockscout_base_url():
    c = EtherscanClient.blockscout(sleep=lambda _: None)
    assert c.key is None
    assert c.chain_id is None
    assert c.base_url == BLOCKSCOUT_BASE
    assert c.source == "blockscout"


def test_blockscout_empty_result_returns_none_like_etherscan():
    c = EtherscanClient.blockscout(
        session=FakeHttp([{"status": "0", "message": "No transactions found", "result": []}]),
        sleep=lambda _: None)
    assert c.first_tx("0xA") is None


def test_default_client_etherscan_still_sends_chainid_and_apikey():
    c = default_client("etherscan", "KEY")
    assert c.source == "etherscan"
    assert c.chain_id == 8453
    assert c.key == "KEY"
    assert c.base_url == EtherscanClient(api_key="KEY").base_url


# --- malformed address: ValueError, not None ---------------------------------------


def test_first_tx_invalid_address_format_raises_value_error_naming_address():
    http = FakeHttp([{"message": "Invalid address format", "result": None, "status": "0"}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    try:
        c.first_tx("0xBAD")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "0xbad" in str(e).lower()
        assert "invalid address" in str(e).lower()
    assert len(http.calls) == 1  # non-retryable: no retry burned


# --- unsupported plan (e.g. Etherscan free plan on Base): non-retryable ------------


def test_first_tx_free_plan_unsupported_chain_raises_plan_error_mentioning_plan():
    http = FakeHttp([{"status": "0", "message": "NOTOK",
                       "result": "Free API access is not supported for this chain, "
                                 "please subscribe to a plan"}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected EtherscanPlanError"
    except EtherscanPlanError as e:
        assert isinstance(e, RuntimeError)
        assert "plan" in str(e).lower()
    assert len(http.calls) == 1  # non-retryable: no retry burned


# --- transient errors: retried with backoff, throttle in finally ------------------


def test_first_tx_retries_body_rate_limit_then_raises():
    payload = {"status": "0", "message": "NOTOK", "result": "Max rate limit reached"}
    http = FakeHttp([payload, payload, payload])
    sleeps = []
    c = EtherscanClient("KEY", session=http, sleep=sleeps.append)
    try:
        c.first_tx("0xA")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "0xa" in str(e).lower()
        assert "n/a" in str(e)
    assert len(http.calls) == 3
    # one rps-throttle sleep per attempt (in `finally`, always paid) plus backoff
    # sleeps between attempts 1->2 and 2->3 (none after the final, exhausted attempt).
    assert len(sleeps) == 5


def test_first_tx_scrubs_429_http_error_and_retries(caplog):
    secret_url = "https://api.etherscan.io/v2/api?chainid=8453&apikey=SECRET"  # pragma: allowlist secret
    http = FakeHttpRaisingOnStatus(429, f"429 Client Error: Too Many Requests for url: {secret_url}")
    sleeps = []
    c = EtherscanClient("SECRET", session=http, retries=3, sleep=sleeps.append)
    with caplog.at_level(logging.DEBUG):
        try:
            c.first_tx("0xA")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            msg = str(e)
    assert "429" in msg and "0xa" in msg.lower()
    assert "SECRET" not in msg
    assert "SECRET" not in caplog.text
    assert len(http.calls) == 3
    assert len(sleeps) > 3  # throttle sleeps (one per attempt) plus >=1 backoff sleep


# --- 429 Retry-After header: honored (capped at 60s) instead of exponential backoff --


def test_first_tx_429_honors_retry_after_header():
    http = FakeHttpRaisingOnStatus(429, "429 too many requests", headers={"Retry-After": "5"})
    sleeps = []
    c = EtherscanClient("KEY", session=http, retries=2, sleep=sleeps.append)
    try:
        c.first_tx("0xA")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    # Two throttle sleeps (one per attempt, `self.gap`) plus one backoff sleep of
    # exactly the Retry-After value (not the exponential min(2**attempt, 30)).
    assert sleeps.count(5.0) == 1


def test_first_tx_429_retry_after_capped_at_60():
    http = FakeHttpRaisingOnStatus(429, "429 too many requests", headers={"Retry-After": "600"})
    sleeps = []
    c = EtherscanClient("KEY", session=http, retries=2, sleep=sleeps.append)
    try:
        c.first_tx("0xA")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert sleeps.count(60.0) == 1
    assert 600.0 not in sleeps


def test_first_tx_429_without_retry_after_falls_back_to_exponential_backoff():
    http = FakeHttpRaisingOnStatus(429, "429 too many requests")
    sleeps = []
    c = EtherscanClient("KEY", session=http, retries=2, sleep=sleeps.append)
    try:
        c.first_tx("0xA")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    # attempt 0's backoff, with no header present, is min(2**0, 30) == 1.0.
    assert sleeps.count(1.0) == 1


def test_first_tx_scrubs_5xx_http_error_and_retries():
    http = FakeHttpRaisingOnStatus(503, "503 Server Error")
    c = EtherscanClient("KEY", session=http, retries=2, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "503" in str(e)
    assert len(http.calls) == 2


def test_first_tx_scrubs_connection_error(caplog):
    exc = requests.ConnectionError(
        "HTTPSConnectionPool: Max retries exceeded ... apikey=SECRET ...")
    http = FakeHttpRaisingOnGet(exc)
    sleeps = []
    c = EtherscanClient("SECRET", session=http, retries=2, sleep=sleeps.append)
    with caplog.at_level(logging.DEBUG):
        try:
            c.first_tx("0xA")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            msg = str(e)
    assert "SECRET" not in msg
    assert "SECRET" not in caplog.text
    assert "n/a" in msg
    assert len(http.calls) == 2


def test_first_tx_non_json_body_retries_then_raises_value_error():
    http = FakeHttpBadJson()
    sleeps = []
    c = EtherscanClient("KEY", session=http, retries=2, sleep=sleeps.append)
    try:
        c.first_tx("0xA")
        assert False, "expected ValueError"
    except ValueError as e:
        msg = str(e)
        assert "0xa" in msg.lower()
        assert "Expecting value" not in msg
    assert len(http.calls) == 2


# --- non-retryable errors: raised immediately --------------------------------------


def test_first_tx_invalid_api_key_raises_immediately():
    http = FakeHttp([{"status": "0", "message": "NOTOK", "result": "Invalid API Key"}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "API Key" in str(e)
    assert len(http.calls) == 1


def test_first_tx_non_retryable_4xx_raises_immediately():
    http = FakeHttpRaisingOnStatus(404, "404 Client Error: Not Found")
    c = EtherscanClient("KEY", session=http, retries=3, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "404" in str(e) and "0xa" in str(e).lower()
    assert len(http.calls) == 1


def test_first_tx_non_dict_body_raises_value_error():
    c = EtherscanClient("KEY", session=FakeHttp([None]), sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "0xa" in str(e).lower()


def test_first_tx_missing_status_raises_value_error():
    c = EtherscanClient("KEY", session=FakeHttp([{"result": []}]), sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "0xa" in str(e).lower()


def test_first_tx_status_ok_non_list_result_raises_value_error():
    c = EtherscanClient("KEY", session=FakeHttp([{"status": "1", "result": "oops"}]), sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "0xa" in str(e).lower()


def test_first_tx_malformed_entry_missing_from_raises_value_error():
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "99"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "0xa" in str(e).lower()


def test_first_tx_unparseable_timestamp_returns_none_with_one_warning(caplog):
    # M4: an unparseable third-party timestamp must not reach the raters cache
    # (a bad first_seen_ts there wedges every later score/report run), and must
    # not raise either -- the address simply goes unprofiled.
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "not-a-number", "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.rater_profile"):
        assert c.first_tx("0xA") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "not-a-number" in warnings[0].getMessage() and "0xa" in warnings[0].getMessage()


# --- estimate_seconds ---------------------------------------------------------------


def test_estimate_seconds():
    assert estimate_seconds(100_000, 4.0) == 25_000.0


def test_estimate_seconds_rejects_non_positive_rps():
    try:
        estimate_seconds(10, 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- enrich_raters: etherscan mode ---------------------------------------------------


def test_enrich_with_client(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    c = EtherscanClient("K", session=FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "99", "from": "0xF"}]}]),
                        sleep=lambda _: None)
    assert enrich_raters(s, c) == 1
    m = s.load_rater_meta().iloc[0]
    assert m["first_seen_ts"] == 99 and m["funder"] == "0xf"
    assert s.get_sync("rater_profile_mode") == "etherscan"


def test_enrich_with_client_no_tx_found_upserts_none(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    c = EtherscanClient("K", session=FakeHttp([{"status": "0", "result": []}]), sleep=lambda _: None)
    assert enrich_raters(s, c) == 1
    m = s.load_rater_meta().iloc[0]
    assert m["first_seen_ts"] is None and m["funder"] is None
    assert s.get_sync("rater_profile_mode") == "etherscan"


def test_enrich_with_client_collects_failures_marks_partial_mode(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa", log_index=0), fb("0xb", log_index=1), fb("0xc", log_index=2)])

    class FlakyClient:
        def first_tx(self, address):
            if address == "0xb":
                raise RuntimeError("boom")
            return (1, 100, "0xf")

    try:
        enrich_raters(s, FlakyClient())
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "0xb" in str(e)
    meta = s.load_rater_meta()
    assert set(meta["rater"]) == {"0xa", "0xc"}
    assert len(meta) == 2
    assert s.get_sync("rater_profile_mode") == "etherscan-partial"


def test_enrich_with_client_failure_message_lists_first_five_plus_more(tmp_path):
    s = Store(tmp_path / "t.db")
    addrs = [f"0xf{i}" for i in range(7)]
    s.upsert_feedback([fb(a, log_index=i) for i, a in enumerate(addrs)])

    class AlwaysFailsClient:
        def first_tx(self, address):
            raise RuntimeError("boom")

    try:
        enrich_raters(s, AlwaysFailsClient())
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        msg = str(e)
    assert "7 address(es)" in msg
    assert "(+2 more)" in msg
    assert s.get_sync("rater_profile_mode") == "etherscan-partial"
    assert len(s.load_rater_meta()) == 0


def test_enrich_logs_scrubbed_failure_message(tmp_path, caplog):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    http = FakeHttpRaisingOnStatus(429, "429 ... apikey=SECRET ...")
    c = EtherscanClient("SECRET", session=http, retries=1, sleep=lambda _: None)
    with caplog.at_level(logging.ERROR, logger="robustrep.sources.rater_profile"):
        try:
            enrich_raters(s, c)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
    assert "SECRET" not in caplog.text
    assert any("failed to enrich" in r.message for r in caplog.records)


def test_enrich_with_client_is_idempotent(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    c = EtherscanClient("K", session=FakeHttp([{"status": "1", "result": [{"blockNumber": "1", "timeStamp": "5", "from": "0xf"}]}]),
                        sleep=lambda _: None)
    assert enrich_raters(s, c) == 1
    assert enrich_raters(s, c) == 0


# --- enrich_raters: blockscout mode ---------------------------------------------------


def test_enrich_with_blockscout_client_sets_blockscout_mode(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "99", "from": "0xF"}]}])
    c = EtherscanClient.blockscout(session=http, sleep=lambda _: None)
    assert enrich_raters(s, c) == 1
    m = s.load_rater_meta().iloc[0]
    assert m["first_seen_ts"] == 99 and m["funder"] == "0xf"
    assert s.get_sync("rater_profile_mode") == "blockscout"


def test_enrich_with_blockscout_client_partial_failure_sets_blockscout_partial_mode(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa", log_index=0), fb("0xb", log_index=1)])

    class FlakyBlockscoutClient:
        source = "blockscout"

        def first_tx(self, address):
            if address == "0xb":
                raise RuntimeError("boom")
            return (1, 100, "0xf")

    try:
        enrich_raters(s, FlakyBlockscoutClient())
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert s.get_sync("rater_profile_mode") == "blockscout-partial"


# --- enrich_raters: EtherscanPlanError aborts on the first address -----------------


def test_enrich_aborts_immediately_when_first_address_hits_plan_error(tmp_path):
    from robustrep.sources.rater_profile import EtherscanPlanError

    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa", log_index=0), fb("0xb", log_index=1), fb("0xc", log_index=2)])

    calls = []

    class PlanRejectedClient:
        def first_tx(self, address):
            calls.append(address)
            raise EtherscanPlanError(f"Etherscan plan does not support this chain for {address}")

    try:
        enrich_raters(s, PlanRejectedClient())
        assert False, "expected EtherscanPlanError"
    except EtherscanPlanError as e:
        assert "plan" in str(e).lower()
    # Only the first address was attempted -- no burning through the rest.
    assert len(calls) == 1
    assert len(s.load_rater_meta()) == 0
    # Aborted before any mode was recorded for this run.
    assert s.get_sync("rater_profile_mode") is None


def test_enrich_plan_error_on_later_address_is_a_regular_failure(tmp_path):
    from robustrep.sources.rater_profile import EtherscanPlanError

    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa", log_index=0), fb("0xb", log_index=1)])

    class PlanRejectedOnSecondClient:
        source = "etherscan"

        def first_tx(self, address):
            if address == "0xb":
                raise EtherscanPlanError(f"Etherscan plan does not support this chain for {address}")
            return (1, 100, "0xf")

    try:
        enrich_raters(s, PlanRejectedOnSecondClient())
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "0xb" in str(e)
    meta = s.load_rater_meta()
    assert set(meta["rater"]) == {"0xa"}
    assert s.get_sync("rater_profile_mode") == "etherscan-partial"


# --- enrich_raters: fallback mode (no client) -- no sticky rows ---------------------


def test_enrich_fallback_writes_no_rows_but_returns_informational_count(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    s.upsert_block_ts([(1, 555)])
    assert enrich_raters(s, None) == 1
    assert len(s.load_rater_meta()) == 0
    assert s.get_sync("rater_profile_mode") == "fallback"


def test_enrich_fallback_warns_once_when_block_ts_missing(tmp_path, caplog):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa", block=1), fb("0xb", block=2, log_index=1)])
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.rater_profile"):
        assert enrich_raters(s, None) == 2
    assert len(s.load_rater_meta()) == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "fill_block_timestamps" in warnings[0].message


def test_enrich_fallback_no_warning_when_block_ts_cached(tmp_path, caplog):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    s.upsert_block_ts([(1, 555)])
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.rater_profile"):
        enrich_raters(s, None)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_enrich_fallback_returns_same_count_every_call_not_sticky(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    s.upsert_block_ts([(1, 555)])
    assert enrich_raters(s, None) == 1
    # No rows were written, so the address is still "unprofiled" -- a later
    # run (fallback again, or with a real key) can still see and profile it.
    assert enrich_raters(s, None) == 1
    assert len(s.load_rater_meta()) == 0


def test_keyed_run_after_fallback_upgrades_rows(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    s.upsert_block_ts([(1, 555)])

    assert enrich_raters(s, None) == 1
    assert len(s.load_rater_meta()) == 0
    assert s.get_sync("rater_profile_mode") == "fallback"

    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "99", "from": "0xF"}]}])
    c = EtherscanClient("K", session=http, sleep=lambda _: None)
    assert enrich_raters(s, c) == 1
    assert len(http.calls) == 1
    m = s.load_rater_meta().iloc[0]
    assert m["rater"] == "0xa" and m["first_seen_ts"] == 99 and m["funder"] == "0xf"
    assert s.get_sync("rater_profile_mode") == "etherscan"


# --- enrich_raters: nothing to do ----------------------------------------------------


def test_enrich_nothing_to_do(tmp_path):
    s = Store(tmp_path / "t.db")
    http = FakeHttp([])
    c = EtherscanClient("K", session=http, sleep=lambda _: None)
    assert enrich_raters(s, c) == 0
    assert http.calls == []


def test_enrich_nothing_to_do_fallback(tmp_path):
    s = Store(tmp_path / "t.db")
    assert enrich_raters(s, None) == 0
    assert s.get_sync("rater_profile_mode") is None


# --- enrich_raters: explicit `addresses` bypasses distinct_clients() ----------------


def test_enrich_uses_given_addresses_instead_of_distinct_clients(tmp_path, monkeypatch):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa"), fb("0xb")])
    calls = []
    monkeypatch.setattr(Store, "distinct_clients", lambda self: calls.append(1) or ["should-not-be-used"])
    assert enrich_raters(s, None, addresses=["0xa"]) == 1
    assert calls == []  # distinct_clients() never called when addresses is given


def test_enrich_addresses_none_falls_back_to_distinct_clients(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    s.upsert_block_ts([(1, 555)])
    assert enrich_raters(s, None) == 1  # default addresses=None -> store.distinct_clients()


# --- client_from_env -----------------------------------------------------------------


def test_client_from_env(monkeypatch):
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    assert client_from_env() is None
    monkeypatch.setenv("ETHERSCAN_API_KEY", "abc123")
    c = client_from_env()
    assert isinstance(c, EtherscanClient) and c.key == "abc123"
    monkeypatch.setenv("ETHERSCAN_API_KEY", "  ")
    assert client_from_env() is None


# --- default_client: --profile-source propagation -----------------------------------


def test_default_client_none_returns_no_client():
    assert default_client("none", "KEY") is None
    assert default_client("none", None) is None


def test_default_client_blockscout_ignores_any_key():
    c = default_client("blockscout", "KEY")
    assert isinstance(c, BlockscoutV2Client)
    assert c.source == "blockscout"


def test_default_client_etherscan_uses_given_key():
    c = default_client("etherscan", "KEY")
    assert c.source == "etherscan" and c.key == "KEY"


def test_default_client_etherscan_without_key_raises_value_error():
    try:
        default_client("etherscan", None)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "etherscan" in str(e).lower()


def test_default_client_auto_with_key_uses_etherscan():
    c = default_client("auto", "KEY")
    assert c.source == "etherscan" and c.key == "KEY"


def test_default_client_auto_without_key_uses_blockscout():
    c = default_client("auto", None)
    assert isinstance(c, BlockscoutV2Client)
    assert c.source == "blockscout"


# --- default_client: --profile-rps/--profile-retries propagation --------------------


def test_default_client_none_rps_retries_leaves_client_defaults():
    """``rps=None``/``retries=None`` (the default) is not forwarded at all --
    each client built by ``default_client`` keeps its own default rate/retry
    count."""
    c = default_client("blockscout", None)
    assert c.gap == 1.0  # BlockscoutV2Client's own default: rps=1.0
    assert c.retries == 5  # BlockscoutV2Client's own default


def test_default_client_blockscout_passes_through_rps_and_retries():
    c = default_client("blockscout", None, rps=4.0, retries=7)
    assert isinstance(c, BlockscoutV2Client)
    assert c.gap == 0.25
    assert c.retries == 7


def test_default_client_etherscan_passes_through_rps_and_retries():
    c = default_client("etherscan", "KEY", rps=0.5, retries=2)
    assert isinstance(c, EtherscanClient)
    assert c.gap == 2.0
    assert c.retries == 2


def test_default_client_auto_with_key_passes_through_rps_and_retries():
    c = default_client("auto", "KEY", rps=10.0, retries=9)
    assert isinstance(c, EtherscanClient)
    assert c.gap == 0.1
    assert c.retries == 9


def test_default_client_auto_without_key_passes_through_rps_and_retries():
    c = default_client("auto", None, rps=4.0, retries=1)
    assert isinstance(c, BlockscoutV2Client)
    assert c.gap == 0.25
    assert c.retries == 1


def test_default_client_none_ignores_rps_and_retries():
    assert default_client("none", None, rps=5.0, retries=9) is None


def test_default_client_unknown_source_raises_value_error():
    try:
        default_client("bogus", None)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "bogus" in str(e)


# --- BlockscoutV2Client: happy paths -------------------------------------------------


def test_blockscout_v2_single_page_returns_oldest_and_lowercases_funder():
    body = {
        "items": [
            bs_tx("0x2", 42502820, "2026-02-22T21:20:19.000000Z", "0x6463AAA"),
            bs_tx("0x1", 42502816, "2026-02-22T21:16:19.000000Z", "0x6463BBB"),
        ],
        "next_page_params": None,
    }
    http = FakeV2Http([body])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    block, ts, funder = c.first_tx("0xA")
    assert block == 42502816
    assert funder == "0x6463bbb"
    assert ts == int(datetime(2026, 2, 22, 21, 16, 19, tzinfo=timezone.utc).timestamp())
    assert http.calls[0]["filter"] == "to"


def test_blockscout_v2_hits_v2_addresses_transactions_url():
    body = {"items": [bs_tx("0x1", 1, "2026-01-01T00:00:00Z", "0xf")], "next_page_params": None}
    http = FakeV2Http([body])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    c.first_tx("0xABCDEF")
    assert c.base_url == BLOCKSCOUT_V2_BASE


def test_blockscout_v2_two_pages_passes_next_page_params_through_and_returns_oldest():
    page1 = {
        "items": [bs_tx("0x2", 200, "2026-02-01T00:00:00Z", "0xnewest")],
        "next_page_params": {"block_number": 100, "index": 5},
    }
    page2 = {
        "items": [bs_tx("0x1", 100, "2026-01-01T00:00:00Z", "0xoldest")],
        "next_page_params": None,
    }
    http = FakeV2Http([page1, page2])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    block, ts, funder = c.first_tx("0xA")
    assert block == 100 and funder == "0xoldest"
    assert len(http.calls) == 2
    assert http.calls[1]["block_number"] == 100
    assert http.calls[1]["index"] == 5
    assert http.calls[1]["filter"] == "to"


def test_blockscout_v2_default_max_pages_and_rps():
    c = BlockscoutV2Client(sleep=lambda _: None)
    assert c.max_pages == 5
    assert c.source == "blockscout"
    # Measured 2026-09-14: Blockscout v2 sustains ~1 request/s; at 2 rps it
    # starts answering 429 after a few hundred requests -- so the default is
    # 1 rps (not 2), with more retries (5, not 3) to absorb transient 429s.
    assert abs(c.gap - 1.0) < 1e-9  # rps=1.0 default -> 1.0s gap
    assert c.retries == 5


def test_blockscout_v2_exceeds_max_pages_returns_none():
    pages = [{"items": [bs_tx(f"0x{i}", i, "2026-01-01T00:00:00Z", "0xf")],
               "next_page_params": {"block_number": i}} for i in range(3)]
    http = FakeV2Http(pages)
    c = BlockscoutV2Client(session=http, max_pages=3, sleep=lambda _: None)
    assert c.first_tx("0xA") is None
    assert len(http.calls) == 3  # stopped at max_pages, never asked for a 4th page


def test_blockscout_v2_no_items_and_no_next_page_returns_none():
    http = FakeV2Http([{"items": [], "next_page_params": None}])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    assert c.first_tx("0xA") is None


# --- BlockscoutV2Client: unknown address (404) ---------------------------------------


def test_blockscout_v2_404_returns_none_without_retry():
    http = FakeV2Http([404])
    c = BlockscoutV2Client(session=http, retries=3, sleep=lambda _: None)
    assert c.first_tx("0xA") is None
    assert len(http.calls) == 1


# --- BlockscoutV2Client: transient errors retried -------------------------------------


def test_blockscout_v2_429_then_200_succeeds_after_retry():
    body = {"items": [bs_tx("0x1", 5, "2026-01-01T00:00:00Z", "0xf")], "next_page_params": None}
    http = FakeV2Http([429, body])
    sleeps = []
    c = BlockscoutV2Client(session=http, sleep=sleeps.append)
    block, ts, funder = c.first_tx("0xA")
    assert block == 5 and funder == "0xf"
    assert len(http.calls) == 2
    assert len(sleeps) >= 2  # throttle sleep(s) plus >=1 backoff sleep


# --- 429 Retry-After header: honored (capped at 60s) instead of exponential backoff --


def test_blockscout_v2_429_honors_retry_after_header():
    body = {"items": [], "next_page_params": None}
    http = FakeV2Http([(429, {"Retry-After": "10"}), body])
    sleeps = []
    c = BlockscoutV2Client(session=http, retries=2, sleep=sleeps.append)
    assert c.first_tx("0xA") is None
    # attempt 0's throttle (gap=1.0), then its backoff -- 10.0 (Retry-After),
    # not the exponential fallback min(2**0, 30) == 1.0 -- then attempt 1's throttle.
    assert sleeps == [1.0, 10.0, 1.0]


def test_blockscout_v2_429_retry_after_capped_at_60():
    body = {"items": [], "next_page_params": None}
    http = FakeV2Http([(429, {"Retry-After": "600"}), body])
    sleeps = []
    c = BlockscoutV2Client(session=http, retries=2, sleep=sleeps.append)
    c.first_tx("0xA")
    assert sleeps.count(60.0) == 1
    assert 600.0 not in sleeps


def test_blockscout_v2_429_without_retry_after_falls_back_to_exponential_backoff():
    body = {"items": [], "next_page_params": None}
    http = FakeV2Http([429, body])
    sleeps = []
    c = BlockscoutV2Client(session=http, retries=2, sleep=sleeps.append)
    c.first_tx("0xA")
    # Two throttle sleeps (gap=1.0 default) plus one backoff of min(2**0, 30) == 1.0
    # -- all three happen to equal 1.0 here, so just check the total count.
    assert len(sleeps) == 3
    assert all(s == 1.0 for s in sleeps)


def test_blockscout_v2_exhausts_retries_then_raises_scrubbed_runtime_error(caplog):
    http = FakeV2Http([503, 503, 503])
    c = BlockscoutV2Client(session=http, retries=3, sleep=lambda _: None)
    with caplog.at_level(logging.DEBUG):
        try:
            c.first_tx("0xA")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            msg = str(e)
    assert "503" in msg and "0xa" in msg.lower()
    assert "SECRET" not in msg
    assert "SECRET" not in caplog.text
    assert "http" not in msg.lower() or "https://" not in msg  # no URL leaked
    assert len(http.calls) == 3


def test_blockscout_v2_scrubs_connection_error_and_retries(caplog):
    exc = requests.ConnectionError("HTTPSConnectionPool: Max retries exceeded ... apikey=SECRET ...")
    http = FakeV2HttpConnErr(exc, then_body={"items": [], "next_page_params": None})
    sleeps = []
    c = BlockscoutV2Client(session=http, retries=2, sleep=sleeps.append)
    with caplog.at_level(logging.DEBUG):
        result = c.first_tx("0xA")
    assert result is None  # recovered on retry, page had no items
    assert "SECRET" not in caplog.text
    assert len(http.calls) == 2


def test_blockscout_v2_non_json_body_retries_then_raises_value_error():
    http = FakeV2HttpBadJson()
    c = BlockscoutV2Client(session=http, retries=2, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected ValueError"
    except ValueError as e:
        msg = str(e)
        assert "0xa" in msg.lower()
        assert "Expecting value" not in msg
    assert len(http.calls) == 2


def test_blockscout_v2_missing_items_key_raises_value_error():
    http = FakeV2Http([{"next_page_params": None}])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "0xa" in str(e).lower()


def test_blockscout_v2_malformed_tx_entry_raises_value_error():
    http = FakeV2Http([{"items": [{"hash": "0x1", "block_number": 1}], "next_page_params": None}])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "0xa" in str(e).lower()


# --- BlockscoutV2Client: timestamp parsing --------------------------------------------


def test_blockscout_v2_parses_timestamp_with_fractional_seconds():
    body = {"items": [bs_tx("0x1", 1, "2026-02-22T21:16:19.123456Z", "0xf")], "next_page_params": None}
    c = BlockscoutV2Client(session=FakeV2Http([body]), sleep=lambda _: None)
    _, ts, _ = c.first_tx("0xA")
    assert ts == int(datetime(2026, 2, 22, 21, 16, 19, tzinfo=timezone.utc).timestamp())


def test_blockscout_v2_parses_timestamp_without_fractional_seconds():
    body = {"items": [bs_tx("0x1", 1, "2026-02-22T21:16:19Z", "0xf")], "next_page_params": None}
    c = BlockscoutV2Client(session=FakeV2Http([body]), sleep=lambda _: None)
    _, ts, _ = c.first_tx("0xA")
    assert ts == int(datetime(2026, 2, 22, 21, 16, 19, tzinfo=timezone.utc).timestamp())


# --- default_client: BlockscoutV2Client wiring ----------------------------------------


def test_default_client_blockscout_returns_blockscout_v2_client():
    c = default_client("blockscout", None)
    assert isinstance(c, BlockscoutV2Client)
    assert c.source == "blockscout"


# --- enrich_raters: BlockscoutV2Client end-to-end -------------------------------------


def test_enrich_with_blockscout_v2_client_records_blockscout_mode(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    body = {"items": [bs_tx("0x1", 3, "2026-01-01T00:00:00Z", "0xF")], "next_page_params": None}
    http = FakeV2Http([body])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    assert enrich_raters(s, c) == 1
    m = s.load_rater_meta().iloc[0]
    assert m["funder"] == "0xf"
    assert m["first_seen_ts"] == int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    assert s.get_sync("rater_profile_mode") == "blockscout"


# --- M1: the Etherscan API key never reaches urllib3's DEBUG log line ----------------


@pytest.fixture
def clean_urllib3_filters():
    """Start from (and restore) a urllib3 logger with no redactors installed.

    The filters live on process-wide loggers and the install registry is a
    module global, so every other test that builds a keyed ``EtherscanClient``
    leaks one in -- strip them for the duration of these tests, then put the
    original list back."""
    names = ("urllib3", "urllib3.connectionpool")
    before = {n: list(logging.getLogger(n).filters) for n in names}
    for n in names:
        lg = logging.getLogger(n)
        lg.filters = [f for f in lg.filters if not isinstance(f, http_util.ApiKeyRedactor)]
    saved = dict(http_util._installed_redactors)
    http_util._installed_redactors.clear()
    yield
    for n, filters in before.items():
        logging.getLogger(n).filters = filters
    http_util._installed_redactors.clear()
    http_util._installed_redactors.update(saved)


def test_api_key_is_redacted_from_urllib3_debug_records(caplog, clean_urllib3_filters):
    EtherscanClient("SECRET-KEY-VALUE", session=FakeHttp([]), sleep=lambda _: None)
    with caplog.at_level(logging.DEBUG, logger="urllib3"):
        logging.getLogger("urllib3").debug(
            '%s://%s:%s "%s %s %s" %s %s', "https", "api.etherscan.io", 443,
            "GET", "/v2/api?module=account&apikey=SECRET-KEY-VALUE", "HTTP/1.1", 200, 512)
    assert "SECRET-KEY-VALUE" not in caplog.text
    assert "[redacted]" in caplog.text


def test_api_key_is_redacted_from_connectionpool_log_during_a_real_call(caplog, clean_urllib3_filters):
    class LoggingHttp(FakeHttp):
        """A session that logs urllib3's connectionpool DEBUG line (the one
        that carries the whole query string, API key included) mid-request."""

        def get(self, url, params=None, timeout=None, stream=None):
            logging.getLogger("urllib3.connectionpool").debug(
                '%s://%s:%s "%s %s %s" %s %s', "https", "api.etherscan.io", 443, "GET",
                f"/v2/api?module=account&apikey={params['apikey']}", "HTTP/1.1", 200, 512)
            return super().get(url, params=params, timeout=timeout, stream=stream)

    http = LoggingHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "99", "from": "0xF"}]}])
    c = EtherscanClient("SECRET-KEY-VALUE", session=http, sleep=lambda _: None)
    with caplog.at_level(logging.DEBUG, logger="urllib3.connectionpool"):
        assert c.first_tx("0xA") == (3, 99, "0xf")
    assert "SECRET-KEY-VALUE" not in caplog.text
    assert "[redacted]" in caplog.text


def test_api_key_redactor_installed_once_per_key(clean_urllib3_filters):
    EtherscanClient("SAME-KEY", session=FakeHttp([]), sleep=lambda _: None)
    EtherscanClient("SAME-KEY", session=FakeHttp([]), sleep=lambda _: None)
    for name in ("urllib3", "urllib3.connectionpool"):
        installed = [f for f in logging.getLogger(name).filters
                     if isinstance(f, http_util.ApiKeyRedactor)]
        assert len(installed) == 1


def test_keyless_client_installs_no_redactor(clean_urllib3_filters):
    EtherscanClient(None, session=FakeHttp([]), sleep=lambda _: None)
    BlockscoutV2Client(session=FakeV2Http([]), sleep=lambda _: None)
    installed = [f for f in logging.getLogger("urllib3").filters
                 if isinstance(f, http_util.ApiKeyRedactor)]
    assert installed == []


# --- M2: capped response bodies ------------------------------------------------------


def test_etherscan_streams_the_response():
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "99", "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    c.first_tx("0xA")
    assert http.streams == [True]
    assert http.responses[0].closed is True


def test_max_profile_response_bytes_is_8_mib():
    assert rater_profile.MAX_PROFILE_RESPONSE_BYTES == 8 * 1024 * 1024


def test_etherscan_oversized_body_raises_without_leaking_key_or_body(caplog, monkeypatch):
    monkeypatch.setattr(rater_profile, "MAX_PROFILE_RESPONSE_BYTES", 1024)
    http = FakeHttp([b'{"status": "1", "leak": "' + b"SECRETBODY" * 2000 + b'"}'])
    c = EtherscanClient("SECRET-KEY-VALUE", session=http, sleep=lambda _: None)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError) as exc:
            c.first_tx("0xA")
    msg = str(exc.value)
    assert "SECRET-KEY-VALUE" not in msg and "SECRETBODY" not in msg
    assert "0xa" in msg.lower()
    assert "SECRET-KEY-VALUE" not in caplog.text and "SECRETBODY" not in caplog.text


def test_etherscan_oversized_body_is_not_retried(monkeypatch):
    monkeypatch.setattr(rater_profile, "MAX_PROFILE_RESPONSE_BYTES", 1024)
    http = FakeHttp([b"x" * 200_000] * 3)
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None, retries=3)
    with pytest.raises(RuntimeError):
        c.first_tx("0xA")
    assert len(http.calls) == 1


def test_blockscout_v2_streams_the_response():
    body = {"items": [bs_tx("0x1", 1, "2026-01-01T00:00:00Z", "0xf")], "next_page_params": None}
    http = FakeV2Http([body])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    c.first_tx("0xA")
    assert http.streams == [True]


def test_blockscout_v2_oversized_body_raises_without_leaking_body(monkeypatch):
    monkeypatch.setattr(rater_profile, "MAX_PROFILE_RESPONSE_BYTES", 1024)
    http = FakeV2Http([b'{"items": [], "leak": "' + b"SECRETBODY" * 2000 + b'"}'])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    with pytest.raises(RuntimeError) as exc:
        c.first_tx("0xA")
    assert "SECRETBODY" not in str(exc.value)
    assert "0xa" in str(exc.value).lower()


# --- M4: implausible timestamps never reach the raters cache ------------------------


def test_etherscan_negative_timestamp_returns_none_with_one_warning(caplog):
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "-1", "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.rater_profile"):
        assert c.first_tx("0xA") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "'-1'" in warnings[0].getMessage()


def test_etherscan_zero_timestamp_returns_none():
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "0", "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    assert c.first_tx("0xA") is None


def test_etherscan_absurd_future_timestamp_returns_none():
    absurd = str(rater_profile.MAX_PLAUSIBLE_TS + 1)
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": absurd, "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    assert c.first_tx("0xA") is None


def test_max_plausible_ts_is_2100_01_01():
    assert rater_profile.MAX_PLAUSIBLE_TS == 4102444800


def test_blockscout_pre_1970_timestamp_returns_none_with_one_warning(caplog):
    body = {"items": [bs_tx("0x1", 1, "1900-01-01T00:00:00Z", "0xf")], "next_page_params": None}
    c = BlockscoutV2Client(session=FakeV2Http([body]), sleep=lambda _: None)
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.rater_profile"):
        assert c.first_tx("0xA") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "1900-01-01" in warnings[0].getMessage()


def test_blockscout_far_future_timestamp_returns_none_with_one_warning(caplog):
    body = {"items": [bs_tx("0x1", 1, "2200-01-01T00:00:00Z", "0xf")], "next_page_params": None}
    c = BlockscoutV2Client(session=FakeV2Http([body]), sleep=lambda _: None)
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.rater_profile"):
        assert c.first_tx("0xA") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "2200-01-01" in warnings[0].getMessage()


def test_blockscout_garbage_timestamp_returns_none_with_one_warning(caplog):
    body = {"items": [bs_tx("0x1", 1, "garbage", "0xf")], "next_page_params": None}
    c = BlockscoutV2Client(session=FakeV2Http([body]), sleep=lambda _: None)
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.rater_profile"):
        assert c.first_tx("0xA") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "garbage" in warnings[0].getMessage()


def test_implausible_timestamp_never_reaches_upsert_rater(tmp_path):
    # The whole point of M4: nothing implausible is cached, so a later
    # score/report run cannot be wedged by one hostile third-party row.
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([fb("0xa")])
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "-99", "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    assert enrich_raters(s, c) == 1
    meta = s.load_rater_meta()
    assert meta["first_seen_ts"].isna().all()


# --- L5: Blockscout pagination params are allowlisted, not echoed verbatim -----------


def test_next_page_params_drops_unsafe_keys_and_values():
    page1 = {"items": [bs_tx("0x2", 200, "2026-02-01T00:00:00Z", "0xnewest")],
             "next_page_params": {"block_number": 100, "index": 5, "../../etc/passwd": "x",
                                  "Uppercase": "y", "nested": {"a": 1}, "listy": [1, 2]}}
    page2 = {"items": [bs_tx("0x1", 100, "2026-01-01T00:00:00Z", "0xoldest")],
             "next_page_params": None}
    http = FakeV2Http([page1, page2])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    assert c.first_tx("0xA")[0] == 100
    sent = http.calls[1]
    assert sent["block_number"] == 100 and sent["index"] == 5 and sent["filter"] == "to"
    assert set(sent) == {"block_number", "index", "filter"}


def test_next_page_params_with_nothing_safe_stops_paging(caplog):
    page1 = {"items": [bs_tx("0x2", 200, "2026-02-01T00:00:00Z", "0xnewest")],
             "next_page_params": {"BAD/KEY": "x"}}
    http = FakeV2Http([page1])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.rater_profile"):
        assert c.first_tx("0xA") is None
    assert len(http.calls) == 1
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_next_page_params_non_dict_stops_paging():
    page1 = {"items": [bs_tx("0x2", 200, "2026-02-01T00:00:00Z", "0xnewest")],
             "next_page_params": ["not", "a", "dict"]}
    http = FakeV2Http([page1])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    assert c.first_tx("0xA") is None
    assert len(http.calls) == 1


def test_next_page_params_key_length_is_bounded():
    page1 = {"items": [bs_tx("0x2", 200, "2026-02-01T00:00:00Z", "0xnewest")],
             "next_page_params": {"a" * 33: 1, "index": 5}}
    page2 = {"items": [bs_tx("0x1", 100, "2026-01-01T00:00:00Z", "0xoldest")],
             "next_page_params": None}
    http = FakeV2Http([page1, page2])
    c = BlockscoutV2Client(session=http, sleep=lambda _: None)
    c.first_tx("0xA")
    assert set(http.calls[1]) == {"index", "filter"}
