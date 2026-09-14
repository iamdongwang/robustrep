import logging

import requests

from robustrep.sources.rater_profile import EtherscanClient, client_from_env, enrich_raters, estimate_seconds
from robustrep.store import Store

FB = dict(chain="base", tx_hash="0x", value="1", value_decimals=0, tag1="", tag2="", endpoint="",
          feedback_uri="", feedback_hash="")


def fb(client, block=1, log_index=0, agent_id="1", feedback_index=0):
    return dict(FB, block=block, log_index=log_index, agent_id=agent_id, client=client,
                feedback_index=feedback_index)


class FakeHttp:
    """Simulates a JSON body response (success or Etherscan-level error body)."""

    def __init__(self, payloads):
        self.payloads, self.calls = list(payloads), []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        p = self.payloads.pop(0)

        class R:
            def json(self_inner):
                return p

            def raise_for_status(self_inner):
                pass

        return R()


class FakeHttpRaisingOnGet:
    """Simulates a transport-level failure (e.g. ConnectionError) raised by ``get`` itself."""

    def __init__(self, exc):
        self.exc, self.calls = exc, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        raise self.exc


class FakeHttpRaisingOnStatus:
    """Simulates an HTTP error surfaced via ``raise_for_status`` (e.g. 429/404)."""

    def __init__(self, status_code, message):
        self.status_code, self.message, self.calls = status_code, message, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)

        class R:
            def raise_for_status(self_inner):
                err = requests.HTTPError(self.message)
                resp = requests.Response()
                resp.status_code = self.status_code
                err.response = resp
                raise err

            def json(self_inner):
                raise AssertionError("json() must not be called when raise_for_status() raises")

        return R()


class FakeHttpBadJson:
    """Simulates a 200 response whose body is not valid JSON."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)

        class R:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                raise ValueError("Expecting value: line 1 column 1 (char 0)")

        return R()


# --- EtherscanClient.first_tx: happy paths -----------------------------------------


def test_first_tx_parses_etherscan_v2():
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "10", "timeStamp": "1700", "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    assert c.first_tx("0xA") == (10, 1700, "0xf")
    assert http.calls[0]["chainid"] == 8453 and http.calls[0]["apikey"] == "KEY"


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
    secret_url = "https://api.etherscan.io/v2/api?chainid=8453&apikey=SECRET"
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


def test_first_tx_malformed_entry_bad_timestamp_raises_value_error():
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "not-a-number", "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "0xa" in str(e).lower()


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


# --- client_from_env -----------------------------------------------------------------


def test_client_from_env(monkeypatch):
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    assert client_from_env() is None
    monkeypatch.setenv("ETHERSCAN_API_KEY", "abc123")
    c = client_from_env()
    assert isinstance(c, EtherscanClient) and c.key == "abc123"
    monkeypatch.setenv("ETHERSCAN_API_KEY", "  ")
    assert client_from_env() is None
