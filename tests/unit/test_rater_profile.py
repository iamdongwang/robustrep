import logging

from robustrep.sources.rater_profile import EtherscanClient, client_from_env, enrich_raters
from robustrep.store import Store


class FakeHttp:
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


def test_first_tx_parses_etherscan_v2():
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "10", "timeStamp": "1700", "from": "0xF"}]}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    assert c.first_tx("0xA") == (10, 1700, "0xf")
    assert http.calls[0]["chainid"] == 8453 and http.calls[0]["apikey"] == "KEY"


def test_first_tx_none_when_empty():
    c = EtherscanClient("KEY", session=FakeHttp([{"status": "0", "result": []}]), sleep=lambda _: None)
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


def test_first_tx_retries_transient_rate_limit_then_raises():
    payload = {"status": "0", "message": "NOTOK", "result": "Max rate limit reached"}
    http = FakeHttp([payload, payload, payload])
    sleeps = []
    c = EtherscanClient("KEY", session=http, sleep=sleeps.append)
    try:
        c.first_tx("0xA")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "0xa" in str(e).lower() or "0xA" in str(e)
    assert len(http.calls) == 3
    # one rps-throttle sleep per attempt (3), plus backoff sleep between attempts
    # 1->2 and 2->3 (2 more) -- none after the final, exhausted attempt.
    assert len(sleeps) == 5


def test_first_tx_invalid_api_key_raises_immediately():
    http = FakeHttp([{"status": "0", "message": "NOTOK", "result": "Invalid API Key"}])
    c = EtherscanClient("KEY", session=http, sleep=lambda _: None)
    try:
        c.first_tx("0xA")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "API Key" in str(e)
    assert len(http.calls) == 1


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


def test_first_tx_never_logs_params(caplog):
    http = FakeHttp([{"status": "1", "result": [{"blockNumber": "1", "timeStamp": "1", "from": "0xf"}]}])
    c = EtherscanClient("SECRET-KEY", session=http, sleep=lambda _: None)
    with caplog.at_level(logging.DEBUG, logger="robustrep.sources.rater_profile"):
        c.first_tx("0xA")
    assert "SECRET-KEY" not in caplog.text


def test_first_tx_none_when_status_ok_but_result_empty():
    c = EtherscanClient("KEY", session=FakeHttp([{"status": "1", "result": []}]), sleep=lambda _: None)
    assert c.first_tx("0xA") is None


def test_enrich_with_client_no_tx_found_upserts_none(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([dict(chain="base", block=1, tx_hash="0x", log_index=0, agent_id="1", client="0xa",
                            feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
                            feedback_uri="", feedback_hash="")])
    c = EtherscanClient("K", session=FakeHttp([{"status": "0", "result": []}]), sleep=lambda _: None)
    assert enrich_raters(s, c) == 1
    m = s.load_rater_meta().iloc[0]
    assert m["first_seen_ts"] is None and m["funder"] is None


def test_enrich_with_client(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([dict(chain="base", block=1, tx_hash="0x", log_index=0, agent_id="1", client="0xa",
                            feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
                            feedback_uri="", feedback_hash="")])
    c = EtherscanClient("K", session=FakeHttp([{"status": "1", "result": [{"blockNumber": "3", "timeStamp": "99", "from": "0xF"}]}]),
                        sleep=lambda _: None)
    assert enrich_raters(s, c) == 1
    m = s.load_rater_meta().iloc[0]
    assert m["first_seen_ts"] == 99 and m["funder"] == "0xf"
    assert s.get_sync("rater_profile_mode") == "etherscan"


def test_enrich_fallback_without_client(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([dict(chain="base", block=1, tx_hash="0x", log_index=0, agent_id="1", client="0xa",
                            feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
                            feedback_uri="", feedback_hash="")])
    s.upsert_block_ts([(1, 555)])
    assert enrich_raters(s, None) == 1
    m = s.load_rater_meta().iloc[0]
    assert m["first_seen_ts"] == 555 and m["funder"] is None
    assert s.get_sync("rater_profile_mode") == "fallback"


def test_enrich_fallback_zero_ts_warns_once(tmp_path, caplog):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([
        dict(chain="base", block=1, tx_hash="0x", log_index=0, agent_id="1", client="0xa",
             feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
             feedback_uri="", feedback_hash=""),
        dict(chain="base", block=2, tx_hash="0x", log_index=0, agent_id="1", client="0xb",
             feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
             feedback_uri="", feedback_hash=""),
    ])
    with caplog.at_level(logging.WARNING, logger="robustrep.sources.rater_profile"):
        assert enrich_raters(s, None) == 2
    m = s.load_rater_meta()
    assert (m["first_seen_ts"] == 0).all()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "fill_block_timestamps" in warnings[0].message


def test_enrich_with_client_collects_failures_but_keeps_successes(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([
        dict(chain="base", block=1, tx_hash="0x", log_index=0, agent_id="1", client="0xa",
             feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
             feedback_uri="", feedback_hash=""),
        dict(chain="base", block=1, tx_hash="0x", log_index=1, agent_id="1", client="0xb",
             feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
             feedback_uri="", feedback_hash=""),
        dict(chain="base", block=1, tx_hash="0x", log_index=2, agent_id="1", client="0xc",
             feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
             feedback_uri="", feedback_hash=""),
    ])

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


def test_enrich_nothing_to_do(tmp_path):
    s = Store(tmp_path / "t.db")
    http = FakeHttp([])
    c = EtherscanClient("K", session=http, sleep=lambda _: None)
    assert enrich_raters(s, c) == 0
    assert http.calls == []


def test_enrich_is_idempotent(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_feedback([dict(chain="base", block=1, tx_hash="0x", log_index=0, agent_id="1", client="0xa",
                            feedback_index=0, value="1", value_decimals=0, tag1="", tag2="", endpoint="",
                            feedback_uri="", feedback_hash="")])
    s.upsert_block_ts([(1, 555)])
    assert enrich_raters(s, None) == 1
    assert enrich_raters(s, None) == 0


def test_client_from_env(monkeypatch):
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    assert client_from_env() is None
    monkeypatch.setenv("ETHERSCAN_API_KEY", "abc123")
    c = client_from_env()
    assert isinstance(c, EtherscanClient) and c.key == "abc123"
    monkeypatch.setenv("ETHERSCAN_API_KEY", "  ")
    assert client_from_env() is None
