"""Tests for the shared HTTP hygiene helpers: bounded bodies (M2), API key
redaction in urllib3's own log records (M1)."""
import json
import logging

import pytest

from robustrep.sources import http_util
from robustrep.sources.http_util import (
    ApiKeyRedactor,
    ResponseTooLarge,
    install_key_redaction,
    read_json_capped,
)


class FakeRaw:
    """Stand-in for ``urllib3.HTTPResponse``: hands out ``data`` in chunks of
    whatever size the reader asks for, recording every requested size."""

    def __init__(self, data: bytes):
        self.data, self.reads = data, []

    def read(self, amt, decode_content=True):
        assert decode_content is True, "must decode transfer/content encodings"
        self.reads.append(amt)
        chunk, self.data = self.data[:amt], self.data[amt:]
        return chunk


class FakeResponse:
    def __init__(self, data: bytes):
        self.raw = FakeRaw(data)


class EndlessRaw:
    """A body that never ends -- what a hostile node can serve (M2)."""

    def __init__(self):
        self.served = 0

    def read(self, amt, decode_content=True):
        self.served += amt
        return b"a" * amt


class EndlessResponse:
    def __init__(self):
        self.raw = EndlessRaw()


def test_parses_body_under_the_limit():
    body = {"jsonrpc": "2.0", "result": ["a", "b"]}
    r = FakeResponse(json.dumps(body).encode())
    assert read_json_capped(r, 1024) == body


def test_parses_body_exactly_at_the_limit():
    payload = json.dumps({"k": "v"}).encode()
    assert read_json_capped(FakeResponse(payload), len(payload)) == {"k": "v"}


def test_raises_past_the_limit():
    payload = json.dumps({"k": "v" * 100}).encode()
    with pytest.raises(ResponseTooLarge):
        read_json_capped(FakeResponse(payload), 16)


def test_message_names_only_the_limit_never_the_body():
    payload = json.dumps({"secret": "SUPERSECRET-BODY-TEXT"}).encode()
    with pytest.raises(ResponseTooLarge) as exc:
        read_json_capped(FakeResponse(payload), 8)
    msg = str(exc.value)
    assert "8" in msg
    assert "SUPERSECRET" not in msg and "secret" not in msg


def test_stops_reading_an_endless_body_soon_after_the_limit():
    r = EndlessResponse()
    with pytest.raises(ResponseTooLarge):
        read_json_capped(r, 1024)
    # Bounded by one chunk past the cap -- never the whole (infinite) body.
    assert r.raw.served <= 1024 + 64 * 1024


def test_reads_in_bounded_chunks_not_one_giant_read():
    r = FakeResponse(b"[]" + b" " * (200 * 1024))
    read_json_capped(r, 1024 * 1024)
    assert r.raw.reads and max(r.raw.reads) <= 64 * 1024


def test_response_too_large_is_a_value_error():
    assert issubclass(ResponseTooLarge, ValueError)


def test_bad_json_raises_value_error_not_response_too_large():
    r = FakeResponse(b"<html>not json</html>")
    with pytest.raises(ValueError) as exc:
        read_json_capped(r, 1024)
    assert not isinstance(exc.value, ResponseTooLarge)


def test_invalid_utf8_raises_value_error():
    r = FakeResponse(b'{"k": "\xff\xfe"}')
    with pytest.raises(ValueError):
        read_json_capped(r, 1024)


def test_empty_body_raises_value_error():
    with pytest.raises(ValueError):
        read_json_capped(FakeResponse(b""), 1024)


def test_requires_a_raw_stream():
    class NoRaw:
        raw = None

    with pytest.raises(ValueError):
        read_json_capped(NoRaw(), 1024)


def test_rejects_a_non_positive_limit():
    with pytest.raises(ValueError):
        read_json_capped(FakeResponse(b"{}"), 0)


# --- M1: API key redaction in third-party (urllib3) log records ---------------------


def _record(msg, args):
    return logging.LogRecord("urllib3", logging.DEBUG, "f.py", 1, msg, args, None)


def test_redactor_scrubs_the_key_from_positional_args():
    redactor = ApiKeyRedactor("SECRET")
    record = _record('%s "%s %s"', ("https", "GET", "/api?apikey=SECRET"))
    assert redactor.filter(record) is True
    assert "SECRET" not in record.getMessage()
    assert "[redacted]" in record.getMessage()


def test_redactor_scrubs_the_key_from_mapping_args():
    # logging accepts a single mapping argument for %(name)s-style formats.
    redactor = ApiKeyRedactor("SECRET")
    record = _record("%(url)s", ({"url": "/api?apikey=SECRET"},))
    assert isinstance(record.args, dict)  # logging unwraps a single mapping arg
    assert redactor.filter(record) is True
    assert record.getMessage() == "/api?apikey=[redacted]"


def test_redactor_scrubs_the_key_from_the_format_string_itself():
    redactor = ApiKeyRedactor("SECRET")
    record = _record("GET /api?apikey=SECRET", None)
    assert redactor.filter(record) is True
    assert "SECRET" not in record.getMessage()


def test_redactor_leaves_unrelated_records_alone():
    redactor = ApiKeyRedactor("SECRET")
    record = _record("%s", ("nothing to hide",))
    assert redactor.filter(record) is True
    assert record.getMessage() == "nothing to hide"


def test_redactor_passes_through_non_string_args():
    redactor = ApiKeyRedactor("SECRET")
    record = _record("%s %s", (200, None))
    assert redactor.filter(record) is True
    assert record.getMessage() == "200 None"


# L2: robustrep's own loggers that echo server-chosen text (a JSON-RPC error
# message at DEBUG, an implausible timestamp or a refused funder at WARNING)
# must be scrubbed too -- a logger-level filter is not consulted for child or
# sibling loggers, so each one needs its own copy.
ROBUSTREP_ECHO_LOGGERS = ("robustrep.sources.alchemy_profile", "robustrep.sources.profile_http")


def _isolated_redaction(names):
    """Snapshot the filters on ``names`` and the install registry, clear both,
    and return a restore callable -- other tests leak redactors for their own
    keys into these process-wide loggers."""
    before = {n: list(logging.getLogger(n).filters) for n in names}
    saved = dict(http_util._installed_redactors)
    for n in names:
        logging.getLogger(n).filters = []
    http_util._installed_redactors.clear()

    def restore():
        for n, filters in before.items():
            logging.getLogger(n).filters = filters
        http_util._installed_redactors.clear()
        http_util._installed_redactors.update(saved)
    return restore


def test_install_key_redaction_covers_the_robustrep_loggers_that_echo_server_text(caplog):
    restore = _isolated_redaction(ROBUSTREP_ECHO_LOGGERS)
    try:
        install_key_redaction("SECRET-KEY-VALUE")
        for name in ROBUSTREP_ECHO_LOGGERS:
            installed = [f for f in logging.getLogger(name).filters
                         if isinstance(f, ApiKeyRedactor) and f.key == "SECRET-KEY-VALUE"]
            assert len(installed) == 1, name
        with caplog.at_level(logging.DEBUG):
            logging.getLogger("robustrep.sources.alchemy_profile").debug(
                "JSON-RPC error for %s (code %r): %r", "0xa", -32602, "echo SECRET-KEY-VALUE back")
            logging.getLogger("robustrep.sources.profile_http").warning(
                "implausible first-tx timestamp for 0xa: 'SECRET-KEY-VALUE'")
        assert "SECRET-KEY-VALUE" not in caplog.text
        assert caplog.text.count("[redacted]") == 2
    finally:
        restore()


def test_install_key_redaction_is_idempotent_on_the_robustrep_loggers_too():
    restore = _isolated_redaction(ROBUSTREP_ECHO_LOGGERS)
    try:
        install_key_redaction("SAME-KEY-VALUE")
        install_key_redaction("SAME-KEY-VALUE")
        install_key_redaction("OTHER-KEY-VALUE")
        for name in ROBUSTREP_ECHO_LOGGERS:
            redactors = [f for f in logging.getLogger(name).filters if isinstance(f, ApiKeyRedactor)]
            assert sorted(f.key for f in redactors) == ["OTHER-KEY-VALUE", "SAME-KEY-VALUE"]
    finally:
        restore()


def test_install_key_redaction_ignores_a_falsy_key():
    before = list(logging.getLogger("urllib3").filters)
    install_key_redaction("")
    install_key_redaction(None)
    assert logging.getLogger("urllib3").filters == before


# --- the cap is exclusive: one byte past the limit is refused -----------------------


def test_rejects_exactly_one_byte_past_the_limit():
    payload = json.dumps({"k": "v"}).encode()
    assert read_json_capped(FakeResponse(payload), len(payload)) == {"k": "v"}
    with pytest.raises(ResponseTooLarge):
        read_json_capped(FakeResponse(payload), len(payload) - 1)


# --- I2: non-finite JSON numbers are refused at the choke point ---------------------


def test_rejects_the_nan_constant():
    with pytest.raises(ValueError) as exc:
        read_json_capped(FakeResponse(b'{"ts": NaN}'), 1024)
    assert not isinstance(exc.value, ResponseTooLarge)


def test_rejects_the_infinity_constant():
    with pytest.raises(ValueError):
        read_json_capped(FakeResponse(b'{"ts": Infinity}'), 1024)


def test_rejects_the_negative_infinity_constant():
    with pytest.raises(ValueError):
        read_json_capped(FakeResponse(b'{"ts": -Infinity}'), 1024)


def test_still_parses_ordinary_numbers():
    assert read_json_capped(FakeResponse(b'{"ts": 1.5, "n": 7}'), 1024) == {"ts": 1.5, "n": 7}


# --- M1: a key too short to match safely is not redacted at all ---------------------


def test_install_key_redaction_refuses_a_short_key(caplog):
    logger_names = ("urllib3", "urllib3.connectionpool")
    before = {n: list(logging.getLogger(n).filters) for n in logger_names}
    try:
        with caplog.at_level(logging.WARNING, logger="robustrep.sources.http_util"):
            install_key_redaction("KEY")
        for n in logger_names:
            assert logging.getLogger(n).filters == before[n]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "KEY" not in warnings[0].getMessage()  # never echo the key itself
        assert "3" in warnings[0].getMessage()  # its length is safe to state
    finally:
        for n, filters in before.items():
            logging.getLogger(n).filters = filters
        http_util._installed_redactors.clear()


def test_install_key_redaction_warns_once_per_short_key(caplog):
    before = {n: list(logging.getLogger(n).filters) for n in ("urllib3", "urllib3.connectionpool")}
    try:
        with caplog.at_level(logging.WARNING, logger="robustrep.sources.http_util"):
            install_key_redaction("short")
            install_key_redaction("short")
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    finally:
        for n, filters in before.items():
            logging.getLogger(n).filters = filters
        http_util._installed_redactors.clear()


def test_redaction_covers_every_urllib3_logger_that_echoes_a_url():
    assert set(http_util._URLLIB3_LOGGERS) >= {
        "urllib3", "urllib3.connectionpool", "urllib3.util.retry", "urllib3.poolmanager"}


def test_short_for_log_scrubs_installed_keys_before_truncating():
    # I-2: the redactor filter replaces the *exact* key in a record's args, but
    # short_for_log has already cut the value to 40 chars by then -- a key at
    # offset >= 9 survived as a 30+ char prefix. Installed keys are therefore
    # scrubbed inside short_for_log itself, before the cut.
    restore = _isolated_redaction(http_util._REDACTED_LOGGERS)
    try:
        key = "K" * 32
        http_util.install_key_redaction(key)
        for value in ("x" * 10 + key, key, key + "tail", "a" + key + "b" + key):
            out = http_util.short_for_log(value)
            assert key not in out and key[:12] not in out, out
            assert http_util.REDACTED in out
        # Unrelated text and the limit are untouched.
        assert http_util.short_for_log("plain") == "plain"
        assert http_util.short_for_log("y" * 50).startswith("y" * 40)
    finally:
        restore()


def test_short_for_log_leaves_a_short_unredactable_key_alone():
    restore = _isolated_redaction(http_util._REDACTED_LOGGERS)
    try:
        http_util.install_key_redaction("KEY")  # under the minimum: registered as None
        assert http_util.short_for_log("MONKEY") == "MONKEY"
    finally:
        restore()
