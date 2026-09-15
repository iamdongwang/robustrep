"""Tests for the shared HTTP hygiene helpers: bounded bodies (M2), API key
redaction in urllib3's own log records (M1)."""
import json
import logging

import pytest

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


def test_install_key_redaction_ignores_a_falsy_key():
    before = list(logging.getLogger("urllib3").filters)
    install_key_redaction("")
    install_key_redaction(None)
    assert logging.getLogger("urllib3").filters == before
