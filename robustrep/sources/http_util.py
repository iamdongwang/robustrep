"""Shared hygiene for every outbound HTTP client: bounded bodies (M2), redacted
secrets in third-party logs (M1).

Every remote this project talks to -- the JSON-RPC node, Etherscan, Blockscout --
is a third party we do not control, and two of them (the RPC node's ``eth_getLogs``
result, Blockscout's transaction list) return bodies whose size is partly chosen by
whoever wrote the on-chain data being read. ``requests``' own ``Response.json()``
buffers the *entire* body into memory before parsing it and offers no cap, so a
hostile or merely broken endpoint -- or an attacker who stuffed megabytes of string
data into a log this pipeline must decode -- could exhaust the process's memory
just by answering a request (security review finding M2).

``read_json_capped`` reads the *streamed* body in fixed-size chunks and gives up
as soon as the accumulated size passes the caller's cap, so at most one chunk
beyond the limit is ever held. Callers must have issued the request with
``stream=True`` (otherwise ``requests`` has already buffered the body and the cap
is pointless) and are responsible for closing the response.

The ``ResponseTooLarge`` message deliberately names only the byte limit: never the
URL (which may embed a provider API key in its path or query string) and never any
part of the body itself, so the error is safe to log, print or paste into a bug
report. ``ResponseTooLarge`` subclasses ``ValueError`` so a caller that already
maps "the body did not parse as JSON" onto its own error type keeps working
unchanged -- but callers that want to treat an oversized body differently (the RPC
client makes it non-retryable, since re-requesting cannot shrink it) catch it
first.

``install_key_redaction`` closes the other half of the same trust problem (M1):
an API key kept scrupulously out of *our* messages still reaches the log if a
library below us writes the request line out. ``urllib3`` does exactly that at
DEBUG -- ``GET /v2/api?...&apikey=<the key> HTTP/1.1`` -- so a logging filter
substitutes the key out of its records before any handler sees them.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Optional

# Bytes pulled from the socket per read. Large enough that a normal body costs
# only a handful of reads, small enough that overshooting the cap costs at most
# this much extra memory.
CHUNK_BYTES = 64 * 1024


class ResponseTooLarge(ValueError):
    """A response body exceeded the caller's byte cap and was abandoned
    unparsed. Its message names only the limit -- never the URL or the body --
    so it is safe to log verbatim."""


def read_json_capped(response, max_bytes: int) -> object:
    """JSON-decode ``response``'s body, reading at most ``max_bytes`` of it.

    ``response`` must come from a request made with ``stream=True``: the body is
    pulled from ``response.raw`` in ``CHUNK_BYTES`` chunks (with
    ``decode_content=True``, so gzip/deflate transfer encodings are undone --
    the cap therefore applies to the *decompressed* size, which is what actually
    lands in memory, not to a compressed size an attacker could shrink).

    Raises ``ResponseTooLarge`` (a ``ValueError``) as soon as the accumulated
    body passes ``max_bytes``, without parsing anything; ``ValueError`` if the
    body is not valid UTF-8 or not valid JSON; and ``ValueError`` if
    ``max_bytes`` is not positive or the response carries no ``raw`` stream
    (i.e. the caller forgot ``stream=True``). Closing the response is the
    caller's job.
    """
    if max_bytes <= 0:
        raise ValueError("read_json_capped: max_bytes must be positive")
    raw = getattr(response, "raw", None)
    if raw is None:
        raise ValueError("read_json_capped: response has no raw stream (request it with stream=True)")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = raw.read(CHUNK_BYTES, decode_content=True)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLarge(f"response body exceeds the {max_bytes} byte limit")
        chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8", errors="strict"))


# What an API key is replaced with in a redacted log record (M1).
REDACTED = "[redacted]"

# Loggers urllib3 writes request lines to. A ``logging.Filter`` installed on a
# logger only sees records logged *on that logger* (filters, unlike handlers,
# are not consulted up the hierarchy), so both the package logger and the
# connectionpool logger that actually emits `GET /api?...&apikey=... HTTP/1.1`
# need their own copy.
_URLLIB3_LOGGERS = ("urllib3", "urllib3.connectionpool")

# Installed redactors, keyed by the API key they scrub, so constructing many
# clients with the same key does not stack duplicate filters on a process-wide
# logger. Guarded by ``_redactor_lock``: a client may be built from
# a worker thread.
_installed_redactors: dict = {}
_redactor_lock = threading.Lock()


class ApiKeyRedactor(logging.Filter):
    """Replaces an exact API key value with ``[redacted]`` in every log record
    passing through urllib3's loggers (M1).

    urllib3 logs each request at DEBUG as ``'%s://%s:%s "%s %s %s" %s %s'``
    with the full path+query string among its ``args`` -- so a run with root
    logging at DEBUG (``robustrep --verbose``) would otherwise write
    ``GET /v2/api?...&apikey=<the real key>`` to wherever logs go. The key is
    substituted in place in both ``record.msg`` and every string in
    ``record.args``, so every handler downstream sees the redacted form; the
    record is always allowed through (this filter redacts, it never drops).
    """

    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._scrub(record.msg)
        if record.args:
            record.args = self._scrub_args(record.args)
        return True

    def _scrub_args(self, args):
        if isinstance(args, dict):
            return {k: self._scrub(v) for k, v in args.items()}
        return tuple(self._scrub(a) for a in args)

    def _scrub(self, value):
        if isinstance(value, str) and self.key in value:
            return value.replace(self.key, REDACTED)
        return value


def short_for_log(value, limit: int = 40) -> str:
    """``value`` bounded to ``limit`` characters, for logging a field a remote
    chose. A log topic, a pagination key or an API error string can be
    arbitrarily long, and a WARNING that echoes megabytes of it is its own
    denial of service. Non-strings are ``repr``-ed first, so this is safe to
    hand any value; callers still log the result with ``%r``."""
    s = value if isinstance(value, str) else repr(value)
    return s if len(s) <= limit else f"{s[:limit]}...({len(s)} chars)"


def install_key_redaction(key: Optional[str]) -> None:
    """Install an ``ApiKeyRedactor`` for ``key`` on urllib3's loggers, once
    per distinct key (M1). A falsy key (Blockscout, or an unconfigured
    Etherscan client) installs nothing -- there is no secret to hide, and an
    empty-string match would redact every record."""
    if not key:
        return
    with _redactor_lock:
        if key in _installed_redactors:
            return
        redactor = ApiKeyRedactor(key)
        for name in _URLLIB3_LOGGERS:
            logging.getLogger(name).addFilter(redactor)
        _installed_redactors[key] = redactor
