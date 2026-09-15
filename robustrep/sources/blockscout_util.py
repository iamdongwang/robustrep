"""Parsing helpers for Blockscout v2 responses, kept honest about their source.

Blockscout is a third party: everything in its response body is chosen by
whoever runs (or compromises) that instance, and two fields are load-bearing
enough to be worth their own hardening, per the security review.

- ``parse_timestamp`` (M4) turns an ISO8601 transaction timestamp into epoch
  seconds and refuses -- rather than guesses -- anything it cannot parse. It
  also pins a naive timestamp (no offset, no trailing ``Z``) to UTC: without
  that, ``datetime.timestamp()`` would silently interpret it in whatever local
  zone the machine happens to run in, so the same response would yield
  different first-seen times on a laptop and in CI. Plausibility (is this
  epoch second one a real chain could have produced?) is the caller's call --
  see ``rater_profile._parse_blockscout_timestamp``.
- ``safe_next_page_params`` (L5) allowlists the pagination parameters that get
  echoed straight back onto the *next* request we make. A server-chosen value
  copied verbatim into our own outbound query string is remote control over
  our request, so only short lowercase keys with small scalar values survive,
  and only a handful of them.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from .http_util import short_for_log

logger = logging.getLogger(__name__)

# Pagination keys we are willing to echo back as query parameters: every real
# one is a short lowercase snake_case name (``block_number``, ``index``,
# ``items_count``, ...).
_KEY_RE = re.compile(r"^[a-z_]{1,32}$")
# Most keys forwarded from one page to the next (Blockscout sends 2-4), and the
# longest rendered value accepted for any of them -- caps on how much
# server-chosen text can end up in a URL we build.
MAX_KEYS = 8
MAX_VALUE_CHARS = 128


def parse_timestamp(ts) -> Optional[int]:
    """Epoch seconds for a Blockscout v2 ISO8601 timestamp (e.g.
    ``"2026-02-22T21:16:19.000000Z"``), or ``None`` if it does not parse.

    ``datetime.fromisoformat`` doesn't accept a trailing ``Z`` (replaced with
    ``+00:00``) and, on Python < 3.11, only accepts a 3- or 6-digit
    fractional-second component -- so fractional seconds, if any, are stripped
    rather than relied upon. A timestamp with no timezone at all is read as
    UTC, which is what Blockscout serves and what makes the result independent
    of the machine's local zone."""
    if not isinstance(ts, str):
        return None
    try:
        text = re.sub(r"\.\d+", "", ts.replace("Z", "+00:00"))
        parsed = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return int(parsed.timestamp())
    except (OverflowError, OSError, ValueError):  # pragma: no cover - platform-dependent
        return None


def _forwardable(key, value) -> bool:
    """True for a pagination key/value pair safe to put in our next request:
    an allowlisted key name, a scalar (``str``/``int``/``bool``, never a float,
    list or nested object) value, and a rendered length small enough that the
    remote cannot stuff a URL with it."""
    if not isinstance(key, str) or not _KEY_RE.match(key):
        return False
    if not isinstance(value, (str, int, bool)):
        return False
    return len(value if isinstance(value, str) else str(value)) <= MAX_VALUE_CHARS


def safe_next_page_params(next_params, address: str) -> dict:
    """The subset of Blockscout's ``next_page_params`` safe to send back (L5).

    Forwarding them verbatim would let the remote inject arbitrary query
    parameters into our next request (overriding our own ``filter=to``, adding
    keys the endpoint treats specially) and, if the shape is not a flat
    mapping, hand ``requests`` something it encodes in surprising ways. Pairs
    that fail ``_forwardable`` are dropped with a WARNING naming the rejected
    key (``%r``, length-bounded -- it is remote text), and no more than
    ``MAX_KEYS`` survive. A non-mapping ``next_params`` yields ``{}``, which
    the caller treats as "stop paging"."""
    if not isinstance(next_params, dict):
        logger.warning("blockscout: next_page_params for %r is not an object (%s) - not paging further",
                        address, type(next_params).__name__)
        return {}
    safe: dict = {}
    for key, value in next_params.items():
        if len(safe) >= MAX_KEYS:
            logger.warning("blockscout: next_page_params for %r has %d keys - forwarding only the "
                            "first %d", address, len(next_params), MAX_KEYS)
            break
        if _forwardable(key, value):
            safe[key] = value
        else:
            logger.warning("blockscout: dropping unsafe pagination key %r for %r",
                            short_for_log(key), address)
    return safe
