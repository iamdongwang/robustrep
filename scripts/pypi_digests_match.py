"""Fail unless every file in a dist/ directory is byte-identical to PyPI's copy.

Usage: python scripts/pypi_digests_match.py robustrep 0.2.2 dist/ [--retries 6]

For each file in the directory the sha256 is compared to the digest PyPI
publishes for the same filename in its JSON API. A filename PyPI does not list,
a digest that differs, or an empty directory all fail. The JSON API can lag an
upload by seconds, so a missing version or filename is retried with a short
sleep before it counts as a failure; a digest mismatch is never retried.

Stdlib only on purpose: this runs in the one job of ``publish.yml`` that holds
a write token, and it must not ``pip install`` anything there.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RETRY_SLEEP_SECONDS = 10.0
DEFAULT_RETRIES = 6


def fetch_digests(project: str, version: str) -> dict[str, str]:
    """Return ``{filename: sha256}`` for the version from PyPI's JSON API."""
    url = f"https://pypi.org/pypi/{project}/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.load(resp)
    return {f["filename"]: f["digests"]["sha256"] for f in payload["urls"]}


def local_digests(dist: Path) -> dict[str, str]:
    """Return ``{filename: sha256}`` for every regular file in ``dist``."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(dist.iterdir())
        if p.is_file()
    }


def compare(local: dict[str, str], remote: dict[str, str]) -> tuple[list[str], list[str]]:
    """Split local files into (missing on PyPI, digest differs)."""
    missing = [name for name in local if name not in remote]
    mismatched = [
        name for name, digest in local.items() if name in remote and remote[name] != digest
    ]
    return missing, mismatched


def check(
    project: str,
    version: str,
    dist: Path,
    retries: int = DEFAULT_RETRIES,
    sleep: float = RETRY_SLEEP_SECONDS,
) -> list[str]:
    """Return the problems found; an empty list means every file matches."""
    local = local_digests(dist)
    if not local:
        return [f"no files in {dist}"]
    missing: list[str] = list(local)
    for attempt in range(1, retries + 1):
        try:
            remote = fetch_digests(project, version)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                return [f"PyPI returned HTTP {exc.code}"]
            remote = {}
        missing, mismatched = compare(local, remote)
        if mismatched:
            return [f"sha256 differs from PyPI: {name}" for name in mismatched]
        if not missing:
            return []
        if attempt < retries:
            time.sleep(sleep)
    return [f"not listed on PyPI after {retries} tries: {name}" for name in missing]


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    retries = DEFAULT_RETRIES
    if "--retries" in args:
        i = args.index("--retries")
        try:
            retries = int(args[i + 1])
        except (IndexError, ValueError):
            print(__doc__, file=sys.stderr)
            return 2
        del args[i : i + 2]
    if len(args) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    project, version, dist = args[0], args[1], Path(args[2])
    try:
        problems = check(project, version, dist, retries=retries)
    except (OSError, KeyError, ValueError) as exc:
        print(f"pypi_digests_match: {exc!r}", file=sys.stderr)
        return 1
    for problem in problems:
        print(f"pypi_digests_match: {problem}", file=sys.stderr)
    if not problems:
        print(f"pypi_digests_match: {len(local_digests(dist))} file(s) match PyPI")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
