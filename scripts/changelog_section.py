"""Print the CHANGELOG body for one version, for use as GitHub release notes.

Usage: python scripts/changelog_section.py CHANGELOG.md 0.2.2

The body is everything under the ``## [0.2.2] — <date>`` heading up to the next
``## `` heading or the trailing link-reference block, with the heading itself
left out (the release title carries the version). Exit 1 with a message on
stderr when the version has no section, so a tag without notes fails loudly in
CI instead of publishing an empty release.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")
_LINK_REF = re.compile(r"^\[[^\]]+\]: \S+$")


def section(changelog: str, version: str) -> str:
    """Return the notes body for ``version``; raise LookupError if absent."""
    lines = changelog.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if (m := _HEADING.match(line)) and m.group("version") == version
        ),
        None,
    )
    if start is None:
        raise LookupError(f"no '## [{version}]' section in changelog")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if _HEADING.match(line) or _LINK_REF.match(line):
            break
        body.append(line)
    return "\n".join(body).strip()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    path, version = Path(argv[1]), argv[2]
    try:
        print(section(path.read_text(encoding="utf-8"), version))
    except (LookupError, OSError) as exc:
        print(f"changelog_section: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
