"""Print the CHANGELOG body for one version, for use as GitHub release notes.

Usage: python scripts/changelog_section.py CHANGELOG.md 0.2.2 [--lint]

The body is everything under the ``## [0.2.2] — <date>`` heading up to the next
``## `` heading or the trailing link-definition block, with the heading itself
left out (the release title carries the version). A link definition inside a
section (``[ref]: https://…`` followed by more prose) is part of the body; only
the block at the very end of the file, where every remaining line is a link
definition or blank, is cut.

Exit 1 with a message on stderr when the version has no section or the section
is empty, so a tag cannot ship an empty release. With ``--lint`` the body is
also checked for things that must not reach the public Releases page and its
notification feed: a bare ``@name`` outside a code span (GitHub linkifies and
notifies it) and the accusatory vocabulary the report wording rule (spec
§5.5b) keeps out of public text. ``publish.yml`` runs the lint in ``build``,
before the irreversible PyPI upload.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")
_LINK_DEF = re.compile(r"^\[[^\]]+\]: \S+\s*$")
_CODE_SPAN = re.compile(r"`[^`]*`")
_MENTION = re.compile(r"(?<![\w./\\-])@[A-Za-z0-9][A-Za-z0-9-]*")
# Whole words that the §5.5b wording rule keeps out of public text: a changelog
# describes measured behaviour, not accusations.
_BANNED_WORDS = ("attack", "cheat", "fraud", "bot")
_BANNED = re.compile(r"\b(" + "|".join(_BANNED_WORDS) + r")\b", re.IGNORECASE)


def _trailing_link_block_start(lines: list[str]) -> int:
    """Index where the file's trailing link-definition block begins.

    Returns ``len(lines)`` when there is no such block. The block is the
    longest suffix made only of link definitions and blank lines that contains
    at least one definition.
    """
    i = len(lines)
    while i > 0 and (not lines[i - 1].strip() or _LINK_DEF.match(lines[i - 1])):
        i -= 1
    return i if any(_LINK_DEF.match(line) for line in lines[i:]) else len(lines)


def section(changelog: str, version: str) -> str:
    """Return the notes body for ``version``.

    Raises LookupError when the section is absent or empty.
    """
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
    end = _trailing_link_block_start(lines)
    body: list[str] = []
    for line in lines[start + 1 : end]:
        if _HEADING.match(line):
            break
        body.append(line)
    text = "\n".join(body).strip()
    if not text:
        raise LookupError(f"empty '## [{version}]' section in changelog")
    return text


def lint(body: str) -> list[str]:
    """Return the problems that make ``body`` unfit for a public release page."""
    prose = _CODE_SPAN.sub("", body)
    problems = [f"bare mention {m.group(0)!r}" for m in _MENTION.finditer(prose)]
    problems += [f"banned word {m.group(0)!r}" for m in _BANNED.finditer(prose)]
    return problems


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if a != "--lint"]
    want_lint = "--lint" in argv[1:]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path, version = Path(args[0]), args[1]
    try:
        body = section(path.read_text(encoding="utf-8"), version)
    except (LookupError, OSError) as exc:
        print(f"changelog_section: {exc}", file=sys.stderr)
        return 1
    if want_lint and (problems := lint(body)):
        for problem in problems:
            print(f"changelog_section: {version}: {problem}", file=sys.stderr)
        return 1
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
