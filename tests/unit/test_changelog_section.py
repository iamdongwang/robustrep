"""scripts/changelog_section.py: the release-notes body for one version."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "changelog_section.py"

SAMPLE = """# Changelog

Preamble that is not part of any release.

## [Unreleased]

- pending item

## [0.2.2] — 2026-10-13

Intro line for 0.2.2.

### Added

- **Owner lookup.** Body text.

## [0.2.1] — 2026-10-13

- 0.2.1 only line

## [0.1.0] — 2026-09-14

Initial release.

[0.2.2]: https://example.invalid/compare/v0.2.1...v0.2.2
[0.2.1]: https://example.invalid/compare/v0.1.0...v0.2.1
"""


def _load():
    spec = importlib.util.spec_from_file_location("changelog_section", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_section_is_body_without_heading_or_neighbours() -> None:
    mod = _load()
    body = mod.section(SAMPLE, "0.2.2")
    assert body.startswith("Intro line for 0.2.2.")
    assert "### Added" in body
    assert "0.2.1 only line" not in body
    assert "## [0.2.2]" not in body
    assert "pending item" not in body


def test_last_section_stops_before_link_references() -> None:
    mod = _load()
    body = mod.section(SAMPLE, "0.1.0")
    assert body == "Initial release."


def test_trailing_link_definition_with_trailing_space_is_still_cut() -> None:
    mod = _load()
    text = "## [1.0.0] — 2026-01-01\n\nBody.\n\n[1.0.0]: https://example.invalid/x   \n"
    assert mod.section(text, "1.0.0") == "Body."


def test_link_definition_inside_a_section_is_kept() -> None:
    mod = _load()
    text = (
        "## [1.1.0] — 2026-01-02\n\nSee [the issue][ref] for context.\n\n"
        "[ref]: https://example.invalid/issue\n\n- a bullet after the definition\n\n"
        "## [1.0.0] — 2026-01-01\n\nOld.\n\n[1.1.0]: https://example.invalid/compare\n"
    )
    body = mod.section(text, "1.1.0")
    assert "[ref]: https://example.invalid/issue" in body
    assert body.endswith("- a bullet after the definition")
    assert "Old." not in body


def test_missing_version_raises() -> None:
    mod = _load()
    with pytest.raises(LookupError, match="0.9.9"):
        mod.section(SAMPLE, "0.9.9")


def test_empty_section_raises() -> None:
    mod = _load()
    text = "## [2.0.0] — 2026-02-02\n\n\n## [1.0.0] — 2026-01-01\n\nOld.\n"
    with pytest.raises(LookupError, match="empty '## \\[2.0.0\\]'"):
        mod.section(text, "2.0.0")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Thanks to @someone for the report.", ["bare mention '@someone'"]),
        ("The `@decorator` form and user\\@example.com stay.", []),
        ("We now reject an Attack on the fetcher.", ["banned word 'Attack'"]),
        ("Rejects `bot` traffic in a code span.", []),
        ("A robot and a botnet are different words.", []),
        ("Clean notes about measured behaviour.", []),
    ],
)
def test_lint_flags_mentions_and_banned_words_outside_code(
    body: str, expected: list[str]
) -> None:
    assert _load().lint(body) == expected


def test_cli_prints_body_and_fails_on_unknown_version(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE, encoding="utf-8")
    ok = subprocess.run(
        [sys.executable, str(SCRIPT), str(changelog), "0.2.1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0
    assert ok.stdout.strip() == "- 0.2.1 only line"
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), str(changelog), "9.9.9"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert "9.9.9" in bad.stderr


def test_cli_lint_fails_and_prints_nothing_to_stdout(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "## [3.0.0] — 2026-03-03\n\nThanks @reporter, this stops the fraud.\n",
        encoding="utf-8",
    )
    res = subprocess.run(
        [sys.executable, str(SCRIPT), str(changelog), "3.0.0", "--lint"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 1
    assert res.stdout == ""
    assert "bare mention '@reporter'" in res.stderr
    assert "banned word 'fraud'" in res.stderr
