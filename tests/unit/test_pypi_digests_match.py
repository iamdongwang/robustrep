"""scripts/pypi_digests_match.py: release assets must equal what PyPI serves."""

from __future__ import annotations

import hashlib
import importlib.util
import urllib.error
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pypi_digests_match.py"


def _load():
    spec = importlib.util.spec_from_file_location("pypi_digests_match", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _dist(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    dist = tmp_path / "dist"
    dist.mkdir()
    digests = {}
    for name, payload in (("pkg-1.0.0-py3-none-any.whl", b"wheel"), ("pkg-1.0.0.tar.gz", b"sdist")):
        (dist / name).write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    return dist, digests


def test_all_files_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    dist, digests = _dist(tmp_path)
    monkeypatch.setattr(mod, "fetch_digests", lambda project, version: dict(digests))
    assert mod.check("pkg", "1.0.0", dist, retries=1, sleep=0) == []


def test_digest_mismatch_fails_without_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    dist, digests = _dist(tmp_path)
    remote = dict(digests)
    remote["pkg-1.0.0.tar.gz"] = "0" * 64
    calls: list[str] = []

    def fetch(project: str, version: str) -> dict[str, str]:
        calls.append(version)
        return remote

    monkeypatch.setattr(mod, "fetch_digests", fetch)
    problems = mod.check("pkg", "1.0.0", dist, retries=5, sleep=0)
    assert problems == ["sha256 differs from PyPI: pkg-1.0.0.tar.gz"]
    assert calls == ["1.0.0"]


def test_missing_file_is_retried_then_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    dist, digests = _dist(tmp_path)
    partial = {k: v for k, v in digests.items() if k.endswith(".whl")}
    attempts: list[int] = []

    def fetch(project: str, version: str) -> dict[str, str]:
        attempts.append(1)
        return partial

    monkeypatch.setattr(mod, "fetch_digests", fetch)
    problems = mod.check("pkg", "1.0.0", dist, retries=3, sleep=0)
    assert problems == ["not listed on PyPI after 3 tries: pkg-1.0.0.tar.gz"]
    assert len(attempts) == 3


def test_404_counts_as_not_yet_listed_and_later_success_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load()
    dist, digests = _dist(tmp_path)
    state = {"n": 0}

    def fetch(project: str, version: str) -> dict[str, str]:
        state["n"] += 1
        if state["n"] == 1:
            raise urllib.error.HTTPError("u", 404, "not found", None, None)  # type: ignore[arg-type]
        return dict(digests)

    monkeypatch.setattr(mod, "fetch_digests", fetch)
    assert mod.check("pkg", "1.0.0", dist, retries=2, sleep=0) == []
    assert state["n"] == 2


def test_non_404_http_error_fails_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    dist, _ = _dist(tmp_path)

    def fetch(project: str, version: str) -> dict[str, str]:
        raise urllib.error.HTTPError("u", 503, "unavailable", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(mod, "fetch_digests", fetch)
    assert mod.check("pkg", "1.0.0", dist, retries=5, sleep=0) == ["PyPI returned HTTP 503"]


def test_empty_dist_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    dist = tmp_path / "dist"
    dist.mkdir()
    monkeypatch.setattr(mod, "fetch_digests", lambda project, version: {})
    assert mod.check("pkg", "1.0.0", dist, retries=1, sleep=0) == [f"no files in {dist}"]


def test_main_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load()
    dist, digests = _dist(tmp_path)
    monkeypatch.setattr(mod, "fetch_digests", lambda project, version: dict(digests))
    assert mod.main(["x", "pkg", "1.0.0", str(dist), "--retries", "1"]) == 0
    assert "2 file(s) match PyPI" in capsys.readouterr().out
    assert mod.main(["x", "pkg", "1.0.0"]) == 2
    assert mod.main(["x", "pkg", "1.0.0", str(dist), "--retries", "many"]) == 2
    monkeypatch.setattr(mod, "fetch_digests", lambda project, version: {})
    assert mod.main(["x", "pkg", "1.0.0", str(dist), "--retries", "1"]) == 1
    assert "not listed on PyPI" in capsys.readouterr().err
