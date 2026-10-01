"""Focused tests for the local provenance manifest."""

import hashlib
import json
import re
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import machine, python_implementation, python_version, release, system

import pytest

from neuroflynav import provenance


def _fake_git(monkeypatch: pytest.MonkeyPatch, status: str = "") -> None:
    def run_git(repository_root: Path, *arguments: str) -> str:
        del repository_root
        if arguments == ("branch", "--show-current"):
            return "feature/test"
        if arguments == ("rev-parse", "HEAD"):
            return "0123456789abcdef0123456789abcdef01234567"
        if arguments == ("status", "--porcelain"):
            return status
        raise AssertionError(f"Unexpected Git arguments: {arguments}")

    monkeypatch.setattr(provenance, "_run_git", run_git)


def _repository_with_environment_file(tmp_path: Path) -> Path:
    (tmp_path / "environment.yml").write_text("name: test\n", encoding="utf-8")
    return tmp_path


def test_manifest_has_documented_schema_and_captures_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_git(monkeypatch)
    repository_root = _repository_with_environment_file(tmp_path)

    manifest = provenance.build_manifest("python -m pytest", repository_root)

    assert manifest["schema_version"] == "1.0"
    assert manifest["command"] == "python -m pytest"
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(manifest["captured_at_utc"])
    )
    assert datetime.fromisoformat(str(manifest["captured_at_utc"])).tzinfo == UTC
    assert manifest["git"] == {
        "branch": "feature/test",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "dirty": False,
    }
    assert manifest["environment"] == {
        "environment_yml_sha256": hashlib.sha256(b"name: test\n").hexdigest()
    }
    assert manifest["python"] == {
        "implementation": python_implementation(),
        "version": python_version(),
    }
    assert manifest["platform"] == {
        "system": system(),
        "release": release(),
        "machine": machine(),
    }
    assert manifest["tools"] == {
        name: version(name) for name in provenance.TOOL_DISTRIBUTIONS
    }
    assert json.loads(json.dumps(manifest)) == manifest


def test_dirty_state_is_boolean_and_does_not_expose_paths_or_contents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_git(monkeypatch, status=" M private-data.csv\n?? sealed-result.json")
    repository_root = _repository_with_environment_file(tmp_path)

    manifest = provenance.build_manifest("python run.py", repository_root)
    encoded = json.dumps(manifest)

    assert manifest["git"] == {
        "branch": "feature/test",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "dirty": True,
    }
    assert "private-data.csv" not in encoded
    assert "sealed-result.json" not in encoded


def test_detached_head_is_explicitly_represented(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        provenance,
        "_run_git",
        lambda repository_root, *arguments: (
            ""
            if arguments == ("branch", "--show-current")
            else "0123456789abcdef0123456789abcdef01234567"
            if arguments == ("rev-parse", "HEAD")
            else ""
        ),
    )
    repository_root = _repository_with_environment_file(tmp_path)

    manifest = provenance.build_manifest("python -m pytest", repository_root)

    assert manifest["git"] == {
        "branch": None,
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "dirty": False,
    }


def test_missing_git_metadata_fails_with_actionable_error(tmp_path: Path) -> None:
    repository_root = _repository_with_environment_file(tmp_path)

    with pytest.raises(provenance.ProvenanceError, match="Git metadata is unavailable"):
        provenance.build_manifest("python -m pytest", repository_root)


def test_missing_environment_file_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_git(monkeypatch)

    with pytest.raises(provenance.ProvenanceError, match="environment file is missing"):
        provenance.build_manifest("python -m pytest", tmp_path)


def test_missing_tool_metadata_fails_with_environment_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_git(monkeypatch)
    _repository_with_environment_file(tmp_path)

    def missing_version(distribution: str) -> str:
        del distribution
        raise PackageNotFoundError

    monkeypatch.setattr(provenance, "version", missing_version)

    with pytest.raises(
        provenance.ProvenanceError, match="neuroflynav Conda environment"
    ):
        provenance.build_manifest("python -m pytest", tmp_path)


def test_empty_command_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_git(monkeypatch)
    _repository_with_environment_file(tmp_path)

    with pytest.raises(provenance.ProvenanceError, match="must not be empty"):
        provenance.build_manifest("  ", tmp_path)


def test_cli_requires_command_argument() -> None:
    with pytest.raises(SystemExit, match="2"):
        provenance.main([])
