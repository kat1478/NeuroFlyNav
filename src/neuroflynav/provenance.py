"""Capture a compact provenance manifest for a local command."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import machine, python_implementation, python_version, release, system

SCHEMA_VERSION = "1.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_DISTRIBUTIONS = ("pytest", "ruff", "mypy")


class ProvenanceError(Exception):
    """Raised when required provenance metadata cannot be collected."""


def _run_git(repository_root: Path, *arguments: str) -> str:
    """Run Git without exposing captured output unless it is an error message."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise ProvenanceError("Git executable was not found on PATH") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "Git returned an error"
        raise ProvenanceError(f"Git metadata is unavailable: {detail}") from error
    except OSError as error:
        raise ProvenanceError(f"Unable to run Git: {error}") from error
    return result.stdout.strip()


def _git_metadata(repository_root: Path) -> dict[str, object]:
    """Return the current Git identity and dirty state, without listing changed paths."""
    branch = _run_git(repository_root, "branch", "--show-current")
    commit = _run_git(repository_root, "rev-parse", "HEAD")
    status = _run_git(repository_root, "status", "--porcelain")
    if not commit:
        raise ProvenanceError("Git did not report a commit for HEAD")
    return {
        "branch": branch or None,
        "commit": commit,
        "dirty": bool(status),
    }


def _environment_sha256(repository_root: Path) -> str:
    environment_file = repository_root / "environment.yml"
    if not environment_file.is_file():
        raise ProvenanceError(
            f"Required environment file is missing: {environment_file}"
        )
    try:
        content = environment_file.read_bytes()
    except OSError as error:
        raise ProvenanceError(f"Unable to read environment file: {error}") from error
    return hashlib.sha256(content).hexdigest()


def _tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in TOOL_DISTRIBUTIONS:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError as error:
            raise ProvenanceError(
                f"Installed metadata for {distribution!r} is unavailable; "
                "run this command in the neuroflynav Conda environment"
            ) from error
    return versions


def build_manifest(
    command: str, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, object]:
    """Collect a versioned provenance manifest using only Python's standard library."""
    if not command.strip():
        raise ProvenanceError("The command description must not be empty")

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "command": command,
        "git": _git_metadata(repository_root),
        "environment": {"environment_yml_sha256": _environment_sha256(repository_root)},
        "python": {
            "implementation": python_implementation(),
            "version": python_version(),
        },
        "platform": {
            "system": system(),
            "release": release(),
            "machine": machine(),
        },
        "tools": _tool_versions(),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a local provenance manifest as JSON."
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Exact command string associated with this provenance capture.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write one UTF-8 JSON manifest to stdout, or a useful error to stderr."""
    arguments = _argument_parser().parse_args(argv)
    try:
        manifest = build_manifest(arguments.command)
    except ProvenanceError as error:
        print(f"provenance: error: {error}", file=sys.stderr)
        return 1

    output = json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n"
    sys.stdout.buffer.write(output.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
