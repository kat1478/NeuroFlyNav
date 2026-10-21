"""Smoke tests for the NeuroFlyNav package."""

import tomllib
from pathlib import Path

import neuroflynav


def test_package_can_be_imported() -> None:
    """The package is importable from the source tree."""
    assert neuroflynav.__name__ == "neuroflynav"


def test_version_matches_pyproject() -> None:
    """The package version and pyproject.toml stay in sync."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    assert neuroflynav.__version__ == project["version"]
