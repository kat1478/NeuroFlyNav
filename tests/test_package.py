"""Smoke tests for the NeuroFlyNav package."""

import neuroflynav


def test_package_can_be_imported() -> None:
    """The package is importable from the source tree."""
    assert neuroflynav.__name__ == "neuroflynav"
