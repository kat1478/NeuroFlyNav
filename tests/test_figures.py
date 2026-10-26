from pathlib import Path

from neuroflynav.reporting.figures import family_of, render_all


def test_family_mapping() -> None:
    assert family_of("EPGt") == "EPG"
    assert family_of("Delta7_a") == "Δ7"
    assert family_of("vDeltaA_a") == "vΔ"
    assert family_of("MBON01") == "other"


def test_all_figures_render_in_both_modes(tmp_path: Path) -> None:
    written = render_all(tmp_path)
    assert len(written) == 8
    assert all(path.stat().st_size > 10_000 for path in written)
