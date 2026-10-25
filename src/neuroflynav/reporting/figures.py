"""Portfolio figures for the README, rendered in a light and a dark variant.

Figures read small committed files in ``docs/figures/data/`` (result summaries and an
aggregated cell-family connectivity matrix), so they can be rebuilt without the raw
connectome. ``--refresh-matrix`` recomputes the matrix from the local hemibrain export.

Colours follow a validated categorical order (blue, orange, aqua) and a single-hue blue
ramp for magnitude; identity is never carried by colour alone (marker shapes, direct
labels), and the numbers behind every chart are tables in ``docs/spike_*.md``.

Example:
    PYTHONPATH=src python -m neuroflynav.reporting.figures --out docs/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure

DATA_DIR = Path("docs/figures/data")

# Cell-type families in rough circuit order: inputs → compass → steering → fan-shaped body.
FAMILIES: tuple[tuple[str, str], ...] = (
    ("ER", r"^ER"),
    ("ExR", r"^ExR"),
    ("EPG", r"^EPG"),
    ("PEN", r"^PEN"),
    ("PEG", r"^PEG"),
    ("Δ7", r"^Delta7"),
    ("EL", r"^EL"),
    ("PFN", r"^PFN"),
    ("PFG", r"^PFG"),
    ("PFR", r"^PFR"),
    ("PFL", r"^PFL"),
    ("hΔ", r"^hDelta"),
    ("vΔ", r"^vDelta"),
    ("FC", r"^FC"),
    ("FS", r"^FS"),
    ("FR", r"^FR"),
    ("FB", r"^FB"),
    ("SA", r"^SA"),
)
OTHER = "other"


@dataclass(frozen=True)
class Theme:
    """Chart chrome and data colours for one mode (reference palette, both modes)."""

    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    baseline: str
    entity: dict[str, str]  # one fixed colour per compared graph
    ramp: tuple[str, ...]  # sequential blue, from near-surface to strongest


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    entity={
        "FlyConnectome": "#2a78d6",
        "ShuffledFly": "#eb6834",
        "RandomRecurrent": "#1baf7a",
        "TypeShuffledFly": "#6250d6",
    },
    ramp=(
        "#fcfcfb",
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ),
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    entity={
        "FlyConnectome": "#3987e5",
        "ShuffledFly": "#d95926",
        "RandomRecurrent": "#199e70",
        "TypeShuffledFly": "#9085e9",
    },
    ramp=(
        "#1a1a19",
        "#0d366b",
        "#184f95",
        "#256abf",
        "#3987e5",
        "#6da7ec",
        "#9ec5f4",
        "#cde2fb",
    ),
)
# Colour follows the entity in every figure; the marker is a second channel for identity.
MARKERS = {
    "FlyConnectome": "o",
    "ShuffledFly": "s",
    "RandomRecurrent": "^",
    "TypeShuffledFly": "D",
}
LABELS = {
    "FlyConnectome": "real CX wiring",
    "ShuffledFly": "degree-shuffled wiring",
    "RandomRecurrent": "random wiring",
    "TypeShuffledFly": "shuffled within cell types",
}


def family_of(cell_type: str) -> str:
    """Map a hemibrain cell type to one of the plotted families."""
    for label, pattern in FAMILIES:
        if re.match(pattern, cell_type):
            return label
    return OTHER


def compute_family_matrix(data_dir: Path) -> tuple[list[str], np.ndarray, int, int]:
    """Total synapses between families in the CX subgraph (rows: source, cols: target)."""
    from neuroflynav.graph.hemibrain import load_cx_subgraph

    subgraph = load_cx_subgraph(data_dir)
    labels = [label for label, _ in FAMILIES] + [OTHER]
    index = {label: i for i, label in enumerate(labels)}
    families = [index[family_of(t)] for t in subgraph.cell_types]
    matrix = np.zeros((len(labels), len(labels)))
    for source, target, weight in subgraph.edges:
        matrix[families[source], families[target]] += weight
    return labels, matrix, len(subgraph.body_ids), int(matrix.sum())


def write_family_matrix(path: Path, data_dir: Path) -> None:
    labels, matrix, neurons, synapses = compute_family_matrix(data_dir)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"# neurons={neurons} synapses={synapses}", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *(int(v) for v in row)])


def read_family_matrix(path: Path) -> tuple[list[str], np.ndarray, str]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, labels = rows[0][0], rows[0][1:]
    matrix = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    return labels, matrix, header


def _style(fig: Figure, theme: Theme) -> None:
    fig.patch.set_facecolor(theme.surface)
    for ax in fig.axes:
        ax.set_facecolor(theme.surface)
        ax.tick_params(colors=theme.ink_secondary, labelsize=9, length=0)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(theme.baseline)


def _titles(fig: Figure, theme: Theme, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.97, title, fontsize=15, weight="bold", color=theme.ink, va="top")
    fig.text(0.02, 0.915, subtitle, fontsize=10, color=theme.ink_secondary, va="top")


def plot_wiring(
    labels: Sequence[str], matrix: np.ndarray, header: str, theme: Theme
) -> Figure:
    """Hero figure: family-to-family synapse counts, log colour scale."""
    neurons, synapses = (int(v) for v in re.findall(r"\d+", header))
    fig, ax = plt.subplots(figsize=(8.0, 7.6), dpi=200)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.84, bottom=0.12)
    cmap = LinearSegmentedColormap.from_list(f"ramp-{theme.name}", theme.ramp)
    shown = np.where(matrix > 0, matrix, np.nan)
    image = ax.imshow(
        shown,
        cmap=cmap,
        norm=LogNorm(vmin=10, vmax=np.nanmax(shown)),
        interpolation="nearest",
    )
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("to (postsynaptic family)", color=theme.ink_secondary, fontsize=10)
    ax.set_ylabel("from (presynaptic family)", color=theme.ink_secondary, fontsize=10)
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=theme.surface, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("synapses (log scale)", color=theme.ink_secondary, fontsize=9)
    bar.ax.tick_params(colors=theme.ink_secondary, labelsize=8, length=0)
    bar.outline.set_visible(False)
    _style(fig, theme)
    ax.spines["bottom"].set_visible(False)
    _titles(
        fig,
        theme,
        "Who talks to whom in the fly's navigation centre",
        f"Hemibrain central complex: {neurons:,} neurons, {synapses / 1e6:.1f} M synapses, "
        "grouped by cell-type family",
    )
    fig.text(
        0.02,
        0.02,
        "Data: hemibrain v1.2, Janelia FlyEM (CC BY 4.0), Scheffer et al. 2020",
        fontsize=8,
        color=theme.muted,
    )
    return fig


def plot_forest(summary: dict[str, object], theme: Theme) -> Figure:
    """Paired differences FlyConnectome − control, one row per pre-registered experiment."""
    rows = summary["forest"]
    assert isinstance(rows, list)
    controls = ("ShuffledFly", "TypeShuffledFly", "RandomRecurrent")
    names = {
        "ShuffledFly": "vs degree-shuffled wiring",
        "TypeShuffledFly": "vs wiring shuffled within cell types",
        "RandomRecurrent": "vs random wiring",
    }
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=200)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.78, bottom=0.16)
    offsets = (0.22, 0.0, -0.22)  # same top-to-bottom order as the legend
    for position, row in enumerate(rows):
        y = len(rows) - 1 - position
        for k, control in enumerate(controls):
            low, mean, high = (100 * v for v in row["diff"][control])
            ax.plot(
                [low, high],
                [y + offsets[k]] * 2,
                color=theme.entity[control],
                linewidth=2,
                solid_capstyle="round",
            )
            ax.plot(
                mean,
                y + offsets[k],
                MARKERS[control],
                color=theme.entity[control],
                markersize=8,
                markeredgecolor=theme.surface,
                markeredgewidth=1.5,
                label=names[control] if position == 0 else None,
            )
    ax.axvline(0, color=theme.ink_secondary, linewidth=1)
    ax.set_yticks(
        range(len(rows)),
        [r["label"] for r in reversed(rows)],
        fontsize=9.5,
        color=theme.ink,
    )
    ax.set_xlabel(
        "FlyConnectome minus control, percentage points of success / accuracy\n"
        "(95% CI, paired over 20 seeds)",
        color=theme.ink_secondary,
        fontsize=9,
    )
    ax.grid(axis="x", color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.3, len(rows) - 0.45, "real wiring better →", color=theme.muted, fontsize=8.5
    )
    ax.text(
        -0.3,
        len(rows) - 0.45,
        "← control better",
        color=theme.muted,
        fontsize=8.5,
        ha="right",
    )
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.32, -0.16),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handlelength=1.2,
        columnspacing=1.2,
    )
    for item in legend.get_texts():
        item.set_color(theme.ink_secondary)
    _style(fig, theme)
    _titles(
        fig,
        theme,
        "Does the real wiring help? A little — and less as models get better",
        "Pre-registered comparisons on fresh seeds; every gap is within ~6 points of zero",
    )
    return fig


def plot_progression(summary: dict[str, object], theme: Theme) -> Figure:
    """Heading accuracy by model family: learning, not wiring, makes the compass."""
    stages = summary["progression"]
    assert isinstance(stages, list)
    references = summary["references"]
    assert isinstance(references, dict)
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=200)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    x = np.arange(len(stages))
    series = (
        ("FlyConnectome", "real CX wiring"),
        ("ShuffledFly", "degree-shuffled wiring"),
    )
    wired = [i for i, s in enumerate(stages) if not s.get("no_wiring")]
    for k, (key, label) in enumerate(series):
        ax.plot(
            x[wired] + (k - 0.5) * 0.12,
            [100 * stages[i][key] for i in wired],
            MARKERS[key],
            color=theme.entity[key],
            markersize=9,
            markeredgecolor=theme.surface,
            markeredgewidth=1.5,
            label=label,
        )
    for i, stage in enumerate(stages):
        if stage.get(
            "no_wiring"
        ):  # no graph at all: one neutral point, not a comparison
            ax.plot(
                x[i],
                100 * stage["FlyConnectome"],
                "X",
                color=theme.muted,
                markersize=8,
                markeredgecolor=theme.surface,
                markeredgewidth=1.5,
            )
    for value, text in (
        (references["oracle"], "perfect memory of the last landmark"),
        (references["chance"], "chance (8 sectors)"),
    ):
        ax.axhline(100 * value, color=theme.muted, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(
            -0.45,
            100 * value + 1.5,
            text,
            va="bottom",
            fontsize=8.5,
            color=theme.ink_secondary,
        )
    ax.set_xticks(x, [s["label"] for s in stages], fontsize=9.5, color=theme.ink)
    ax.set_xlim(-0.5, len(stages) - 0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("heading accuracy, %", color=theme.ink_secondary, fontsize=9)
    ax.grid(axis="y", color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for item in legend.get_texts():
        item.set_color(theme.ink_secondary)
    _style(fig, theme)
    _titles(
        fig,
        theme,
        "Learning, not wiring, makes the compass",
        "Keeping track of heading from self-motion between landmarks; mean of 20 seeds per stage",
    )
    return fig


def plot_learning_curve(summary: dict[str, object], theme: Theme) -> Figure:
    """Accuracy vs training data for trained masks: does the real wiring learn faster?"""
    learning = summary["learning"]
    assert isinstance(learning, dict)
    sizes = learning["sizes"]
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=200)
    fig.subplots_adjust(left=0.09, right=0.8, top=0.78, bottom=0.14)
    frozen = [100 * v for v in learning["FrozenFly"]]
    ax.plot(sizes, frozen, color=theme.muted, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(
        sizes[-1] * 1.08,
        frozen[-1],
        "frozen weights\n(no training)",
        va="center",
        fontsize=8.5,
        color=theme.ink_secondary,
    )
    for key in ("RandomRecurrent", "ShuffledFly", "FlyConnectome"):
        values = [100 * v for v in learning[key]]
        ax.plot(
            sizes,
            values,
            color=theme.entity[key],
            linewidth=2,
            marker=MARKERS[key],
            markersize=8,
            markeredgecolor=theme.surface,
            markeredgewidth=1.5,
            label=LABELS[key],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(s) for s in sizes], fontsize=9.5, color=theme.ink)
    ax.minorticks_off()
    ax.set_ylim(30, 90)
    ax.set_xlabel("training sequences", color=theme.ink_secondary, fontsize=9)
    ax.set_ylabel("heading accuracy, %", color=theme.ink_secondary, fontsize=9)
    ax.grid(axis="y", color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for item in legend.get_texts():
        item.set_color(theme.ink_secondary)
    _style(fig, theme)
    _titles(
        fig,
        theme,
        "More data helps every mask — the real wiring is not a head start",
        "Trained weights on the compass-core mask (409 neurons); mean of 20 seeds per point "
        "(n = 256 from a separate seed set)",
    )
    return fig


def render_all(out: Path) -> list[Path]:
    """Render every figure in both modes; return the written paths."""
    summary = json.loads((DATA_DIR / "summary.json").read_text())
    labels, matrix, header = read_family_matrix(DATA_DIR / "cx_family_matrix.csv")
    written: list[Path] = []
    for theme in (LIGHT, DARK):
        for name, fig in (
            ("cx-wiring", plot_wiring(labels, matrix, header, theme)),
            ("results-forest", plot_forest(summary, theme)),
            ("heading-progression", plot_progression(summary, theme)),
            ("learning-curve", plot_learning_curve(summary, theme)),
        ):
            path = out / f"{name}-{theme.name}.png"
            fig.savefig(path, facecolor=theme.surface)
            plt.close(fig)
            written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render README figures.")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--refresh-matrix", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/hemibrain/v1.2/exported-traced-adjacencies-v1.2"),
    )
    arguments = parser.parse_args(argv)
    if arguments.refresh_matrix:
        write_family_matrix(DATA_DIR / "cx_family_matrix.csv", arguments.data_dir)
    for path in render_all(arguments.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
