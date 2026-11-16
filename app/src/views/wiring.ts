import * as Plot from "@observablehq/plot";
import { loadWiring, type Wiring } from "../data";
import {
  cellValue,
  FAMILY_GROUPS,
  FAMILY_INFO,
  familyName,
  REGIONS,
  topPartners,
  type Mode,
  type Partner,
} from "../families";
import { onWidthChange, type Scope } from "../lifecycle";
import { onScheme, palette, type Palette } from "../theme";
import { card, el, pct, table } from "../ui";

interface Cell {
  source: string;
  target: string;
  si: number;
  ti: number;
  synapses: number;
  share: number;
}

const fmt = (n: number): string => n.toLocaleString("en");

function cells(wiring: Wiring): Cell[] {
  const { labels, matrix } = wiring;
  return labels.flatMap((source, si) =>
    labels.map((target, ti) => ({
      source,
      target,
      si,
      ti,
      synapses: matrix[si][ti],
      share: cellValue(matrix, si, ti, "share"),
    })),
  );
}

function heatmap(
  wiring: Wiring,
  data: Cell[],
  mode: Mode,
  selected: number,
  p: Palette,
  width: number,
): ReturnType<typeof Plot.plot> {
  const nonzero = data.filter((d) => d.synapses > 0);
  const value = (d: Cell): number => (mode === "synapses" ? d.synapses : d.share);
  const values = nonzero.map(value);
  const size = Math.min(width, 640);
  const label = wiring.labels[selected];
  return Plot.plot({
    width: size,
    height: size,
    marginLeft: 64,
    marginTop: size < 560 ? 84 : 64,
    marginBottom: 8,
    marginRight: 8,
    padding: 0.06,
    style: { background: "transparent", color: p.ink2, fontSize: "11px" },
    x: {
      domain: wiring.labels,
      axis: "top",
      label: "to (receives) →",
      labelAnchor: "center",
      labelOffset: size < 560 ? 64 : 40,
      tickSize: 0,
      tickRotate: size < 560 ? -90 : 0,
    },
    y: {
      domain: wiring.labels,
      label: "↓ from (sends)",
      labelAnchor: "center",
      labelOffset: 50,
      tickSize: 0,
    },
    color: {
      type: "log",
      domain: [Math.min(...values), Math.max(...values)],
      range: [p.rampLo, p.rampHi],
      interpolate: "hcl",
      legend: true,
      ticks: 5,
      label:
        mode === "synapses" ? "Synapses (log scale)" : "Share of the source family's output (log)",
      tickFormat: mode === "synapses" ? "~s" : (d: number) => pct(d, d < 0.01 ? 1 : 0),
      width: Math.min(size, 360),
    },
    marks: [
      Plot.cell(data, {
        x: "target",
        y: "source",
        fill: p.surface,
        stroke: p.grid,
        strokeWidth: 0.5,
      }),
      Plot.cell(nonzero, { x: "target", y: "source", fill: value }),
      // No synapses at all: a small dot, so "none" never looks like the lowest colour.
      Plot.dot(
        data.filter((d) => d.synapses === 0),
        { x: "target", y: "source", r: 1.6, fill: p.muted },
      ),
      Plot.cell(
        data.filter((d) => d.source === label),
        { x: "target", y: "source", fill: "none", stroke: p.ink, strokeWidth: 1.5 },
      ),
      Plot.tip(
        data,
        Plot.pointer({
          x: "target",
          y: "source",
          title: (d: Cell) =>
            `${fmt(d.synapses)} synapses\n` +
            `from ${d.source} · ${familyName(d.source)}\n` +
            `to ${d.target} · ${familyName(d.target)}\n` +
            `${pct(d.share)} of ${d.source}'s output in the CX`,
          fill: p.raised,
          stroke: p.grid,
          fontSize: 12,
        }),
      ),
    ],
  });
}

function partnerList(title: string, partners: Partner[]): HTMLElement[] {
  const top = partners[0]?.share || 1;
  return [
    el("p", { className: "partners-title", text: title }),
    el(
      "ul",
      { className: "partners" },
      ...partners.map((partner) => {
        const name = el("abbr", { text: partner.family });
        name.title = familyName(partner.family);
        const bar = el("span", { className: "partner-bar" });
        bar.style.width = `${Math.max(4, (100 * partner.share) / top)}%`;
        return el(
          "li",
          {},
          name,
          el("span", {}, bar),
          el("span", { className: "num", text: pct(partner.share, 0) }),
        );
      }),
    ),
  ];
}

function familyCard(wiring: Wiring, index: number): HTMLElement {
  const label = wiring.labels[index];
  const info = FAMILY_INFO[label] ?? { name: label, role: "" };
  const out = wiring.matrix[index].reduce((a, b) => a + b, 0);
  const into = wiring.matrix.reduce((sum, row) => sum + row[index], 0);
  const stat = (value: string, name: string): HTMLElement =>
    el(
      "div",
      {},
      el("span", { className: "stat-value", text: value }),
      el("span", { className: "stat-label", text: name }),
    );
  return el(
    "aside",
    { className: "family-card" },
    el("h3", { text: info.name }),
    el("p", { text: info.role }),
    el(
      "div",
      { className: "stats" },
      stat(fmt(wiring.neurons[index] ?? 0), "neurons"),
      stat(fmt(out), "synapses out"),
      stat(fmt(into), "synapses in"),
    ),
    ...partnerList(
      "Sends most to (share of its output)",
      topPartners(wiring.labels, wiring.matrix, index, "out"),
    ),
    ...partnerList(
      "Hears most from (share of its input)",
      topPartners(wiring.labels, wiring.matrix, index, "in"),
    ),
  );
}

function glossary(): HTMLElement {
  const regions = el(
    "dl",
    { className: "regions" },
    ...REGIONS.flatMap(([term, text]) => [el("dt", { text: term }), el("dd", { text })]),
  );
  const groups = FAMILY_GROUPS.map((group) =>
    el(
      "div",
      { className: "glossary-group" },
      el("h4", { text: group.title }),
      el(
        "dl",
        {},
        ...group.families.flatMap((label) => [
          el("dt", {}, el("strong", { text: label }), ` — ${familyName(label)}`),
          el("dd", { text: FAMILY_INFO[label]?.role ?? "" }),
        ]),
      ),
    ),
  );
  return el(
    "details",
    { className: "glossary" },
    el("summary", {
      text: "What do the abbreviations mean? (cell-type families and brain regions)",
    }),
    el(
      "p",
      { className: "note" },
      "Neuroscientists name these neurons after the regions they connect: P-EN, for example, " +
        "runs from the protocerebral bridge (P) to the ellipsoid body (E) and the noduli (N). " +
        "Δ (delta) types are local interneurons. Many roles are still hypotheses.",
    ),
    el("h4", { text: "Brain regions" }),
    regions,
    ...groups,
  );
}

export async function renderWiring(root: HTMLElement, scope: Scope): Promise<void> {
  const wiring = await loadWiring();
  if (!scope.active) return;
  const data = cells(wiring);
  let mode: Mode = "synapses";
  let selected = wiring.labels.indexOf("EPG");

  root.append(
    el("h1", { text: "The wiring the agent is built on" }),
    el("p", {
      className: "lede",
      text:
        `The central complex of the hemibrain connectome: ${fmt(wiring.total_neurons)} neurons and ` +
        `${fmt(wiring.total_synapses)} synapses, grouped here into ${wiring.labels.length} cell-type ` +
        "families. The agent uses the neuron-level version of this graph; the controls scramble it.",
    }),
  );

  const view = card(
    "Who talks to whom",
    "Rows send, columns receive: each cell counts synapses from the row family to the column " +
      "family. Stronger colour = more synapses (see the scale); a small dot = none. Hover or tap " +
      "a cell for numbers; click it, or pick a family, to read about the sending family. The " +
      "abbreviations are explained below the chart.",
  );
  const picker = el("select");
  picker.setAttribute("aria-label", "Family");
  wiring.labels.forEach((label, i) => {
    const option = el("option", { text: FAMILY_INFO[label]?.name ?? label });
    option.value = String(i);
    picker.append(option);
  });
  const modeSwitch = el("div", { className: "segmented" });
  modeSwitch.setAttribute("role", "group");
  modeSwitch.setAttribute("aria-label", "Colour by");
  const modeButtons = (
    [
      ["synapses", "synapse count"],
      ["share", "share of output"],
    ] as [Mode, string][]
  ).map(([value, text]) => {
    const button = el("button", { text });
    button.type = "button";
    button.addEventListener("click", () => {
      mode = value;
      draw();
    });
    modeSwitch.append(button);
    return { value, button };
  });
  const controls = el(
    "div",
    { className: "controls" },
    el("label", {}, "Family ", picker),
    el("span", { text: "Colour by" }),
    modeSwitch,
  );
  const side = el("div");
  const layout = el("div", { className: "wiring-layout" }, view.chart, side);
  view.legend.replaceWith(controls, layout, glossary()); // the chart now lives inside the layout
  view.tableSlot.append(
    table(
      data.filter((d) => d.synapses > 0).sort((a, b) => b.synapses - a.synapses),
      [
        { label: "From", value: (d) => d.source },
        { label: "To", value: (d) => d.target },
        { label: "Synapses", value: (d) => fmt(d.synapses), numeric: true },
        { label: "Share of source output", value: (d) => pct(d.share), numeric: true },
      ],
      "Synapses between cell-type families, strongest first",
    ),
  );
  root.append(
    view.root,
    el("p", {
      className: "note",
      text:
        "Data: hemibrain v1.2 (Scheffer et al., 2020), Janelia FlyEM, CC BY 4.0; neurons with a " +
        "synapse in the CX regions, connections with weight ≥ 1. Family descriptions follow " +
        "Hulse et al. (2021). Only these family-level totals are published with the app.",
    }),
  );

  const draw = (): void => {
    const p = palette();
    picker.value = String(selected);
    for (const { value, button } of modeButtons) {
      button.setAttribute("aria-pressed", String(mode === value));
    }
    const width = layout.clientWidth > 860 ? layout.clientWidth - 320 : layout.clientWidth;
    const plot = heatmap(wiring, data, mode, selected, p, width || 640);
    plot.addEventListener("click", () => {
      const cell = plot.value as Cell | null;
      if (cell) {
        selected = cell.si;
        draw();
        if (layout.clientWidth <= 860)
          side.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });
    view.chart.replaceChildren(plot);
    side.replaceChildren(familyCard(wiring, selected));
  };

  picker.addEventListener("change", () => {
    selected = Number(picker.value);
    draw();
  });
  const stopScheme = onScheme(draw);
  const stopResize = onWidthChange(root, draw);
  scope.onLeave(() => {
    stopScheme();
    stopResize();
  });
}
