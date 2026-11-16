import * as Plot from "@observablehq/plot";
import { loadWiring, type Wiring } from "../data";
import { cellValue, FAMILY_INFO, topPartners, type Mode, type Partner } from "../families";
import { onLeave, onWidthChange } from "../lifecycle";
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
    marginLeft: 44,
    marginTop: size < 560 ? 64 : 44,
    marginBottom: 8,
    marginRight: 8,
    padding: 0.06,
    style: { background: "transparent", color: p.ink2, fontSize: "11px" },
    x: {
      domain: wiring.labels,
      axis: "top",
      label: null,
      tickSize: 0,
      tickRotate: size < 560 ? -90 : 0,
    },
    y: { domain: wiring.labels, label: null, tickSize: 0 },
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
            `${fmt(d.synapses)} synapses\n${d.source} → ${d.target}\n` +
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
        const bar = el("span", { className: "partner-bar" });
        bar.style.width = `${Math.max(4, (100 * partner.share) / top)}%`;
        return el(
          "li",
          {},
          el("span", { text: partner.family }),
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
    ...partnerList("Strongest targets", topPartners(wiring.labels, wiring.matrix, index, "out")),
    ...partnerList("Strongest sources", topPartners(wiring.labels, wiring.matrix, index, "in")),
  );
}

export async function renderWiring(root: HTMLElement): Promise<void> {
  const wiring = await loadWiring();
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
    "Rows send, columns receive: each cell counts synapses from the row family to the " +
      "column family; darker = more. " +
      "Hover a cell for numbers; click it (or pick a family below) to read about the row family.",
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
  const controls = el(
    "div",
    { className: "controls" },
    el("label", {}, "Family ", picker),
    el("span", { text: "Colour by" }),
    modeSwitch,
  );
  const side = el("div");
  const layout = el("div", { className: "wiring-layout" }, view.chart, side);
  view.legend.replaceWith(controls, layout); // the chart now lives inside the layout
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
    modeSwitch.replaceChildren(
      ...(
        [
          ["synapses", "synapse count"],
          ["share", "share of output"],
        ] as [Mode, string][]
      ).map(([value, text]) => {
        const button = el("button", { text });
        button.type = "button";
        button.setAttribute("aria-pressed", String(mode === value));
        button.addEventListener("click", () => {
          mode = value;
          draw();
        });
        return button;
      }),
    );
    const width = layout.clientWidth > 860 ? layout.clientWidth - 320 : layout.clientWidth;
    const plot = heatmap(wiring, data, mode, selected, p, width || 640);
    plot.addEventListener("click", () => {
      const cell = plot.value as Cell | null;
      if (cell) {
        selected = cell.si;
        draw();
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
  onLeave(() => {
    stopScheme();
    stopResize();
  });
}
