import * as Plot from "@observablehq/plot";
import { loadResults, type Control, type PerSeedExperiment, type Results } from "../data";
import { onWidthChange, type Scope } from "../lifecycle";
import {
  onScheme,
  palette,
  SERIES_LABEL,
  SERIES_SYMBOL,
  type Palette,
  type Series,
} from "../theme";
import { card, el, GLYPH, legendItem, pct, points, table } from "../ui";

const CONTROLS: Control[] = ["ShuffledFly", "TypeShuffledFly", "RandomRecurrent"];
const CONTROL_NAME: Record<Control, string> = {
  ShuffledFly: "vs degree-shuffled",
  TypeShuffledFly: "vs shuffled within cell types",
  RandomRecurrent: "vs random (same size)",
};

function axisStyle(p: Palette): Partial<Plot.PlotOptions> {
  return {
    style: { background: "transparent", color: p.ink2, fontSize: "12px", overflow: "visible" },
  };
}

function tipStyle(p: Palette): Plot.TipOptions {
  return { fill: p.raised, stroke: p.grid, textPadding: 8, fontSize: 12 };
}

function legend(slot: HTMLElement, p: Palette, series: Series[]): void {
  slot.replaceChildren(
    ...series.map((s) => legendItem(SERIES_LABEL[s], p.series[s], GLYPH[SERIES_SYMBOL[s]])),
  );
}

/* ---------- headline ---------- */

function headline(results: Results): HTMLElement {
  const row = results.forest[results.forest.length - 1];
  const [low, mean, high] = row.diff.RandomRecurrent;
  return el(
    "section",
    { className: "hero" },
    el(
      "div",
      { className: "hero-number" },
      el("span", { className: "hero-value", text: points(mean) }),
      el("span", { className: "hero-unit", text: "percentage points of heading accuracy" }),
      el("span", {
        className: "hero-ci",
        text: `95% confidence interval ${points(low)} to ${points(high)}`,
      }),
    ),
    el(
      "p",
      { className: "hero-text" },
      "Recurrent weights trained on the ",
      el("strong", { text: "real wiring of the whole fly central complex" }),
      " (2,966 neurons) keep track of heading better than the same network trained on a random " +
        "wiring with the same number of connections. The test was fixed in writing before it ran " +
        "(pre-registered) and repeated with 20 independent random starts (seeds), as 100 jobs in " +
        "the cloud. On the small, dense compass core there was no such edge.",
    ),
  );
}

/* ---------- forest plot ---------- */

/** Row label without the seed range (kept in tooltips and tables). */
const shortLabel = (label: string): string => label.replace(/ \(seeds [^)]*\)/, "");

interface ForestPoint {
  row: string;
  position: number;
  control: Control;
  low: number;
  mean: number;
  high: number;
}

const OFFSET: Record<Control, number> = {
  ShuffledFly: -0.22,
  TypeShuffledFly: 0,
  RandomRecurrent: 0.22,
};

function forestData(results: Results): ForestPoint[] {
  return results.forest.flatMap((row, index) =>
    CONTROLS.map((control) => {
      const [low, mean, high] = row.diff[control];
      return {
        row: row.label,
        position: index + OFFSET[control],
        control,
        low: 100 * low,
        mean: 100 * mean,
        high: 100 * high,
      };
    }),
  );
}

function forestChart(data: ForestPoint[], rows: string[], p: Palette, width: number): Element {
  const narrow = width < 560;
  const headline = rows.length - 1;
  const label = (i: number): string =>
    (i === headline ? "★ " : "") +
    (shortLabel(rows[i] ?? "").replace("\n", narrow ? " · " : "\n") || "");
  return Plot.plot({
    ...axisStyle(p),
    width,
    height: rows.length * (narrow ? 96 : 78) + 76,
    marginLeft: narrow ? 8 : 250,
    marginRight: 16,
    marginBottom: 48,
    x: {
      label: "Real wiring minus control (percentage points) →",
      labelAnchor: "center",
      labelOffset: 40,
      inset: 8,
    },
    y: {
      domain: [rows.length - 0.5, narrow ? -0.85 : -0.5],
      ticks: narrow ? [] : rows.map((_, i) => i),
      tickFormat: label,
      tickSize: 0,
      label: null,
      axis: narrow ? null : "left",
    },
    marks: [
      Plot.rect([headline], {
        y1: (i: number) => i - 0.46,
        y2: (i: number) => i + 0.46,
        fill: p.grid,
        fillOpacity: 0.55,
      }),
      Plot.gridX({ stroke: p.grid, strokeOpacity: 1 }),
      Plot.ruleX([0], { stroke: p.ink2 }),
      Plot.link(data, {
        x1: "low",
        x2: "high",
        y1: "position",
        y2: "position",
        stroke: (d: ForestPoint) => p.series[d.control],
        strokeWidth: 2,
        strokeLinecap: "round",
      }),
      Plot.dot(data, {
        x: "mean",
        y: "position",
        symbol: (d: ForestPoint) => SERIES_SYMBOL[d.control],
        fill: (d: ForestPoint) => p.series[d.control],
        stroke: p.surface,
        strokeWidth: 1.5,
        r: 5,
      }),
      ...(narrow
        ? [
            Plot.text(
              rows.map((_, i) => i),
              {
                frameAnchor: "left",
                y: (i: number) => i - 0.42,
                text: label,
                textAnchor: "start",
                fill: p.ink,
                fontWeight: 600,
                stroke: p.raised,
                strokeWidth: 4,
              },
            ),
          ]
        : []),
      Plot.tip(
        data,
        Plot.pointer({
          x: "mean",
          y: "position",
          maxRadius: 28,
          title: (d: ForestPoint) =>
            `${points(d.mean / 100)} pts ${CONTROL_NAME[d.control]}\n` +
            `95% CI ${points(d.low / 100)} to ${points(d.high / 100)}\n` +
            d.row.replace("\n", " — "),
          ...tipStyle(p),
        }),
      ),
    ],
  });
}

/* ---------- per-seed differences ---------- */

interface SeedPoint {
  seed: number;
  control: Control;
  diff: number;
  fly: number;
  other: number;
}

function seedData(experiment: PerSeedExperiment): SeedPoint[] {
  return experiment.accuracy.flatMap((row, i) =>
    CONTROLS.filter((c) => c in row).map((control) => ({
      seed: experiment.seeds[i],
      control,
      diff: 100 * (row.FlyConnectome - row[control]),
      fly: row.FlyConnectome,
      other: row[control],
    })),
  );
}

function seedChart(data: SeedPoint[], p: Palette, width: number): Element {
  const narrow = width < 560;
  const means = CONTROLS.map((control) => {
    const values = data.filter((d) => d.control === control).map((d) => d.diff);
    return { control, mean: values.reduce((a, b) => a + b, 0) / values.length };
  });
  const dodge = (options: Plot.DotOptions): Plot.DotOptions =>
    Plot.dodgeY("middle", { x: "diff", fy: "control", r: 4.5, padding: 1, ...options });
  return Plot.plot({
    ...axisStyle(p),
    width,
    height: narrow ? 380 : 330,
    marginLeft: narrow ? 8 : 200,
    marginRight: 16,
    marginTop: narrow ? 24 : 8,
    marginBottom: 48,
    x: {
      label: "Real wiring minus control, per seed (percentage points) →",
      labelAnchor: "center",
      labelOffset: 40,
      inset: 14,
    },
    y: { axis: null },
    fy: {
      domain: CONTROLS,
      tickFormat: (c: Control) => CONTROL_NAME[c],
      label: null,
      axis: narrow ? null : "left",
      tickSize: 0,
      padding: 0.15,
    },
    marks: [
      Plot.gridX({ stroke: p.grid, strokeOpacity: 1 }),
      Plot.ruleX([0], { stroke: p.ink2 }),
      Plot.dot(
        data,
        dodge({
          symbol: (d: SeedPoint) => SERIES_SYMBOL[d.control],
          fill: (d: SeedPoint) => p.series[d.control],
          stroke: p.surface,
          strokeWidth: 1,
        }),
      ),
      Plot.tickX(means, { x: "mean", fy: "control", stroke: p.ink, strokeWidth: 2.5, inset: 6 }),
      ...(narrow
        ? [
            Plot.text(CONTROLS, {
              fy: (c: Control) => c,
              frameAnchor: "top-left",
              text: (c: Control) => CONTROL_NAME[c],
              fill: p.ink,
              fontWeight: 600,
              stroke: p.raised,
              strokeWidth: 4,
              dy: -14,
            }),
          ]
        : []),
      Plot.tip(
        data,
        Plot.pointer(
          dodge({
            maxRadius: 24,
            title: (d: SeedPoint) =>
              `${points(d.diff / 100)} pts — seed ${d.seed}\n` +
              `real ${pct(d.fly)} · control ${pct(d.other)}\n${CONTROL_NAME[d.control]}`,
            ...tipStyle(p),
          } as Plot.DotOptions),
        ) as Plot.TipOptions,
      ),
    ],
  });
}

/* ---------- learning curve ---------- */

const CURVE_SERIES: Series[] = [
  "FlyConnectome",
  "RandomRecurrent",
  "ShuffledFly",
  "TypeShuffledFly",
  "FrozenFly",
];

/** The real wiring is drawn last (on top): its line nearly coincides with the shuffled one. */
const DRAW_ORDER: Series[] = [
  ...CURVE_SERIES.filter((s) => s !== "FlyConnectome"),
  "FlyConnectome",
];

interface CurvePoint {
  series: Series;
  size: number;
  accuracy: number;
}

function curveData(results: Results): CurvePoint[] {
  const { sizes } = results.learning;
  return CURVE_SERIES.flatMap((series) =>
    (results.learning[series] ?? []).map((accuracy, i) => ({ series, size: sizes[i], accuracy })),
  );
}

function curveChart(data: CurvePoint[], p: Palette, width: number): Element {
  const sizes = [...new Set(data.map((d) => d.size))];
  const readout = sizes.map((size) => ({
    size,
    text: CURVE_SERIES.map((s) => {
      const point = data.find((d) => d.series === s && d.size === size);
      return point ? `${pct(point.accuracy)}  ${SERIES_LABEL[s]}` : "";
    })
      .filter(Boolean)
      .join("\n"),
  }));
  return Plot.plot({
    ...axisStyle(p),
    width,
    height: 360,
    marginLeft: 48,
    marginRight: 16,
    marginBottom: 48,
    x: {
      type: "log",
      ticks: sizes,
      tickFormat: (d: number) => String(d),
      label: "Training sequences (log scale) →",
      labelAnchor: "center",
      labelOffset: 40,
    },
    y: {
      domain: [0.3, 0.9],
      tickFormat: (d: number) => pct(d, 0),
      label: "↑ Accuracy",
      grid: true,
    },
    marks: [
      Plot.gridY({ stroke: p.grid, strokeOpacity: 1 }),
      ...DRAW_ORDER.map((s) =>
        Plot.line(
          data.filter((d) => d.series === s),
          {
            x: "size",
            y: "accuracy",
            stroke: p.series[s],
            strokeWidth: s === "FlyConnectome" ? 3 : 2,
            strokeDasharray: s === "FrozenFly" ? "4 4" : undefined,
          },
        ),
      ),
      ...DRAW_ORDER.map((s) =>
        Plot.dot(
          data.filter((d) => d.series === s),
          {
            x: "size",
            y: "accuracy",
            symbol: SERIES_SYMBOL[s],
            fill: p.series[s],
            stroke: p.surface,
            strokeWidth: 1.5,
            r: 4.5,
          },
        ),
      ),
      Plot.ruleX(readout, Plot.pointerX({ x: "size", stroke: p.baseline })),
      Plot.tip(
        readout,
        Plot.pointerX({
          x: "size",
          y: 0.9,
          title: (d: { size: number; text: string }) => `${d.size} training sequences\n${d.text}`,
          ...tipStyle(p),
        }),
      ),
    ],
  });
}

/* ---------- view ---------- */

export async function renderResults(root: HTMLElement, scope: Scope): Promise<void> {
  const results = await loadResults();
  if (!scope.active) return;
  root.append(
    el("h1", { text: "Does real fly wiring make a better network?" }),
    el("p", {
      className: "lede",
      text:
        "Each experiment compares an agent built on the real central-complex wiring with controls " +
        "that keep the size but scramble the wiring. Every comparison below was fixed in writing " +
        "before it ran, on seeds never used before. Exploratory research: one model family, " +
        "small tasks — read the gaps as leads, not conclusions.",
    }),
    headline(results),
  );

  const rows = results.forest.map((r) => r.label);
  const forest = card(
    "All pre-registered comparisons",
    "Each row is one experiment; each mark compares the real wiring with one control. Points: " +
      "average difference over 20 seeds; bars: 95% confidence interval. Right of zero = real " +
      "wiring better. ★ marks the headline result above. The hidden-cue grid task asks the agent " +
      "to reach a goal it saw only briefly; the heading task asks it to keep track of where it " +
      "faces. Hover or tap a mark for numbers and seeds.",
  );
  const forestPoints = forestData(results);
  forest.tableSlot.append(
    table(
      forestPoints,
      [
        { label: "Experiment", value: (d) => d.row.replace("\n", " — ") },
        { label: "Control", value: (d) => CONTROL_NAME[d.control] },
        { label: "Difference (pts)", value: (d) => points(d.mean / 100), numeric: true },
        {
          label: "95% CI",
          value: (d) => `${points(d.low / 100)} to ${points(d.high / 100)}`,
          numeric: true,
        },
      ],
      "Real wiring minus control, percentage points",
    ),
  );

  const seeds = card(
    "Seed by seed",
    "Each mark is one seed: how much better (right) or worse (left) the real wiring did than " +
      "that control on the same seed. The thick vertical tick is the average.",
  );
  const picker = el("div", { className: "segmented", text: "" });
  picker.setAttribute("role", "group");
  picker.setAttribute("aria-label", "Experiment");
  seeds.root.insertBefore(picker, seeds.legend);
  let selected = results.per_seed[results.per_seed.length - 1].id;
  const pickerButtons = results.per_seed.map((e) => {
    const button = el("button", { text: e.label.replace("Heading task, ", "") });
    button.type = "button";
    button.addEventListener("click", () => {
      selected = e.id;
      drawSeeds(palette());
    });
    picker.append(button);
    return { id: e.id, button };
  });

  const curve = card(
    "How much data does each wiring need?",
    "Trained compass core (409 neurons): accuracy against the number of training sequences. " +
      "The dashed line is the real wiring with its connections frozen (not trained). With little " +
      "data the random wiring is slightly ahead; with 256 sequences all trained networks converge.",
  );
  const curvePoints = curveData(results);
  curve.tableSlot.append(
    table(
      results.learning.sizes,
      [
        { label: "Sequences", value: (n) => String(n), numeric: true },
        ...CURVE_SERIES.map((s) => ({
          label: SERIES_LABEL[s],
          value: (n: number) => {
            const point = curvePoints.find((d) => d.series === s && d.size === n);
            return point ? pct(point.accuracy) : "—";
          },
          numeric: true,
        })),
      ],
      "Accuracy by training-set size",
    ),
  );

  root.append(forest.root, seeds.root, curve.root);

  const drawSeeds = (p: Palette): void => {
    const experiment = results.per_seed.find((e) => e.id === selected) ?? results.per_seed[0];
    for (const { id, button } of pickerButtons) {
      button.setAttribute("aria-pressed", String(id === selected));
    }
    const data = seedData(experiment);
    seeds.chart.replaceChildren(seedChart(data, p, seeds.chart.clientWidth || 800));
    seeds.tableSlot.replaceChildren(
      table(
        data,
        [
          { label: "Seed", value: (d) => String(d.seed), numeric: true },
          { label: "Compared with", value: (d) => CONTROL_NAME[d.control] },
          { label: "Real wiring accuracy", value: (d) => pct(d.fly), numeric: true },
          { label: "Control accuracy", value: (d) => pct(d.other), numeric: true },
          { label: "Difference (pts)", value: (d) => points(d.diff / 100), numeric: true },
        ],
        experiment.label,
      ),
    );
  };

  const draw = (): void => {
    const p = palette();
    legend(forest.legend, p, CONTROLS);
    legend(seeds.legend, p, CONTROLS);
    legend(curve.legend, p, CURVE_SERIES);
    forest.chart.replaceChildren(
      forestChart(forestPoints, rows, p, forest.chart.clientWidth || 800),
    );
    drawSeeds(p);
    curve.chart.replaceChildren(curveChart(curvePoints, p, curve.chart.clientWidth || 800));
  };

  const stopScheme = onScheme(draw);
  const stopResize = onWidthChange(root, draw);
  scope.onLeave(() => {
    stopScheme();
    stopResize();
  });
}
