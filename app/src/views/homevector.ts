/** City view card: the pre-registered home-vector result (the network sums the vector itself). */
import * as Plot from "@observablehq/plot";
import type { HomeVector } from "../data";
import { onWidthChange, type Scope } from "../lifecycle";
import { onScheme, palette, type Palette, type Series } from "../theme";
import { card, el, table } from "../ui";

type Variant = "FlyConnectome" | "TypeShuffledFly" | "RandomMask" | "ShuffledFly" | "FrozenFly";

const NAME: Record<Variant, string> = {
  FlyConnectome: "real CX wiring",
  TypeShuffledFly: "shuffled within cell types",
  RandomMask: "random wiring (same size)",
  ShuffledFly: "degree-shuffled wiring",
  FrozenFly: "real wiring, frozen (not trained)",
};
/** Colour and marker of each control, as in the Results view (random mask ≙ random wiring). */
const SERIES: Record<Exclude<Variant, "FlyConnectome">, Series> = {
  TypeShuffledFly: "TypeShuffledFly",
  RandomMask: "RandomRecurrent",
  ShuffledFly: "ShuffledFly",
  FrozenFly: "FrozenFly",
};
const SYMBOL: Record<Exclude<Variant, "FlyConnectome">, string> = {
  TypeShuffledFly: "diamond",
  RandomMask: "triangle",
  ShuffledFly: "square",
  FrozenFly: "times",
};
const CONTROLS: Exclude<Variant, "FlyConnectome">[] = [
  "TypeShuffledFly",
  "RandomMask",
  "ShuffledFly",
];

const metres = (v: number): string => `${v.toFixed(1)} m`;
const signed = (v: number): string => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;

/** The verdict of the pre-registered primary test (real − random), read from its interval. */
export function verdict(hv: HomeVector): string {
  const d = hv.fly_minus_error_m.RandomMask;
  const ci = `${signed(d.low)} to ${signed(d.high)} m`;
  if (d.high < 0) {
    return (
      `The real wiring lands ${Math.abs(d.mean).toFixed(1)} m closer to the nest than a random ` +
      `wiring of the same size, and the 95% confidence interval (${ci}) lies below zero: ` +
      "a clear advantage."
    );
  }
  if (d.low > 0) {
    return (
      `The real wiring lands ${d.mean.toFixed(1)} m further from the nest, and the 95% ` +
      `confidence interval (${ci}) lies above zero: the random wiring does better.`
    );
  }
  return (
    `The difference is ${signed(d.mean)} m, but the 95% confidence interval (${ci}) includes ` +
    "zero: no measurable advantage."
  );
}

interface Point {
  variant: Exclude<Variant, "FlyConnectome">;
  index: number;
  mean: number;
  low: number;
  high: number;
}

function chart(points: Point[], p: Palette, width: number): Element {
  const narrow = width < 560;
  return Plot.plot({
    style: { background: "transparent", color: p.ink2, fontSize: "12px", overflow: "visible" },
    width,
    height: points.length * (narrow ? 64 : 48) + 70,
    marginLeft: narrow ? 8 : 210,
    marginRight: 16,
    marginBottom: 46,
    x: {
      label: narrow
        ? "← real wiring closer · difference (m)"
        : "← real wiring closer to the nest · real minus control (m)",
      labelAnchor: "center",
      labelOffset: 38,
      inset: 8,
    },
    y: {
      domain: narrow ? [points.length - 0.5, -0.9] : [points.length - 0.5, -0.5],
      ticks: narrow ? [] : points.map((d) => d.index),
      tickFormat: (i: number) => NAME[points[i]?.variant ?? "RandomMask"],
      tickSize: 0,
      label: null,
      axis: narrow ? null : "left",
    },
    marks: [
      Plot.gridX({ stroke: p.grid, strokeOpacity: 1 }),
      Plot.ruleX([0], { stroke: p.ink2 }),
      Plot.link(points, {
        x1: "low",
        x2: "high",
        y1: "index",
        y2: "index",
        stroke: (d: Point) => p.series[SERIES[d.variant]],
        strokeWidth: 2,
        strokeLinecap: "round",
      }),
      Plot.dot(points, {
        x: "mean",
        y: "index",
        symbol: (d: Point) => SYMBOL[d.variant],
        fill: (d: Point) => p.series[SERIES[d.variant]],
        stroke: p.surface,
        strokeWidth: 1.5,
        r: 5,
      }),
      ...(narrow
        ? [
            Plot.text(points, {
              frameAnchor: "left",
              y: (d: Point) => d.index - 0.42,
              text: (d: Point) => `vs ${NAME[d.variant]}`,
              textAnchor: "start",
              fill: p.ink,
              fontWeight: 600,
              stroke: p.raised,
              strokeWidth: 4,
            }),
          ]
        : []),
      Plot.tip(
        points,
        Plot.pointer({
          x: "mean",
          y: "index",
          maxRadius: 28,
          title: (d: Point) =>
            `${signed(d.mean)} m vs ${NAME[d.variant]}\n95% CI ${signed(d.low)} to ${signed(d.high)} m`,
          fill: p.raised,
          stroke: p.grid,
          textPadding: 8,
          fontSize: 12,
        }),
      ),
    ],
  });
}

/** Build the card, append it to `parent` and keep it in sync with width and colour scheme. */
export function homeVectorCard(parent: HTMLElement, hv: HomeVector, scope: Scope): void {
  const seeds = `${hv.seeds[0]}–${hv.seeds[hv.seeds.length - 1]}`;
  const view = card(
    "When the network computes the home vector itself",
    "Above, the network only reports its heading and code adds up the steps. Here a network of " +
      "the same size is trained to report, at every step, where the nest is — so its wiring has to " +
      "do the adding up, as the fan-shaped body is thought to do in insects. Tested on the same " +
      `Warsaw flights, pre-registered on fresh seeds ${seeds} (${hv.seeds.length} networks of ` +
      "each kind, 100 flights each).",
  );
  const lede = el("p", { className: "lede", text: verdict(hv) });
  view.root.querySelector("h2")?.after(lede);
  const points: Point[] = CONTROLS.map((variant, index) => ({
    variant,
    index,
    ...hv.fly_minus_error_m[variant],
  }));
  const variants: Variant[] = [
    "FlyConnectome",
    "TypeShuffledFly",
    "RandomMask",
    "ShuffledFly",
    "FrozenFly",
  ];
  view.tableSlot.append(
    table(
      variants,
      [
        { label: "Network", value: (v) => NAME[v] },
        {
          label: "Lands from the nest",
          value: (v) => metres(hv.mean_error_m[v].mean),
          numeric: true,
        },
        {
          label: "95% CI",
          value: (v) =>
            `${hv.mean_error_m[v].low.toFixed(1)}–${hv.mean_error_m[v].high.toFixed(1)}`,
          numeric: true,
        },
        {
          label: "Real minus this (m)",
          value: (v) => {
            if (v === "FlyConnectome") return "—";
            const d = hv.fly_minus_error_m[v];
            return `${signed(d.mean)} [${signed(d.low)}, ${signed(d.high)}]`;
          },
          numeric: true,
        },
      ],
      "Mean distance from the nest after flying back along the network's own home vector " +
        `(seeds ${seeds}, 95% confidence intervals over seeds)`,
    ),
  );
  view.root.append(
    el(
      "p",
      { className: "note" },
      `Average distance from the nest: real wiring ${metres(hv.mean_error_m.FlyConnectome.mean)}, ` +
        `random wiring ${metres(hv.mean_error_m.RandomMask.mean)}, frozen (untrained) ` +
        `${metres(hv.mean_error_m.FrozenFly.mean)}, a fly that stays put ` +
        `${metres(hv.stay_error_m.mean)}. A rewiring that keeps which cell types connect does ` +
        `almost as well as the real wiring (${signed(hv.fly_minus_error_m.TypeShuffledFly.mean)} m, ` +
        "interval includes zero): as on the heading task, cell-type structure carries most of the " +
        "advantage. Caveats: the network's own vector is still less accurate than code adding up " +
        "its heading (above), because these flights run up to three times longer than the training " +
        `flights (on training-length flights it misses by ${metres(hv.mean_synthetic_error_m.FlyConnectome.mean)}); ` +
        "and a pilot criterion of this test was not met. Exploratory.",
    ),
    el(
      "p",
      { className: "note" },
      el("a", {
        text: "Details and pre-registration",
      }),
    ),
  );
  const link = view.root.querySelector<HTMLAnchorElement>("p.note:last-child a");
  if (link)
    link.href = "https://github.com/kat1478/NeuroFlyNav/blob/develop/docs/spike_homevector.md";
  parent.append(view.root);

  const draw = (): void => {
    const p = palette();
    view.chart.replaceChildren(chart(points, p, view.chart.clientWidth || 800));
  };
  draw();
  const stopScheme = onScheme(draw);
  const stopResize = onWidthChange(view.root, draw);
  scope.onLeave(() => {
    stopScheme();
    stopResize();
  });
}
