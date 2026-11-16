import { loadReplay, type Replay } from "../data";
import { onLeave } from "../lifecycle";
import { polar, runningScore, sectorPath, stepsSinceLandmark } from "../replay-geometry";
import { onScheme, palette, SERIES_LABEL, type Palette } from "../theme";
import { el, pct } from "../ui";

const SVG = "http://www.w3.org/2000/svg";
type VariantKey = "FlyConnectome" | "RandomMask";
const VARIANTS: { key: VariantKey; series: "FlyConnectome" | "RandomRecurrent" }[] = [
  { key: "FlyConnectome", series: "FlyConnectome" },
  { key: "RandomMask", series: "RandomRecurrent" },
];

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | number> = {},
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  return node;
}

/** One compass ring: 8 sectors, the predicted one filled, the true heading as a needle. */
function ring(
  replay: Replay,
  p: Palette,
  color: string,
  predicted: number,
  theta: number,
  landmark: boolean,
  correct: boolean | null,
): SVGSVGElement {
  const size = 220;
  const c = size / 2;
  const outer = 96;
  const inner = 62;
  const root = svg("svg", { viewBox: `0 0 ${size} ${size}`, class: "ring", role: "img" });
  root.setAttribute(
    "aria-label",
    `Predicted sector ${predicted + 1} of ${replay.sectors}; ` +
      (correct === null ? "landmark visible" : correct ? "correct" : "wrong"),
  );
  if (landmark) {
    root.append(
      svg("circle", {
        cx: c,
        cy: c,
        r: outer + 8,
        fill: "none",
        stroke: p.ink2,
        "stroke-width": 2,
        "stroke-dasharray": "3 5",
      }),
    );
  }
  for (let k = 0; k < replay.sectors; k += 1) {
    const isPredicted = k === predicted;
    root.append(
      svg("path", {
        d: sectorPath(c, c, inner, outer, k, replay.sectors),
        fill: isPredicted ? color : p.surface,
        stroke: p.grid,
        "stroke-width": 1.5,
      }),
    );
  }
  const tip = polar(c, c, outer + 2, theta);
  const tail = polar(c, c, 18, theta + Math.PI);
  root.append(
    svg("line", {
      x1: tail.x,
      y1: tail.y,
      x2: tip.x,
      y2: tip.y,
      stroke: p.ink,
      "stroke-width": 3,
      "stroke-linecap": "round",
    }),
    svg("circle", { cx: c, cy: c, r: 5, fill: p.ink }),
  );
  const mark = svg("text", {
    x: c,
    y: c + 34,
    "text-anchor": "middle",
    "font-size": 15,
    "font-weight": 700,
    fill: p.ink2,
  });
  mark.textContent = correct === null ? "landmark" : correct ? "✓ right" : "✕ wrong";
  root.append(mark);
  return root;
}

/** Timeline: one column per step; landmark row, then ✓ / ✕ per network. */
function timeline(
  replay: Replay,
  sequence: number,
  step: number,
  p: Palette,
  onSeek: (step: number) => void,
): SVGSVGElement {
  const left = 92;
  const cell = 16;
  const width = left + replay.steps * cell + 8;
  const rowY = [14, 40, 62];
  const root = svg("svg", { viewBox: `0 0 ${width} 80`, class: "timeline" });
  root.setAttribute("aria-label", "Timeline: landmarks and right or wrong answers per step");
  const label = (y: number, text: string): void => {
    const node = svg("text", {
      x: left - 8,
      y: y + 4,
      "text-anchor": "end",
      "font-size": 11,
      fill: p.ink2,
    });
    node.textContent = text;
    root.append(node);
  };
  label(rowY[0], "landmark");
  label(rowY[1], "real CX");
  label(rowY[2], "random");
  const truth = replay.sequences[sequence].sector;
  for (let t = 0; t < replay.steps; t += 1) {
    const x = left + t * cell + cell / 2;
    if (t === step) {
      root.append(
        svg("rect", { x: x - cell / 2, y: 0, width: cell, height: 80, fill: p.grid, rx: 3 }),
      );
    }
    if (replay.landmark[t]) root.append(svg("circle", { cx: x, cy: rowY[0], r: 4, fill: p.ink2 }));
    VARIANTS.forEach(({ key, series }, row) => {
      const y = rowY[row + 1];
      if (!replay.scored[t]) {
        root.append(
          svg("line", {
            x1: x - 3,
            y1: y,
            x2: x + 3,
            y2: y,
            stroke: p.baseline,
            "stroke-width": 1.5,
          }),
        );
        return;
      }
      const right = replay.variants[key].predicted[sequence][t] === truth[t];
      if (right) {
        root.append(svg("circle", { cx: x, cy: y, r: 4.5, fill: p.series[series] }));
      } else {
        root.append(
          svg("path", {
            d: `M${x - 4},${y - 4} L${x + 4},${y + 4} M${x + 4},${y - 4} L${x - 4},${y + 4}`,
            stroke: p.ink2,
            "stroke-width": 1.8,
            "stroke-linecap": "round",
          }),
        );
      }
    });
    const hit = svg("rect", {
      x: x - cell / 2,
      y: 0,
      width: cell,
      height: 80,
      fill: "transparent",
    });
    hit.style.cursor = "pointer";
    hit.addEventListener("click", () => onSeek(t));
    root.append(hit);
  }
  return root;
}

export async function renderReplay(root: HTMLElement): Promise<void> {
  const replay = await loadReplay();
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let sequence = 0;
  let step = 0;
  let playing = !reduceMotion;
  let speed = 1;
  let timer = 0;

  const fly = replay.variants.FlyConnectome.accuracy;
  const random = replay.variants.RandomMask.accuracy;
  root.append(
    el("h1", { text: "Watch a network keep track of heading" }),
    el("p", {
      className: "lede",
      text:
        "The task: a simulated fly turns at random. Every 8 steps it sees a landmark that " +
        "tells it where it is facing; in between it only feels how fast it turns, and the " +
        "network must integrate that to report its heading — one of 8 sectors. The black needle " +
        "is the true heading; the filled sector is the network's answer.",
    }),
  );

  const sequenceButtons = el("div", { className: "segmented" });
  sequenceButtons.setAttribute("role", "group");
  sequenceButtons.setAttribute("aria-label", "Test sequence");
  const play = el("button", { className: "play" });
  play.type = "button";
  const scrub = el("input");
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = String(replay.steps - 1);
  scrub.setAttribute("aria-label", "Step");
  const speedSelect = el("select");
  speedSelect.setAttribute("aria-label", "Speed");
  for (const [value, text] of [
    ["0.5", "0.5×"],
    ["1", "1×"],
    ["2", "2×"],
  ]) {
    const option = el("option", { text });
    option.value = value;
    speedSelect.append(option);
  }
  speedSelect.value = "1";
  const stepLabel = el("span", { className: "step-label" });
  const rings = el("div", { className: "rings" });
  const timelineSlot = el("div", { className: "timeline-slot" });

  const section = el(
    "section",
    { className: "card" },
    el("div", { className: "controls" }, el("span", { text: "Sequence" }), sequenceButtons),
    rings,
    el("div", { className: "player" }, play, scrub, stepLabel, speedSelect),
    timelineSlot,
    el("p", {
      className: "note",
      text:
        `Both networks were trained on the whole CX graph (${replay.nodes.toLocaleString("en")} ` +
        `neurons) with the same budget; one uses the real wiring, the other a random wiring with ` +
        `the same number of connections. On this seed (${replay.seed}) they score about the same over ` +
        `64 test sequences — real ${pct(fly)}, random ${pct(random)}. The real wiring's edge ` +
        "(+2.8 points) only shows when averaged over 20 seeds: see Results. These 6 sequences " +
        "are the first 6 test sequences, not hand-picked.",
    }),
  );
  root.append(section);

  const draw = (): void => {
    const p = palette();
    const data = replay.sequences[sequence];
    const landmark = replay.landmark[step];
    sequenceButtons.replaceChildren(
      ...replay.sequences.map((_, i) => {
        const button = el("button", { text: String(i + 1) });
        button.type = "button";
        button.setAttribute("aria-pressed", String(i === sequence));
        button.addEventListener("click", () => {
          sequence = i;
          step = 0;
          draw();
        });
        return button;
      }),
    );
    rings.replaceChildren(
      ...VARIANTS.map(({ key, series }) => {
        const predicted = replay.variants[key].predicted[sequence];
        const score = runningScore(predicted, data.sector, replay.scored, step);
        const correct = replay.scored[step] ? predicted[step] === data.sector[step] : null;
        return el(
          "figure",
          { className: "ring-figure" },
          ring(replay, p, p.series[series], predicted[step], data.theta[step], landmark, correct),
          el(
            "figcaption",
            {},
            el("strong", { text: SERIES_LABEL[series] }),
            el("span", {
              text: score.total
                ? `${score.hits} / ${score.total} right so far (${pct(score.hits / score.total, 0)})`
                : "no scored step yet",
            }),
          ),
        );
      }),
    );
    const since = stepsSinceLandmark(replay.landmark, step);
    stepLabel.textContent =
      `Step ${step + 1} / ${replay.steps} · ` +
      (landmark ? "landmark visible" : `${since} step${since === 1 ? "" : "s"} since landmark`);
    scrub.value = String(step);
    play.textContent = playing ? "❚❚ Pause" : "▶ Play";
    play.setAttribute("aria-pressed", String(playing));
    timelineSlot.replaceChildren(
      timeline(replay, sequence, step, p, (t) => {
        step = t;
        playing = false;
        schedule();
        draw();
      }),
    );
  };

  const schedule = (): void => {
    window.clearTimeout(timer);
    if (!playing) return;
    timer = window.setTimeout(() => {
      step = (step + 1) % replay.steps;
      draw();
      schedule();
    }, 450 / speed);
  };

  play.addEventListener("click", () => {
    playing = !playing;
    draw();
    schedule();
  });
  scrub.addEventListener("input", () => {
    step = Number(scrub.value);
    playing = false;
    schedule();
    draw();
  });
  speedSelect.addEventListener("change", () => {
    speed = Number(speedSelect.value);
    schedule();
  });
  const onKey = (event: KeyboardEvent): void => {
    if (event.key !== " " || (event.target as HTMLElement).closest("input, select, button")) return;
    event.preventDefault();
    playing = !playing;
    draw();
    schedule();
  };
  document.addEventListener("keydown", onKey);

  const stopScheme = onScheme(draw);
  schedule();
  onLeave(() => {
    window.clearTimeout(timer);
    document.removeEventListener("keydown", onKey);
    stopScheme();
  });
}
