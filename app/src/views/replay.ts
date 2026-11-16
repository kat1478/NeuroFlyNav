import { loadReplay, type Replay } from "../data";
import type { Scope } from "../lifecycle";
import { polar, runningScore, sectorPath, stepsSinceLandmark } from "../replay-geometry";
import { onScheme, palette, SERIES_LABEL, type Palette } from "../theme";
import { el, pct } from "../ui";

const SVG = "http://www.w3.org/2000/svg";
type VariantKey = "FlyConnectome" | "RandomMask";
const VARIANTS: { key: VariantKey; series: "FlyConnectome" | "RandomRecurrent" }[] = [
  { key: "FlyConnectome", series: "FlyConnectome" },
  { key: "RandomMask", series: "RandomRecurrent" },
];
/** Short names used both in the ring captions and the timeline rows. */
const SHORT: Record<VariantKey, string> = {
  FlyConnectome: "real wiring",
  RandomMask: "random wiring",
};

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | number> = {},
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG, tag);
  set(node, attributes);
  return node;
}

function set(node: Element, attributes: Record<string, string | number>): void {
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
}

/* ---------- compass ring: built once, updated per step ---------- */

interface Ring {
  root: SVGSVGElement;
  update: (p: Palette, color: string, predicted: number, theta: number, landmark: boolean) => void;
}

function buildRing(replay: Replay, markerId: string): Ring {
  const size = 220;
  const c = size / 2;
  const outer = 96;
  const inner = 62;
  const root = svg("svg", { viewBox: `0 0 ${size} ${size}`, class: "ring", role: "img" });
  const defs = svg("defs");
  const marker = svg("marker", {
    id: markerId,
    viewBox: "0 0 10 10",
    refX: 6,
    refY: 5,
    markerWidth: 5,
    markerHeight: 5,
    orient: "auto",
  });
  const arrow = svg("path", { d: "M0,0 L10,5 L0,10 Z" });
  marker.append(arrow);
  defs.append(marker);
  const halo = svg("circle", {
    cx: c,
    cy: c,
    r: outer + 8,
    fill: "none",
    "stroke-width": 2,
    "stroke-dasharray": "3 5",
  });
  const sectors = Array.from({ length: replay.sectors }, (_, k) =>
    svg("path", { d: sectorPath(c, c, inner, outer, k, replay.sectors), "stroke-width": 1.5 }),
  );
  const needle = svg("line", {
    "stroke-width": 3,
    "stroke-linecap": "round",
    "marker-end": `url(#${markerId})`,
  });
  const hub = svg("circle", { cx: c, cy: c, r: 5 });
  root.append(defs, halo, ...sectors, needle, hub);

  return {
    root,
    update(p, color, predicted, theta, landmark) {
      set(halo, { stroke: p.ink2, opacity: landmark ? 1 : 0 });
      sectors.forEach((path, k) =>
        set(path, { fill: k === predicted ? color : p.surface, stroke: p.grid }),
      );
      const tip = polar(c, c, outer - 6, theta);
      const tail = polar(c, c, 16, theta + Math.PI);
      set(needle, { x1: tail.x, y1: tail.y, x2: tip.x, y2: tip.y, stroke: p.ink });
      set(arrow, { fill: p.ink });
      set(hub, { fill: p.ink });
    },
  };
}

/* ---------- timeline: rebuilt only when the sequence, theme or size class changes ---------- */

interface Timeline {
  root: SVGSVGElement;
  moveTo: (step: number) => void;
}

function buildTimeline(
  replay: Replay,
  sequence: number,
  p: Palette,
  compact: boolean,
  onSeek: (step: number) => void,
): Timeline {
  const left = compact ? 84 : 96;
  const cell = compact ? 10 : 16;
  const width = left + replay.steps * cell + 4;
  const rowY = [14, 40, 62];
  const half = compact ? 3 : 4;
  const root = svg("svg", { viewBox: `0 0 ${width} 80`, class: "timeline", role: "img" });
  root.setAttribute("aria-label", "Timeline of landmarks and right or wrong answers per step");
  const highlight = svg("rect", {
    y: 1,
    width: cell,
    height: 78,
    rx: 3,
    fill: "none",
    stroke: p.ink,
    "stroke-width": 1.5,
  });
  root.append(highlight);
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
  VARIANTS.forEach(({ key }, row) =>
    label(rowY[row + 1], compact ? SHORT[key].replace(" wiring", "") : SHORT[key]),
  );
  const truth = replay.sequences[sequence].sector;
  for (let t = 0; t < replay.steps; t += 1) {
    const x = left + t * cell + cell / 2;
    if (replay.landmark[t]) {
      root.append(svg("circle", { cx: x, cy: rowY[0], r: half, fill: p.ink2 }));
    }
    VARIANTS.forEach(({ key, series }, row) => {
      const y = rowY[row + 1];
      if (!replay.scored[t]) {
        root.append(
          svg("line", { x1: x - 3, y1: y, x2: x + 3, y2: y, stroke: p.muted, "stroke-width": 1.5 }),
        );
      } else if (replay.variants[key].predicted[sequence][t] === truth[t]) {
        root.append(svg("circle", { cx: x, cy: y, r: half + 0.5, fill: p.series[series] }));
      } else {
        root.append(
          svg("path", {
            d:
              `M${x - half},${y - half} L${x + half},${y + half} ` +
              `M${x + half},${y - half} L${x - half},${y + half}`,
            stroke: p.ink,
            "stroke-width": 1.8,
            "stroke-linecap": "round",
          }),
        );
      }
    });
  }
  // One listener for the whole timeline: map the click position to a step.
  root.style.cursor = "pointer";
  root.addEventListener("click", (event) => {
    const matrix = root.getScreenCTM();
    if (!matrix) return;
    const x = (event.clientX - matrix.e) / matrix.a;
    const step = Math.floor((x - left) / cell);
    if (step >= 0 && step < replay.steps) onSeek(step);
  });
  return {
    root,
    moveTo(step) {
      highlight.setAttribute("x", String(left + step * cell));
    },
  };
}

/* ---------- view ---------- */

export async function renderReplay(root: HTMLElement, scope: Scope): Promise<void> {
  const replay = await loadReplay();
  if (!scope.active) return;
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let sequence = 0;
  let step = 0;
  let playing = !reduceMotion;
  let speed = 1;
  let timer = 0;
  let timeline: Timeline | undefined;

  const fly = replay.variants.FlyConnectome.accuracy;
  const random = replay.variants.RandomMask.accuracy;
  const resultsLink = el("a", { text: "Results" });
  resultsLink.href = "#/results";
  root.append(
    el("h1", { text: "Watch a network keep track of heading" }),
    el("p", {
      className: "lede",
      text:
        "The task: a simulated fly turns at random. Every 8 steps it sees a landmark that tells it " +
        "where it is facing; in between it only feels how fast it turns, and the network must add " +
        "those turns up to report its heading as one of 8 sectors. The needle is the true heading; " +
        "the filled sector is the network's answer.",
    }),
  );

  // Controls are built once; playback only updates attributes, so clicks and focus survive.
  const sequenceButtons = el("div", { className: "segmented" });
  sequenceButtons.setAttribute("role", "group");
  sequenceButtons.setAttribute("aria-label", "Test sequence");
  const seqButtons = replay.sequences.map((_, i) => {
    const button = el("button", { text: String(i + 1) });
    button.type = "button";
    button.addEventListener("click", () => {
      sequence = i;
      step = 0;
      rebuildTimeline();
      update();
    });
    sequenceButtons.append(button);
    return button;
  });
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

  const rings = VARIANTS.map(({ key, series }, i) => {
    const ring = buildRing(replay, `needle-arrow-${i}`);
    const verdict = el("span", { className: "verdict" });
    const score = el("span");
    const figure = el(
      "figure",
      { className: "ring-figure" },
      ring.root,
      el("figcaption", {}, verdict, el("strong", { text: SERIES_LABEL[series] }), score),
    );
    return { key, series, ring, verdict, score, figure };
  });
  const timelineSlot = el("div", { className: "timeline-slot" });
  const key = el(
    "p",
    { className: "cell-key" },
    el("span", { text: "● right answer" }),
    el("span", { text: "✕ wrong answer" }),
    el("span", { text: "– not scored (landmark visible)" }),
    el("span", { text: "▢ current step; click the timeline to jump" }),
  );

  root.append(
    el(
      "section",
      { className: "card" },
      el("h2", { text: "One test run, step by step" }),
      el("div", { className: "controls" }, el("span", { text: "Sequence" }), sequenceButtons),
      el("div", { className: "rings" }, ...rings.map((r) => r.figure)),
      el("div", { className: "player" }, play, scrub, stepLabel, speedSelect),
      timelineSlot,
      key,
      el(
        "p",
        { className: "note" },
        `Both networks were trained on the whole CX graph (${replay.nodes.toLocaleString("en")} ` +
          "neurons) with the same budget; one uses the real wiring, the other a random wiring with " +
          `the same number of connections. On this seed (${replay.seed}) they score about the same ` +
          `over 64 test sequences: real ${pct(fly)}, random ${pct(random)}. The real wiring's edge ` +
          "(+2.8 percentage points) only shows when averaged over 20 seeds — see ",
        resultsLink,
        ". These six are the first six test sequences, not hand-picked.",
      ),
    ),
  );

  function update(): void {
    const p = palette();
    const data = replay.sequences[sequence];
    const landmark = replay.landmark[step];
    seqButtons.forEach((button, i) => button.setAttribute("aria-pressed", String(i === sequence)));
    for (const { key: variant, series, ring, verdict, score, figure } of rings) {
      const predicted = replay.variants[variant].predicted[sequence];
      ring.update(p, p.series[series], predicted[step], data.theta[step], landmark);
      const correct = replay.scored[step] ? predicted[step] === data.sector[step] : null;
      verdict.textContent = correct === null ? "landmark visible" : correct ? "✓ right" : "✕ wrong";
      const total = runningScore(predicted, data.sector, replay.scored, step);
      score.textContent = total.total
        ? `${total.hits} / ${total.total} right so far (${pct(total.hits / total.total, 0)})`
        : "no scored step yet";
      ring.root.setAttribute(
        "aria-label",
        `${SHORT[variant]}: answers sector ${predicted[step] + 1}, true sector ` +
          `${data.sector[step] + 1} of ${replay.sectors}`,
      );
      figure.dataset.verdict = correct === null ? "landmark" : correct ? "right" : "wrong";
    }
    const since = stepsSinceLandmark(replay.landmark, step);
    stepLabel.textContent =
      `Step ${step + 1} / ${replay.steps} · ` +
      (landmark ? "landmark visible" : `${since} step${since === 1 ? "" : "s"} since landmark`);
    scrub.value = String(step);
    play.textContent = playing ? "❚❚ Pause" : "▶ Play";
    play.setAttribute("aria-pressed", String(playing));
    timeline?.moveTo(step);
  }

  function rebuildTimeline(): void {
    const compact = timelineSlot.clientWidth > 0 && timelineSlot.clientWidth < 560;
    timeline = buildTimeline(replay, sequence, palette(), compact, (t) => {
      step = t;
      playing = false;
      schedule();
      update();
    });
    timelineSlot.replaceChildren(timeline.root);
  }

  function schedule(): void {
    window.clearTimeout(timer);
    if (!playing) return;
    timer = window.setTimeout(() => {
      step = (step + 1) % replay.steps;
      update();
      schedule();
    }, 450 / speed);
  }

  play.addEventListener("click", () => {
    playing = !playing;
    update();
    schedule();
  });
  scrub.addEventListener("input", () => {
    step = Number(scrub.value);
    playing = false;
    schedule();
    update();
  });
  speedSelect.addEventListener("change", () => {
    speed = Number(speedSelect.value);
    schedule();
  });
  const onKey = (event: KeyboardEvent): void => {
    const target = event.target as HTMLElement;
    if (event.key !== " " || target.closest("input, select, button, a")) return;
    event.preventDefault();
    playing = !playing;
    update();
    schedule();
  };
  document.addEventListener("keydown", onKey);

  // Rebuild the timeline only when it crosses the compact breakpoint.
  let compactNow: boolean | undefined;
  const resize = new ResizeObserver(() => {
    const compact = timelineSlot.clientWidth < 560;
    if (compact === compactNow) return;
    compactNow = compact;
    rebuildTimeline();
    update();
  });
  resize.observe(timelineSlot);

  const stopScheme = onScheme(() => {
    rebuildTimeline();
    update();
  });
  schedule();
  scope.onLeave(() => {
    window.clearTimeout(timer);
    document.removeEventListener("keydown", onKey);
    resize.disconnect();
    stopScheme();
  });
}
