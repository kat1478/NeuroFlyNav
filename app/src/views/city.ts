import type { Feature, FeatureCollection } from "geojson";
import {
  addProtocol,
  Map as MapLibreMap,
  Marker,
  NavigationControl,
  setWorkerUrl,
  type GeoJSONSource,
  type LngLatBoundsLike,
  type StyleSpecification,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { Protocol } from "pmtiles";
import { loadCity, loadTreats, type City, type CityFlight } from "../data";
import type { Scope } from "../lifecycle";
import { onScheme, palette, SERIES_LABEL, type Palette } from "../theme";
import { el, pct } from "../ui";

type Net = "FlyConnectome" | "RandomMask";
const NETS: { key: Net; series: "FlyConnectome" | "RandomRecurrent"; short: string }[] = [
  { key: "FlyConnectome", series: "FlyConnectome", short: "real wiring" },
  { key: "RandomMask", series: "RandomRecurrent", short: "random wiring" },
];
const BOUNDS: LngLatBoundsLike = [
  [20.992, 52.217],
  [21.048, 52.25],
];

let protocolAdded = false;

/** Map colours derived from the app tokens, so the map follows light and dark mode. */
function mapColors(dark: boolean) {
  return dark
    ? {
        land: "#1f1f1d",
        water: "#1d3047",
        park: "#22301f",
        road: "#3a3a37",
        major: "#4a4a46",
        building: "#2a2a28",
      }
    : {
        land: "#f4f3ee",
        water: "#cfe0f3",
        park: "#dfeccf",
        road: "#ffffff",
        major: "#ffffff",
        building: "#e2e0d8",
      };
}

function baseStyle(url: string, p: Palette, dark: boolean): StyleSpecification {
  const c = mapColors(dark);
  const major = [
    "in",
    ["get", "highway"],
    ["literal", ["primary", "secondary", "tertiary", "trunk"]],
  ];
  return {
    version: 8,
    sources: {
      osm: {
        type: "vector",
        url: `pmtiles://${url}`,
        attribution:
          '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors (ODbL)',
      },
      trips: { type: "geojson", data: { type: "FeatureCollection", features: [] } },
      treats: { type: "geojson", data: { type: "FeatureCollection", features: [] } },
    },
    layers: [
      { id: "land", type: "background", paint: { "background-color": c.land } },
      {
        id: "parks",
        type: "fill",
        source: "osm",
        "source-layer": "parks",
        paint: { "fill-color": c.park },
      },
      {
        id: "water",
        type: "fill",
        source: "osm",
        "source-layer": "water",
        paint: { "fill-color": c.water },
      },
      {
        id: "roads",
        type: "line",
        source: "osm",
        "source-layer": "roads",
        filter: ["!", major] as never,
        paint: {
          "line-color": c.road,
          "line-width": ["interpolate", ["linear"], ["zoom"], 13, 0.6, 16, 3],
        },
      },
      {
        id: "roads-major",
        type: "line",
        source: "osm",
        "source-layer": "roads",
        filter: major as never,
        paint: {
          "line-color": c.major,
          "line-width": ["interpolate", ["linear"], ["zoom"], 13, 1.5, 16, 7],
        },
      },
      {
        id: "buildings",
        type: "fill",
        source: "osm",
        "source-layer": "buildings",
        paint: { "fill-color": c.building, "fill-outline-color": dark ? "#353532" : "#d2d0c6" },
      },
      {
        id: "treats",
        type: "circle",
        source: "treats",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 13, 1.8, 16, 4],
          "circle-color": p.muted,
          "circle-opacity": 0.7,
        },
      },
      {
        id: "outbound",
        type: "line",
        source: "trips",
        filter: ["==", ["get", "kind"], "outbound"],
        paint: { "line-color": p.ink, "line-width": 2.5, "line-dasharray": [1.5, 1.2] },
      },
      {
        id: "return-error",
        type: "line",
        source: "trips",
        filter: ["==", ["get", "kind"], "error"],
        paint: { "line-color": ["get", "color"], "line-width": 1.5, "line-dasharray": [0.6, 1.4] },
      },
      {
        id: "return",
        type: "line",
        source: "trips",
        filter: ["==", ["get", "kind"], "return"],
        paint: { "line-color": ["get", "color"], "line-width": 3.5 },
      },
    ],
  };
}

function marker(className: string, text = ""): HTMLElement {
  return el("div", { className: `map-marker ${className}`, text });
}

/** A small fly drawn in SVG (emoji fonts are not available everywhere). */
function flyIcon(): HTMLElement {
  const wrap = el("div", { className: "fly-icon" });
  wrap.innerHTML =
    '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">' +
    '<ellipse cx="7.5" cy="9" rx="5" ry="3" transform="rotate(-25 7.5 9)" fill="currentColor" opacity="0.35"/>' +
    '<ellipse cx="16.5" cy="9" rx="5" ry="3" transform="rotate(25 16.5 9)" fill="currentColor" opacity="0.35"/>' +
    '<ellipse cx="12" cy="13" rx="3" ry="5.5" fill="currentColor"/>' +
    '<circle cx="12" cy="6.8" r="2.2" fill="currentColor"/></svg>';
  return wrap;
}

/** Landing dot at the exact point; its label sits above, real wiring left, random right. */
function landingDot(net: Net): HTMLElement {
  return el("span", { className: `landing-dot landing-${net}` });
}

const toRad = (deg: number): number => (deg * Math.PI) / 180;
/** Metres between two lon/lat points (equirectangular; fine at city scale). */
function metres(a: [number, number], b: [number, number]): number {
  const x = toRad(b[0] - a[0]) * Math.cos(toRad((a[1] + b[1]) / 2)) * 6_371_000;
  const y = toRad(b[1] - a[1]) * 6_371_000;
  return Math.hypot(x, y);
}

function lerp(a: [number, number], b: [number, number], t: number): [number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

/** Trip geometry at animation progress: outbound steps done, then return fraction 0..1. */
export function tripFeatures(
  city: City,
  flight: CityFlight,
  outboundSteps: number,
  returnFraction: number,
  colors: Record<Net, string>,
): FeatureCollection {
  const features: Feature[] = [];
  const path = flight.path.slice(0, Math.max(1, outboundSteps + 1));
  if (path.length > 1) {
    features.push({
      type: "Feature",
      properties: { kind: "outbound" },
      geometry: { type: "LineString", coordinates: path },
    });
  }
  if (returnFraction > 0) {
    const start = flight.path[flight.path.length - 1];
    for (const { key } of NETS) {
      const landing = flight.returns[key].landing;
      const end = lerp(start, landing, Math.min(1, returnFraction));
      features.push({
        type: "Feature",
        properties: { kind: "return", color: colors[key] },
        geometry: { type: "LineString", coordinates: [start, end] },
      });
      if (returnFraction >= 1) {
        features.push({
          type: "Feature",
          properties: { kind: "error", color: colors[key] },
          geometry: { type: "LineString", coordinates: [landing, city.nest] },
        });
      }
    }
  }
  return { type: "FeatureCollection", features };
}

export async function renderCity(root: HTMLElement, scope: Scope): Promise<void> {
  const [city, treats] = await Promise.all([loadCity(), loadTreats()]);
  if (!scope.active) return;
  if (!protocolAdded) {
    // Vite bundles MapLibre's module worker separately; tell MapLibre where it is.
    setWorkerUrl(workerUrl);
    addProtocol("pmtiles", new Protocol().tile);
    protocolAdded = true;
  }
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pmtilesUrl = new URL(`${import.meta.env.BASE_URL}map/warsaw.pmtiles`, location.href).href;

  root.append(
    el("h1", { text: "A fly forages in central Warsaw" }),
    el("p", {
      className: "lede",
      text:
        "Insects that forage keep a running sum of every turn and every stretch they fly — a " +
        "home vector — so they can fly straight back to their nest. Here a simulated fly leaves " +
        "its nest in Park Świętokrzyski, searches until it smells a real café, bakery or ice-cream " +
        "shop from OpenStreetMap, and then flies home. Its heading at every step comes from the " +
        "trained network of the Replay view: one built on the real central-complex wiring, one on " +
        "a random wiring of the same size.",
    }),
  );

  const flightButtons = el("div", { className: "segmented" });
  flightButtons.setAttribute("role", "group");
  flightButtons.setAttribute("aria-label", "Flight");
  const play = el("button", { className: "play" });
  play.type = "button";
  const status = el("span", { className: "step-label" });
  const mapBox = el("div", { className: "city-map" });
  mapBox.setAttribute("role", "img");
  const side = el("aside", { className: "family-card city-side" });
  const legend = el("p", { className: "cell-key" });

  root.append(
    el(
      "section",
      { className: "card" },
      el("h2", { text: "One foraging trip" }),
      el("div", { className: "controls" }, el("span", { text: "Flight" }), flightButtons),
      el("div", { className: "city-layout" }, mapBox, side),
      el("div", { className: "player" }, play, status),
      legend,
      el("p", {
        className: "note",
        text:
          "An illustration, not a navigation system: the fly flies over buildings, does not see " +
          "the map, and the network reports only 8 heading sectors (45° each), so even perfect " +
          `sector answers miss the nest by ${city.references_m.quantised.mean.toFixed(0)} m on ` +
          `average. Flights are the first six of seed ${city.seed}, not hand-picked; flying ` +
          "5 m per step, with a landmark glimpse every 8 steps as in training.",
      }),
    ),
  );
  const resultCard = el("section", { className: "card" });
  root.append(resultCard);

  let flightIndex = 0;
  let step = 0;
  let returnFraction = 0;
  let playing = !reduceMotion;
  let frame = 0;
  let last = 0;

  const map = new MapLibreMap({
    container: mapBox,
    style: baseStyle(pmtilesUrl, palette(), isDark()),
    bounds: BOUNDS,
    maxBounds: [
      [20.97, 52.205],
      [21.07, 52.262],
    ],
    minZoom: 12.5,
    maxZoom: 17.5,
    attributionControl: { compact: true },
    cooperativeGestures: true,
  });
  map.addControl(new NavigationControl({ showCompass: false }), "top-right");

  new Marker({ element: el("div", { className: "nest-dot" }) }).setLngLat(city.nest).addTo(map);
  new Marker({ element: marker("nest", "⌂ nest"), anchor: "bottom", offset: [0, -10] })
    .setLngLat(city.nest)
    .addTo(map);
  const treatMarker = new Marker({ element: marker("treat"), anchor: "bottom", offset: [0, -16] });
  const flyMarker = new Marker({ element: flyIcon() });
  // Landing points are often metres from the nest, so their distances are listed in the
  // side panel rather than as map labels (which would collide).
  const landingMarkers = NETS.map(({ key }) => ({
    key,
    dot: new Marker({ element: landingDot(key) }),
  }));

  function isDark(): boolean {
    const forced = document.documentElement.dataset.theme;
    if (forced) return forced === "dark";
    return matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function colors(p: Palette): Record<Net, string> {
    return { FlyConnectome: p.series.FlyConnectome, RandomMask: p.series.RandomRecurrent };
  }

  function flight(): CityFlight {
    return city.flights[flightIndex];
  }

  function drawData(): void {
    const source = map.getSource("trips") as GeoJSONSource | undefined;
    if (!source) return;
    const current = flight();
    source.setData(tripFeatures(city, current, step, returnFraction, colors(palette())));
    const position =
      returnFraction > 0
        ? current.path[current.path.length - 1]
        : current.path[Math.min(step, current.path.length - 1)];
    flyMarker.setLngLat(position).addTo(map);
    for (const { key, dot } of landingMarkers) {
      if (returnFraction >= 1) dot.setLngLat(current.returns[key].landing).addTo(map);
      else dot.remove();
    }
    const outboundTotal = current.path.length - 1;
    status.textContent =
      returnFraction > 0
        ? returnFraction >= 1
          ? "Back home — dotted lines show how far each fly landed from the nest"
          : "Flying home along the estimated home vector…"
        : `Searching · step ${Math.min(step, outboundTotal)} / ${outboundTotal}`;
    play.textContent = playing
      ? "❚❚ Pause"
      : step === 0 && returnFraction === 0
        ? "▶ Play"
        : "▶ Replay";
    mapBox.setAttribute(
      "aria-label",
      `Map of central Warsaw: flight ${flightIndex + 1} from the nest to ` +
        `${current.treat ? current.treat.name || `a ${current.treat.kind}` : "no treat"}; ` +
        NETS.map(
          ({ key, short }) => `${short} lands ${current.returns[key].error_m} m from the nest`,
        ).join(", "),
    );
  }

  function drawSide(): void {
    const current = flight();
    const end = current.path[current.path.length - 1];
    const treatText = current.treat
      ? `${current.treat.name || "unnamed"} (${current.treat.kind})`
      : "none within reach";
    treatMarker.getElement().textContent = current.treat
      ? current.treat.name || current.treat.kind
      : "";
    if (current.treat) treatMarker.setLngLat(end).addTo(map);
    else treatMarker.remove();
    const rows = NETS.map(({ key, series }) =>
      el(
        "li",
        {},
        el("span", { className: "swatch-line", text: "" }),
        el("span", { text: SERIES_LABEL[series] }),
        el("strong", { className: "num", text: `${current.returns[key].error_m.toFixed(0)} m` }),
      ),
    );
    rows.forEach((row, i) => {
      (row.firstChild as HTMLElement).style.background = colors(palette())[NETS[i].key];
    });
    side.replaceChildren(
      el("h3", { text: `Flight ${flightIndex + 1}` }),
      el(
        "div",
        { className: "stats" },
        stat(`${((current.path.length - 1) * city.homing.step_m).toFixed(0)} m`, "flown out"),
        stat(`${metres(city.nest, end).toFixed(0)} m`, "straight home"),
        stat(String(current.path.length - 1), "steps"),
      ),
      el("p", {}, el("strong", { text: "Treat: " }), treatText),
      el("p", { className: "partners-title", text: "Lands this far from the nest" }),
      el("ul", { className: "partners city-errors" }, ...rows),
    );
  }

  function stat(value: string, name: string): HTMLElement {
    return el(
      "div",
      {},
      el("span", { className: "stat-value", text: value }),
      el("span", { className: "stat-label", text: name }),
    );
  }

  function drawResult(): void {
    const s = city.summary;
    const pilot = city.pilot;
    const head = el("h2", { text: "Does the real wiring fly home closer?" });
    if (!s) {
      resultCard.replaceChildren(
        head,
        el("p", {
          className: "note",
          text:
            `Pilot only (seed ${city.seed}, ${pilot.FlyConnectome.flights} flights): mean distance ` +
            `from the nest ${pilot.FlyConnectome.error_m.mean.toFixed(1)} m with the real wiring, ` +
            `${pilot.RandomMask.error_m.mean.toFixed(1)} m with the random wiring. The ` +
            "pre-registered test on 20 fresh seeds is still running.",
        }),
      );
      return;
    }
    const d = s.fly_minus_error_m.RandomMask;
    resultCard.replaceChildren(
      head,
      el("p", {
        className: "lede",
        text:
          `Pre-registered test, seeds ${s.seeds[0]}–${s.seeds[s.seeds.length - 1]}, 100 flights ` +
          `per seed: the real wiring lands ${Math.abs(d.mean).toFixed(1)} m ` +
          `${d.mean < 0 ? "closer to" : "further from"} the nest than a random wiring of the same ` +
          `size (95% CI ${d.low.toFixed(1)} to ${d.high.toFixed(1)} m). ` +
          (d.high < 0
            ? "The interval is below zero: a small but clear homing advantage."
            : "The interval includes zero or lies above it: no homing advantage."),
      }),
      el(
        "p",
        { className: "note" },
        `Mean distance from the nest: real ${s.mean_error_m.FlyConnectome.mean.toFixed(1)} m, ` +
          `random ${s.mean_error_m.RandomMask.mean.toFixed(1)} m; perfect sector answers ` +
          `${s.quantised_error_m.mean.toFixed(1)} m; no integration ` +
          `${s.no_integration_error_m.mean.toFixed(1)} m. Heading accuracy on the flights: real ` +
          `${pct(s.heading_accuracy.FlyConnectome.mean)}, random ` +
          `${pct(s.heading_accuracy.RandomMask.mean)}.`,
      ),
    );
  }

  function selectFlight(i: number): void {
    flightIndex = i;
    step = 0;
    returnFraction = 0;
    buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
    const path = city.flights[i].path;
    const lons = [...path.map((p) => p[0]), city.nest[0]];
    const lats = [...path.map((p) => p[1]), city.nest[1]];
    map.fitBounds(
      [
        [Math.min(...lons), Math.min(...lats)],
        [Math.max(...lons), Math.max(...lats)],
      ],
      { padding: 70, maxZoom: 16.5, duration: reduceMotion ? 0 : 600 },
    );
    if (reduceMotion && !playing) {
      step = path.length - 1;
      returnFraction = 1;
    }
    drawSide();
    drawData();
  }

  const buttons = city.flights.map((_, i) => {
    const button = el("button", { text: String(i + 1) });
    button.type = "button";
    button.addEventListener("click", () => {
      selectFlight(i);
      if (!reduceMotion) start();
    });
    flightButtons.append(button);
    return button;
  });

  function tick(time: number): void {
    const dt = last ? (time - last) / 1000 : 0;
    last = time;
    const total = flight().path.length - 1;
    if (step < total) step = Math.min(total, step + dt * 14);
    else if (returnFraction < 1) returnFraction = Math.min(1, returnFraction + dt / 1.4);
    else playing = false;
    stepView();
    if (playing) frame = requestAnimationFrame(tick);
  }

  function stepView(): void {
    const s = step;
    step = Math.floor(s);
    drawData();
    step = s;
  }

  function start(): void {
    cancelAnimationFrame(frame);
    if (step >= flight().path.length - 1 && returnFraction >= 1) {
      step = 0;
      returnFraction = 0;
    }
    playing = true;
    last = 0;
    frame = requestAnimationFrame(tick);
  }

  play.addEventListener("click", () => {
    if (playing) {
      playing = false;
      cancelAnimationFrame(frame);
      stepView();
    } else start();
  });

  legend.replaceChildren(
    el("span", { text: "⌂ nest" }),
    el("span", { text: "• treats (OpenStreetMap)" }),
    el("span", { text: "- - outbound search" }),
    el("span", { text: "━ flight home, real wiring (blue)" }),
    el("span", { text: "━ flight home, random wiring (green)" }),
    el("span", { text: "● where each fly lands · ··· distance left to the nest" }),
  );

  map.on("load", () => {
    (map.getSource("treats") as GeoJSONSource).setData(treats as unknown as FeatureCollection);
    selectFlight(0);
    if (playing) start();
  });

  const restyle = (): void => {
    if (!map.isStyleLoaded()) return;
    const p = palette();
    const c = mapColors(isDark());
    map.setPaintProperty("land", "background-color", c.land);
    map.setPaintProperty("parks", "fill-color", c.park);
    map.setPaintProperty("water", "fill-color", c.water);
    map.setPaintProperty("roads", "line-color", c.road);
    map.setPaintProperty("roads-major", "line-color", c.major);
    map.setPaintProperty("buildings", "fill-color", c.building);
    map.setPaintProperty("treats", "circle-color", p.muted);
    map.setPaintProperty("outbound", "line-color", p.ink);
    drawSide();
    stepView();
  };
  const stopScheme = onScheme(restyle);
  drawResult();
  scope.onLeave(() => {
    cancelAnimationFrame(frame);
    stopScheme();
    map.remove();
  });
}
