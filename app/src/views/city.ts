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
import {
  distance,
  FLIGHT,
  flyOut,
  headingSector,
  landing as landingPoint,
  sectorCentre,
  toLocal,
  toLonLat,
  type Point,
} from "../fly/flight";
import { DOWNLOAD_MB_COMPRESSED, getNetworks, runSectors, type Networks } from "../fly/model";
import type { Scope } from "../lifecycle";
import { onScheme, palette, type Palette } from "../theme";
import { el, pct } from "../ui";
import { homeVectorCard } from "./homevector";

type Net = "FlyConnectome" | "RandomMask";
const NETS: { key: Net; series: "FlyConnectome" | "RandomRecurrent"; short: string }[] = [
  { key: "FlyConnectome", series: "FlyConnectome", short: "real wiring" },
  { key: "RandomMask", series: "RandomRecurrent", short: "random wiring" },
];
const BOUNDS: LngLatBoundsLike = [
  [20.992, 52.217],
  [21.048, 52.25],
];

/** The area covered by the map data (geo/build_map.py BBOX). */
const MAPPED: [[number, number], [number, number]] = [
  [21.0, 52.222],
  [21.04, 52.245],
];

/** The mapped area as a closed line, shown while the visitor places a nest. */
function mappedFrame(): Feature {
  const [[w, s], [e, n]] = MAPPED;
  return {
    type: "Feature",
    properties: {},
    geometry: {
      type: "LineString",
      coordinates: [
        [w, s],
        [e, s],
        [e, n],
        [w, n],
        [w, s],
      ],
    },
  };
}

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
      mapped: { type: "geojson", data: mappedFrame() },
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
        id: "mapped-edge",
        type: "line",
        source: "mapped",
        layout: { visibility: "none" },
        paint: { "line-color": p.muted, "line-width": 1.5, "line-dasharray": [3, 2] },
      },
      {
        id: "treats",
        type: "circle",
        source: "treats",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 13, 1.8, 16, 4],
          "circle-color": p.muted,
          "circle-opacity": dark ? 0.9 : 0.7,
        },
      },
      {
        id: "treat-found",
        type: "circle",
        source: "treats",
        filter: ["==", ["get", "id"], -1],
        paint: {
          "circle-radius": 7,
          "circle-color": p.ink,
          "circle-stroke-color": p.surface,
          "circle-stroke-width": 2.5,
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
      // Identity is not colour alone: random wiring = wide dashed line underneath,
      // real wiring = narrow solid line on top (they often nearly coincide).
      {
        id: "return-random",
        type: "line",
        source: "trips",
        filter: ["all", ["==", ["get", "kind"], "return"], ["==", ["get", "net"], "RandomMask"]],
        paint: { "line-color": ["get", "color"], "line-width": 7, "line-dasharray": [1.2, 0.8] },
      },
      {
        id: "return-real",
        type: "line",
        source: "trips",
        filter: ["all", ["==", ["get", "kind"], "return"], ["==", ["get", "net"], "FlyConnectome"]],
        paint: { "line-color": ["get", "color"], "line-width": 2.5 },
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

/** Landing marker: circle for the real wiring, triangle for the random wiring. */
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
        properties: { kind: "return", net: key, color: colors[key] },
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
        "its nest (in Park Świętokrzyski, or wherever you drop it), searches until it smells a real " +
        "café, bakery or ice-cream " +
        "shop from OpenStreetMap, and then flies home. At every step a trained network (the one in the " +
        "Replay view) estimates the fly's compass heading as one of 8 sectors; adding up these " +
        "estimates gives the home vector the fly follows back. Two networks are compared: one " +
        "built on the real central-complex wiring, one on a random wiring of the same size.",
    }),
  );

  const flightButtons = el("div", { className: "segmented" });
  flightButtons.setAttribute("role", "group");
  flightButtons.setAttribute("aria-label", "Flight");
  const modeSwitch = el("div", { className: "segmented" });
  modeSwitch.setAttribute("role", "group");
  modeSwitch.setAttribute("aria-label", "What to show");
  const dropKind = el("div", { className: "segmented" });
  dropKind.setAttribute("role", "group");
  dropKind.setAttribute("aria-label", "How the fly finds a treat");
  const dropHint = el("p", { className: "drop-hint" });
  dropHint.setAttribute("aria-live", "polite");
  // Step-by-step progress is not announced (too chatty); the hint above says when it is done.
  const dropProgress = el("span", { className: "drop-progress num" });
  const randomButton = el("button", { className: "link-button", text: "Drop at a random spot" });
  randomButton.type = "button";
  const pilotControls = el(
    "div",
    { className: "controls" },
    el("span", { text: "Flight" }),
    flightButtons,
  );
  const dropControls = el(
    "div",
    { className: "drop-controls" },
    el("div", { className: "controls" }, el("span", { text: "The fly" }), dropKind, randomButton),
    dropHint,
  );
  dropControls.hidden = true;
  const dropNote = el("p", {
    className: "note",
    text:
      "Each fly you drop is a new random flight computed live in your browser, with the same " +
      "trained networks; a single flight says nothing about which wiring is better (one network " +
      "can land closer by chance) — the pre-registered test below compares 2,000 flights. In " +
      `“pick a treat” the turns are gently pulled towards the treat, and a fly that gets no closer ` +
      `than ${FLIGHT.detectM} m within ${FLIGHT.maxSteps} steps flies home anyway. Place the nest ` +
      "inside the mapped area. Flies from other nests, and steered flies, tend to be longer than " +
      "the tested ones (Park Świętokrzyski nest, unsteered search), and longer flights miss " +
      "the nest by more.",
  });
  dropNote.hidden = true;
  const play = el("button", { className: "play" });
  play.type = "button";
  const status = el("span", { className: "step-label" });
  const mapBox = el("div", { className: "city-map" });
  const mapSummary = el("p", { className: "sr-only" });
  mapSummary.id = "city-map-summary";
  mapBox.setAttribute("role", "region");
  mapBox.setAttribute("aria-label", "Map of central Warsaw with the flight");
  mapBox.setAttribute("aria-describedby", mapSummary.id);
  const zoomButton = el("button", { className: "link-button", text: "Close-up of the nest" });
  zoomButton.type = "button";
  const mapError = el("p", { className: "error" });
  mapError.hidden = true;
  const side = el("aside", { className: "family-card city-side" });
  const legend = el("p", { className: "cell-key" });

  root.append(
    el(
      "section",
      { className: "card" },
      el("h2", { text: "One foraging trip" }),
      el("div", { className: "controls" }, modeSwitch),
      pilotControls,
      dropControls,
      el(
        "div",
        { className: "city-layout" },
        el(
          "div",
          {},
          mapBox,
          mapError,
          mapSummary,
          el("div", { className: "player" }, play, status, dropProgress, zoomButton),
          legend,
        ),
        side,
      ),
      dropNote,
      el("p", {
        className: "note pilot-note",
        text:
          "An illustration, not a navigation system: the fly flies over buildings, does not see " +
          "the map, and the network reports only 8 heading sectors (45° each), so even perfect " +
          `sector answers miss the nest by ${city.references_m.quantised.mean.toFixed(0)} m on ` +
          `average. The six flights are the first six of pilot seed ${city.seed} (not part of the ` +
          "test below), not hand-picked; the fly moves 5 m per step and glimpses a landmark every " +
          "8 steps, as in training.",
      }),
    ),
  );
  const resultCard = el("section", { className: "card" });
  root.append(resultCard);

  // The flight on screen: one of the pilot flights, or a fly dropped by the visitor.
  let active: { nest: [number, number]; flight: CityFlight; title: string; live: boolean } = {
    nest: city.nest,
    flight: city.flights[0],
    title: "Flight 1",
    live: false,
  };
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
    maxZoom: 19.5,
    attributionControl: { compact: true },
    cooperativeGestures: true,
  });
  map.addControl(new NavigationControl({ showCompass: false }), "top-right");

  const nestDot = new Marker({ element: el("div", { className: "nest-dot" }) })
    .setLngLat(city.nest)
    .addTo(map);
  const nestLabel = new Marker({
    element: marker("nest", "⌂ nest"),
    anchor: "bottom",
    offset: [0, -10],
  })
    .setLngLat(city.nest)
    .addTo(map);
  const placeNest = (lngLat: [number, number]): void => {
    nestDot.setLngLat(lngLat).addTo(map);
    nestLabel.setLngLat(lngLat).addTo(map);
  };
  const treatMarker = new Marker({ element: marker("treat"), anchor: "bottom", offset: [0, -16] });
  const flyMarker = new Marker({ element: flyIcon() });
  // Landing points are often metres from the nest, so their distances are listed in the
  // side panel rather than as map labels (which would collide).
  const treatPositions = new Map(
    treats.features.map((f) => [f.properties.id, f.geometry.coordinates] as const),
  );
  let closeUp = false;
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
    return active.flight;
  }

  function drawData(): void {
    const source = map.getSource("trips") as GeoJSONSource | undefined;
    if (!source) return;
    const current = flight();
    source.setData(
      tripFeatures(
        { ...city, nest: active.nest },
        current,
        step,
        returnFraction,
        colors(palette()),
      ),
    );
    // The single fly is shown while searching; on the way home the two networks' lines take over.
    if (returnFraction > 0) flyMarker.remove();
    else flyMarker.setLngLat(current.path[Math.min(step, current.path.length - 1)]).addTo(map);
    for (const { key, dot } of landingMarkers) {
      if (returnFraction >= 1) dot.setLngLat(current.returns[key].landing).addTo(map);
      else dot.remove();
    }
    const outboundTotal = current.path.length - 1;
    status.textContent =
      returnFraction > 0
        ? returnFraction >= 1
          ? "Back home — dotted lines show how far each landed from the nest"
          : "Flying home along the estimated home vector…"
        : `Searching · step ${Math.min(step, outboundTotal)} / ${outboundTotal}`;
    const label = playing ? "Pause" : step === 0 && returnFraction === 0 ? "Play" : "Replay";
    play.textContent = `${playing ? "❚❚" : "▶"} ${label}`;
    play.setAttribute("aria-label", label);
    play.setAttribute("aria-pressed", String(playing));
    zoomButton.hidden = returnFraction < 1;
  }

  function describe(): void {
    const current = flight();
    mapSummary.textContent =
      `${active.title} from the nest to ` +
      `${current.treat ? current.treat.name || `a ${current.treat.kind}` : "no treat"}; ` +
      NETS.map(
        ({ key, short }) =>
          `${short} lands ${current.returns[key].error_m.toFixed(0)} m from the nest`,
      ).join(", ") +
      ".";
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
    const treatAt = current.treat ? treatPositions.get(current.treat.id) : undefined;
    if (treatAt) treatMarker.setLngLat(treatAt).addTo(map);
    else treatMarker.remove();
    if (map.getLayer("treat-found")) {
      map.setFilter("treat-found", ["==", ["get", "id"], current.treat ? current.treat.id : -1]);
    }
    const rows = NETS.map(({ key, short }) =>
      el(
        "li",
        {},
        el("span", { className: `key-line key-${key}` }),
        el("span", { text: short }),
        el("strong", { className: "num", text: `${current.returns[key].error_m.toFixed(0)} m` }),
      ),
    );
    side.replaceChildren(
      el("h3", { text: active.title }),
      el(
        "div",
        { className: "stats" },
        stat(`${((current.path.length - 1) * city.homing.step_m).toFixed(0)} m`, "flown out"),
        stat(`${metres(active.nest, end).toFixed(0)} m`, "to the nest"),
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
    const framing =
      "Here the network reports only its heading; code adds up the steps into the home vector. ";
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
    const signed = (v: number): string => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
    const ci = `${signed(d.low)} to ${signed(d.high)} m`;
    let verdict: string;
    if (d.high < 0) {
      verdict =
        `The real wiring lands ${Math.abs(d.mean).toFixed(1)} m closer to the nest on average, ` +
        `and the 95% confidence interval of the difference (${ci}) lies below zero: a small ` +
        "but clear homing advantage.";
    } else if (d.low > 0) {
      verdict =
        `The real wiring lands ${d.mean.toFixed(1)} m further from the nest on average, and ` +
        `the 95% confidence interval of the difference (${ci}) lies above zero: the random ` +
        "wiring homes better.";
    } else {
      verdict =
        `The real wiring lands ${Math.abs(d.mean).toFixed(1)} m ${d.mean < 0 ? "closer to" : "further from"} ` +
        `the nest on average, but the 95% confidence interval of the difference (${ci}) ` +
        "includes zero: no measurable homing advantage.";
    }
    resultCard.replaceChildren(
      head,
      el("p", {
        className: "lede",
        text:
          framing +
          `Pre-registered test on fresh seeds ${s.seeds[0]}–${s.seeds[s.seeds.length - 1]} ` +
          `(${s.seeds.length} trained networks of each kind, 100 flights each). ${verdict}`,
      }),
      el(
        "p",
        { className: "note" },
        `Average distance from the nest: real wiring ${s.mean_error_m.FlyConnectome.mean.toFixed(1)} m, ` +
          `random wiring ${s.mean_error_m.RandomMask.mean.toFixed(1)} m. For comparison, a fly ` +
          `that keeps the heading it saw at the last landmark lands ${s.no_integration_error_m.mean.toFixed(0)} m ` +
          `away, and perfect compass answers (true sectors) ${s.quantised_error_m.mean.toFixed(0)} m. ` +
          `The real wiring's heading answers were slightly more accurate on these flights ` +
          `(${pct(s.heading_accuracy.FlyConnectome.mean)} vs ${pct(s.heading_accuracy.RandomMask.mean)}), ` +
          "but that did not turn into measurably closer landings. Negative differences mean the " +
          "real wiring lands closer.",
      ),
    );
  }

  function selectFlight(i: number): void {
    active = { nest: city.nest, flight: city.flights[i], title: `Flight ${i + 1}`, live: false };
    placeNest(city.nest);
    step = 0;
    returnFraction = 0;
    buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
    const path = city.flights[i].path;
    overview();
    closeUp = false;
    zoomButton.textContent = "Close-up of the nest";
    if (reduceMotion && !playing) {
      step = path.length - 1;
      returnFraction = 1;
    }
    drawSide();
    drawData();
    describe();
  }

  function overview(): void {
    const path = flight().path;
    const lons = [...path.map((p) => p[0]), active.nest[0]];
    const lats = [...path.map((p) => p[1]), active.nest[1]];
    map.fitBounds(
      [
        [Math.min(...lons), Math.min(...lats)],
        [Math.max(...lons), Math.max(...lats)],
      ],
      { padding: 70, maxZoom: 16.5, duration: reduceMotion ? 0 : 700 },
    );
  }

  /** Zoom to the nest and both landing points: errors are metres, the trip hundreds. */
  function nestCloseUp(): void {
    const points = [active.nest, ...NETS.map(({ key }) => flight().returns[key].landing)];
    const lons = points.map((p) => p[0]);
    const lats = points.map((p) => p[1]);
    map.fitBounds(
      [
        [Math.min(...lons), Math.min(...lats)],
        [Math.max(...lons), Math.max(...lats)],
      ],
      { padding: 90, maxZoom: 19.5, duration: reduceMotion ? 0 : 900 },
    );
  }

  zoomButton.addEventListener("click", () => {
    closeUp = !closeUp;
    zoomButton.textContent = closeUp ? "Show the whole trip" : "Close-up of the nest";
    if (closeUp) nestCloseUp();
    else overview();
  });

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
    else {
      playing = false;
      closeUp = true;
      zoomButton.textContent = "Show the whole trip";
      nestCloseUp();
    }
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
      if (closeUp) {
        closeUp = false;
        zoomButton.textContent = "Close-up of the nest";
        overview();
      }
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

  /* ---------- "Drop a fly": the trained networks run live in the browser ---------- */

  type Mode = "pilot" | "drop";
  type Kind = "pick" | "search";
  /** nest: waiting for a nest; treat: waiting for a treat; busy: computing; done: flight shown. */
  type Phase = "nest" | "treat" | "busy" | "done";
  let mode: Mode = "pilot";
  let kind: Kind = "pick";
  let phase: Phase = "nest";
  let dropNest: [number, number] | null = null;
  let runId = 0;
  let loadedOnce = false;
  const treatLonLat = treats.features.map((f) => f.geometry.coordinates as Point);
  const treatIndex = new Map(treats.features.map((f, i) => [f.properties.id, i]));
  /** A treat further than this cannot be reached within the step limit. */
  const MAX_TREAT_M = 1400;

  const modeButtons = (
    [
      ["pilot", "Pilot flights"],
      ["drop", "Drop a fly anywhere"],
    ] as [Mode, string][]
  ).map(([value, text]) => {
    const button = el("button", { text });
    button.type = "button";
    button.addEventListener("click", () => setMode(value));
    modeSwitch.append(button);
    return { value, button };
  });
  const kindButtons = (
    [
      ["pick", "flies to a treat you pick"],
      ["search", "searches on its own"],
    ] as [Kind, string][]
  ).map(([value, text]) => {
    const button = el("button", { text });
    button.type = "button";
    button.addEventListener("click", () => {
      kind = value;
      resetDrop();
    });
    dropKind.append(button);
    return { value, button };
  });
  randomButton.addEventListener("click", () => {
    const [[w, s], [e, n]] = MAPPED;
    const margin = 0.1;
    const lngLat: [number, number] = [
      w + (e - w) * (margin + (1 - 2 * margin) * Math.random()),
      s + (n - s) * (margin + (1 - 2 * margin) * Math.random()),
    ];
    kind = "search";
    resetDrop();
    dropNestAt(lngLat);
  });

  function setMode(next: Mode): void {
    mode = next;
    modeButtons.forEach(({ value, button }) =>
      button.setAttribute("aria-pressed", String(value === mode)),
    );
    pilotControls.hidden = mode !== "pilot";
    dropControls.hidden = mode !== "drop";
    dropNote.hidden = mode !== "drop";
    root.querySelectorAll<HTMLElement>(".pilot-note").forEach((n) => (n.hidden = mode !== "pilot"));
    mapBox.classList.toggle("is-picking", mode === "drop");
    if (map.getLayer("mapped-edge")) {
      map.setLayoutProperty("mapped-edge", "visibility", mode === "drop" ? "visible" : "none");
    }
    runId++;
    if (mode === "pilot") {
      play.hidden = false;
      selectFlight(0);
      if (!reduceMotion) start();
    } else {
      resetDrop();
      map.fitBounds(MAPPED, { padding: 30, duration: reduceMotion ? 0 : 700 });
    }
  }

  function setBusy(busy: boolean): void {
    mapBox.setAttribute("aria-busy", String(busy));
    if (!busy) dropProgress.textContent = "";
  }

  function resetDrop(): void {
    runId++;
    phase = "nest";
    playing = false;
    cancelAnimationFrame(frame);
    setBusy(false);
    dropNest = null;
    active = { ...active, live: false };
    kindButtons.forEach(({ value, button }) =>
      button.setAttribute("aria-pressed", String(value === kind)),
    );
    nestDot.remove();
    nestLabel.remove();
    flyMarker.remove();
    treatMarker.remove();
    landingMarkers.forEach(({ dot }) => dot.remove());
    (map.getSource("trips") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection",
      features: [],
    });
    if (map.getLayer("treat-found")) map.setFilter("treat-found", ["==", ["get", "id"], -1]);
    zoomButton.hidden = true;
    status.textContent = "";
    play.hidden = true;
    mapSummary.textContent = "No fly dropped yet.";
    dropHint.textContent =
      "Click anywhere inside the dashed frame to place the nest" +
      (kind === "pick" ? ", then click a treat (grey dot) for the fly to visit." : ".");
    side.replaceChildren(
      el("h3", { text: "Your fly" }),
      el("p", {
        text:
          "Two trained networks — real fly wiring and random wiring, the ones in the Replay view — " +
          "run here in your browser (ONNX Runtime Web). The first fly downloads them once (about " +
          `${DOWNLOAD_MB_COMPRESSED} MB compressed).`,
      }),
    );
  }

  function inside([lng, lat]: [number, number]): boolean {
    const [[w, s], [e, n]] = MAPPED;
    return lng >= w && lng <= e && lat >= s && lat <= n;
  }

  /** Place the nest; a searching fly starts at once, a picking fly waits for a treat. */
  function dropNestAt(lngLat: [number, number]): void {
    dropNest = lngLat;
    placeNest(lngLat);
    if (kind === "search") void launch(-1);
    else {
      phase = "treat";
      dropHint.textContent =
        "Now click a treat (grey dot) for the fly to visit — or click elsewhere to move the nest.";
    }
  }

  /** The treat under the pointer (a small box around it, so the dots are easy to hit). */
  function treatAt(point: { x: number; y: number }): number {
    const box = 12;
    const hits = map.queryRenderedFeatures(
      [
        [point.x - box, point.y - box],
        [point.x + box, point.y + box],
      ],
      { layers: ["treats"] },
    );
    let best = -1;
    let bestD = Infinity;
    for (const hit of hits) {
      const index = treatIndex.get(hit.properties?.id as number);
      if (index === undefined) continue;
      const p = map.project(treatLonLat[index]);
      const d = Math.hypot(p.x - point.x, p.y - point.y);
      if (d < bestD) [best, bestD] = [index, d];
    }
    return best;
  }

  async function launch(target: number): Promise<void> {
    if (!dropNest) return;
    const nest = dropNest;
    const id = ++runId;
    phase = "busy";
    play.hidden = true;
    setBusy(true);
    dropHint.textContent = loadedOnce
      ? "Computing the flight…"
      : `Loading the two networks (about ${DOWNLOAD_MB_COMPRESSED} MB compressed, once)…`;
    const fail = (error: unknown): void => {
      if (id !== runId || !scope.active) return;
      console.error(error);
      setBusy(false);
      phase = "nest";
      dropHint.textContent =
        "Sorry — the networks could not be run in this browser. Check the connection and click " +
        "the map to try again.";
    };
    let loaded: Networks;
    try {
      loaded = await getNetworks(import.meta.env.BASE_URL);
      loadedOnce = true;
    } catch (error) {
      fail(error);
      return;
    }
    if (id !== runId || !scope.active) return;
    dropHint.textContent = "Computing the flight…";
    const seed = Math.floor(Math.random() * 2 ** 31);
    const local = treatLonLat.map((t) => toLocal(t, nest));
    const out = flyOut(local, seed, target);
    const steps = out.inputs.length;
    let results: number[][] | null;
    try {
      results = await runSectors(
        loaded,
        out.inputs,
        (done, total) => {
          if (id === runId)
            dropProgress.textContent = `Heading estimates: step ${done} of ${total}`;
        },
        () => id !== runId || !scope.active,
      );
    } catch (error) {
      fail(error);
      return;
    }
    if (id !== runId || !scope.active || results === null) return;
    const predictedBy = results;
    const returns = Object.fromEntries(
      loaded.nets.map((net, i) => {
        const predicted = predictedBy[i];
        const land = landingPoint(out.path, predicted.map(sectorCentre));
        return [
          net.name,
          {
            landing: toLonLat(land, nest),
            error_m: Math.round(distance(land) * 10) / 10,
            predicted,
          },
        ];
      }),
    ) as CityFlight["returns"];
    const reached = out.treat >= 0 ? treats.features[out.treat].properties : null;
    setBusy(false);
    phase = "done";
    active = {
      nest,
      title: `Your fly (seed ${seed})`,
      live: true,
      flight: {
        path: out.path.map((p) => toLonLat(p, nest)),
        sectors: out.theta.map(headingSector),
        treat: reached,
        returns,
      },
    };
    dropHint.textContent =
      (reached
        ? `Found ${reached.name || `a ${reached.kind}`} after ${steps} steps (${steps * FLIGHT.stepM} m).`
        : `No treat within ${FLIGHT.maxSteps} steps — flying home anyway.`) +
      " Click the map to drop another fly.";
    play.hidden = false;
    zoomButton.hidden = false;
    step = 0;
    returnFraction = 0;
    closeUp = false;
    zoomButton.textContent = "Close-up of the nest";
    overview();
    drawSide();
    describe();
    if (reduceMotion) {
      step = out.path.length - 1;
      returnFraction = 1;
      drawData();
    } else start();
  }

  map.on("click", (event) => {
    if (mode !== "drop") return;
    if (phase === "busy") {
      dropHint.textContent = "Still computing this fly — one moment…";
      return;
    }
    const lngLat: [number, number] = [event.lngLat.lng, event.lngLat.lat];
    if (phase === "treat" && dropNest) {
      const index = treatAt(event.point);
      if (index >= 0) {
        if (metres(dropNest, treatLonLat[index]) > MAX_TREAT_M) {
          dropHint.textContent =
            `That treat is more than ${(MAX_TREAT_M / 1000).toFixed(1)} km from the nest — the ` +
            "fly would run out of steps. Pick a closer one.";
          return;
        }
        if (map.getLayer("treat-found")) {
          map.setFilter("treat-found", ["==", ["get", "id"], treats.features[index].properties.id]);
        }
        void launch(index);
        return;
      }
    }
    if (!inside(lngLat)) {
      dropHint.textContent = "Place the nest inside the dashed frame (where the map is drawn).";
      return;
    }
    if (phase === "done" || phase === "treat") resetDrop();
    dropNestAt(lngLat);
  });

  const keyItem = (sample: HTMLElement, text: string): HTMLElement =>
    el("span", { className: "key-item" }, sample, el("span", { text }));
  legend.replaceChildren(
    keyItem(el("span", { className: "key-dot key-nest" }), "nest"),
    keyItem(el("span", { className: "key-dot key-treat" }), "treats (OpenStreetMap)"),
    keyItem(el("span", { className: "key-line key-outbound" }), "search flight"),
    keyItem(el("span", { className: "key-line key-FlyConnectome" }), "home, real wiring (●)"),
    keyItem(el("span", { className: "key-line key-RandomMask" }), "home, random wiring (▲)"),
    keyItem(el("span", { className: "key-line key-error" }), "distance left to the nest"),
  );

  map.on("error", (event) => {
    const message = String(event.error?.message ?? "");
    if (/pmtiles|tile|fetch|range/i.test(message)) {
      mapError.textContent = "The map tiles could not be loaded; the flights are still listed.";
      mapError.hidden = false;
    }
  });

  map.on("load", () => {
    (map.getSource("treats") as GeoJSONSource).setData(treats as unknown as FeatureCollection);
    setMode("pilot");
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
    map.setPaintProperty("treats", "circle-opacity", isDark() ? 0.9 : 0.7);
    map.setPaintProperty("treat-found", "circle-color", p.ink);
    map.setPaintProperty("treat-found", "circle-stroke-color", p.surface);
    map.setPaintProperty("outbound", "line-color", p.ink);
    map.setPaintProperty("mapped-edge", "line-color", p.muted);
    if (mode === "drop" && !active.live) return; // nothing drawn yet
    drawSide();
    stepView();
  };
  const stopScheme = onScheme(restyle);
  drawResult();
  if (city.homevector) {
    homeVectorCard(root, city.homevector, scope, city.summary?.mean_error_m.FlyConnectome.mean);
  }
  scope.onLeave(() => {
    runId++;
    cancelAnimationFrame(frame);
    stopScheme();
    map.remove();
  });
}
