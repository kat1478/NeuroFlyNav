/** Types and loaders for the JSON files written by `neuroflynav.reporting.app_data`. */

export type Control = "ShuffledFly" | "TypeShuffledFly" | "RandomRecurrent";
export type Interval = [low: number, mean: number, high: number];

export interface ForestRow {
  label: string;
  diff: Record<Control, Interval>;
}

export interface PerSeedExperiment {
  id: string;
  label: string;
  seeds: number[];
  accuracy: Record<string, number>[];
}

export interface Results {
  _source: string;
  forest: ForestRow[];
  progression: { label: string; FlyConnectome: number; ShuffledFly: number }[];
  learning: { sizes: number[] } & Record<string, number[]>;
  references: Record<string, unknown>;
  per_seed: PerSeedExperiment[];
}

export interface Wiring {
  _source: string;
  labels: string[];
  matrix: number[][];
  neurons: number[];
  total_neurons: number;
  total_synapses: number;
}

export interface ReplayVariant {
  accuracy: number;
  predicted: number[][];
}

export interface Replay {
  graph: string;
  seed: number;
  nodes: number;
  edges: number;
  sectors: number;
  sector_width: number;
  steps: number;
  landmark: boolean[];
  scored: boolean[];
  sequences: { theta: number[]; sector: number[] }[];
  variants: Record<"FlyConnectome" | "RandomMask", ReplayVariant>;
}

const cache = new Map<string, Promise<unknown>>();

/** Fetch a data file once; later calls reuse the same promise (a failed fetch is retried). */
function load<T>(name: string): Promise<T> {
  let promise = cache.get(name) as Promise<T> | undefined;
  if (!promise) {
    const file = name === "treats" ? "treats.geojson" : `${name}.json`;
    promise = fetch(`${import.meta.env.BASE_URL}data/${file}`).then(async (response) => {
      if (!response.ok) throw new Error(`could not load ${file} (${response.status})`);
      return (await response.json()) as T;
    });
    promise.catch(() => cache.delete(name));
    cache.set(name, promise);
  }
  return promise;
}

export const loadResults = (): Promise<Results> => load<Results>("results");
export const loadWiring = (): Promise<Wiring> => load<Wiring>("wiring");
export const loadReplay = (): Promise<Replay> => load<Replay>("replay");

/** Fraction of scored steps where `predicted` equals the true sector. */
export function scoredAccuracy(predicted: number[], truth: number[], scored: boolean[]): number {
  let hits = 0;
  let total = 0;
  scored.forEach((counts, step) => {
    if (!counts) return;
    total += 1;
    if (predicted[step] === truth[step]) hits += 1;
  });
  return total === 0 ? 0 : hits / total;
}

export interface CityReturn {
  landing: [number, number];
  error_m: number;
  predicted: number[];
}

export interface CityFlight {
  path: [number, number][];
  sectors: number[];
  treat: { id: number; kind: string; name: string } | null;
  returns: Record<"FlyConnectome" | "RandomMask", CityReturn>;
}

interface Estimate {
  mean: number;
  low: number;
  high: number;
}

export interface City {
  _source: string;
  seed: number;
  nest: [number, number];
  homing: { step_m: number; detect_m: number; max_steps: number; landmark_every: number };
  pilot: Record<
    "FlyConnectome" | "RandomMask",
    { heading_accuracy: number; error_m: { mean: number; median: number }; flights: number }
  >;
  references_m: { quantised: { mean: number }; no_integration: { mean: number } };
  summary: {
    seeds: number[];
    mean_error_m: Record<string, Estimate>;
    heading_accuracy: Record<string, Estimate>;
    fly_minus_error_m: Record<string, Estimate>;
    quantised_error_m: Estimate;
    no_integration_error_m: Estimate;
  } | null;
  flights: CityFlight[];
  homevector?: HomeVector;
}

type HomeVectorVariant =
  "FlyConnectome" | "TypeShuffledFly" | "RandomMask" | "ShuffledFly" | "FrozenFly";

/** Pre-registered home-vector result (the network reports the home vector itself). */
export interface HomeVector {
  _source: string;
  seeds: number[];
  mean_error_m: Record<HomeVectorVariant, Estimate>;
  mean_along_error_m: Record<HomeVectorVariant, Estimate>;
  mean_synthetic_error_m: Record<HomeVectorVariant, Estimate>;
  fly_minus_error_m: Record<Exclude<HomeVectorVariant, "FlyConnectome">, Estimate>;
  stay_error_m: Estimate;
  quantised_error_m: Estimate;
  no_integration_error_m: Estimate;
}

export interface TreatFeature {
  type: "Feature";
  properties: { id: number; kind: string; name: string };
  geometry: { type: "Point"; coordinates: [number, number] };
}

export const loadCity = (): Promise<City> => load<City>("city");
export const loadTreats = (): Promise<{ type: "FeatureCollection"; features: TreatFeature[] }> =>
  load("treats");
