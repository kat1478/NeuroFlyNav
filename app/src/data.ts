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

async function load<T>(name: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${name}.json`);
  if (!response.ok) throw new Error(`could not load ${name}.json (${response.status})`);
  return (await response.json()) as T;
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
