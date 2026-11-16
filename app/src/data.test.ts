import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { scoredAccuracy, type Replay, type Results, type Wiring } from "./data";
import { viewFromHash } from "./main";

const read = <T>(name: string): T =>
  JSON.parse(readFileSync(resolve("public/data", `${name}.json`), "utf8")) as T;

describe("app data", () => {
  it("has the five forest rows and four per-seed experiments", () => {
    const results = read<Results>("results");
    expect(results.forest).toHaveLength(5);
    expect(results.per_seed.map((e) => e.seeds.length)).toEqual([20, 20, 20, 20]);
  });

  it("wiring neurons add up to the CX total", () => {
    const wiring = read<Wiring>("wiring");
    expect(wiring.neurons.reduce((a, b) => a + b, 0)).toBe(wiring.total_neurons);
    expect(wiring.matrix).toHaveLength(wiring.labels.length);
  });

  it("replay accuracy on the shown sequences is consistent with the predictions", () => {
    const replay = read<Replay>("replay");
    const fly = replay.variants.FlyConnectome;
    replay.sequences.forEach((sequence, i) => {
      const accuracy = scoredAccuracy(fly.predicted[i], sequence.sector, replay.scored);
      expect(accuracy).toBeGreaterThanOrEqual(0);
      expect(accuracy).toBeLessThanOrEqual(1);
    });
  });
});

describe("routing", () => {
  it("falls back to results for unknown hashes", () => {
    expect(viewFromHash("#/wiring")).toBe("wiring");
    expect(viewFromHash("#/nope")).toBe("results");
    expect(viewFromHash("")).toBe("results");
  });
});
