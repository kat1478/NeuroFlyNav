import { describe, expect, it } from "vitest";
import { cellValue, FAMILY_INFO, topPartners } from "./families";

const labels = ["A", "B", "C"];
const matrix = [
  [0, 30, 10],
  [5, 0, 0],
  [1, 2, 3],
];

describe("families", () => {
  it("normalises by the source row", () => {
    expect(cellValue(matrix, 0, 1, "synapses")).toBe(30);
    expect(cellValue(matrix, 0, 1, "share")).toBeCloseTo(0.75);
  });

  it("ranks partners in both directions", () => {
    expect(topPartners(labels, matrix, 0, "out", 2).map((p) => p.family)).toEqual(["B", "C"]);
    expect(topPartners(labels, matrix, 0, "in", 1)[0]).toEqual({
      family: "B",
      synapses: 5,
      share: 5 / 6,
    });
  });

  it("describes every family in the committed wiring data", async () => {
    const { readFileSync } = await import("node:fs");
    const wiring = JSON.parse(readFileSync("public/data/wiring.json", "utf8")) as {
      labels: string[];
    };
    for (const label of wiring.labels) expect(FAMILY_INFO[label]).toBeDefined();
  });
});
