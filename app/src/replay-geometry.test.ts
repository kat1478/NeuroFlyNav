import { describe, expect, it } from "vitest";
import { polar, runningScore, sectorPath, stepsSinceLandmark } from "./replay-geometry";

describe("replay geometry", () => {
  it("uses counter-clockwise angles with y pointing down on screen", () => {
    const top = polar(0, 0, 10, Math.PI / 2);
    expect(top.x).toBeCloseTo(0);
    expect(top.y).toBeCloseTo(-10);
  });

  it("draws a closed ring segment", () => {
    const path = sectorPath(50, 50, 20, 40, 0, 8);
    expect(path.startsWith("M90.00,50.00")).toBe(true);
    expect(path.endsWith("Z")).toBe(true);
  });

  it("scores only scored steps up to the playhead", () => {
    const scored = [false, true, true, true];
    expect(runningScore([0, 1, 2, 9], [5, 1, 2, 3], scored, 2)).toEqual({ hits: 2, total: 2 });
    expect(runningScore([0, 1, 2, 9], [5, 1, 2, 3], scored, 3)).toEqual({ hits: 2, total: 3 });
  });

  it("counts steps since the last landmark", () => {
    const landmark = [true, false, false, true, false];
    expect(stepsSinceLandmark(landmark, 2)).toBe(2);
    expect(stepsSinceLandmark(landmark, 3)).toBe(0);
  });
});
