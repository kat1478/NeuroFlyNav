import { describe, expect, it } from "vitest";
import { FLIGHT, flyOut, landing, landmarkVisible, toLocal, toLonLat, type Point } from "./flight";

describe("flight", () => {
  it("projects to metres and back", () => {
    const origin: Point = [21.0035, 52.2357];
    const back = toLonLat(toLocal([21.02, 52.24], origin), origin);
    expect(back[0]).toBeCloseTo(21.02, 9);
    expect(back[1]).toBeCloseTo(52.24, 9);
    expect(toLocal([0, 0.01], [0, 0])[1]).toBeCloseTo(1105.4, 1);
  });

  it("shows the heading on cue and landmark steps only, like the training task", () => {
    expect([0, 1, 2, 3, 7, 8, 16].map(landmarkVisible)).toEqual([
      true,
      true,
      true,
      false,
      false,
      true,
      true,
    ]);
    const f = flyOut([], 7);
    f.inputs.forEach((row, step) => {
      expect(row[0]).toBe(1);
      expect(row[2]).toBe(landmarkVisible(step) ? 1 : 0);
      if (!row[2]) expect([row[3], row[4]]).toEqual([0, 0]);
    });
  });

  it("is reproducible and comes home exactly with true headings", () => {
    const a = flyOut([], 3);
    expect(flyOut([], 3)).toEqual(a);
    expect(a.path).toHaveLength(FLIGHT.maxSteps + 1);
    const end = landing(a.path, a.theta);
    expect(Math.hypot(end[0], end[1])).toBeLessThan(1e-6);
  });

  it("stops at a treat and, with a target, only at that treat", () => {
    const probe = flyOut([], 11);
    const near = probe.path[30];
    const treats: Point[] = [[near[0] + 3, near[1]]];
    expect(flyOut(treats, 11).treat).toBe(0);
    const goal: Point = [400, 0];
    const f = flyOut([[1e6, 1e6], goal], 5, 1);
    if (f.treat === 1) {
      const end = f.path[f.path.length - 1];
      expect(Math.hypot(end[0] - goal[0], end[1] - goal[1])).toBeLessThanOrEqual(FLIGHT.detectM);
    }
    f.inputs.forEach((row) => expect(Math.abs(row[1])).toBeLessThanOrEqual(3 + 1e-9));
  });
});
