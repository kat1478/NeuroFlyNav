/**
 * Foraging flights for the browser demo — a TypeScript port of `neuroflynav.tasks.homing`
 * and the heading task's input encoding, so the exported networks see the inputs they were
 * trained on. Local metres around the nest: x east, y north; angles counter-clockwise from east.
 */
import { Rng } from "./rng";

export const FLIGHT = {
  stepM: 5,
  detectM: 15,
  maxSteps: 300,
  cueSteps: 3,
  landmarkEvery: 8,
  turnStd: 0.15,
  persistence: 0.8,
  sectors: 8,
} as const;

export type Point = [number, number];

export interface Flight {
  /** Network input per step: bias, turn rate / turnStd, landmark flag, cos θ, sin θ (if visible). */
  inputs: number[][];
  theta: number[];
  /** Positions in metres, path[0] = nest. */
  path: Point[];
  /** Index of the treat reached, or -1. */
  treat: number;
}

const M_PER_DEG_LAT = 110_540;
const M_PER_DEG_LON_EQUATOR = 111_320;

export function toLocal(lonLat: Point, origin: Point): Point {
  const scale = M_PER_DEG_LON_EQUATOR * Math.cos((origin[1] * Math.PI) / 180);
  return [(lonLat[0] - origin[0]) * scale, (lonLat[1] - origin[1]) * M_PER_DEG_LAT];
}

export function toLonLat(xy: Point, origin: Point): Point {
  const scale = M_PER_DEG_LON_EQUATOR * Math.cos((origin[1] * Math.PI) / 180);
  return [origin[0] + xy[0] / scale, origin[1] + xy[1] / M_PER_DEG_LAT];
}

export const landmarkVisible = (step: number): boolean =>
  step < FLIGHT.cueSteps || step % FLIGHT.landmarkEvery === 0;

const wrap = (a: number): number => Math.atan2(Math.sin(a), Math.cos(a));

/**
 * Fly from the nest until a treat is within `detectM` (or `maxSteps`).
 * `target` = index of a chosen treat: turns are gently biased towards it (and only that treat
 * ends the flight); otherwise the fly searches like in the experiment and stops at any treat.
 * The turn rate stays within ±3 standard deviations of the training distribution.
 */
export function flyOut(treats: Point[], seed: number, target = -1): Flight {
  const rng = new Rng(seed);
  const innovation = FLIGHT.turnStd * Math.sqrt(1 - FLIGHT.persistence ** 2);
  const limit = 3 * FLIGHT.turnStd;
  let omega = rng.normal() * FLIGHT.turnStd;
  let theta = rng.next() * 2 * Math.PI;
  const inputs: number[][] = [];
  const thetas: number[] = [];
  const path: Point[] = [[0, 0]];
  let found = -1;
  for (let step = 0; step < FLIGHT.maxSteps && found < 0; step++) {
    if (step > 0) omega = FLIGHT.persistence * omega + rng.normal() * innovation;
    if (target >= 0) {
      const [x, y] = path[path.length - 1];
      const bearing = Math.atan2(treats[target][1] - y, treats[target][0] - x);
      omega += 0.08 * wrap(bearing - (theta + omega));
    }
    omega = Math.max(-limit, Math.min(limit, omega));
    theta += omega;
    const visible = landmarkVisible(step);
    inputs.push([
      1,
      omega / FLIGHT.turnStd,
      visible ? 1 : 0,
      visible ? Math.cos(theta) : 0,
      visible ? Math.sin(theta) : 0,
    ]);
    thetas.push(theta);
    const [x, y] = path[path.length - 1];
    const next: Point = [x + FLIGHT.stepM * Math.cos(theta), y + FLIGHT.stepM * Math.sin(theta)];
    path.push(next);
    const candidates = target >= 0 ? [target] : treats.map((_, i) => i);
    for (const i of candidates) {
      if (Math.hypot(treats[i][0] - next[0], treats[i][1] - next[1]) <= FLIGHT.detectM) {
        found = i;
        break;
      }
    }
  }
  return { inputs, theta: thetas, path, treat: found };
}

export const sectorCentre = (sector: number): number =>
  ((sector + 0.5) * 2 * Math.PI) / FLIGHT.sectors;

/** Where the fly lands after flying back along the home vector built from these headings. */
export function landing(path: Point[], headings: number[]): Point {
  let hx = 0;
  let hy = 0;
  for (const h of headings) {
    hx += FLIGHT.stepM * Math.cos(h);
    hy += FLIGHT.stepM * Math.sin(h);
  }
  const end = path[path.length - 1];
  return [end[0] - hx, end[1] - hy];
}

export const distance = (p: Point): number => Math.hypot(p[0], p[1]);
