/** Geometry and scoring helpers for the heading replay (pure functions, unit-tested). */

export interface Point {
  x: number;
  y: number;
}

/** Point on a circle; angle 0 points right and angles grow counter-clockwise (maths convention). */
export function polar(cx: number, cy: number, r: number, angle: number): Point {
  return { x: cx + r * Math.cos(angle), y: cy - r * Math.sin(angle) };
}

/** SVG path of the ring segment for one heading sector (sector k spans k·w … (k+1)·w). */
export function sectorPath(
  cx: number,
  cy: number,
  inner: number,
  outer: number,
  sector: number,
  sectors: number,
): string {
  const width = (2 * Math.PI) / sectors;
  const a0 = sector * width;
  const a1 = a0 + width;
  const p0 = polar(cx, cy, outer, a0);
  const p1 = polar(cx, cy, outer, a1);
  const p2 = polar(cx, cy, inner, a1);
  const p3 = polar(cx, cy, inner, a0);
  const f = (n: number): string => n.toFixed(2);
  return (
    `M${f(p0.x)},${f(p0.y)} A${outer},${outer} 0 0 0 ${f(p1.x)},${f(p1.y)} ` +
    `L${f(p2.x)},${f(p2.y)} A${inner},${inner} 0 0 1 ${f(p3.x)},${f(p3.y)} Z`
  );
}

/** Hits and scored steps up to and including `step`. */
export function runningScore(
  predicted: number[],
  truth: number[],
  scored: boolean[],
  step: number,
): { hits: number; total: number } {
  let hits = 0;
  let total = 0;
  for (let t = 0; t <= step && t < scored.length; t += 1) {
    if (!scored[t]) continue;
    total += 1;
    if (predicted[t] === truth[t]) hits += 1;
  }
  return { hits, total };
}

/** Steps since the last landmark (0 when a landmark is visible now). */
export function stepsSinceLandmark(landmark: boolean[], step: number): number {
  for (let t = step; t >= 0; t -= 1) if (landmark[t]) return step - t;
  return step + 1;
}
