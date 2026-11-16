/** Short descriptions of the plotted cell-type families (after Hulse et al., 2021, eLife). */

export const FAMILY_INFO: Record<string, { name: string; role: string }> = {
  ER: {
    name: "Ring neurons (ER)",
    role: "Bring sensory input — visual landmarks, sky polarisation, wind — into the ellipsoid body. The main inputs of the compass.",
  },
  ExR: {
    name: "Extrinsic ring neurons (ExR)",
    role: "Connect the ellipsoid body with other brain regions; several types are thought to be modulatory.",
  },
  EPG: {
    name: "Compass neurons (E-PG)",
    role: "Their activity forms a 'bump' around the ellipsoid body whose position tracks the fly's heading.",
  },
  PEN: {
    name: "Shift neurons (P-EN)",
    role: "Receive angular-velocity input and push the heading bump when the fly turns.",
  },
  PEG: {
    name: "P-EG neurons",
    role: "Close another loop from the protocerebral bridge back to the ellipsoid body; part of the recurrent compass circuit.",
  },
  Δ7: {
    name: "Δ7 neurons",
    role: "Inhibitory interneurons of the protocerebral bridge, thought to shape the heading bump into a sinusoid-like profile.",
  },
  EL: {
    name: "EL neurons",
    role: "A small group linking the ellipsoid body to the lateral accessory lobe, a region close to motor output.",
  },
  PFN: {
    name: "P-FN neurons",
    role: "Carry heading together with self-motion signals from the bridge into the fan-shaped body.",
  },
  PFG: {
    name: "P-FG neurons",
    role: "Columnar neurons linking the protocerebral bridge, the fan-shaped body and the gall.",
  },
  PFR: {
    name: "P-FR neurons",
    role: "Columnar neurons linking the protocerebral bridge, the fan-shaped body and the round body.",
  },
  PFL: {
    name: "Steering neurons (P-FL)",
    role: "Output neurons to premotor regions; thought to compare current heading with a goal direction to steer.",
  },
  hΔ: {
    name: "hΔ neurons",
    role: "Columnar interneurons within the fan-shaped body connecting distant columns.",
  },
  vΔ: {
    name: "vΔ neurons",
    role: "Columnar interneurons within the fan-shaped body connecting its layers.",
  },
  FC: {
    name: "FC neurons",
    role: "Fan-shaped body columnar neurons projecting to the crepine region.",
  },
  FS: {
    name: "FS neurons",
    role: "Fan-shaped body columnar neurons projecting to the superior protocerebrum — a large output family.",
  },
  FR: {
    name: "FR neurons",
    role: "Fan-shaped body columnar neurons projecting to the round body.",
  },
  FB: {
    name: "Fan-shaped body tangential neurons (FB)",
    role: "Span whole layers of the fan-shaped body and bring context from the rest of the brain.",
  },
  SA: {
    name: "SA neurons",
    role: "A small group of neuron types connecting the fan-shaped body with neighbouring regions.",
  },
  other: {
    name: "Other neurons",
    role: "Neurons with synapses in the CX that belong to none of the families above (or have no cell-type name).",
  },
};

/** Families grouped by where they sit in the circuit, for the glossary. */
export const FAMILY_GROUPS: { title: string; families: string[] }[] = [
  { title: "Inputs to the compass (ellipsoid body)", families: ["ER", "ExR"] },
  { title: "Heading compass", families: ["EPG", "PEN", "PEG", "Δ7"] },
  { title: "Towards motor output", families: ["EL", "PFL"] },
  {
    title: "Fan-shaped body (context, goals, vectors)",
    families: ["PFN", "PFG", "PFR", "hΔ", "vΔ", "FC", "FS", "FR", "FB", "SA"],
  },
  { title: "Everything else", families: ["other"] },
];

/** Brain regions named in the family descriptions. */
export const REGIONS: [string, string][] = [
  ["Central complex (CX)", "a set of midline brain regions that insects use for navigation"],
  ["Ellipsoid body (EB)", "a ring-shaped region where the heading 'compass' lives"],
  ["Protocerebral bridge (PB)", "a handlebar-shaped region; a second map of heading"],
  ["Fan-shaped body (FB)", "a layered region that combines heading with context and goals"],
  [
    "Gall, round body, lateral accessory lobe",
    "nearby regions linking the CX to the rest of the brain and to motor control",
  ],
];

export const familyName = (label: string): string => FAMILY_INFO[label]?.name ?? label;

export type Mode = "synapses" | "share";

export interface Partner {
  family: string;
  synapses: number;
  share: number;
}

/** Synapses from row family to column family, or the share of the row's total output. */
export function cellValue(matrix: number[][], source: number, target: number, mode: Mode): number {
  const value = matrix[source][target];
  if (mode === "synapses") return value;
  const total = matrix[source].reduce((a, b) => a + b, 0);
  return total === 0 ? 0 : value / total;
}

/** The `count` strongest output (or input) partners of one family. */
export function topPartners(
  labels: string[],
  matrix: number[][],
  index: number,
  direction: "out" | "in",
  count = 3,
): Partner[] {
  const values = labels.map((_, j) => (direction === "out" ? matrix[index][j] : matrix[j][index]));
  const total = values.reduce((a, b) => a + b, 0);
  return values
    .map((synapses, j) => ({ family: labels[j], synapses, share: total ? synapses / total : 0 }))
    .sort((a, b) => b.synapses - a.synapses)
    .slice(0, count);
}
