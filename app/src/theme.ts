/** Chart colours read from the CSS tokens, so charts follow light/dark mode. */

export type Series =
  "FlyConnectome" | "ShuffledFly" | "RandomRecurrent" | "TypeShuffledFly" | "FrozenFly";

export interface Palette {
  surface: string;
  raised: string;
  ink: string;
  ink2: string;
  muted: string;
  grid: string;
  baseline: string;
  rampLo: string;
  rampHi: string;
  series: Record<Series, string>;
}

/** Readable names; identity is also carried by marker shape and legend, never colour alone. */
export const SERIES_LABEL: Record<Series, string> = {
  FlyConnectome: "real CX wiring",
  ShuffledFly: "degree-shuffled wiring",
  RandomRecurrent: "random wiring (same size)",
  TypeShuffledFly: "shuffled within cell types",
  FrozenFly: "real wiring, frozen (not trained)",
};

export const SERIES_SYMBOL: Record<Series, string> = {
  FlyConnectome: "circle",
  ShuffledFly: "square",
  RandomRecurrent: "triangle",
  TypeShuffledFly: "diamond",
  FrozenFly: "times",
};

export function palette(): Palette {
  const style = getComputedStyle(document.documentElement);
  const v = (name: string): string => style.getPropertyValue(name).trim();
  return {
    surface: v("--surface"),
    raised: v("--raised"),
    ink: v("--ink"),
    ink2: v("--ink-2"),
    muted: v("--muted"),
    grid: v("--grid"),
    baseline: v("--baseline"),
    rampLo: v("--ramp-lo"),
    rampHi: v("--ramp-hi"),
    series: {
      FlyConnectome: v("--fly"),
      ShuffledFly: v("--shuffled"),
      RandomRecurrent: v("--random"),
      TypeShuffledFly: v("--typeshuffled"),
      FrozenFly: v("--muted"),
    },
  };
}

/** Call `render` now and again whenever the colour scheme changes; returns an unsubscribe. */
export function onScheme(render: () => void): () => void {
  const query = matchMedia("(prefers-color-scheme: dark)");
  render();
  query.addEventListener("change", render);
  return () => query.removeEventListener("change", render);
}
