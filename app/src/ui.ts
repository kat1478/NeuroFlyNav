/** Small DOM helpers shared by the views. Text always goes in via textContent. */

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  options: { className?: string; text?: string } = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  node.append(...children);
  return node;
}

export interface Column<T> {
  label: string;
  value: (row: T) => string;
  numeric?: boolean;
}

export function table<T>(rows: T[], columns: Column<T>[], caption: string): HTMLTableElement {
  const head = el(
    "tr",
    {},
    ...columns.map((c) => el("th", { className: c.numeric ? "num" : "", text: c.label })),
  );
  const body = rows.map((row) =>
    el(
      "tr",
      {},
      ...columns.map((c) => el("td", { className: c.numeric ? "num" : "", text: c.value(row) })),
    ),
  );
  return el(
    "table",
    { className: "data-table" },
    el("caption", { text: caption }),
    el("thead", {}, head),
    el("tbody", {}, ...body),
  );
}

let tableCount = 0;

/** A chart card: title, short explanation, chart slot and a "show table" toggle. */
export function card(
  title: string,
  description: string,
): { root: HTMLElement; chart: HTMLElement; tableSlot: HTMLElement; legend: HTMLElement } {
  const chart = el("div", { className: "chart" });
  const legend = el("div", { className: "legend" });
  const tableSlot = el("div", { className: "table-slot" });
  tableSlot.hidden = true;
  tableSlot.id = `table-${++tableCount}`;
  const toggle = el("button", { className: "link-button", text: "Show table" });
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-controls", tableSlot.id);
  toggle.addEventListener("click", () => {
    tableSlot.hidden = !tableSlot.hidden;
    toggle.textContent = tableSlot.hidden ? "Show table" : "Hide table";
    toggle.setAttribute("aria-expanded", String(!tableSlot.hidden));
  });
  const root = el(
    "section",
    { className: "card" },
    el("h2", { text: title }),
    el("p", { className: "note", text: description }),
    legend,
    chart,
    toggle,
    tableSlot,
  );
  return { root, chart, tableSlot, legend };
}

/** Legend items mirror the mark: a marker glyph in the series colour, the name in ink. */
export function legendItem(name: string, color: string, glyph: string): HTMLElement {
  const swatch = el("span", { className: "legend-glyph", text: glyph });
  swatch.style.color = color;
  return el("span", { className: "legend-item" }, swatch, el("span", { text: name }));
}

export const GLYPH: Record<string, string> = {
  circle: "●",
  square: "■",
  triangle: "▲",
  diamond: "◆",
  times: "✕",
};

export const pct = (value: number, digits = 1): string => `${(100 * value).toFixed(digits)}%`;
export const points = (value: number, digits = 1): string =>
  `${value >= 0 ? "+" : "−"}${Math.abs(100 * value).toFixed(digits)}`;
