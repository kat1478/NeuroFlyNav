import "./styles.css";
import { leave } from "./lifecycle";
import { renderReplay } from "./views/replay";
import { renderResults } from "./views/results";
import { renderWiring } from "./views/wiring";

const views: Record<string, (root: HTMLElement) => Promise<void>> = {
  results: renderResults,
  wiring: renderWiring,
  replay: renderReplay,
};

export function viewFromHash(hash: string): string {
  const name = hash.replace(/^#\/?/, "");
  return name in views ? name : "results";
}

async function route(): Promise<void> {
  const root = document.getElementById("view");
  if (!root) return;
  leave();
  const name = viewFromHash(location.hash);
  document.querySelectorAll<HTMLAnchorElement>(".tabs a").forEach((link) => {
    link.toggleAttribute("aria-current", link.dataset.view === name);
  });
  root.replaceChildren();
  try {
    await views[name](root);
  } catch (error) {
    root.innerHTML = `<p class="error">Could not load this view: ${String(error)}</p>`;
  }
}

if (typeof window !== "undefined" && document.getElementById("view")) {
  window.addEventListener("hashchange", () => void route());
  void route();
}
