import "./styles.css";
import { Scope } from "./lifecycle";
import { renderReplay } from "./views/replay";
import { renderResults } from "./views/results";
import { renderWiring } from "./views/wiring";

type Render = (root: HTMLElement, scope: Scope) => Promise<void>;

const views: Record<string, { title: string; render: Render }> = {
  results: { title: "Results", render: renderResults },
  wiring: { title: "Wiring", render: renderWiring },
  replay: { title: "Replay", render: renderReplay },
};

export function viewFromHash(hash: string): string {
  const name = hash.replace(/^#\/?/, "");
  return name in views ? name : "results";
}

let scope = new Scope();

async function route(): Promise<void> {
  const root = document.getElementById("view");
  if (!root) return;
  scope.dispose();
  const mine = (scope = new Scope());
  const name = viewFromHash(location.hash);
  document.title = `${views[name].title} · NeuroFlyNav`;
  document.querySelectorAll<HTMLAnchorElement>(".tabs a").forEach((link) => {
    if (link.dataset.view === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  const container = document.createElement("div");
  const loading = Object.assign(document.createElement("p"), {
    className: "loading",
    textContent: "Loading…",
  });
  container.append(loading);
  root.replaceChildren(container);
  root.setAttribute("aria-busy", "true");
  try {
    await views[name].render(container, mine);
  } catch (error) {
    if (!mine.active) return;
    const message = document.createElement("p");
    message.className = "error";
    message.textContent = `Could not load this view: ${String(error)}`;
    container.replaceChildren(message);
  } finally {
    if (mine.active) {
      loading.remove();
      root.removeAttribute("aria-busy");
      root.focus({ preventScroll: true });
    }
  }
}

if (typeof window !== "undefined" && document.getElementById("view")) {
  window.addEventListener("hashchange", () => void route());
  void route();
}
