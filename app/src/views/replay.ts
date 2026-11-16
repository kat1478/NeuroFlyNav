import { loadReplay } from "../data";

export async function renderReplay(root: HTMLElement): Promise<void> {
  const replay = await loadReplay();
  root.innerHTML = `<h1>Replay</h1><p class="lede">${replay.sequences.length} test sequences on the ${replay.graph}.</p>`;
}
