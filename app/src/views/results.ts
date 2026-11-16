import { loadResults } from "../data";

export async function renderResults(root: HTMLElement): Promise<void> {
  const results = await loadResults();
  root.innerHTML = `<h1>Results</h1><p class="lede">${results.forest.length} pre-registered comparisons.</p>`;
}
