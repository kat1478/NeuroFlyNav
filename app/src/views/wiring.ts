import { loadWiring } from "../data";

export async function renderWiring(root: HTMLElement): Promise<void> {
  const wiring = await loadWiring();
  root.innerHTML = `<h1>Wiring</h1><p class="lede">${wiring.total_neurons.toLocaleString("en")} neurons in ${wiring.labels.length} cell-type families.</p>`;
}
