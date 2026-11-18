/**
 * Runs the exported one-step ONNX cells (see `neuroflynav.reporting.export_onnx`) with
 * ONNX Runtime Web (WebAssembly, single thread). Nothing here loads until the visitor drops
 * a fly; the runtime and the two models are then kept for the rest of the page's life.
 */
import type { InferenceSession, Tensor } from "onnxruntime-web";

type Ort = typeof import("onnxruntime-web/wasm");

export interface Network {
  name: "FlyConnectome" | "RandomMask";
  session: InferenceSession;
  nodes: number;
}

export interface Networks {
  ort: Ort;
  nets: Network[];
}

interface Manifest {
  nodes: number;
  models: Record<string, { file: string; bytes: number; sha256: string }>;
}

const MODEL_FILES: [Network["name"], string][] = [
  ["FlyConnectome", "fly"],
  ["RandomMask", "random"],
];

/** Approximate compressed download (runtime + both models), for honest copy in the UI. */
export const DOWNLOAD_MB_COMPRESSED = 9;

let cached: Promise<Networks> | undefined;

/** Load the runtime and both models once (in parallel); a failed load is retried next time. */
export function getNetworks(base: string): Promise<Networks> {
  cached ??= (async () => {
    const manifest = (await (await fetch(`${base}models/manifest.json`)).json()) as Manifest;
    // Version the model URLs by content so a new release is never served from a stale cache.
    const files = MODEL_FILES.map(async ([, key]) => {
      const model = manifest.models[key];
      const response = await fetch(`${base}models/${model.file}?v=${model.sha256.slice(0, 12)}`);
      if (!response.ok) throw new Error(`${model.file}: HTTP ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    });
    const ortPromise = import("onnxruntime-web/wasm").then((ort) => {
      ort.env.wasm.numThreads = 1; // multithreading needs COOP/COEP headers (not on Pages)
      return ort;
    });
    const [ort, ...bytes] = await Promise.all([ortPromise, ...files]);
    const nets = await Promise.all(
      MODEL_FILES.map(async ([name], i) => ({
        name,
        session: await ort.InferenceSession.create(bytes[i], { executionProviders: ["wasm"] }),
        nodes: manifest.nodes,
      })),
    );
    return { ort, nets };
  })();
  cached.catch(() => (cached = undefined));
  return cached;
}

const argmax = (values: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
};

/** Let the browser paint and handle input (not throttled like setTimeout in background tabs). */
const yieldToBrowser = (): Promise<void> =>
  new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(null);
  });

let queue: Promise<unknown> = Promise.resolve();

/**
 * Predicted heading sector of every network at every step. Both networks advance together,
 * and the loop yields every few steps so the page stays responsive. Runs are serialised, so
 * a cancelled run and a new one never use the same session at once. Returns null if cancelled.
 */
export function runSectors(
  networks: Networks,
  inputs: number[][],
  onProgress?: (done: number, total: number) => void,
  isCancelled?: () => boolean,
): Promise<number[][] | null> {
  const run = async (): Promise<number[][] | null> => {
    const { ort, nets } = networks;
    let states: Tensor[] = nets.map(
      (net) => new ort.Tensor("float32", new Float32Array(net.nodes), [1, net.nodes]),
    );
    const sectors: number[][] = nets.map(() => []);
    for (let k = 0; k < inputs.length; k++) {
      if (isCancelled?.()) return null;
      const input = new ort.Tensor("float32", Float32Array.from(inputs[k]), [1, inputs[k].length]);
      const next: Tensor[] = [];
      for (let n = 0; n < nets.length; n++) {
        const out = await nets[n].session.run({ state: states[n], input });
        next.push(out.next_state);
        sectors[n].push(argmax(out.logits.data as Float32Array));
      }
      states = next;
      if (k % 5 === 4 || k === inputs.length - 1) {
        onProgress?.(k + 1, inputs.length);
        await yieldToBrowser();
      }
    }
    return sectors;
  };
  const result = queue.then(run, run);
  queue = result.catch(() => undefined);
  return result;
}
