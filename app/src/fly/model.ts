/**
 * Runs the exported one-step ONNX cells (see `neuroflynav.reporting.export_onnx`) with
 * ONNX Runtime Web (WebAssembly, single thread). Loaded lazily: nothing here is imported
 * until the visitor opens "Drop a fly".
 */
import type { InferenceSession, Tensor } from "onnxruntime-web";

type Ort = typeof import("onnxruntime-web/wasm");

let ortPromise: Promise<Ort> | undefined;

/** The WebAssembly binary is emitted by Vite next to the bundle and found automatically. */
export function loadRuntime(): Promise<Ort> {
  ortPromise ??= import("onnxruntime-web/wasm").then((ort) => {
    ort.env.wasm.numThreads = 1; // multithreading needs COOP/COEP headers (not on Pages)
    return ort;
  });
  return ortPromise;
}

export interface Network {
  name: string;
  session: InferenceSession;
  nodes: number;
}

export async function loadNetwork(ort: Ort, name: string, url: string): Promise<Network> {
  const session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  const nodes = Number(
    session.inputMetadata?.[0] && "shape" in session.inputMetadata[0]
      ? (session.inputMetadata[0].shape as number[])[1]
      : 2966,
  );
  return { name, session, nodes };
}

const argmax = (values: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
};

/**
 * Predicted heading sector at every step. `onStep` lets the caller animate while computing;
 * `isCancelled` stops early when the visitor starts a new flight.
 */
export async function runSectors(
  ort: Ort,
  network: Network,
  inputs: number[][],
  onStep?: (step: number) => void,
  isCancelled?: () => boolean,
): Promise<number[] | null> {
  let state: Tensor = new ort.Tensor("float32", new Float32Array(network.nodes), [
    1,
    network.nodes,
  ]);
  const sectors: number[] = [];
  for (let k = 0; k < inputs.length; k++) {
    if (isCancelled?.()) return null;
    const out = await network.session.run({
      state,
      input: new ort.Tensor("float32", Float32Array.from(inputs[k]), [1, inputs[k].length]),
    });
    state = out.next_state;
    sectors.push(argmax(out.logits.data as Float32Array));
    onStep?.(k);
  }
  return sectors;
}
