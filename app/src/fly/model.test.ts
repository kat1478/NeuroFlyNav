// @vitest-environment node
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";
import golden from "./golden.json";

const models = resolve("public/models");
const available =
  existsSync(resolve(models, "fly.onnx")) && existsSync(resolve(models, "random.onnx"));

// The models are a release asset (fetched by scripts/fetch-models.sh); CI fetches them first.
describe.skipIf(!available)("exported networks in ONNX Runtime Web", () => {
  for (const name of ["fly", "random"] as const) {
    it(`${name}: reproduces PyTorch's heading sectors`, async () => {
      ort.env.wasm.numThreads = 1;
      const session = await ort.InferenceSession.create(
        new Uint8Array(readFileSync(resolve(models, `${name}.onnx`))),
      );
      const expected = (golden.sectors as Record<string, number[][]>)[name];
      for (let b = 0; b < golden.inputs.length; b++) {
        let state = new ort.Tensor("float32", new Float32Array(2966), [1, 2966]);
        const got: number[] = [];
        for (const row of golden.inputs[b] as number[][]) {
          const out = await session.run({
            state,
            input: new ort.Tensor("float32", Float32Array.from(row), [1, 5]),
          });
          state = out.next_state as typeof state;
          const logits = out.logits.data as Float32Array;
          got.push(logits.indexOf(Math.max(...logits)));
        }
        expect(got).toEqual(expected[b]);
      }
    }, 120_000);
  }
});
