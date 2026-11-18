# Interactive web app

A static web app in [`app/`](../app/) shows the results to visitors:
**[kat1478.github.io/NeuroFlyNav](https://kat1478.github.io/NeuroFlyNav/)**
(AWS mirror: [d20rmxnlajencb.cloudfront.net](https://d20rmxnlajencb.cloudfront.net)).

| View | What it shows |
| --- | --- |
| Results | Headline result, all pre-registered comparisons, per-seed differences, learning curve |
| Wiring | Synapses between the 19 cell-type families of the CX, family cards, glossary of abbreviations |
| Replay | Two trained whole-CX networks (real vs random wiring) tracking heading, step by step |
| City | A fly forages over central Warsaw (OpenStreetMap) and flies home by path integration; real vs random wiring ([spike_homing.md](spike_homing.md)). **Drop a fly anywhere:** the visitor places a nest and lets the fly search or picks a treat; the trained networks run live in the browser |

There is no server: the app reads four small JSON files (plus `treats.geojson`) in
`app/public/data/`, written by `neuroflynav.reporting.app_data` from committed result
summaries (and, with `--refresh`, from local results and the hemibrain export; with `--city`,
from the homing pilot traces and summary). Only family-level aggregates and synthetic-task traces
are published; no neuron-level connectome data. The City view also loads
`app/public/map/warsaw.pmtiles` (OpenStreetMap, ODbL; built by `geo/build_map.py`, see
`app/public/map/manifest.json`) and is loaded lazily, so MapLibre does not slow the other views.

## Drop a fly: the trained networks in the browser

The "Drop a fly anywhere" mode runs the seed-100 whole-CX networks (real wiring and random
wiring, 2,966 neurons, 467,579 connections each) with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WebAssembly, single thread).

- **Export** (`neuroflynav.reporting.export_onnx`): retrains the two networks exactly as in the
  City pilot and writes a one-step cell, `(state, input) → (next_state, logits)`, per network.
  The recurrence stays sparse — gather the state at each edge's source, multiply by the edge
  weight, scatter-add at the target (`ScatterElements`, reduction `add`) — with int32 edge
  indices: 5.8 MB per model instead of a dense 2,966 × 2,966 matrix. The exported cells predict
  the same sector as PyTorch at every step of 64 test sequences (agreement 1.0; heading accuracy
  0.8253 real, 0.8277 random), and a Vitest golden test checks the browser code path against
  PyTorch's outputs.

  ```bash
  PYTHONPATH=src python -m neuroflynav.reporting.export_onnx   # → results/models/ (gitignored)
  ```

- **Hosting:** the models carry neuron-level connectivity derived from hemibrain (CC BY 4.0), so
  they are published as the GitHub release asset `models-v1` and on the site, never committed to
  Git. `app/scripts/fetch-models.sh` downloads them and checks their SHA-256 against
  `app/public/models/manifest.json`; the Pages workflow, `make deploy-aws` and CI call it before
  building.
- **In the browser** (`app/src/fly/`): a TypeScript port of the flight generator (seeded PRNG,
  AR(1) turns, landmark every 8 steps, 5 m steps, treat detection at 15 m; in "pick a treat" the
  turns are gently pulled towards the treat, clipped to the training range), the network runner
  and the home-vector maths. The runtime and the models (about 9 MB compressed) load only when
  the first fly is dropped; both networks step together (~10 ms per step) and yield to the
  browser every few steps, so the page stays responsive.
- **Honesty:** a dropped fly is an illustration, not the test — single flights vary a lot, and
  flights from other nests or steered to a treat are longer than the tested ones. The result
  stays the pre-registered one in [spike_homing.md](spike_homing.md).

## Develop

```bash
mamba env create -f app/environment.yml        # Node 22 from conda-forge (separate env)
cd app
conda run -n neuroflynav-app npm ci
scripts/fetch-models.sh                         # trained models for "Drop a fly" (needs gh)
conda run -n neuroflynav-app npm run dev        # http://localhost:5173
conda run -n neuroflynav-app npm run check      # lint, format, typecheck, tests (also in CI)
```

Refresh the data after new results (from the repository root):

```bash
PYTHONPATH=src python -m neuroflynav.reporting.app_data            # from committed data
PYTHONPATH=src python -m neuroflynav.reporting.app_data --refresh  # + local results, ~15 min
```

Stack: Vite, TypeScript, Observable Plot; colours match the README figures and were checked
for colour-vision deficiency in light and dark mode; every chart has a table view.

## Deploy

- **GitHub Pages (canonical):** `.github/workflows/pages.yml` builds and publishes `app/dist`
  on every push to `master` that touches `app/` (that is, with each release), or on demand
  from the Actions tab.
- **AWS mirror:** a private S3 bucket behind CloudFront with Origin Access Control
  (`infra/site.tf`; no public bucket, HTTPS only, AWS-managed security headers). After
  `terraform apply`, upload with `conda run -n neuroflynav-app make deploy-aws` (needs
  `aws login --profile neuroflynav` and `AWS_PROFILE=neuroflynav`). The mirror lives on
  the project's AWS Free plan and is not guaranteed to outlast it. The `.wasm` runtime and
  the `.onnx` models are uploaded pre-gzipped (`Content-Encoding: gzip`), because CloudFront
  compresses neither files over 10 MB nor `application/octet-stream`.
