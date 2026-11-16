# Interactive web app

A static web app in [`app/`](../app/) shows the results to visitors:
**[kat1478.github.io/NeuroFlyNav](https://kat1478.github.io/NeuroFlyNav/)**
(AWS mirror: [d20rmxnlajencb.cloudfront.net](https://d20rmxnlajencb.cloudfront.net)).

| View | What it shows |
| --- | --- |
| Results | Headline result, all pre-registered comparisons, per-seed differences, learning curve |
| Wiring | Synapses between the 19 cell-type families of the CX, family cards, glossary of abbreviations |
| Replay | Two trained whole-CX networks (real vs random wiring) tracking heading, step by step |

There is no server and no model in the browser: the app reads three small JSON files in
`app/public/data/`, written by `neuroflynav.reporting.app_data` from committed result
summaries (and, with `--refresh`, from local results and the hemibrain export). Only
family-level aggregates and synthetic-task traces are published; no neuron-level connectome data.

## Develop

```bash
mamba env create -f app/environment.yml        # Node 22 from conda-forge (separate env)
cd app
conda run -n neuroflynav-app npm ci
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
  the project's AWS Free plan and is not guaranteed to outlast it.
