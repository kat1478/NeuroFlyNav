# Development spike (M0+): a comparison that can discriminate topologies

> **Non-authoritative development spike.** Everything here is synthetic and every dynamics/training
> choice is an M0+ placeholder, not a frozen scientific decision. The comparison below is
> *illustrative only*: it has no statistical or biological meaning and must never be read as evidence
> that any topology is better. Its purpose is that the machinery runs, is reproducible, and is *able*
> to tell topologies apart — the plain M0 task could not (all policies tied).

## Why M0+ exists

In the merged M0 slice, `FlyConnectome` and `ShuffledFly` scored identically: the task was trivially
solvable and the control could still solve it, so the setup could not distinguish wirings. M0+ fixes
the *machinery* on synthetic data, before any real connectome, so that a topology difference *can*
show up and so the training/evaluation/metrics code is ready to reuse at M2.

## What changed

- **Harder task with memory** (`neuroflynav.env.nav_memory`): the goal direction is shown only for the
  first `cue_steps` steps, then hidden. A memoryless policy cannot finish after the cue disappears; a
  policy with recurrent memory can. This is what lets topology matter.
- **Fixed graph substrate + trained readout** (`neuroflynav.agent.reservoir`): the graph becomes a
  frozen recurrent weight matrix (a reservoir). The observation is injected at sensory nodes; the state
  evolves through the fixed wiring; only a small linear readout is trained. This mirrors the project's
  intended shape (frozen connectome topology, trained heads). The readout reads the whole reservoir
  state (standard echo-state practice); a biologically-constrained motor-only readout is a later
  decision (TASK-013/014), not an M0+ choice.
- **Closed-form imitation training** (`neuroflynav.training.imitation`): a privileged greedy expert
  produces demonstrations; the readout is fit by ridge regression. No gradient loop, fully reproducible.
- **Controls that can discriminate** (`neuroflynav.experiments.run_m0plus`): `FlyConnectome` (the
  synthetic graph) vs `ShuffledFly` (degree-preserving shuffle) vs `RandomRecurrent` (matched-edge
  random graph) vs `Reactive` (no memory). Same task, seeds and training budget.
- **Metrics that set up M2's vocabulary:** success rate, mean steps, and a **robustness** probe (a
  shorter cue window at test time) — robustness is where prior art (e.g. FLYNN) found the connectome
  signal, so the hook is built now.

## How to run

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_m0plus --seed 0
```

An execution example from this branch (illustrative, not a result):

```json
{"base_seed": 0, "cue_steps": 3, "eval_episodes": 40, "graph_edges": 160, "graph_nodes": 64,
 "grid_size": 7, "robust_cue_steps": 1, "train_episodes": 80, "policies": [
  {"policy": "FlyConnectome",   "graph": "synthetic",              "success_rate": 0.55,  "mean_steps": 30.7,  "robust_success_rate": 0.30},
  {"policy": "ShuffledFly",     "graph": "shuffled(seed=0)",       "success_rate": 0.525, "mean_steps": 31.275,"robust_success_rate": 0.20},
  {"policy": "RandomRecurrent", "graph": "random(matched-edges)",  "success_rate": 0.425, "mean_steps": 36.525,"robust_success_rate": 0.125},
  {"policy": "Reactive",        "graph": "none",                   "success_rate": 0.275, "mean_steps": 44.725,"robust_success_rate": 0.125}]}
```

How to read this (and how not to): the memory-capable reservoirs beat the memoryless baseline, which
only confirms the task now needs memory and the machinery can use it. Any ordering *among* the
reservoirs is noise at this scale on a synthetic graph — **not** evidence about biology. A real test
needs the frozen protocol, matched controls, many seeds and statistics decided at M2.

## Quality gate

```bash
make quality
```

runs pytest, `ruff format --check`, `ruff check`, and `mypy` (strict) over the spike code. The spike
adds one dependency, `numpy` (conda-forge), recorded in `environment.yml`.

## Next steps (not part of M0+)

1. **M1:** replace the synthetic graph with a real CX subgraph loader behind the same `GraphLoader`
   protocol, only after D-034 preconditions (exact file identity + license) are met.
2. **M2:** freeze the scientific task, representation, signs/dynamics, controls, metrics and protocol;
   this spike's trainer/evaluator/metrics are the reusable starting point.
