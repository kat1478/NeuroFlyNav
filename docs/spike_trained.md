# Exploratory spike: trained recurrent weights on the CX mask

> **Exploratory.** One task (heading integration), one budget, the compass core only.

## Model

`neuroflynav.training.masked_rnn.MaskedRateRNN`: a leaky rate RNN,
`x ← (1−α)x + α·tanh(W x + B u)` with α = 0.2. `W` has a weight **only on edges of the graph**
(the mask); each of those weights — strength and sign — is trained by backpropagation through
time, as are the input weights into the ring (`ER*`) neurons and a linear readout. Weights start
from the graph's `log1p` synapse counts scaled to spectral radius 1.1, so an untrained model is
the leaky reservoir. Learned signs are model parameters, not biological evidence.

Every variant gets the same budget: 300 Adam steps (lr 3e-3, batch 32) on 256 training
sequences, scored on 64 held-out sequences.

## Compass core

`EPG*`, `PEN*`, `PEG*`, `Delta7` and ring `ER*` neurons of the hemibrain CX subgraph:
409 neurons, 43,629 edges (26% of all pairs).

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_trained --core --seeds 20
```

## Pre-registered result (seeds 0–19)

| Variant | Accuracy | Fly − variant (paired) |
| --- | --- | --- |
| FlyConnectome (real mask, trained) | 0.844 [0.836, 0.852] | — |
| RandomMask (same edge count) | 0.845 [0.837, 0.853] | −0.001 [−0.007, +0.005] |
| ShuffledFly (degree-preserving) | 0.844 [0.837, 0.852] | −0.000 [−0.005, +0.004] |
| TypeShuffledFly (within cell types) | 0.837 [0.827, 0.847] | +0.007 [+0.0004, +0.014] |
| FrozenFly (recurrent weights frozen) | 0.519 [0.510, 0.528] | +0.325 [+0.319, +0.331] |
| hold-last-landmark oracle | 0.565 [0.555, 0.576] | — |

Reading:
- Training the recurrent weights is what makes the network a good integrator (+0.33 over the
  frozen reservoir, +0.28 over the memory-only oracle).
- The real CX mask is **no better a scaffold** than a random or degree-shuffled mask of the same
  density. The tiny edge over the type-shuffled mask is at the margin of significance and not
  matched against the other controls.
- The core is dense (26%), so the mask barely constrains learning. Next: a low-data regime and
  the sparse whole CX (5%), where a structural prior has a chance to matter.

## Low-data regime (pre-registered, seeds 20–39)

Same model and budget; only the number of training sequences changes (16, 32, 64). The plan fixed
one primary test before the run: FlyConnectome − RandomMask averaged over the three sizes.

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_trained --core --train 32 --first-seed 20 --seeds 20
```

| Training sequences | FlyConnectome | RandomMask | ShuffledFly | TypeShuffledFly | FrozenFly |
| --- | --- | --- | --- | --- | --- |
| 16 | 0.371 | 0.392 | 0.380 | 0.367 | 0.450 |
| 32 | 0.570 | 0.605 | 0.578 | 0.551 | 0.520 |
| 64 | 0.762 | 0.780 | 0.763 | 0.744 | 0.496 |

| Paired difference, pooled over sizes | Accuracy |
| --- | --- |
| **FlyConnectome − RandomMask (primary)** | **−0.024 [−0.029, −0.020]** |
| FlyConnectome − ShuffledFly | −0.006 [−0.010, −0.002] |
| FlyConnectome − TypeShuffledFly | +0.014 [+0.010, +0.018] |

Reading:
- With little data the real mask is a slightly **worse** scaffold than an even random mask — the
  opposite of the hoped-for inductive bias.
- The degree-preserving shuffle is almost as good as the real mask; the cell-type shuffle is the
  worst. A likely reason (post hoc): the real CX has strong hubs (in-degree spread 6.7× that of
  the random mask), and hub-heavy masks learn less easily here.
- Training needs ~32 sequences to beat the frozen reservoir; with 16 it overfits.
