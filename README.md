# NeuralEvolve

Hardware-aware multi-objective neural architecture search (NAS) for the **Apple Neural Engine (ANE)**,
using a genetic algorithm (**NSGA-II**).

Semestral project for NIE-MVI (Computational Intelligence Methods), FIT CTU.

> **Status:** early work in progress. The search space and genome (`genome.py`) are being written.
> Nothing below the "Genome" section exists yet.

## Idea

We look for small image classifiers that are, all at the same time:

1. **accurate**: low validation error,
2. **fast on the Apple Neural Engine**: low latency on Mac and iPhone,
3. **small**: few bytes after compression.

These goals conflict, so the result is not one network but a **Pareto front**, a set of trade-offs
from "most accurate" to "fastest".

The genetic algorithm evolves **only the architecture and the compression bits for each stage**.
Weights are always trained by SGD, never evolved.

## How it works

| Part | Approach |
|---|---|
| Search algorithm | NSGA-II (`pymoo`), compared against random search, regularized evolution and local search |
| Accuracy | Weight-sharing supernet (Once-for-All style, sandwich rule), so each candidate is scored in seconds. Fallback: short stand-alone training on a smaller search space. |
| Latency | MLP predictor trained on real Core ML measurements on the ANE |
| Size | Parameter count × bits per stage (palettization with `coremltools`) |
| Dataset | Imagenette at 96–128 px (CIFAR-10 only for debugging) |
| Final check | The Pareto front is measured for real on Mac and iPhone (Core ML / Xcode performance report) |

## Genome

Each candidate network is described by a list of **91 integers**. Each integer is an index into a
menu of choices.

| Position | Genes | Meaning | Options |
|---|---|---|---|
| 0 | 1 | input resolution | 96, 112, 128 px |
| 1–18 | 6 stages × 3 | depth, width multiplier, compression bits | depth 1–4, width 0.75/1.0/1.25, bits 4/6/8/16 |
| 19–90 | 24 block slots × 3 | kernel size, expansion ratio, squeeze-excite | kernel 3/5/7, expand 3/4/6, SE off/on |

Each stage has room for 4 blocks. If its depth is smaller, the remaining block slots are ignored.
`repair` resets those unused genes so identical networks get identical genomes, which lets the
evaluation cache catch duplicates.

`genome.py` (planned contents): `random_genome`, `mutate`, `crossover_uniform`, `repair`, and
`decode` (genome → PyTorch model, input resolution, bits per stage).

## Research questions

1. NSGA-II vs random search vs regularized evolution vs local search, equal budget, 5 seeds (hypervolume).
2. Hardware objective: FLOPs vs params vs latency predictor vs real measurement.
3. Joint architecture + compression search vs sequential.
4. GA ablations: population 20/50/100, mutation rate, crossover type, repair on/off.
5. Supernet accuracy vs from-scratch training (Kendall tau).
6. Transfer of the front: Mac ANE vs iPhone ANE vs CPU-only (Spearman).
7. Which genes appear on the front (what avoids ANE latency cliffs).

## Setup

Requires macOS on Apple Silicon and Python 3.

```bash
pip install torch pymoo coremltools
python3 genome.py      # smoke test
```

## Roadmap

- [ ] `genome.py`: search space, genetic operators, decode
- [ ] `latency.py`: Core ML conversion, ANE measurements, latency predictor
- [ ] `supernet.py`: weight-sharing supernet training on Imagenette
- [ ] `search.py`: NSGA-II with an evaluation cache and a log of every genome
- [ ] Experiments for RQ1–7
- [ ] Notebook, technical documentation, blog post, 5-minute pitch
