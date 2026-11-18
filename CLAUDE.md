# CLAUDE.md

Semestral project for NIE-MVI (Computational Intelligence Methods, FIT CTU).

## Project
Hardware-aware multi-objective neural architecture search (NAS) with a genetic algorithm (NSGA-II)
for the Apple Neural Engine. The GA evolves **only the architecture + per-stage compression bits**;
weights are always trained by SGD (the instructor warned that evolving weights does not work).

- Objectives: minimize validation error, ANE latency (learned MLP predictor on real Core ML measurements), model size.
- Dataset: Imagenette at 96–128 px (CIFAR-10 only for debugging).
- Evaluation: weight-sharing supernet (Once-for-All style, sandwich rule) so each candidate takes seconds.
  Fallback if supernet ranking is unreliable: short stand-alone training with a smaller search space.
- Final Pareto front is measured for real on Mac and iPhone (Core ML / Xcode performance report).

## Files
- `genome.py` — search space, 91-gene integer genome, `random_genome`, `mutate`, `crossover_uniform`,
  `repair`, and `decode` (genome -> PyTorch model, input resolution, per-stage bits). Run `python genome.py` to smoke-test.

## Planned experiments (research questions)
1. NSGA-II vs random search vs regularized evolution vs local search, equal budget, 5 seeds (hypervolume).
2. Hardware objective: FLOPs vs params vs latency predictor vs real measurement.
3. Joint architecture + compression search vs sequential.
4. GA ablations: population 20/50/100, mutation rate, crossover type, repair on/off.
5. Supernet accuracy vs from-scratch training (Kendall tau).
6. Transfer of the front Mac ANE vs iPhone ANE vs CPU-only (Spearman).
7. Analysis of genes on the front (what avoids ANE latency cliffs).

## Environment
- macOS, Apple Silicon, PyTorch 2.13 with MPS available. Python: `python3`.
- Planned libraries: `pymoo` (NSGA-II, hypervolume), `coremltools` (conversion, palettization).

## Conventions
- Log every evaluated genome (accuracy, latency, params, FLOPs, bits) to CSV/SQLite; all plots come from this log.
- Cache evaluations by genome so duplicates are never re-evaluated.
- Deliverables: Jupyter notebook + README, technical documentation, blog post, 5-minute pitch.
