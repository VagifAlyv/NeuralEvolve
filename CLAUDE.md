# CLAUDE.md

Semestral project for NIE-MVI (Computational Intelligence Methods, FIT CTU). Project name: NeuralEvolve.

## Project
Hardware-aware multi-objective neural architecture search (NAS) with a genetic algorithm (NSGA-II)
for the Apple Neural Engine. The GA evolves **only the architecture + per-stage compression bits**;
weights are always trained by SGD (the instructor warned that evolving weights does not work).

- Objectives: minimize validation error, ANE latency, model size (MB after palettization).
- Dataset: Imagenette 160px, trained at 128 px, searched at 96/112/128; 1000 train images held out
  as search-validation; official val split = test (final numbers only). CIFAR-10 only for debugging.
- Accuracy: weight-sharing supernet (sandwich rule + in-place distillation), subnet extraction,
  BN recalibration, simulated palettization; plus an MLP accuracy predictor for fast experiments.
- Latency: Core ML (float16, per-stage palettization) on the Mac's CPU_AND_NE; MLP latency predictor.
- Final front: re-scored with the supernet and real latency, measured on iPhone, trained from scratch.

## Files
- `genome.py` — search space, 91-gene genome, operators (`mutate`, three crossovers, `repair` = canonical
  form), `decode` -> plain `Net`, `cost_measures` (params, MACs, size).
- `common.py`, `data.py` — paths/device/SQLite cache; data splits.
- `supernet.py` — supernet, `extract(genome)`, training CLI.
- `evaluate.py` — BN recalibration, palettization simulation, `SubnetEvaluator`, accuracy dataset CLI.
- `latency.py` — Core ML conversion/measurement, latency dataset, iPhone export, MobileNetV2 baseline.
- `predictors.py` — one-hot features, MLP predictors.
- `search.py` — `Objective` (logs every call), NSGA-II (pymoo), random, regularized evolution, local search, hypervolume.
- `experiments.py` — RQ1–RQ4 grids, final search, post-processing.
- `train_standalone.py` — from-scratch training (RQ5, final front, baseline).
- `colab.ipynb` — runs the GPU steps (supernet, accuracy dataset, from-scratch training) on Colab; clones
  the repo from GitHub, results on Google Drive. Mac-only: Core ML latency, experiments, iPhone.
- `analysis.py`, `notebook.ipynb` — tables/figures. `docs/` — report, blog post, pitch.
- Outputs go to `runs/` (gitignored) and `figures/`. README has the full command order.

## Research questions
1. NSGA-II vs random search vs regularized evolution vs local search, equal budget, 5 seeds (hypervolume).
2. Speed objective: MACs vs params vs latency predictor (vs measured in the loop), judged by measured latency.
3. Joint architecture + compression search vs architecture-only vs sequential.
4. GA ablations: population 20/50/100, mutation rate, crossover type, repair on/off.
5. Supernet accuracy vs from-scratch training (Kendall tau).
6. Latency rank transfer: Mac CPU/GPU/ANE vs iPhone (Spearman).
7. Gene statistics on the final front (what the ANE likes).

## Environment
- macOS, Apple Silicon, PyTorch with MPS. Python: `python3`. `pip install -r requirements.txt`.
- Status (2026-11-19): code written by Claude at the user's request. First run done up to a 1-epoch debug
  supernet: venv is Python 3.12 (`.venv`; coremltools does not support 3.14). Fixes from the first run:
  BatchNorm folded before Core ML conversion (`latency.fold_bn`, else stage weights lose their names),
  scikit-learn added, supernet batch 64 / lr 0.05 on the 16 GB M4 (batch 128 swaps to disk).
  Measured on M4: 3.4 min/supernet epoch, ~4 s per subnet accuracy, ~2 s per Core ML latency.

## Conventions
- Every evaluation is logged (CSV per run) and cached (`runs/cache.sqlite`); runs are resumable.
- Analysis cuts every run at exactly its budget so algorithms are compared fairly.
- Deliverables: notebook + README, technical report, blog post, 5-minute pitch.
