# CLAUDE.md

Semestral project for NIE-MVI (Computational Intelligence Methods, FIT CTU). Project name: NeuralEvolve.
Goal: a result strong and clear enough to be noticed by Apple.

## Project
Hardware-aware multi-objective neural architecture search with NSGA-II for the Apple Neural Engine.
The GA evolves only the architecture + per-stage compression bits; weights are always trained by SGD.
Objectives (all minimized): validation error, Neural Engine latency, Core ML model size.
Every candidate is converted to Core ML, palettized, and its accuracy/latency/size are measured on the
Mac's Neural Engine (no simulation, no predictors).

## Files (rewritten from scratch on 2026-11-26, kept small on purpose)
- `model.py` — search space, 31-gene genome (values, not indices), mutate/crossover, `Network`,
  `SuperNetwork` (same layer names, so `extract` just cuts weights to shape).
- `train.py` — Imagenette data, supernetwork training (sandwich rule + in-place distillation),
  training from scratch of the front + MobileNetV2. GPU (Kaggle).
- `search.py` — BN recalibration, Core ML conversion/palettization, NSGA-II (hand-written), final
  measurements, figures. Mac only.
- `kaggle.ipynb` — clones the repo and runs `train.py` on Kaggle.
- Outputs: `runs/` and `figures/`. `runs_old/` = results of the first version (gitignored, not compatible).

## Style (the user asked for this)
Simple, short, readable by everyone. Full variable names, no one-letter names, no lambdas, no `**`
dict tricks, no clever one-liners. Few files. Research questions are added only after the core works.

## Environment
macOS, Apple Silicon (M4, 16 GB), Python 3.12 venv in `.venv` (coremltools does not support 3.14).
On the Mac train with `--batch-size 64 --learning-rate 0.05`. Commands are in the README.
