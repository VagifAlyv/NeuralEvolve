# NeuralEvolve

Hardware-aware multi-objective neural architecture search (NAS) for the **Apple Neural Engine (ANE)**,
using a genetic algorithm (**NSGA-II**).

## Idea

We look for small image classifiers that are, at the same time:

1. **accurate**: low validation error,
2. **fast on the Apple Neural Engine**: low latency on Mac and iPhone,
3. **small**: few bytes after compression.

These goals conflict, so the result is a **Pareto front**: a set of trade-offs from "most accurate"
to "fastest". The genetic algorithm evolves **only the architecture and the compression bits of each
stage**. Weights are always trained by SGD, never evolved.

## How it works

```
                 ┌──────────────── train once (SGD) ────────────────┐
Imagenette ──►   supernet.py: weight-sharing supernet (sandwich rule)
                 └──────────────────────────────────────────────────┘
                          │ extract subnet + BN recalibration + simulated palettization
                          ▼
evaluate.py collect ──► accuracy of ~1000 random genomes ──► predictors.py accuracy ──► accuracy MLP
latency.py collect  ──► Core ML latency of ~400 genomes  ──► predictors.py latency  ──► latency MLP
                          │
                          ▼
experiments.py: NSGA-II / random / regularized evolution / local search
                objective(genome) = [1 - accuracy, latency, size MB]   (every call logged)
                          │
                          ▼
final front ──► validated with the supernet + real latency (Mac, iPhone) ──► trained from scratch
                          │
                          ▼
analysis.py / notebook.ipynb ──► tables and figures for RQ1–RQ7
```

| Part | Approach |
|---|---|
| Search | NSGA-II (`pymoo`) with our own sampling, crossover, mutation, repair |
| Baselines | random search, regularized (aging) evolution, Pareto local search, MobileNetV2 |
| Accuracy | weight-sharing supernet (Once-for-All / BigNAS style) + MLP accuracy predictor |
| Latency | Core ML on the ANE, measured on the Mac + MLP latency predictor; iPhone for the final front |
| Size | weights × bits per stage (Core ML palettization; tensors < 2048 weights stay 16-bit) |
| Data | Imagenette 160px, train at 128px; 1000 train images held out for search; official val = test |

## Genome (91 integers)

| Position | Genes | Meaning | Options |
|---|---|---|---|
| 0 | 1 | input resolution | 96, 112, 128 px |
| 1–18 | 6 stages × 3 | depth, width multiplier, compression bits | depth 1–4, width 0.75/1.0/1.25, bits 4/6/8/16 |
| 19–90 | 24 block slots × 3 | kernel size, expansion ratio, squeeze-excite | kernel 3/5/7, expand 3/4/6, SE off/on |

Each stage has room for 4 blocks; slots beyond the stage's depth are inactive. `repair` sets inactive
genes to 0, so identical networks get identical genomes (cache key; RQ4 tests the GA without it).

## Files

| File | What it does |
|---|---|
| `genome.py` | search space, genome, genetic operators, `decode` → plain PyTorch `Net`, params/MACs/size |
| `common.py` | paths, device, seeding, SQLite cache |
| `data.py` | Imagenette splits, in-memory evaluation tensors |
| `supernet.py` | weight-sharing supernet, `extract(genome)`, sandwich-rule training |
| `evaluate.py` | BN recalibration, palettization simulation, `SubnetEvaluator`, accuracy dataset |
| `latency.py` | Core ML conversion + palettization, latency measurement, latency dataset, iPhone export |
| `predictors.py` | one-hot genome features, MLP predictors for accuracy and latency |
| `search.py` | `Objective` (logs every evaluation), NSGA-II, random, regularized evolution, local search, hypervolume |
| `experiments.py` | RQ1–RQ4 grids, final search, post-processing |
| `train_standalone.py` | training from scratch (RQ5, final front, MobileNetV2 baseline) |
| `analysis.py`, `notebook.ipynb` | all tables and figures |
| `docs/` | technical report, blog post draft, pitch outline |

## Reproduce

Requires macOS on Apple Silicon (for Core ML on the Neural Engine) and Python 3.10+.

```bash
pip install -r requirements.txt
```

Step by step (times are rough guesses, measure your own):

```bash
python genome.py                                   # 1. smoke test: build min/mid/max/random networks
python latency.py names                            # 2. check that Core ML weight names contain the stage index
python supernet.py --epochs 3 --tag debug          # 3. short run to check the pipeline (then delete runs/supernet/debug*)
python supernet.py --epochs 100 --tag main         # 4. real supernet (hours)
python evaluate.py check                           # 5. accuracy of min/mid/max subnets
python evaluate.py collect --n 1000                # 6. accuracy dataset (~10 s per genome)
python latency.py collect --n 400                  # 7. latency dataset (~10 s per genome)
python latency.py baseline                         # 8. MobileNetV2 latencies
python predictors.py latency                       # 9. latency MLP
python predictors.py accuracy                      # 10. accuracy MLP
python experiments.py rq1                          # 11. RQ1 (4 algorithms x 5 seeds)
python experiments.py rq2 && python experiments.py rq2-measure
python experiments.py rq3 && python experiments.py rq3-post
python experiments.py rq4
python experiments.py final && python experiments.py validate-final
python latency.py measure --csv runs/final/front_validated.csv          # RQ6: Mac CPU/GPU/ANE
python latency.py export-iphone --csv runs/final/front_validated.csv    # RQ6: iPhone (see below)
python train_standalone.py --sample runs/accuracy_dataset_main.csv --k 15 --epochs 40 --out runs/rq5/standalone_40ep.csv
python train_standalone.py --from-csv runs/final/front_validated.csv --epochs 40 --out runs/final/front_trained.csv
python train_standalone.py --baseline --epochs 40 --out runs/final/baseline_trained.csv
jupyter notebook notebook.ipynb                    # figures for everything
```

**Training on Google Colab instead.** The GPU-heavy steps (supernet, accuracy dataset, all
from-scratch training) can run on Colab with `colab.ipynb`, which clones this repository and saves
results to Google Drive. Core ML latency, the search experiments and the iPhone export stay on the Mac.
Push your latest code before opening the notebook. `kaggle.ipynb` does the same on Kaggle (about
30 free GPU hours per week): choose a `TASK`, attach the needed `runs/` files as a private dataset and use
"Save & Run All", which keeps running with the browser closed.

Every step can be stopped and restarted: finished runs are skipped and evaluations are cached in
`runs/cache.sqlite`. Use `--dataset cifar10` on `supernet.py` only to debug quickly.

**iPhone latency.** Open each `runs/iphone/front_XX.mlpackage` in Xcode, select the *Performance* tab,
choose your connected iPhone and run the report. Copy the median prediction time into
`runs/iphone/iphone_latency.csv`.

## Research questions

| RQ | Question | Command | Metric |
|---|---|---|---|
| 1 | Is NSGA-II better than random search, regularized evolution and local search at equal budget? | `experiments.py rq1` | hypervolume, 5 seeds, Mann-Whitney U |
| 2 | Which speed objective works best in reality: MACs, params, or the latency predictor? | `rq2`, `rq2-measure` | hypervolume with measured latency |
| 3 | Is searching architecture and compression together better than one after the other? | `rq3`, `rq3-post` | hypervolume |
| 4 | Which GA settings matter (population, mutation rate, crossover, repair)? | `rq4` | hypervolume, 3 seeds |
| 5 | Does the supernet rank networks like training from scratch? | `train_standalone.py --sample` | Kendall τ |
| 6 | Do latency rankings transfer between Mac ANE, Mac CPU/GPU and iPhone? | `latency.py measure`, iPhone | Spearman ρ |
| 7 | Which design choices survive on the front (what does the ANE like)? | notebook | gene statistics |

## Known risks (check these on the first run)

- **Core ML weight names.** Per-stage palettization finds weights by the regex `stages_<i>_` in their
  Core ML names. `python latency.py names` prints them; if nothing matches, adjust `STAGE_RE` in
  `latency.py`. The code warns when nothing was palettized.
- **Library versions.** `coremltools` may lag behind the newest PyTorch; the pymoo operator API is for
  pymoo 0.6.x.
- **Supernet quality.** If RQ5 shows a weak correlation (Kendall τ < ~0.5), use the fallback: score
  candidates with `--accuracy supernet` on a smaller space, or short stand-alone training.

## Related work

APQ (Wang et al., CVPR 2020) is the closest work: joint architecture + quantization search with
evolution on a Once-for-All network. Also: Once-for-All (Cai et al., 2019), BigNAS (Yu et al., 2020),
NSGA-Net (Lu et al., 2018), MnasNet (Tan et al., 2018), ProxylessNAS (Cai et al., 2018),
Regularized Evolution (Real et al., 2019), Li & Talwalkar (2019), MobileNetV2 (Sandler et al., 2018),
NSGA-II (Deb et al., 2002).
