# NeuralEvolve: technical report

*NIE-MVI semestral project. Sections marked **TODO** need results from the experiments.*

## 1. Problem

Find image classifiers for the Apple Neural Engine (ANE) that trade off three objectives:
classification error, inference latency and model size. Because the ANE runs only some operations
efficiently and falls back to GPU/CPU otherwise, latency does not follow FLOPs/MACs, so a
hardware-aware search is needed. The result is a Pareto front rather than a single network.

## 2. Search space and genome

MobileNetV2-style network: fixed stem (3×3 conv, 16 channels, stride 2), 6 searchable stages,
fixed head (1×1 conv to 640 channels, global pooling, linear classifier, 10 classes).

| Gene group | Count | Options |
|---|---|---|
| input resolution | 1 | 96 / 112 / 128 |
| per stage: depth | 6 | 1–4 blocks |
| per stage: width multiplier | 6 | 0.75 / 1.0 / 1.25 × base [16, 24, 40, 80, 112, 160], rounded to a multiple of 4 |
| per stage: weight bits | 6 | 4 / 6 / 8 / 16 (Core ML k-means palettization) |
| per block: kernel | 24 | 3 / 5 / 7 (depthwise) |
| per block: expansion | 24 | 3 / 4 / 6 |
| per block: squeeze-excite | 24 | off / on |

91 integer genes, 3 × (4·3·4)^6 × (3·3·2)^24 ≈ 5·10^40 genomes (fewer distinct networks, because
inactive slots do not matter). Each block is an inverted-residual (MBConv) block with ReLU; residual
connections are used when stride is 1 and channels match.

**Design decisions**
- *Fixed-length genome with inactive genes* instead of variable length: simple crossover (gene i always
  means the same thing), and inactive genes act as hidden memory that mutation of the depth gene can
  re-activate.
- *Repair = canonical form* (inactive genes set to 0): makes duplicate detection and caching exact.
- *Bits per stage, not per layer*: keeps the genome small while still letting the search compress
  late, wide stages more than early ones.

## 3. Evaluation pipeline

### 3.1 Weight-sharing supernet
One network contains all subnets (elastic depth, width, expansion, kernel via center crop of a 7×7
kernel, optional SE, elastic resolution). Trained with the sandwich rule (largest + smallest + 2 random
subnets per batch), the largest learning from labels (label smoothing 0.1) and the others by in-place
knowledge distillation from it. SGD with Nesterov momentum, lr 0.1, cosine schedule, 3 warm-up epochs.
BatchNorm in the supernet always uses batch statistics; every extracted subnet gets its own statistics
by recalibration on 2048 training images.

### 3.2 Compression
Core ML palettization replaces each weight by one of 2^bits shared values (k-means). We simulate it in
PyTorch on the BatchNorm-folded weights (as Core ML palettizes after folding), for weight tensors with at
least 2048 elements (the Core ML default threshold). Size = 16-bit for everything except palettized
stage weights.

### 3.3 Latency
Each network is converted to a Core ML ML Program (float16), palettized per stage, and timed on the Mac
with compute units `CPU_AND_NE` (median of 50 predictions after 10 warm-up calls). Latency does not
depend on weight values, so untrained networks are timed.

### 3.4 Surrogates
Two MLPs (input: one-hot of active genes, 261 features; 2 hidden layers of 128) predict accuracy
(trained on ~1000 supernet evaluations) and log-latency (trained on ~400 measurements). They make one
evaluation take microseconds, so experiments with several seeds are affordable. Final fronts are always
re-checked with the supernet and real measurements.

**TODO:** predictor held-out Spearman / Kendall / MAPE (table from `analysis.predictor_quality()`).

## 4. Search algorithms

All algorithms receive the same objective `f(g) = [1 − accuracy, latency, size]` and the same budget
of evaluations; every evaluation is logged.

| Algorithm | Details |
|---|---|
| NSGA-II | pymoo; random initial population; uniform crossover (p = 0.9); reset mutation 1/91 per gene; canonical repair; duplicates removed by network identity |
| Random search | uniform random genomes |
| Regularized evolution | population 50, tournament of 10, parent = random non-dominated member of the sample, oldest dies |
| Pareto local search | archive of non-dominated genomes, change exactly one active gene of a random member |

**Metric.** Hypervolume of all evaluated points after normalizing each objective with the min/max over
all runs of the same research question (reference point (1, 1, 1)). Runs are cut at exactly the budget.

## 5. Experiments

| RQ | Setups | Seeds | Budget |
|---|---|---|---|
| 1 | NSGA-II, random, regularized evolution, local search | 5 | 2000 |
| 2 | speed objective = MACs, params, latency predictor (+ measured latency in the loop) | 3 (1) | 2000 (300) |
| 3 | joint; architecture only (16-bit); sequential (architecture front + uniform 8/6/4 bits) | 3 | 2000 |
| 4 | default; population 20/100; mutation 0.5/91, 2/91, 4/91; one-point / stage-wise crossover; no repair | 3 | 2000 |
| 5 | 15 genomes spread over supernet accuracy, trained 15 epochs from scratch | – | – |
| 6 | final front on Mac CPU / GPU / ANE / ALL and iPhone | – | – |
| 7 | gene statistics of the final front (NSGA-II, population 100, 5000 evaluations) | 1 | 5000 |

That is 4 + 4 + 3 + 9 = 20 search setups, plus RQ5–RQ7.

## 6. Results

**TODO** for each RQ: the table and figure from `notebook.ipynb`, and two or three sentences on what it
means. Also report the search cost in hours (supernet training, datasets, searches).

## 7. Limitations

- Supernet accuracy is a proxy; RQ5 measures how good it is.
- The predictors can be exploited by the search; final fronts are re-evaluated without them.
- Latency is measured from Python and includes a small constant overhead; the iPhone numbers come from
  Xcode and are measured differently.
- Imagenette has 10 easy classes; results may not transfer to harder datasets.
- Palettization accuracy is post-training (no fine-tuning), which favours higher bit widths.

## 8. References

Deb et al. 2002 (NSGA-II); Sandler et al. 2018 (MobileNetV2); Tan et al. 2018 (MnasNet); Cai et al.
2018 (ProxylessNAS); Lu et al. 2018 (NSGA-Net); Real et al. 2019 (Regularized Evolution); Li &
Talwalkar 2019 (random search for NAS); Cai et al. 2019 (Once-for-All); Yu et al. 2020 (BigNAS);
Wang et al. 2020 (APQ).
