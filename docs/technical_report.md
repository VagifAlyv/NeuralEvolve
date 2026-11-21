# NeuralEvolve: technical report

*NIE-MVI semestral project. Status 2026-11-21: supernet, surrogates, RQ1, RQ2 and RQ5 are done; sections
marked **TODO** still need results (RQ3, RQ4, RQ6, RQ7, final front).*

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

Held-out quality (20 % of each dataset):

| Predictor | Train / val | Spearman | Kendall | MAPE |
|---|---|---|---|---|
| latency (ANE, Mac) | 323 / 80 | 0.82 | 0.62 | 6.6 % |
| accuracy (supernet) | 803 / 200 | 0.63 | – | – |

For comparison, MACs alone rank the 403 measured latencies with Spearman 0.80 and parameter count with
only 0.36. The latency predictor is therefore only slightly better than MACs at ranking; RQ2 tests
whether that difference matters for the search. The accuracy predictor is weaker because the supernet
accuracies of different subnets lie close together (80 % of them between 81.0 and 84.7 %).

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
| 5 | 15 genomes spread over supernet accuracy, trained 40 epochs from scratch | – | – |
| 6 | final front on Mac CPU / GPU / ANE / ALL and iPhone | – | – |
| 7 | gene statistics of the final front (NSGA-II, population 100, 5000 evaluations) | 1 | 5000 |

That is 4 + 4 + 3 + 9 = 20 search setups, plus RQ5–RQ7.

## 6. Results

### 6.1 Supernet and surrogates
The supernet was trained for 50 epochs on a Colab T4 (82 GPU minutes). On the 1000-image search-validation
set the largest, middle and smallest subnets reach 88.7 / 84.2 / 83.5 % (`figures/supernet_training.png`).
The accuracy dataset (1003 random subnets, mean 82.9 %, range 75.1–88.7 %) and the latency dataset
(403 networks, about 2 s each on the M4 including Core ML conversion) were collected once and reused by every experiment.

Latency of untrained networks on the Mac ANE ranges from 0.15 to 0.65 ms (median 0.22 ms). The
MobileNetV2 baselines take 0.16 ms (width 0.35, 96 px) to 0.25 ms (width 1.0, 128 px) on the ANE but
0.30–0.79 ms on the CPU and 0.66–1.08 ms on the GPU, so the ANE is 2–4× faster than either, even for
these small networks (`runs/baseline_latency.csv`, `figures/macs_vs_latency.png`).

### 6.2 RQ1: which search algorithm?
Four algorithms, 2000 evaluations each, 5 seeds, predictor objectives, one shared normalization
(`figures/rq1.png`).

| Algorithm | Hypervolume (mean ± std) | at 500 / 1000 evals | Front size | p vs NSGA-II |
|---|---|---|---|---|
| **NSGA-II** | **0.940 ± 0.009** | 0.819 / 0.883 | 119 | – |
| Pareto local search | 0.876 ± 0.005 | 0.766 / 0.824 | 161 | 0.008 |
| Regularized evolution | 0.827 ± 0.020 | 0.748 / 0.788 | 90 | 0.008 |
| Random search | 0.791 ± 0.017 | 0.761 / 0.775 | 37 | 0.008 |

NSGA-II is ahead from about 150 evaluations on and still improving at 2000. Every NSGA-II seed beats every
seed of the other algorithms; p = 0.008 (two-sided Mann–Whitney U) is the smallest value possible with
5 vs 5 seeds. Local search is second: changing one gene at a time works well because the predictors make
the landscape smooth, but it spends its budget near the archive and finds fewer fast networks. Regularized
evolution, built for one objective, keeps too few non-dominated parents and only passes random search after
about 750 evaluations. Random search stops improving after about 500 evaluations: in a space of ~10^40
genomes, random samples rarely reach the extremes of the front. Each run takes about one minute.

### 6.3 RQ2: which speed objective?
NSGA-II was run three times (2000 evaluations, 3 seeds each) with a different speed objective: MACs,
parameter count, or the latency predictor. Every network on each final front (208–392 per objective) was
then converted to Core ML and timed on the ANE, and the fronts were scored with the *measured* latency
(`figures/rq2.png`).

| Speed objective | HV, measured (3 objectives) | HV, measured (error–latency only) | Front size | Fastest / median latency |
|---|---|---|---|---|
| **latency predictor** | **0.789 ± 0.018** | **0.889** | 131 | 0.134 / 0.175 ms |
| MACs | 0.772 ± 0.027 | 0.831 | 118 | 0.123 / 0.177 ms |
| params | 0.745 ± 0.024 | 0.789 | 69 | 0.159 / 0.213 ms |

The predictor gives the best fronts and parameter count the worst, matching how well each proxy ranks
latency (Spearman 0.82 / 0.80 / 0.36 on the latency dataset). The differences are small and not
significant with 3 seeds (Mann–Whitney p = 0.4 predictor vs MACs, 0.2 predictor vs params). The gap is
largest when size is left out, because parameter count is already the size objective: optimizing it for
speed too only finds small networks, not fast ones. On the MACs fronts, measured latency follows MACs with
only Spearman 0.57: among networks with similar MACs, latency still differs by operation type and layer
shape, which MACs cannot see (RQ7 looks at which genes matter). MACs still found the single fastest network
(0.123 ms), so it is a reasonable free proxy; the predictor is better at the middle of the front.

### 6.4 RQ3: joint architecture + compression search
**TODO** (`analysis.rq3()`).

### 6.5 RQ4: GA ablations
**TODO** (`analysis.rq4()`).

### 6.6 RQ5: does the supernet rank networks like training from scratch?
15 genomes sampled across the supernet-accuracy range (75.1–88.7 %) were trained from scratch for 40 epochs on
Kaggle (T4, 3.3 GPU hours in total) and evaluated with and without the palettization from their genome
(`figures/rq5.png`).

| From-scratch accuracy compared with | Kendall τ | p | Spearman ρ | Mean (range) |
|---|---|---|---|---|
| search-val, uncompressed | 0.22 | 0.25 | 0.31 | 82.7 % (80.7–84.8) |
| search-val, palettized | 0.36 | 0.06 | 0.47 | 81.5 % (75.6–84.5) |
| test, uncompressed | 0.33 | 0.09 | 0.49 | 82.6 % (80.6–84.6) |
| **test, palettized** | **0.50** | **0.008** | **0.67** | 81.7 % (77.1–84.3) |

The supernet score includes the simulated palettization, so the fair comparison is the last row: a
moderate, significant rank correlation. Two things limit it. First, from-scratch accuracies span only 4
points, while the noise of one 3925-image test evaluation is about ±0.6 points, so many pairs are swapped
by chance. Second, the supernet spreads networks over 14 points that shrink to 4 after real training, so
it exaggerates differences between small and large networks. With 15 epochs instead of 40 the correlation
was only τ = 0.11: under-trained networks rank by how fast they learn, not by their final accuracy.
The search therefore uses the supernet as a coarse filter, and the final front is retrained from scratch.

### 6.7 RQ6: latency transfer between devices
**TODO** (`analysis.rq6()`).

### 6.8 RQ7: what does the ANE like?
**TODO** (`analysis.rq7()`).

### 6.9 Final front vs MobileNetV2
**TODO** (`analysis.final_front()`).

### 6.10 Cost
So far: supernet 1.4 GPU hours (Colab), RQ5 3.3 GPU hours (Kaggle), latency dataset about 15 min on
the Mac, RQ1 about 20 min and RQ2 9 min of search plus about 4.5 h
of ANE measurement (954 front networks, Core ML conversion dominates) on the Mac. **TODO:** add the
remaining experiments.

## 7. Limitations

- Supernet accuracy is a proxy with only a moderate rank correlation to real training (RQ5, τ = 0.50),
  and RQ5 uses 15 networks, so its confidence interval is wide.
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
