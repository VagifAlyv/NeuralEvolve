# NeuralEvolve

A genetic algorithm (NSGA-II) that designs image classifiers for the **Apple Neural Engine**.

It searches for networks that are **accurate, fast on the Neural Engine and small**, all at the same
time. The algorithm evolves only the design of the network and how much each part is compressed.
The weights are always learned by normal training (SGD).

Semestral project for NIE-MVI (Computational Intelligence Methods), FIT CTU.

## How it works

1. **One supernetwork for everything.** Training each candidate network would take far too long, so
   we train one large network once. Every smaller network is a part of it and borrows its weights.
2. **Every candidate is measured on the real hardware.** Each candidate is converted to Core ML,
   compressed, and run on the Mac's Neural Engine. Its accuracy, speed and file size all come from
   that real Core ML model.
3. **NSGA-II** keeps the networks that no other network beats in all three objectives (the Pareto
   front), and breeds new networks from them with crossover and mutation.
4. **Final check.** Networks from the front are trained from scratch and compared with
   MobileNetV2, a well-known network designed by hand for phones.

A genome has 31 numbers: the input resolution, and then for each of the 6 stages the depth, width,
kernel size, expansion and compression bits (4, 6, 8 or 16). See `model.py`.

Data: [Imagenette](https://github.com/fastai/imagenette) (10 ImageNet classes). 1000 training images
are kept apart to score networks during the search. The official test split is used only for the
final numbers.

## Files

| File | What it does | Runs on |
|---|---|---|
| `model.py` | search space, genome, network, supernetwork | everywhere |
| `train.py` | data, supernetwork training, training from scratch | GPU (Kaggle) |
| `search.py` | Core ML measuring, NSGA-II, final measurements | Mac |
| `kaggle.ipynb` | runs `train.py` on Kaggle's free GPU | Kaggle |

## How to run

Install (Python 3.12, because coremltools does not support newer versions yet):

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

1. Train the supernetwork on Kaggle (`kaggle.ipynb`, `TASK = "supernetwork"`), then put
   `supernetwork.pt` into `runs/`.
2. Search on the Mac (about 8 seconds per network, 2 to 3 hours in total):

   ```bash
   .venv/bin/python search.py search
   ```

3. Train the front from scratch on Kaggle (`TASK = "scratch"`, with `runs/front.csv` as input), then
   put the `trained/` folder into `runs/`.
4. Measure the final networks on the Mac:

   ```bash
   .venv/bin/python search.py final
   ```

5. iPhone: open a model from `runs/iphone/` in Xcode, then use *Performance -> Create report* on a
   connected iPhone.

Results: `runs/evaluations.csv` (every network scored), `runs/front.csv`, `runs/final.csv`,
`figures/front.png`, `figures/final.png`. Each step continues where it stopped if you start it again.
