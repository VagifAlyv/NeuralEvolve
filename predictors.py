"""Small MLPs that predict a genome's accuracy or latency in microseconds instead of seconds.

Input features: one-hot encoding of every ACTIVE gene (inactive block slots are all zeros).

    python predictors.py latency    # runs/latency_dataset.csv          -> runs/latency_predictor.pt
    python predictors.py accuracy   # runs/accuracy_dataset_main.csv    -> runs/accuracy_predictor.pt
"""
import argparse
import csv
import functools
import json

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr

import genome as G
from common import RUNS_DIR, seed_everything

OFFSETS = np.concatenate([[0], np.cumsum(G.N_CHOICES)[:-1]])
FEATURE_DIM = int(sum(G.N_CHOICES))


def features(g):
    x = np.zeros(FEATURE_DIM, dtype=np.float32)
    for i, (v, active) in enumerate(zip(g, G.active_mask(g))):
        if active:
            x[OFFSETS[i] + v] = 1.0
    return x


class GenomeMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Predictor:
    """Wraps the MLP and the target scaling. predict(g) returns a value in the original unit."""

    def __init__(self, mlp, mean, std, log_target):
        self.mlp, self.mean, self.std, self.log_target = mlp.eval(), mean, std, log_target
        self._cached = functools.lru_cache(maxsize=500_000)(self._predict_key)

    def _predict_key(self, key):
        g = [int(v) for v in key.split("-")]
        with torch.no_grad():
            y = self.mlp(torch.from_numpy(features(g))).item() * self.std + self.mean
        return float(np.exp(y)) if self.log_target else float(y)

    def predict(self, g):
        return self._cached(G.genome_key(g))

    def save(self, path):
        torch.save({"state": self.mlp.state_dict(), "hidden": self.mlp.net[0].out_features,
                    "mean": self.mean, "std": self.std, "log_target": self.log_target}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, map_location="cpu")
        mlp = GenomeMLP(ck["hidden"])
        mlp.load_state_dict(ck["state"])
        return Predictor(mlp, ck["mean"], ck["std"], ck["log_target"])


def fit(csv_path, target, log_target, out_path, hidden=128, epochs=3000, lr=1e-3, wd=1e-4, val_frac=0.2, seed=0):
    """Train on 80% of the rows, report rank correlation on the other 20%, save the model."""
    seed_everything(seed)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    X = np.stack([features(json.loads(r["genome"])) for r in rows])
    y = np.array([float(r[target]) for r in rows])
    t = np.log(y) if log_target else y
    perm = np.random.permutation(len(rows))
    n_val = int(len(rows) * val_frac)
    val, tr = perm[:n_val], perm[n_val:]
    mean, std = float(t[tr].mean()), float(t[tr].std() + 1e-8)

    mlp = GenomeMLP(hidden)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    Xt, Tt = torch.from_numpy(X[tr]), torch.from_numpy(((t[tr] - mean) / std).astype(np.float32))
    for _ in range(epochs):                      # full batch: the datasets have only ~1000 rows
        opt.zero_grad()
        loss = nn.functional.mse_loss(mlp(Xt), Tt)
        loss.backward()
        opt.step()

    pred = Predictor(mlp, mean, std, log_target)
    with torch.no_grad():
        p = mlp(torch.from_numpy(X[val])).numpy() * std + mean
    p = np.exp(p) if log_target else p
    metrics = {"n_train": len(tr), "n_val": n_val,
               "spearman": float(spearmanr(p, y[val])[0]), "kendall": float(kendalltau(p, y[val])[0]),
               "mape": float(np.mean(np.abs(p - y[val]) / np.abs(y[val]))),
               "rmse": float(np.sqrt(np.mean((p - y[val]) ** 2)))}
    pred.save(out_path)
    with open(str(out_path).replace(".pt", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(str(out_path).replace(".pt", "_val.npz"), predicted=p, true=y[val])   # for parity plots
    print(metrics)
    return pred, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["latency", "accuracy"])
    p.add_argument("--csv")
    p.add_argument("--unit", default="CPU_AND_NE")
    p.add_argument("--tag", default="main")
    args = p.parse_args()
    if args.which == "latency":
        fit(args.csv or RUNS_DIR / "latency_dataset.csv", f"latency_{args.unit}", True,
            RUNS_DIR / "latency_predictor.pt")
    else:
        fit(args.csv or RUNS_DIR / f"accuracy_dataset_{args.tag}.csv", "acc", False,
            RUNS_DIR / "accuracy_predictor.pt")
