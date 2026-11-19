"""Scoring one genome with the trained supernet, and the accuracy dataset for the accuracy predictor.

Evaluating genome g:
  1. extract its weights from the supernet          (supernet.extract)
  2. recalibrate BatchNorm statistics on ~2000 images (recalibrate_bn)
  3. simulate Core ML palettization with each stage's bits (simulate_palettization)
  4. top-1 accuracy on the search-validation images

    python evaluate.py check   --supernet runs/supernet/main.pt
    python evaluate.py collect --supernet runs/supernet/main.pt --n 1000
"""
import argparse
import csv
import json
import random
import time

import torch
import torch.nn as nn

import genome as G
from common import RUNS_DIR, JsonCache, get_device
from data import Data, prepare


@torch.no_grad()
def recalibrate_bn(net, calib_x, resolution, device, batch_size=256):
    """Recompute BatchNorm running mean/var for this exact subnet (cumulative average over calib_x)."""
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    net.train()
    for i in range(0, len(calib_x), batch_size):
        net(prepare(calib_x[i:i + batch_size], device, resolution))
    net.eval()
    return net


def kmeans_1d(v, k, iters=20):
    """Cluster the values of v into k centers and replace every value by its center (a 'palette')."""
    flat = v.detach().flatten().float().cpu()
    c = torch.quantile(flat, torch.linspace(0, 1, k))
    for _ in range(iters):
        c, _ = torch.sort(c)
        idx = torch.bucketize(flat, (c[1:] + c[:-1]) / 2)
        sums = torch.zeros(k).index_add_(0, idx, flat)
        counts = torch.bincount(idx, minlength=k).float()
        c = torch.where(counts > 0, sums / counts.clamp(min=1), c)
    c, _ = torch.sort(c)
    return c[torch.bucketize(flat, (c[1:] + c[:-1]) / 2)].view_as(v).to(v.device)


@torch.no_grad()
def _palettize_conv_bn(conv, bn, bits):
    """Core ML folds BatchNorm into the conv BEFORE palettizing, so we cluster the folded weights.
    Writing back w_pal / scale keeps the module structure and gives exactly bn(conv_{w_pal/scale}(x))."""
    if conv.weight.numel() < G.PALETTIZE_MIN_WEIGHTS:
        return
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    scale = torch.where(scale.abs() < 1e-8, torch.full_like(scale, 1e-8), scale).view(-1, 1, 1, 1)
    conv.weight.copy_(kmeans_1d(conv.weight * scale, 2 ** bits) / scale)


@torch.no_grad()
def _palettize_plain(conv, bits):
    if conv.weight.numel() >= G.PALETTIZE_MIN_WEIGHTS:
        conv.weight.copy_(kmeans_1d(conv.weight, 2 ** bits))


@torch.no_grad()
def simulate_palettization(net, bits):
    """In place. Call AFTER recalibrate_bn (folding needs the running statistics)."""
    for stage, b in zip(net.stages, bits):
        if b >= 16:
            continue
        for blk in stage:
            _palettize_conv_bn(blk.expand_conv, blk.expand_bn, b)
            _palettize_conv_bn(blk.dw_conv, blk.dw_bn, b)
            _palettize_conv_bn(blk.proj_conv, blk.proj_bn, b)
            if blk.se is not None:
                _palettize_plain(blk.se.fc1, b)
                _palettize_plain(blk.se.fc2, b)
    return net


@torch.no_grad()
def accuracy(net, x, y, resolution, device, batch_size=500):
    net.eval()
    correct = 0
    for i in range(0, len(y), batch_size):
        logits = net(prepare(x[i:i + batch_size], device, resolution))
        correct += (logits.argmax(1).cpu() == y[i:i + batch_size]).sum().item()
    return correct / len(y)


class SubnetEvaluator:
    """genome -> {'acc', 'params', 'macs', 'size_mb'} using the supernet. Results are cached on disk."""

    def __init__(self, supernet, data, device, tag="main"):
        self.supernet, self.data, self.device = supernet, data, device
        self.cache = JsonCache(f"subnet_acc_{tag}")

    def __call__(self, g):
        key = G.genome_key(g)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        res, bits = G.RESOLUTIONS[g[0]], G.bits_per_stage(g)
        net = self.supernet.extract(g).to(self.device)
        recalibrate_bn(net, self.data.calib, res, self.device)
        simulate_palettization(net, bits)
        rec = {"acc": accuracy(net, *self.data.search_val, res, self.device), **G.cost_measures(g)}
        self.cache.put(key, rec)
        return rec


def quick_check(supernet, data, device):
    """Accuracy of the largest, middle and smallest subnets (16-bit), used while training the supernet."""
    out = {}
    for name in ["max", "mid", "min"]:
        g = G.extreme_genome(name)
        for s in range(G.N_STAGES):              # no compression for this check
            g[G.stage_index(s) + 2] = G.BITS.index(16)
        net = supernet.extract(g).to(device)
        res = G.RESOLUTIONS[g[0]]
        recalibrate_bn(net, data.calib, res, device)
        out[f"acc_{name}"] = round(accuracy(net, *data.search_val, res, device), 4)
    supernet.train()
    return out


def collect(args):
    """Score n random genomes with the supernet -> CSV used to train the accuracy predictor (and RQ5)."""
    from supernet import load_supernet
    device = get_device()
    data = Data(need_train=False, workers=args.workers)
    ev = SubnetEvaluator(load_supernet(args.supernet, device), data, device, tag=args.tag)
    out = RUNS_DIR / f"accuracy_dataset_{args.tag}.csv"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {row["key"] for row in csv.DictReader(f)}
    rng = random.Random(args.seed)
    genomes = [G.extreme_genome(w) for w in ("min", "mid", "max")] + [G.random_genome(rng) for _ in range(args.n)]
    fields = ["key", "genome", "acc", "params", "macs", "size_mb", "seconds"]
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not done:
            w.writeheader()
        for i, g in enumerate(genomes):
            key = G.genome_key(g)
            if key in done:
                continue
            t0 = time.time()
            rec = ev(g)
            w.writerow({"key": key, "genome": json.dumps(G.repair(g)), **rec, "seconds": round(time.time() - t0, 2)})
            f.flush()
            done.add(key)
            print(f"[{i + 1}/{len(genomes)}] acc={rec['acc']:.3f} size={rec['size_mb']:.2f}MB")


def check(args):
    from supernet import load_supernet
    device = get_device()
    data = Data(need_train=False, workers=args.workers)
    print(quick_check(load_supernet(args.supernet, device), data, device))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["check", "collect"])
    p.add_argument("--supernet", default=str(RUNS_DIR / "supernet" / "main.pt"))
    p.add_argument("--tag", default="main")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    {"check": check, "collect": collect}[args.command](args)
