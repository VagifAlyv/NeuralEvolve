"""Train networks from scratch (no supernet). Used for:

- RQ5: does the supernet rank networks like real training does?
      python train_standalone.py --sample runs/accuracy_dataset_main.csv --k 15 --epochs 40 --out runs/rq5/standalone_40ep.csv
- final results: properly train the final Pareto front and the MobileNetV2 baseline
      python train_standalone.py --from-csv runs/final/front_selected.csv --epochs 40 --out runs/final/front_trained.csv
      python train_standalone.py --baseline --epochs 40 --out runs/final/baseline_trained.csv
"""
import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import genome as G
from common import get_device, seed_everything
from data import TRAIN_RES, Data, prepare
from evaluate import accuracy, simulate_palettization


def train_model(model, res, data, device, epochs, lr=0.05, wd=4e-5):
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    total = epochs * len(data.train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * i / total)))
    for epoch in range(epochs):
        model.train()
        for x, y in data.train_loader:
            x, y = prepare(x, device, TRAIN_RES), y.to(device)
            if res != TRAIN_RES:
                x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y, label_smoothing=0.1).backward()
            opt.step()
            sched.step()
    return model.eval()


def scores(model, res, bits, data, device):
    out = {"val_acc": accuracy(model, *data.search_val, res, device),
           "test_acc": accuracy(model, *data.test, res, device)}
    if bits is not None and min(bits) < 16:
        pal = simulate_palettization(copy.deepcopy(model), bits)
        out["val_acc_pal"] = accuracy(pal, *data.search_val, res, device)
        out["test_acc_pal"] = accuracy(pal, *data.test, res, device)
    else:
        out["val_acc_pal"], out["test_acc_pal"] = out["val_acc"], out["test_acc"]
    return out


def jobs_from_args(args):
    if args.baseline:
        import torchvision
        return [{"key": f"mobilenetv2_w{w}_r{r}", "genome": "",
                 "make": (lambda w=w: torchvision.models.mobilenet_v2(width_mult=w, num_classes=G.NUM_CLASSES)),
                 "res": r, "bits": None, "supernet_acc": ""}
                for w in (0.35, 0.5, 0.75) for r in G.RESOLUTIONS]
    with open(args.from_csv or args.sample) as f:
        rows = list(csv.DictReader(f))
    if args.sample:   # k genomes spread evenly over the supernet-accuracy range
        rows.sort(key=lambda r: float(r["acc"]))
        idx = sorted({round(i * (len(rows) - 1) / max(1, args.k - 1)) for i in range(args.k)})
        rows = [rows[i] for i in idx]
    jobs = []
    for r in rows:
        g = json.loads(r["genome"])
        jobs.append({"key": G.genome_key(g), "genome": r["genome"], "make": (lambda g=g: G.decode(g)[0]),
                     "res": G.RESOLUTIONS[g[0]], "bits": G.bits_per_stage(g),
                     "supernet_acc": r.get("acc_supernet") or r.get("acc", "")})
    return jobs


def main(args):
    seed_everything(args.seed)
    device = get_device()
    data = Data(batch_size=args.batch_size, workers=args.workers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r["key"] for r in csv.DictReader(f)}
    fields = ["key", "genome", "resolution", "supernet_acc", "val_acc", "val_acc_pal", "test_acc", "test_acc_pal",
              "params", "macs", "epochs", "minutes"]
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not done:
            w.writeheader()
        for job in jobs_from_args(args):
            if job["key"] in done:
                continue
            t0 = time.time()
            model = job["make"]()
            row = {"key": job["key"], "genome": job["genome"], "resolution": job["res"],
                   "supernet_acc": job["supernet_acc"], "params": G.count_params(model),
                   "macs": G.count_macs(model, job["res"]), "epochs": args.epochs}
            model = train_model(model, job["res"], data, device, args.epochs, args.lr)
            row.update(scores(model, job["res"], job["bits"], data, device))
            row["minutes"] = round((time.time() - t0) / 60, 2)
            w.writerow(row)
            f.flush()
            print(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-csv", help="CSV with a 'genome' column, e.g. runs/final/front_validated.csv")
    src.add_argument("--sample", help="accuracy dataset CSV to sample --k genomes from (RQ5)")
    src.add_argument("--baseline", action="store_true", help="train MobileNetV2 baselines")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    main(p.parse_args())
