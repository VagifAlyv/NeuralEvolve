
import argparse
import copy
import csv
import json
import random
import re
import shutil
import statistics
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch

import genome as G
from common import RUNS_DIR, JsonCache

STAGE_RE = re.compile(r"stages[_.](\d+)[_.]")
COMPUTE_UNITS = ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]


def _ct():
    import coremltools as ct   # imported lazily so the rest of the project works without it
    return ct


def fold_bn(model):
    """Copy of model with every BatchNorm folded into the conv before it (`x_conv` + `x_bn`).

    Core ML would fold them anyway, but its folded weights get anonymous names (`const_102`), so
    STAGE_RE could not find them and per-stage palettization would skip all convs. Folding here keeps
    the PyTorch names (`stages_3_1_proj_conv_weight`).
    """
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    model = copy.deepcopy(model).eval()
    for mod in model.modules():
        for name in [n for n in mod._modules if n.endswith("_bn")]:
            conv_name = name[:-3] + "_conv"
            conv = getattr(mod, conv_name, None)
            if isinstance(conv, torch.nn.Conv2d):
                setattr(mod, conv_name, fuse_conv_bn_eval(conv, getattr(mod, name)))
                setattr(mod, name, torch.nn.Identity())
    return model


def to_coreml(model, resolution, bits=None):
    """PyTorch model -> Core ML ML Program (float16), palettized per stage when bits < 16."""
    ct = _ct()
    model = fold_bn(model)
    example = torch.rand(1, 3, resolution, resolution)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(traced,
                         inputs=[ct.TensorType(name="image", shape=example.shape)],
                         outputs=[ct.TensorType(name="logits")],
                         convert_to="mlprogram",
                         compute_precision=ct.precision.FLOAT16,
                         minimum_deployment_target=ct.target.iOS17)
    n_pal = 0
    if bits is not None and min(bits) < 16:
        from coremltools.optimize.coreml import (OpPalettizerConfig, OptimizationConfig, get_weights_metadata,
                                                 palettize_weights)
        configs = {}
        for name in get_weights_metadata(mlmodel, weight_threshold=G.PALETTIZE_MIN_WEIGHTS):
            m = STAGE_RE.search(name)
            if m and bits[int(m.group(1))] < 16:
                configs[name] = OpPalettizerConfig(mode="kmeans", nbits=bits[int(m.group(1))],
                                                   weight_threshold=G.PALETTIZE_MIN_WEIGHTS)
        n_pal = len(configs)
        if n_pal == 0:
            warnings.warn("No Core ML weight name matched STAGE_RE: nothing was palettized. "
                          "Run `python latency.py names` and adjust STAGE_RE.")
        else:
            mlmodel = palettize_weights(mlmodel, OptimizationConfig(op_name_configs=configs))
    return mlmodel, n_pal


def measure(path, resolution, compute_units="CPU_AND_NE", warmup=10, runs=50):
    """Median latency in ms of one prediction (batch 1). Includes a small, constant Python overhead."""
    ct = _ct()
    mlmodel = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, compute_units))
    x = {"image": np.random.rand(1, 3, resolution, resolution).astype(np.float32)}
    for _ in range(warmup):            # the first calls compile the model for the device
        mlmodel.predict(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        mlmodel.predict(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def measure_genome(g, compute_units=("CPU_AND_NE",), keep_path=None):
    """Convert genome g and time it. Returns {'latency_<unit>': ms, ..., 'n_palettized': k}."""
    model, res, bits = G.decode(g)
    mlmodel, n_pal = to_coreml(model, res, bits)
    tmp = Path(tempfile.mkdtemp())
    path = Path(keep_path) if keep_path else tmp / "model.mlpackage"
    mlmodel.save(str(path))
    out = {f"latency_{u}": measure(path, res, u) for u in compute_units}
    out["n_palettized"] = n_pal
    shutil.rmtree(tmp, ignore_errors=True)
    return out


class MeasuredLatency:
    """genome -> measured ms on the Mac (cached). Used as the 'measured' objective in RQ2."""

    def __init__(self, compute_units="CPU_AND_NE"):
        self.unit = compute_units
        self.cache = JsonCache(f"latency_{compute_units}")

    def __call__(self, g):
        key = G.genome_key(g)
        hit = self.cache.get(key)
        if hit is None:
            hit = measure_genome(g, (self.unit,))
            self.cache.put(key, hit)
        return hit[f"latency_{self.unit}"]


def _read_genomes(csv_path):
    with open(csv_path) as f:
        return [json.loads(row["genome"]) for row in csv.DictReader(f)]


def cmd_names(args):
    model, res, bits = G.decode(G.extreme_genome("mid"))
    mlmodel, _ = to_coreml(model, res, None)
    from coremltools.optimize.coreml import get_weights_metadata
    for name, meta in get_weights_metadata(mlmodel, weight_threshold=0).items():
        print(f"{name:70s} stage={STAGE_RE.search(name).group(1) if STAGE_RE.search(name) else '-':>2} "
              f"shape={tuple(meta.val.shape)}")


def cmd_collect(args):
    """Training data for the latency predictor."""
    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {row["key"] for row in csv.DictReader(f)}
    rng = random.Random(args.seed)
    genomes = [G.extreme_genome(w) for w in ("min", "mid", "max")] + [G.random_genome(rng) for _ in range(args.n)]
    fields = ["key", "genome", "resolution", "params", "macs", "size_mb", "n_palettized", f"latency_{args.unit}"]
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not done:
            w.writeheader()
        for i, g in enumerate(genomes):
            key = G.genome_key(g)
            if key in done:
                continue
            row = {"key": key, "genome": json.dumps(G.repair(g)), "resolution": G.RESOLUTIONS[g[0]],
                   **G.cost_measures(g), **measure_genome(g, (args.unit,))}
            w.writerow(row)
            f.flush()
            done.add(key)
            print(f"[{i + 1}/{len(genomes)}] {row[f'latency_{args.unit}']:.2f} ms  {row['macs'] / 1e6:.0f} MMACs")


def cmd_measure(args):
    """Time every genome of a CSV (e.g. a final front) on several compute units (RQ2, RQ6)."""
    src = Path(args.csv)
    out = src.with_name(src.stem + "_measured.csv")
    rows = []
    for i, g in enumerate(_read_genomes(src)):
        rows.append({"key": G.genome_key(g), "genome": json.dumps(G.repair(g)), **G.cost_measures(g),
                     **measure_genome(g, args.units)})
        print(f"[{i + 1}] " + ", ".join(f"{u}={rows[-1][f'latency_{u}']:.2f}ms" for u in args.units))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


def cmd_export_iphone(args):
    """Save .mlpackage files + a CSV template. Fill latency_iphone_ms from Xcode's Core ML performance report."""
    out_dir = RUNS_DIR / "iphone"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, g in enumerate(_read_genomes(args.csv)):
        model, res, bits = G.decode(g)
        mlmodel, _ = to_coreml(model, res, bits)
        name = f"front_{i:02d}.mlpackage"
        mlmodel.save(str(out_dir / name))
        rows.append({"file": name, "key": G.genome_key(g), "genome": json.dumps(G.repair(g)),
                     "latency_iphone_ms": ""})
    with open(out_dir / "iphone_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"exported {len(rows)} models to {out_dir}; fill in iphone_latency.csv")


def cmd_baseline(args):
    """Hand-designed baselines: torchvision MobileNetV2 at a few widths."""
    import torchvision
    rows = []
    for width in [0.35, 0.5, 0.75, 1.0]:
        for res in G.RESOLUTIONS:
            model = torchvision.models.mobilenet_v2(width_mult=width, num_classes=G.NUM_CLASSES)
            mlmodel, _ = to_coreml(model, res, None)
            tmp = Path(tempfile.mkdtemp())
            mlmodel.save(str(tmp / "m.mlpackage"))
            row = {"name": f"mobilenetv2_w{width}_r{res}", "width": width, "resolution": res,
                   "params": G.count_params(model), "macs": G.count_macs(model, res)}
            for u in args.units:
                row[f"latency_{u}"] = measure(tmp / "m.mlpackage", res, u)
            shutil.rmtree(tmp, ignore_errors=True)
            rows.append(row)
            print(row)
    with open(RUNS_DIR / "baseline_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["names", "collect", "measure", "export-iphone", "baseline"])
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unit", default="CPU_AND_NE", choices=COMPUTE_UNITS)
    p.add_argument("--units", nargs="+", default=COMPUTE_UNITS, choices=COMPUTE_UNITS)
    p.add_argument("--csv")
    p.add_argument("--out", default=str(RUNS_DIR / "latency_dataset.csv"))
    args = p.parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    {"names": cmd_names, "collect": cmd_collect, "measure": cmd_measure,
     "export-iphone": cmd_export_iphone, "baseline": cmd_baseline}[args.command](args)
