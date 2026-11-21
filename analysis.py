"""Tables and figures for every research question. Used by notebook.ipynb; figures go to figures/.

Every function returns a pandas DataFrame and saves a PNG. Missing results are reported, not fatal.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr

import genome as G
from common import FIG_DIR, RUNS_DIR
from search import hypervolume, nondominated_mask

OBJ = ["f_error", "f_latency", "f_size"]
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _missing(path):
    if not Path(path).exists():
        print(f"missing: {path}")
        return True
    return False


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    return fig


# ---------------------------------------------------------------- loading runs

def load_run(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "done.json").read_text())
    df = pd.read_csv(run_dir / "evals.csv").head(meta["budget"])    # equal budget for everyone
    return meta, df


def load_rq(rq):
    """{variant: [(seed, df), ...]} for all finished runs of an RQ."""
    out = {}
    for done in sorted((RUNS_DIR / rq).glob("*/seed*/done.json")):
        meta, df = load_run(done.parent)
        out.setdefault(meta["variant"], []).append((meta["seed"], df))
    return out


def common_bounds(frames, cols=OBJ):
    """Same normalization for every variant of one RQ: min and max over everything evaluated."""
    F = np.vstack([df[cols].to_numpy(float) for df in frames])
    lo, hi = F.min(0), F.max(0)
    return lo, hi + 1e-9


def hv_curve(df, lo, hi, step=50, cols=OBJ):
    F = df[cols].to_numpy(float)
    xs = list(range(step, len(F) + 1, step))
    return np.array(xs), np.array([hypervolume(F[:n], lo, hi) for n in xs])


def front(df, cols=OBJ):
    return df[nondominated_mask(df[cols].to_numpy(float))].drop_duplicates("key")


def _hv_table(runs, lo, hi, cols=OBJ, reference=None):
    rows = []
    finals = {v: [hypervolume(df[cols].to_numpy(float), lo, hi) for _, df in lst] for v, lst in runs.items()}
    for v, hv in finals.items():
        row = {"variant": v, "hv_mean": np.mean(hv), "hv_std": np.std(hv), "seeds": len(hv)}
        if reference and v != reference and reference in finals and len(hv) > 1:
            row[f"p_vs_{reference}"] = mannwhitneyu(finals[reference], hv, alternative="two-sided").pvalue
        rows.append(row)
    return pd.DataFrame(rows).sort_values("hv_mean", ascending=False)


# ---------------------------------------------------------------- supernet and predictors

def supernet_training(tag="main"):
    path = RUNS_DIR / "supernet" / f"{tag}_log.csv"
    if _missing(path):
        return None
    df = pd.read_csv(path)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(df["epoch"], df["loss_max"])
    ax[0].set(xlabel="epoch", ylabel="loss of largest subnet")
    for c in ["acc_max", "acc_mid", "acc_min"]:
        d = df.dropna(subset=[c])
        ax[1].plot(d["epoch"], d[c], marker="o", label=c.replace("acc_", ""))
    ax[1].set(xlabel="epoch", ylabel="search-val accuracy")
    ax[1].legend()
    _save(fig, "supernet_training")
    return df


def predictor_quality():
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for i, name in enumerate(["latency", "accuracy"]):
        m, v = RUNS_DIR / f"{name}_predictor_metrics.json", RUNS_DIR / f"{name}_predictor_val.npz"
        if _missing(m) or _missing(v):
            continue
        rows.append({"predictor": name, **json.loads(m.read_text())})
        d = np.load(v)
        ax[i].scatter(d["true"], d["predicted"], s=8, alpha=0.6)
        lim = [min(d["true"].min(), d["predicted"].min()), max(d["true"].max(), d["predicted"].max())]
        ax[i].plot(lim, lim, "k--", lw=1)
        ax[i].set(xlabel=f"true {name}", ylabel=f"predicted {name}", title=f"{name} predictor (held-out)")
    _save(fig, "predictor_parity")
    return pd.DataFrame(rows)


def macs_vs_latency(unit="CPU_AND_NE"):
    """The core motivation: is MACs a good proxy for ANE latency?"""
    path = RUNS_DIR / "latency_dataset.csv"
    if _missing(path):
        return None
    df = pd.read_csv(path)
    col = f"latency_{unit}"
    rho = spearmanr(df["macs"], df[col])[0]
    df["has_se"] = df["genome"].apply(lambda s: any(b["se"] for st in G.describe(json.loads(s))["stages"]
                                                    for b in st["blocks"]))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for se, d in df.groupby("has_se"):
        ax.scatter(d["macs"] / 1e6, d[col], s=10, alpha=0.6, label="contains SE" if se else "no SE")
    ax.set(xlabel="MMACs", ylabel=f"latency {unit} [ms]", title=f"MACs vs measured latency (Spearman {rho:.2f})")
    ax.legend()
    _save(fig, "macs_vs_latency")
    return pd.DataFrame([{"spearman_macs_latency": rho,
                          "spearman_params_latency": spearmanr(df["params"], df[col])[0], "n": len(df)}])


# ---------------------------------------------------------------- RQ1-RQ4

def rq1():
    runs = load_rq("rq1")
    if not runs:
        print("no rq1 runs")
        return None
    lo, hi = common_bounds([df for lst in runs.values() for _, df in lst])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for v, lst in runs.items():
        curves = [hv_curve(df, lo, hi) for _, df in lst]
        n = min(len(c[0]) for c in curves)
        Y = np.array([c[1][:n] for c in curves])
        ax[0].plot(curves[0][0][:n], Y.mean(0), label=v)
        ax[0].fill_between(curves[0][0][:n], Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), alpha=0.2)
        fr = front(lst[0][1], cols=["f_error", "f_latency"]).sort_values("f_latency")
        ax[1].step(fr["f_latency"], 1 - fr["f_error"], where="post", marker=".", label=v)
    ax[0].set(xlabel="evaluations", ylabel="hypervolume", title="RQ1: search progress (mean ± std)")
    ax[1].set(xlabel="predicted latency [ms]", ylabel="accuracy",
              title="Error-latency front (seed 0, size ignored)")
    ax[0].legend()
    ax[1].legend()
    _save(fig, "rq1")
    return _hv_table(runs, lo, hi, reference="nsga2")


def rq2():
    """Fronts found with each speed objective, judged by MEASURED latency."""
    rows, frames = [], {}
    for path in sorted((RUNS_DIR / "rq2").glob("*/seed*/front_measured.csv")):
        variant = path.parent.parent.name
        frames.setdefault(variant, []).append(pd.read_csv(path))
    if not frames:
        print("no rq2 fronts measured (run: python experiments.py rq2-measure)")
        return None
    cols = ["f_error", "latency_measured", "f_size"]
    lo, hi = common_bounds([df for lst in frames.values() for df in lst], cols)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for v, lst in frames.items():
        hv = [hypervolume(df[cols].to_numpy(float), lo, hi) for df in lst]
        rows.append({"speed_objective": v, "hv_measured_mean": np.mean(hv), "hv_measured_std": np.std(hv),
                     "seeds": len(hv)})
        d = lst[0]
        d = d[nondominated_mask(d[cols[:2]].to_numpy(float))].sort_values("latency_measured")
        ax.step(d["latency_measured"], 1 - d["f_error"], where="post", marker=".", label=v)
    ax.set(xlabel="measured ANE latency [ms]", ylabel="accuracy", title="RQ2: which speed objective?")
    ax.legend()
    _save(fig, "rq2")
    return pd.DataFrame(rows).sort_values("hv_measured_mean", ascending=False)


def rq3():
    runs = load_rq("rq3")
    if not runs:
        print("no rq3 runs")
        return None
    seq = []
    for seed, df in runs.get("arch_only", []):
        post = RUNS_DIR / "rq3" / "arch_only" / f"seed{seed}" / "posthoc.csv"
        if not _missing(post):
            seq.append((seed, pd.concat([df, pd.read_csv(post)], ignore_index=True)))
    if seq:
        runs["sequential"] = seq
    lo, hi = common_bounds([df for lst in runs.values() for _, df in lst])
    table = _hv_table(runs, lo, hi, reference="joint")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for v, lst in runs.items():
        fr = front(lst[0][1], cols=["f_error", "f_size"]).sort_values("f_size")
        ax.step(fr["f_size"], 1 - fr["f_error"], where="post", marker=".", label=v)
    ax.set(xlabel="size [MB]", ylabel="accuracy", title="RQ3: joint vs sequential compression")
    ax.legend()
    _save(fig, "rq3")
    return table


def rq4():
    runs = load_rq("rq4")
    if not runs:
        print("no rq4 runs")
        return None
    lo, hi = common_bounds([df for lst in runs.values() for _, df in lst])
    table = _hv_table(runs, lo, hi, reference="default")
    fig, ax = plt.subplots(figsize=(8, 4))
    t = table.sort_values("variant")
    ax.bar(t["variant"], t["hv_mean"], yerr=t["hv_std"], capsize=3)
    ax.set(ylabel="final hypervolume", title="RQ4: GA ablations (one change at a time)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "rq4")
    return table


# ---------------------------------------------------------------- RQ5-RQ7 and the final front

def rq5(path=RUNS_DIR / "rq5" / "standalone_40ep.csv", target="test_acc_pal"):
    """Supernet accuracy vs from-scratch accuracy. `target` is the column plotted; the table has all four
    (search-val / test, plain / after the same palettization)."""
    if _missing(path):
        return None
    df = pd.read_csv(path).dropna(subset=["supernet_acc"])
    rows = []
    for col in ["val_acc", "val_acc_pal", "test_acc", "test_acc_pal"]:
        tau, p = kendalltau(df["supernet_acc"], df[col])
        rows.append({"from_scratch": col, "kendall_tau": tau, "p": p,
                     "spearman": spearmanr(df["supernet_acc"], df[col])[0],
                     "mean": df[col].mean(), "min": df[col].min(), "max": df[col].max(), "n": len(df)})
    t = pd.DataFrame(rows).set_index("from_scratch")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(df["supernet_acc"], df[target])
    ax.set(xlabel="supernet accuracy (inherited weights)", ylabel=f"from scratch: {target}",
           title=f"RQ5: Kendall τ = {t.loc[target, 'kendall_tau']:.2f}, "
                 f"Spearman ρ = {t.loc[target, 'spearman']:.2f}")
    _save(fig, "rq5")
    return t


def rq6():
    """Do latencies rank the same way on Mac ANE, Mac CPU/GPU and iPhone?"""
    mac = RUNS_DIR / "final" / "front_validated_measured.csv"
    if _missing(mac):
        return None
    df = pd.read_csv(mac)
    iphone = RUNS_DIR / "iphone" / "iphone_latency.csv"
    if iphone.exists():
        ip = pd.read_csv(iphone)[["key", "latency_iphone_ms"]].dropna()
        df = df.merge(ip, on="key", how="left")
    cols = [c for c in df.columns if c.startswith("latency_")]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(cols)), [c.replace("latency_", "") for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), [c.replace("latency_", "") for c in cols])
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im)
    ax.set_title("RQ6: Spearman correlation of latencies")
    _save(fig, "rq6")
    return corr


def _gene_table(genomes):
    rows = []
    for g in genomes:
        d = G.describe(g)
        for s, st in enumerate(d["stages"]):
            for blk in st["blocks"]:
                rows.append({"stage": s, "kernel": blk["kernel"], "expand": blk["expand"], "se": blk["se"],
                             "bits": st["bits"], "depth": st["depth"]})
    return pd.DataFrame(rows)


def rq7(run_dir=RUNS_DIR / "final" / "nsga2" / "seed0"):
    """Which choices survive on the front, compared with everything the search evaluated?"""
    if _missing(Path(run_dir) / "done.json"):
        return None
    _, df = load_run(run_dir)
    fr = front(df)
    all_t = _gene_table([json.loads(s) for s in df["genome"]]).assign(set="all evaluated")
    fr_t = _gene_table([json.loads(s) for s in fr["genome"]]).assign(set="Pareto front")
    t = pd.concat([all_t, fr_t])
    summary = t.groupby(["set", "stage"]).agg(se_rate=("se", "mean"), kernel_mean=("kernel", "mean"),
                                              expand_mean=("expand", "mean"), bits_mean=("bits", "mean"),
                                              depth_mean=("depth", "mean")).reset_index()
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
    for i, col in enumerate(["se_rate", "kernel_mean", "expand_mean", "bits_mean"]):
        for name, d in summary.groupby("set"):
            ax[i].plot(d["stage"], d[col], marker="o", label=name)
        ax[i].set(xlabel="stage", title=col)
    ax[0].legend()
    _save(fig, "rq7")
    return summary


def final_front():
    """Trained front vs MobileNetV2 baselines: test accuracy (palettized) vs measured ANE latency."""
    tr, val = RUNS_DIR / "final" / "front_trained.csv", RUNS_DIR / "final" / "front_validated.csv"
    if _missing(tr) or _missing(val):
        return None
    df = pd.read_csv(tr).merge(pd.read_csv(val)[["key", "latency_measured", "size_mb"]], on="key")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    d = df.sort_values("latency_measured")
    ax.plot(d["latency_measured"], d["test_acc_pal"], "o-", label="evolved (NSGA-II)")
    bt, bl = RUNS_DIR / "final" / "baseline_trained.csv", RUNS_DIR / "baseline_latency.csv"
    if bt.exists() and bl.exists():
        b = pd.read_csv(bt).merge(pd.read_csv(bl)[["name", "latency_CPU_AND_NE"]], left_on="key", right_on="name")
        ax.scatter(b["latency_CPU_AND_NE"], b["test_acc"], marker="x", color="k", label="MobileNetV2")
    ax.set(xlabel="measured ANE latency on Mac [ms]", ylabel="test accuracy", title="Final front vs baseline")
    ax.legend()
    _save(fig, "final_front")
    return df
