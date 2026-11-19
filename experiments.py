"""All search experiments (RQ1-RQ4) and the final search. Every run writes runs/<rq>/<variant>/seed<k>/evals.csv.

Order (see README):
    python experiments.py rq1            # algorithms at equal budget, 5 seeds
    python experiments.py rq2            # speed objective: MACs vs params vs latency predictor
    python experiments.py rq2 --include-measured   # + real Core ML latency in the loop (slow, 1 seed)
    python experiments.py rq2-measure    # real Mac latency of every RQ2 front
    python experiments.py rq3            # joint vs architecture-only search
    python experiments.py rq3-post       # 'sequential' = architecture-only front + uniform 8/6/4 bits
    python experiments.py rq4            # GA ablations, 3 seeds
    python experiments.py final          # long NSGA-II run used for RQ5-RQ7
    python experiments.py validate-final # supernet accuracy + real latency of the final front

By default accuracy comes from the accuracy predictor (fast). Use --accuracy supernet to score
every candidate with the supernet instead (slower, ~10 s per new genome, cached).
"""
import argparse
import csv
import json
import random
import time

import numpy as np

import genome as G
from common import RUNS_DIR
from search import ALGORITHMS, Objective, nondominated_mask


def force_16_bits(g):
    """RQ3 'architecture only': compression genes are ignored, every stage stays at 16 bits."""
    g = list(g)
    for s in range(G.N_STAGES):
        g[G.stage_index(s) + 2] = G.BITS.index(16)
    return g


def set_all_bits(g, bits):
    g = list(g)
    for s in range(G.N_STAGES):
        g[G.stage_index(s) + 2] = G.BITS.index(bits)
    return g


def V(algo="nsga2", latency_mode="predictor", transform=None, seeds=None, budget=None, **params):
    return {"algo": algo, "latency_mode": latency_mode, "transform": transform, "seeds": seeds,
            "budget": budget, "params": params}


def grid(rq, args):
    n = G.GENOME_LEN
    if rq == "rq1":
        return {"nsga2": V("nsga2"), "random": V("random"), "regevo": V("regevo"), "local": V("local")}
    if rq == "rq2":
        out = {"macs": V(latency_mode="macs"), "params": V(latency_mode="params"),
               "predictor": V(latency_mode="predictor")}
        if args.include_measured:
            out["measured"] = V(latency_mode="measured", seeds=[0], budget=args.measured_budget, pop_size=20)
        return out
    if rq == "rq3":
        return {"joint": V(), "arch_only": V(transform=force_16_bits)}
    if rq == "rq4":
        return {"default": V(), "pop20": V(pop_size=20), "pop100": V(pop_size=100),
                "rate0.5": V(rate=0.5 / n), "rate2": V(rate=2 / n), "rate4": V(rate=4 / n),
                "one_point": V(crossover="one_point"), "stagewise": V(crossover="stagewise"),
                "no_repair": V(repair=False)}
    if rq == "final":
        return {"nsga2": V(pop_size=100, seeds=[0], budget=args.final_budget)}
    raise ValueError(rq)


DEFAULT_SEEDS = {"rq1": [0, 1, 2, 3, 4], "rq2": [0, 1, 2], "rq3": [0, 1, 2], "rq4": [0, 1, 2], "final": [0]}


class Components:
    """Loads the accuracy source, latency predictor and Core ML timer only when needed."""

    def __init__(self, args):
        self.args = args
        self._acc = self._lat = self._measured = self._evaluator = None

    def evaluator(self):
        if self._evaluator is None:
            from data import Data
            from evaluate import SubnetEvaluator
            from supernet import load_supernet
            from common import get_device
            device = get_device()
            self._evaluator = SubnetEvaluator(load_supernet(self.args.supernet, device),
                                              Data(need_train=False), device)
        return self._evaluator

    def acc_fn(self):
        if self._acc is None:
            if self.args.accuracy == "supernet":
                ev = self.evaluator()
                self._acc = lambda g: ev(g)["acc"]
            else:
                from predictors import Predictor
                self._acc = Predictor.load(RUNS_DIR / "accuracy_predictor.pt").predict
        return self._acc

    def latency_pred(self):
        if self._lat is None:
            from predictors import Predictor
            path = RUNS_DIR / "latency_predictor.pt"
            self._lat = Predictor.load(path) if path.exists() else None
        return self._lat

    def measured(self):
        if self._measured is None:
            from latency import MeasuredLatency
            self._measured = MeasuredLatency("CPU_AND_NE")
        return self._measured

    def objective(self, latency_mode, log_path=None, transform=None):
        if latency_mode == "predictor" and self.latency_pred() is None:
            raise SystemExit("runs/latency_predictor.pt missing: run `python predictors.py latency` first")
        return Objective(self.acc_fn(), latency_mode, self.latency_pred(),
                         self.measured() if latency_mode == "measured" else None, log_path, transform)


def run(rq, args, comps):
    variants = grid(rq, args)
    for name, v in variants.items():
        seeds = v["seeds"] or args.seeds or DEFAULT_SEEDS[rq]
        budget = v["budget"] or args.budget
        for seed in seeds:
            run_dir = RUNS_DIR / rq / name / f"seed{seed}"
            if (run_dir / "done.json").exists():
                print(f"skip {run_dir} (done)")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"== {rq}/{name}/seed{seed}  budget={budget}")
            t0 = time.time()
            obj = comps.objective(v["latency_mode"], run_dir / "evals.csv", v["transform"])
            rng = random.Random(seed)
            algo = ALGORITHMS[v["algo"]]
            if v["algo"] == "nsga2":
                algo(obj, budget, rng, seed=seed, **v["params"])
            else:
                algo(obj, budget, rng, **v["params"])
            obj.close()
            with open(run_dir / "done.json", "w") as f:
                json.dump({"rq": rq, "variant": name, "seed": seed, "budget": budget, "algo": v["algo"],
                           "latency_mode": v["latency_mode"], "accuracy": args.accuracy,
                           "transform": getattr(v["transform"], "__name__", None),
                           "params": v["params"], "evaluations": obj.n,
                           "minutes": round((time.time() - t0) / 60, 2)}, f, indent=2)


# ---------------------------------------------------------------- post-processing

def read_evals(run_dir):
    with open(run_dir / "done.json") as f:
        budget = json.load(f)["budget"]
    with open(run_dir / "evals.csv") as f:
        rows = list(csv.DictReader(f))[:budget]          # equal budget for every algorithm
    return rows


def front_rows(rows):
    F = np.array([[float(r["f_error"]), float(r["f_latency"]), float(r["f_size"])] for r in rows])
    keep = nondominated_mask(F)
    seen, out = set(), []
    for r, k in zip(rows, keep):
        if k and r["key"] not in seen:
            seen.add(r["key"])
            out.append(r)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rq2_measure(args, comps):
    """The real question of RQ2: how good are the fronts when latency is MEASURED on the ANE?"""
    meas = comps.measured()
    for run_dir in sorted((RUNS_DIR / "rq2").glob("*/seed*")):
        if not (run_dir / "done.json").exists():
            continue
        out = []
        for r in front_rows(read_evals(run_dir)):
            g = json.loads(r["genome"])
            out.append({"key": r["key"], "genome": r["genome"], "f_error": r["f_error"], "f_size": r["f_size"],
                        "macs": r["macs"], "params": r["params"], "latency_pred": r["latency_pred"],
                        "latency_measured": meas(g)})
        write_csv(run_dir / "front_measured.csv", out)
        print(f"{run_dir}: measured {len(out)} front members")


def rq3_post(args, comps):
    """Sequential baseline: take the architecture-only front, then compress every stage uniformly."""
    for run_dir in sorted((RUNS_DIR / "rq3" / "arch_only").glob("seed*")):
        if not (run_dir / "done.json").exists():
            continue
        obj = comps.objective("predictor", run_dir / "posthoc.csv")
        for r in front_rows(read_evals(run_dir)):
            for bits in (8, 6, 4):
                obj(set_all_bits(json.loads(r["genome"]), bits))
        obj.close()
        print(f"{run_dir}: {obj.n} post-hoc evaluations")


def validate_final(args, comps):
    """Score the final front with the supernet (instead of the predictor) and time it on the Mac."""
    run_dir = RUNS_DIR / "final" / "nsga2" / "seed0"
    ev, meas = comps.evaluator(), comps.measured()
    out = []
    for r in front_rows(read_evals(run_dir)):
        g = json.loads(r["genome"])
        rec = ev(g)
        out.append({"key": r["key"], "genome": r["genome"], "acc_predicted": r["acc"], "acc_supernet": rec["acc"],
                    "latency_pred": r["latency_pred"], "latency_measured": meas(g), "size_mb": rec["size_mb"],
                    "macs": rec["macs"], "params": rec["params"]})
        print(out[-1]["key"], out[-1]["acc_supernet"], out[-1]["latency_measured"])
    write_csv(RUNS_DIR / "final" / "front_validated.csv", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["rq1", "rq2", "rq2-measure", "rq3", "rq3-post", "rq4", "final",
                                       "validate-final"])
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--final-budget", type=int, default=5000)
    p.add_argument("--measured-budget", type=int, default=300)
    p.add_argument("--include-measured", action="store_true")
    p.add_argument("--accuracy", default="predictor", choices=["predictor", "supernet"])
    p.add_argument("--supernet", default=str(RUNS_DIR / "supernet" / "main.pt"))
    args = p.parse_args()
    comps = Components(args)
    post = {"rq2-measure": rq2_measure, "rq3-post": rq3_post, "validate-final": validate_final}
    if args.command in post:
        post[args.command](args, comps)
    else:
        run(args.command, args, comps)
