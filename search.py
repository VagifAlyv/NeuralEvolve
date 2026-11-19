"""Search algorithms. All of them minimize three objectives: [error, latency, size_mb].

- nsga2:      NSGA-II from pymoo with our own genome operators
- random:     random search (baseline)
- regevo:     regularized (aging) evolution, multi-objective tournament (Real et al. 2019)
- local:      Pareto local search: mutate one gene of an archive member

Every algorithm only calls `objective(g)`. The Objective counts and logs every call, so all
algorithms are compared at the same number of evaluations (the budget).
"""
import csv
import json
import time
from collections import deque

import numpy as np

import genome as G

LOG_FIELDS = ["eval", "key", "genome", "acc", "latency", "latency_pred", "macs", "params", "size_mb",
              "f_error", "f_latency", "f_size", "seconds"]


class Objective:
    """genome -> np.array([error, latency, size_mb]), logging every call to a CSV.

    acc_fn:        genome -> accuracy in [0, 1] (accuracy predictor or supernet)
    latency_mode:  'predictor' (ms), 'macs' (millions), 'params' (millions) or 'measured' (ms, Core ML)
    transform:     optional genome -> genome applied before evaluating (e.g. force 16-bit for RQ3)
    """

    def __init__(self, acc_fn, latency_mode="predictor", latency_pred=None, measured=None,
                 log_path=None, transform=None):
        self.acc_fn, self.mode, self.latency_pred, self.measured = acc_fn, latency_mode, latency_pred, measured
        self.transform, self.n = transform, 0
        self._file = self._writer = None
        if log_path is not None:
            self._file = open(log_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=LOG_FIELDS)
            self._writer.writeheader()

    def __call__(self, g):
        t0 = time.time()
        g = [int(v) for v in g]
        if self.transform is not None:
            g = self.transform(g)
        acc = float(np.clip(self.acc_fn(g), 0.0, 1.0))
        costs = G.cost_measures(g)
        lat_pred = self.latency_pred.predict(g) if self.latency_pred is not None else float("nan")
        latency = {"predictor": lambda: lat_pred,
                   "macs": lambda: costs["macs"] / 1e6,
                   "params": lambda: costs["params"] / 1e6,
                   "measured": lambda: self.measured(g)}[self.mode]()
        f = np.array([1.0 - acc, latency, costs["size_mb"]])
        self.n += 1
        if self._writer is not None:
            self._writer.writerow({"eval": self.n, "key": G.genome_key(g), "genome": json.dumps(g), "acc": acc,
                                   "latency": latency, "latency_pred": lat_pred, **costs,
                                   "f_error": f[0], "f_latency": f[1], "f_size": f[2],
                                   "seconds": round(time.time() - t0, 3)})
            self._file.flush()
        return f

    def close(self):
        if self._file is not None:
            self._file.close()


# ---------------------------------------------------------------- Pareto helpers

def dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_mask(F):
    F = np.asarray(F, dtype=float)
    mask = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        better_eq = np.all(F <= F[i], axis=1)
        strictly = np.any(F < F[i], axis=1)
        if np.any(better_eq & strictly):
            mask[i] = False
    return mask


def hypervolume(F, lower, upper):
    """Hypervolume of the points F after scaling every objective to [0, 1] with fixed bounds.
    Reference point = (1, 1, 1); points outside the bounds contribute nothing."""
    from pymoo.indicators.hv import HV
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return 0.0
    Z = (F - np.asarray(lower)) / (np.asarray(upper) - np.asarray(lower))
    Z = Z[np.all(Z < 1.0, axis=1)]
    if len(Z) == 0:
        return 0.0
    Z = Z[nondominated_mask(Z)]
    return float(HV(ref_point=np.ones(F.shape[1]))(Z))


# ---------------------------------------------------------------- operators shared by the algorithms

def mutate_changed(g, rate, rng, tries=50):
    """Mutate until the NETWORK changes (a change in an inactive gene alone does not count)."""
    key = G.genome_key(g)
    for _ in range(tries):
        child = G.mutate(g, rate, rng)
        if G.genome_key(child) != key:
            return child
    return mutate_one(g, rng)


def mutate_one(g, rng):
    """Change exactly one active gene to a different option."""
    active = [i for i, a in enumerate(G.active_mask(g)) if a]
    i = rng.choice(active)
    child = list(g)
    child[i] = rng.choice([v for v in range(G.N_CHOICES[i]) if v != g[i]])
    return child


CROSSOVERS = {"uniform": G.crossover_uniform, "one_point": G.crossover_one_point,
              "stagewise": G.crossover_stagewise}


# ---------------------------------------------------------------- algorithms

def random_search(objective, budget, rng):
    for _ in range(budget):
        objective(G.random_genome(rng))


def regularized_evolution(objective, budget, rng, pop_size=50, sample_size=10, rate=None):
    """Aging evolution: the OLDEST individual dies, not the worst. The parent is a random
    non-dominated member of a random sample (multi-objective tournament)."""
    pop = deque()
    for _ in range(min(pop_size, budget)):
        g = G.random_genome(rng)
        pop.append((g, objective(g)))
    for _ in range(budget - len(pop)):
        sample = rng.sample(list(pop), min(sample_size, len(pop)))
        nd = np.flatnonzero(nondominated_mask([f for _, f in sample]))
        parent = sample[int(rng.choice(list(nd)))][0]
        child = mutate_changed(parent, rate, rng)
        pop.append((child, objective(child)))
        pop.popleft()


def pareto_local_search(objective, budget, rng, n_init=10):
    """Keep an archive of non-dominated genomes; repeatedly change one gene of a random member."""
    archive = []

    def add(g, f):
        if any(dominates(a, f) or np.array_equal(a, f) for _, a in archive):
            return
        archive[:] = [(h, a) for h, a in archive if not dominates(f, a)] + [(g, f)]

    for _ in range(min(n_init, budget)):
        g = G.random_genome(rng)
        add(g, objective(g))
    for _ in range(budget - min(n_init, budget)):
        parent = rng.choice(archive)[0]
        child = mutate_one(parent, rng)
        add(child, objective(child))


def nsga2(objective, budget, rng, seed, pop_size=50, crossover="uniform", crossover_prob=0.9, rate=None,
          repair=True):
    """NSGA-II from pymoo with our sampling, crossover, mutation and (optional) repair.
    pymoo finishes the current generation, so it may go slightly over budget; analysis cuts every
    log at exactly `budget` evaluations."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.crossover import Crossover
    from pymoo.core.duplicate import ElementwiseDuplicateElimination
    from pymoo.core.mutation import Mutation
    from pymoo.core.problem import Problem
    from pymoo.core.repair import NoRepair, Repair
    from pymoo.core.sampling import Sampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    n_choices = np.array(G.N_CHOICES)
    cross_fn = CROSSOVERS[crossover]
    mut_rate = 1.0 / G.GENOME_LEN if rate is None else rate

    class GenomeProblem(Problem):
        def __init__(self):
            super().__init__(n_var=G.GENOME_LEN, n_obj=3, xl=np.zeros(G.GENOME_LEN), xu=n_choices - 1, vtype=int)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.array([objective(x) for x in X])

    class GenomeSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            return np.array([G.random_genome(rng) for _ in range(n_samples)])

    class GenomeCrossover(Crossover):
        def __init__(self):
            super().__init__(n_parents=2, n_offsprings=2, prob=crossover_prob)

        def _do(self, problem, X, **kwargs):
            _, n_matings, n_var = X.shape
            Y = np.empty((2, n_matings, n_var), dtype=X.dtype)
            for k in range(n_matings):
                a, b = list(X[0, k]), list(X[1, k])
                Y[0, k], Y[1, k] = cross_fn(a, b, rng), cross_fn(b, a, rng)
            return Y

    class ResetMutation(Mutation):
        def _do(self, problem, X, **kwargs):
            X = X.copy()
            for row in X:
                for i in range(len(row)):
                    if rng.random() < mut_rate:
                        row[i] = rng.randrange(int(n_choices[i]))
            return X

    class CanonicalRepair(Repair):
        def _do(self, problem, X, **kwargs):
            return np.array([G.repair(list(x)) for x in X])

    class SameNetwork(ElementwiseDuplicateElimination):
        def is_equal(self, a, b):
            return G.genome_key(a.X) == G.genome_key(b.X)

    algorithm = NSGA2(pop_size=pop_size, sampling=GenomeSampling(), crossover=GenomeCrossover(),
                      mutation=ResetMutation(), repair=CanonicalRepair() if repair else NoRepair(),
                      eliminate_duplicates=SameNetwork())
    minimize(GenomeProblem(), algorithm, get_termination("n_eval", budget), seed=seed, verbose=False)


ALGORITHMS = {"nsga2": nsga2, "random": random_search, "regevo": regularized_evolution,
              "local": pareto_local_search}
