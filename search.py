"""The search. Runs on the Mac, because every network is measured on the Apple Neural Engine.

    python search.py search     NSGA-II with the trained supernetwork    -> runs/evaluations.csv, runs/front.csv
    python search.py final      measure the networks trained from scratch -> runs/final.csv, runs/iphone/

How one network (one genome) is scored:
    1. take its weights out of the supernetwork
    2. recompute its BatchNorm statistics on 2000 training images
    3. convert it to Core ML (float16) and compress every stage to the bits in the genome
    4. accuracy:  the Core ML model classifies the 1000 validation images on the Neural Engine
       latency:   median time of one prediction on the Neural Engine
       size:      size of the Core ML model file in MB
The search minimizes three objectives: error (1 - accuracy), latency and size.
"""
import argparse
import csv
import json
import random
import re
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import coremltools
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, get_weights_metadata, palettize_weights
from torch.nn.utils.fusion import fuse_conv_bn_eval

import model
from train import RUNS_FOLDER, TRAINING_RESOLUTION, evaluation_images, get_device, mobilenet, normalize

PROJECT_FOLDER = Path(__file__).resolve().parent
FIGURES_FOLDER = PROJECT_FOLDER / "figures"
SMALLEST_COMPRESSED_WEIGHT = 2048   # Core ML leaves smaller weights uncompressed
STAGE_NAME = re.compile(r"stages_(\d+)_")   # Core ML weight names look like "stages_3_1_project_weight"


# ------------------------------------------------------------------ scoring one network

@torch.no_grad()
def recalibrate_batchnorm(network, calibration_images, resolution, device):
    """A network taken out of the supernetwork has no BatchNorm statistics yet: compute them."""
    for layer in network.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.reset_running_stats()
            layer.momentum = None   # None = plain average over all batches
    network.train()
    for start in range(0, len(calibration_images), 250):
        network(normalize(calibration_images[start:start + 250], resolution, device))
    network.eval()


def fold_batchnorm(network):
    """Merge every BatchNorm into the convolution before it (layer "x" and "x_bn").
    Core ML would do it too, but then the weights lose their names and we could not find the stages."""
    network = network.cpu().eval()
    for layer in list(network.modules()):
        for name in list(layer._modules):
            if name.endswith("_bn"):
                convolution_name = name[:-len("_bn")]
                convolution = getattr(layer, convolution_name)
                setattr(layer, convolution_name, fuse_conv_bn_eval(convolution, getattr(layer, name)))
                setattr(layer, name, nn.Identity())
    return network


def convert_to_coreml(network, resolution, stage_bits):
    """PyTorch -> Core ML in float16. Then every stage is compressed with k-means palettization:
    each weight is replaced by one of 2^bits shared values. stage_bits = None means no compression."""
    example = torch.rand(1, 3, resolution, resolution)
    traced = torch.jit.trace(network, example)
    coreml_model = coremltools.convert(
        traced,
        inputs=[coremltools.TensorType(name="image", shape=example.shape)],
        outputs=[coremltools.TensorType(name="logits")],
        convert_to="mlprogram",
        compute_precision=coremltools.precision.FLOAT16,
        minimum_deployment_target=coremltools.target.iOS17,
    )
    if stage_bits is None:
        return coreml_model
    settings = {}
    for weight_name in get_weights_metadata(coreml_model, weight_threshold=SMALLEST_COMPRESSED_WEIGHT):
        match = STAGE_NAME.search(weight_name)
        if match is None:
            continue   # stem, head and classifier stay at 16 bits
        bits = stage_bits[int(match.group(1))]
        if bits < 16:
            settings[weight_name] = OpPalettizerConfig(mode="kmeans", nbits=bits,
                                                       weight_threshold=SMALLEST_COMPRESSED_WEIGHT)
    if not settings:
        return coreml_model
    return palettize_weights(coreml_model, OptimizationConfig(op_name_configs=settings))


def folder_size_mb(folder):
    total_bytes = 0
    for file in folder.rglob("*"):
        if file.is_file():
            total_bytes += file.stat().st_size
    return total_bytes / 1e6


def measure_on_neural_engine(coreml_model, images, labels, resolution, save_to=None):
    """Accuracy, latency and size of a Core ML model, running on the Neural Engine."""
    temporary_folder = Path(tempfile.mkdtemp())
    path = temporary_folder / "model.mlpackage"
    if save_to is not None:
        path = save_to
    coreml_model.save(str(path))
    size_mb = folder_size_mb(path)
    loaded = coremltools.models.MLModel(str(path), compute_units=coremltools.ComputeUnit.CPU_AND_NE)

    inputs = normalize(images, resolution, torch.device("cpu")).numpy()
    correct = 0
    for image, label in zip(inputs, labels.tolist()):
        logits = loaded.predict({"image": image[numpy.newaxis]})["logits"]
        if int(logits.argmax()) == label:
            correct += 1

    times = []
    for repeat in range(50):
        start = time.perf_counter()
        loaded.predict({"image": inputs[:1]})
        times.append((time.perf_counter() - start) * 1000)
    shutil.rmtree(temporary_folder, ignore_errors=True)
    return {"accuracy": correct / len(labels), "latency_ms": statistics.median(times), "size_mb": size_mb}


class Scorer:
    """genome -> {accuracy, latency_ms, size_mb}. Every result is written to runs/evaluations.csv,
    and results already in that file are reused, so a stopped search can simply be started again."""

    columns = ["genome", "accuracy", "latency_ms", "size_mb", "seconds"]

    def __init__(self, supernetwork_path):
        self.device = get_device()
        self.supernetwork = model.SuperNetwork()
        self.supernetwork.load_state_dict(torch.load(supernetwork_path, map_location="cpu")["weights"])
        self.images = evaluation_images()
        self.log_path = RUNS_FOLDER / "evaluations.csv"
        self.results = {}
        if self.log_path.exists():
            with open(self.log_path) as file:
                for row in csv.DictReader(file):
                    self.results[row["genome"]] = {"accuracy": float(row["accuracy"]),
                                                   "latency_ms": float(row["latency_ms"]),
                                                   "size_mb": float(row["size_mb"])}
        else:
            with open(self.log_path, "w", newline="") as file:
                csv.writer(file).writerow(self.columns)

    def __call__(self, genome):
        key = json.dumps(genome)
        if key in self.results:
            return self.results[key]
        start = time.time()
        resolution, stages = model.describe(genome)
        stage_bits = [stage["bits"] for stage in stages]
        network = model.extract(self.supernetwork, genome).to(self.device)
        calibration_images, calibration_labels = self.images["calibration"]
        recalibrate_batchnorm(network, calibration_images, resolution, self.device)
        coreml_model = convert_to_coreml(fold_batchnorm(network), resolution, stage_bits)
        validation_images, validation_labels = self.images["validation"]
        result = measure_on_neural_engine(coreml_model, validation_images, validation_labels, resolution)
        self.results[key] = result
        with open(self.log_path, "a", newline="") as file:
            csv.writer(file).writerow([key, result["accuracy"], result["latency_ms"], result["size_mb"],
                                       round(time.time() - start, 1)])
        return result


def objectives(result):
    """All three are minimized."""
    return (1 - result["accuracy"], result["latency_ms"], result["size_mb"])


# ------------------------------------------------------------------ NSGA-II

def dominates(first, second):
    """First is at least as good in every objective and better in at least one."""
    no_worse = all(mine <= theirs for mine, theirs in zip(first, second))
    strictly_better = any(mine < theirs for mine, theirs in zip(first, second))
    return no_worse and strictly_better


def sort_into_fronts(scores):
    """Front 0 = solutions nobody dominates, front 1 = dominated only by front 0, and so on."""
    dominated_solutions = [[] for score in scores]
    domination_count = [0 for score in scores]
    fronts = [[]]
    for first in range(len(scores)):
        for second in range(len(scores)):
            if dominates(scores[first], scores[second]):
                dominated_solutions[first].append(second)
            elif dominates(scores[second], scores[first]):
                domination_count[first] += 1
        if domination_count[first] == 0:
            fronts[0].append(first)
    while fronts[-1]:
        next_front = []
        for solution in fronts[-1]:
            for dominated in dominated_solutions[solution]:
                domination_count[dominated] -= 1
                if domination_count[dominated] == 0:
                    next_front.append(dominated)
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(scores, front):
    """How empty the space around each solution of a front is. Larger = more worth keeping."""
    distance = {solution: 0.0 for solution in front}
    for objective in range(len(scores[front[0]])):
        pairs = [(scores[solution][objective], solution) for solution in front]
        pairs.sort()
        ordered = [solution for value, solution in pairs]
        lowest = scores[ordered[0]][objective]
        highest = scores[ordered[-1]][objective]
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        if highest == lowest:
            continue
        for position in range(1, len(ordered) - 1):
            gap = scores[ordered[position + 1]][objective] - scores[ordered[position - 1]][objective]
            distance[ordered[position]] += gap / (highest - lowest)
    return distance


def rank_population(scores):
    """For every solution: (front number, crowding distance)."""
    rank = {}
    for front_number, front in enumerate(sort_into_fronts(scores)):
        for solution, distance in crowding_distance(scores, front).items():
            rank[solution] = (front_number, distance)
    return rank


def better(first, second, rank):
    """Lower front wins; in the same front, the larger crowding distance wins."""
    first_front, first_distance = rank[first]
    second_front, second_distance = rank[second]
    if first_front != second_front:
        return first_front < second_front
    return first_distance > second_distance


def tournament(rank, generator):
    """Pick two random solutions, keep the better one."""
    first = generator.randrange(len(rank))
    second = generator.randrange(len(rank))
    if better(first, second, rank):
        return first
    return second


def nsga2(score, population_size, generations, seed):
    generator = random.Random(seed)
    population = [model.random_genome(generator) for number in range(population_size)]
    scores = [objectives(score(genome)) for genome in population]

    for generation in range(generations):
        rank = rank_population(scores)
        children = []
        while len(children) < population_size:
            mother = population[tournament(rank, generator)]
            father = population[tournament(rank, generator)]
            child = model.mutate(model.crossover(mother, father, generator), generator)
            if child not in population and child not in children:
                children.append(child)
        everyone = population + children
        everyone_scores = scores + [objectives(score(child)) for child in children]

        # keep the best half: whole fronts first, the last front by crowding distance
        survivors = []
        for front in sort_into_fronts(everyone_scores):
            if len(survivors) + len(front) <= population_size:
                survivors.extend(front)
            else:
                distance = crowding_distance(everyone_scores, front)
                pairs = [(distance[solution], solution) for solution in front]
                pairs.sort(reverse=True)
                most_isolated = [solution for value, solution in pairs]
                survivors.extend(most_isolated[:population_size - len(survivors)])
                break
        population = [everyone[solution] for solution in survivors]
        scores = [everyone_scores[solution] for solution in survivors]
        best_accuracy = 1 - min(error for error, latency, size in scores)
        print(f"generation {generation + 1}/{generations}  best accuracy {best_accuracy:.3f}", flush=True)


def save_front(scorer):
    """The Pareto front of every network evaluated during the search -> runs/front.csv and a figure."""
    genomes = list(scorer.results)
    results = [scorer.results[genome] for genome in genomes]
    front = sort_into_fronts([objectives(result) for result in results])[0]
    with open(RUNS_FOLDER / "front.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["genome", "accuracy", "latency_ms", "size_mb"])
        for solution in front:
            result = results[solution]
            writer.writerow([genomes[solution], result["accuracy"], result["latency_ms"], result["size_mb"]])
    print(f"{len(front)} networks on the Pareto front -> runs/front.csv")

    plt.figure(figsize=(6, 4))
    plt.scatter([result["latency_ms"] for result in results], [result["accuracy"] for result in results],
                s=6, color="lightgray", label="all evaluated networks")
    plotted = plt.scatter([results[solution]["latency_ms"] for solution in front],
                          [results[solution]["accuracy"] for solution in front],
                          c=[results[solution]["size_mb"] for solution in front], s=20, label="Pareto front")
    plt.colorbar(plotted, label="size (MB)")
    plt.xlabel("latency on the Neural Engine (ms)")
    plt.ylabel("validation accuracy (supernetwork)")
    plt.legend()
    plt.tight_layout()
    FIGURES_FOLDER.mkdir(exist_ok=True)
    plt.savefig(FIGURES_FOLDER / "front.png", dpi=150)


# ------------------------------------------------------------------ final networks

def measure_final_networks():
    """Convert every network trained from scratch to Core ML, measure it on the test images,
    and keep the .mlpackage files for the iPhone (Xcode -> Performance report)."""
    iphone_folder = RUNS_FOLDER / "iphone"
    iphone_folder.mkdir(parents=True, exist_ok=True)
    test_images, test_labels = evaluation_images()["test"]
    rows = []
    for path in sorted((RUNS_FOLDER / "trained").glob("*.pt")):
        job = torch.load(path, map_location="cpu")
        if job["genome"] is None:
            network = mobilenet(job["width"])
            resolution = TRAINING_RESOLUTION
            stage_bits = None
        else:
            network = model.Network(job["genome"])
            resolution, stages = model.describe(job["genome"])
            stage_bits = [stage["bits"] for stage in stages]
        network.load_state_dict(job["weights"])
        coreml_model = convert_to_coreml(fold_batchnorm(network), resolution, stage_bits)
        save_to = iphone_folder / f"{job['name']}.mlpackage"
        shutil.rmtree(save_to, ignore_errors=True)
        result = measure_on_neural_engine(coreml_model, test_images, test_labels, resolution, save_to)
        result["name"] = job["name"]
        rows.append(result)
        print(f"{job['name']:18s} test accuracy {result['accuracy']:.3f}  "
              f"latency {result['latency_ms']:.2f} ms  size {result['size_mb']:.2f} MB", flush=True)

    with open(RUNS_FOLDER / "final.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "accuracy", "latency_ms", "size_mb"])
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(6, 4))
    for prefix, label in [("front", "ours (searched)"), ("mobilenetv2", "MobileNetV2")]:
        chosen = [row for row in rows if row["name"].startswith(prefix)]
        plt.scatter([row["latency_ms"] for row in chosen], [row["accuracy"] for row in chosen], label=label)
    plt.xlabel("latency on the Neural Engine (ms)")
    plt.ylabel("test accuracy (trained from scratch)")
    plt.legend()
    plt.tight_layout()
    FIGURES_FOLDER.mkdir(exist_ok=True)
    plt.savefig(FIGURES_FOLDER / "final.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search networks for the Neural Engine.")
    parser.add_argument("command", choices=["search", "final"])
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    RUNS_FOLDER.mkdir(exist_ok=True)
    if arguments.command == "search":
        scorer = Scorer(RUNS_FOLDER / "supernetwork.pt")
        nsga2(scorer, arguments.population, arguments.generations, arguments.seed)
        save_front(scorer)
    else:
        measure_final_networks()
