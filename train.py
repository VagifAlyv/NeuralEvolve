"""Data and training. Runs on a GPU (Kaggle), works on the Mac too, only slower.

    python train.py supernetwork     train the supernetwork             -> runs/supernetwork.pt
    python train.py scratch          train the chosen networks of runs/front.csv and MobileNetV2
                                     from scratch                       -> runs/trained/*.pt

Both commands continue where they stopped if they are started again.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import model

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
RUNS_FOLDER = PROJECT_FOLDER / "runs"

TRAINING_RESOLUTION = 128
VALIDATION_SIZE = 1000    # training images kept apart to score networks during the search
CALIBRATION_SIZE = 2000   # training images used to recompute BatchNorm statistics
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STANDARD_DEVIATION = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ data

def convert_to_rgb(image):
    return image.convert("RGB")   # a few Imagenette images are black and white


TRAINING_TRANSFORM = transforms.Compose([
    convert_to_rgb,
    transforms.RandomResizedCrop(TRAINING_RESOLUTION, scale=(0.35, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.PILToTensor(),
])
EVALUATION_TRANSFORM = transforms.Compose([
    convert_to_rgb,
    transforms.Resize(146),
    transforms.CenterCrop(TRAINING_RESOLUTION),
    transforms.PILToTensor(),
])


def imagenette(split, transform):
    """Imagenette: 10 ImageNet classes, about 9500 training and 3900 test images."""
    already_downloaded = (DATA_FOLDER / "imagenette2-160").exists()
    return torchvision.datasets.Imagenette(str(DATA_FOLDER), split=split, size="160px",
                                           download=not already_downloaded, transform=transform)


def split_training_images():
    """The same random split every time: validation images, calibration images, training images."""
    DATA_FOLDER.mkdir(exist_ok=True)
    number_of_images = len(imagenette("train", None))
    order = torch.randperm(number_of_images, generator=torch.Generator().manual_seed(0)).tolist()
    validation = order[:VALIDATION_SIZE]
    training = order[VALIDATION_SIZE:]
    calibration = training[:CALIBRATION_SIZE]
    return validation, calibration, training


def training_loader(batch_size):
    validation, calibration, training = split_training_images()
    dataset = Subset(imagenette("train", TRAINING_TRANSFORM), training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4,
                      persistent_workers=True)


def load_into_memory(dataset):
    images = []
    labels = []
    for batch_images, batch_labels in DataLoader(dataset, batch_size=250, num_workers=4):
        images.append(batch_images)
        labels.append(batch_labels)
    return torch.cat(images), torch.cat(labels)


def evaluation_images():
    """Small image sets kept in memory as 8-bit tensors.
    validation and calibration come from the training split; test is the official test split
    and is used only for the final numbers."""
    validation, calibration, training = split_training_images()
    training_images = imagenette("train", EVALUATION_TRANSFORM)
    return {
        "validation": load_into_memory(Subset(training_images, validation)),
        "calibration": load_into_memory(Subset(training_images, calibration)),
        "test": load_into_memory(imagenette("val", EVALUATION_TRANSFORM)),
    }


def normalize(images, resolution, device):
    """8-bit images -> normalized floats on the device, resized to the resolution of the network."""
    images = images.to(device).float() / 255
    if images.shape[-1] != resolution:
        images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return (images - MEAN.to(device)) / STANDARD_DEVIATION.to(device)


@torch.no_grad()
def accuracy(network, images, labels, resolution, device):
    network.eval()
    correct = 0
    for start in range(0, len(labels), 500):
        batch = normalize(images[start:start + 500], resolution, device)
        predictions = network(batch).argmax(dim=1).cpu()
        correct += (predictions == labels[start:start + 500]).sum().item()
    return correct / len(labels)


# ------------------------------------------------------------------ training

def make_optimizer(network, learning_rate, steps):
    """SGD with a cosine learning rate and a short warm-up. No weight decay on BatchNorm and biases."""
    matrices = [parameter for parameter in network.parameters() if parameter.dim() > 1]
    vectors = [parameter for parameter in network.parameters() if parameter.dim() <= 1]
    optimizer = torch.optim.SGD([{"params": matrices, "weight_decay": 4e-5},
                                 {"params": vectors, "weight_decay": 0}],
                                lr=learning_rate, momentum=0.9, nesterov=True)
    warmup_steps = steps // 20

    def learning_rate_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    return optimizer, scheduler


def distillation_loss(student_logits, teacher_logits):
    """The student learns to copy the predictions of the teacher."""
    teacher_probabilities = teacher_logits.detach().softmax(dim=1)
    return F.kl_div(student_logits.log_softmax(dim=1), teacher_probabilities, reduction="batchmean")


def train_supernetwork(epochs, batch_size, learning_rate):
    """Sandwich rule: every batch trains the largest network on the labels, and the smallest network
    plus two random networks on the predictions of the largest one (in-place distillation)."""
    device = get_device()
    loader = training_loader(batch_size)
    supernetwork = model.SuperNetwork().to(device)
    optimizer, scheduler = make_optimizer(supernetwork, learning_rate, epochs * len(loader))
    checkpoint_path = RUNS_FOLDER / "supernetwork.pt"
    first_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        supernetwork.load_state_dict(checkpoint["weights"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first_epoch = checkpoint["epoch"] + 1
    generator = random.Random(first_epoch)

    for epoch in range(first_epoch, epochs):
        supernetwork.train()
        total_loss = 0
        for images, labels in loader:
            images = normalize(images, TRAINING_RESOLUTION, device)
            labels = labels.to(device)
            optimizer.zero_grad()

            largest_logits = supernetwork(images, model.LARGEST_GENOME)
            loss = F.cross_entropy(largest_logits, labels, label_smoothing=0.1)
            loss.backward()
            total_loss += loss.item()

            for genome in [model.SMALLEST_GENOME, model.random_genome(generator), model.random_genome(generator)]:
                resolution = genome[0]
                smaller_images = F.interpolate(images, size=(resolution, resolution), mode="bilinear",
                                               align_corners=False)
                distillation_loss(supernetwork(smaller_images, genome), largest_logits).backward()

            torch.nn.utils.clip_grad_norm_(supernetwork.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

        print(f"epoch {epoch + 1}/{epochs}  loss of the largest network {total_loss / len(loader):.3f}", flush=True)
        torch.save({"weights": supernetwork.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch}, checkpoint_path)


def train_from_scratch(network, resolution, epochs, batch_size, learning_rate):
    device = get_device()
    loader = training_loader(batch_size)
    network = network.to(device)
    optimizer, scheduler = make_optimizer(network, learning_rate, epochs * len(loader))
    for epoch in range(epochs):
        network.train()
        for images, labels in loader:
            images = normalize(images, resolution, device)
            loss = F.cross_entropy(network(images), labels.to(device), label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"  epoch {epoch + 1}/{epochs}  loss {loss.item():.3f}", flush=True)
    return network


def mobilenet(width):
    return torchvision.models.mobilenet_v2(width_mult=width, num_classes=model.NUMBER_OF_CLASSES)


def latency_of(row):
    return float(row["latency_ms"])


def choose_from_front(count):
    """`count` networks of the Pareto front, spread evenly from the fastest to the slowest."""
    with open(RUNS_FOLDER / "front.csv") as file:
        rows = list(csv.DictReader(file))
    rows.sort(key=latency_of)
    if len(rows) <= count:
        return rows
    positions = [round(number * (len(rows) - 1) / (count - 1)) for number in range(count)]
    return [rows[position] for position in positions]


def train_final_networks(count, epochs, batch_size, learning_rate):
    """Train the chosen networks of the front and MobileNetV2 (hand-designed baseline) from scratch."""
    output_folder = RUNS_FOLDER / "trained"
    output_folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for number, row in enumerate(choose_from_front(count)):
        genome = json.loads(row["genome"])
        jobs.append({"name": f"front_{number:02d}", "genome": genome, "width": None})
    for width in [0.35, 0.5, 0.75, 1.0]:
        jobs.append({"name": f"mobilenetv2_{width}", "genome": None, "width": width})

    images = evaluation_images()
    for job in jobs:
        path = output_folder / f"{job['name']}.pt"
        if path.exists():
            continue
        print(job["name"], flush=True)
        if job["genome"] is None:
            network = mobilenet(job["width"])
            resolution = TRAINING_RESOLUTION
        else:
            network = model.Network(job["genome"])
            resolution = job["genome"][0]
        network = train_from_scratch(network, resolution, epochs, batch_size, learning_rate)
        test_images, test_labels = images["test"]
        test_accuracy = accuracy(network, test_images, test_labels, resolution, get_device())
        print(f"  test accuracy in PyTorch {test_accuracy:.3f}", flush=True)
        job["weights"] = network.cpu().state_dict()
        torch.save(job, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the supernetwork or the final networks.")
    parser.add_argument("command", choices=["supernetwork", "scratch"])
    parser.add_argument("--epochs", type=int, default=None, help="default: 50 for the supernetwork, 40 from scratch")
    # On a 16 GB Mac use --batch-size 64 --learning-rate 0.05 (batch 128 does not fit in memory).
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--count", type=int, default=10, help="how many networks of the front to train")
    arguments = parser.parse_args()
    RUNS_FOLDER.mkdir(exist_ok=True)
    if arguments.command == "supernetwork":
        epochs = 50 if arguments.epochs is None else arguments.epochs
        train_supernetwork(epochs, arguments.batch_size, arguments.learning_rate)
    else:
        epochs = 40 if arguments.epochs is None else arguments.epochs
        train_final_networks(arguments.count, epochs, arguments.batch_size, arguments.learning_rate)
