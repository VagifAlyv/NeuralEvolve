"""Imagenette data.

- train_loader:  augmented training images (the official train split minus the search-validation part)
- search_val:    ~1000 held-out train images, used to score candidates DURING the search
- calib:         ~2000 train images used only to recalibrate BatchNorm of an extracted subnet
- test:          the official Imagenette val split, used ONLY for final reported numbers

Images are stored as uint8 at 128x128; `prepare` normalizes them and resizes to the genome's resolution.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from common import DATA_DIR

TRAIN_RES = 128
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_rgb(img):
    return img.convert("RGB")   # a few Imagenette images are grayscale


def train_transform(dataset):
    if dataset == "cifar10":    # debugging only: upscale the 32x32 images
        return T.Compose([T.Resize(TRAIN_RES), T.RandomHorizontalFlip(), T.PILToTensor()])
    return T.Compose([to_rgb, T.RandomResizedCrop(TRAIN_RES, scale=(0.35, 1.0)),
                      T.RandomHorizontalFlip(), T.PILToTensor()])


def eval_transform(dataset):
    if dataset == "cifar10":
        return T.Compose([T.Resize(TRAIN_RES), T.PILToTensor()])
    return T.Compose([to_rgb, T.Resize(int(TRAIN_RES * 1.14)), T.CenterCrop(TRAIN_RES), T.PILToTensor()])


def prepare(x, device, resolution):
    """uint8 [N,3,128,128] -> normalized float on `device`, resized to `resolution`."""
    x = x.to(device, non_blocking=True).float().div_(255)
    if x.shape[-1] != resolution:
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return (x - MEAN.to(device)) / STD.to(device)


def _raw_datasets(dataset):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if dataset == "cifar10":
        make = lambda train, tf: torchvision.datasets.CIFAR10(str(DATA_DIR), train=train, download=True, transform=tf)
        return make(True, train_transform(dataset)), make(True, eval_transform(dataset)), make(False, eval_transform(dataset))
    # torchvision refuses to download into an existing folder, so only download the first time
    download = not (DATA_DIR / "imagenette2-160").exists()
    make = lambda split, tf, dl: torchvision.datasets.Imagenette(str(DATA_DIR), split=split, size="160px",
                                                                 download=dl, transform=tf)
    train_aug = make("train", train_transform(dataset), download)
    train_eval = make("train", eval_transform(dataset), False)
    test = make("val", eval_transform(dataset), False)
    return train_aug, train_eval, test


def _to_tensors(ds, cache_path, workers):
    """Load a whole (small) dataset into memory once and keep it as a .pt file."""
    if cache_path.exists():
        return torch.load(cache_path)
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=256, num_workers=workers):
        xs.append(x)
        ys.append(torch.as_tensor(y))
    out = (torch.cat(xs), torch.cat(ys))
    torch.save(out, cache_path)
    return out


class Data:
    def __init__(self, dataset="imagenette", batch_size=128, workers=4, n_search_val=1000, n_calib=2048,
                 seed=0, need_train=True):
        train_aug, train_eval, test = _raw_datasets(dataset)
        perm = torch.randperm(len(train_aug), generator=torch.Generator().manual_seed(seed)).tolist()
        val_idx, train_idx = perm[:n_search_val], perm[n_search_val:]
        tag = f"{dataset}_seed{seed}_val{n_search_val}"

        self.train_loader = None
        if need_train:
            self.train_loader = DataLoader(Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True,
                                           num_workers=workers, drop_last=True, persistent_workers=workers > 0)
        self.search_val = _to_tensors(Subset(train_eval, val_idx), DATA_DIR / f"{tag}_searchval.pt", workers)
        self.calib = _to_tensors(Subset(train_eval, train_idx[:n_calib]),
                                 DATA_DIR / f"{tag}_calib{n_calib}.pt", workers)[0]
        self.test = _to_tensors(test, DATA_DIR / f"{dataset}_test.pt", workers)
