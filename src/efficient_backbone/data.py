from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
NUM_CLASSES = {"cifar100": 100, "synthetic": 100}


class SyntheticImages(Dataset):
    """Fixed random tensors so smoke tests do not download CIFAR."""

    def __init__(self, length: int, img_size: int, num_classes: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand(length, 3, img_size, img_size, generator=g)
        self.targets = torch.randint(0, num_classes, (length,), generator=g)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.images[index], int(self.targets[index])


def _train_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def build_loaders(cfg: dict, smoke: bool) -> tuple[DataLoader, DataLoader, int]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    name = "synthetic" if smoke else data_cfg["name"]
    if name not in NUM_CLASSES:
        known = ", ".join(sorted(NUM_CLASSES))
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}")
    num_classes = NUM_CLASSES[name]
    img_size = int(data_cfg.get("img_size", 32))
    batch_size = int(train_cfg["batch_size"])
    num_workers = 0 if smoke else int(train_cfg.get("num_workers", 4))
    pin = not smoke

    if name == "synthetic":
        length = int(data_cfg.get("synthetic_size", 32))
        train_set = SyntheticImages(length, img_size, num_classes, seed=int(cfg.get("seed", 0)))
        eval_set = SyntheticImages(length, img_size, num_classes, seed=int(cfg.get("seed", 0)) + 1)
    else:
        root = data_cfg["root"]
        train_set = datasets.CIFAR100(
            root, train=True, download=True, transform=_train_transform(img_size)
        )
        eval_set = datasets.CIFAR100(
            root, train=False, download=True, transform=_eval_transform()
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, **loader_kwargs
    )
    eval_loader = DataLoader(
        eval_set, shuffle=False, drop_last=False, **loader_kwargs
    )
    return train_loader, eval_loader, num_classes
