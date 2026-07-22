"""Dataset and dataloader helpers for the CIFAR reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class CIFARLoaders:
    train: DataLoader
    id_test: DataLoader
    ood_test: DataLoader


def cifar10_cifar100_loaders(
    data_dir: str | Path,
    *,
    batch_size: int = 200,
    eval_batch_size: int = 512,
    num_workers: int = 2,
    seed: int = 100,
    download: bool = True,
    train_augmentation: bool = True,
) -> CIFARLoaders:
    data_dir = Path(data_dir)
    train_transform = transforms.Compose(
        ([transforms.RandomHorizontalFlip(p=0.5)] if train_augmentation else [])
        + [transforms.ToTensor()]
    )
    test_transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(
        data_dir / "cifar10", train=True, download=download, transform=train_transform
    )
    id_test = datasets.CIFAR10(
        data_dir / "cifar10", train=False, download=download, transform=test_transform
    )
    ood_test = datasets.CIFAR100(
        data_dir / "cifar100", train=False, download=download, transform=test_transform
    )
    generator = torch.Generator().manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return CIFARLoaders(
        train=DataLoader(
            train_set, batch_size=batch_size, shuffle=True, generator=generator, **common
        ),
        id_test=DataLoader(id_test, batch_size=eval_batch_size, shuffle=False, **common),
        ood_test=DataLoader(ood_test, batch_size=eval_batch_size, shuffle=False, **common),
    )
