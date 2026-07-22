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
    train_batch_size: int = 200,
    id_batch_size: int = 200,
    ood_batch_size: int = 4000,
    num_workers: int = 0,
    seed: int = 100,
    download: bool = True,
    train_augmentation: bool = True,
    shuffle_train: bool = True,
    shuffle_id: bool = True,
    shuffle_ood: bool = True,
    historical_rng: bool = True,
) -> CIFARLoaders:
    """Build CIFAR-10 train/test and CIFAR-100 test loaders.

    ``historical_rng=True`` intentionally avoids a private DataLoader generator.
    This mirrors the original script, in which the sampler and transforms consumed
    the process-wide PyTorch RNG seeded before loader construction.
    """

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

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    generator = None if historical_rng else torch.Generator().manual_seed(seed)

    def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
        kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **common)
        if generator is not None:
            kwargs["generator"] = generator
        return DataLoader(**kwargs)

    return CIFARLoaders(
        train=make_loader(train_set, train_batch_size, shuffle_train),
        id_test=make_loader(id_test, id_batch_size, shuffle_id),
        ood_test=make_loader(ood_test, ood_batch_size, shuffle_ood),
    )
