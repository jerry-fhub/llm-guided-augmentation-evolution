from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    num_classes: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


DATASET_META: dict[str, DatasetMeta] = {
    "fake": DatasetMeta("fake", 10, 32, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "cifar10": DatasetMeta("cifar10", 10, 32, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "flowers102": DatasetMeta("flowers102", 102, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "eurosat": DatasetMeta("eurosat", 10, 64, (0.3444, 0.3808, 0.4083), (0.2037, 0.1366, 0.1148)),
}


def get_meta(dataset_name: str, image_size: int | None = None) -> DatasetMeta:
    meta = DATASET_META[dataset_name.lower()]
    if image_size is None:
        return meta
    return DatasetMeta(meta.name, meta.num_classes, image_size, meta.mean, meta.std)


def _targets_from_dataset(dataset: Dataset) -> np.ndarray:
    for attr in ("targets", "labels", "_labels"):
        if hasattr(dataset, attr):
            return np.asarray(getattr(dataset, attr))
    labels = []
    original_transform = getattr(dataset, "transform", None)
    if hasattr(dataset, "transform"):
        dataset.transform = None
    try:
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(int(label))
    finally:
        if hasattr(dataset, "transform"):
            dataset.transform = original_transform
    return np.asarray(labels)


def stratified_indices(
    targets: np.ndarray,
    per_class: int | None,
    fraction: float | None,
    seed: int,
    max_total: int | None = None,
) -> list[int]:
    rng = np.random.default_rng(seed)
    all_indices: list[int] = []
    classes = sorted(np.unique(targets).tolist())
    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        if per_class is not None:
            take = min(per_class, len(cls_idx))
        elif fraction is not None:
            take = max(1, int(round(len(cls_idx) * fraction)))
        else:
            take = len(cls_idx)
        all_indices.extend(cls_idx[:take].tolist())
    rng.shuffle(all_indices)
    if max_total is not None:
        all_indices = all_indices[:max_total]
    return all_indices


def split_indices_from_train(
    targets: np.ndarray,
    train_per_class: int | None,
    val_per_class: int | None,
    train_fraction: float | None,
    seed: int,
    max_train: int | None = None,
    max_val: int | None = None,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    classes = sorted(np.unique(targets).tolist())
    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        if train_per_class is not None:
            n_train = min(train_per_class, len(cls_idx))
        elif train_fraction is not None:
            n_train = max(1, int(round(len(cls_idx) * train_fraction)))
        else:
            n_train = max(1, int(round(len(cls_idx) * 0.8)))
        n_val = min(val_per_class or max(1, len(cls_idx) // 10), max(0, len(cls_idx) - n_train))
        train_indices.extend(cls_idx[:n_train].tolist())
        val_indices.extend(cls_idx[n_train:n_train + n_val].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if max_train is not None:
        train_indices = train_indices[:max_train]
    if max_val is not None:
        val_indices = val_indices[:max_val]
    return train_indices, val_indices


def split_indices_train_val_test(
    targets: np.ndarray,
    train_per_class: int | None,
    val_per_class: int | None,
    train_fraction: float | None,
    seed: int,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Create a deterministic stratified split for datasets without official splits."""
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    classes = sorted(np.unique(targets).tolist())
    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        if train_per_class is not None:
            n_train = min(train_per_class, len(cls_idx))
        elif train_fraction is not None:
            n_train = max(1, int(round(len(cls_idx) * train_fraction)))
        else:
            n_train = max(1, int(round(len(cls_idx) * 0.6)))
        remaining = max(0, len(cls_idx) - n_train)
        if val_per_class is not None:
            n_val = min(val_per_class, remaining)
        else:
            n_val = max(1, int(round(len(cls_idx) * 0.2))) if remaining > 1 else remaining
            n_val = min(n_val, remaining)
        train_indices.extend(cls_idx[:n_train].tolist())
        val_indices.extend(cls_idx[n_train:n_train + n_val].tolist())
        test_indices.extend(cls_idx[n_train + n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    if max_train is not None:
        train_indices = train_indices[:max_train]
    if max_val is not None:
        val_indices = val_indices[:max_val]
    if max_test is not None:
        test_indices = test_indices[:max_test]
    return train_indices, val_indices, test_indices


def build_base_dataset(
    dataset_name: str,
    root: str | Path,
    split: str,
    transform: Callable | None,
    download: bool = True,
    fake_size: int = 600,
    num_classes: int | None = None,
    image_size: int | None = None,
    download_url: str | None = None,
    archive_filename: str | None = None,
    archive_md5: str | None = None,
) -> Dataset:
    name = dataset_name.lower()
    root = Path(root)
    if name == "fake":
        meta = get_meta("fake", image_size)
        return datasets.FakeData(
            size=fake_size,
            image_size=(3, meta.image_size, meta.image_size),
            num_classes=num_classes or meta.num_classes,
            transform=transform,
            random_offset=0 if split != "test" else 100_000,
        )
    if name == "cifar10":
        if download_url:
            datasets.CIFAR10.url = download_url
        if archive_filename:
            datasets.CIFAR10.filename = archive_filename
        if archive_md5 is not None:
            datasets.CIFAR10.tgz_md5 = archive_md5 or None
        return datasets.CIFAR10(root=str(root), train=(split != "test"), transform=transform, download=download)
    if name == "flowers102":
        flower_split = "val" if split == "val" else split
        return datasets.Flowers102(root=str(root), split=flower_split, transform=transform, download=download)
    if name == "eurosat":
        return datasets.EuroSAT(root=str(root), transform=transform, download=download)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    meta: DatasetMeta
    train_size: int
    val_size: int
    test_size: int


@dataclass
class SSLDataBundle:
    labeled_loader: DataLoader
    unlabeled_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    meta: DatasetMeta
    labeled_size: int
    unlabeled_size: int
    val_size: int
    test_size: int


class TransformTwiceSubset(Dataset):
    """Return weakly and strongly augmented views of the same unlabeled image."""

    def __init__(self, base_dataset: Dataset, indices: list[int], weak_transform: Callable, strong_transform: Callable) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, _ = self.base_dataset[self.indices[idx]]
        return self.weak_transform(image), self.strong_transform(image)


def remaining_stratified_indices(
    targets: np.ndarray,
    excluded: set[int],
    seed: int,
    per_class: int | None = None,
    max_total: int | None = None,
) -> list[int]:
    rng = np.random.default_rng(seed)
    all_indices: list[int] = []
    classes = sorted(np.unique(targets).tolist())
    for cls in classes:
        cls_idx = [int(i) for i in np.where(targets == cls)[0].tolist() if int(i) not in excluded]
        rng.shuffle(cls_idx)
        take = len(cls_idx) if per_class is None else min(per_class, len(cls_idx))
        all_indices.extend(cls_idx[:take])
    rng.shuffle(all_indices)
    if max_total is not None:
        all_indices = all_indices[:max_total]
    return all_indices


def build_dataloaders(
    dataset_name: str,
    root: str | Path,
    train_transform: Callable,
    eval_transform: Callable,
    batch_size: int,
    num_workers: int,
    seed: int,
    download: bool = True,
    train_per_class: int | None = None,
    val_per_class: int | None = None,
    train_fraction: float | None = None,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
    image_size: int | None = None,
    download_url: str | None = None,
    archive_filename: str | None = None,
    archive_md5: str | None = None,
) -> DataBundle:
    name = dataset_name.lower()
    meta = get_meta(name, image_size)
    if name in {"cifar10", "fake"}:
        train_full_for_targets = build_base_dataset(
            name,
            root,
            "train",
            None,
            download,
            fake_size=max(600, (max_train or 0) + (max_val or 0) + 100),
            image_size=meta.image_size,
            download_url=download_url,
            archive_filename=archive_filename,
            archive_md5=archive_md5,
        )
        targets = _targets_from_dataset(train_full_for_targets)
        train_idx, val_idx = split_indices_from_train(
            targets,
            train_per_class=train_per_class,
            val_per_class=val_per_class,
            train_fraction=train_fraction,
            seed=seed,
            max_train=max_train,
            max_val=max_val,
        )
        train_base = build_base_dataset(
            name,
            root,
            "train",
            train_transform,
            download,
            fake_size=len(train_full_for_targets),
            image_size=meta.image_size,
            download_url=download_url,
            archive_filename=archive_filename,
            archive_md5=archive_md5,
        )
        val_base = build_base_dataset(
            name,
            root,
            "train",
            eval_transform,
            download,
            fake_size=len(train_full_for_targets),
            image_size=meta.image_size,
            download_url=download_url,
            archive_filename=archive_filename,
            archive_md5=archive_md5,
        )
        test_base = build_base_dataset(
            name,
            root,
            "test",
            eval_transform,
            download,
            fake_size=max_test or 200,
            image_size=meta.image_size,
            download_url=download_url,
            archive_filename=archive_filename,
            archive_md5=archive_md5,
        )
        test_indices = list(range(len(test_base)))
        if max_test is not None:
            test_indices = test_indices[:max_test]
        train_ds: Dataset = Subset(train_base, train_idx)
        val_ds: Dataset = Subset(val_base, val_idx)
        test_ds: Dataset = Subset(test_base, test_indices)
    elif name == "eurosat":
        full_for_targets = build_base_dataset(name, root, "train", None, download, image_size=meta.image_size)
        targets = _targets_from_dataset(full_for_targets)
        train_idx, val_idx, test_idx = split_indices_train_val_test(
            targets,
            train_per_class=train_per_class,
            val_per_class=val_per_class,
            train_fraction=train_fraction,
            seed=seed,
            max_train=max_train,
            max_val=max_val,
            max_test=max_test,
        )
        train_base = build_base_dataset(name, root, "train", train_transform, download, image_size=meta.image_size)
        val_base = build_base_dataset(name, root, "train", eval_transform, download, image_size=meta.image_size)
        test_base = build_base_dataset(name, root, "train", eval_transform, download, image_size=meta.image_size)
        train_ds = Subset(train_base, train_idx)
        val_ds = Subset(val_base, val_idx)
        test_ds = Subset(test_base, test_idx)
    elif name == "flowers102":
        train_base = build_base_dataset(name, root, "train", train_transform, download, image_size=meta.image_size)
        train_eval_base = build_base_dataset(name, root, "train", eval_transform, download, image_size=meta.image_size)
        val_base = build_base_dataset(name, root, "val", eval_transform, download, image_size=meta.image_size)
        test_base = build_base_dataset(name, root, "test", eval_transform, download, image_size=meta.image_size)
        targets = _targets_from_dataset(train_base)
        train_idx = stratified_indices(targets, train_per_class, train_fraction, seed, max_train)
        val_indices = list(range(len(val_base)))
        if max_val is not None:
            val_indices = val_indices[:max_val]
        test_indices = list(range(len(test_base)))
        if max_test is not None:
            test_indices = test_indices[:max_test]
        train_ds = Subset(train_base, train_idx)
        val_ds = Subset(val_base, val_indices)
        test_ds = Subset(test_base, test_indices)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return DataBundle(train_loader, val_loader, test_loader, meta, len(train_ds), len(val_ds), len(test_ds))


def build_ssl_dataloaders(
    dataset_name: str,
    root: str | Path,
    labeled_transform: Callable,
    weak_transform: Callable,
    strong_transform: Callable,
    eval_transform: Callable,
    labeled_batch_size: int,
    unlabeled_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    download: bool = True,
    train_per_class: int | None = None,
    val_per_class: int | None = None,
    train_fraction: float | None = None,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
    unlabeled_per_class: int | None = None,
    max_unlabeled: int | None = None,
    image_size: int | None = None,
    download_url: str | None = None,
    archive_filename: str | None = None,
    archive_md5: str | None = None,
) -> SSLDataBundle:
    """Build labeled/unlabeled/validation/test loaders for FixMatch-style SSL.

    Labeled and validation splits match the supervised pipeline. The unlabeled
    split is drawn from the remaining training images, with labels ignored by
    the returned dataset.
    """

    name = dataset_name.lower()
    meta = get_meta(name, image_size)
    if name in {"cifar10", "fake"}:
        train_full_for_targets = build_base_dataset(
            name,
            root,
            "train",
            None,
            download,
            fake_size=max(600, (max_train or 0) + (max_val or 0) + (max_unlabeled or 0) + 100),
            image_size=meta.image_size,
            download_url=download_url,
            archive_filename=archive_filename,
            archive_md5=archive_md5,
        )
        targets = _targets_from_dataset(train_full_for_targets)
        train_idx, val_idx = split_indices_from_train(
            targets,
            train_per_class=train_per_class,
            val_per_class=val_per_class,
            train_fraction=train_fraction,
            seed=seed,
            max_train=max_train,
            max_val=max_val,
        )
        excluded = set(train_idx) | set(val_idx)
        unlabeled_idx = remaining_stratified_indices(
            targets,
            excluded,
            seed=seed + 17,
            per_class=unlabeled_per_class,
            max_total=max_unlabeled,
        )
        labeled_base = build_base_dataset(
            name,
            root,
            "train",
            labeled_transform,
            download,
            fake_size=len(train_full_for_targets),
            image_size=meta.image_size,
            download_url=download_url,
            archive_filename=archive_filename,
            archive_md5=archive_md5,
        )
        unlabeled_base = build_base_dataset(
            name,
            root,
            "train",
            None,
            download,
            fake_size=len(train_full_for_targets),
            image_size=meta.image_size,
            download_url=download_url,
            archive_filename=archive_filename,
            archive_md5=archive_md5,
        )
        val_base = build_base_dataset(
            name,
            root,
            "train",
            eval_transform,
            download,
            fake_size=len(train_full_for_targets),
            image_size=meta.image_size,
            download_url=download_url,
            archive_filename=archive_filename,
            archive_md5=archive_md5,
        )
        test_base = build_base_dataset(
            name,
            root,
            "test",
            eval_transform,
            download,
            fake_size=max_test or 200,
            image_size=meta.image_size,
            download_url=download_url,
            archive_filename=archive_filename,
            archive_md5=archive_md5,
        )
        test_indices = list(range(len(test_base)))
        if max_test is not None:
            test_indices = test_indices[:max_test]
        labeled_ds: Dataset = Subset(labeled_base, train_idx)
        unlabeled_ds: Dataset = TransformTwiceSubset(unlabeled_base, unlabeled_idx, weak_transform, strong_transform)
        val_ds: Dataset = Subset(val_base, val_idx)
        test_ds: Dataset = Subset(test_base, test_indices)
    elif name == "eurosat":
        full_for_targets = build_base_dataset(name, root, "train", None, download, image_size=meta.image_size)
        targets = _targets_from_dataset(full_for_targets)
        train_idx, val_idx = split_indices_from_train(
            targets,
            train_per_class=train_per_class,
            val_per_class=val_per_class,
            train_fraction=train_fraction,
            seed=seed,
            max_train=max_train,
            max_val=max_val,
        )
        excluded = set(train_idx) | set(val_idx)
        test_idx = remaining_stratified_indices(targets, excluded, seed=seed + 101, max_total=max_test)
        excluded |= set(test_idx)
        unlabeled_idx = remaining_stratified_indices(
            targets,
            excluded,
            seed=seed + 17,
            per_class=unlabeled_per_class,
            max_total=max_unlabeled,
        )
        labeled_base = build_base_dataset(name, root, "train", labeled_transform, download, image_size=meta.image_size)
        unlabeled_base = build_base_dataset(name, root, "train", None, download, image_size=meta.image_size)
        val_base = build_base_dataset(name, root, "train", eval_transform, download, image_size=meta.image_size)
        test_base = build_base_dataset(name, root, "train", eval_transform, download, image_size=meta.image_size)
        labeled_ds = Subset(labeled_base, train_idx)
        unlabeled_ds = TransformTwiceSubset(unlabeled_base, unlabeled_idx, weak_transform, strong_transform)
        val_ds = Subset(val_base, val_idx)
        test_ds = Subset(test_base, test_idx)
    elif name == "flowers102":
        train_full_for_targets = build_base_dataset(name, root, "train", None, download, image_size=meta.image_size)
        targets = _targets_from_dataset(train_full_for_targets)
        train_idx = stratified_indices(targets, train_per_class, train_fraction, seed, max_train)
        excluded = set(train_idx)
        unlabeled_idx = remaining_stratified_indices(
            targets,
            excluded,
            seed=seed + 17,
            per_class=unlabeled_per_class,
            max_total=max_unlabeled,
        )
        val_base = build_base_dataset(name, root, "val", eval_transform, download, image_size=meta.image_size)
        test_base = build_base_dataset(name, root, "test", eval_transform, download, image_size=meta.image_size)
        val_indices = list(range(len(val_base)))
        if max_val is not None:
            val_indices = val_indices[:max_val]
        test_indices = list(range(len(test_base)))
        if max_test is not None:
            test_indices = test_indices[:max_test]
        labeled_base = build_base_dataset(name, root, "train", labeled_transform, download, image_size=meta.image_size)
        unlabeled_base = build_base_dataset(name, root, "train", None, download, image_size=meta.image_size)
        labeled_ds = Subset(labeled_base, train_idx)
        unlabeled_ds = TransformTwiceSubset(unlabeled_base, unlabeled_idx, weak_transform, strong_transform)
        val_ds = Subset(val_base, val_indices)
        test_ds = Subset(test_base, test_indices)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if len(unlabeled_ds) == 0:
        raise ValueError("FixMatch requires at least one unlabeled sample; increase max_unlabeled or reduce labeled/validation split sizes.")

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=labeled_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=unlabeled_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed + 31),
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return SSLDataBundle(
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        meta=meta,
        labeled_size=len(labeled_ds),
        unlabeled_size=len(unlabeled_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
    )
