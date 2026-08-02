from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_aug_evolution.augmentation.baselines import build_named_baseline
from image_aug_evolution.data.datasets import build_dataloaders, get_meta
from image_aug_evolution.utils.config import load_config


def validate(config_path: str | Path) -> None:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    meta = get_meta(dataset_cfg["name"], dataset_cfg.get("image_size"))
    eval_transform = build_named_baseline("none", meta.image_size, meta.mean, meta.std)
    bundle = build_dataloaders(
        dataset_name=dataset_cfg["name"],
        root=dataset_cfg.get("root", "data/raw/image_datasets"),
        train_transform=eval_transform,
        eval_transform=eval_transform,
        batch_size=int(training_cfg.get("batch_size", 64)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        seed=int(config.get("seed", 0)),
        download=bool(dataset_cfg.get("download", True)),
        train_per_class=dataset_cfg.get("train_per_class"),
        val_per_class=dataset_cfg.get("val_per_class"),
        train_fraction=dataset_cfg.get("train_fraction"),
        max_train=dataset_cfg.get("max_train"),
        max_val=dataset_cfg.get("max_val"),
        max_test=dataset_cfg.get("max_test"),
        image_size=dataset_cfg.get("image_size"),
        download_url=dataset_cfg.get("download_url"),
        archive_filename=dataset_cfg.get("archive_filename"),
        archive_md5=dataset_cfg.get("archive_md5"),
    )
    x, y = next(iter(bundle.train_loader))
    print(f"dataset={bundle.meta.name}")
    print(f"num_classes={bundle.meta.num_classes}")
    print(f"image_size={bundle.meta.image_size}")
    print(f"train_size={bundle.train_size}")
    print(f"val_size={bundle.val_size}")
    print(f"test_size={bundle.test_size}")
    print(f"first_batch_shape={tuple(x.shape)}")
    print(f"first_batch_labels={y[: min(10, len(y))].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset download and split settings.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    args = parser.parse_args()
    validate(args.config)


if __name__ == "__main__":
    main()
