from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_aug_evolution.augmentation.builder import build_train_transform
from image_aug_evolution.augmentation.policy import AugPolicy
from image_aug_evolution.augmentation.validator import PolicyValidator
from image_aug_evolution.data.datasets import build_base_dataset, get_meta
from image_aug_evolution.utils.config import load_config
from image_aug_evolution.utils.io import read_json


def _unnormalise(x: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (x * std_t + mean_t).clamp(0.0, 1.0)


def visualise(config_path: str | Path, policy_path: str | Path, output_path: str | Path, count: int, repeats: int) -> None:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    meta = get_meta(dataset_cfg["name"], dataset_cfg.get("image_size"))
    policy = AugPolicy.from_dict(read_json(policy_path))
    validation = PolicyValidator(dataset_cfg["name"]).validate(policy)
    if not validation.ok:
        raise ValueError(f"Policy is invalid: {validation.errors}")
    raw_transform = transforms.Compose([
        transforms.Resize((meta.image_size, meta.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(meta.mean, meta.std),
    ])
    aug_transform = build_train_transform(policy, meta.image_size, meta.mean, meta.std)
    raw_dataset = build_base_dataset(
        dataset_cfg["name"],
        dataset_cfg.get("root", "data/raw/image_datasets"),
        split="train",
        transform=raw_transform,
        download=bool(dataset_cfg.get("download", True)),
        image_size=meta.image_size,
        download_url=dataset_cfg.get("download_url"),
        archive_filename=dataset_cfg.get("archive_filename"),
        archive_md5=dataset_cfg.get("archive_md5"),
    )
    aug_dataset = build_base_dataset(
        dataset_cfg["name"],
        dataset_cfg.get("root", "data/raw/image_datasets"),
        split="train",
        transform=aug_transform,
        download=bool(dataset_cfg.get("download", True)),
        image_size=meta.image_size,
        download_url=dataset_cfg.get("download_url"),
        archive_filename=dataset_cfg.get("archive_filename"),
        archive_md5=dataset_cfg.get("archive_md5"),
    )
    rows: list[torch.Tensor] = []
    for idx in range(min(count, len(raw_dataset))):
        original, label = raw_dataset[idx]
        samples = [_unnormalise(original, meta.mean, meta.std)]
        for _ in range(repeats):
            augmented, _ = aug_dataset[idx]
            samples.append(_unnormalise(augmented, meta.mean, meta.std))
        rows.extend(samples)
    grid = make_grid(rows, nrow=repeats + 1, padding=3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(2.2 * (repeats + 1), 2.2 * count))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title(f"Policy: {policy.policy_id} | first column is original")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved policy visualisation to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise a learned augmentation policy on real images.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--policy", required=True, help="Policy JSON path.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--count", type=int, default=6, help="Number of source images.")
    parser.add_argument("--repeats", type=int, default=3, help="Augmented samples per source image.")
    args = parser.parse_args()
    visualise(args.config, args.policy, args.output, args.count, args.repeats)


if __name__ == "__main__":
    main()
