from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_aug_evolution.augmentation.baselines import build_named_baseline  # noqa: E402
from image_aug_evolution.augmentation.builder import (  # noqa: E402
    build_eval_transform,
    build_fixmatch_weak_transform,
    build_train_transform,
)
from image_aug_evolution.augmentation.policy import AugPolicy  # noqa: E402
from image_aug_evolution.data.datasets import build_base_dataset, get_meta  # noqa: E402
from image_aug_evolution.evaluation.fixmatch_runner import _fixmatch_strong_named_transform  # noqa: E402
from image_aug_evolution.utils.io import read_json  # noqa: E402


DEFAULT_METHODS = [
    "original",
    "standard",
    "standard_color",
    "standard_erasing",
    "mixup",
    "cutmix",
    "randaugment",
    "trivialaugment",
    "supervised_llm",
    "fixmatch_weak",
    "fixmatch_randaugment",
    "fixmatch_trivialaugment",
    "fixmatch_llm_child",
]

METHOD_LABELS = {
    "original": "Original",
    "standard": "Standard",
    "standard_color": "Crop+Flip+Color",
    "standard_erasing": "RandomErasing",
    "mixup": "Mixup",
    "cutmix": "CutMix",
    "randaugment": "RandAugment",
    "trivialaugment": "TrivialAugment",
    "supervised_llm": "LLM supervised",
    "fixmatch_weak": "FixMatch weak",
    "fixmatch_randaugment": "FixMatch RA",
    "fixmatch_trivialaugment": "FixMatch TA",
    "fixmatch_llm_child": "LLM FixMatch child",
}

METHOD_NOTES = {
    "original": "Resize and display without training augmentation.",
    "standard": "Random crop and horizontal flip baseline.",
    "standard_color": "Standard baseline with light colour jitter.",
    "standard_erasing": "Standard baseline with random erasing after tensor conversion.",
    "mixup": "Visual demonstration of batch-level Mixup using a paired image.",
    "cutmix": "Visual demonstration of batch-level CutMix using a paired image.",
    "randaugment": "Torchvision RandAugment baseline.",
    "trivialaugment": "Torchvision TrivialAugment baseline.",
    "supervised_llm": "Best CIFAR-10 supervised OpenAI-evolved JSON policy.",
    "fixmatch_weak": "Weak augmentation branch used for FixMatch pseudo-label views.",
    "fixmatch_randaugment": "FixMatch strong branch using RandAugment.",
    "fixmatch_trivialaugment": "FixMatch strong branch using TrivialAugment.",
    "fixmatch_llm_child": "Best repaired LLM-generated FixMatch strong-branch child.",
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_name(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "sample"


def _parse_methods(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_METHODS
    methods = [_safe_name(item) for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported methods: {DEFAULT_METHODS}")
    return methods


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _unnormalise(
    x: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (x * mean_t.new_tensor(std).view(3, 1, 1) + mean_t).clamp(0.0, 1.0)


def _tensor_to_display_image(x: torch.Tensor, display_size: int) -> Image.Image:
    arr = (x.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    resample = Image.Resampling.NEAREST if max(img.size) <= 64 else Image.Resampling.BICUBIC
    return img.resize((display_size, display_size), resample)


def _resize_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])(image)


def _apply_mixup(image: Image.Image, pair: Image.Image, image_size: int, display_size: int, lam: float = 0.65) -> Image.Image:
    x1 = _resize_to_tensor(image, image_size)
    x2 = _resize_to_tensor(pair, image_size)
    mixed = (lam * x1 + (1.0 - lam) * x2).clamp(0.0, 1.0)
    return _tensor_to_display_image(mixed, display_size)


def _apply_cutmix(image: Image.Image, pair: Image.Image, image_size: int, display_size: int) -> Image.Image:
    x1 = _resize_to_tensor(image, image_size)
    x2 = _resize_to_tensor(pair, image_size)
    mixed = x1.clone()
    _, h, w = mixed.shape
    x_start, x_end = int(w * 0.45), int(w * 0.82)
    y_start, y_end = int(h * 0.18), int(h * 0.64)
    mixed[:, y_start:y_end, x_start:x_end] = x2[:, y_start:y_end, x_start:x_end]
    return _tensor_to_display_image(mixed.clamp(0.0, 1.0), display_size)


def _policy_path(relative_path: str) -> Path:
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Required policy file is missing: {path}. "
            "Run the relevant experiment first or choose methods that do not require learned policies."
        )
    return path


def _build_transform(method: str, dataset_name: str, image_size: int, mean, std):
    if method == "original":
        return build_eval_transform(image_size, mean, std)
    if method in {"standard", "standard_color", "standard_erasing", "randaugment", "trivialaugment"}:
        return build_named_baseline(method, image_size, mean, std)
    if method == "supervised_llm":
        policy = AugPolicy.from_dict(read_json(_policy_path("results/cifar10_resnet18cifar_strong_openai/policies/gen02_mut_005.json")))
        return build_train_transform(policy, image_size, mean, std)
    if method == "fixmatch_weak":
        return build_fixmatch_weak_transform(dataset_name, image_size, mean, std)
    if method == "fixmatch_randaugment":
        return _fixmatch_strong_named_transform("randaugment", dataset_name, image_size, mean, std)
    if method == "fixmatch_trivialaugment":
        return _fixmatch_strong_named_transform("trivialaugment", dataset_name, image_size, mean, std)
    if method == "fixmatch_llm_child":
        policy = AugPolicy.from_dict(
            read_json(_policy_path("results/cifar10_fixmatch_repaired_llm_children/policies/fm_gen01_mut_001_repaired.json"))
        )
        return build_train_transform(policy, image_size, mean, std)
    raise ValueError(f"Method does not use a transform: {method}")


def _apply_method(
    method: str,
    image: Image.Image,
    pair: Image.Image,
    dataset_name: str,
    image_size: int,
    mean,
    std,
    display_size: int,
    seed: int,
) -> Image.Image:
    _seed_everything(seed)
    if method == "mixup":
        return _apply_mixup(image, pair, image_size, display_size)
    if method == "cutmix":
        return _apply_cutmix(image, pair, image_size, display_size)
    transform = _build_transform(method, dataset_name, image_size, mean, std)
    x = transform(image)
    return _tensor_to_display_image(_unnormalise(x, mean, std), display_size)


def _dataset_download_kwargs(dataset_name: str) -> dict:
    if dataset_name.lower() != "cifar10":
        return {}
    return {
        "download_url": "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
        "archive_filename": "cifar-10-python.tar.gz",
        "archive_md5": "c58f30108f718f92721af3b95e74349a",
    }


def _load_samples_from_dataset(args, image_size: int) -> list[dict]:
    dataset = build_base_dataset(
        args.dataset,
        args.data_root,
        split=args.split,
        transform=None,
        download=not args.no_download,
        image_size=image_size,
        **_dataset_download_kwargs(args.dataset),
    )
    rng = random.Random(args.seed)
    count = min(args.count, len(dataset))
    indices = rng.sample(range(len(dataset)), count)
    pair_indices = rng.sample(range(len(dataset)), count)
    classes = getattr(dataset, "classes", None)
    samples = []
    for row, (idx, pair_idx) in enumerate(zip(indices, pair_indices, strict=False)):
        if pair_idx == idx:
            pair_idx = (idx + 1) % len(dataset)
        image, label = dataset[idx]
        pair, pair_label = dataset[pair_idx]
        label_name = classes[int(label)] if classes and int(label) < len(classes) else f"class_{int(label)}"
        pair_label_name = classes[int(pair_label)] if classes and int(pair_label) < len(classes) else f"class_{int(pair_label)}"
        samples.append({
            "sample_id": f"sample_{row:02d}_{_safe_name(label_name)}",
            "source": f"{args.dataset}:{args.split}:{idx}",
            "label": label_name,
            "pair_source": f"{args.dataset}:{args.split}:{pair_idx}",
            "pair_label": pair_label_name,
            "image": image.convert("RGB"),
            "pair": pair.convert("RGB"),
        })
    return samples


def _load_samples_from_image(args) -> list[dict]:
    image = _load_rgb(Path(args.image))
    if args.pair_image:
        pair = _load_rgb(Path(args.pair_image))
        pair_source = str(Path(args.pair_image))
    else:
        pair = ImageOps.mirror(ImageOps.autocontrast(image))
        pair_source = "auto-generated mirror/autocontrast pair"
    return [{
        "sample_id": f"sample_00_{_safe_name(Path(args.image).stem)}",
        "source": str(Path(args.image)),
        "label": "custom_image",
        "pair_source": pair_source,
        "pair_label": "pair_image",
        "image": image,
        "pair": pair,
    }]


def _draw_grid_cell(
    canvas: Image.Image,
    image: Image.Image,
    x: int,
    y: int,
    cell_w: int,
    cell_h: int,
    title: str,
    subtitle: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((x, y, x + cell_w, y + cell_h), fill=(248, 249, 250), outline=(210, 214, 219))
    draw.text((x + 8, y + 8), title, fill=(20, 28, 38), font=font)
    draw.text((x + 8, y + 22), subtitle[:36], fill=(91, 101, 115), font=font)
    img_x = x + (cell_w - image.width) // 2
    img_y = y + 42
    canvas.paste(image, (img_x, img_y))


def _save_grid(rows: list[dict], methods: list[str], output_dir: Path, display_size: int) -> Path:
    cell_w = max(display_size + 28, 180)
    cell_h = display_size + 64
    label_w = 148
    margin = 18
    header_h = 38
    width = margin * 2 + label_w + cell_w * len(methods)
    height = margin * 2 + header_h + cell_h * len(rows)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin + 8), "Input sample", fill=(20, 28, 38), font=font)
    for col, method in enumerate(methods):
        x = margin + label_w + col * cell_w
        draw.text((x + 8, margin + 8), METHOD_LABELS[method], fill=(20, 28, 38), font=font)
    for row_idx, row in enumerate(rows):
        y = margin + header_h + row_idx * cell_h
        draw.rectangle((margin, y, margin + label_w, y + cell_h), fill=(243, 245, 247), outline=(210, 214, 219))
        draw.text((margin + 8, y + 12), row["label"][:24], fill=(20, 28, 38), font=font)
        draw.text((margin + 8, y + 28), row["source"][:24], fill=(91, 101, 115), font=font)
        for col, method in enumerate(methods):
            x = margin + label_w + col * cell_w
            _draw_grid_cell(canvas, row["outputs"][method], x, y, cell_w, cell_h, METHOD_LABELS[method], method)
    output_path = output_dir / "augmentation_grid.png"
    canvas.save(output_path)
    return output_path


def export_examples(args) -> None:
    methods = _parse_methods(args.methods)
    meta = get_meta(args.dataset, args.image_size)
    samples = _load_samples_from_image(args) if args.image else _load_samples_from_dataset(args, meta.image_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(exist_ok=True)

    rows = []
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "image_size": meta.image_size,
        "display_size": args.display_size,
        "seed": args.seed,
        "methods": [{"name": method, "label": METHOD_LABELS[method], "note": METHOD_NOTES[method]} for method in methods],
        "samples": [],
    }
    for row_idx, sample in enumerate(samples):
        row = {"label": sample["label"], "source": sample["source"], "outputs": {}}
        sample_manifest = {
            "sample_id": sample["sample_id"],
            "source": sample["source"],
            "label": sample["label"],
            "pair_source": sample["pair_source"],
            "pair_label": sample["pair_label"],
            "outputs": {},
        }
        for method_idx, method in enumerate(methods):
            image = _apply_method(
                method,
                sample["image"],
                sample["pair"],
                meta.name,
                meta.image_size,
                meta.mean,
                meta.std,
                args.display_size,
                args.seed + row_idx * 1000 + method_idx * 97,
            )
            filename = f"{sample['sample_id']}_{method}.png"
            path = individual_dir / filename
            image.save(path)
            row["outputs"][method] = image
            sample_manifest["outputs"][method] = str(path.relative_to(output_dir))
        rows.append(row)
        manifest["samples"].append(sample_manifest)

    grid_path = _save_grid(rows, methods, output_dir, args.display_size)
    manifest["grid"] = str(grid_path.relative_to(output_dir))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved augmentation grid: {grid_path}")
    print(f"Saved individual images: {individual_dir}")
    print(f"Saved manifest: {output_dir / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export visual examples for the augmentation methods used in the project.")
    parser.add_argument("--image", help="Optional path to a custom input image. If omitted, random dataset samples are used.")
    parser.add_argument("--pair-image", help="Optional second image for Mixup/CutMix visualisation when --image is used.")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "flowers102", "eurosat", "fake"])
    parser.add_argument("--data-root", default="data/raw/image_datasets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=4, help="Number of dataset samples to export when --image is omitted.")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--image-size", type=int, help="Override dataset image size.")
    parser.add_argument("--display-size", type=int, default=192, help="Square display size for saved example PNGs.")
    parser.add_argument("--methods", help="Comma-separated method list. Defaults to all supported methods.")
    parser.add_argument("--output-dir", default="docs/assets/augmentation_examples/cifar10_default")
    parser.add_argument("--no-download", action="store_true", help="Do not download datasets if missing.")
    args = parser.parse_args()
    export_examples(args)


if __name__ == "__main__":
    main()
