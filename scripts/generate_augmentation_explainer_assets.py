from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_aug_evolution.augmentation.builder import (  # noqa: E402
    build_fixmatch_weak_transform,
    build_train_transform,
)
from image_aug_evolution.augmentation.policy import AugPolicy  # noqa: E402
from image_aug_evolution.augmentation.seed_policies import build_seed_policy  # noqa: E402
from image_aug_evolution.data.datasets import build_base_dataset, get_meta  # noqa: E402
from image_aug_evolution.evaluation.fixmatch_runner import _fixmatch_strong_named_transform  # noqa: E402
from image_aug_evolution.utils.io import read_json  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "combined" / "completed_experiment_visuals"
DATA_ROOT = ROOT / "data" / "raw" / "image_datasets"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _unnormalise(x: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (x * std_t + mean_t).clamp(0.0, 1.0)


def _tensor_to_pil(x: torch.Tensor, scale: int = 5) -> Image.Image:
    arr = (x.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


def _draw_cell(
    canvas: Image.Image,
    image: Image.Image,
    x: int,
    y: int,
    title: str,
    subtitle: str | None,
    cell_w: int,
    cell_h: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    subtitle_font = ImageFont.load_default()
    draw.rectangle((x, y, x + cell_w, y + cell_h), fill=(248, 249, 250), outline=(210, 214, 219))
    text_y = y + 8
    draw.text((x + 8, text_y), title, fill=(20, 28, 38), font=title_font)
    if subtitle:
        draw.text((x + 8, text_y + 14), subtitle, fill=(91, 101, 115), font=subtitle_font)
    img_x = x + (cell_w - image.width) // 2
    img_y = y + 42
    canvas.paste(image, (img_x, img_y))


def _make_grid(samples: list[dict], methods: list[dict], output_path: Path) -> None:
    cell_w = 190
    cell_h = 224
    header_h = 42
    label_w = 132
    margin = 18
    width = margin * 2 + label_w + cell_w * len(methods)
    height = margin * 2 + header_h + cell_h * len(samples)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((margin, margin + 8), "CIFAR-10 sample", fill=(20, 28, 38), font=font)
    for c, method in enumerate(methods):
        x = margin + label_w + c * cell_w
        draw.text((x + 8, margin + 8), method["title"], fill=(20, 28, 38), font=font)

    for r, sample in enumerate(samples):
        y = margin + header_h + r * cell_h
        draw.rectangle((margin, y, margin + label_w, y + cell_h), fill=(243, 245, 247), outline=(210, 214, 219))
        draw.text((margin + 8, y + 12), f"{sample['class_name']}", fill=(20, 28, 38), font=font)
        draw.text((margin + 8, y + 30), f"index {sample['idx']}", fill=(91, 101, 115), font=font)
        for c, method in enumerate(methods):
            image = method["images"][r]
            x = margin + label_w + c * cell_w
            _draw_cell(canvas, image, x, y, method["short"], method.get("subtitle"), cell_w, cell_h)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def _apply_transform(transform, dataset, idx: int, seed: int, mean, std) -> Image.Image:
    _set_seed(seed)
    original_transform = dataset.transform
    try:
        dataset.transform = transform
        x, _ = dataset[idx]
    finally:
        dataset.transform = original_transform
    return _tensor_to_pil(_unnormalise(x, mean, std))


def generate_augmentation_grid() -> Path:
    meta = get_meta("cifar10")
    raw_transform = transforms.Compose([
        transforms.Resize((meta.image_size, meta.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(meta.mean, meta.std),
    ])
    dataset = build_base_dataset(
        "cifar10",
        DATA_ROOT,
        split="train",
        transform=raw_transform,
        download=False,
        image_size=meta.image_size,
    )

    chosen_indices = [7, 12, 21, 42]
    samples = []
    for idx in chosen_indices:
        _, label = dataset[idx]
        samples.append({"idx": idx, "label": int(label), "class_name": CIFAR10_CLASSES[int(label)]})

    supervised_llm_policy = AugPolicy.from_dict(
        read_json(ROOT / "results" / "cifar10_resnet18cifar_strong_openai" / "policies" / "gen02_mut_005.json")
    )
    fixmatch_llm_child = AugPolicy.from_dict(
        read_json(ROOT / "results" / "cifar10_fixmatch_repaired_llm_children" / "policies" / "fm_gen01_mut_001_repaired.json")
    )
    standard_policy = build_seed_policy("cifar10", "standard")

    method_specs = [
        {
            "title": "Original",
            "short": "Original",
            "subtitle": "no train aug",
            "transform": raw_transform,
        },
        {
            "title": "Standard",
            "short": "Crop + Flip",
            "subtitle": "baseline",
            "transform": build_train_transform(standard_policy, meta.image_size, meta.mean, meta.std),
        },
        {
            "title": "LLM supervised",
            "short": "LLM evolved",
            "subtitle": "crop/flip/jitter",
            "transform": build_train_transform(supervised_llm_policy, meta.image_size, meta.mean, meta.std),
        },
        {
            "title": "RandAugment",
            "short": "RandAug",
            "subtitle": "FixMatch strong",
            "transform": _fixmatch_strong_named_transform("randaugment", "cifar10", meta.image_size, meta.mean, meta.std),
        },
        {
            "title": "TrivialAugment",
            "short": "TrivialAug",
            "subtitle": "FixMatch strong",
            "transform": _fixmatch_strong_named_transform("trivialaugment", "cifar10", meta.image_size, meta.mean, meta.std),
        },
        {
            "title": "FixMatch weak",
            "short": "Weak branch",
            "subtitle": "pseudo-label view",
            "transform": build_fixmatch_weak_transform("cifar10", meta.image_size, meta.mean, meta.std),
        },
        {
            "title": "LLM FixMatch child",
            "short": "LLM child",
            "subtitle": "repaired strong",
            "transform": build_train_transform(fixmatch_llm_child, meta.image_size, meta.mean, meta.std),
        },
    ]

    for method in method_specs:
        method["images"] = [
            _apply_transform(method["transform"], dataset, sample["idx"], seed=20260726 + i * 97, mean=meta.mean, std=meta.std)
            for i, sample in enumerate(samples)
        ]

    output_path = OUTPUT_DIR / "augmentation_method_examples_cifar10.png"
    _make_grid(samples, method_specs, output_path)
    return output_path


def generate_mixup_cutmix_demo() -> Path:
    meta = get_meta("cifar10")
    raw_transform = transforms.Compose([
        transforms.Resize((meta.image_size, meta.image_size)),
        transforms.ToTensor(),
    ])
    dataset = build_base_dataset(
        "cifar10",
        DATA_ROOT,
        split="train",
        transform=raw_transform,
        download=False,
        image_size=meta.image_size,
    )
    idx_a, idx_b = 7, 12
    img_a, label_a = dataset[idx_a]
    img_b, label_b = dataset[idx_b]
    lam = 0.65
    mixup = (lam * img_a + (1.0 - lam) * img_b).clamp(0.0, 1.0)
    cutmix = img_a.clone()
    _, h, w = cutmix.shape
    x1, x2 = int(w * 0.45), int(w * 0.82)
    y1, y2 = int(h * 0.18), int(h * 0.64)
    cutmix[:, y1:y2, x1:x2] = img_b[:, y1:y2, x1:x2]

    cell_w = 210
    cell_h = 232
    margin = 20
    width = margin * 2 + cell_w * 4
    height = margin * 2 + cell_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    cells = [
        (_tensor_to_pil(img_a), f"Image A: {CIFAR10_CLASSES[int(label_a)]}", "source sample"),
        (_tensor_to_pil(img_b), f"Image B: {CIFAR10_CLASSES[int(label_b)]}", "source sample"),
        (_tensor_to_pil(mixup), "Mixup", "weighted blend"),
        (_tensor_to_pil(cutmix), "CutMix", "patch replacement"),
    ]
    for i, (image, title, subtitle) in enumerate(cells):
        _draw_cell(canvas, image, margin + i * cell_w, margin, title, subtitle, cell_w, cell_h)

    output_path = OUTPUT_DIR / "mixup_cutmix_visual_demo_cifar10.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved: {output_path}")
    return output_path


def generate_policy_summary_chart() -> Path:
    labels = [
        "Supervised Mixup",
        "Supervised LLM best",
        "FixMatch standard",
        "FixMatch reused LLM",
        "FixMatch LLM child",
        "FixMatch TrivialAug",
        "FixMatch RandAug",
    ]
    accuracy = [48.50, 48.45, 56.75, 59.00, 59.05, 61.50, 62.05]
    colors = ["#9CA3AF", "#2F80ED", "#A7C957", "#56CC9D", "#27AE60", "#F2C94C", "#F2994A"]

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    y = np.arange(len(labels))
    ax.barh(y, accuracy, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(44, 64)
    ax.set_title("Local CIFAR-10 Results: Strong Augmentation Helps Most Inside FixMatch")
    for i, value in enumerate(accuracy):
        ax.text(value + 0.18, i, f"{value:.2f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "augmentation_explainer_result_bar.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    generate_augmentation_grid()
    generate_mixup_cutmix_demo()
    generate_policy_summary_chart()


if __name__ == "__main__":
    main()
