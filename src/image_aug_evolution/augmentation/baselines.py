from __future__ import annotations

from torchvision import transforms


def build_named_baseline(
    name: str,
    image_size: int,
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
):
    """Build common augmentation baselines supported by torchvision."""
    name = name.lower()
    final = [transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(normalize_mean, normalize_std)]
    if name in {"none", "no_aug"}:
        return transforms.Compose(final)
    if name == "standard":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=max(2, image_size // 8), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    if name in {"standard_color", "standard_jitter"}:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=max(2, image_size // 8), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.16, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    if name in {"standard_erasing", "standard_cutout"}:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=max(2, image_size // 8), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), value="random"),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    if name in {"standard_mixup", "standard_cutmix"}:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=max(2, image_size // 8), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    if name == "autoaugment":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    if name == "randaugment":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    if name in {"trivialaugment", "trivialaugmentwide"}:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    if name == "augmix":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    raise ValueError(f"Unknown baseline transform: {name}")
