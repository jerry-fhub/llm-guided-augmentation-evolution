from __future__ import annotations

import random
from typing import Callable

import torch
from torchvision import transforms

from .policy import AugOperation, AugPolicy


class RandomPolicyApply:
    """Apply one sub-policy sampled uniformly from a policy."""

    def __init__(self, sub_transforms: list[list[Callable]]) -> None:
        self.sub_transforms = sub_transforms

    def __call__(self, img):
        if not self.sub_transforms:
            return img
        for transform in random.choice(self.sub_transforms):
            img = transform(img)
        return img


def _maybe(op: AugOperation, transform: Callable) -> Callable:
    return transforms.RandomApply([transform], p=op.probability)


def _build_op(op: AugOperation, image_size: int) -> Callable:
    m = max(0.0, min(1.0, op.magnitude))
    if op.name == "RandomCrop":
        padding = int(2 + round(6 * m))
        return _maybe(op, transforms.RandomCrop(image_size, padding=padding, padding_mode="reflect"))
    if op.name == "RandomResizedCrop":
        min_scale = max(0.35, 1.0 - 0.6 * m)
        return _maybe(op, transforms.RandomResizedCrop(image_size, scale=(min_scale, 1.0)))
    if op.name == "HorizontalFlip":
        return transforms.RandomHorizontalFlip(p=op.probability)
    if op.name == "VerticalFlip":
        return transforms.RandomVerticalFlip(p=op.probability)
    if op.name == "Rotation":
        return _maybe(op, transforms.RandomRotation(degrees=round(5 + 40 * m)))
    if op.name == "Affine":
        degrees = round(2 + 18 * m)
        translate = min(0.25, 0.02 + 0.18 * m)
        scale = (max(0.6, 1 - 0.3 * m), 1 + 0.3 * m)
        shear = round(2 + 12 * m)
        return _maybe(op, transforms.RandomAffine(degrees=degrees, translate=(translate, translate), scale=scale, shear=shear))
    if op.name == "ColorJitter":
        strength = 0.05 + 0.75 * m
        hue = min(0.25, 0.02 + 0.12 * m)
        return _maybe(op, transforms.ColorJitter(brightness=strength, contrast=strength, saturation=strength, hue=hue))
    if op.name == "Grayscale":
        return transforms.RandomGrayscale(p=op.probability)
    if op.name == "GaussianBlur":
        kernel_size = 3 if image_size <= 64 else 5
        sigma = (0.1, 0.1 + 2.0 * m)
        return _maybe(op, transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma))
    if op.name == "Solarize":
        threshold = int(255 * (1.0 - 0.85 * m))
        return _maybe(op, transforms.RandomSolarize(threshold=threshold, p=1.0))
    if op.name == "Posterize":
        bits = max(2, int(round(8 - 5 * m)))
        return _maybe(op, transforms.RandomPosterize(bits=bits, p=1.0))
    if op.name == "RandomErasing":
        # RandomErasing is tensor-only; handled after ToTensor.
        scale_max = min(0.5, 0.05 + 0.35 * m)
        return transforms.RandomErasing(p=op.probability, scale=(0.02, scale_max), value="random")
    raise ValueError(f"Unknown operation: {op.name}")


def build_train_transform(
    policy: AugPolicy | None,
    image_size: int,
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
) -> Callable:
    pil_ops: list[list[Callable]] = []
    tensor_ops: list[Callable] = []
    if policy is not None:
        for sub in policy.sub_policies:
            built: list[Callable] = []
            for op in sub:
                built_op = _build_op(op, image_size)
                if op.name == "RandomErasing":
                    tensor_ops.append(built_op)
                else:
                    built.append(built_op)
            pil_ops.append(built)
    pipeline: list[Callable] = []
    if pil_ops:
        pipeline.append(RandomPolicyApply(pil_ops))
    pipeline.extend([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    pipeline.extend(tensor_ops)
    pipeline.append(transforms.Normalize(normalize_mean, normalize_std))
    return transforms.Compose(pipeline)


def build_eval_transform(
    image_size: int,
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])


def build_fixmatch_weak_transform(
    dataset_name: str,
    image_size: int,
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
) -> Callable:
    """Build the weak branch used for pseudo-label generation in FixMatch."""
    dataset = dataset_name.lower()
    ops: list[Callable] = [transforms.Resize((image_size, image_size))]
    if dataset == "flowers102":
        ops.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        ops.extend([
            transforms.RandomCrop(image_size, padding=max(2, image_size // 8), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ])
        if dataset == "eurosat":
            ops.append(transforms.RandomVerticalFlip())
    ops.extend([transforms.ToTensor(), transforms.Normalize(normalize_mean, normalize_std)])
    return transforms.Compose(ops)


def mix_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup or CutMix. Returns x, y_a, y_b, lambda."""
    alpha = max(float(mixup_alpha), float(cutmix_alpha))
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    index = torch.randperm(x.size(0), device=x.device)
    if cutmix_alpha > 0 and random.random() < 0.5:
        _, _, h, w = x.shape
        cut_rat = (1.0 - lam) ** 0.5
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y2 = min(cy + cut_h // 2, h)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    else:
        x = lam * x + (1 - lam) * x[index]
    return x, y, y[index], lam
