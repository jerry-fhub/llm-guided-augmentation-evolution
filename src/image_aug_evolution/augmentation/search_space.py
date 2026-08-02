from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationSpec:
    name: str
    min_magnitude: float = 0.0
    max_magnitude: float = 1.0
    description: str = ""


DEFAULT_OPERATION_SPECS: dict[str, OperationSpec] = {
    "RandomCrop": OperationSpec("RandomCrop", 0.0, 1.0, "Padding/crop strength."),
    "RandomResizedCrop": OperationSpec("RandomResizedCrop", 0.0, 1.0, "Scale jitter strength."),
    "HorizontalFlip": OperationSpec("HorizontalFlip", 0.0, 1.0, "Horizontal flip."),
    "VerticalFlip": OperationSpec("VerticalFlip", 0.0, 1.0, "Vertical flip."),
    "Rotation": OperationSpec("Rotation", 0.0, 1.0, "Maximum rotation angle."),
    "Affine": OperationSpec("Affine", 0.0, 1.0, "Affine translation/scale/shear strength."),
    "ColorJitter": OperationSpec("ColorJitter", 0.0, 1.0, "Brightness/contrast/saturation/hue strength."),
    "Grayscale": OperationSpec("Grayscale", 0.0, 1.0, "Random grayscale."),
    "GaussianBlur": OperationSpec("GaussianBlur", 0.0, 1.0, "Blur kernel/sigma strength."),
    "Solarize": OperationSpec("Solarize", 0.0, 1.0, "Solarization threshold."),
    "Posterize": OperationSpec("Posterize", 0.0, 1.0, "Posterization strength."),
    "RandomErasing": OperationSpec("RandomErasing", 0.0, 1.0, "Erasing area strength."),
}


DATASET_CONSTRAINTS: dict[str, dict[str, list[str]]] = {
    "cifar10": {"discouraged": ["VerticalFlip"]},
    "flowers102": {"discouraged": ["Grayscale", "Solarize", "Posterize"]},
    "eurosat": {"discouraged": []},
    "fake": {"discouraged": []},
}
