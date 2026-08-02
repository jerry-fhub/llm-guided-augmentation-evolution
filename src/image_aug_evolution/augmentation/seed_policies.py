from __future__ import annotations

from image_aug_evolution.augmentation.policy import AugOperation, AugPolicy
from image_aug_evolution.augmentation.search_space import DATASET_CONSTRAINTS


def _op(name: str, probability: float, magnitude: float) -> AugOperation:
    return AugOperation(name=name, probability=round(probability, 2), magnitude=round(magnitude, 2))


def _filter_discouraged(dataset_name: str, sub_policies: list[list[AugOperation]]) -> list[list[AugOperation]]:
    discouraged = set(DATASET_CONSTRAINTS.get(dataset_name.lower(), {}).get("discouraged", []))
    return [[op for op in sub if op.name not in discouraged] for sub in sub_policies]


def build_seed_policy(dataset_name: str, seed_name: str, policy_id: str | None = None) -> AugPolicy:
    """Create strong, human-designed seed policies for evolutionary search.

    These policies approximate common augmentation baselines within the local
    JSON search space, so they can be used as parents by LLM/evolution operators.
    """
    dataset = dataset_name.lower()
    name = seed_name.lower()
    policy_id = policy_id or f"seed_{name}"

    if name in {"none", "no_aug"}:
        sub_policies: list[list[AugOperation]] = []
    elif name == "standard":
        if dataset == "eurosat":
            sub_policies = [
                [
                _op("RandomCrop", 1.0, 0.35),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("VerticalFlip", 0.5, 0.0),
                ],
                [_op("Rotation", 0.45, 0.25)],
            ]
        elif dataset == "flowers102":
            sub_policies = [[
                _op("RandomResizedCrop", 1.0, 0.35),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("ColorJitter", 0.35, 0.20),
            ]]
        else:
            sub_policies = [[
                _op("RandomCrop", 1.0, 0.45),
                _op("HorizontalFlip", 0.5, 0.0),
            ]]
    elif name == "conservative":
        if dataset == "flowers102":
            sub_policies = [
                [_op("RandomResizedCrop", 1.0, 0.25), _op("HorizontalFlip", 0.5, 0.0)],
                [_op("ColorJitter", 0.25, 0.15), _op("GaussianBlur", 0.15, 0.10)],
            ]
        elif dataset == "eurosat":
            sub_policies = [
                [_op("RandomCrop", 1.0, 0.25), _op("HorizontalFlip", 0.5, 0.0), _op("VerticalFlip", 0.5, 0.0)],
                [_op("Rotation", 0.35, 0.20), _op("GaussianBlur", 0.15, 0.10)],
            ]
        else:
            sub_policies = [
                [_op("RandomCrop", 1.0, 0.30), _op("HorizontalFlip", 0.5, 0.0)],
                [_op("ColorJitter", 0.25, 0.15)],
            ]
    elif name in {"standard_color", "color"}:
        if dataset == "flowers102":
            sub_policies = [[
                _op("RandomResizedCrop", 1.0, 0.30),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("ColorJitter", 0.35, 0.18),
            ]]
        elif dataset == "eurosat":
            sub_policies = [
                [
                _op("RandomCrop", 1.0, 0.25),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("VerticalFlip", 0.5, 0.0),
                ],
                [_op("ColorJitter", 0.20, 0.10)],
            ]
        else:
            sub_policies = [[
                _op("RandomCrop", 1.0, 0.35),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("ColorJitter", 0.30, 0.16),
            ]]
    elif name in {"standard_erasing", "erasing", "cutout"}:
        if dataset == "flowers102":
            sub_policies = [[
                _op("RandomResizedCrop", 1.0, 0.30),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("RandomErasing", 0.12, 0.10),
            ]]
        elif dataset == "eurosat":
            sub_policies = [
                [
                _op("RandomCrop", 1.0, 0.25),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("VerticalFlip", 0.5, 0.0),
                ],
                [_op("RandomErasing", 0.08, 0.08)],
            ]
        else:
            sub_policies = [[
                _op("RandomCrop", 1.0, 0.35),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("RandomErasing", 0.15, 0.10),
            ]]
    elif name in {"standard_mixup", "mixup"}:
        if dataset == "flowers102":
            sub_policies = [[
                _op("RandomResizedCrop", 1.0, 0.28),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("ColorJitter", 0.25, 0.12),
            ]]
        elif dataset == "eurosat":
            sub_policies = [
                [
                _op("RandomCrop", 1.0, 0.25),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("VerticalFlip", 0.5, 0.0),
                ],
                [_op("Rotation", 0.25, 0.12)],
            ]
        else:
            sub_policies = [[
                _op("RandomCrop", 1.0, 0.32),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("ColorJitter", 0.18, 0.10),
            ]]
    elif name in {"standard_cutmix", "cutmix"}:
        if dataset == "flowers102":
            sub_policies = [[
                _op("RandomResizedCrop", 1.0, 0.28),
                _op("HorizontalFlip", 0.5, 0.0),
            ]]
        elif dataset == "eurosat":
            sub_policies = [[
                _op("RandomCrop", 1.0, 0.25),
                _op("HorizontalFlip", 0.5, 0.0),
                _op("VerticalFlip", 0.5, 0.0),
            ]]
        else:
            sub_policies = [[
                _op("RandomCrop", 1.0, 0.32),
                _op("HorizontalFlip", 0.5, 0.0),
            ]]
    elif name in {"randaugment", "randaugment_inspired"}:
        if dataset == "flowers102":
            sub_policies = [
                [_op("RandomResizedCrop", 1.0, 0.35), _op("HorizontalFlip", 0.5, 0.0)],
                [_op("ColorJitter", 0.65, 0.35), _op("Rotation", 0.45, 0.25)],
                [_op("Affine", 0.45, 0.25), _op("GaussianBlur", 0.25, 0.20)],
            ]
        elif dataset == "eurosat":
            sub_policies = [
                [_op("RandomCrop", 1.0, 0.35), _op("HorizontalFlip", 0.5, 0.0), _op("VerticalFlip", 0.5, 0.0)],
                [_op("Rotation", 0.60, 0.35), _op("Affine", 0.45, 0.25)],
                [_op("ColorJitter", 0.35, 0.25), _op("GaussianBlur", 0.20, 0.15)],
            ]
        else:
            sub_policies = [
                [_op("RandomCrop", 1.0, 0.45), _op("HorizontalFlip", 0.5, 0.0)],
                [_op("ColorJitter", 0.55, 0.35), _op("Rotation", 0.45, 0.30)],
                [_op("Affine", 0.35, 0.30), _op("Solarize", 0.20, 0.30)],
                [_op("Posterize", 0.15, 0.25), _op("GaussianBlur", 0.20, 0.20)],
            ]
    elif name in {"trivialaugment", "trivialaugmentwide", "trivialaugment_inspired"}:
        if dataset == "flowers102":
            sub_policies = [
                [_op("RandomResizedCrop", 1.0, 0.35)],
                [_op("HorizontalFlip", 0.5, 0.0)],
                [_op("ColorJitter", 0.65, 0.30)],
                [_op("Rotation", 0.45, 0.25)],
            ]
        elif dataset == "eurosat":
            sub_policies = [
                [_op("RandomCrop", 1.0, 0.35)],
                [_op("HorizontalFlip", 0.5, 0.0)],
                [_op("VerticalFlip", 0.5, 0.0)],
                [_op("Rotation", 0.50, 0.30)],
            ]
        else:
            sub_policies = [
                [_op("RandomCrop", 1.0, 0.40)],
                [_op("HorizontalFlip", 0.5, 0.0)],
                [_op("ColorJitter", 0.55, 0.30)],
                [_op("Rotation", 0.40, 0.25)],
            ]
    else:
        raise ValueError(f"Unknown seed policy: {seed_name}")

    sub_policies = [sub for sub in _filter_discouraged(dataset, sub_policies) if sub]
    mixing = {"mixup_alpha": 0.0, "cutmix_alpha": 0.0}
    if name in {"standard_mixup", "mixup"}:
        mixing["mixup_alpha"] = 0.20 if dataset != "flowers102" else 0.10
    if name in {"standard_cutmix", "cutmix"}:
        mixing["cutmix_alpha"] = 0.40 if dataset != "flowers102" else 0.20

    return AugPolicy(
        policy_id=policy_id,
        sub_policies=sub_policies,
        mixing=mixing,
        source=f"baseline_seed:{name}",
        parent_ids=[],
        notes=f"Baseline-seeded parent approximating {seed_name}.",
    )


def build_seed_policies(dataset_name: str, seed_names: list[str] | None = None) -> list[AugPolicy]:
    names = seed_names or ["standard", "conservative", "randaugment", "trivialaugment"]
    return [
        build_seed_policy(dataset_name, name, policy_id=f"seed_{i:02d}_{name.lower()}")
        for i, name in enumerate(names)
    ]
