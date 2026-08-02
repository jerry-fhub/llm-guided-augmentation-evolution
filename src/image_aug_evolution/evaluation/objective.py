from __future__ import annotations

from image_aug_evolution.augmentation.policy import AugPolicy


DESTRUCTIVE_OP_WEIGHTS = {
    "ColorJitter": 1.15,
    "GaussianBlur": 1.20,
    "Grayscale": 1.35,
    "Posterize": 1.45,
    "Solarize": 1.45,
    "RandomErasing": 1.35,
    "RandomResizedCrop": 1.10,
}


def policy_complexity(policy: AugPolicy) -> float:
    op_count = len(policy.operations)
    if op_count == 0:
        return 0.0
    sub_policy_penalty = len(policy.sub_policies) / 4.0
    op_penalty = op_count / 12.0
    mixing_penalty = (
        float(policy.mixing.get("mixup_alpha", 0.0))
        + float(policy.mixing.get("cutmix_alpha", 0.0))
    ) / 2.0
    return min(2.0, 0.50 * sub_policy_penalty + 0.35 * op_penalty + 0.15 * mixing_penalty)


def distortion_proxy(policy: AugPolicy) -> float:
    """Estimate augmentation aggressiveness without an extra image pass.

    The value is a lightweight proxy, not a perceptual metric. It is useful for
    selection pressure because overly aggressive policies often score well in a
    noisy rough run but fail under fuller evaluation.
    """
    ops = policy.operations
    if not ops:
        return 0.0
    weighted = [
        op.probability * op.magnitude * DESTRUCTIVE_OP_WEIGHTS.get(op.name, 1.0)
        for op in ops
    ]
    mixing = (
        float(policy.mixing.get("mixup_alpha", 0.0))
        + float(policy.mixing.get("cutmix_alpha", 0.0))
    )
    return min(2.0, sum(weighted) / len(weighted) + 0.20 * mixing)


def policy_objective(
    result: dict,
    policy: AugPolicy,
    objective_cfg: dict | None = None,
) -> dict:
    cfg = objective_cfg or {}
    accuracy_weight = float(cfg.get("accuracy_weight", 1.0))
    macro_f1_weight = float(cfg.get("macro_f1_weight", 0.0))
    distortion_weight = float(cfg.get("distortion_weight", 0.0))
    complexity_weight = float(cfg.get("complexity_weight", 0.0))

    accuracy = float(result.get("val_accuracy", 0.0) or 0.0)
    macro_f1 = float(result.get("val_macro_f1", 0.0) or 0.0)
    complexity = policy_complexity(policy)
    distortion = distortion_proxy(policy)
    score = (
        accuracy_weight * accuracy
        + macro_f1_weight * macro_f1
        - distortion_weight * distortion
        - complexity_weight * complexity
    )
    return {
        "objective_score": score,
        "policy_complexity": complexity,
        "distortion_proxy": distortion,
    }
