from __future__ import annotations

from pathlib import Path

import pandas as pd
from torchvision import transforms

from image_aug_evolution.augmentation.builder import (
    build_eval_transform,
    build_fixmatch_weak_transform,
    build_train_transform,
)
from image_aug_evolution.augmentation.policy import AugPolicy
from image_aug_evolution.augmentation.seed_policies import build_seed_policy
from image_aug_evolution.augmentation.validator import PolicyValidator
from image_aug_evolution.data.datasets import build_ssl_dataloaders, get_meta
from image_aug_evolution.models.fixmatch import FixMatchConfig, train_fixmatch_and_evaluate
from image_aug_evolution.utils.io import read_json, write_json
from image_aug_evolution.utils.seeding import seed_everything


def _zero_mixing(policy: AugPolicy) -> AugPolicy:
    cloned = AugPolicy.from_dict(policy.to_dict())
    cloned.mixing = {"mixup_alpha": 0.0, "cutmix_alpha": 0.0}
    return cloned


def _load_policy(path: str | Path, policy_id: str | None = None, source: str | None = None) -> AugPolicy:
    policy = AugPolicy.from_dict(read_json(path))
    if policy_id:
        policy.policy_id = policy_id
    if source:
        policy.source = source
    return policy


def _fixmatch_strong_named_transform(
    name: str,
    dataset_name: str,
    image_size: int,
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
):
    """Canonical strong branches for FixMatch-style comparisons."""
    name = name.lower()
    ops = [transforms.Resize((image_size, image_size))]
    if dataset_name.lower() == "flowers102":
        ops.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        ops.extend([
            transforms.RandomCrop(image_size, padding=max(2, image_size // 8), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ])
        if dataset_name.lower() == "eurosat":
            ops.append(transforms.RandomVerticalFlip())
    if name in {"standard", "weak", "weak_standard"}:
        pass
    elif name == "randaugment":
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif name in {"trivialaugment", "trivialaugmentwide"}:
        ops.append(transforms.TrivialAugmentWide())
    elif name == "color_jitter":
        ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04))
    else:
        raise ValueError(f"Unknown FixMatch named strong transform: {name}")
    ops.extend([transforms.ToTensor(), transforms.Normalize(normalize_mean, normalize_std)])
    return transforms.Compose(ops)


def _policy_to_method(policy: AugPolicy) -> str:
    policy_id = policy.policy_id.lower()
    for prefix in ("seed_", "strong_"):
        if policy_id.startswith(prefix):
            policy_id = policy_id[len(prefix):]
    return f"fixmatch_{policy_id}"


def evaluate_fixmatch_policy(
    strong_policy: AugPolicy | None,
    config: dict,
    output_dir: str | Path,
    seed: int,
    method: str,
    named_strong_transform: str | None = None,
    training_override: dict | None = None,
    fixmatch_override: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    seed_everything(int(seed))
    dataset_cfg = config["dataset"]
    training_cfg = {**config["training"], **config.get("fixmatch_training", {}), **(training_override or {})}
    fixmatch_cfg = {**config.get("fixmatch", {}), **(fixmatch_override or {})}
    meta = get_meta(dataset_cfg["name"], dataset_cfg.get("image_size"))
    weak_transform = build_fixmatch_weak_transform(dataset_cfg["name"], meta.image_size, meta.mean, meta.std)
    labeled_transform = weak_transform
    eval_transform = build_eval_transform(meta.image_size, meta.mean, meta.std)
    if named_strong_transform is not None:
        strong_transform = _fixmatch_strong_named_transform(
            named_strong_transform,
            dataset_cfg["name"],
            meta.image_size,
            meta.mean,
            meta.std,
        )
        policy_id = f"strong_{named_strong_transform}"
        source = "fixmatch_named_strong"
    else:
        assert strong_policy is not None
        strong_policy = _zero_mixing(strong_policy)
        strong_transform = build_train_transform(strong_policy, meta.image_size, meta.mean, meta.std)
        policy_id = strong_policy.policy_id
        source = strong_policy.source
    batch_size = int(training_cfg.get("batch_size", 64))
    mu = int(fixmatch_cfg.get("mu", 2))
    bundle = build_ssl_dataloaders(
        dataset_name=dataset_cfg["name"],
        root=dataset_cfg.get("root", "data/raw/image_datasets"),
        labeled_transform=labeled_transform,
        weak_transform=weak_transform,
        strong_transform=strong_transform,
        eval_transform=eval_transform,
        labeled_batch_size=batch_size,
        unlabeled_batch_size=int(training_cfg.get("unlabeled_batch_size", batch_size * mu)),
        eval_batch_size=int(training_cfg.get("eval_batch_size", batch_size)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        seed=seed,
        download=bool(dataset_cfg.get("download", True)),
        train_per_class=dataset_cfg.get("train_per_class"),
        val_per_class=dataset_cfg.get("val_per_class"),
        train_fraction=dataset_cfg.get("train_fraction"),
        max_train=dataset_cfg.get("max_train"),
        max_val=dataset_cfg.get("max_val"),
        max_test=dataset_cfg.get("max_test"),
        unlabeled_per_class=dataset_cfg.get("unlabeled_per_class"),
        max_unlabeled=dataset_cfg.get("max_unlabeled"),
        image_size=dataset_cfg.get("image_size"),
        download_url=dataset_cfg.get("download_url"),
        archive_filename=dataset_cfg.get("archive_filename"),
        archive_md5=dataset_cfg.get("archive_md5"),
    )
    cfg = FixMatchConfig(
        model_name=training_cfg.get("model_name", "resnet18_cifar"),
        pretrained=bool(training_cfg.get("pretrained", False)),
        epochs=int(training_cfg.get("epochs", 20)),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        optimizer=training_cfg.get("optimizer", "adamw"),
        device=training_cfg.get("device", "auto"),
        save_model=bool(training_cfg.get("save_model", False)),
        threshold=float(fixmatch_cfg.get("threshold", 0.95)),
        lambda_u=float(fixmatch_cfg.get("lambda_u", 1.0)),
        max_steps_per_epoch=fixmatch_cfg.get("max_steps_per_epoch"),
    )
    result = train_fixmatch_and_evaluate(
        bundle.labeled_loader,
        bundle.unlabeled_loader,
        bundle.val_loader,
        bundle.test_loader if training_cfg.get("evaluate_test", True) else None,
        bundle.meta.num_classes,
        cfg,
        output_model_path=output_dir / "models" / f"{policy_id}_fixmatch.pt",
    )
    return {
        "method": method,
        "stage": "fixmatch",
        "policy_id": policy_id,
        "source": source,
        "seed": seed,
        "valid": True,
        "train_size": bundle.labeled_size,
        "unlabeled_size": bundle.unlabeled_size,
        "val_size": bundle.val_size,
        "test_size": bundle.test_size,
        "epochs": cfg.epochs,
        "threshold": cfg.threshold,
        "lambda_u": cfg.lambda_u,
        **{k: v for k, v in result.items() if k not in {"history", "val_confusion_matrix", "test_confusion_matrix"}},
    }


def run_fixmatch(config: dict, output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("logs", "policies", "tables", "figures", "models"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    seeds = [int(s) for s in config.get("seeds", [seed])]
    fixmatch_cfg = config.get("fixmatch", {})
    validator = PolicyValidator(config["dataset"]["name"], allow_discouraged=True)
    results_csv = output_dir / "tables" / "fixmatch_results.csv"
    if results_csv.exists() and not bool(config.get("overwrite_results", False)):
        try:
            rows: list[dict] = pd.read_csv(results_csv).to_dict(orient="records")
        except pd.errors.EmptyDataError:
            rows = []
    else:
        rows = []

    def has_completed(method: str, policy_id: str, eval_seed: int) -> bool:
        for record in rows:
            if str(record.get("method")) != str(method):
                continue
            if str(record.get("policy_id")) != str(policy_id):
                continue
            try:
                record_seed = int(record.get("seed", -1))
            except (TypeError, ValueError):
                continue
            if record_seed != int(eval_seed):
                continue
            if pd.notna(record.get("test_accuracy")) or pd.notna(record.get("val_accuracy")):
                return True
        return False

    named_strong_transforms = list(fixmatch_cfg.get("named_strong_transforms", ["standard", "randaugment", "trivialaugment"]))
    for name in named_strong_transforms:
        method = f"fixmatch_{name}"
        policy_id = f"strong_{name}"
        for eval_seed in seeds:
            if has_completed(method, policy_id, eval_seed):
                print(f"[fixmatch] skip existing named strong={name} seed={eval_seed}", flush=True)
                continue
            print(f"[fixmatch] named strong={name} seed={eval_seed}", flush=True)
            result = evaluate_fixmatch_policy(
                strong_policy=None,
                config=config,
                output_dir=output_dir,
                seed=eval_seed,
                method=method,
                named_strong_transform=name,
            )
            rows.append(result)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
            print(
                f"[fixmatch] completed {name} seed={eval_seed}: val_acc={result.get('val_accuracy'):.4f} "
                f"test_acc={result.get('test_accuracy'):.4f} mask={result.get('final_pseudo_mask_rate'):.3f}",
                flush=True,
            )

    for i, seed_name in enumerate(fixmatch_cfg.get("seed_policy_names", [])):
        policy = build_seed_policy(config["dataset"]["name"], seed_name, policy_id=f"fix_seed_{i:02d}_{seed_name.lower()}")
        policy = _zero_mixing(policy)
        vr = validator.validate(policy)
        if not vr.ok:
            for eval_seed in seeds:
                rows.append({
                    "method": _policy_to_method(policy),
                    "stage": "fixmatch",
                    "policy_id": policy.policy_id,
                    "source": policy.source,
                    "seed": eval_seed,
                    "valid": False,
                    "errors": "; ".join(vr.errors),
                })
            continue
        write_json(policy.to_dict(), output_dir / "policies" / f"{policy.policy_id}.json")
        method = _policy_to_method(policy)
        for eval_seed in seeds:
            if has_completed(method, policy.policy_id, eval_seed):
                print(f"[fixmatch] skip existing seed policy={policy.policy_id} seed={eval_seed}", flush=True)
                continue
            print(f"[fixmatch] seed policy={policy.policy_id} seed={eval_seed}", flush=True)
            result = evaluate_fixmatch_policy(policy, config, output_dir, eval_seed, method=method)
            rows.append(result)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
            print(
                f"[fixmatch] completed {policy.policy_id} seed={eval_seed}: val_acc={result.get('val_accuracy'):.4f} "
                f"test_acc={result.get('test_accuracy'):.4f} mask={result.get('final_pseudo_mask_rate'):.3f}",
                flush=True,
            )

    for item in fixmatch_cfg.get("policy_files", []):
        if isinstance(item, str):
            path = item
            method = f"fixmatch_{Path(path).stem}"
            policy_id = f"fix_{Path(path).stem}"
            source = "external_policy"
        else:
            path = item["path"]
            method = item.get("method", f"fixmatch_{Path(path).stem}")
            policy_id = item.get("policy_id", f"fix_{Path(path).stem}")
            source = item.get("source", "external_policy")
        policy = _zero_mixing(_load_policy(path, policy_id=policy_id, source=source))
        vr = validator.validate(policy)
        if not vr.ok:
            for eval_seed in seeds:
                rows.append({
                    "method": method,
                    "stage": "fixmatch",
                    "policy_id": policy.policy_id,
                    "source": policy.source,
                    "seed": eval_seed,
                    "valid": False,
                    "errors": "; ".join(vr.errors),
                })
            continue
        write_json(policy.to_dict(), output_dir / "policies" / f"{policy.policy_id}.json")
        for eval_seed in seeds:
            if has_completed(method, policy.policy_id, eval_seed):
                print(f"[fixmatch] skip existing external policy={policy.policy_id} seed={eval_seed}", flush=True)
                continue
            print(f"[fixmatch] external policy={policy.policy_id} seed={eval_seed}", flush=True)
            result = evaluate_fixmatch_policy(policy, config, output_dir, eval_seed, method=method)
            rows.append(result)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
            print(
                f"[fixmatch] completed {policy.policy_id} seed={eval_seed}: val_acc={result.get('val_accuracy'):.4f} "
                f"test_acc={result.get('test_accuracy'):.4f} mask={result.get('final_pseudo_mask_rate'):.3f}",
                flush=True,
            )

    pd.DataFrame(rows).to_csv(results_csv, index=False)
    return rows
