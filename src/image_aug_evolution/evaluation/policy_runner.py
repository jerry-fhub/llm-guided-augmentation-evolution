from __future__ import annotations

from pathlib import Path

import pandas as pd

from image_aug_evolution.augmentation.builder import build_eval_transform, build_train_transform
from image_aug_evolution.augmentation.policy import AugPolicy
from image_aug_evolution.augmentation.validator import PolicyValidator
from image_aug_evolution.data.datasets import build_dataloaders, get_meta
from image_aug_evolution.llm.factory import build_policy_generator
from image_aug_evolution.models.train import TrainConfig, train_and_evaluate
from image_aug_evolution.utils.io import read_json, write_json
from image_aug_evolution.utils.seeding import seed_everything


def evaluate_single_policy(
    policy: AugPolicy,
    config: dict,
    output_dir: str | Path,
    seed: int,
    stage: str,
    training_override: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    seed_everything(int(seed))
    dataset_cfg = config["dataset"]
    train_cfg = {**config["training"], **(training_override or {})}
    meta = get_meta(dataset_cfg["name"], dataset_cfg.get("image_size"))
    train_transform = build_train_transform(policy, meta.image_size, meta.mean, meta.std)
    eval_transform = build_eval_transform(meta.image_size, meta.mean, meta.std)
    bundle = build_dataloaders(
        dataset_name=dataset_cfg["name"],
        root=dataset_cfg.get("root", "data/raw/image_datasets"),
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        seed=seed,
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
    cfg = TrainConfig(
        model_name=train_cfg.get("model_name", "resnet18"),
        pretrained=bool(train_cfg.get("pretrained", False)),
        epochs=int(train_cfg.get("epochs", 10)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        optimizer=train_cfg.get("optimizer", "adamw"),
        device=train_cfg.get("device", "auto"),
        save_model=bool(train_cfg.get("save_model", False)),
    )
    result = train_and_evaluate(
        bundle.train_loader,
        bundle.val_loader,
        bundle.test_loader if train_cfg.get("evaluate_test", True) else None,
        bundle.meta.num_classes,
        cfg,
        mixing=policy.mixing,
        output_model_path=output_dir / "models" / f"{policy.policy_id}_{stage}.pt",
    )
    return {
        "policy_id": policy.policy_id,
        "source": policy.source,
        "seed": seed,
        "stage": stage,
        "train_size": bundle.train_size,
        "val_size": bundle.val_size,
        "test_size": bundle.test_size,
        "epochs": cfg.epochs,
        **{k: v for k, v in result.items() if k not in {"history", "val_confusion_matrix", "test_confusion_matrix"}},
    }


def load_external_policies(path: str | Path) -> list[AugPolicy]:
    data = read_json(path)
    if isinstance(data, dict) and "policies" in data:
        data = data["policies"]
    if isinstance(data, dict):
        data = [data]
    return [AugPolicy.from_dict(item) for item in data]


def run_one_shot(config: dict, output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    seeds = [int(s) for s in config.get("seeds", [seed])]
    count = int(config.get("one_shot_count", 1))
    if count <= 0:
        pd.DataFrame().to_csv(output_dir / "tables" / "one_shot_results.csv", index=False)
        return []
    generator = build_policy_generator(config["dataset"]["name"], seed=seed, config=config)
    policy_path = config.get("llm_policy_file")
    if policy_path:
        policies = load_external_policies(policy_path)
    else:
        policies = [generator.random_policy("one_shot_000", source="one_shot_heuristic_llm")]
    validator = PolicyValidator(config["dataset"]["name"])
    results_csv = output_dir / "tables" / "one_shot_results.csv"
    if results_csv.exists() and not bool(config.get("overwrite_results", False)):
        try:
            rows = pd.read_csv(results_csv).to_dict(orient="records")
        except pd.errors.EmptyDataError:
            rows = []
    else:
        rows = []

    def has_completed(policy_id: str, eval_seed: int) -> bool:
        for record in rows:
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

    method = str(config.get("policy_eval_method", "one_shot_llm"))
    for policy in policies[:count]:
        write_json(policy.to_dict(), output_dir / "policies" / f"{policy.policy_id}.json")
        for eval_seed in seeds:
            if has_completed(policy.policy_id, eval_seed):
                print(f"[one-shot] skip existing policy={policy.policy_id} seed={eval_seed}", flush=True)
                continue
            print(f"[one-shot] policy={policy.policy_id} seed={eval_seed}", flush=True)
            vr = validator.validate(policy)
            if not vr.ok:
                rows.append({
                    "method": method,
                    "policy_id": policy.policy_id,
                    "seed": eval_seed,
                    "valid": False,
                    "errors": "; ".join(vr.errors),
                })
                continue
            result = evaluate_single_policy(
                policy,
                config,
                output_dir,
                seed=eval_seed,
                stage="one_shot",
                training_override=config.get("full_training", {}),
            )
            rows.append({"method": method, "valid": True, **result})
            pd.DataFrame(rows).to_csv(results_csv, index=False)
            print(
                f"[one-shot] completed policy={policy.policy_id} seed={eval_seed} "
                f"val_acc={result.get('val_accuracy'):.4f} "
                f"test_acc={result.get('test_accuracy'):.4f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    return rows


def run_random_search(config: dict, output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    seed = int(config.get("seed", 0))
    search_cfg = config.get("random_search", {})
    count = int(search_cfg.get("count", 6))
    top_k = int(search_cfg.get("full_top_k", 2))
    if count <= 0:
        pd.DataFrame().to_csv(output_dir / "tables" / "random_search_results.csv", index=False)
        return []
    generator = build_policy_generator(config["dataset"]["name"], seed=seed + 999, config=config)
    validator = PolicyValidator(config["dataset"]["name"])
    policies = [generator.random_policy(f"random_{i:03d}", source="random_search") for i in range(count)]
    rows = []
    rough_rows = []
    for policy in policies:
        print(f"[random] rough policy={policy.policy_id}", flush=True)
        vr = validator.validate(policy)
        if not vr.ok:
            row = {"method": "random_search", "policy_id": policy.policy_id, "valid": False, "errors": "; ".join(vr.errors)}
            rows.append(row)
            rough_rows.append(row)
            continue
        result = evaluate_single_policy(
            policy,
            config,
            output_dir,
            seed=seed,
            stage="random_rough",
            training_override=config.get("rough_training", {}),
        )
        row = {"method": "random_search", "valid": True, **result}
        rows.append(row)
        rough_rows.append(row)
        write_json(policy.to_dict(), output_dir / "policies" / f"{policy.policy_id}.json")
        pd.DataFrame(rows).to_csv(output_dir / "tables" / "random_search_results.csv", index=False)
        print(
            f"[random] completed rough policy={policy.policy_id} "
            f"val_acc={result.get('val_accuracy'):.4f}",
            flush=True,
        )
    ranked = sorted([r for r in rough_rows if r.get("valid")], key=lambda r: r.get("val_accuracy", -1), reverse=True)[:top_k]
    policy_by_id = {p.policy_id: p for p in policies}
    for r in ranked:
        policy = policy_by_id[r["policy_id"]]
        print(f"[random] full policy={policy.policy_id}", flush=True)
        result = evaluate_single_policy(
            policy,
            config,
            output_dir,
            seed=seed + 10_000,
            stage="random_full",
            training_override=config.get("full_training", {}),
        )
        rows.append({"method": "random_search_full", "valid": True, **result})
        pd.DataFrame(rows).to_csv(output_dir / "tables" / "random_search_results.csv", index=False)
        print(
            f"[random] completed full policy={policy.policy_id} "
            f"val_acc={result.get('val_accuracy'):.4f} "
            f"test_acc={result.get('test_accuracy'):.4f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(output_dir / "tables" / "random_search_results.csv", index=False)
    return rows
