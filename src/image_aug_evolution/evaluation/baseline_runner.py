from __future__ import annotations

from pathlib import Path

import pandas as pd

from image_aug_evolution.augmentation.baselines import build_named_baseline
from image_aug_evolution.data.datasets import build_dataloaders, get_meta
from image_aug_evolution.models.train import TrainConfig, train_and_evaluate
from image_aug_evolution.utils.io import write_json
from image_aug_evolution.utils.seeding import seed_everything


def baseline_mixing(name: str) -> dict[str, float]:
    name = name.lower()
    if name == "standard_mixup":
        return {"mixup_alpha": 0.20, "cutmix_alpha": 0.0}
    if name == "standard_cutmix":
        return {"mixup_alpha": 0.0, "cutmix_alpha": 0.40}
    return {"mixup_alpha": 0.0, "cutmix_alpha": 0.0}


def run_baselines(config: dict, output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("logs", "tables", "figures", "models"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    baselines = config.get("baselines", ["none", "standard", "randaugment", "trivialaugment"])
    seeds = config.get("seeds", [config.get("seed", 0)])
    meta = get_meta(dataset_cfg["name"], dataset_cfg.get("image_size"))
    results_csv = output_dir / "tables" / "baseline_results.csv"
    if results_csv.exists() and not bool(config.get("overwrite_results", False)):
        try:
            records: list[dict] = pd.read_csv(results_csv).to_dict(orient="records")
        except pd.errors.EmptyDataError:
            records = []
    else:
        records = []

    def has_completed(method: str, seed: int) -> bool:
        for record in records:
            if str(record.get("method")) != str(method):
                continue
            try:
                record_seed = int(record.get("seed", -1))
            except (TypeError, ValueError):
                continue
            if record_seed != int(seed):
                continue
            if pd.notna(record.get("test_accuracy")) or pd.notna(record.get("val_accuracy")):
                return True
        return False

    for baseline in baselines:
        for seed in seeds:
            if has_completed(str(baseline), int(seed)):
                print(f"[baseline] skip existing method={baseline} seed={seed}", flush=True)
                continue
            seed_everything(int(seed))
            print(f"[baseline] method={baseline} seed={seed}", flush=True)
            train_transform = build_named_baseline(baseline, meta.image_size, meta.mean, meta.std)
            eval_transform = build_named_baseline("none", meta.image_size, meta.mean, meta.std)
            bundle = build_dataloaders(
                dataset_name=dataset_cfg["name"],
                root=dataset_cfg.get("root", "data/raw/image_datasets"),
                train_transform=train_transform,
                eval_transform=eval_transform,
                batch_size=int(training_cfg.get("batch_size", 64)),
                num_workers=int(training_cfg.get("num_workers", 0)),
                seed=int(seed),
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
                model_name=training_cfg.get("model_name", "resnet18"),
                pretrained=bool(training_cfg.get("pretrained", False)),
                epochs=int(training_cfg.get("epochs", 10)),
                lr=float(training_cfg.get("lr", 1e-3)),
                weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
                optimizer=training_cfg.get("optimizer", "adamw"),
                device=training_cfg.get("device", "auto"),
                save_model=bool(training_cfg.get("save_model", False)),
            )
            result = train_and_evaluate(
                bundle.train_loader,
                bundle.val_loader,
                bundle.test_loader if training_cfg.get("evaluate_test", True) else None,
                bundle.meta.num_classes,
                cfg,
                mixing=baseline_mixing(str(baseline)),
                output_model_path=output_dir / "models" / f"{baseline}_seed{seed}.pt",
            )
            record = {
                "method": baseline,
                "seed": int(seed),
                "train_size": bundle.train_size,
                "val_size": bundle.val_size,
                "test_size": bundle.test_size,
                **{k: v for k, v in result.items() if k not in {"history", "val_confusion_matrix", "test_confusion_matrix"}},
            }
            records.append(record)
            write_json(result, output_dir / "logs" / f"baseline_{baseline}_seed{seed}.json")
            pd.DataFrame(records).to_csv(results_csv, index=False)
            print(
                f"[baseline] completed method={baseline} seed={seed} "
                f"val_acc={record.get('val_accuracy'):.4f} "
                f"test_acc={record.get('test_accuracy'):.4f}",
                flush=True,
            )
    df = pd.DataFrame(records)
    df.to_csv(results_csv, index=False)
    return records
