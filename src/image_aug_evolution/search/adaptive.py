from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from image_aug_evolution.augmentation.builder import build_eval_transform, build_train_transform
from image_aug_evolution.augmentation.policy import AugPolicy
from image_aug_evolution.augmentation.seed_policies import build_seed_policy
from image_aug_evolution.augmentation.validator import PolicyValidator
from image_aug_evolution.data.datasets import build_dataloaders, get_meta
from image_aug_evolution.evaluation.objective import policy_objective
from image_aug_evolution.llm.factory import build_policy_generator
from image_aug_evolution.models.classifiers import build_model
from image_aug_evolution.models.train import TrainConfig, _build_optimizer, evaluate, get_device, train_one_epoch
from image_aug_evolution.utils.io import write_json
from image_aug_evolution.utils.seeding import seed_everything


def _build_bundle(policy: AugPolicy, dataset_cfg: dict, train_cfg: dict, seed: int):
    meta = get_meta(dataset_cfg["name"], dataset_cfg.get("image_size"))
    train_transform = build_train_transform(policy, meta.image_size, meta.mean, meta.std)
    eval_transform = build_eval_transform(meta.image_size, meta.mean, meta.std)
    return build_dataloaders(
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


def _ranked_items(policy_results: list[dict]) -> list[dict]:
    return sorted(
        [
            {
                "policy": item["policy"],
                "score": item["result"].get("objective_score", item["result"].get("val_accuracy", 0.0)),
                "result": item["result"],
            }
            for item in policy_results
        ],
        key=lambda item: float(item["score"] or 0.0),
    )


def run_adaptive_policy_updates(config: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("logs", "policies", "tables", "figures", "models"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 0))
    seed_everything(seed)
    dataset_cfg = config["dataset"]
    adaptive_cfg = config.get("adaptive", {})
    train_cfg_dict = {**config["training"], **adaptive_cfg.get("training", {})}
    total_epochs = int(adaptive_cfg.get("epochs", train_cfg_dict.get("epochs", 10)))
    update_interval = max(1, int(adaptive_cfg.get("update_interval", 2)))
    objective_cfg = adaptive_cfg.get("objective", config.get("search", {}).get("objective", {}))
    use_ranked_feedback = bool(adaptive_cfg.get("ranked_feedback", True))
    generator = build_policy_generator(dataset_cfg["name"], seed=seed, config=config)
    validator = PolicyValidator(dataset_cfg["name"], allow_discouraged=bool(adaptive_cfg.get("allow_discouraged", True)))

    initial_seed = adaptive_cfg.get("initial_seed_policy", "standard")
    if initial_seed:
        active_policy = build_seed_policy(dataset_cfg["name"], str(initial_seed), policy_id="adaptive_policy_000")
    else:
        active_policy = generator.random_policy("adaptive_policy_000", source="adaptive_initial")
    write_json(active_policy.to_dict(), output_dir / "policies" / f"{active_policy.policy_id}.json")

    bundle = _build_bundle(active_policy, dataset_cfg, train_cfg_dict, seed)
    cfg = TrainConfig(
        model_name=train_cfg_dict.get("model_name", "resnet18"),
        pretrained=bool(train_cfg_dict.get("pretrained", False)),
        epochs=total_epochs,
        lr=float(train_cfg_dict.get("lr", 1e-3)),
        weight_decay=float(train_cfg_dict.get("weight_decay", 1e-4)),
        optimizer=train_cfg_dict.get("optimizer", "adamw"),
        device=train_cfg_dict.get("device", "auto"),
        save_model=bool(train_cfg_dict.get("save_model", False)),
    )
    device = get_device(cfg.device)
    model = build_model(cfg.model_name, num_classes=bundle.meta.num_classes, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, cfg)

    print(
        f"[adaptive] training {cfg.model_name} for {total_epochs} epochs on {device} "
        f"with policy updates every {update_interval} epoch(s)",
        flush=True,
    )

    history: list[dict] = []
    policy_results: list[dict] = []
    best_val = -1.0
    best_state = None
    active_bundle = bundle

    for epoch in range(total_epochs):
        vr = validator.validate(active_policy)
        if not vr.ok:
            raise ValueError(f"Adaptive policy became invalid: {vr.errors}")
        mixing = active_policy.mixing or {}
        loss = train_one_epoch(
            model,
            active_bundle.train_loader,
            criterion,
            optimizer,
            device,
            mixup_alpha=float(mixing.get("mixup_alpha", 0.0)),
            cutmix_alpha=float(mixing.get("cutmix_alpha", 0.0)),
        )
        val_metrics = evaluate(model, active_bundle.val_loader, device, active_bundle.meta.num_classes)
        result = {
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch": epoch + 1,
        }
        result.update(policy_objective(result, active_policy, objective_cfg))
        row = {
            "epoch": epoch + 1,
            "policy_id": active_policy.policy_id,
            "policy_source": active_policy.source,
            "train_loss": loss,
            **result,
        }
        history.append(row)
        print(
            f"[adaptive] epoch {epoch + 1:03d}/{total_epochs}: "
            f"policy={active_policy.policy_id} loss={loss:.4f} "
            f"val_acc={result['val_accuracy']:.4f} objective={result['objective_score']:.4f}",
            flush=True,
        )
        if result["val_accuracy"] > best_val:
            best_val = result["val_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        should_update = (epoch + 1) % update_interval == 0 and (epoch + 1) < total_epochs
        if not should_update:
            continue

        policy_results.append({"policy": active_policy, "result": result})
        ranked = _ranked_items(policy_results)
        next_id = f"adaptive_policy_{len(policy_results):03d}"
        feedback = {
            "epoch": epoch + 1,
            "total_epochs": total_epochs,
            "recent_result": result,
            "objective": objective_cfg,
        }
        if use_ranked_feedback and ranked and hasattr(generator, "mutate_from_ranked_population"):
            next_policy = generator.mutate_from_ranked_population(ranked, policy_id=next_id, feedback=feedback)
        else:
            next_policy = generator.mutate_from_feedback(active_policy, policy_id=next_id, feedback=feedback)
        vr_next = validator.validate(next_policy)
        if not vr_next.ok:
            next_policy = active_policy
            next_policy.notes = f"{next_policy.notes} Invalid adaptive update rejected: {vr_next.errors}".strip()
        else:
            write_json(next_policy.to_dict(), output_dir / "policies" / f"{next_policy.policy_id}.json")
            active_policy = next_policy
            active_bundle = _build_bundle(active_policy, dataset_cfg, train_cfg_dict, seed)

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate(model, active_bundle.val_loader, device, active_bundle.meta.num_classes)
    summary = {
        "method": "adaptive_ranked_llm_evolution" if use_ranked_feedback else "adaptive_llm_feedback",
        "epochs": total_epochs,
        "update_interval": update_interval,
        "train_size": active_bundle.train_size,
        "val_size": active_bundle.val_size,
        "test_size": active_bundle.test_size,
        "best_val_accuracy": best_val,
        "final_val_accuracy": final_val["accuracy"],
        "final_val_macro_f1": final_val["macro_f1"],
        "num_policy_updates": len(policy_results),
        "final_policy": active_policy.to_dict(),
        "objective": objective_cfg,
    }
    if train_cfg_dict.get("evaluate_test", True):
        test_metrics = evaluate(model, active_bundle.test_loader, device, active_bundle.meta.num_classes)
        summary.update({
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        })
    if cfg.save_model:
        torch.save(model.state_dict(), output_dir / "models" / "adaptive_best_model.pt")

    pd.DataFrame(history).to_csv(output_dir / "tables" / "adaptive_history.csv", index=False)
    write_json(summary, output_dir / "adaptive_summary.json")
    return summary
