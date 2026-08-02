from __future__ import annotations

import statistics
import json
from pathlib import Path

import pandas as pd

from image_aug_evolution.augmentation.builder import build_eval_transform, build_train_transform
from image_aug_evolution.augmentation.policy import AugPolicy
from image_aug_evolution.augmentation.seed_policies import build_seed_policies
from image_aug_evolution.augmentation.validator import PolicyValidator
from image_aug_evolution.data.datasets import build_dataloaders, get_meta
from image_aug_evolution.evaluation.objective import policy_objective
from image_aug_evolution.llm.factory import build_policy_generator
from image_aug_evolution.models.train import TrainConfig, train_and_evaluate
from image_aug_evolution.search.population import PolicyRecord
from image_aug_evolution.utils.io import append_jsonl, write_json
from image_aug_evolution.utils.seeding import seed_everything


def evaluate_policy(
    policy: AugPolicy,
    dataset_cfg: dict,
    train_cfg: dict,
    seed: int,
    output_dir: str | Path,
    stage: str,
) -> dict:
    seed_everything(int(seed))
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
        bundle.test_loader if train_cfg.get("evaluate_test", False) else None,
        bundle.meta.num_classes,
        cfg,
        mixing=policy.mixing,
        output_model_path=Path(output_dir) / "models" / f"{policy.policy_id}_{stage}.pt",
    )
    result.update({
        "policy_id": policy.policy_id,
        "stage": stage,
        "train_size": bundle.train_size,
        "val_size": bundle.val_size,
        "test_size": bundle.test_size,
        "epochs": cfg.epochs,
    })
    return result


def _record_result(record: PolicyRecord, stage: str) -> dict:
    return (record.rough_result if stage == "rough" else record.full_result) or {}


def _record_score(record: PolicyRecord, stage: str, objective_cfg: dict | None = None) -> float:
    result = _record_result(record, stage)
    if not result:
        return float("-inf")
    if "objective_score" in result:
        return float(result["objective_score"])
    return float(policy_objective(result, record.policy, objective_cfg).get("objective_score", float("-inf")))


def _select_top(
    records: list[PolicyRecord],
    k: int,
    stage: str = "rough",
    objective_cfg: dict | None = None,
) -> list[PolicyRecord]:
    return sorted(records, key=lambda r: _record_score(r, stage, objective_cfg), reverse=True)[:k]


def _ranked_population_payload(
    records: list[PolicyRecord],
    stage: str,
    objective_cfg: dict | None = None,
    max_items: int = 8,
) -> list[dict]:
    ranked = sorted(
        [r for r in records if r.validation_ok and _record_result(r, stage)],
        key=lambda r: _record_score(r, stage, objective_cfg),
    )
    if len(ranked) > max_items:
        half = max_items // 2
        ranked = ranked[:half] + ranked[-(max_items - half):]
    payload = []
    for record in ranked:
        result = dict(_record_result(record, stage))
        result.update(policy_objective(result, record.policy, objective_cfg))
        payload.append({
            "policy": record.policy,
            "score": _record_score(record, stage, objective_cfg),
            "result": result,
            "generation": record.generation,
        })
    return payload


def _policy_signature(policy: AugPolicy) -> str:
    payload = {
        "sub_policies": [
            [
                {
                    "name": op.name,
                    "probability": round(float(op.probability), 3),
                    "magnitude": round(float(op.magnitude), 3),
                }
                for op in sub
            ]
            for sub in policy.sub_policies
        ],
        "mixing": {
            "mixup_alpha": round(float(policy.mixing.get("mixup_alpha", 0.0)), 3),
            "cutmix_alpha": round(float(policy.mixing.get("cutmix_alpha", 0.0)), 3),
        },
    }
    return json.dumps(payload, sort_keys=True)


def _load_baseline_reference(output_dir: Path) -> list[dict]:
    path = output_dir / "tables" / "baseline_results.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return []
    rows: list[dict] = []
    for record in df.to_dict(orient="records"):
        rows.append({
            "method": record.get("method"),
            "val_accuracy": record.get("val_accuracy"),
            "val_macro_f1": record.get("val_macro_f1"),
            "test_accuracy": record.get("test_accuracy"),
            "test_macro_f1": record.get("test_macro_f1"),
        })
    return sorted(rows, key=lambda row: float(row.get("val_accuracy") or 0.0), reverse=True)


def run_evolutionary_search(config: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("logs", "policies", "tables", "figures", "models"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    dataset_cfg = config["dataset"]
    search_cfg = config["search"]
    objective_cfg = search_cfg.get("objective", {})
    validator = PolicyValidator(dataset_cfg["name"], allow_discouraged=bool(search_cfg.get("allow_discouraged", True)))
    generator = build_policy_generator(dataset_cfg["name"], seed=int(config.get("seed", 0)), config=config)
    population_size = int(search_cfg.get("population_size", 6))
    generations = int(search_cfg.get("generations", 2))
    elite_count = int(search_cfg.get("elite_count", 2))
    offspring_count = int(search_cfg.get("offspring_count", population_size))
    full_top_k = int(search_cfg.get("full_top_k", 2))
    use_feedback = bool(search_cfg.get("use_feedback", True))
    ranked_prompting = bool(search_cfg.get("ranked_prompting", False))
    seed_with_baselines = bool(search_cfg.get("seed_with_baselines", False))
    max_unique_attempts = int(search_cfg.get("max_unique_attempts", 4))
    method_name = "ranked_llm_evolution" if use_feedback and ranked_prompting else ("llm_guided_evolution" if use_feedback else "pure_evolution")
    seed = int(config.get("seed", 0))
    rough_train_cfg = {**config["training"], **config.get("rough_training", {})}
    full_train_cfg = {**config["training"], **config.get("full_training", {})}
    baseline_reference = _load_baseline_reference(output_dir)
    records: list[PolicyRecord] = []
    initial_policies: list[AugPolicy] = []
    if seed_with_baselines:
        seed_names = list(search_cfg.get("baseline_seed_names", ["standard", "conservative", "randaugment", "trivialaugment"]))
        initial_policies.extend(build_seed_policies(dataset_cfg["name"], seed_names)[:population_size])
    remaining = max(0, population_size - len(initial_policies))
    if remaining:
        initial_policies.extend(generator.initial_population(remaining))
    if not initial_policies:
        initial_policies.extend(generator.initial_population(population_size))
    for policy in initial_policies:
        records.append(PolicyRecord(policy=policy, generation=0))
        write_json(policy.to_dict(), output_dir / "policies" / f"{policy.policy_id}.json")
    seen_signatures = {_policy_signature(policy) for policy in initial_policies}
    log_path = output_dir / "evolution_log.jsonl"
    all_records: list[PolicyRecord] = []

    for gen in range(generations + 1):
        print(f"[evolution] generation={gen}/{generations}", flush=True)
        gen_records = [r for r in records if r.generation == gen]
        for record in gen_records:
            print(f"[evolution] rough policy={record.policy.policy_id}", flush=True)
            vr = validator.validate(record.policy)
            record.validation_ok = vr.ok
            record.validation_errors = vr.errors
            record.validation_warnings = vr.warnings
            if not vr.ok:
                record.status = "invalid"
                append_jsonl(record.to_dict(), log_path)
                continue
            write_json(record.policy.to_dict(), output_dir / "policies" / f"{record.policy.policy_id}.json")
            record.status = "rough_evaluated"
            record.rough_result = evaluate_policy(
                record.policy,
                dataset_cfg,
                rough_train_cfg,
                seed=seed + gen,
                output_dir=output_dir,
                stage="rough",
            )
            record.rough_result.update(policy_objective(record.rough_result, record.policy, objective_cfg))
            append_jsonl(record.to_dict(), log_path)
            print(
                f"[evolution] completed rough policy={record.policy.policy_id} "
                f"val_acc={record.rough_result.get('val_accuracy'):.4f} "
                f"objective={record.rough_result.get('objective_score'):.4f}",
                flush=True,
            )
        valid_records = [r for r in records if r.validation_ok and r.rough_result is not None]
        all_records = valid_records
        if gen == generations:
            break
        elites = _select_top(valid_records, elite_count, stage="rough", objective_cfg=objective_cfg)
        scores = [_record_score(r, "rough", objective_cfg) for r in valid_records if _record_score(r, "rough", objective_cfg) > float("-inf")]
        pop_mean = statistics.mean(scores) if scores else 0.0
        ranked_population = _ranked_population_payload(
            valid_records,
            stage="rough",
            objective_cfg=objective_cfg,
            max_items=int(search_cfg.get("ranked_prompt_max_items", 8)),
        )
        next_records: list[PolicyRecord] = []
        for i in range(offspring_count):
            child_policy = None
            for attempt in range(max(1, max_unique_attempts)):
                policy_id = f"gen{gen+1:02d}_{'cross' if i % 3 == 0 and len(ranked_population) >= 2 else 'mut'}_{i:03d}"
                if attempt:
                    policy_id = f"{policy_id}_try{attempt}"
                if ranked_prompting and use_feedback and ranked_population:
                    feedback = {
                        "population_mean": pop_mean,
                        "generation": gen,
                        "objective": objective_cfg,
                        "baseline_reference": baseline_reference,
                    }
                    if len(ranked_population) >= 2 and i % 3 == 0 and hasattr(generator, "crossover_from_ranked_population"):
                        child_policy = generator.crossover_from_ranked_population(ranked_population, policy_id=policy_id, feedback=feedback)
                    elif hasattr(generator, "mutate_from_ranked_population"):
                        child_policy = generator.mutate_from_ranked_population(ranked_population, policy_id=policy_id, feedback=feedback)
                    else:
                        parent = elites[i % len(elites)] if elites else valid_records[i % len(valid_records)]
                        child_policy = generator.mutate_from_feedback(parent.policy, policy_id=policy_id, feedback=feedback)
                elif i < len(elites):
                    parent = elites[i]
                    child_policy = generator.mutate_from_feedback(
                        parent.policy,
                        policy_id=policy_id,
                        feedback={**(parent.rough_result or {}), "population_mean": pop_mean} if use_feedback else None,
                    )
                elif len(elites) >= 2 and i % 3 == 0:
                    p1, p2 = elites[0], elites[(i % len(elites))]
                    child_policy = generator.crossover(p1.policy, p2.policy, policy_id=policy_id)
                    if not use_feedback:
                        child_policy.source = "pure_evolution_crossover"
                else:
                    parent = elites[i % len(elites)] if elites else valid_records[i % len(valid_records)]
                    child_policy = generator.mutate_from_feedback(
                        parent.policy,
                        policy_id=policy_id,
                        feedback={**(parent.rough_result or {}), "population_mean": pop_mean} if use_feedback else None,
                    )
                signature = _policy_signature(child_policy)
                if signature not in seen_signatures or attempt + 1 >= max(1, max_unique_attempts):
                    seen_signatures.add(signature)
                    break
                child_policy.notes = f"{child_policy.notes} Duplicate retry requested by novelty filter.".strip()
            next_records.append(PolicyRecord(policy=child_policy, generation=gen + 1))
        records.extend(next_records)

    finalists = _select_top(all_records, full_top_k, stage="rough", objective_cfg=objective_cfg)
    include_policy_ids = set(search_cfg.get("full_include_policy_ids", []))
    include_source_prefixes = tuple(search_cfg.get("full_include_source_prefixes", []))
    if include_policy_ids or include_source_prefixes:
        finalist_ids = {record.policy.policy_id for record in finalists}
        for record in all_records:
            source = str(record.policy.source)
            if (
                record.policy.policy_id in include_policy_ids
                or (include_source_prefixes and source.startswith(include_source_prefixes))
            ) and record.policy.policy_id not in finalist_ids:
                finalists.append(record)
                finalist_ids.add(record.policy.policy_id)
    for record in finalists:
        print(f"[evolution] full policy={record.policy.policy_id}", flush=True)
        record.status = "full_evaluated"
        record.full_result = evaluate_policy(
            record.policy,
            dataset_cfg,
            full_train_cfg,
            seed=seed + 10_000,
            output_dir=output_dir,
            stage="full",
        )
        record.full_result.update(policy_objective(record.full_result, record.policy, objective_cfg))
        append_jsonl(record.to_dict(), log_path)
        print(
            f"[evolution] completed full policy={record.policy.policy_id} "
            f"val_acc={record.full_result.get('val_accuracy'):.4f} "
            f"test_acc={record.full_result.get('test_accuracy'):.4f} "
            f"objective={record.full_result.get('objective_score'):.4f}",
            flush=True,
        )
    summary = {
        "method": method_name,
        "best_rough": finalists[0].to_dict() if finalists else None,
        "best_full": _select_top(finalists, 1, stage="full", objective_cfg=objective_cfg)[0].to_dict() if finalists and finalists[0].full_result else None,
        "num_records": len(records),
        "num_valid": len([r for r in records if r.validation_ok]),
        "num_full_evaluated": len(finalists),
        "seed_with_baselines": seed_with_baselines,
        "ranked_prompting": ranked_prompting,
        "objective": objective_cfg,
    }
    write_json(summary, output_dir / "evolution_summary.json")
    write_json([r.to_dict() for r in records], output_dir / "all_policy_records.json")
    return summary
