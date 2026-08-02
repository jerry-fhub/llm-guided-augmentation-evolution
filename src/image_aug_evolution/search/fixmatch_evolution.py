from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from image_aug_evolution.augmentation.policy import AugPolicy
from image_aug_evolution.augmentation.seed_policies import build_seed_policies
from image_aug_evolution.augmentation.validator import PolicyValidator
from image_aug_evolution.evaluation.fixmatch_runner import _load_policy, _zero_mixing, evaluate_fixmatch_policy
from image_aug_evolution.evaluation.objective import policy_objective
from image_aug_evolution.llm.factory import build_policy_generator
from image_aug_evolution.search.population import PolicyRecord
from image_aug_evolution.utils.io import append_jsonl, write_json


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
        "mixing": {"mixup_alpha": 0.0, "cutmix_alpha": 0.0},
    }
    return json.dumps(payload, sort_keys=True)


def repair_policy_for_search_space(policy: AugPolicy, max_sub_policies: int = 4, max_ops_per_sub_policy: int = 3) -> AugPolicy:
    """Keep LLM intent but enforce the local JSON policy shape."""
    repaired = AugPolicy.from_dict(policy.to_dict())
    repaired.sub_policies = [sub[:max_ops_per_sub_policy] for sub in repaired.sub_policies[:max_sub_policies]]
    repaired.sub_policies = [sub for sub in repaired.sub_policies if sub]
    repaired.mixing = {"mixup_alpha": 0.0, "cutmix_alpha": 0.0}
    if "repair" not in repaired.notes.lower():
        repaired.notes = f"{repaired.notes} Repaired to satisfy local max-subpolicy/max-operation constraints.".strip()
    return repaired


def _record_result(record: PolicyRecord, stage: str) -> dict:
    return (record.rough_result if stage == "rough" else record.full_result) or {}


def _record_score(record: PolicyRecord, stage: str, objective_cfg: dict | None = None) -> float:
    result = _record_result(record, stage)
    if not result:
        return float("-inf")
    if "objective_score" in result:
        return float(result["objective_score"])
    return float(policy_objective(result, record.policy, objective_cfg).get("objective_score", float("-inf")))


def _select_top(records: list[PolicyRecord], k: int, stage: str, objective_cfg: dict | None = None) -> list[PolicyRecord]:
    return sorted(records, key=lambda r: _record_score(r, stage, objective_cfg), reverse=True)[:k]


def _ranked_population_payload(
    records: list[PolicyRecord],
    stage: str,
    objective_cfg: dict | None,
    max_items: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        [r for r in records if r.validation_ok and _record_result(r, stage)],
        key=lambda r: _record_score(r, stage, objective_cfg),
    )
    if len(ranked) > max_items:
        half = max_items // 2
        ranked = ranked[:half] + ranked[-(max_items - half):]
    payload: list[dict[str, Any]] = []
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


def _load_fixmatch_baseline_reference(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return []
    rows: list[dict] = []
    for record in df.to_dict(orient="records"):
        if not bool(record.get("valid", True)):
            continue
        rows.append({
            "method": record.get("method"),
            "policy_id": record.get("policy_id"),
            "val_accuracy": record.get("val_accuracy"),
            "val_macro_f1": record.get("val_macro_f1"),
            "test_accuracy": record.get("test_accuracy"),
            "test_macro_f1": record.get("test_macro_f1"),
            "final_pseudo_mask_rate": record.get("final_pseudo_mask_rate"),
            "final_pseudo_confidence": record.get("final_pseudo_confidence"),
        })
    return sorted(rows, key=lambda row: float(row.get("val_accuracy") or 0.0), reverse=True)


def _make_initial_policies(config: dict, population_size: int, output_dir: Path) -> list[AugPolicy]:
    dataset_name = config["dataset"]["name"]
    evo_cfg = config.get("fixmatch_evolution", {})
    policies: list[AugPolicy] = []
    if evo_cfg.get("seed_with_baselines", True):
        seed_names = list(evo_cfg.get("baseline_seed_names", ["standard", "standard_color", "randaugment", "trivialaugment", "conservative"]))
        policies.extend(build_seed_policies(dataset_name, seed_names))
    for item in evo_cfg.get("initial_policy_files", []):
        if isinstance(item, str):
            path = item
            policy_id = f"init_{Path(path).stem}"
            source = "external_initial_policy"
        else:
            path = item["path"]
            policy_id = item.get("policy_id", f"init_{Path(path).stem}")
            source = item.get("source", "external_initial_policy")
        policies.append(_load_policy(path, policy_id=policy_id, source=source))
    generator = build_policy_generator(dataset_name, seed=int(config.get("seed", 0)), config=config)
    remaining = max(0, population_size - len(policies))
    if remaining:
        policies.extend(generator.initial_population(remaining))
    deduped: list[AugPolicy] = []
    seen: set[str] = set()
    for policy in policies:
        policy = repair_policy_for_search_space(_zero_mixing(policy))
        signature = _policy_signature(policy)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(policy)
        write_json(policy.to_dict(), output_dir / "policies" / f"{policy.policy_id}.json")
        if len(deduped) >= population_size:
            break
    return deduped


def _result_row(record: PolicyRecord, result: dict, method_name: str, seed: int) -> dict:
    return {
        "method": method_name,
        "stage": "fixmatch",
        "policy_id": record.policy.policy_id,
        "source": record.policy.source,
        "seed": seed,
        "generation": record.generation,
        "valid": True,
        **{
            key: value
            for key, value in result.items()
            if key not in {"history", "val_confusion_matrix", "test_confusion_matrix", "method", "stage", "policy_id", "source", "seed", "valid"}
        },
    }


def run_fixmatch_evolution(config: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("logs", "policies", "tables", "figures", "models"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset_name = config["dataset"]["name"]
    evo_cfg = config.get("fixmatch_evolution", {})
    objective_cfg = evo_cfg.get("objective", {})
    seed = int(config.get("seed", 0))
    population_size = int(evo_cfg.get("population_size", 6))
    generations = int(evo_cfg.get("generations", 1))
    elite_count = int(evo_cfg.get("elite_count", 2))
    offspring_count = int(evo_cfg.get("offspring_count", population_size))
    full_top_k = int(evo_cfg.get("full_top_k", 3))
    max_unique_attempts = int(evo_cfg.get("max_unique_attempts", 3))
    method_name = str(evo_cfg.get("method_name", "fixmatch_ranked_llm_evolution"))

    validator = PolicyValidator(dataset_name, allow_discouraged=bool(evo_cfg.get("allow_discouraged", True)))
    generator = build_policy_generator(dataset_name, seed=seed, config=config)
    baseline_reference = _load_fixmatch_baseline_reference(evo_cfg.get("baseline_reference_csv"))
    records = [PolicyRecord(policy=policy, generation=0) for policy in _make_initial_policies(config, population_size, output_dir)]
    seen_signatures = {_policy_signature(record.policy) for record in records}
    log_path = output_dir / "fixmatch_evolution_log.jsonl"

    for gen in range(generations + 1):
        print(f"[fixmatch-evolution] generation={gen}/{generations}", flush=True)
        gen_records = [record for record in records if record.generation == gen]
        for record in gen_records:
            print(f"[fixmatch-evolution] rough policy={record.policy.policy_id}", flush=True)
            record.policy = repair_policy_for_search_space(_zero_mixing(record.policy))
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
            record.rough_result = evaluate_fixmatch_policy(
                record.policy,
                config,
                output_dir,
                seed=seed,
                method=method_name,
                training_override=evo_cfg.get("rough_training", config.get("rough_fixmatch_training", {})),
                fixmatch_override=evo_cfg.get("rough_fixmatch", {}),
            )
            record.rough_result.update(policy_objective(record.rough_result, record.policy, objective_cfg))
            append_jsonl(record.to_dict(), log_path)
            print(
                f"[fixmatch-evolution] completed rough policy={record.policy.policy_id} "
                f"val_acc={record.rough_result.get('val_accuracy'):.4f} "
                f"objective={record.rough_result.get('objective_score'):.4f}",
                flush=True,
            )
        valid_records = [r for r in records if r.validation_ok and r.rough_result is not None]
        if gen == generations:
            break
        elites = _select_top(valid_records, elite_count, stage="rough", objective_cfg=objective_cfg)
        scores = [_record_score(r, "rough", objective_cfg) for r in valid_records if _record_score(r, "rough", objective_cfg) > float("-inf")]
        pop_mean = statistics.mean(scores) if scores else 0.0
        ranked_population = _ranked_population_payload(
            valid_records,
            stage="rough",
            objective_cfg=objective_cfg,
            max_items=int(evo_cfg.get("ranked_prompt_max_items", 8)),
        )
        next_records: list[PolicyRecord] = []
        for i in range(offspring_count):
            child_policy = None
            for attempt in range(max(1, max_unique_attempts)):
                policy_id = f"fm_gen{gen+1:02d}_{'cross' if i % 3 == 0 and len(ranked_population) >= 2 else 'mut'}_{i:03d}"
                if attempt:
                    policy_id = f"{policy_id}_try{attempt}"
                feedback = {
                    "learning_algorithm": "FixMatch",
                    "search_target": "strong augmentation branch for unlabeled consistency training",
                    "population_mean": pop_mean,
                    "generation": gen,
                    "objective": objective_cfg,
                    "baseline_reference": baseline_reference,
                    "constraints": {
                        "disable_mixup_cutmix": True,
                        "prefer_valid_tensor_safe_policy": True,
                        "avoid_overly_weak_policy": True,
                        "avoid_pseudo_label_noise": True,
                    },
                }
                crossover_enabled = bool(evo_cfg.get("crossover_enabled", True))
                if crossover_enabled and len(ranked_population) >= 2 and i % 3 == 0 and hasattr(generator, "crossover_from_ranked_population"):
                    child_policy = generator.crossover_from_ranked_population(ranked_population, policy_id=policy_id, feedback=feedback)
                elif hasattr(generator, "mutate_from_ranked_population"):
                    child_policy = generator.mutate_from_ranked_population(ranked_population, policy_id=policy_id, feedback=feedback)
                else:
                    parent = elites[i % len(elites)] if elites else valid_records[i % len(valid_records)]
                    child_policy = generator.mutate_from_feedback(parent.policy, policy_id=policy_id, feedback=feedback)
                child_policy = repair_policy_for_search_space(_zero_mixing(child_policy))
                signature = _policy_signature(child_policy)
                if signature not in seen_signatures or attempt + 1 >= max(1, max_unique_attempts):
                    seen_signatures.add(signature)
                    break
                child_policy.notes = f"{child_policy.notes} Duplicate retry requested by FixMatch novelty filter.".strip()
            next_records.append(PolicyRecord(policy=child_policy, generation=gen + 1))
        records.extend(next_records)

    valid_records = [r for r in records if r.validation_ok and r.rough_result is not None]
    finalists = _select_top(valid_records, full_top_k, stage="rough", objective_cfg=objective_cfg)
    include_policy_ids = set(evo_cfg.get("full_include_policy_ids", []))
    finalist_ids = {r.policy.policy_id for r in finalists}
    for record in valid_records:
        if record.policy.policy_id in include_policy_ids and record.policy.policy_id not in finalist_ids:
            finalists.append(record)
            finalist_ids.add(record.policy.policy_id)

    full_rows: list[dict] = []
    for record in finalists:
        print(f"[fixmatch-evolution] full policy={record.policy.policy_id}", flush=True)
        record.status = "full_evaluated"
        record.full_result = evaluate_fixmatch_policy(
            record.policy,
            config,
            output_dir,
            seed=seed,
            method=method_name,
            training_override=evo_cfg.get("full_training", config.get("full_fixmatch_training", {})),
            fixmatch_override=evo_cfg.get("full_fixmatch", {}),
        )
        record.full_result.update(policy_objective(record.full_result, record.policy, objective_cfg))
        full_rows.append(_result_row(record, record.full_result, method_name, seed))
        append_jsonl(record.to_dict(), log_path)
        print(
            f"[fixmatch-evolution] completed full policy={record.policy.policy_id} "
            f"val_acc={record.full_result.get('val_accuracy'):.4f} "
            f"test_acc={record.full_result.get('test_accuracy'):.4f} "
            f"objective={record.full_result.get('objective_score'):.4f}",
            flush=True,
        )

    pd.DataFrame(full_rows).to_csv(output_dir / "tables" / "fixmatch_results.csv", index=False)
    evolution_rows: list[dict] = []
    for record in records:
        rough = record.rough_result or {}
        full = record.full_result or {}
        evolution_rows.append({
            "method": method_name,
            "policy_id": record.policy.policy_id,
            "generation": record.generation,
            "source": record.policy.source,
            "valid": record.validation_ok,
            "rough_val_accuracy": rough.get("val_accuracy"),
            "rough_val_macro_f1": rough.get("val_macro_f1"),
            "rough_objective_score": rough.get("objective_score"),
            "rough_pseudo_mask_rate": rough.get("final_pseudo_mask_rate"),
            "rough_pseudo_confidence": rough.get("final_pseudo_confidence"),
            "rough_distortion_proxy": rough.get("distortion_proxy"),
            "rough_policy_complexity": rough.get("policy_complexity"),
            "full_val_accuracy": full.get("val_accuracy"),
            "full_val_macro_f1": full.get("val_macro_f1"),
            "full_test_accuracy": full.get("test_accuracy"),
            "full_test_macro_f1": full.get("test_macro_f1"),
            "full_objective_score": full.get("objective_score"),
            "full_pseudo_mask_rate": full.get("final_pseudo_mask_rate"),
            "full_pseudo_confidence": full.get("final_pseudo_confidence"),
            "full_distortion_proxy": full.get("distortion_proxy"),
            "full_policy_complexity": full.get("policy_complexity"),
            "runtime_sec": (rough.get("runtime_sec") or 0) + (full.get("runtime_sec") or 0),
        })
    pd.DataFrame(evolution_rows).to_csv(output_dir / "tables" / "fixmatch_evolution_records.csv", index=False)
    write_json([r.to_dict() for r in records], output_dir / "all_policy_records.json")
    best_full = _select_top(finalists, 1, stage="full", objective_cfg=objective_cfg)
    summary = {
        "method": method_name,
        "learning_algorithm": "FixMatch",
        "num_records": len(records),
        "num_valid": len(valid_records),
        "num_full_evaluated": len(finalists),
        "baseline_reference_csv": evo_cfg.get("baseline_reference_csv"),
        "best_full": best_full[0].to_dict() if best_full else None,
        "objective": objective_cfg,
    }
    write_json(summary, output_dir / "fixmatch_evolution_summary.json")
    return summary
