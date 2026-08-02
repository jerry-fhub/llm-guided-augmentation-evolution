from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from image_aug_evolution.analysis.reporting import generate_experiment_report, plot_accuracy, summarize_csv
from image_aug_evolution.evaluation.baseline_runner import run_baselines
from image_aug_evolution.evaluation.fixmatch_runner import run_fixmatch
from image_aug_evolution.evaluation.policy_runner import run_one_shot, run_random_search
from image_aug_evolution.search.adaptive import run_adaptive_policy_updates
from image_aug_evolution.search.evolution import run_evolutionary_search
from image_aug_evolution.search.fixmatch_evolution import run_fixmatch_evolution
from image_aug_evolution.utils.config import ensure_dir, load_config
from image_aug_evolution.utils.io import write_json
from image_aug_evolution.utils.seeding import seed_everything


def run(config_path: str | Path, mode: str) -> None:
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 0)))
    exp_name = config.get("experiment_name", Path(config_path).stem)
    output_dir = ensure_dir(Path(config.get("output_root", "results")) / exp_name)
    for subdir in ("logs", "policies", "tables", "figures", "models"):
        ensure_dir(output_dir / subdir)
    write_json(config, output_dir / "config_resolved.json")

    if mode in {"baselines", "all"}:
        baseline_records = run_baselines(config, output_dir)
        baseline_csv = output_dir / "tables" / "baseline_results.csv"
        if baseline_records:
            summarize_csv(baseline_csv, output_dir / "tables" / "baseline_summary.csv")
            plot_accuracy(baseline_csv, output_dir / "figures" / "baseline_accuracy.png")

    if mode in {"policies", "all"}:
        run_one_shot(config, output_dir)
        run_random_search(config, output_dir)

    if mode in {"evolution", "all"}:
        summary = run_evolutionary_search(config, output_dir)
        write_json(summary, output_dir / "evolution_summary.json")
        # A compact CSV for thesis tables.
        rows = []
        records_path = output_dir / "all_policy_records.json"
        if records_path.exists():
            import json
            records = json.loads(records_path.read_text(encoding="utf-8"))
            for r in records:
                rough = r.get("rough_result") or {}
                full = r.get("full_result") or {}
                rows.append({
                    "method": summary.get("method", "llm_guided_evolution"),
                    "policy_id": r["policy"]["policy_id"],
                    "generation": r["generation"],
                    "source": r["policy"].get("source", ""),
                    "valid": r.get("validation_ok", False),
                    "rough_val_accuracy": rough.get("val_accuracy"),
                    "rough_val_macro_f1": rough.get("val_macro_f1"),
                    "rough_objective_score": rough.get("objective_score"),
                    "rough_distortion_proxy": rough.get("distortion_proxy"),
                    "rough_policy_complexity": rough.get("policy_complexity"),
                    "full_val_accuracy": full.get("val_accuracy"),
                    "full_test_accuracy": full.get("test_accuracy"),
                    "full_test_macro_f1": full.get("test_macro_f1"),
                    "full_objective_score": full.get("objective_score"),
                    "full_distortion_proxy": full.get("distortion_proxy"),
                    "full_policy_complexity": full.get("policy_complexity"),
                    "runtime_sec": (rough.get("runtime_sec") or 0) + (full.get("runtime_sec") or 0),
                })
            pd.DataFrame(rows).to_csv(output_dir / "tables" / "evolution_records.csv", index=False)

    if mode == "adaptive" or (mode == "all" and config.get("adaptive", {}).get("enabled", False)):
        summary = run_adaptive_policy_updates(config, output_dir)
        write_json(summary, output_dir / "adaptive_summary.json")

    if mode == "fixmatch" or (mode == "all" and config.get("fixmatch", {}).get("enabled", False)):
        fixmatch_records = run_fixmatch(config, output_dir)
        fixmatch_csv = output_dir / "tables" / "fixmatch_results.csv"
        if fixmatch_records:
            summarize_csv(fixmatch_csv, output_dir / "tables" / "fixmatch_summary.csv")
            plot_accuracy(fixmatch_csv, output_dir / "figures" / "fixmatch_accuracy.png")

    if mode == "fixmatch_evolution" or (mode == "all" and config.get("fixmatch_evolution", {}).get("enabled", False)):
        summary = run_fixmatch_evolution(config, output_dir)
        write_json(summary, output_dir / "fixmatch_evolution_summary.json")

    generate_experiment_report(output_dir)
    print(f"Experiment complete. Results written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run augmentation policy experiments.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--mode",
        choices=["baselines", "policies", "evolution", "adaptive", "fixmatch", "fixmatch_evolution", "all"],
        default="all",
    )
    args = parser.parse_args()
    run(args.config, args.mode)


if __name__ == "__main__":
    main()
