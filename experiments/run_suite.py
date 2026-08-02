from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SUITES: dict[str, list[str]] = {
    "smoke": [
        "configs/smoke.yaml",
        "configs/smoke_pure_evolution.yaml",
    ],
    "pilot": [
        "configs/cifar10_pilot.yaml",
        "configs/cifar10_pilot_pure_evolution.yaml",
    ],
    "formal": [
        "configs/cifar10_lowdata.yaml",
        "configs/cifar10_lowdata_pure_evolution.yaml",
        "configs/flowers102_lowdata.yaml",
        "configs/flowers102_lowdata_pure_evolution.yaml",
    ],
    "stretch": [
        "configs/eurosat_stretch.yaml",
        "configs/eurosat_stretch_pure_evolution.yaml",
    ],
    "openai-pilot": [
        "configs/cifar10_openai_policy_pilot.yaml",
    ],
}


def _config_to_experiment_name(config_path: Path) -> str:
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return str(data.get("experiment_name", config_path.stem))


def _run_command(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def run_suite(suite: str, mode: str, skip_existing: bool, collect: bool) -> None:
    if suite == "all":
        configs = [cfg for name in ("smoke", "pilot", "formal", "stretch") for cfg in SUITES[name]]
    elif suite in SUITES:
        configs = SUITES[suite]
    else:
        raise ValueError(f"Unknown suite: {suite}. Available: {', '.join([*SUITES, 'all'])}")

    for rel_config in configs:
        config_path = ROOT / rel_config
        if skip_existing:
            exp_name = _config_to_experiment_name(config_path)
            marker = ROOT / "results" / exp_name / "tables" / "all_methods_results.csv"
            if marker.exists():
                print(f"Skipping existing experiment: {exp_name}", flush=True)
                continue
        _run_command([sys.executable, "experiments/run_experiment.py", "--config", rel_config, "--mode", mode])

    if collect:
        _run_command([sys.executable, "experiments/collect_results.py", "--results-root", "results", "--output-dir", "results/combined"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a predefined suite of augmentation-policy experiments.")
    parser.add_argument("--suite", choices=[*SUITES.keys(), "all"], default="smoke")
    parser.add_argument("--mode", choices=["baselines", "policies", "evolution", "all"], default="all")
    parser.add_argument("--skip-existing", action="store_true", help="Skip configs whose all_methods_results.csv already exists.")
    parser.add_argument("--no-collect", action="store_true", help="Do not regenerate results/combined after the suite.")
    args = parser.parse_args()
    run_suite(args.suite, args.mode, args.skip_existing, collect=not args.no_collect)


if __name__ == "__main__":
    main()
