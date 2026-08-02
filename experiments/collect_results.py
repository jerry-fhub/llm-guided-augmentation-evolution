from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _experiment_dirs(results_root: Path) -> list[Path]:
    return sorted(
        p
        for p in results_root.iterdir()
        if p.is_dir() and (p / "tables" / "all_methods_results.csv").exists()
    )


def collect(results_root: str | Path, output_dir: str | Path) -> None:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for exp_dir in _experiment_dirs(results_root):
        all_path = exp_dir / "tables" / "all_methods_results.csv"
        summary_path = exp_dir / "tables" / "method_summary.csv"
        config_path = exp_dir / "config_resolved.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        dataset_name = config.get("dataset", {}).get("name", "")
        model_name = config.get("training", {}).get("model_name", "")
        all_df = pd.read_csv(all_path)
        all_df.insert(0, "experiment_name", exp_dir.name)
        all_df.insert(1, "dataset", dataset_name)
        all_df.insert(2, "model", model_name)
        frames.append(all_df)
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_df.insert(0, "experiment_name", exp_dir.name)
            summary_df.insert(1, "dataset", dataset_name)
            summary_df.insert(2, "model", model_name)
            summaries.append(summary_df)
    if not frames:
        raise FileNotFoundError(f"No experiment results found under {results_root}")
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(output_dir / "all_experiment_results.csv", index=False)
    if summaries:
        all_summaries = pd.concat(summaries, ignore_index=True)
        all_summaries.to_csv(output_dir / "all_method_summaries.csv", index=False)
    final = all_results[(all_results["is_final_comparison"]) & (all_results["valid"].fillna(True))].copy()
    metric = "test_accuracy" if final["test_accuracy"].notna().any() else "val_accuracy"
    final[metric] = pd.to_numeric(final[metric], errors="coerce")
    final = final[final[metric].notna()]
    if not final.empty:
        pivot = final.pivot_table(index=["experiment_name", "dataset"], columns="method", values=metric, aggfunc="mean")
        pivot.to_csv(output_dir / "final_accuracy_pivot.csv")
        plot_df = final.groupby(["dataset", "method"])[metric].mean().reset_index()
        labels = [f"{row.dataset}:{row.method}" for row in plot_df.itertuples()]
        plt.figure(figsize=(max(8, len(labels) * 0.55), 4.8))
        plt.bar(labels, plot_df[metric])
        plt.ylabel(metric)
        plt.xticks(rotation=55, ha="right")
        plt.ylim(0, max(1.0, float(plot_df[metric].max()) * 1.15))
        plt.tight_layout()
        plt.savefig(output_dir / "cross_experiment_method_accuracy.png", dpi=180)
        plt.close()
    print(f"Collected {len(frames)} experiments into: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect result tables across experiment folders.")
    parser.add_argument("--results-root", default="results", help="Root directory containing experiment folders.")
    parser.add_argument("--output-dir", default="results/combined", help="Directory for combined tables and figures.")
    args = parser.parse_args()
    collect(args.results_root, args.output_dir)


if __name__ == "__main__":
    main()
