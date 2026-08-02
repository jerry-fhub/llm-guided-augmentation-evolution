from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FINAL_STAGES = {"baseline", "one_shot", "random_full", "evolution_full", "adaptive", "fixmatch"}
METRIC_COLS = [
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "runtime_sec",
    "final_pseudo_mask_rate",
    "final_pseudo_confidence",
    "best_pseudo_mask_rate",
    "best_pseudo_confidence",
    "best_unsupervised_loss",
]


def summarize_csv(input_csv: str | Path, output_csv: str | Path, group_col: str = "method") -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    summary = df.groupby(group_col)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join([a, b]) for a, b in summary.columns]
    summary = summary.reset_index()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return summary


def plot_accuracy(input_csv: str | Path, output_png: str | Path, metric: str = "test_accuracy") -> None:
    df = pd.read_csv(input_csv)
    if metric not in df.columns:
        metric = "val_accuracy"
    df = df[df[metric].notna()]
    if df.empty:
        return
    means = df.groupby("method")[metric].mean().sort_values(ascending=False)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    means.plot(kind="bar")
    plt.ylabel(metric)
    plt.ylim(0, max(1.0, float(means.max()) * 1.15))
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalise_metric_frame(df: pd.DataFrame, default_stage: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "stage" not in out.columns:
        out["stage"] = default_stage
    if "valid" not in out.columns:
        out["valid"] = True
    for col in ["policy_id", "source", "seed", "generation", "train_size", "unlabeled_size", "val_size", "test_size", "epochs", "device"]:
        if col not in out.columns:
            out[col] = pd.NA
    for col in METRIC_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    out["is_final_comparison"] = out["stage"].isin(FINAL_STAGES)
    columns = [
        "method",
        "stage",
        "policy_id",
        "source",
        "seed",
        "generation",
        "valid",
        "is_final_comparison",
        "train_size",
        "unlabeled_size",
        "val_size",
        "test_size",
        "epochs",
        "device",
    ]
    for col in METRIC_COLS:
        if col not in columns:
            columns.insert(-1, col)
    return out[[c for c in columns if c in out.columns]]


def _normalise_evolution_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows: list[dict] = []
    for record in df.to_dict(orient="records"):
        base = {
            "method": record.get("method", "llm_guided_evolution"),
            "policy_id": record.get("policy_id"),
            "source": record.get("source"),
            "seed": pd.NA,
            "generation": record.get("generation"),
            "valid": record.get("valid", True),
            "train_size": pd.NA,
            "val_size": pd.NA,
            "test_size": pd.NA,
            "epochs": pd.NA,
            "runtime_sec": record.get("runtime_sec"),
            "device": pd.NA,
        }
        if pd.notna(record.get("rough_val_accuracy")):
            rows.append({
                **base,
                "stage": "evolution_rough",
                "is_final_comparison": False,
                "val_accuracy": record.get("rough_val_accuracy"),
                "val_macro_f1": record.get("rough_val_macro_f1"),
                "test_accuracy": pd.NA,
                "test_macro_f1": pd.NA,
            })
        if pd.notna(record.get("full_val_accuracy")) or pd.notna(record.get("full_test_accuracy")):
            rows.append({
                **base,
                "stage": "evolution_full",
                "is_final_comparison": True,
                "val_accuracy": record.get("full_val_accuracy"),
                "val_macro_f1": pd.NA,
                "test_accuracy": record.get("full_test_accuracy"),
                "test_macro_f1": record.get("full_test_macro_f1"),
            })
    return pd.DataFrame(rows)


def aggregate_result_tables(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    frames: list[pd.DataFrame] = []
    baseline_df = _read_csv_if_exists(tables_dir / "baseline_results.csv")
    if not baseline_df.empty:
        baseline_df["stage"] = "baseline"
        frames.append(_normalise_metric_frame(baseline_df, "baseline"))
    one_shot_df = _read_csv_if_exists(tables_dir / "one_shot_results.csv")
    if not one_shot_df.empty:
        one_shot_df["stage"] = one_shot_df.get("stage", "one_shot")
        one_shot_df["stage"] = one_shot_df["stage"].fillna("one_shot")
        frames.append(_normalise_metric_frame(one_shot_df, "one_shot"))
    random_df = _read_csv_if_exists(tables_dir / "random_search_results.csv")
    if not random_df.empty:
        frames.append(_normalise_metric_frame(random_df, "random_rough"))
    fixmatch_df = _read_csv_if_exists(tables_dir / "fixmatch_results.csv")
    if not fixmatch_df.empty:
        fixmatch_df["stage"] = fixmatch_df.get("stage", "fixmatch")
        fixmatch_df["stage"] = fixmatch_df["stage"].fillna("fixmatch")
        frames.append(_normalise_metric_frame(fixmatch_df, "fixmatch"))
    evolution_df = _read_csv_if_exists(tables_dir / "evolution_records.csv")
    if not evolution_df.empty:
        frames.append(_normalise_evolution_records(evolution_df))
    adaptive_summary_path = output_dir / "adaptive_summary.json"
    if adaptive_summary_path.exists():
        try:
            adaptive = json.loads(adaptive_summary_path.read_text(encoding="utf-8"))
            frames.append(_normalise_metric_frame(pd.DataFrame([{
                "method": adaptive.get("method", "adaptive_ranked_llm_evolution"),
                "stage": "adaptive",
                "policy_id": adaptive.get("final_policy", {}).get("policy_id"),
                "source": adaptive.get("final_policy", {}).get("source"),
                "valid": True,
                "train_size": adaptive.get("train_size"),
                "val_size": adaptive.get("val_size"),
                "test_size": adaptive.get("test_size"),
                "epochs": adaptive.get("epochs"),
                "val_accuracy": adaptive.get("final_val_accuracy"),
                "val_macro_f1": adaptive.get("final_val_macro_f1"),
                "test_accuracy": adaptive.get("test_accuracy"),
                "test_macro_f1": adaptive.get("test_macro_f1"),
            }]), "adaptive"))
        except (OSError, json.JSONDecodeError):
            pass
    if frames:
        all_results = pd.concat(frames, ignore_index=True)
    else:
        all_results = pd.DataFrame()
    if all_results.empty:
        return all_results
    for col in METRIC_COLS:
        if col in all_results.columns:
            all_results[col] = pd.to_numeric(all_results[col], errors="coerce")
    tables_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(tables_dir / "all_methods_results.csv", index=False)
    final = all_results[(all_results["is_final_comparison"]) & (all_results["valid"].fillna(True))]
    metric_cols = [c for c in METRIC_COLS if c in final.columns]
    if not final.empty and metric_cols:
        summary = final.groupby("method")[metric_cols].agg(["mean", "std", "count"])
        summary.columns = ["_".join([a, b]) for a, b in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(tables_dir / "method_summary.csv", index=False)
    return all_results


def plot_method_comparison(all_results: pd.DataFrame, output_png: str | Path) -> None:
    if all_results.empty:
        return
    final = all_results[(all_results["is_final_comparison"]) & (all_results["valid"].fillna(True))].copy()
    for col in METRIC_COLS:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors="coerce")
    metric = "test_accuracy" if final["test_accuracy"].notna().any() else "val_accuracy"
    final = final[final[metric].notna()]
    if final.empty:
        return
    means = final.groupby("method")[metric].mean().sort_values(ascending=False)
    stds = final.groupby("method")[metric].std().reindex(means.index).fillna(0.0)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 4.8))
    means.plot(kind="bar", yerr=stds, capsize=3)
    plt.ylabel(metric)
    plt.xlabel("")
    plt.ylim(0, max(1.0, float(means.max()) * 1.15))
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def _policy_records_path(output_dir: Path) -> Path:
    return output_dir / "all_policy_records.json"


def _load_policy_records(output_dir: Path) -> list[dict]:
    path = _policy_records_path(output_dir)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def analyse_policy_behaviour(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    records = _load_policy_records(output_dir)
    if not records:
        return {}
    complexity_rows: list[dict] = []
    operation_rows: list[dict] = []
    lineage_rows: list[dict] = []
    for item in records:
        policy = item.get("policy", {})
        operations = [
            op
            for sub_policy in policy.get("sub_policies", [])
            for op in sub_policy
        ]
        rough = item.get("rough_result") or {}
        full = item.get("full_result") or {}
        policy_id = policy.get("policy_id")
        generation = item.get("generation")
        source = policy.get("source")
        probabilities = [float(op.get("probability", 0.0)) for op in operations]
        magnitudes = [float(op.get("magnitude", 0.0)) for op in operations]
        complexity_rows.append({
            "policy_id": policy_id,
            "generation": generation,
            "source": source,
            "status": item.get("status"),
            "validation_ok": item.get("validation_ok"),
            "num_sub_policies": len(policy.get("sub_policies", [])),
            "num_operations": len(operations),
            "mean_probability": sum(probabilities) / len(probabilities) if probabilities else 0.0,
            "mean_magnitude": sum(magnitudes) / len(magnitudes) if magnitudes else 0.0,
            "mixup_alpha": float(policy.get("mixing", {}).get("mixup_alpha", 0.0)),
            "cutmix_alpha": float(policy.get("mixing", {}).get("cutmix_alpha", 0.0)),
            "rough_val_accuracy": rough.get("val_accuracy"),
            "rough_val_macro_f1": rough.get("val_macro_f1"),
            "full_val_accuracy": full.get("val_accuracy"),
            "full_test_accuracy": full.get("test_accuracy"),
        })
        lineage_rows.append({
            "policy_id": policy_id,
            "generation": generation,
            "source": source,
            "parent_ids": ";".join(policy.get("parent_ids", [])),
            "status": item.get("status"),
            "rough_val_accuracy": rough.get("val_accuracy"),
            "full_val_accuracy": full.get("val_accuracy"),
            "full_test_accuracy": full.get("test_accuracy"),
        })
        for op in operations:
            operation_rows.append({
                "policy_id": policy_id,
                "generation": generation,
                "source": source,
                "operation": op.get("name"),
                "probability": op.get("probability"),
                "magnitude": op.get("magnitude"),
            })
    complexity = pd.DataFrame(complexity_rows)
    operations = pd.DataFrame(operation_rows)
    lineage = pd.DataFrame(lineage_rows)
    complexity.to_csv(tables_dir / "policy_complexity.csv", index=False)
    operations.to_csv(tables_dir / "policy_operations_long.csv", index=False)
    lineage.to_csv(tables_dir / "policy_lineage.csv", index=False)
    if not operations.empty:
        frequency = (
            operations.groupby(["generation", "source", "operation"])
            .agg(count=("operation", "count"), mean_probability=("probability", "mean"), mean_magnitude=("magnitude", "mean"))
            .reset_index()
        )
        frequency.to_csv(tables_dir / "policy_operation_frequency.csv", index=False)
        top_ops = Counter(operations["operation"]).most_common()
        if top_ops:
            labels = [item[0] for item in top_ops]
            counts = [item[1] for item in top_ops]
            plt.figure(figsize=(8, 4.5))
            plt.barh(labels[::-1], counts[::-1])
            plt.xlabel("count")
            plt.tight_layout()
            plt.savefig(figures_dir / "policy_operation_frequency.png", dpi=180)
            plt.close()
    correlation = {}
    paired = complexity[complexity["rough_val_accuracy"].notna() & complexity["full_val_accuracy"].notna()].copy()
    if len(paired) >= 2:
        correlation = {
            "n": int(len(paired)),
            "pearson": float(paired["rough_val_accuracy"].corr(paired["full_val_accuracy"], method="pearson")),
            "spearman": float(paired["rough_val_accuracy"].corr(paired["full_val_accuracy"], method="spearman")),
        }
        (tables_dir / "rough_full_correlation.json").write_text(json.dumps(correlation, indent=2), encoding="utf-8")
        plt.figure(figsize=(5.5, 4.5))
        plt.scatter(paired["rough_val_accuracy"], paired["full_val_accuracy"])
        for _, row in paired.iterrows():
            plt.annotate(str(row["policy_id"]), (row["rough_val_accuracy"], row["full_val_accuracy"]), fontsize=7, alpha=0.75)
        plt.xlabel("rough validation accuracy")
        plt.ylabel("full validation accuracy")
        plt.tight_layout()
        plt.savefig(figures_dir / "rough_vs_full_accuracy.png", dpi=180)
        plt.close()
    return correlation


def generate_experiment_report(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    all_results = aggregate_result_tables(output_dir)
    if not all_results.empty:
        plot_method_comparison(all_results, output_dir / "figures" / "method_comparison_accuracy.png")
    analyse_policy_behaviour(output_dir)
