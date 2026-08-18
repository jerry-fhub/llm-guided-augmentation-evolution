from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUPERVISED_DIR = ROOT / "results" / "cifar10_final_supervised_multiseed"
FIXMATCH_DIR = ROOT / "results" / "cifar10_final_fixmatch_multiseed"
OUTPUT_DIR = ROOT / "results" / "combined" / "final_multiseed"
REPORT_PATH = ROOT / "FINAL_Multiseed_Experiment_Report.md"


DISPLAY_NAMES = {
    "none": "None",
    "standard": "Standard",
    "standard_mixup": "Mixup",
    "standard_cutmix": "CutMix",
    "randaugment": "RandAugment",
    "trivialaugment": "TrivialAugment",
    "supervised_llm_evolved_best": "LLM evolved",
    "fixmatch_standard": "FixMatch Standard",
    "fixmatch_randaugment": "FixMatch RandAugment",
    "fixmatch_trivialaugment": "FixMatch TrivialAugment",
    "fixmatch_llm_evolved_cifar_best": "FixMatch LLM reused",
    "fixmatch_llm_repaired_mut_001": "FixMatch LLM repaired",
}


SUPERVISED_ORDER = [
    "none",
    "standard",
    "standard_mixup",
    "standard_cutmix",
    "randaugment",
    "trivialaugment",
    "supervised_llm_evolved_best",
]


FIXMATCH_ORDER = [
    "fixmatch_standard",
    "fixmatch_randaugment",
    "fixmatch_trivialaugment",
    "fixmatch_llm_evolved_cifar_best",
    "fixmatch_llm_repaired_mut_001",
]


def _load_summary(path: Path, order: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    df = df.sort_values("method").reset_index(drop=True)
    df["display_name"] = df["method"].astype(str).map(DISPLAY_NAMES).fillna(df["method"].astype(str))
    return df


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pct_std(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}"


def _format_summary_table(df: pd.DataFrame, include_mask: bool = False) -> str:
    headers = ["Method", "Test accuracy", "Test macro-F1", "Validation accuracy", "Seeds"]
    if include_mask:
        headers.append("Final pseudo-label mask")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in df.to_dict(orient="records"):
        values = [
            row["display_name"],
            f"{_pct(row['test_accuracy_mean'])} +/- {_pct_std(row['test_accuracy_std'])}",
            f"{_pct(row['test_macro_f1_mean'])} +/- {_pct_std(row['test_macro_f1_std'])}",
            f"{_pct(row['val_accuracy_mean'])} +/- {_pct_std(row['val_accuracy_std'])}",
            str(int(row["test_accuracy_count"])),
        ]
        if include_mask:
            values.append(f"{row['final_pseudo_mask_rate_mean']:.3f} +/- {row['final_pseudo_mask_rate_std']:.3f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _plot_summary(df: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y = df["test_accuracy_mean"] * 100
    yerr = df["test_accuracy_std"].fillna(0) * 100
    colors = ["#7f8c8d" if "LLM" not in name else "#c0392b" for name in df["display_name"]]
    ax.bar(df["display_name"], y, yerr=yerr, capsize=4, color=colors, alpha=0.88)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(70, float((y + yerr).max()) + 5))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=28)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_fixmatch_mask(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
    x = range(len(df))
    acc = df["test_accuracy_mean"] * 100
    mask = df["final_pseudo_mask_rate_mean"]
    ax1.bar(x, acc, color="#2c7fb8", alpha=0.78, label="Test accuracy")
    ax1.set_ylabel("Test accuracy (%)", color="#2c7fb8")
    ax1.tick_params(axis="y", labelcolor="#2c7fb8")
    ax1.set_ylim(0, max(70, float(acc.max()) + 5))
    ax2 = ax1.twinx()
    ax2.plot(x, mask, color="#d95f0e", marker="o", linewidth=2.4, label="Pseudo-label mask")
    ax2.set_ylabel("Final pseudo-label mask rate", color="#d95f0e")
    ax2.tick_params(axis="y", labelcolor="#d95f0e")
    ax2.set_ylim(0, 1)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["display_name"], rotation=28, ha="right")
    ax1.set_title("FixMatch Accuracy and Pseudo-Label Activation")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_combined_tables(supervised: pd.DataFrame, fixmatch: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    supervised.to_csv(OUTPUT_DIR / "supervised_multiseed_summary.csv", index=False)
    fixmatch.to_csv(OUTPUT_DIR / "fixmatch_multiseed_summary.csv", index=False)
    combined = pd.concat(
        [
            supervised.assign(setting="supervised"),
            fixmatch.assign(setting="fixmatch"),
        ],
        ignore_index=True,
    )
    combined.to_csv(OUTPUT_DIR / "combined_multiseed_summary.csv", index=False)


def _write_report(supervised: pd.DataFrame, fixmatch: pd.DataFrame) -> None:
    supervised_plot = "results/combined/final_multiseed/supervised_multiseed_accuracy.png"
    fixmatch_plot = "results/combined/final_multiseed/fixmatch_multiseed_accuracy.png"
    mask_plot = "results/combined/final_multiseed/fixmatch_mask_vs_accuracy.png"
    best_sup = supervised.loc[supervised["test_accuracy_mean"].idxmax()]
    best_fm = fixmatch.loc[fixmatch["test_accuracy_mean"].idxmax()]
    llm_sup = supervised[supervised["method"].astype(str).eq("supervised_llm_evolved_best")].iloc[0]
    llm_fm = fixmatch[fixmatch["method"].astype(str).eq("fixmatch_llm_evolved_cifar_best")].iloc[0]
    repaired = fixmatch[fixmatch["method"].astype(str).eq("fixmatch_llm_repaired_mut_001")].iloc[0]
    text = f"""# Final Multi-Seed Experiment Report

Updated: 2026-08-19

## Purpose

This report consolidates the final multi-seed experiments for the dissertation project. Earlier results showed that LLM-guided augmentation evolution could generate valid and interpretable policies, but several key comparisons were based on single-seed runs. The new experiments re-evaluate the most important supervised and FixMatch methods on CIFAR-10 using three matched seeds.

## Supervised Low-Data CIFAR-10

Setting: CIFAR-10 with 100 labelled training images per class, 100 validation images per class, 2,000 test images, `resnet18_cifar`, 15 epochs, and seeds 47, 48, and 49.

{_format_summary_table(supervised)}

![Supervised multi-seed accuracy]({supervised_plot})

The best supervised mean accuracy is **{best_sup['display_name']}** at **{_pct(best_sup['test_accuracy_mean'])} +/- {_pct_std(best_sup['test_accuracy_std'])}**. The LLM-evolved policy reaches **{_pct(llm_sup['test_accuracy_mean'])} +/- {_pct_std(llm_sup['test_accuracy_std'])}**, close to Standard, Mixup, and RandAugment, but below CutMix in mean accuracy. This weakens the earlier single-seed interpretation that the LLM policy nearly matched the best supervised baseline at 48.45-48.50%. The more reliable conclusion is that the supervised LLM policy is competitive with several local baselines, while not consistently outperforming the strongest one.

## FixMatch CIFAR-10

Setting: CIFAR-10 with 100 labelled training images per class, 10,000 unlabelled images, 100 validation images per class, 2,000 test images, `resnet18_cifar`, 15 epochs, FixMatch threshold 0.8, and seeds 61, 62, and 63.

{_format_summary_table(fixmatch, include_mask=True)}

![FixMatch multi-seed accuracy]({fixmatch_plot})

![FixMatch pseudo-label mask analysis]({mask_plot})

The best FixMatch mean accuracy is **{best_fm['display_name']}** at **{_pct(best_fm['test_accuracy_mean'])} +/- {_pct_std(best_fm['test_accuracy_std'])}**. The reused supervised LLM policy reaches **{_pct(llm_fm['test_accuracy_mean'])} +/- {_pct_std(llm_fm['test_accuracy_std'])}**, which is clearly above the collapsed Standard mean and below RandAugment. The repaired in-loop LLM child reaches **{_pct(repaired['test_accuracy_mean'])} +/- {_pct_std(repaired['test_accuracy_std'])}** because two of its three runs collapsed after early pseudo-label activation. This revises the earlier single-seed result: the repaired child is not a stable improvement over the reused LLM policy.

## Main Conclusions

The project has now completed a stronger final evidence package. The supervised LLM-evolved policy is competitive but not superior to the best supervised baseline under multi-seed evaluation. In FixMatch, RandAugment remains the strongest and most stable local baseline. The reused LLM policy is a valid and reasonably stable strong-augmentation branch, but the repaired in-loop LLM child is unstable under the current short training budget and constraint-repair design.

The thesis should therefore make a framework-oriented contribution rather than a state-of-the-art performance claim. The defensible claim is that baseline-seeded, ranked, constraint-checked LLM evolution can generate interpretable and runnable augmentation policies, and that empirical validation is essential because single-seed improvements can disappear under multi-seed testing.

## Remaining Work

The code and experiments are now sufficient for a solid dissertation result section. The remaining work is mostly analysis and writing:

- Add a compact ablation table using existing early-stage evidence and, if time permits, run one small no-baseline-seeding or no-ranked-feedback OpenAI ablation.
- Add class-wise analysis for the final supervised LLM policy and FixMatch RandAugment/LLM reused methods.
- Update the thesis Results, Discussion, and Conclusion chapters to reflect the multi-seed findings.
- Avoid claiming that the LLM-generated policy exceeds RandAugment; instead emphasise interpretability, constraint handling, ranked feedback, and the importance of robust evaluation.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    supervised = _load_summary(SUPERVISED_DIR / "tables" / "method_summary.csv", SUPERVISED_ORDER)
    fixmatch = _load_summary(FIXMATCH_DIR / "tables" / "method_summary.csv", FIXMATCH_ORDER)
    _write_combined_tables(supervised, fixmatch)
    _plot_summary(supervised, OUTPUT_DIR / "supervised_multiseed_accuracy.png", "Supervised Low-Data CIFAR-10 Multi-Seed Accuracy")
    _plot_summary(fixmatch, OUTPUT_DIR / "fixmatch_multiseed_accuracy.png", "FixMatch CIFAR-10 Multi-Seed Accuracy")
    _plot_fixmatch_mask(fixmatch, OUTPUT_DIR / "fixmatch_mask_vs_accuracy.png")
    _write_report(supervised, fixmatch)
    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote summaries and figures: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
