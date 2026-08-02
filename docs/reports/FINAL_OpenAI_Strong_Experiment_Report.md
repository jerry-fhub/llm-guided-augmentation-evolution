# Final OpenAI Strong Experiment Report

Last updated: 2026-07-09

## Purpose

The original heuristic LLM-evolution experiments were not strong enough: evolved policies did not consistently outperform conventional augmentation baselines, and the formal runs did not use a real LLM. The project was therefore upgraded and re-run with a true OpenAI-compatible `gpt-5.5` policy generator, baseline-seeded ranked evolution, stronger augmentation seeds, and a CIFAR-appropriate ResNet18 model.

## Main Implementation Changes

The most important technical changes are:

- Added real `OPENAI_BASE_URL` support in `OpenAIPolicyGenerator`.
- Added ranked-population OpenAI mutation and crossover.
- Added baseline-conditioned seeds: `none`, `standard_color`, `standard_erasing`, `standard_mixup`, and `standard_cutmix`.
- Added full-evaluation inclusion for important seeds, avoiding rough-evaluation bias against Mixup/CutMix.
- Added `resnet18_cifar`, replacing the ImageNet-style 7x7 stride-2 stem with a CIFAR-suitable 3x3 stride-1 stem and no maxpool.
- Tightened prompts so the LLM first identifies the strongest baseline reference and avoids harmful over-combination.

## CIFAR-10 Result

Config: `configs/cifar10_resnet18cifar_strong_openai.yaml`

Dataset setting: CIFAR-10, 100 training images per class, 100 validation images per class, 2000-image test subset.

Model: `resnet18_cifar`, 15 epochs, AdamW.

### Best Results

| Method / Policy | Source | Test Accuracy | Test Macro-F1 |
|---|---|---:|---:|
| `standard_mixup` | baseline | 0.4850 | 0.4630 |
| `gen02_mut_005` | OpenAI ranked mutation | 0.4845 | 0.4722 |
| `standard_cutmix` | baseline | 0.4825 | 0.4697 |
| `randaugment` | baseline | 0.4435 | 0.4072 |
| `standard` | baseline | 0.4100 | 0.4083 |
| `none` | baseline | 0.4065 | 0.3976 |

The best OpenAI-evolved policy was:

```json
{
  "sub_policies": [
    [
      {"name": "RandomCrop", "probability": 1.0, "magnitude": 0.32},
      {"name": "HorizontalFlip", "probability": 0.5, "magnitude": 0.0},
      {"name": "ColorJitter", "probability": 0.16, "magnitude": 0.12}
    ]
  ],
  "mixing": {"mixup_alpha": 0.2, "cutmix_alpha": 0.0}
}
```

Interpretation: the LLM did not invent a completely new augmentation family. Instead, it successfully acted as a semantic local-search operator around the strongest baseline prior, preserving Mixup and adding only light color jitter. This is a defensible positive result because the evolved policy nearly matches the best baseline accuracy and slightly improves macro-F1.

## EuroSAT Result

Config: `configs/eurosat_resnet18cifar_strong_openai.yaml`

Dataset setting: EuroSAT, 100 training images per class, 100 validation images per class, 2000-image test subset.

Model: `resnet18_cifar`, 12 epochs, AdamW.

### Best Results

| Method / Policy | Source | Test Accuracy | Test Macro-F1 |
|---|---|---:|---:|
| `none` | baseline | 0.7395 | 0.7306 |
| `gen01_mut_002` | OpenAI ranked mutation | 0.7245 | 0.7083 |
| `seed_00_none` | baseline seed inside evolution | 0.7225 | 0.7091 |
| `standard` | baseline | 0.7195 | 0.7122 |
| `standard_mixup` | baseline | 0.7070 | 0.6911 |
| `randaugment` | baseline | 0.5815 | 0.5531 |

Interpretation: EuroSAT behaves differently from CIFAR-10. Strong augmentation is often harmful; no augmentation or very mild augmentation is best. Adding the `none` seed was therefore essential. The LLM-evolved policy became competitive, but it did not beat the no-augmentation baseline.

## Final Research Claim

The strongest honest claim is:

> Baseline-seeded ranked LLM evolution is effective as a constrained semantic local-search mechanism. It is not reliable as a from-scratch augmentation inventor. On CIFAR-10, it learns to preserve the strongest Mixup prior and reaches near-best accuracy with improved macro-F1. On EuroSAT, it reveals that conservative or no augmentation is preferable, showing the importance of dataset-aware constraints and allowing no-augmentation candidates.

## What Should Be Used in the Dissertation

Use CIFAR-10 as the main positive result because the upgraded method achieves a real improvement over the earlier project results and produces an interpretable LLM-evolved policy.

Use EuroSAT as a dataset-sensitivity case study: it demonstrates that LLM/evolution must be constrained by baseline evidence and that aggressive augmentation is not universally beneficial.

Treat adaptive OpenAI policy updates as an ablation/negative result, not the main method. The small adaptive pilot was runnable but unstable.

## Key Output Files

- `results/cifar10_resnet18cifar_strong_openai/tables/baseline_results.csv`
- `results/cifar10_resnet18cifar_strong_openai/tables/evolution_records.csv`
- `results/eurosat_resnet18cifar_strong_openai/tables/baseline_results.csv`
- `results/eurosat_resnet18cifar_strong_openai/tables/evolution_records.csv`
- `results/combined/final_accuracy_pivot.csv`
- `results/combined/all_experiment_results.csv`
- `EXPERIMENT_STATUS.md`
- `LLM_Evolution_Core_Paradigm_Upgrade.md`
