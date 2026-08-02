# Experiment Results Summary

This file summarises the selected experiment outputs included in `results/`.

## Core Research Trail

The project moved through four experimental stages:

1. Establish traditional low-data image classification baselines.
2. Represent augmentation policies as constrained JSON genotypes and evaluate one-shot or random LLM-like policies.
3. Use baseline-seeded ranked LLM evolution to generate mutation and crossover children.
4. Move the search into the FixMatch strong-augmentation branch, where strong augmentation has a clearer role in semi-supervised learning.

## Supervised Experiments

| Dataset | Best local baseline | Best LLM/evolution result | Interpretation |
|---|---:|---:|---|
| CIFAR-10 early supervised run | RandAugment 43.97% | One-shot LLM 45.25%; LLM evolution 40.48% | Early LLM search was runnable but unstable. |
| Flowers102 early supervised run | TrivialAugment 47.47% | LLM evolution 31.86% | Conventional augmentation remained stronger. |
| EuroSAT early supervised run | Standard 75.75% | LLM evolution 68.68% | Strong augmentation can harm domain-specific signals. |
| CIFAR-10 strong OpenAI run | Mixup 48.50% | OpenAI-evolved `gen02_mut_005` 48.45% | Baseline-seeded ranked LLM evolution became competitive locally. |
| EuroSAT strong OpenAI run | No augmentation 73.95% | OpenAI-evolved `gen01_mut_002` 72.45% | Conservative policies were preferred in this setting. |

## FixMatch Experiments

| Method | Test accuracy | Test macro-F1 | Validation accuracy | Interpretation |
|---|---:|---:|---:|---|
| Supervised Mixup baseline | 48.50% | 46.30% | 47.40% | Best supervised local accuracy baseline. |
| Supervised LLM best | 48.45% | 47.22% | 50.20% | Best supervised LLM-evolved policy. |
| FixMatch standard | 56.75% | 55.12% | 56.60% | Unlabelled data improves absolute performance. |
| FixMatch reused LLM policy | 59.00% | 58.38% | 61.40% | Supervised LLM policy transfers usefully to FixMatch. |
| FixMatch repaired LLM child `mut_001` | 59.05% | 58.05% | 62.70% | In-loop LLM evolution is functional with a small gain over reuse. |
| FixMatch TrivialAugment | 61.50% | 61.08% | 63.00% | Strong canonical augmentation baseline. |
| FixMatch RandAugment | 62.05% | 61.05% | 63.90% | Current best local result. |

## Visual Results

![Method dashboard](results/combined/completed_experiment_visuals/supervisor_result_dashboard.png)

![Method pipeline](results/combined/completed_experiment_visuals/supervisor_method_pipeline.png)

![Augmentation method examples](results/combined/completed_experiment_visuals/augmentation_method_examples_cifar10.png)

![Mixup and CutMix demo](results/combined/completed_experiment_visuals/mixup_cutmix_visual_demo_cifar10.png)

![SOTA context comparison](results/combined/completed_experiment_visuals/sota_context_comparison.png)

## Included Result Directories

The included `results/` directory keeps selected successful experiment artefacts: tables, JSON summaries, policies, lineages, logs, and figures. It excludes model checkpoints and raw datasets.

Important files:

- `results/combined/all_method_summaries.csv`
- `results/combined/all_experiment_results.csv`
- `results/cifar10_resnet18cifar_strong_openai/policies/gen02_mut_005.json`
- `results/cifar10_fixmatch_in_loop_openai/fixmatch_evolution_summary.json`
- `results/cifar10_fixmatch_repaired_llm_children/policies/fm_gen01_mut_001_repaired.json`
- `results/cifar10_fixmatch_llm_strong_policy/tables/fixmatch_results.csv`
