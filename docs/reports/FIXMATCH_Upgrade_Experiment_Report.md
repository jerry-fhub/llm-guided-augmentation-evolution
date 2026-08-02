# FixMatch Upgrade Experiment Report

Last updated: 2026-07-11

## Purpose

The previous supervised low-data experiments showed that baseline-seeded ranked LLM evolution can generate competitive augmentation policies, but the best CIFAR-10 result only nearly matched Mixup/CutMix rather than clearly exceeding strong baselines. This upgrade tests a more research-motivated direction: using the LLM-evolved policy as the strong augmentation branch inside FixMatch-style semi-supervised learning.

This is not only a way to improve the headline accuracy. It is methodologically meaningful because FixMatch relies directly on the consistency between weakly augmented unlabeled images and strongly augmented unlabeled images. Therefore, the choice of strong augmentation policy is part of the core learning mechanism, not a peripheral training trick. This makes FixMatch a better setting for studying LLM-guided augmentation policy design than purely supervised training.

Relevant methodological background includes FixMatch, UDA, RandAugment, and TrivialAugment:

| Method | Relevance |
|---|---|
| [FixMatch](https://arxiv.org/abs/2001.07685) | Combines pseudo-labeling with weak/strong augmentation consistency. |
| [UDA](https://arxiv.org/abs/1904.12848) | Shows strong data augmentation is central to consistency training. |
| [RandAugment](https://arxiv.org/abs/1909.13719) | Provides a strong augmentation baseline with a reduced search space. |
| [TrivialAugment](https://arxiv.org/abs/2103.10158) | Provides a tuning-light strong augmentation baseline. |

## Implementation

The codebase was extended with a new FixMatch pipeline rather than replacing the existing supervised pipeline. The new components include semi-supervised dataloaders, weak/strong dual-view unlabeled datasets, a FixMatch training loop, pseudo-label confidence masking, policy-file evaluation, and integration with the existing `run_experiment.py` entry point.

The main new files and configs are:

| File | Purpose |
|---|---|
| `src/image_aug_evolution/models/fixmatch.py` | FixMatch training loop and evaluation. |
| `src/image_aug_evolution/evaluation/fixmatch_runner.py` | Runs FixMatch policies and writes result tables. |
| `configs/cifar10_fixmatch_llm_strong_policy.yaml` | Main CIFAR-10 FixMatch pilot. |
| `configs/cifar10_fixmatch_geo_candidate.yaml` | Exploratory geometric refinement candidate. |
| `fixmatch_geo_candidate_policy.json` | Manual FixMatch-aware refinement candidate. |

The main CIFAR-10 setting used 100 labeled training images per class, 100 validation images per class, 10,000 unlabeled training images, and a 2,000-image test subset. The model was `resnet18_cifar`, trained for 15 epochs with threshold `0.8` and unlabeled loss weight `1.0`. The threshold was intentionally lower than the canonical `0.95` because this local pilot uses a short training budget; with very short runs, a high threshold often produces near-zero pseudo-label usage.

## Results

| Method | Test Accuracy | Test Macro-F1 | Val Accuracy | Notes |
|---|---:|---:|---:|---|
| Supervised `standard_mixup` | 0.4850 | 0.4630 | 0.4740 | Best supervised accuracy baseline from the previous strong run. |
| Supervised LLM best `gen02_mut_005` | 0.4845 | 0.4722 | 0.5020 | Best supervised LLM-evolved policy. |
| FixMatch `standard` | 0.5675 | 0.5512 | 0.5660 | Clear gain from unlabeled data, but weaker than stronger augmentations. |
| FixMatch LLM geo-refined candidate | 0.5640 | 0.5532 | 0.5940 | Manual geometry refinement did not improve over the original LLM policy. |
| FixMatch LLM-evolved policy | 0.5900 | 0.5838 | 0.6140 | Reuses the OpenAI-evolved supervised policy as the FixMatch strong branch. |
| FixMatch `trivialaugment` | 0.6150 | 0.6108 | 0.6300 | Strong canonical SSL augmentation baseline. |
| FixMatch `randaugment` | 0.6205 | 0.6105 | 0.6390 | Best result in this pilot. |

![FixMatch upgrade summary](../../results/combined/completed_experiment_visuals/fixmatch_upgrade_summary.png)

The upgrade is successful in one important sense: FixMatch substantially improves the absolute CIFAR-10 result. The best previous supervised result was approximately 0.485 test accuracy, while FixMatch with RandAugment reaches 0.6205. The original LLM-evolved policy also improves from its supervised 0.4845 to 0.5900 when used as a FixMatch strong augmentation policy. This confirms that the semi-supervised direction is not merely cosmetic; it changes the learning regime in a way that strongly benefits low-data image recognition.

However, the LLM-evolved policy does not yet outperform the best canonical FixMatch strong augmentation baseline. RandAugment remains stronger than the reused LLM policy by 3.05 percentage points in test accuracy. TrivialAugment also remains ahead by 2.50 percentage points. The manual geometry-refined candidate underperforms the original LLM policy, suggesting that simple hand-designed strengthening is not enough.

## Analysis

The results support a more precise research claim. The problem should not be framed as “LLM-generated policies already beat strong augmentation baselines.” The better claim is that FixMatch provides a more appropriate experimental setting for studying LLM-guided augmentation search, because strong augmentation directly controls the pseudo-label consistency objective. In this setting, the current LLM-evolved policy is competitive and useful, but not yet optimal.

The result also explains why directly transferring the best supervised policy is insufficient. The supervised LLM policy was deliberately conservative: crop, flip, light color jitter, and Mixup prior. In FixMatch, Mixup/CutMix are disabled for the strong branch, and the strong view needs to be sufficiently diverse to regularize pseudo-label learning. As a result, RandAugment and TrivialAugment are stronger because they provide broader transformation diversity. The next LLM-evolution stage should therefore use FixMatch validation feedback directly, rather than reusing supervised-search feedback.

The failed geometry-refined candidate is informative. It added mild rotation and affine transformations to the LLM policy, but this did not improve performance. This suggests that the useful strong branch is not obtained by simply adding more geometric distortion. Future search should learn which transformation families, probabilities, and magnitudes improve pseudo-label consistency under the FixMatch objective.

## Conclusion

This upgrade is worth continuing. It produces a meaningful performance lift and, more importantly, places LLM-guided augmentation search inside a learning algorithm where augmentation policy quality is central to the method. The strongest next step is not more manual policy editing; it is FixMatch-in-the-loop LLM evolution, where the LLM receives ranked FixMatch results and mutates/crosses strong augmentation policies based on pseudo-label consistency performance.

The thesis claim should be updated as follows:

> Preliminary FixMatch experiments show that semi-supervised consistency learning is a more suitable setting for LLM-guided augmentation policy optimization than purely supervised low-data training. Although the reused supervised LLM-evolved policy improves substantially under FixMatch, it does not yet outperform RandAugment. This motivates a stronger future method: FixMatch-aware ranked LLM evolution, where strong augmentation policies are evolved directly using semi-supervised validation feedback.

## 2026-07-11 In-loop Evolution Update

A compact FixMatch-in-the-loop OpenAI evolution experiment was implemented and run. The pipeline evaluates initial policies under rough FixMatch training, sends ranked FixMatch feedback to the LLM, generates mutation children, applies a local repair layer when generated policies exceed search-space constraints, and then fully evaluates repaired children.

The best repaired LLM child reached 0.5905 test accuracy and 0.5805 macro-F1. This is essentially tied with the reused supervised LLM policy under FixMatch (0.5900 accuracy), but it remains below TrivialAugment (0.6150) and RandAugment (0.6205). Therefore, the in-loop method is functional and research-relevant, but not yet stronger than canonical FixMatch augmentation baselines.

Detailed report: `FIXMATCH_In_Loop_LLM_Evolution_Report.md`.
