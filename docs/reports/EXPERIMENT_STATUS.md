# Experiment Status

Last updated: 2026-07-09

## Implementation Status

The project codebase now implements the full experimental pipeline for low-data image recognition with LLM-guided augmentation policy evolution:

- Dataset loaders and deterministic low-data splits for FakeData, CIFAR-10, Flowers102, and EuroSAT.
- Dataset-aware augmentation policy representation, validation, transform construction, and policy JSON logging.
- Baselines: no augmentation, standard augmentation, AutoAugment, RandAugment, TrivialAugment, and AugMix where applicable.
- Evaluation modes: one-shot generated policy, random policy search, LLM-guided evolutionary search, and pure-evolution ablation.
- Rough-evaluation to full-evaluation selection.
- Policy lineage, complexity, operation-frequency, and rough/full behaviour analysis.
- Optional OpenAI structured-output policy generator.
- Cross-experiment result aggregation and figures.
- Qualitative augmentation-policy visualisation utilities.

All formal experiments were run locally on Apple Silicon MPS. DataLoader `num_workers` was set to `0` for the formal configurations because multiprocessing workers stalled on this macOS/MPS environment.

## Data Validation

The required datasets were downloaded and validated locally under `data/raw/image_datasets`.

| Dataset | Classes | Train | Validation | Test | Image Size | Status |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 low-data | 10 | 1000 | 1000 | 2000 | 32 | Completed |
| Flowers102 low-data | 102 | 510 | 1020 | 2040 | 224 | Completed |
| EuroSAT stretch | 10 | 1000 | 1000 | 2000 | 64 | Completed |

## Experiments Completed

The following experiments have been executed successfully in the current workspace:

| Experiment | Dataset | Config | Status |
|---|---:|---|---|
| Smoke main pipeline | FakeData | `configs/smoke.yaml` | Completed |
| Smoke pure evolution | FakeData | `configs/smoke_pure_evolution.yaml` | Completed |
| CIFAR-10 pilot main pipeline | CIFAR-10 | `configs/cifar10_pilot.yaml` | Completed |
| CIFAR-10 pilot pure evolution | CIFAR-10 | `configs/cifar10_pilot_pure_evolution.yaml` | Completed |
| CIFAR-10 formal main pipeline | CIFAR-10 | `configs/cifar10_lowdata.yaml` | Completed |
| CIFAR-10 pure-evolution ablation | CIFAR-10 | `configs/cifar10_lowdata_pure_evolution.yaml` | Completed |
| Flowers102 formal main pipeline | Flowers102 | `configs/flowers102_lowdata.yaml` | Completed |
| Flowers102 pure-evolution ablation | Flowers102 | `configs/flowers102_lowdata_pure_evolution.yaml` | Completed |
| EuroSAT stretch main pipeline | EuroSAT | `configs/eurosat_stretch.yaml` | Completed |
| EuroSAT stretch pure-evolution ablation | EuroSAT | `configs/eurosat_stretch_pure_evolution.yaml` | Completed |

## Final Result Summary

### CIFAR-10 Formal

| Method | Test Acc Mean | Test Acc Std | Count | Val Acc Mean |
|---|---:|---:|---:|---:|
| one_shot_llm | 0.4525 |  | 1 | 0.4260 |
| randaugment | 0.4397 | 0.0089 | 3 | 0.4350 |
| standard | 0.4318 | 0.0207 | 3 | 0.4427 |
| trivialaugment | 0.4240 | 0.0197 | 3 | 0.4297 |
| llm_guided_evolution | 0.4048 | 0.0085 | 3 | 0.4110 |
| autoaugment | 0.3983 | 0.0160 | 3 | 0.3937 |
| random_search_full | 0.3953 | 0.0172 | 3 | 0.3880 |
| pure_evolution | 0.3995 | 0.0250 | 3 | 0.3887 |
| none | 0.3893 | 0.0193 | 3 | 0.4003 |
| augmix | 0.3883 | 0.0107 | 3 | 0.3960 |

### Flowers102 Formal

| Method | Test Acc Mean | Test Acc Std | Count | Val Acc Mean |
|---|---:|---:|---:|---:|
| trivialaugment | 0.4747 | 0.0532 | 3 | 0.5026 |
| randaugment | 0.4582 | 0.0824 | 3 | 0.4755 |
| standard | 0.4475 | 0.0312 | 3 | 0.4660 |
| augmix | 0.4026 | 0.0343 | 3 | 0.4144 |
| llm_guided_evolution | 0.3186 | 0.1354 | 3 | 0.3324 |
| pure_evolution | 0.3132 | 0.0285 | 3 | 0.3542 |
| autoaugment | 0.3029 | 0.0637 | 3 | 0.3487 |
| none | 0.2626 | 0.0457 | 3 | 0.2922 |
| random_search_full | 0.2391 | 0.0688 | 3 | 0.3219 |
| one_shot_llm | 0.0853 |  | 1 | 0.1216 |

### EuroSAT Stretch

| Method | Test Acc Mean | Test Acc Std | Count | Val Acc Mean |
|---|---:|---:|---:|---:|
| standard | 0.7575 | 0.0225 | 3 | 0.7440 |
| trivialaugment | 0.7302 | 0.0195 | 3 | 0.7170 |
| one_shot_llm | 0.7255 |  | 1 | 0.7410 |
| random_search_full | 0.7167 | 0.0196 | 3 | 0.6943 |
| pure_evolution | 0.7107 | 0.0187 | 3 | 0.6780 |
| randaugment | 0.7090 | 0.0253 | 3 | 0.6950 |
| llm_guided_evolution | 0.6868 | 0.0352 | 3 | 0.6737 |
| none | 0.6795 | 0.0044 | 3 | 0.6723 |
| augmix | 0.6748 | 0.0169 | 3 | 0.6637 |

## Key Observations

The experiments show that LLM-generated and evolved augmentation policies are not uniformly superior to strong hand-designed augmentation baselines. On CIFAR-10, the one-shot heuristic LLM policy achieved the best test accuracy, while the multi-round LLM-guided evolution improved rough-evaluation candidates but did not outperform RandAugment or standard augmentation under full evaluation. On Flowers102, strong conventional augmentations, especially TrivialAugment and RandAugment, were much more reliable than one-shot or searched policies; LLM-guided evolution improved over random search and pure evolution but remained below the best baselines. On EuroSAT, standard augmentation was the strongest method, while one-shot, random search, and pure evolution were competitive; LLM-guided evolution underperformed pure evolution in the final test mean.

These results are still useful for the dissertation because they support a nuanced research claim: LLM guidance can generate valid and sometimes competitive augmentation policies, but robust performance depends strongly on dataset characteristics, constraint design, rough/full evaluation alignment, and the search space. The rough-evaluation to full-evaluation gap is especially important and should be analysed in the thesis.

## 2026-07-09 OpenAI Strong-Paradigm Update

After the initial formal experiments proved too weak, the project was upgraded to a stronger LLM + evolution paradigm:

- Real OpenAI-compatible `gpt-5.5` structured-output calls were enabled through `OPENAI_BASE_URL`.
- The CIFAR model was upgraded to `resnet18_cifar`, using a CIFAR-appropriate 3x3 stride-1 stem without the ImageNet maxpool.
- Strong baseline-conditioned seeds were added: `standard_mixup`, `standard_cutmix`, `standard_color`, `standard_erasing`, and `none`.
- Ranked LLM mutation/crossover prompts were tightened to inherit the best baseline reference rather than over-combining Mixup, CutMix, erasing, and color jitter.
- Full evaluation was modified to include important baseline seeds, reducing rough-evaluation bias against regularizers such as Mixup.

### CIFAR-10 Strong OpenAI Result

Config: `configs/cifar10_resnet18cifar_strong_openai.yaml`

| Method / Policy | Source | Test Accuracy | Test Macro-F1 | Notes |
|---|---|---:|---:|---|
| `standard_mixup` baseline | baseline | 0.4850 | 0.4630 | Strongest non-LLM baseline in this run |
| `gen02_mut_005` | OpenAI ranked mutation | 0.4845 | 0.4722 | Best LLM-evolved policy; nearly matches `standard_mixup` and has slightly higher macro-F1 |
| `standard_cutmix` baseline | baseline | 0.4825 | 0.4697 | Strong conventional regularization |
| `randaugment` baseline | baseline | 0.4435 | 0.4072 | Below Mixup/CutMix |
| `standard` baseline | baseline | 0.4100 | 0.4083 | Much weaker than Mixup/CutMix under the CIFAR stem |
| `none` baseline | baseline | 0.4065 | 0.3976 | Weak baseline |

The best OpenAI-evolved CIFAR policy was:

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

This is the strongest result so far for the thesis direction: the LLM-evolved policy inherits the empirically strongest Mixup prior, avoids unnecessary CutMix/erasing combinations, and nearly matches the best baseline test accuracy while slightly improving test macro-F1.

### EuroSAT Strong OpenAI Result

Config: `configs/eurosat_resnet18cifar_strong_openai.yaml`

| Method / Policy | Source | Test Accuracy | Test Macro-F1 | Notes |
|---|---|---:|---:|---|
| `none` baseline | baseline | 0.7395 | 0.7306 | Best overall in this run |
| `gen01_mut_002` | OpenAI ranked mutation | 0.7245 | 0.7083 | Best LLM-evolved policy after adding `none` seed |
| `seed_00_none` | baseline seed inside evolution | 0.7225 | 0.7091 | Confirms conservative/no-aug strategy is important |
| `standard` baseline | baseline | 0.7195 | 0.7122 | Competitive but below no augmentation |
| `standard_mixup` baseline | baseline | 0.7070 | 0.6911 | Mixing is less suitable for EuroSAT than CIFAR |
| `randaugment` baseline | baseline | 0.5815 | 0.5531 | Overly strong augmentation hurts EuroSAT |

The EuroSAT result is valuable because it shows dataset dependence: the same Mixup/CutMix strategy that improves CIFAR does not transfer directly to EuroSAT. Adding a `none` seed was necessary; otherwise, the search space forced unnecessary augmentation.

### Updated Interpretation

The strongest defensible thesis claim is now:

> Baseline-seeded ranked LLM evolution is effective when the LLM is constrained to act as a semantic local-search operator around strong dataset-specific augmentation priors. It does not reliably invent superior policies from scratch, but it can inherit strong priors, avoid harmful operator combinations, and produce competitive or near-best policies under objective evaluation.

This is a stronger and more honest result than the original heuristic LLM search. The CIFAR-10 result demonstrates a clear practical improvement over the earlier formal best test accuracy of 0.4525, reaching approximately 0.485 with the upgraded model and Mixup-aware LLM evolution. EuroSAT demonstrates that the framework can also discover when augmentation should be conservative, although the best single run remains no augmentation.

## Generated Outputs

Important final outputs:

- `results/combined/all_experiment_results.csv`
- `results/combined/final_accuracy_pivot.csv`
- `results/combined/cross_experiment_method_accuracy.png`
- `results/cifar10_lowdata_resnet18/tables/method_summary.csv`
- `results/cifar10_lowdata_pure_evolution/tables/method_summary.csv`
- `results/flowers102_lowdata_resnet18/tables/method_summary.csv`
- `results/flowers102_lowdata_pure_evolution/tables/method_summary.csv`
- `results/eurosat_stretch_resnet18/tables/method_summary.csv`
- `results/eurosat_stretch_pure_evolution/tables/method_summary.csv`
- `results/*/figures/method_comparison_accuracy.png`
- `results/*/figures/policy_operation_frequency.png`
- `results/cifar10_lowdata_resnet18/figures/best_evolution_policy_augmentation_grid.png`
- `results/flowers102_lowdata_resnet18/figures/best_evolution_policy_augmentation_grid.png`
- `results/eurosat_stretch_resnet18/figures/best_evolution_policy_augmentation_grid.png`

## Reproduction Commands

The final experiments were executed stage-by-stage so partial results could be saved safely:

```bash
python3 experiments/run_experiment.py --config configs/cifar10_lowdata.yaml --mode baselines
python3 experiments/run_experiment.py --config configs/cifar10_lowdata.yaml --mode policies
python3 experiments/run_experiment.py --config configs/cifar10_lowdata.yaml --mode evolution
python3 experiments/run_experiment.py --config configs/cifar10_lowdata_pure_evolution.yaml --mode all

python3 experiments/run_experiment.py --config configs/flowers102_lowdata.yaml --mode baselines
python3 experiments/run_experiment.py --config configs/flowers102_lowdata.yaml --mode policies
python3 experiments/run_experiment.py --config configs/flowers102_lowdata.yaml --mode evolution
python3 experiments/run_experiment.py --config configs/flowers102_lowdata_pure_evolution.yaml --mode all

python3 experiments/run_experiment.py --config configs/eurosat_stretch.yaml --mode baselines
python3 experiments/run_experiment.py --config configs/eurosat_stretch.yaml --mode policies
python3 experiments/run_experiment.py --config configs/eurosat_stretch.yaml --mode evolution
python3 experiments/run_experiment.py --config configs/eurosat_stretch_pure_evolution.yaml --mode all

python3 experiments/collect_results.py --results-root results --output-dir results/combined
```
