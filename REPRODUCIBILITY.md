# Reproducibility Guide

## Environment

The project was developed with Python, PyTorch, torchvision, NumPy, pandas, scikit-learn, matplotlib, Pillow, and PyYAML. The local environment used during the final packaging had:

- torch 2.10.0
- torchvision 0.25.0
- numpy 2.4.1
- pandas 3.0.0
- scikit-learn 1.8.0
- matplotlib 3.10.8
- Pillow 11.3.0
- PyYAML 6.0.3
- openai 2.29.0

Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data

Raw datasets are excluded. The configs download datasets into `data/raw/image_datasets/` when `download: true`.

- CIFAR-10 is used as the main low-data image recognition benchmark.
- Flowers102 is used as an additional fine-grained visual recognition setting.
- EuroSAT is used to test whether augmentation policies transfer to remote-sensing imagery.

## Recommended Execution Order

1. Run the smoke suite:

```bash
python experiments/run_suite.py --suite smoke
```

2. Validate real-data loading:

```bash
python experiments/validate_dataset.py --config configs/cifar10_pilot.yaml
```

3. Run a small CIFAR-10 pilot:

```bash
python experiments/run_suite.py --suite pilot
```

4. Re-run key final experiments if sufficient compute is available:

```bash
python experiments/run_experiment.py --config configs/cifar10_resnet18cifar_strong_openai.yaml --mode all
python experiments/run_experiment.py --config configs/cifar10_fixmatch_llm_strong_policy.yaml --mode fixmatch
python experiments/run_experiment.py --config configs/cifar10_fixmatch_in_loop_openai.yaml --mode fixmatch_evolution
python experiments/run_experiment.py --config configs/cifar10_fixmatch_repaired_llm_children.yaml --mode fixmatch
```

5. Run the final multi-seed thesis comparisons:

```bash
python experiments/run_experiment.py --config configs/cifar10_final_supervised_multiseed.yaml --mode baselines
python experiments/run_experiment.py --config configs/cifar10_final_supervised_multiseed.yaml --mode policies
python experiments/run_experiment.py --config configs/cifar10_final_fixmatch_multiseed.yaml --mode fixmatch
```

The final multi-seed runners are resumable. If a method/policy/seed already has a completed validation or test result in the target CSV, it is skipped unless `overwrite_results: true` is added to the config.

6. Rebuild combined summaries:

```bash
python experiments/collect_results.py --results-root results --output-dir results/combined
python scripts/generate_augmentation_explainer_assets.py
python scripts/summarize_final_experiments.py
```

## Compute Notes

The smoke suite is lightweight and should run on CPU or Apple Silicon MPS. The real experiments are more expensive because they train image classifiers repeatedly during policy search. A CUDA GPU or Apple Silicon MPS is recommended for the full experiments. CPU execution is possible for small smoke or pilot runs but will be slow for repeated FixMatch or multi-seed experiments.

## LLM API Configuration

OpenAI-backed policy generation is optional. Offline heuristic generation works without network access or credentials. To run OpenAI-backed configs, set `OPENAI_API_KEY` in your shell or environment. If using the default OpenAI endpoint, leave `OPENAI_BASE_URL` unset.

The repository includes `.env.example` as a template. A real `.env` file is intentionally ignored by Git:

```bash
cp .env.example .env
```

Fill in `.env`:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

Load the variables before running OpenAI-backed configs:

```bash
set -a
source .env
set +a
```

If you use an OpenAI-compatible endpoint rather than the default OpenAI endpoint, set `OPENAI_BASE_URL` to that provider's base URL. The model name is controlled by each YAML config under `llm.model`, for example:

```yaml
llm:
  provider: openai
  model: gpt-5.5
```

See `docs/LLM_CONFIGURATION.md` for the full setup guide and troubleshooting notes.

No key, `.env`, or private endpoint is included in this repository.
