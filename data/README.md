# Data Directory

Raw datasets are intentionally excluded from the GitHub-ready folder.

When an experiment config has `download: true`, torchvision or the configured mirror downloads data into:

```text
data/raw/image_datasets/
```

Datasets used by the project:

- CIFAR-10: main low-data image recognition benchmark.
- Flowers102: fine-grained flower classification benchmark.
- EuroSAT: remote-sensing image classification benchmark.

Generated processed data, raw archives, and downloaded image files should not be committed.
