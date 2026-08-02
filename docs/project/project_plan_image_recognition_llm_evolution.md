# Project Plan

## LLM-Guided Evolutionary Search for Data Augmentation Policy Optimization in Low-Data Image Recognition

### Problem, Question and Aim

Image recognition models often require large amounts of labelled training data, but many practical classification tasks have limited examples, fine-grained class boundaries, or domain-specific visual characteristics. Data augmentation can improve generalisation, but policies are often selected manually or from generic recipes. A policy that works for natural images may not suit fine-grained flower recognition or remote-sensing images, and overly strong transformations may damage class-discriminative information.

This project investigates whether large language models (LLMs), combined with evolutionary search, can discover effective augmentation policies for low-data image recognition. The hypothesis is that an LLM can use dataset descriptions, validation feedback and previous policy performance to propose better candidate policies than random search or one-shot prompting, while a classifier-based evaluator provides objective selection pressure.

The main research question is:

**Can LLM-guided evolutionary search improve low-data image classification by automatically discovering dataset-specific augmentation policies under a limited training budget?**

### Existing Work and Context

Automated data augmentation is an established area in computer vision. AutoAugment introduced the idea of learning augmentation policies from data using validation performance as the search objective. RandAugment simplified this process by reducing the search space, making automated augmentation more practical. TrivialAugment further showed that simple dataset-independent augmentation can be a strong baseline, while AugMix demonstrated that augmentation can improve robustness and uncertainty under distribution shift.

Recent work has also explored LLMs in optimisation and AutoML. LLMs can generate structured candidates, explain failures and suggest revisions, while evolutionary algorithms can maintain diversity and evaluate candidates objectively. This project keeps the scope manageable: the LLM will not classify images or train models directly. Instead, it will generate and revise augmentation policies encoded in a constrained JSON format, which will then be validated, converted into executable transforms, evaluated and evolved.

### Method and Work Plan

The project will use CIFAR-10 and Oxford Flowers102 as the main datasets. CIFAR-10 supports fast pipeline development, while Flowers102 tests the method on a fine-grained recognition problem. EuroSAT will be considered as a stretch dataset for cross-domain evaluation. The main classifier will be ResNet18, with MobileNetV3-Small or ViT-Tiny as optional extensions.

The implementation will consist of six phases: literature review and protocol design; dataset loading, low-data splits and baseline training; augmentation policy representation with JSON schema, operation set, validator and transform builder; LLM prompting, response parsing, caching and one-shot policy generation; evolutionary search with mutation, selection, rough evaluation and full evaluation; and finally main experiments, analysis and dissertation writing.

### Evaluation

The project will be evaluated quantitatively and qualitatively. The main metrics will be top-1 accuracy, macro-F1, per-class accuracy, mean and standard deviation across random seeds, training/search time and number of valid generated policies. Baselines will include no augmentation, standard augmentation, AutoAugment, RandAugment, TrivialAugment, AugMix, random policy search, one-shot LLM recommendation and pure evolutionary search without LLM feedback.

To evaluate the search process, I will compare rough-evaluation and full-evaluation rankings to test whether the rough-to-full mechanism reduces cost without discarding strong policies. I will also analyse operation frequencies, policy lineage, validation curves, confusion matrices and augmentation examples to understand whether datasets favour different augmentation types.

### Timetable

| Period | Phase | Planned work | Output |
|---|---|---|---|
| Late June | Planning and setup | Finalise ethics/DMP, project plan, literature review structure, repository setup | Approved plan, project structure |
| Week 1 | Literature and protocol | Review AutoAugment, RandAugment, AugMix, LLM-guided optimisation and evolutionary search; define datasets, metrics and baselines | Literature notes, experiment specification |
| Week 2 | Data and baselines | Implement CIFAR-10/Flowers102 loaders, low-data splits, ResNet18 training, standard augmentation baselines | Working baseline pipeline |
| Week 3 | Policy system | Implement augmentation policy JSON schema, validator, transform builder and policy preview visualisations | Executable policy module |
| Week 4 | LLM policy generation | Design prompts, parse LLM responses, cache outputs, generate initial policies and run one-shot LLM baseline | LLM policy pool and validation report |
| Weeks 5-6 | Evolutionary search | Implement population management, mutation, selection, rough evaluation and full evaluation; run pilot search on CIFAR-10 | Search loop and pilot results |
| Weeks 7-8 | Main experiments | Run experiments on CIFAR-10 and Flowers102 with key baselines and ablations | Main result tables and figures |
| Final period | Analysis and writing | Analyse policy behaviour, create visualisations, write method, experiments and discussion chapters | Dissertation draft and final results |

### Risks and Mitigation

The main risk is computational cost, which will be mitigated through rough-to-full evaluation, small candidate pools and limited search rounds. A second risk is invalid LLM-generated policies; this will be mitigated with a strict JSON schema, operation whitelist and parameter validation. A third risk is limited accuracy improvement. To address this, the project will also evaluate policy validity, search efficiency, policy evolution, dataset-specific preferences and failure cases.

### Stretch Goals

If time allows, I will add EuroSAT, evaluate a second classifier such as MobileNetV3-Small, include Grad-CAM or t-SNE/UMAP visualisations, and test robustness using corrupted or shifted image variants. These extensions are not required for the core project but would strengthen the analysis.

### Key References

Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., & Le, Q. V. (2019). AutoAugment: Learning augmentation strategies from data. *CVPR*.

Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical automated data augmentation with a reduced search space. *CVPR Workshops*.

Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., & Lakshminarayanan, B. (2020). AugMix: A simple data processing method to improve robustness and uncertainty. *ICLR*.

Duru, A., & Temizel, A. (2025). Adaptive augmentation policy optimization with LLM feedback. *ICMV*.
