## LLM-Guided Evolutionary Search for Data Augmentation Policy Optimization in Low-Data Image Recognition

### 1. Background and Motivation

Image recognition models often require substantial labelled data to generalise well. In many practical settings, however, labelled images are limited, class boundaries are fine-grained, or the test distribution differs from the training distribution. Data augmentation is one of the most widely used techniques for improving generalisation, but augmentation policies are usually designed manually or selected from generic recipes. A policy that works well for natural images may not be optimal for fine-grained flower recognition or remote-sensing image classification. Excessive colour distortion, cropping, erasing, or mixing may even damage class-discriminative information.

Automated data augmentation has addressed this problem by searching for better augmentation policies. AutoAugment showed that augmentation policies can be learned from validation performance, while RandAugment and TrivialAugment reduced the search cost and made automated augmentation more practical. AugMix further demonstrated the value of augmentation for robustness and uncertainty under distribution shift. More recently, large language models (LLMs) have been explored as feedback-driven optimisers for augmentation policy selection, and broader work on LLM-assisted evolutionary search suggests that LLMs can generate and revise candidate strategies while an external evaluator provides objective performance feedback.

This project proposes a feasible master’s-level study that combines computer vision, automated data augmentation, and LLM-guided evolutionary search. The central idea is not to use an LLM as an image classifier, but to use it as a policy proposer and improver. Candidate augmentation policies will be represented in a constrained JSON format, validated automatically, evaluated through image classification training, and evolved over several rounds using validation feedback.

### 2. Aim and Research Questions

The aim of this project is to develop and evaluate an LLM-guided evolutionary framework for discovering effective data augmentation policies in low-data image recognition.

The study will address the following research questions:

RQ1: Can LLM-generated augmentation policies improve low-data image classification compared with standard manual augmentation and established automated augmentation baselines?

RQ2: Does combining LLM feedback with evolutionary search produce more stable and effective policies than one-shot LLM recommendation or random policy search?

RQ3: Can a rough-to-full evaluation mechanism reduce search cost while preserving high-quality candidate policies?

RQ4: Do different visual domains, such as natural images, fine-grained flower images, and remote-sensing images, lead to different augmentation preferences?

RQ5: Can policy lineage, operation-frequency analysis, augmentation examples, confusion matrices, and feature visualisations help explain why the discovered policies work or fail?

### 3. Proposed Method

The proposed framework will consist of four stages. First, low-data splits will be created for selected image classification datasets. Second, an augmentation search space will be defined using operations such as random crop, resized crop, flip, rotation, affine transformation, colour jitter, grayscale conversion, Gaussian blur, random erasing, Mixup, and CutMix. Each policy will be encoded as a structured JSON object containing operation names, probabilities, magnitudes, and optional mixing parameters. A validator will check whether the policy is syntactically valid, uses only allowed operations, and keeps all parameters within safe ranges.

Third, an LLM will generate initial policies and later propose mutations or combinations of promising policies based on validation feedback. The LLM will not directly modify model weights or make final decisions; it will only suggest candidate augmentation configurations. Fourth, an evolutionary loop will maintain a policy population. Candidate policies will first undergo a rough evaluation using a small training budget, such as fewer epochs or a subset of the training data. The top-ranked policies will then enter a fuller evaluation with a longer training budget. The framework will track each policy’s source, parent policies, validation performance, operation composition, failure mode, and computational cost.

The main classifier will be ResNet18 because it is widely used, efficient, and easy to reproduce. MobileNetV3-Small or ViT-Tiny may be included as an optional second model if time permits. The implementation will use PyTorch / torchvision, with augmentation operations implemented through torchvision transforms or albumentations.

### 4. Datasets and Experimental Design

The main datasets will be CIFAR-10 and Oxford Flowers102. CIFAR-10 provides a fast and standard benchmark for pipeline development and controlled comparison. Flowers102 is more fine-grained, with 102 flower categories, intra-class variation, and inter-class similarity, making it suitable for testing whether augmentation policies should adapt to dataset characteristics. EuroSAT may be used as an optional cross-domain dataset; it contains 27,000 Sentinel-2 remote-sensing images across 10 land-use and land-cover classes.

The experimental baselines will include no augmentation, standard augmentation, AutoAugment, RandAugment, TrivialAugment, AugMix, random policy search, one-shot LLM policy recommendation, pure evolutionary search without LLM feedback, and the proposed LLM-guided evolutionary search. Evaluation metrics will include top-1 accuracy, macro-F1, per-class accuracy, mean and standard deviation across random seeds, search time, number of valid policies, and rough-to-full ranking correlation.

In addition to performance tables, the project will include qualitative and analytical outputs: augmentation examples, policy evolution curves, operation-frequency statistics, confusion matrices, failure cases, and optional t-SNE/UMAP or Grad-CAM visualisations. These analyses are important because the project aims not only to improve accuracy, but also to understand how LLM-guided search behaves across different image domains.

### 5. Feasibility and Resources

The project is feasible within a master’s dissertation timeline because it relies on publicly available datasets, standard image classifiers, and established augmentation baselines. CIFAR-10 and Flowers102 can be accessed through torchvision, while EuroSAT is publicly available through its project repository and common dataset libraries. The project does not require training large foundation models or generating synthetic images.

The recommended compute resource is one GPU with 8GB to 16GB VRAM, such as an NVIDIA T4, RTX 3060, RTX 4060, or an equivalent cloud GPU. Google Colab, Kaggle Notebook, or a university GPU server should be sufficient for the main experiments. To control cost, the search will use a rough-to-full evaluation design: many policies will be evaluated cheaply, and only a small subset will receive longer training. LLM calls will be cached with prompts, responses, parsed policies, validation results, and evaluation outcomes to support reproducibility.

### 6. Expected Contributions

This project is expected to make four contributions. First, it will propose a reproducible LLM-guided evolutionary framework for data augmentation policy search in low-data image recognition. Second, it will introduce a constrained policy representation and validation pipeline to reduce invalid or unsafe LLM outputs. Third, it will evaluate whether rough-to-full policy evaluation can reduce computational cost while retaining strong candidate policies. Fourth, it will analyse dataset-specific augmentation preferences and provide interpretable visual evidence through policy statistics and classification diagnostics.

### References

Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., & Le, Q. V. (2019). AutoAugment: Learning augmentation strategies from data. *CVPR*. https://openaccess.thecvf.com/content_CVPR_2019/html/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.html

Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical automated data augmentation with a reduced search space. *CVPR Workshops*. https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html

Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., & Lakshminarayanan, B. (2020). AugMix: A simple data processing method to improve robustness and uncertainty. *ICLR*. https://arxiv.org/abs/1912.02781

Duru, A., & Temizel, A. (2025). Adaptive augmentation policy optimization with LLM feedback. *ICMV*. https://arxiv.org/abs/2410.13453

PyTorch. (2026). CIFAR10 dataset documentation. https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html

PyTorch. (2026). Flowers102 dataset documentation. https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html

Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE JSTARS*. https://github.com/phelber/eurosat
