# 技术实现方案：LLM 引导进化搜索的图像增强策略优化

## 1. 项目定位

本方案面向图像识别中的自动数据增强策略优化。系统目标是：给定图像分类数据集、模型架构和训练预算，自动搜索适合该任务的数据增强策略，并验证 LLM 引导进化搜索是否优于手工增强、随机增强和现有自动增强基线。

该项目的核心不是训练一个特别大的图像模型，而是构建一个可复现的 AutoML-style 实验框架。LLM 负责提出和改写增强策略，进化算法负责维护策略种群、选择和变异，图像分类模型负责提供客观性能反馈。最终论文围绕分类准确率、策略演化过程、增强策略行为分析和可解释性图表展开。

## 2. 推荐代码结构

```text
Project/
  data/
    raw/
      cifar10/
      flowers102/
      eurosat/
    processed/
      low_data_splits/
      augmentation_previews/
  src/
    image_aug_evolution/
      data/
        datasets.py
        splits.py
        datamodule.py
      models/
        classifiers.py
        train.py
        evaluate.py
      augmentation/
        search_space.py
        policy.py
        policy_parser.py
        validator.py
        transforms_builder.py
        preview.py
      evolution/
        population.py
        selection.py
        mutation.py
        crossover.py
        lineage.py
      llm/
        prompts.py
        client.py
        cache.py
        feedback.py
      evaluation/
        rough_eval.py
        full_eval.py
        metrics.py
        runner.py
        ablation.py
      analysis/
        policy_behavior.py
        result_tables.py
        confusion_matrix.py
        feature_visualization.py
      visualization/
        plot_curves.py
        plot_policy_tree.py
        plot_aug_examples.py
  experiments/
    configs/
  results/
    tables/
    figures/
    logs/
    policies/
  thesis_assets/
```

## 3. 数据与任务设置

建议将 CIFAR-10 作为主调试数据集，因为它下载方便、训练速度快、相关 baseline 多。正式实验至少保留 CIFAR-10 和 Flowers102；如果时间允许，加入 EuroSAT 作为跨领域验证。CIFAR-100 可以作为更难的自然图像扩展，但不是必做。

低数据设置是本项目的重要场景。建议对每个数据集构造 10%、20% 和 100% 三种训练比例，或者每类固定抽取 20、50、100 张图片。验证集和测试集保持独立，所有方法使用相同随机种子和相同数据划分。这样可以回答增强策略在数据不足时是否更有价值。

## 4. 增强策略表示

增强策略采用 JSON 配置表示，避免让 LLM 直接输出不可控代码。一个策略包含若干 `sub_policies`，每个子策略由 1 到 3 个操作组成。每个操作包含 `name`、`probability` 和 `magnitude`，例如：

```json
{
  "policy_id": "llm_gen_001",
  "sub_policies": [
    [
      {"name": "RandomResizedCrop", "probability": 1.0, "magnitude": 0.75},
      {"name": "ColorJitter", "probability": 0.6, "magnitude": 0.35}
    ],
    [
      {"name": "RandomHorizontalFlip", "probability": 0.5, "magnitude": 1.0},
      {"name": "RandomErasing", "probability": 0.25, "magnitude": 0.2}
    ]
  ],
  "mixing": {"mixup_alpha": 0.2, "cutmix_alpha": 0.0}
}
```

允许的操作库建议包括 RandomCrop、RandomResizedCrop、HorizontalFlip、VerticalFlip、Rotation、Affine、ColorJitter、Grayscale、GaussianBlur、Solarize、Posterize、RandomErasing、Mixup 和 CutMix。不同数据集可以定义不同约束，例如 CIFAR 可允许更强颜色扰动，Flowers 对颜色扰动需适中，EuroSAT 可允许旋转和翻转但限制过强颜色改变。

## 5. LLM 生成与校验模块

LLM 输入应包含数据集描述、类别数、图像尺寸、模型架构、低数据比例、可选增强操作列表、禁止操作、上一轮策略表现和失败原因。输出必须是 JSON，不允许输出自然语言解释作为策略主体。系统解析 JSON 后进入 validator。

Validator 至少检查以下内容：JSON 是否可解析；增强操作是否在白名单中；概率是否在 0 到 1 之间；强度是否在允许范围内；是否存在重复无意义操作；是否包含与数据集语义冲突的操作；Mixup/CutMix 参数是否合理。校验失败的策略记录失败原因，不进入训练。

## 6. 进化搜索模块

策略种群由三类个体组成：手工 baseline、随机采样策略和 LLM 生成策略。每个个体保存策略配置、来源、父策略、校验结果、粗评估分数、精评估分数、训练时间和行为特征。

每轮演化分为初始化、校验、粗评估、选择、LLM 反馈式变异/融合、精评估和日志记录。选择函数综合验证准确率、Macro-F1、训练稳定性和训练成本。变异可以改变某个增强操作、概率、强度、子策略顺序或 mixup/cutmix 参数。交叉可以融合两个优秀策略的子策略。LLM 的作用是根据结果文本反馈提出更有方向性的变异建议，例如减少过强颜色扰动、增加几何鲁棒性或降低 random erasing 概率。

## 7. 粗评估-精评估机制

粗评估用于快速筛选策略，建议配置为 ResNet18 或 MobileNetV3-Small，训练 5 到 10 个 epoch，使用训练集子集或低分辨率输入。每轮保留前 20% 至 30% 的策略进入精评估。

精评估用于正式结果，建议训练 30 到 80 个 epoch，具体取决于数据集大小和算力。最终报告多随机种子均值和标准差。论文中需要分析粗评估排名与精评估排名的 Spearman correlation，以说明粗评估是否足以作为筛选代理。

## 8. 模型与训练配置

主模型建议使用 ResNet18，因为结构经典、训练速度快、解释容易。第二模型可选择 MobileNetV3-Small，用于验证策略是否对轻量模型也有效。如果想更贴近现代视觉模型，可加入 ViT-Tiny 或 ConvNeXt-Tiny，但不建议作为必做。

优化器建议使用 SGD momentum 或 AdamW。CIFAR-10 可训练 50 到 100 epoch，Flowers102 和 EuroSAT 可使用 ImageNet 预训练模型进行 fine-tuning，训练 20 到 50 epoch。所有方法保持相同 batch size、learning rate schedule、weight decay 和 random seed。

## 9. 实验组设计

正式实验建议包含以下对照组：No Augmentation、Standard Augmentation、AutoAugment、RandAugment、TrivialAugment、AugMix、Random Search Policy、One-shot LLM Policy、Pure Evolution without LLM Feedback、LLM-guided Evolutionary Policy。若时间紧张，至少保留前六个基线和完整方法。

消融实验建议包括：移除 LLM 反馈、移除进化选择、移除策略校验、移除粗评估-精评估、只使用通用 prompt、不使用数据集描述。这样可以证明各模块不是装饰，而是对最终结果有贡献。

## 10. 指标与可视化

分类性能指标包括 Top-1 Accuracy、Macro-F1、per-class accuracy、mean/std across seeds。效率指标包括总训练时间、LLM 调用次数、候选策略数、每个有效策略平均成本。策略质量指标包括校验通过率、策略存活率、操作频率、增强强度分布和策略多样性。

可视化建议包括增强样例图、策略演化谱系图、验证准确率收敛曲线、不同方法准确率柱状图、混淆矩阵、错误样本分析和 t-SNE/UMAP 特征图。若时间允许，可加入 Grad-CAM 展示模型关注区域是否更稳定。

## 11. 实现顺序

第一阶段实现数据加载、低数据划分、ResNet18 训练和常规 baseline。第二阶段实现增强策略 JSON 表示、validator 和 transforms builder。第三阶段实现随机搜索与纯进化搜索。第四阶段接入 LLM prompt、缓存和反馈式策略生成。第五阶段实现粗评估-精评估和正式实验 runner。第六阶段完成可视化、结果表格和论文分析。

## 12. 计算资源

最低配置为 16GB RAM 的个人电脑，推荐配置为 8GB 至 16GB 显存 GPU。CIFAR-10 可在普通 GPU 上快速完成，Flowers102 和 EuroSAT 建议使用预训练模型 fine-tuning。LLM 使用 API 即可，不建议本地部署大模型。所有 LLM 调用必须缓存，所有实验配置必须保存 YAML 或 JSON，以便复现。

## 13. 取舍边界

必须完成：CIFAR-10 或 Flowers102 数据集、ResNet18 baseline、标准增强基线、LLM 策略生成、策略校验、进化搜索、粗评估-精评估、最终分类实验和可视化分析。

应该完成：CIFAR-10 + Flowers102 两个数据集、EuroSAT 跨领域验证、MobileNetV3 第二模型、消融实验、多随机种子。

可选完成：ViT/ConvNeXt、MedMNIST、Grad-CAM、CIFAR-C robustness、贝叶斯优化对比、更多 LLM 模型对比。
