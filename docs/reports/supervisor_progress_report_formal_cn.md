# 项目进度汇报

项目最初的设想是让大语言模型直接生成图像数据增强策略，随后逐步收敛为一个更可控的研究框架：将大语言模型作为语义层面的变异和交叉算子，嵌入由本地评估器控制的进化搜索循环中。候选增强策略被表示为受约束的 JSON 对象，经过验证器检查，并在必要时进行修复；随后通过监督学习训练或 FixMatch 半监督训练进行评估，再根据排序反馈选择更有潜力的策略。

目前的实验结果支持一个更聚焦的研究判断：无约束或弱约束的大语言模型增强策略搜索稳定性不足，而基于强基线初始化和排序反馈的大语言模型进化搜索更可靠。在 CIFAR-10 低数据量监督分类实验中，最佳 OpenAI 进化策略在测试准确率上几乎追平最强本地 Mixup 基线，并取得了略高的 macro-F1。在 EuroSAT 数据集上，结果表明当前设置下较保守的增强策略或无增强更合适。在 FixMatch 实验中，半监督学习显著提升了 CIFAR-10 的绝对准确率，同时 FixMatch-in-the-loop 的大语言模型进化原型已经能够在修复后生成有效的强增强分支策略。当前 LLM 进化得到的 FixMatch 策略仍低于 RandAugment 和 TrivialAugment，因此下一阶段应重点推进多随机种子评估、消融实验，以及更大搜索预算下的 FixMatch-in-the-loop 策略进化。

![当前实验证据概览](../../results/combined/completed_experiment_visuals/supervisor_result_dashboard.png)

## 当前研究问题

**基于强基线初始化和排序反馈的大语言模型进化搜索，能否为低数据量图像识别生成有效、可解释且具备竞争力的数据增强策略，尤其是在 FixMatch 的强增强分支中？**

## 方法

采用由评估器控制的大语言模型进化循环。大语言模型负责提出候选数据增强策略，而策略是否被保留由本地训练评估结果决定。

![方法流程图](../../results/combined/completed_experiment_visuals/supervisor_method_pipeline.png)

整体流程包含五个主要部分。

1. 数据增强策略被编码为 JSON 基因型，其中包含增强操作、概率、强度，以及可选的 Mixup 和 CutMix 参数。

2. 所有策略都会通过验证器检查，包括操作名称、参数范围、子策略长度和数据集相关风险提示。
3. 初始种群由强传统基线构成，包括 standard augmentation、Mixup、CutMix、RandAugment、TrivialAugment 和 no augmentation。
4. 大语言模型接收带排序的种群反馈，并生成变异或交叉后的子代策略。
5. 系统采用粗评估到精评估的机制，先用较低成本筛选候选策略，再对入选策略进行更完整的训练评估。

最新阶段的方法将这一循环扩展到了 FixMatch。在 FixMatch 中，弱增强用于生成伪标签，强增强用于一致性训练，因此强增强分支天然适合作为大语言模型引导策略进化的目标。

## 实验结果

### 监督学习实验

| 数据集 | 最佳本地基线 | 最佳 LLM/进化结果 | 结果解读 |
|---|---:|---:|---|
| CIFAR-10 早期监督实验 | RandAugment 43.97% | One-shot LLM 45.25%；LLM evolution 40.48% | 早期 LLM 搜索可以运行，但稳定性不足。 |
| Flowers102 早期监督实验 | TrivialAugment 47.47% | LLM evolution 31.86% | 传统增强方法明显更强。 |
| EuroSAT 早期监督实验 | Standard 75.75% | LLM evolution 68.68% | 较强增强可能破坏领域相关视觉信号。 |
| CIFAR-10 强 OpenAI 实验 | Mixup 48.50% | OpenAI-evolved `gen02_mut_005` 48.45% | 基于强基线初始化和排序反馈的 LLM 进化搜索已具备竞争力。 |
| EuroSAT 强 OpenAI 实验 | No augmentation 73.95% | OpenAI-evolved `gen01_mut_002` 72.45% | 当前设置下保守策略更合适。 |

监督学习阶段最强结果出现在 CIFAR-10 上。最佳 OpenAI 进化策略保留了 RandomCrop、HorizontalFlip、轻量 ColorJitter，以及 `mixup_alpha=0.2` 的 Mixup。这一结果具有分析价值，因为大语言模型保留了强本地基线中的有效先验，并进行了一处较小且可解释的修改，同时避免了过度复杂的策略组合。

### FixMatch 实验

| 方法 | 测试准确率 | 测试 macro-F1 | 验证准确率 | 结果解读 |
|---|---:|---:|---:|---|
| Supervised Mixup baseline | 48.50% | 46.30% | 47.40% | 最强监督学习本地准确率基线。 |
| Supervised LLM best | 48.45% | 47.22% | 50.20% | 最佳监督学习 LLM 进化策略。 |
| FixMatch standard | 56.75% | 55.12% | 56.60% | 无标签数据显著提升性能。 |
| FixMatch reused LLM policy | 59.00% | 58.38% | 61.40% | 监督阶段得到的 LLM 策略可以有效迁移到 FixMatch。 |
| FixMatch repaired LLM child `mut_001` | 59.05% | 58.05% | 62.70% | In-loop LLM 进化流程已经可运行，并相对复用策略取得很小提升。 |
| FixMatch TrivialAugment | 61.50% | 61.08% | 63.00% | 强规范增强基线。 |
| FixMatch RandAugment | 62.05% | 61.05% | 63.90% | 当前最佳本地结果。 |

FixMatch 实验证明，半监督学习是研究强增强策略更合适的设置。CIFAR-10 上最佳本地准确率从监督训练中的 48.50% 提升到 FixMatch RandAugment 的 62.05%。最佳 in-loop LLM 子代策略达到 59.05%，说明该方法能够生成有效策略，但与规范强增强方法相比仍有差距。

## SOTA 与强基线背景

![SOTA 背景对照](../../results/combined/completed_experiment_visuals/sota_context_comparison.png)

已有外部论文为本项目提供了重要参照。UDA 在 250-label CIFAR-10 半监督设置下报告了 94.57% 的准确率，FixMatch 在相近协议下报告了 94.93% 的准确率。全数据自动数据增强方法，如 AutoAugment 和 Population Based Augmentation，在更大训练预算和搜索预算下报告了约 98.5% 的 CIFAR-10 准确率。这些结果与我当前 15 epoch、ResNet18-CIFAR、学生项目规模的本地实验设置并非同一实验条件下的直接对比，但它们清楚展示了性能差距，并帮助界定后续改进目标。

因此，论文中应主要主张本地强基线对比下的发现，并将 SOTA 文献作为外部背景。本项目当前贡献在于实现并分析了一个受控的大语言模型引导数据增强策略进化框架，其中包括强基线初始化、排序反馈、策略验证、策略修复、粗评估到精评估机制，以及面向 FixMatch 强增强分支的策略搜索。

## 已完成工作与剩余工作

| 工作项 | 状态 | 证据 | 后续行动 |
|---|---|---|---|
| 文献综述与研究问题收敛 | 已完成，仍需最终更新 | Introduction/background 和参考文献库已更新 | 在 Discussion 中加入最终实验解释。 |
| 数据集准备与低数据量划分 | 已完成 | CIFAR-10、Flowers102、EuroSAT 数据加载器和确定性划分 | 将 CIFAR-10 和 EuroSAT 保留为最终核心数据集。 |
| 监督学习基线流程 | 已完成 | 基线结果表和综合汇总 | 增加选定方法的多随机种子重复实验。 |
| JSON 策略 schema、transform builder、validator | 已完成 | Policy、builder 和 validator 模块 | 作为核心工程贡献进行呈现。 |
| 基于强基线初始化和排序反馈的 OpenAI 进化搜索 | 已完成 | CIFAR-10 和 EuroSAT 强 OpenAI 实验结果 | 分析策略谱系和操作频率。 |
| FixMatch 训练与策略复用 | 已完成 | FixMatch standard、RandAugment、TrivialAugment 和 LLM 策略结果 | 增加类别级诊断。 |
| FixMatch-in-the-loop LLM 进化 | 原型已完成 | 修复后的 LLM 子代已在 FixMatch 下评估 | 扩展到 2 至 3 代，并扩大子代池。 |
| 策略修复层 | 原型已完成 | 无效 LLM 子代已修复并评估 | 增加修复率和修复影响分析。 |
| SOTA/强基线对照 | 已用于进度报告 | SOTA 对照图和已核实参考文献 | 转化为论文 Discussion 小节。 |
| 消融实验 | 尚未完成 | 暂无完整消融表 | 运行强基线初始化、排序反馈、修复机制和粗/精评估消融。 |
| 统计可靠性 | 尚未完成 | 多数强实验仍为单随机种子 | 对最终策略和基线运行 3 seed 对比。 |
| 论文 Methodology 章节 | 草稿已完成 | Methodology LaTeX 章节已创建 | 根据导师反馈继续修改。 |
| Results、Discussion、Conclusion 章节 | 尚未完成 | 现有报告和图表可用 | 在补充实验后撰写最终章节。 |

下一阶段应优先提升实验可靠性并服务于论文写作。近期计划包括运行多随机种子的最终对比实验、补充消融实验、扩展 FixMatch-in-the-loop 搜索预算，并将当前实验报告转化为论文的 Results 和 Discussion 章节。

当前最主要的风险是统计可靠性。若干关键实验仍为单随机种子结果，并且最佳监督 LLM 策略与 Mixup 之间的差距非常小。为缓解这一问题，我计划至少对最终 CIFAR-10 FixMatch 基线和最佳 LLM 进化策略运行三个随机种子的重复实验。

第二个风险是粗评估到精评估之间的排序噪声。粗评估可以帮助低成本筛掉较弱策略，但它并不总能可靠预测最终性能，尤其是在 FixMatch 场景下。我会分析粗评估与精评估之间的相关性，并考虑对入选候选策略使用重复粗评估种子。

第三个风险是搜索空间匹配不足。当前 JSON 策略空间近似 RandAugment 风格的行为，而 torchvision 中的 RandAugment 和 TrivialAugment 是更成熟的规范实现。我会考虑将 RandAugment 或 TrivialAugment 作为宏操作加入搜索空间。

