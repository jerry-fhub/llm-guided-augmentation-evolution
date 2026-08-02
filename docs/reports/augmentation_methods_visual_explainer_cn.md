# 图像增强策略可视化说明

在低数据量图像识别任务中，训练样本有限，模型容易记住训练集中的局部细节，泛化能力不足。数据增强的作用是为同一张训练图片生成多个不同但语义一致的视图，例如轻微裁剪、翻转、颜色扰动、旋转、模糊，或将两张图片按比例混合。模型在训练过程中看到这些变化后的图片，可以学习更稳定的视觉特征。

本项目研究的问题是：能否让大语言模型参与数据增强策略的设计，并通过进化搜索逐步改进这些策略。具体来说，大语言模型不会直接决定最终结果；它负责根据已有策略和验证集反馈提出新的候选增强策略，随后本地训练评估器负责判断这些策略是否真的有效。整个系统关注三件事：增强策略是否有效、是否符合约束、是否可解释。

## 同一批图片经过不同增强方法后的效果

下图选取了 CIFAR-10 训练集中的四张图片，并展示了当前实验中使用过的几类增强方法。由于 CIFAR-10 原始图片只有 32×32 像素，图中的图片经过放大显示，因此看起来会有明显像素感。这里的重点是观察不同方法对图像内容的改变方式。

![CIFAR-10 不同增强方法可视化](../../results/combined/completed_experiment_visuals/augmentation_method_examples_cifar10.png)

从图中可以直观看到，Standard 方法主要做随机裁剪和水平翻转，变化相对温和。LLM supervised 策略是在监督学习实验中进化得到的最佳策略，它保留 crop/flip 的基础结构，并加入轻量 ColorJitter，因此整体仍然偏保守。RandAugment 和 TrivialAugment 是更强的规范增强方法，它们可能带来更明显的颜色、几何或强度变化，在 FixMatch 实验中表现更强。FixMatch weak 分支用于生成伪标签，因此增强幅度较小。LLM FixMatch child 是在 FixMatch-in-the-loop 实验中由大语言模型生成并经过修复的强增强子代策略，它主要保留 crop/flip，并加入适中的颜色扰动。

这张图展示了一个重要现象：当前 LLM 进化策略更倾向于生成“安全、温和、可解释”的增强组合，而 RandAugment/TrivialAugment 往往覆盖更强的扰动空间。这个差异也解释了为什么 LLM 策略已经有效，但在 FixMatch 中仍落后于当前最强本地规范增强基线。

## Mixup 和 CutMix：样本级增强

除了对单张图片做裁剪、翻转、颜色扰动之外，实验中还使用了 Mixup 和 CutMix。它们属于样本级增强：增强发生在一个 batch 内，图片和标签会一起被混合。

![Mixup 和 CutMix 示意图](../../results/combined/completed_experiment_visuals/mixup_cutmix_visual_demo_cifar10.png)

Mixup 会把两张图片按比例线性叠加，同时把标签也按相同比例混合。CutMix 会把一张图片中的局部区域替换成另一张图片的区域，标签也按照替换面积进行混合。在本项目的监督学习实验中，Mixup 是非常强的本地基线；最佳监督 LLM 进化策略保留了 `mixup_alpha=0.2`，并在其基础上加入轻量图像级扰动。这说明大语言模型在排序反馈下学会了保留强基线中的有效成分。

## 比较了哪些方法

当前实验脉络可以分为三层。第一层是传统本地基线，包括 no augmentation、standard augmentation、Mixup、CutMix、RandAugment 和 TrivialAugment，用于确认任务难度和可达到的本地性能。第二层是 LLM 参与的策略生成，包括 one-shot LLM、随机搜索、基于强基线初始化的 ranked LLM evolution。第三层是 FixMatch 半监督扩展，在这个设置中，弱增强用于生成伪标签，强增强用于一致性训练，因此强增强分支成为 LLM 策略搜索的主要目标。

| 方法类别 | 图像增强方式 | 在项目中的作用 |
|---|---|---|
| Standard augmentation | 随机裁剪、水平翻转 | 最基础的本地增强基线 |
| Mixup / CutMix | batch 内图片与标签混合 | 监督学习中的强基线，用于测试 LLM 是否能保留有效先验 |
| RandAugment / TrivialAugment | 更强、更随机的颜色和几何扰动 | FixMatch 中最强的本地规范增强参照 |
| LLM supervised policy | LLM 根据监督学习排序反馈生成的策略 | 验证 LLM 是否能提出有效且可解释的策略变体 |
| FixMatch weak branch | 温和 crop/flip | 生成稳定伪标签 |
| LLM FixMatch child | LLM 在 FixMatch 搜索循环中生成的强分支策略 | 验证 LLM 能否在半监督训练框架内进行策略进化 |

## 5. 整个项目的工作流程

本项目的完整流程可以理解为“生成策略、检查策略、训练评估、反馈进化”。首先，增强策略被写成 JSON 形式，每个策略包含若干增强操作、概率和强度，例如 `RandomCrop`、`HorizontalFlip`、`ColorJitter`、`Rotation` 等。其次，验证器检查策略是否符合搜索空间约束，例如操作名称是否合法、概率是否在有效范围内、子策略长度是否过长。对于 FixMatch-in-the-loop 生成的策略，如果 LLM 输出的策略超出本地实现约束，系统会进行修复，使其能够被实际训练代码执行。然后，候选策略进入训练评估流程：较多候选先通过低成本粗评估筛选，较有希望的策略再进入完整训练评估。最后，系统把表现较好的策略、较差的策略、准确率和复杂度反馈给大语言模型，让它生成下一代候选策略。

这个流程的研究价值在于，它把大语言模型的语义生成能力和本地实验评估结合起来。大语言模型可以提出人类可读的策略修改理由，例如“保留 crop/flip，降低破坏性颜色扰动，保护伪标签质量”；本地评估器则提供真实训练结果，避免策略选择完全依赖语言模型判断。

## 当前结果说明

下图汇总了当前 CIFAR-10 本地实验中的关键结果。这里的数值来自当前项目已完成实验，训练预算较小，主要用于比较同一代码框架下不同方法的相对表现。

![CIFAR-10 当前关键结果](../../results/combined/completed_experiment_visuals/augmentation_explainer_result_bar.png)

在监督学习设置下，最佳 LLM 进化策略达到了 48.45% 的测试准确率，几乎追平 Mixup 的 48.50%，并在 macro-F1 上略有优势。在 FixMatch 设置下，半监督学习显著提高了整体准确率：standard FixMatch 达到 56.75%，复用监督 LLM 策略达到 59.00%，FixMatch-in-the-loop 生成并修复的 LLM 子代达到 59.05%。当前最强本地结果仍然是 FixMatch RandAugment，达到 62.05%。

这些结果表明，本项目已经完成了一个可运行、可验证、可解释的 LLM-guided augmentation evolution 框架。当前方法已经能够生成有效策略，并在部分监督学习实验中接近强本地基线。在 FixMatch 场景中，LLM 策略已经带来可用的强增强分支，但搜索预算、策略空间和多随机种子可靠性仍需要进一步提升。



论文可以重点讨论一个清晰结论：大语言模型适合作为数据增强策略搜索中的语义算子，尤其适合提出可解释、受约束的候选策略；但要在强半监督设置中超过 RandAugment/TrivialAugment，还需要更大的搜索预算、更精细的策略空间，以及多随机种子实验来验证稳定性。

