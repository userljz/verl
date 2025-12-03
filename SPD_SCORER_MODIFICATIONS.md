# SPD Scorer 对 verl 库的修改记录

> **文档版本**: 1.4  
> **更新日期**: 2025-11-26  
> **目的**: 详细记录为实现 Speculative Decoding Scoring Model 对原始 verl 库所做的所有修改及新增模块的功能说明。

---

## 📋 修改概览

为了实现 Speculative Decoding Scorer (SPD Scorer) 的训练，我们基于 verl 框架进行了扩展。所有的修改旨在实现以下核心流程：
1.  **自定义模型**: 使用 `ScoringActor` 替代标准 LLM，输出对 Draft Token 的 Accept/Reject 概率。
2.  **自定义 Rollout**: 使用 `SPDRollout` 执行一次 Forward 并进行 Bernoulli 采样，而非自回归生成。
3.  **自定义 Reward**: 使用 `spd_scorer_reward` 基于 Draft 接受长度和最终答案正确性计算复杂奖励。
4.  **自定义数据管道**: 预处理 Context + Draft + Target 的特殊输入格式。

### 文件清单

#### ✨ 新增文件 (New Files)
| 文件路径 | 模块/类 | 说明 |
|:---|:---|:---|
| `spd_scorer.py` | `ScoringActor` | SPD Scorer 模型核心实现。新增 `AutoModelForSPDScoring` 工厂类适配 verl 加载流程。 |
| `verl/workers/rollout/spd_rollout.py` | `SPDRollout` | 自定义 Rollout 策略，执行非自回归的 Bernoulli 采样。 |
| `verl/utils/reward_score/spd_scorer_reward.py` | `compute_score` | 自定义 Reward Function，集成 vLLM 离线推理进行补全验证。 |
| `verl/utils/dataset/spd_dataset.py` | `SPDRLHFDataset` | 自定义 Dataset，支持预计算的 `input_ids` 和索引偏移修正。 |
| `train_spd_scorer.py` | `run_training` | 训练入口脚本。**包含 Monkey Patch 逻辑** 以强制加载 SPD 模型。 |

#### ✏️ 修改文件 (Modified Files)
| 文件路径 | 修改内容 | 说明 |
|:---|:---|:---|
| `verl/utils/reward_score/__init__.py` | `default_compute_score` | 注册 `spd_scorer` 数据源，分发参数到新的 Reward Function。 |
| `verl/workers/rollout/base.py` | `_ROLLOUT_REGISTRY` | 注册 `("spd", "sync")` 对应的 Rollout 类。 |

---

## 🔍 详细模块说明

### 1. 模型层: `spd_scorer.py`

此文件定义了 SPD Scorer 的模型架构。模型基于 Llama-3-8B (Backbone) + LoRA + Score Head。

*   **`ScoringModelConfig` (Class)**: 配置类，定义了 `hidden_size`, `lora_rank` 等超参数。
*   **`ScoreHead` (Class)**: 简单的 MLP，将 Hidden States 映射为 Accept/Reject Logit。
*   **`ScoringActor` (Class)**:
    *   **Mismatch Mask**: 在 Forward 中，强制 Match 位置的 logit 为极大概率，确保 Ground Truth 必定被 Accept。
*   **`AutoModelForSPDScoring` (Class)**:
    *   **新增**: 一个工厂类，模拟 `AutoModel` 的接口 (`from_pretrained`, `from_config`)。
    *   作用：作为适配器，将 verl 的标准加载调用转换为 `ScoringActor` 的初始化调用。

### 2. 执行层: `verl/workers/rollout/spd_rollout.py`

此文件实现了 SPD 专用的 Rollout 策略，替代了 verl 默认的自回归生成。

*   **`SPDRollout` (Class)**:
    *   继承自 `BaseRollout`。
    *   **`generate_sequences`**: 执行单次 Forward -> Bernoulli 采样 -> 构造 Loss Mask (屏蔽 Match 位置和 Padding)。

### 3. 评估层: `verl/utils/reward_score/spd_scorer_reward.py`

此文件实现了复杂的 Reward 计算逻辑。

*   **`compute_score` (Function)**:
    *   利用 `response_ids` 计算有效长度 L。
    *   构造 `Context` + `Draft[:L]` 并调用 vLLM 进行补全。
    *   验证补全结果，应用四场景奖励公式。

### 4. 数据层: `verl/utils/dataset/spd_dataset.py`

自定义 Dataset，优化了数据加载流程。

*   **`SPDRLHFDataset` (Class)**:
    *   自动修正 Left Padding 带来的索引 (`draft_start_idx`) 偏移。
    *   跳过默认的 Chat Template 处理，直接使用预处理好的 `input_ids`。

### 5. 训练入口: `train_spd_scorer.py`

负责数据准备和启动训练。

*   **Monkey Patch (关键 Hack)**:
    *   为了在不修改 `verl/utils/model.py` 的前提下让 verl 加载自定义的 `ScoringActor`，我们在脚本开头执行了 Monkey Patch。
    *   **被替换函数**: `verl.utils.model.create_huggingface_actor`
    *   **替换逻辑**: 拦截调用，直接返回 `AutoModelForSPDScoring.from_config(...)` 创建的 `ScoringActor` 实例。

### 6. 注册修改 (原有文件)

为了让 verl 识别上述自定义模块，对原有文件进行了少量修改：

*   **`verl/utils/reward_score/__init__.py`**: 在 `default_compute_score` 中增加了 `spd_scorer` 分支。
*   **`verl/workers/rollout/base.py`**: 注册了 `("spd", "sync")` Rollout。

---

## 🚀 训练流程总结

1.  **启动**: 运行 `train_spd_scorer.py`。
2.  **Patch**: 脚本首先应用 Monkey Patch，劫持模型加载逻辑。
3.  **加载**: verl Trainer 调用 `create_huggingface_actor`，被重定向到 `AutoModelForSPDScoring`，加载 `ScoringActor`。
4.  **数据**: `SPDRLHFDataset` 加载数据并修正索引。
5.  **Rollout & Reward**: `SPDRollout` 和 `spd_scorer_reward` 执行采样和评分。
6.  **更新**: GRPO 更新模型参数。
