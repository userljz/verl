# SPD Scorer 对 verl 库的修改记录

> **文档版本**: 1.5  
> **更新日期**: 2025-12-24  
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
| `verl/workers/rollout/spd_rollout.py` | `SPDRollout` | 自定义 Rollout 策略，执行非自回归的 Bernoulli 采样，集成 vLLM 进行 Hybrid 补全验证。 |
| `verl/utils/reward_score/spd_scorer_reward.py` | `compute_score` | 自定义 Reward Function，基于有效长度和答案正确性计算复杂奖励。 |
| `verl/utils/dataset/spd_dataset.py` | `SPDRLHFDataset` | 自定义 Dataset，支持预计算的 `input_ids` 和索引偏移修正。 |
| `train_spd_scorer.py` | `run_training` | 训练入口脚本。**包含 Monkey Patch 逻辑** 以强制加载 SPD 模型。 |
| `scripts/train_spd_scorer.sh` | Shell Script | 训练启动脚本，配置环境变量、模型路径、训练超参数。DEBUG 日志级别。 |
| `scripts/train_spd_scorer_info.sh` | Shell Script | 训练启动脚本的 INFO 日志级别版本，用于正式实验减少日志输出。 |
| `verl/workers/rollout/spd_rollout_dep.py` | (Deprecated) | 旧版 SPD Rollout 备份，保留供参考。 |

#### ✏️ 修改文件 (Modified Files)
| 文件路径 | 修改内容 | 说明 |
|:---|:---|:---|
| `verl/utils/reward_score/__init__.py` | `default_compute_score` | 注册 `spd_scorer` 数据源，分发参数到新的 Reward Function。 |
| `verl/workers/rollout/base.py` | `_ROLLOUT_REGISTRY` | 注册 `("spd", "sync")` 对应的 Rollout 类。 |
| `verl/trainer/ppo/ray_trainer.py` | 添加 Debug 日志 | 在数据流转的关键节点（DataLoader读取、Reward计算前）打印 Batch Size，用于验证数据维度变化。 |
| `verl/workers/actor/dp_actor.py` | 添加 Debug 日志 | 在 `update_policy` 方法中添加详细的训练指标日志 (advantages, log_prob, loss 等)，用于调试策略更新过程。 |
| `verl/workers/reward_manager/batch.py` | `verify` 方法 | 修改参数传递，新增 `prompt_ids`、`response_ids`、`attention_mask`、`batch_tensors` 传递给 reward function，支持 SPD Scorer 访问完整 Batch Tensor。 |

---

## 🔍 详细模块说明

### 1. 模型层: `spd_scorer.py`

此文件定义了 SPD Scorer 的模型架构。模型基于 Llama-3-8B (Backbone) + LoRA + Score Head。

*   **`ScoringModelConfig` (Class)**: 配置类，定义了 `hidden_size`, `lora_rank`, `mismatch_logit_value` 等超参数。
*   **`SPDInputData` (Class)**: SPD 模型的输入数据结构（dataclass）。
*   **`ScoreHead` (Class)**: 轻量级回归头，结构：`LayerNorm → Linear(H→H/4) → GELU → Linear(H/4→1)`。
*   **`ScoringActor` (Class)**:
    *   **Mismatch Mask**: 在 Forward 中，强制 Match 位置的 logit 为极大概率 (50.0)，确保 Ground Truth 必定被 Accept。
    *   **自动加载 Peft Adapter**: 支持从 `adapter_path` 加载 LoRA + ScoreHead 权重。
*   **`AutoModelForSPDScoring` (Class)**: 工厂类，模拟 `AutoModel` 接口，作为 verl 加载流程的适配器。
*   **辅助函数**: `create_hybrid_attention_mask`, `create_spd_attention_mask`, `create_position_ids` 等，用于构造 SPD 场景的 4D Attention Mask。

### 2. 执行层: `verl/workers/rollout/spd_rollout.py`

此文件实现了 SPD 专用的 Rollout 策略，**是整个 SPD Scorer 的核心执行模块**。

*   **`VllmEngineServer` (Class)**: vLLM HTTP 客户端，通过 REST API 调用 vLLM 服务器进行补全。
*   **`SPDRollout` (Class)**:
    *   继承自 `BaseRollout`。
    *   **多 vLLM 服务器负载均衡**: 根据 `CUDA_VISIBLE_DEVICES` 选择对应的 vLLM 服务器（从 `SPD_VLLM_URLS` 环境变量读取）。
    *   **`generate_sequences`**: 
        1. Forward 获取 Accept 概率
        2. 温度采样 (`SPD_SAMPLE_TEMPERATURE`) 控制探索强度
        3. Bernoulli 采样生成 Accept/Reject 序列
        4. 计算有效长度 L (cumprod)
        5. **Heavy Rollout**: 构造 Hybrid Context 并调用 vLLM 补全验证
        6. 将 `effective_len` 和 `is_correct_hybrid` 注入 `extra_info` 供 Reward 使用
    *   **L=0 特殊处理**: 退化为 Baseline，正确性等于 `is_correct_baseline`。

### 3. 评估层: `verl/utils/reward_score/spd_scorer_reward.py`

此文件实现了**轻量级** Reward 计算逻辑。

*   **设计说明**: 
    *   vLLM 补全验证在 Rollout 阶段完成（`spd_rollout.py`）
    *   Reward 函数只负责读取结果并应用奖励公式
*   **`compute_score` (Function)**:
    *   从 `extra_info` 读取 `effective_len` (L)、`is_correct_hybrid` (S_h)、`is_correct_baseline` (S_t)
    *   **L=0 时直接返回 0** (不参与学习)
    *   应用四场景奖励公式:
        - 场景A: `S_t * S_h * (alpha * L)` — 加速成功
        - 场景B: `S_t * (1-S_h) * penalty_break` — 破坏正确
        - 场景C: `(1-S_t) * (1-S_h) * reward_useless` — 无用尝试
        - 场景D: `(1-S_t) * S_h * (reward_correct_base + alpha * L)` — 纠正错误

### 4. 数据层: `verl/utils/dataset/spd_dataset.py`

自定义 Dataset，优化了数据加载流程。

*   **`SPDRLHFDataset` (Class)**:
    *   自动修正 Left Padding 带来的索引偏移 (同时修正 `extra_info` 和 `rollout_info` 中的索引)。
    *   跳过默认的 Chat Template 处理，直接使用预处理好的 `input_ids`。
    *   支持截断到 `max_prompt_length` (只保留最后 N 个 token)。

### 5. 训练入口: `train_spd_scorer.py`

负责数据准备和启动训练。

*   **`setup_loguru_rank0`**: 配置 loguru 只让 rank0 输出 DEBUG/INFO。
*   **Monkey Patch (关键 Hack)**:
    *   **被替换函数**: `verl.utils.model.create_huggingface_actor`
    *   **替换逻辑**: 拦截调用，返回 `AutoModelForSPDScoring.from_config(...)` 创建的 `ScoringActor`。
*   **`prepare_spd_data_from_real_source`**: 从 SPD 生成数据 + Metadata 构造训练数据。
    *   构造 `input_ids`: `[Context] + [SEP] + [Draft] + [SEP] + [Target] + [SEP]`
    *   跳过 `draft_ids == target_ids[:-1]` 的样本（无学习价值）
*   **`build_training_command`**: 构建 verl GRPO 训练命令（含所有超参数）。
*   **`_create_training_env`**: 生成训练环境变量（Reward 系数、Model 路径、SEP Token ID 等）。

### 6. 训练脚本: `scripts/train_spd_scorer.sh` & `scripts/train_spd_scorer_info.sh`

Shell 脚本，用于配置和启动训练。主要功能：

*   **环境清理**: 自动停止残留 Ray 进程，防止连接旧集群。
*   **环境变量配置**: 
    - `CUDA_VISIBLE_DEVICES`: GPU 可见性
    - `HF_HOME`, `HF_HUB_OFFLINE`: HuggingFace 缓存配置
    - `SPD_VLLM_URLS`: vLLM 服务器 URL 列表 (支持多服务器负载均衡)
*   **模型配置**: `MODEL_PATH`, `TOKENIZER_PATH`, `ADAPTER_PATH`, LoRA 参数
*   **数据配置**: 数据目录、训练数据文件、元数据文件路径
*   **训练超参数**: Batch Size, Rollout N, Epochs, PPO Mini Batch Size 等
*   **奖励配置**: `REWARD_ALPHA`, `REWARD_PENALTY_BREAK`, `REWARD_CORRECT_BASE`, `REWARD_USELESS`

区别: `train_spd_scorer.sh` 使用 `LOGURU_LEVEL=DEBUG`，`train_spd_scorer_info.sh` 使用 `LOGURU_LEVEL=INFO`。

### 7. 注册修改 (原有文件)

为了让 verl 识别上述自定义模块，对原有文件进行了少量修改：

*   **`verl/utils/reward_score/__init__.py`**: 在 `default_compute_score` 中增加了 `spd_scorer` 分支。
*   **`verl/workers/rollout/base.py`**: 注册了 `("spd", "sync")` Rollout。
*   **`verl/trainer/ppo/ray_trainer.py`**: 
    - 添加 Debug 日志：打印 Batch Size 变化
    - **关键修改**: `_get_gen_batch()` 方法移除了对 `extra_info` 的过滤，使其能传递给 Rollout 阶段（注释标记 `[SPD Fix]`）
*   **`verl/workers/actor/dp_actor.py`**: 添加 loguru 调试日志，输出 Actor 更新过程中的关键指标：
    - Micro Batch 信息: `response_mask`, `old_log_prob`, `advantages` 统计
    - 当前模型 `log_prob` 与 `old_log_prob` 差异 (策略偏移观察)
    - Policy Loss, Entropy Loss, KL Loss 等损失值
    - Gradient Norm 和最终训练指标汇总
*   **`verl/workers/reward_manager/batch.py`**: 修改 `BatchRewardManager.verify()` 方法，传递完整 Batch Tensor：
    ```python
    scores = self.compute_score(
        ...
        # 新增参数
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        attention_mask=attention_mask,
        batch_tensors=data.batch,  # 完整 batch 供 SPD Scorer 使用
        **self.reward_kwargs,
    )
    ```

---

## 🚀 训练流程总结

1.  **启动**: 运行 `bash scripts/train_spd_scorer.sh` (或 `train_spd_scorer_info.sh`)。
2.  **Patch**: 脚本首先应用 Monkey Patch，劫持模型加载逻辑。
3.  **加载**: verl Trainer 调用 `create_huggingface_actor`，被重定向到 `AutoModelForSPDScoring`，加载 `ScoringActor`。
4.  **数据**: `SPDRLHFDataset` 加载数据并修正索引。
5.  **Rollout**: `SPDRollout` 执行 Forward → Bernoulli 采样 → vLLM Hybrid 补全验证。
6.  **Reward**: `spd_scorer_reward` 基于有效长度和正确性计算奖励。
7.  **更新**: GRPO 更新模型参数 (Actor 更新过程有详细日志)。
