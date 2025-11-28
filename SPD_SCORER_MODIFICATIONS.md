# SPD Scorer 对 verl 库的修改记录

> **文档版本**: 1.2  
> **更新日期**: 2025-11-25  
> **目的**: 记录为实现 Speculative Decoding Scoring Model 对原始 verl 库所做的所有修改

---

## 📋 修改概览

| 类型 | 文件路径 | 说明 |
|------|----------|------|
| ✨ 新增 | `spd_scorer.py` | SPD Scorer 模型核心实现 |
| ✨ 新增 | `train_spd_scorer.py` | GRPO 训练脚本 (支持 API 补全) |
| ✨ 新增 | `verl/workers/rollout/spd_rollout.py` | 自定义 Rollout 策略 (Bernoulli 采样) |
| ✨ 新增 | `verl/utils/reward_score/spd_scorer_reward.py` | 自定义 Reward Function (支持 API 补全) |
| ✏️ 修改 | `verl/utils/reward_score/__init__.py` | 注册新的 Reward Function |
| ✏️ 修改 | `verl/workers/rollout/base.py` | 注册新的 Rollout 类 |

---

## 1. 核心架构更新：补全与验证

为了解决显存限制问题，我们采用 **Remote vLLM Completion** 架构：

1.  **Rollout Worker** (`spd_rollout.py`):
    - 只持有 **Scorer (Actor)** 模型。
    - 负责推理并生成 Accept/Reject 序列。
    - **不进行** Target Model 的补全生成 (避免 OOM)。

2.  **Reward Function** (`spd_scorer_reward.py`):
    - 负责奖励计算。
    - 通过 HTTP API 调用外部 **vLLM 服务** (持有 Target Model)。
    - 流程:
        1. 接收 Rollout 生成的决策序列。
        2. 构造 Hybrid Prefix = Context + Accepted Tokens。
        3. 调用 API 进行确定性补全 (`temperature=0`)。
        4. 验证补全结果是否包含 Ground Truth。

---

## 2. 新增文件

### 2.1 `spd_scorer.py` (根目录)

**位置**: `verl/spd_scorer.py`

**关键功能**:
- `ScoringActor`: 核心模型，Score Head，Mismatch Mask。
- `compute_reward_tensor`: 独立训练用的张量版本奖励函数。

### 2.2 `train_spd_scorer.py` (根目录)

**位置**: `verl/train_spd_scorer.py`

**更新**:
- 支持 `--target_model_url` 参数。
- 将 `context_text` 等关键信息通过 `extra_info` 传递给 Reward Function。

**使用方法**:
```bash
# 启动训练 (需要先启动一个 vLLM 服务作为 Target Model)
python train_spd_scorer.py \
    --model_path meta-llama/Llama-3-8B \
    --target_model_url http://localhost:8000/v1/completions \
    --n_gpus 8
```

### 2.3 `verl/workers/rollout/spd_rollout.py`

**位置**: `verl/workers/rollout/spd_rollout.py`

**说明**:
- 专用于 SPD 任务的 Rollout。
- 使用 `ScoringActor` 进行推理。
- 执行 Bernoulli 采样生成 N 个 0/1 序列。

### 2.4 `verl/utils/reward_score/spd_scorer_reward.py`

**位置**: `verl/utils/reward_score/spd_scorer_reward.py`

**关键更新**:
- 新增 `vllm_generate` 函数，封装 HTTP 请求。
- `verify_hybrid_correctness` 支持调用远程 API 进行补全验证。
- 支持本地 Tokenizer 缓存，用于编解码。

---

## 3. 修改的文件

### 3.1 `verl/utils/reward_score/__init__.py`

**修改**:
- 注册 `spd_scorer` data_source。
- 从 `extra_info` 中解包 `context_text`, `target_model_url`, `model_path` 等参数并传递给 `compute_score`。

### 3.2 `verl/workers/rollout/base.py`

**修改**:
- 注册 `("spd", "sync")` 到 `_ROLLOUT_REGISTRY`。

---

## 4. 训练流程详解

1.  **准备阶段**:
    - 启动 Target Model 的 vLLM 服务 (例如在另一组 GPU 上)。
    - 运行 `train_spd_scorer.py`。

2.  **Rollout 阶段**:
    - `SPDRollout` 使用 Scorer 生成 Accept/Reject 掩码。

3.  **Evaluation 阶段**:
    - `RewardManager` 调用 `spd_scorer_reward.py`。
    - 如果配置了 API URL，脚本将构造 Hybrid Prefix 并请求 vLLM 补全。
    - 验证补全结果，计算四场景奖励 (A/B/C/D)。

4.  **Update 阶段**:
    - GRPO 更新 Scorer 参数。

---

## 5. 依赖说明

- 需要安装 `requests`: `pip install requests`
- 需要安装 `vllm` (用于 Rollout 和 外部服务)

