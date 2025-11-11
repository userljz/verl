# verl库GRPO算法训练完整教程

本教程面向从未使用过verl库但了解GRPO算法和Python基础的工程师，将详细讲解如何使用verl库进行GRPO训练。

## 目录

1. [配置文件位置与自定义](#1-配置文件位置与自定义)
2. [启动命令与入口文件](#2-启动命令与入口文件)
3. [GRPO训练相关代码文件](#3-grpo训练相关代码文件)
4. [完整训练流程解析](#4-完整训练流程解析)

---

## 1. 配置文件位置与自定义

### 1.1 配置文件结构

verl使用Hydra框架管理配置，配置文件采用YAML格式，采用分层结构：

```
verl/trainer/config/
├── ppo_trainer.yaml          # 主配置文件（GRPO也使用这个）
├── actor/                     # Actor相关配置
│   ├── dp_actor.yaml         # 数据并行Actor配置
│   └── actor.yaml            # Actor基础配置
├── data/                      # 数据相关配置
│   └── legacy_data.yaml
├── rollout/                   # Rollout生成配置
│   └── rollout.yaml
├── ref/                       # Reference模型配置
│   └── dp_ref.yaml
└── algorithm/                 # 算法相关配置
    └── rollout_correction.yaml
```

### 1.2 主配置文件位置

**主配置文件**：`verl/trainer/config/ppo_trainer.yaml`

这是GRPO训练的核心配置文件。虽然文件名是`ppo_trainer.yaml`，但GRPO和PPO使用相同的训练框架，只是通过`algorithm.adv_estimator`参数来区分。

### 1.3 如何自定义配置

#### 方法1：命令行参数覆盖（推荐）

在启动命令中直接覆盖配置参数，这是最常用的方式：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \                    # 设置使用GRPO算法
    data.train_files=/path/to/train.parquet \        # 训练数据路径
    actor_rollout_ref.rollout.n=5 \                  # 每组采样5次
    actor_rollout_ref.actor.ppo_mini_batch_size=256  # mini batch大小
```

#### 方法2：创建自定义配置文件

1. **复制主配置文件**：
```bash
cp verl/trainer/config/ppo_trainer.yaml my_grpo_config.yaml
```

2. **修改配置文件**，例如：
```yaml
algorithm:
  adv_estimator: grpo  # 设置为grpo
  
actor_rollout_ref:
  rollout:
    n: 5  # 每组采样5次
  actor:
    ppo_mini_batch_size: 256
    use_kl_loss: True
    kl_loss_coef: 0.001
```

3. **使用自定义配置**：
```bash
python3 -m verl.trainer.main_ppo \
    --config-path=/path/to/config/dir \
    --config-name=my_grpo_config
```

#### 方法3：使用config groups

verl支持config groups，可以在`verl/trainer/config`下创建新的配置组，然后在主配置中引用。

### 1.4 GRPO关键配置参数说明

#### 算法相关配置（`algorithm`部分）

```yaml
algorithm:
  adv_estimator: grpo                    # 必须设置为"grpo"
  norm_adv_by_std_in_grpo: True         # GRPO中是否用标准差归一化优势
  use_kl_in_reward: False                # GRPO中KL惩罚在loss中，不在reward中
```

#### Actor配置（`actor_rollout_ref.actor`部分）

```yaml
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 256             # PPO更新时的mini batch大小
    ppo_epochs: 1                         # 对同一批数据训练的轮数
    clip_ratio: 0.2                       # PPO clip范围
    use_kl_loss: True                     # 使用KL loss（GRPO必须）
    kl_loss_coef: 0.001                   # KL loss系数
    kl_loss_type: low_var_kl              # KL loss类型
    loss_agg_mode: token-mean             # 损失聚合模式
```

#### Rollout配置（`actor_rollout_ref.rollout`部分）

```yaml
actor_rollout_ref:
  rollout:
    n: 5                                  # 每个prompt采样n次（GRPO的关键）
    name: vllm                            # 使用vllm进行推理
    tensor_model_parallel_size: 2         # 张量并行大小
    gpu_memory_utilization: 0.6           # GPU内存利用率
```

#### 数据配置（`data`部分）

```yaml
data:
  train_files: /path/to/train.parquet    # 训练数据路径
  val_files: /path/to/val.parquet        # 验证数据路径
  train_batch_size: 1024                 # 全局batch size（prompt数量）
  max_prompt_length: 512                 # 最大prompt长度
  max_response_length: 1024               # 最大response长度
```

**注意**：实际生成的response数量 = `train_batch_size × rollout.n`

---

## 2. 启动命令与入口文件

### 2.1 入口文件

**入口文件**：`verl/trainer/main_ppo.py`

这是GRPO训练的入口点。虽然文件名包含"ppo"，但它同时支持PPO、GRPO等多种算法。

入口函数：
```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management."""
    run_ppo(config)
```

### 2.2 基本启动命令

#### 最简单的GRPO启动命令

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/path/to/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.rollout.n=5
```

#### 完整示例（参考`examples/grpo_trainer/run_qwen3-8b.sh`）

```bash
python3 -m verl.trainer.main_ppo \
    # 算法配置
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    
    # 数据配置
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    
    # 模型配置
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.model.use_remove_padding=True \
    
    # Actor训练配置
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    
    # Rollout配置（GRPO关键）
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    
    # Trainer配置
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.project_name='verl_grpo_example' \
    trainer.experiment_name='qwen3_8b_grpo'
```

### 2.3 多节点分布式训练

```bash
# 在每个节点上运行
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.nnodes=4 \
    trainer.n_gpus_per_node=8 \
    # ... 其他配置
```

### 2.4 使用自定义配置文件

```bash
python3 -m verl.trainer.main_ppo \
    --config-path=/path/to/config/dir \
    --config-name=my_grpo_config \
    # 仍然可以通过命令行覆盖参数
    data.train_files=/path/to/train.parquet
```

---

## 3. GRPO训练相关代码文件

### 3.1 核心代码文件结构

```
verl/
├── trainer/
│   ├── main_ppo.py                    # 入口文件
│   ├── ppo/
│   │   ├── ray_trainer.py            # Ray分布式训练器（核心训练循环）
│   │   └── core_algos.py             # GRPO算法实现
│   ├── config/
│   │   ├── ppo_trainer.yaml          # 主配置文件
│   │   └── algorithm.py              # 算法配置类定义
│   └── constants_ppo.py              # PPO相关常量
├── workers/
│   ├── fsdp_workers.py               # FSDP worker实现
│   └── engine/                        # 训练引擎
└── protocol.py                        # 数据传输协议定义
```

### 3.2 关键代码文件详解

#### 3.2.1 入口文件：`verl/trainer/main_ppo.py`

**作用**：程序入口，初始化Ray集群，创建TaskRunner

**关键代码**：
```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)

def run_ppo(config, task_runner_class=None):
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(**ray_init_kwargs)
    
    # 创建TaskRunner并运行
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))
```

#### 3.2.2 训练器：`verl/trainer/ppo/ray_trainer.py`

**作用**：实现GRPO训练的核心循环逻辑

**关键类**：`RayPPOTrainer`

**关键方法**：
- `fit()`: 主训练循环
- `compute_advantage()`: 计算优势值的函数（调用GRPO算法，定义在模块级别）

**训练流程**（在`fit()`方法中）：
```python
def fit(self):
    # 1. 初始化logger和checkpoint
    # 2. 验证（可选）
    
    for epoch in range(total_epochs):
        for batch in train_dataloader:
            # 3. 生成rollout（采样多个response）
            gen_batch = self._get_gen_batch(batch)  # 准备生成用的batch
            # 先repeat batch，使每个prompt重复n次
            gen_batch_output = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, 
                interleave=True
            )
            # 然后生成sequences
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
            
            # 4. 计算reward
            rewards = self.reward_fn(...)
            
            # 5. 计算advantage（GRPO核心）
            advantages = compute_advantage(
                adv_estimator=AdvantageEstimator.GRPO,
                ...
            )
            
            # 6. 更新Actor
            actor_output = self.actor_rollout_wg.update_actor(batch)
            # 内部会调用 worker.update_actor() -> actor.update_policy()
            # 在update_policy中会进行mini-batch循环和PPO更新
```

#### 3.2.3 GRPO算法实现：`verl/trainer/ppo/core_algos.py`

**作用**：实现GRPO的优势估计函数

**关键函数**：`compute_grpo_outcome_advantage()`

**位置**：`verl/trainer/ppo/core_algos.py` 第264-330行

**算法逻辑**：
```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,  # 用于分组的索引
    norm_adv_by_std_in_grpo: bool = True,
    ...
):
    """
    GRPO算法核心：
    1. 计算每个response的总reward
    2. 按index分组（同一个prompt的多个response为一组）
    3. 计算组内平均reward作为baseline
    4. advantage = reward - baseline
    5. 可选：用标准差归一化
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    
    # 按组计算均值和标准差
    for group_id in groups:
        group_scores = scores[group_id]
        mean = group_scores.mean()
        std = group_scores.std()
        
        # 计算advantage
        advantages = group_scores - mean
        if norm_adv_by_std_in_grpo:
            advantages = advantages / (std + epsilon)
    
    return advantages, advantages  # returns = advantages for GRPO
```

**调用位置**：在`ray_trainer.py`的`compute_advantage()`函数中

```python
elif adv_estimator == AdvantageEstimator.GRPO:
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=grpo_calculation_mask,
        index=data.non_tensor_batch["uid"],  # 分组索引
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
```

#### 3.2.4 Worker实现：`verl/workers/fsdp_workers.py`

**作用**：实现FSDP训练和rollout生成的worker

**关键类**：
- `ActorRolloutRefWorker`: 执行Actor训练和rollout生成
- `AsyncActorRolloutRefWorker`: 异步版本

**关键方法**：
- `generate_sequences()`: 生成rollout（采样response）
- `update_actor()`: 更新Actor模型参数
- `compute_log_probs()`: 计算log概率（用于KL loss）

#### 3.2.5 配置文件：`verl/trainer/config/ppo_trainer.yaml`

**作用**：定义所有可配置参数及其默认值

**关键部分**：
- `algorithm`: 算法配置（包括`adv_estimator`）
- `actor_rollout_ref`: Actor、Rollout、Reference模型配置
- `data`: 数据配置
- `trainer`: 训练器配置

#### 3.2.6 数据传输协议：`verl/protocol.py`

**作用**：定义训练过程中数据传输的格式

**关键类**：`DataProto`

用于在训练循环的不同阶段（rollout生成、reward计算、advantage计算、模型更新）之间传递数据。

---

## 4. 完整训练流程解析

### 4.1 GRPO训练流程图

```
开始
  ↓
初始化Ray集群
  ↓
加载配置（ppo_trainer.yaml + 命令行参数）
  ↓
创建Workers（Actor、Rollout、Reference）
  ↓
加载模型和checkpoint
  ↓
┌─────────────────────────────────────┐
│ 训练循环（每个epoch）                │
│                                     │
│  for epoch in range(total_epochs):  │
│      for batch in dataloader:      │
│          ↓                          │
│      [1] Rollout生成阶段            │
│          - 对每个prompt采样n次      │
│          - 生成n个response          │
│          ↓                          │
│      [2] Reward计算阶段             │
│          - 调用reward_function      │
│          - 计算每个response的reward │
│          ↓                          │
│      [3] Advantage计算阶段（GRPO）  │
│          - 按prompt分组             │
│          - 计算组内平均reward       │
│          - advantage = reward - mean│
│          - 可选：标准差归一化       │
│          ↓                          │
│      [4] Actor更新阶段              │
│          - 将数据分成mini batches   │
│          - 对每个mini batch:        │
│            * 计算policy loss        │
│            * 计算KL loss            │
│            * 反向传播               │
│            * 更新参数               │
│          ↓                          │
│      [5] 日志记录和验证             │
│                                     │
└─────────────────────────────────────┘
  ↓
保存checkpoint
  ↓
结束
```

### 4.2 详细代码执行路径

#### 步骤1：程序启动

```python
# verl/trainer/main_ppo.py
main(config)
  → run_ppo(config)
    → ray.init()  # 初始化Ray
    → TaskRunner.remote().run(config)
```

#### 步骤2：初始化训练器

```python
# verl/trainer/main_ppo.py -> TaskRunner.run()
  → RayPPOTrainer(config, ...)
    → 创建各种Workers
    → 加载模型
    → 初始化优化器
```

#### 步骤3：训练循环开始

```python
# verl/trainer/ppo/ray_trainer.py -> RayPPOTrainer.fit()
for epoch in range(total_epochs):
    for batch in train_dataloader:
        # 3.1 生成rollout
        gen_batch = self._get_gen_batch(batch)  # 准备生成用的batch
        # 先repeat batch，使每个prompt重复n次（GRPO关键：n>1）
        gen_batch_output = gen_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n, 
            interleave=True
        )
        # 然后生成sequences，此时会为每个prompt生成n个response
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
        # 此时有 n * batch_size 个response
        
        # 3.2 计算reward
        rewards = self.reward_fn(gen_batch_output)
        
        # 3.3 计算advantage（GRPO核心）
        data = compute_advantage(
            data,
            adv_estimator=AdvantageEstimator.GRPO,  # 使用GRPO
            ...
        )
        # 内部调用：core_algos.compute_grpo_outcome_advantage()
        
        # 3.4 更新Actor
        actor_output = self.actor_rollout_wg.update_actor(batch)
        # 内部流程：
        #   - verl/workers/fsdp_workers.py -> ActorRolloutRefWorker.update_actor()
        #     - 调用 self.actor.update_policy(data)
        #   - verl/workers/actor/dp_actor.py -> DataParallelPPOActor.update_policy()
        #     - 分成mini batches
        #     - 对每个mini batch计算policy loss + KL loss
        #     - 反向传播和参数更新
```

#### 步骤4：GRPO优势计算（核心）

```python
# verl/trainer/ppo/core_algos.py -> compute_grpo_outcome_advantage()

# 输入：
# - token_level_rewards: (batch_size * n, seq_len)
# - index: (batch_size * n,)  # 用于分组，相同prompt的n个response有相同的index

# 步骤：
scores = token_level_rewards.sum(dim=-1)  # (batch_size * n,)

# 按index分组
for group_id in unique_indices:
    group_mask = (index == group_id)
    group_scores = scores[group_mask]  # (n,) - 同一prompt的n个response
    
    # 计算组内统计量
    mean = group_scores.mean()
    std = group_scores.std()
    
    # 计算advantage
    group_advantages = group_scores - mean
    if norm_adv_by_std_in_grpo:
        group_advantages = group_advantages / (std + epsilon)
    
    advantages[group_mask] = group_advantages

return advantages, advantages  # returns = advantages for GRPO
```

#### 步骤5：Actor更新

**调用路径**：
1. `ray_trainer.py` (第1206行) → `self.actor_rollout_wg.update_actor(batch)`
2. `verl/workers/fsdp_workers.py` (第865行) → `ActorRolloutRefWorker.update_actor()`
3. `verl/workers/actor/dp_actor.py` (第359行) → `DataParallelPPOActor.update_policy()`

**核心逻辑**（在`update_policy`中）：
```python
# verl/workers/actor/dp_actor.py -> DataParallelPPOActor.update_policy()

# 1. 分成mini batches
mini_batches = data.split(self.config.ppo_mini_batch_size)

# 2. 对每个epoch和mini batch
for epoch in range(self.config.ppo_epochs):
    for mini_batch in mini_batches:
        # 3. 前向传播获取log_probs
        log_prob = self._forward_micro_batch(mini_batch)
        
        # 4. 计算policy loss
        ratio = exp(log_prob - old_log_prob)
        policy_loss = -min(
            ratio * advantages, 
            clip(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
        )
        
        # 5. 计算KL loss（GRPO特有，如果use_kl_loss=True）
        if self.config.use_kl_loss:
            kl_loss = compute_kl_divergence(
                log_prob, 
                ref_log_prob,
                type=self.config.kl_loss_type
            )
            total_loss = policy_loss.mean() + self.config.kl_loss_coef * kl_loss.mean()
        else:
            total_loss = policy_loss.mean()
        
        # 6. 反向传播和更新
        total_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
```

### 4.3 关键数据结构

#### DataProto

训练过程中传递的数据结构，包含：

```python
class DataProto:
    batch: dict[str, torch.Tensor]  # 张量数据
        - "input_ids": (bs, seq_len)
        - "attention_mask": (bs, seq_len)
        - "response_mask": (bs, seq_len)  # response部分的mask
        - "token_level_rewards": (bs, seq_len)
        - "advantages": (bs, seq_len)
        - "log_probs": (bs, seq_len)
        - "ref_log_probs": (bs, seq_len)
    
    non_tensor_batch: dict[str, Any]  # 非张量数据
        - "uid": (bs,)  # 用于GRPO分组
        - "prompt": List[str]
        - "response": List[str]
```

### 4.4 GRPO与PPO的区别

| 特性 | PPO | GRPO |
|------|-----|------|
| Critic模型 | 需要 | 不需要 |
| 优势估计 | GAE（需要value函数） | 组内相对优势 |
| 采样方式 | 每个prompt采样1次 | 每个prompt采样n次（n>1） |
| KL惩罚 | 可在reward中 | 在loss中 |
| 配置参数 | `adv_estimator=gae` | `adv_estimator=grpo` |

---

## 5. 常见问题与调试

### 5.1 如何确认使用的是GRPO？

检查配置中：
```yaml
algorithm:
  adv_estimator: grpo  # 必须是"grpo"
```

检查日志中应该看到：
```
Using advantage estimator: grpo
```

### 5.2 如何调试GRPO训练？

1. **检查分组是否正确**：
   - 确认`rollout.n > 1`
   - 检查`data.non_tensor_batch["uid"]`是否正确分组

2. **检查advantage计算**：
   - 在`core_algos.py`的`compute_grpo_outcome_advantage()`中添加print
   - 检查组内均值和标准差是否合理

3. **检查loss**：
   - 确认`use_kl_loss=True`
   - 检查policy loss和KL loss的比例

### 5.3 性能优化建议

1. **调整`rollout.n`**：更大的n提供更稳定的baseline，但增加计算成本
2. **使用`norm_adv_by_std_in_grpo=False`**：可以加速（DrGRPO变体）
3. **调整`ppo_mini_batch_size`**：平衡内存和训练稳定性

---

## 6. 参考资源

- **官方文档**：https://verl.readthedocs.io/en/latest/algo/grpo.html
- **示例脚本**：`examples/grpo_trainer/run_qwen3-8b.sh`
- **GRPO论文**：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **配置文件**：`verl/trainer/config/ppo_trainer.yaml`

---

## 总结

1. **配置文件**：主要在`verl/trainer/config/ppo_trainer.yaml`，可通过命令行参数覆盖
2. **启动命令**：`python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...`
3. **核心代码**：
   - 入口：`verl/trainer/main_ppo.py`
   - 训练循环：`verl/trainer/ppo/ray_trainer.py`
   - GRPO算法：`verl/trainer/ppo/core_algos.py`中的`compute_grpo_outcome_advantage()`

希望这个教程能帮助你快速上手verl的GRPO训练！

