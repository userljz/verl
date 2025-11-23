# Speculative Decoding 评分模型训练方案

## 1. 流程分析

### 1.1 你的场景特点

你的训练场景与 verl 的默认场景有显著差异：

**verl 默认场景**：
- 训练一个 LLM 生成 response
- Rollout：Actor 模型生成多个 response
- Reward：计算每个 response 的 reward
- 目标：最大化生成高质量 response 的概率

**你的场景**：
- 训练一个评分模型（Scoring Model），不生成 response
- Rollout：执行 speculative decoding 流程
  - Draft model 生成 drafter
  - Target model 验证
  - 评分模型对 mismatch 位置打分
  - 根据阈值决定接受/拒绝
- Reward：基于最终答案是否正确
- 目标：在保证答案正确的前提下，尽可能少地拒绝 mismatch

### 1.2 关键挑战

1. **Rollout 流程不同**：需要实现 speculative decoding，而不是简单的文本生成
2. **模型输出不同**：评分模型输出分数，而不是 token logits
3. **Reward 计算不同**：需要基于最终答案的正确性，而不是 token-level reward
4. **多模型协作**：需要 draft model、target model 和评分模型协同工作

## 2. 解决方案架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    verl 训练框架                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  Actor       │      │  Rollout     │                     │
│  │  (评分模型)   │◄─────┤  (自定义)     │                     │
│  │  LoRA+LMHead │      │  SPD流程     │                     │
│  └──────────────┘      └──────────────┘                     │
│         │                    │                               │
│         │                    │                               │
│         ▼                    ▼                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  Reward      │      │  Draft/Target│                     │
│  │  (自定义)     │      │  (冻结模型)   │                     │
│  └──────────────┘      └──────────────┘                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

1. **自定义评分模型**：继承 LLaMA3，修改 LM Head 输出分数
2. **自定义 Rollout**：实现 speculative decoding 流程
3. **自定义 Reward 函数**：计算最终答案正确性
4. **自定义 Trainer**：适配评分模型的训练流程

## 3. 实现方案

### 3.1 自定义评分模型

#### 3.1.1 模型定义

创建 `scoring_model.py`：

```python
from transformers import LlamaForCausalLM, LlamaConfig
import torch
import torch.nn as nn

class ScoringModel(LlamaForCausalLM):
    """
    评分模型：基于 LLaMA3，修改 LM Head 输出分数而不是 logits
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        # 替换 LM Head 为评分头
        # 原始 LM Head: hidden_size -> vocab_size
        # 评分头: hidden_size -> 1 (输出单个分数)
        self.scoring_head = nn.Linear(config.hidden_size, 1)
        
        # 冻结基础模型，只训练 LoRA 和 scoring_head
        for param in self.model.parameters():
            param.requires_grad = False
        
        # scoring_head 需要训练
        for param in self.scoring_head.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        前向传播，输出分数而不是 logits
        
        Returns:
            scores: (batch_size, seq_len, 1) 每个位置的分数
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        scores = self.scoring_head(hidden_states)  # (batch_size, seq_len, 1)
        
        # 返回格式兼容 verl
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            logits=scores.squeeze(-1),  # 为了兼容，但实际是分数
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
    
    def score_tokens(self, input_ids, attention_mask=None, position_ids=None):
        """
        对特定位置的 token 进行评分
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            position_ids: (batch_size, seq_len)
        
        Returns:
            scores: (batch_size, seq_len) 每个位置的分数
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        return outputs.logits  # 实际是分数
```

#### 3.1.2 LoRA 配置

使用 verl 的 LoRA 支持，配置如下：

```yaml
actor_rollout_ref:
  model:
    path: "path/to/llama3-8b"  # 基础模型
    lora_rank: 64
    lora_alpha: 32
    target_modules: "all-linear"
    # 需要加载自定义模型类
    trust_remote_code: True
```

### 3.2 自定义 Rollout：实现 Speculative Decoding

创建 `speculative_decoding_rollout.py`：

```python
from verl.workers.rollout.base import BaseRollout
from verl import DataProto
import torch
from typing import Optional

class SpeculativeDecodingRollout(BaseRollout):
    """
    实现 Speculative Decoding 流程的 Rollout
    """
    def __init__(
        self,
        config,
        model_config,
        device_mesh,
        draft_model,  # 冻结的 draft model
        target_model,  # 冻结的 target model
        scoring_model,  # 可训练的评分模型
        score_threshold: float = 0.5,  # 接受 mismatch 的阈值
    ):
        super().__init__(config, model_config, device_mesh)
        self.draft_model = draft_model
        self.target_model = target_model
        self.scoring_model = scoring_model
        self.score_threshold = score_threshold
        
        # 冻结 draft 和 target 模型
        for param in draft_model.parameters():
            param.requires_grad = False
        for param in target_model.parameters():
            param.requires_grad = False
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        执行 Speculative Decoding 流程
        
        流程：
        1. Draft model 生成 drafter
        2. Target model 验证（传统 SPD）
        3. 对于 mismatch 位置，使用评分模型打分
        4. 如果分数 > 阈值，接受；否则拒绝
        5. 返回最终生成的序列
        """
        batch_size = prompts.batch["input_ids"].shape[0]
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch.get("attention_mask")
        
        # 1. Draft model 生成 drafter
        with torch.no_grad():
            draft_outputs = self.draft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.response_length,
                do_sample=True,
                temperature=self.config.temperature,
            )
        
        draft_tokens = draft_outputs[:, input_ids.shape[1]:]  # 只取新生成的 tokens
        
        # 2. Target model 验证（传统 SPD 流程）
        # 这里简化处理，实际需要逐 token 验证
        with torch.no_grad():
            # 构建完整序列用于验证
            full_sequence = torch.cat([input_ids, draft_tokens], dim=1)
            
            # Target model 前向传播
            target_outputs = self.target_model(
                input_ids=full_sequence,
                attention_mask=torch.ones_like(full_sequence),
            )
            
            # 获取 target model 的 logits 和预测
            target_logits = target_outputs.logits
            target_predictions = torch.argmax(target_logits, dim=-1)
        
        # 3. 找出 mismatch 位置
        # draft_tokens 和 target_predictions 在对应位置比较
        # 简化：假设逐位置比较
        mismatch_positions = []
        accepted_tokens = []
        
        # 构建用于评分的输入（包含上下文和 mismatch 位置）
        for batch_idx in range(batch_size):
            batch_mismatches = []
            batch_accepted = []
            
            # 逐 token 处理
            current_sequence = input_ids[batch_idx:batch_idx+1]
            
            for pos in range(draft_tokens.shape[1]):
                draft_token = draft_tokens[batch_idx, pos:pos+1]
                target_token = target_predictions[batch_idx, current_sequence.shape[1]-1:current_sequence.shape[1]]
                
                if draft_token.item() != target_token.item():
                    # Mismatch！使用评分模型打分
                    # 构建评分输入：prompt + 已接受的 tokens + mismatch token
                    scoring_input = torch.cat([current_sequence, draft_token.unsqueeze(0)], dim=1)
                    
                    # 评分模型打分
                    scores = self.scoring_model.score_tokens(
                        input_ids=scoring_input,
                        attention_mask=torch.ones_like(scoring_input),
                    )
                    
                    # 获取最后一个位置的分数（mismatch token 的分数）
                    token_score = scores[0, -1].item()
                    
                    if token_score > self.score_threshold:
                        # 接受 mismatch
                        batch_accepted.append(draft_token.item())
                        current_sequence = torch.cat([current_sequence, draft_token.unsqueeze(0)], dim=1)
                    else:
                        # 拒绝，使用 target model 的 token
                        batch_accepted.append(target_token.item())
                        current_sequence = torch.cat([current_sequence, target_token.unsqueeze(0)], dim=1)
                else:
                    # Match，直接接受
                    batch_accepted.append(draft_token.item())
                    current_sequence = torch.cat([current_sequence, draft_token.unsqueeze(0)], dim=1)
            
            accepted_tokens.append(batch_accepted)
        
        # 4. 构建返回的 DataProto
        # 将 accepted_tokens 转换为 tensor
        max_len = max(len(tokens) for tokens in accepted_tokens)
        padded_tokens = []
        for tokens in accepted_tokens:
            padded = tokens + [0] * (max_len - len(tokens))  # 用 0 padding
            padded_tokens.append(padded)
        
        response_tokens = torch.tensor(padded_tokens, device=input_ids.device)
        
        # 构建完整的序列
        full_sequence = torch.cat([input_ids, response_tokens], dim=1)
        
        # 返回 DataProto
        output = DataProto.from_dict(
            tensors={
                "input_ids": full_sequence,
                "attention_mask": torch.ones_like(full_sequence),
                "responses": response_tokens,
            },
            non_tensors={
                "prompt": prompts.non_tensor_batch.get("prompt", []),
            }
        )
        
        return output
    
    async def resume(self, tags: list[str]):
        """Resume rollout weights"""
        pass
    
    async def update_weights(self, weights, **kwargs):
        """Update scoring model weights"""
        # 只更新评分模型的权重
        for name, weight in weights:
            if name.startswith("scoring_head") or name.startswith("lora"):
                # 更新评分模型的权重
                # 这里需要根据实际权重同步机制实现
                pass
    
    async def release(self):
        """Release resources"""
        pass
```

### 3.3 自定义 Reward 函数

创建 `spd_reward.py`：

```python
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl import DataProto
import torch
from typing import Optional

def compute_spd_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """
    计算 Speculative Decoding 场景下的 reward
    
    Args:
        data_source: 数据源名称
        solution_str: 模型生成的最终答案
        ground_truth: 正确答案
    
    Returns:
        reward: 如果答案正确返回 1.0，否则返回 0.0
        可以扩展为更细粒度的评分
    """
    # 简单的字符串匹配（可以根据任务调整）
    solution_clean = solution_str.strip().lower()
    ground_truth_clean = ground_truth.strip().lower()
    
    if solution_clean == ground_truth_clean:
        return 1.0
    else:
        # 可以添加更复杂的匹配逻辑
        # 例如：部分匹配、数值提取等
        return 0.0

class SPDRewardManager(AbstractRewardManager):
    """
    Speculative Decoding 专用的 Reward Manager
    """
    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,
        reward_fn_key: str = "ground_truth",
        **kwargs
    ):
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key)
        self.compute_score = compute_score or compute_spd_reward
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        计算 reward
        
        Args:
            data: DataProto 包含生成的 responses 和 ground_truth
            return_dict: 是否返回字典格式
        
        Returns:
            reward_tensor: (batch_size, response_length) token-level rewards
        """
        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]
        
        # 解码 responses
        responses_str = self.tokenizer.batch_decode(
            data.batch["responses"],
            skip_special_tokens=True
        )
        
        # 获取 ground_truth
        ground_truths = data.non_tensor_batch.get(self.reward_fn_key, [])
        
        # 计算每个 response 的 reward
        rewards = []
        for i in range(batch_size):
            response_str = responses_str[i]
            ground_truth = ground_truths[i] if i < len(ground_truths) else ""
            
            # 计算最终答案的 reward
            final_reward = self.compute_score(
                data_source=data.non_tensor_batch.get("data_source", [""])[i] if "data_source" in data.non_tensor_batch else "",
                solution_str=response_str,
                ground_truth=ground_truth,
            )
            
            # 将 reward 分配到所有 token（sparse reward）
            # 只在最后一个 token 或 EOS token 位置给予 reward
            token_rewards = torch.zeros(response_length)
            # 找到 EOS token 的位置
            eos_positions = (data.batch["responses"][i] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                # 在最后一个 EOS token 位置给予 reward
                token_rewards[eos_positions[-1]] = final_reward
            else:
                # 如果没有 EOS，在最后一个位置给予 reward
                token_rewards[-1] = final_reward
            
            rewards.append(token_rewards)
        
        reward_tensor = torch.stack(rewards).to(data.batch["responses"].device)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {}
            }
        return reward_tensor
```

### 3.4 自定义 Trainer

创建 `spd_trainer.py`：

```python
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl import DataProto

class SPDRayTrainer(RayPPOTrainer):
    """
    适配 Speculative Decoding 场景的 Trainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 可以添加 SPD 特定的初始化逻辑
    
    def fit(self):
        """
        重写 fit 方法以适配 SPD 流程
        主要差异：
        1. Rollout 使用自定义的 SpeculativeDecodingRollout
        2. Reward 计算基于最终答案正确性
        3. Actor 更新时需要考虑评分模型的特殊性
        """
        # 大部分逻辑可以复用父类
        # 主要需要确保：
        # 1. Rollout 使用自定义实现
        # 2. Reward 函数正确计算
        # 3. Actor 更新时只更新评分模型的参数
        
        return super().fit()
```

## 4. 配置示例

### 4.1 训练配置

创建 `spd_train_config.yaml`：

```yaml
algorithm:
  adv_estimator: grpo  # 使用 GRPO
  gamma: 1.0
  lam: 1.0
  norm_adv_by_std_in_grpo: true

data:
  train_files: "path/to/train.parquet"
  val_files: "path/to/val.parquet"
  train_batch_size: 16
  max_prompt_length: 512
  max_response_length: 1024
  reward_fn_key: "ground_truth"

actor_rollout_ref:
  model:
    path: "meta-llama/Meta-Llama-3-8B"
    lora_rank: 64
    lora_alpha: 32
    target_modules: "all-linear"
    trust_remote_code: true
    # 自定义模型类路径
    model_class: "scoring_model.ScoringModel"
  
  actor:
    optim:
      lr: 3e-6  # LoRA 训练建议提高学习率
    ppo_mini_batch_size: 16
    use_kl_loss: true
    kl_loss_coef: 0.001
  
  rollout:
    name: "speculative_decoding"  # 自定义 rollout
    n: 5  # 每个 prompt 采样 5 次
    response_length: 1024
    temperature: 1.0
    # SPD 特定参数
    score_threshold: 0.5
    draft_model_path: "path/to/draft-model"
    target_model_path: "path/to/target-model"

reward_model:
  reward_manager: "spd"  # 自定义 reward manager
  custom_reward_function:
    path: "spd_reward.py"
    name: "compute_spd_reward"

trainer:
  n_gpus_per_node: 4
  total_epochs: 10
```

### 4.2 训练命令

```bash
python3 -m verl.trainer.main_ppo \
    --config-path spd_train_config.yaml \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.rollout.load_format=safetensors \
    data.train_files=/path/to/train.parquet \
    trainer.n_gpus_per_node=4
```

## 5. 关键实现细节

### 5.1 评分模型的 forward 兼容性

评分模型需要返回类似 `CausalLMOutputWithPast` 的格式，但 `logits` 字段实际是分数。在 Actor 的 `compute_log_prob` 中需要特殊处理：

```python
# 在 Actor 的 compute_log_prob 中
# 评分模型返回的 "logits" 实际是分数
# 需要转换为 log_probs 格式
scores = output.logits  # (batch_size, seq_len)
# 将分数转换为 log_probs（可能需要 sigmoid 或其他变换）
log_probs = torch.log(torch.sigmoid(scores) + 1e-8)
```

### 5.2 Rollout 中的模型加载

需要加载三个模型：
1. Draft model（冻结）
2. Target model（冻结）
3. Scoring model（可训练，带 LoRA）

### 5.3 Reward 的稀疏性

由于 reward 只在最终答案正确时给予，是稀疏的。可以考虑：
1. **Sparse Reward**：只在最后一个 token 或 EOS token 位置给予 reward
2. **Dense Reward**：如果能够评估中间步骤的正确性，可以给予中间 reward
3. **Reward Shaping**：添加辅助 reward 信号帮助训练

### 5.4 阈值调整策略

`score_threshold` 可以：
1. **固定阈值**：设置为固定值（如 0.5）
2. **自适应阈值**：根据训练进度动态调整
3. **可学习阈值**：将阈值作为可学习参数

## 6. 训练流程总结

```
1. 初始化阶段
   ├── 加载 Draft Model（冻结）
   ├── 加载 Target Model（冻结）
   └── 加载 Scoring Model（LoRA + 修改的 LM Head）

2. 训练循环（每个 epoch）
   └── 对每个 batch：
       ├── Rollout 阶段
       │   ├── Draft Model 生成 drafter
       │   ├── Target Model 验证
       │   ├── 找出 mismatch 位置
       │   ├── Scoring Model 对 mismatch 打分
       │   └── 根据阈值决定接受/拒绝
       │
       ├── Reward 计算阶段
       │   ├── 解码最终生成的序列
       │   ├── 与 ground_truth 比较
       │   └── 计算 reward（正确=1.0，错误=0.0）
       │
       ├── Advantage 计算阶段（GRPO）
       │   ├── 按 prompt 分组
       │   ├── 计算组内平均 reward
       │   └── advantage = reward - mean_reward
       │
       └── Actor 更新阶段
           ├── 计算 policy loss
           ├── 计算 KL loss（可选）
           └── 更新 Scoring Model 的参数（LoRA + LM Head）
```

## 7. 注意事项

### 7.1 模型兼容性

- 确保评分模型继承自 `PreTrainedModelForCausalLM`
- `forward` 方法返回格式需要兼容 verl 的接口
- 可能需要实现 `dtensor_weight_loader`（如果使用 FSDP+vLLM）

### 7.2 内存管理

- Draft 和 Target 模型可以共享内存（如果相同）
- 考虑使用模型量化减少内存占用
- 使用 `layered_summon=True` 优化大模型加载

### 7.3 训练稳定性

- 评分模型的输出范围可能需要归一化（如 sigmoid）
- 考虑添加正则化项防止过拟合
- 监控训练过程中的指标（acceptance rate, accuracy 等）

### 7.4 调试建议

1. **先实现简化版本**：先实现基本的 SPD 流程，再逐步完善
2. **单元测试**：分别测试各个组件（rollout, reward, model）
3. **可视化**：记录和可视化训练过程中的关键指标
4. **小规模实验**：先用小模型和小数据集验证流程

## 8. 扩展方向

1. **多步 Speculative Decoding**：支持多步 lookahead
2. **动态阈值**：根据上下文动态调整接受阈值
3. **更细粒度的 Reward**：不仅考虑最终答案，还考虑中间步骤
4. **集成学习**：训练多个评分模型并集成

## 9. 参考资源

- verl 文档：`docs/advance/ppo_lora.rst`
- GRPO 论文和实现
- Speculative Decoding 相关论文
- verl 示例：`examples/grpo_trainer/`

