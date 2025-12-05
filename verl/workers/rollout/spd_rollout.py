# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SPD Rollout: Speculative Decoding Scorer 的专用 Rollout 实现

核心区别:
1. 输入是 [Context] + [Draft] + [Target]
2. 不需要自回归生成 (Auto-regressive Generation)
3. 只需要对 Draft 部分进行一次 Forward，得到 logits
4. 基于 logits 进行 N 次 Bernoulli 采样，得到 Accept/Reject 序列
"""

import torch
import torch.distributed
import os
from typing import Generator, List

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.workers.config import RolloutConfig

from spd_scorer import ScoringActor, ScoringModelConfig

class SPDRollout(BaseRollout):
    def __init__(self, config: RolloutConfig, model_config, device_mesh):
        super().__init__(config, model_config, device_mesh)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化 Scoring Model
        # 注意: 这里我们需要一个能够加载权重的 ScoringActor
        # 在 verl 架构中，Actor 和 Rollout 可能共享权重
        self.model_config = model_config
        
        # 创建模型实例 (这里假设模型结构与 spd_scorer.py 中定义的一致)
        # 实际运行时，权重会通过 update_weights 加载
        
        # 从环境变量获取 SPD 特定配置
        lora_rank = int(os.getenv("SPD_LORA_RANK", "16"))
        lora_alpha = int(os.getenv("SPD_LORA_ALPHA", "32"))
        # 获取 SEP token ID (如果是 "eot" 或其他非数字字符串，后续逻辑可能需要调整，这里假设传进来的是数字字符串或 eot)
        # 注意: ScoringModelConfig 期望 sep_token_id 是 int
        sep_token_id_str = os.getenv("SPD_SEP_TOKEN_ID", "128009")
        sep_token_id = int(sep_token_id_str)
        
        spd_config = ScoringModelConfig(
            model_name_or_path=model_config.path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            sep_token_id=sep_token_id
        )
        self.model = ScoringActor(spd_config)
        self.model.to(self.device)
        self.model.eval()
        
        # 获取 SEP token ID
        self.sep_token_id = spd_config.sep_token_id

    async def resume(self, tags: list[str]):
        """无状态，不需要 resume"""
        pass

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """更新模型权重"""
        state_dict = {}
        for name, param in weights:
            state_dict[name] = param
        
        # 加载权重
        # 注意处理 LoRA 和 Score Head 的权重名称匹配
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        # 只需要关注 Score Head 和 LoRA 参数是否加载成功
        
    async def release(self):
        """释放资源"""
        pass

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        执行 SPD Rollout
        
        逻辑:
        1. 获取输入 input_ids (包含 Context+Draft+Target)
        2. 解析 input_ids 得到 Draft/Target 并计算 Match Mask
        3. 模型 Forward 得到 logits
        4. 采样 N 次 Accept/Reject 序列
        5. 构造 response_mask = ~match_mask (只对 Mismatch 位置计算 Loss)
        """
        # 1. 准备输入
        idx = prompts.batch['input_ids'] # [Batch, Seq_Len]
        # 这里的 attention_mask 是 padding mask，是2D的 [Batch, Seq_Len]
        attention_mask = prompts.batch['attention_mask']
        batch_size = idx.shape[0]
        
        # 解析输入序列以获取 Draft 和 Target Tokens
        # 我们直接从 prompts.non_tensor_batch['extra_info'] 中获取这些信息
        # 注意: extra_info 是一个 object array，每个元素是一个 dict
        extra_infos = prompts.non_tensor_batch['extra_info']
        
        draft_tokens_list = [info['draft_tokens'] for info in extra_infos]
        target_tokens_list = [info['target_tokens'] for info in extra_infos]
        
        # 转换为 Tensor
        draft_tokens = torch.tensor(draft_tokens_list, device=self.device, dtype=torch.long)
        target_tokens = torch.tensor(target_tokens_list, device=self.device, dtype=torch.long)
        
        # 获取位置索引
        draft_start_idx = torch.tensor([info['draft_start_idx'] for info in extra_infos], device=self.device, dtype=torch.long)
        draft_end_idx = torch.tensor([info['draft_end_idx'] for info in extra_infos], device=self.device, dtype=torch.long)
        
        # 计算 Match Mask: [Batch, Draft_Len]
        match_mask = (draft_tokens == target_tokens).to(self.device)
        
        # 2. 模型前向传播
        # 传入解析出的 tokens 以应用 mask (Rollout 阶段必须应用 mask 以强制 accept)
        # 注意：ScoringActor 内部会再次计算 match_mask 并应用 logits mask
        outputs = self.model(
            input_ids=idx.to(self.device),
            attention_mask=attention_mask.to(self.device),
            draft_tokens=draft_tokens,
            target_tokens=target_tokens,
            # 传入位置索引以提高效率
            draft_start_idx=draft_start_idx,
            draft_end_idx=draft_end_idx,
        )
        
        # outputs["probs"]: [Batch, Draft_Len]
        probs = outputs["probs"]
        draft_len = probs.shape[1]
        
        # 确保 match_mask 维度正确
        # 理论上，match_mask 应该与 draft_len 完全一致，因为 match_mask 是由 draft_tokens 和 target_tokens 计算的，
        # 而两者应与模型输出 draft 长度一致（都取自 SEP 分割后的 Draft 段 token）。
        # 如果出现长度不一致，很可能是上游解析 input_ids 时出现了意外情况导致 draft_tokens 与实际 draft 区间不匹配。
        # 正常流程下，不应有 mismatch。这里加断言直接抛错，便于定位根因。

        assert match_mask.shape[1] == draft_len, (
            f"match_mask.shape={match_mask.shape}, draft_len={draft_len}，请检查 input_ids 序列解析与 Draft 区段一致性。"
        )
        
        # 3. 执行 Rollout (采样 N 次)
        rollout_n = self.config.n # GRPO 的 N
        
        # 扩展维度以进行并行采样: [Batch, 1, Draft_Len] -> [Batch, N, Draft_Len]
        probs_expanded = probs.unsqueeze(1).expand(-1, rollout_n, -1)
        
        # Bernoulli 采样
        # 结果: [Batch, N, Draft_Len] (0 或 1)
        actions = torch.bernoulli(probs_expanded)
        
        # 4. 构造返回结果
        
        # 展平 Batch 和 N 维度: [Batch * N, Draft_Len]
        sequences = actions.reshape(-1, draft_len).long()
        
        # 构造 log_probs
        probs_flat = probs_expanded.reshape(-1, draft_len)
        log_probs = sequences * torch.log(probs_flat + 1e-10) + (1 - sequences) * torch.log(1 - probs_flat + 1e-10)
        
        # 构造 response_mask (Loss Mask)
        # 核心逻辑: Loss Mask = ~Match Mask
        # Match 的位置不计算 Loss (因为是被强制 Accept 的)
        loss_mask = (~match_mask).float() # [Batch, Draft_Len]
        
        # 扩展 loss_mask 以匹配 sequences 维度 [Batch * N, Draft_Len]
        loss_mask_expanded = loss_mask.unsqueeze(1).expand(-1, rollout_n, -1).reshape(-1, draft_len)
        
        # 还需要返回 input_ids (prompt)，需要重复 N 次以匹配 sequence
        prompts_expanded = idx.repeat_interleave(rollout_n, dim=0)
        attention_mask_expanded = attention_mask.repeat_interleave(rollout_n, dim=0)
        
        # 构造输出 DataProto
        # 注意: response 在这里就是 Accept/Reject 序列
        output_batch = DataProto(
            batch={
                'input_ids': prompts_expanded,           # Prompt (Context+Draft+Target)
                'attention_mask': attention_mask_expanded,
                'responses': sequences,                  # Rollout 结果 (0/1 序列)
                'log_probs': log_probs,                  # 对应的 log 概率
                'response_mask': loss_mask_expanded      # 关键: 只对 Mismatch 位置计算 Loss
            },
            non_tensor_batch=prompts.non_tensor_batch, # 透传非 Tensor 数据 (Extra Info)
            meta_info={'rollout_n': rollout_n}
        )
        
        return output_batch

