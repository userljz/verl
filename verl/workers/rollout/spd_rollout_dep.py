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
import os
import numpy as np
from typing import Generator, List

from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.workers.config import RolloutConfig

from spd_scorer import ScoringActor, ScoringModelConfig
from transformers import AutoTokenizer

# 新增: vLLM 相关 import
from vllm import LLM, SamplingParams
from loguru import logger
import requests 

# 全局缓存 (单例模式)
# _VLLM_ENGINE_CACHE = {}



# def get_vllm_engine(
#     model_path: str,
#     tensor_parallel_size: int = 1,
#     gpu_memory_utilization: float = 0.9,
#     max_model_len: int = None,
#     trust_remote_code: bool = True,
#     dtype: str = "auto",
#     **kwargs
# ):
#     """获取或初始化 vLLM LLM 实例 (单例模式)"""
#     cache_key = model_path
#     if cache_key in _VLLM_ENGINE_CACHE:
#         logger.info(f"[vLLM] Reusing cached engine for: {model_path}")
#         return _VLLM_ENGINE_CACHE[cache_key]
    
#     logger.info(f"[vLLM] Initializing new engine for: {model_path}")
#     llm = LLM(
#         model=model_path,
#         tensor_parallel_size=tensor_parallel_size,
#         gpu_memory_utilization=gpu_memory_utilization,
#         max_model_len=max_model_len,
#         trust_remote_code=trust_remote_code,
#         dtype=dtype,
#         distributed_executor_backend="mp",
#         **kwargs
#     )
#     _VLLM_ENGINE_CACHE[cache_key] = llm
#     return llm

# def batch_vllm_generate(
#     prompt_token_ids: List[List[int]],
#     vllm_engine,
#     temperature: float = 0.0,
#     max_tokens: int = 1024,
#     top_p: float = 1.0,
#     tensor_parallel_size: int = 1,
#     gpu_memory_utilization: float = 0.9,
#     max_model_len: int = None,
#     **engine_kwargs
# ) -> List[str]:
#     """使用离线 vLLM 批量生成 (直接传入 Token IDs)"""
#     if not prompt_token_ids:
#         raise ValueError("prompt_token_ids 列表不能为空")

#     logger.info(f"[vLLM] Generating {len(prompt_token_ids)} sequences with temperature={temperature}, max_tokens={max_tokens}")

#     sampling_params = SamplingParams(
#         temperature=temperature,
#         top_p=top_p,
#         max_tokens=max_tokens,
#     )
    
#     outputs = vllm_engine.generate(
#         prompt_token_ids=prompt_token_ids,
#         sampling_params=sampling_params,
#     )
    
#     results = []
#     for output in outputs:
#         results.append(output.outputs[0].text)
    
#     return results


class VllmEngineServer:
    def __init__(self, url, server_name):
        self.url = url
        self.server_name = server_name
        logger.info(f"[vLLM] Initializing new engine for: {server_name}")
        
    def batch_complete_ids(self, batch_token_ids, max_tokens):
        """
        同步发送 Batch Token IDs 给 vLLM 进行续写
        batch_token_ids: List[List[int]] (已去除 Padding)
        """
        
        payload = {
            "model": self.server_name,
            # 2. 直接传入 ID 列表，vLLM 会并行处理
            "prompt": batch_token_ids, 
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        
        logger.info(f"[Client] Sending {len(batch_token_ids)} sequences to vLLM via ID...")
        
        try:
            # vLLM 处理 Batch 可能会花几秒，Timeout 设长一点
            response = requests.post(self.url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            # 提取 text (注意: completions 接口返回的是 text 字段)
            # vLLM 保证返回顺序与输入顺序一致
            completions = [choice['text'] for choice in result['choices']]
            return completions
        except Exception as e:
            logger.info(f"Error calling vLLM: {e}")
            return ["Error"] * len(batch_token_ids)


class SPDRollout(BaseRollout):
    def __init__(self, config: RolloutConfig, model_config, device_mesh):
        super().__init__(config, model_config, device_mesh)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化 Scoring Model
        # 注意: 这里我们需要一个能够加载权重的 ScoringActor
        # 在 verl 架构中，Actor 和 Rollout 可能共享权重
        
        # 创建模型实例 (这里假设模型结构与 spd_scorer.py 中定义的一致)
        # 实际运行时，权重会通过 update_weights 加载
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.path, trust_remote_code=True)

        # target_model_path = os.getenv("SPD_TARGET_MODEL_PATH")
        # _tp_size = int(os.getenv("SPD_VLLM_TENSOR_PARALLEL_SIZE"))
        # _tp_size = self.config.tensor_model_parallel_size
        # _gpu_util = float(os.getenv("SPD_VLLM_GPU_MEMORY_UTILIZATION"))
        # _gpu_util = self.config.gpu_memory_utilization
        # _max_len = int(os.getenv("SPD_VLLM_MAX_MODEL_LEN"))
        # _max_len = self.config.gpu_memory_utilization
        self.vllm_engine = VllmEngineServer(url="http://localhost:8000/v1/completions", server_name="model-b")

        # 从环境变量获取 SPD 特定配置
        lora_rank = int(os.getenv("SPD_LORA_RANK", "16"))
        lora_alpha = int(os.getenv("SPD_LORA_ALPHA", "32"))
        sep_token_id = int(os.getenv("SPD_SEP_TOKEN_ID"))
        
        spd_config = ScoringModelConfig(
            model_name_or_path=model_config.path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            sep_token_id=sep_token_id
        )
        self.model = ScoringActor(spd_config)
        self.model.to(self.device)
        self.model.eval()
     

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
        context_draft_target_ids = prompts.batch['input_ids'] # [Batch, Seq_Len]
        # 这里的 attention_mask 是 padding mask，是2D的 [Batch, Seq_Len]
        attention_mask = prompts.batch['attention_mask']
        batch_size = context_draft_target_ids.shape[0]

        # [DEBUG] 打印 rollout 接收到的 batch size
        logger.info(f"[DEBUG SPDRollout] Received prompts batch size: {batch_size}")
        
        # 从 rollout_info 获取信息 (这是我们在 train_spd_scorer.py 中显式透传的)
        if 'rollout_info' not in prompts.non_tensor_batch:
             raise KeyError(f"'rollout_info' not found in non_tensor_batch. Available keys: {list(prompts.non_tensor_batch.keys())}")
             
        rollout_infos = prompts.non_tensor_batch['rollout_info']
        
        draft_tokens_list = [info['draft_tokens'] for info in rollout_infos]
        target_tokens_list = [info['target_tokens'] for info in rollout_infos]
        
        # 转换为 Tensor
        # 注意: 如果不同样本 draft 长度不一致，需要 padding 到最大长度
        # 但 verl 框架通常会处理 batch 使得长度可控，这里我们先假设同一 batch 内 draft tokens 长度一致
        # 如果不一致，使用嵌套 tensor 或 padding
        
        # 假设数据集构造保证了同一 batch 内 (甚至全局) draft_len 一致，或者直接取当前 batch 的长度
        # 且 draft_len 已在 rollout_info 中透传
        draft_len = rollout_infos[0]['draft_len']
        
        # 构造 Tensor (无 Padding)
        draft_tokens = torch.zeros((batch_size, draft_len), dtype=torch.long, device=self.device)
        target_tokens = torch.zeros((batch_size, draft_len), dtype=torch.long, device=self.device)
        
        for i in range(batch_size):
            draft_tokens[i] = torch.tensor(draft_tokens_list[i], device=self.device)
            target_tokens[i] = torch.tensor(target_tokens_list[i], device=self.device)
        
        # 获取位置索引
        draft_start_idx = torch.tensor([info['draft_start_idx'] for info in rollout_infos], device=self.device, dtype=torch.long)
        draft_end_idx = torch.tensor([info['draft_end_idx'] for info in rollout_infos], device=self.device, dtype=torch.long)
        
        # 计算 Match Mask: [Batch, Draft_Len]
        # 无需处理 Padding
        match_mask = (draft_tokens == target_tokens).to(self.device)
        
        # 2. 模型前向传播
        # 传入解析出的 tokens 以应用 mask (Rollout 阶段必须应用 mask 以强制 accept)
        # 注意：ScoringActor 内部会再次计算 match_mask 并应用 logits mask
        
        # [Fix] OOM 优化: 使用 Mini-Batch 进行推理
        # 从 config 中读取 micro_batch_size, 默认为 32
        mini_batch_size = self.config.log_prob_micro_batch_size_per_gpu
        if mini_batch_size is None:
            mini_batch_size = 32
            
        logger.info(f"[DEBUG SPDRollout] Using mini_batch_size: {mini_batch_size}")
        
        probs_list = []
        
        for i in range(0, batch_size, mini_batch_size):
            end_idx = min(i + mini_batch_size, batch_size)
            
            # 切片
            mb_input_ids = context_draft_target_ids[i:end_idx]
            mb_attention_mask = attention_mask[i:end_idx]
            mb_draft_tokens = draft_tokens[i:end_idx]
            mb_target_tokens = target_tokens[i:end_idx]
            mb_draft_start_idx = draft_start_idx[i:end_idx]
            mb_draft_end_idx = draft_end_idx[i:end_idx]
            
            with torch.no_grad():
                mb_outputs = self.model(
                    input_ids=mb_input_ids.to(self.device),
                    attention_mask=mb_attention_mask.to(self.device),
                    draft_tokens=mb_draft_tokens,
                    target_tokens=mb_target_tokens,
                    draft_start_idx=mb_draft_start_idx,
                    draft_end_idx=mb_draft_end_idx,
                )
                probs_list.append(mb_outputs["probs"])
                
        # 合并结果 [Batch, Draft_Len]
        probs = torch.cat(probs_list, dim=0)
        
        draft_len = probs.shape[1]
        
        # 确保 match_mask 维度正确
        # 理论上，match_mask 应该与 draft_len 完全一致，因为 match_mask 是由 draft_tokens 和 target_tokens 计算的，
        # 而两者应与模型输出 draft 长度一致（都取自 SEP 分割后的 Draft 段 token）。
        # 如果出现长度不一致，很可能是上游解析 input_ids 时出现了意外情况导致 draft_tokens 与实际 draft 区间不匹配。
        # 正常流程下，不应有 mismatch。这里加断言直接抛错，便于定位根因。

        assert match_mask.shape[1] == draft_len, (
            f"match_mask.shape={match_mask.shape}, draft_len={draft_len}，请检查 input_ids 序列解析与 Draft 区段一致性。"
        )
        
        # 3. 执行 Rollout
        # [Important] 这里必须设为 1。
        # 原因: RayTrainer 已经在外部根据 config.rollout.n 对输入数据进行了扩展 (repeat)。
        # 例如: rollout.n=8, 原始 batch=64 -> 输入给这里的 batch 已经是 512。
        # 所以这里只需要对每条输入生成 1 个结果，最终总数就是 512，与 RayTrainer 期望的一致。
        rollout_n = 1
        
        # 3.1 Bernoulli 采样
        # 输入 probs: [Batch, Draft_Len] (注意: 此时 Batch 已经是外部扩展过的)
        # 输出 actions: [Batch, Draft_Len] (0 或 1)
        actions = torch.bernoulli(probs)
        sequences = actions.long()
        
        # 3.2 构造 log_probs
        # log_probs: [Batch, Draft_Len]
        # P(Accept) = prob, P(Reject) = 1 - prob
        # 如果 action=1, log_prob = log(prob)
        # 如果 action=0, log_prob = log(1 - prob)
        log_probs = sequences * torch.log(probs + 1e-10) + (1 - sequences) * torch.log(1 - probs + 1e-10)
        
        # 3.3 构造 response_mask (Loss Mask)
        # 核心逻辑: Loss Mask = ~Match Mask
        # Match 的位置不计算 Loss (因为是被强制 Accept 的)
        loss_mask = (~match_mask).float() # [Batch, Draft_Len]
        
        # === Heavy Rollout (vLLM 补全) ===
        
        # 3.4 计算有效长度 L (Tensor)
        # actions 是 [Batch, Draft_Len] 的 0/1 序列
        # valid_mask = cumprod(actions) -> L = sum(valid_mask)
        accept_decisions = actions.float()
        valid_mask_tensor = torch.cumprod(accept_decisions, dim=1)
        L_tensor = valid_mask_tensor.sum(dim=1).long() # [Batch]
        L_list = L_tensor.tolist()
        
        # 初始化 is_correct_hybrid_list
        is_correct_hybrid_list = [False] * batch_size
        
        # 3.5 构造 Hybrid Context 并调用 vLLM
        # 需要 vLLM 配置和 Tokenizer (用于解码检查)
        
        
        
        # 提取相关 Tokens
        context_ids_list = [info['context_ids'] for info in rollout_infos]
        draft_tokens_list = [info['draft_tokens'] for info in rollout_infos]
        target_tokens_list = [info['target_tokens'] for info in rollout_infos]
        
        prompts_for_vllm = []
        indices_requiring_vllm = []
        
        for i in range(batch_size):
            l_val = L_list[i]
            
            # L=0 时退化为 Baseline，无需调用 vLLM
            if l_val == 0:
                prompts_for_vllm.append(None) 
            else:
                curr_context = context_ids_list[i]
                curr_draft = draft_tokens_list[i]
                
                hybrid_ids = curr_context + curr_draft[:l_val]
                prompts_for_vllm.append(hybrid_ids)
                indices_requiring_vllm.append(i)
        
        # 3.6 批量调用 vLLM 生成
        valid_prompts = [p for p in prompts_for_vllm if p is not None]
        completions = []
        
        if valid_prompts:
            # _temperature = float(os.getenv("SPD_VLLM_TEMPERATURE", "0.0"))
            _max_tokens = int(os.getenv("SPD_VLLM_MAX_TOKENS", "1024"))
            
            # completions = batch_vllm_generate(
            #     prompt_token_ids=valid_prompts,
            #     vllm_engine=self.vllm_engine,
            #     temperature=_temperature,
            #     max_tokens=_max_tokens,
            # )

            completions = self.vllm_engine.batch_complete_ids(valid_prompts, max_tokens=_max_tokens)
            
        

        # 3.7 验证正确性 (Ground Truth Match)
        ground_truths = [info['ground_truth'] for info in rollout_infos]
        
        
        comp_idx = 0
        for i in range(batch_size):
            if i in indices_requiring_vllm:
                # 获取 Hybrid Prompt 文本
                prompt_ids_val = valid_prompts[comp_idx]
                completion_text = completions[comp_idx]
                comp_idx += 1
                
                # 拼接完整答案并检查 GT
                prompt_text = self.tokenizer.decode(prompt_ids_val, skip_special_tokens=True)
                final_answer = prompt_text + completion_text
                
                gt = ground_truths[i]
                if "\\boxed" in gt:
                    is_correct = gt in final_answer
                else:
                    is_correct = gt.strip() in final_answer.strip()
                    
                is_correct_hybrid_list[i] = is_correct
                     
        # 4. 构造返回结果
        # 由于 rollout_n=1，且输入已经扩展过，所以不需要再 expand/repeat
        
        # 将 is_correct_hybrid 转为 Tensor，通过 batch 传递给 Reward 函数
        is_correct_hybrid_tensor = torch.tensor(is_correct_hybrid_list, dtype=torch.float32, device=self.device)
        
        # 构造输出 DataProto
        batch = TensorDict(
            {
                'prompts': context_draft_target_ids,     # 原始 Prompt (用于 BatchRewardManager)
                'input_ids': context_draft_target_ids,   # Prompt (Context+Draft+Target)
                'attention_mask': attention_mask,
                'responses': sequences,                  # Rollout 结果 (0/1 序列)
                'old_log_probs': log_probs,              # 对应的 log 概率
                'response_mask': loss_mask,              # 关键: 只对 Mismatch 位置计算 Loss
                'effective_len': L_tensor,               # [Batch] 有效长度 (用于 Reward 计算)
                'is_correct_hybrid': is_correct_hybrid_tensor,  # [Batch] Hybrid 验证结果 (用于 Reward 计算)
            },
            batch_size=batch_size,
        )
        
        # non_tensor_batch: 保持为空
        # (rollout 阶段无法访问 extra_info 等，它们保留在原 batch 中，会在 union 时合并)
        non_tensor_batch = {}
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={'rollout_n': rollout_n})


