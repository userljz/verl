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

import copy
import torch
import torch.distributed as dist
import os
import sys
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


# ==============================================================================
# Loguru Rank0-Only 初始化（与 train_spd_scorer.py 保持一致）
# ==============================================================================
_LOGURU_RANK0_INIT = False

def _setup_loguru_rank0():
    """
    配置 loguru 只让 rank0 输出 DEBUG/INFO，WARNING/ERROR 全 rank 输出。
    输出到环境变量 SPD_LOG_FILE 指定的文件（与 train_spd_scorer.py 保持一致）。
    该函数只会执行一次（幂等）。
    """
    global _LOGURU_RANK0_INIT
    if _LOGURU_RANK0_INIT:
        return
    
    log_file = os.getenv("SPD_LOG_FILE")
    if not log_file:
        # 如果没有设置 SPD_LOG_FILE，跳过配置（保持默认行为）
        return
    
    def _filter(record):
        # WARNING/ERROR 全 rank 输出；DEBUG/INFO 只 rank0 输出
        if record["level"].no >= 30:
            return True
        return (not dist.is_initialized()) or dist.get_rank() == 0

    level = os.getenv("LOGURU_LEVEL", "INFO")
    
    logger.remove()
    # 输出到文件，与 train_spd_scorer.py 保持一致
    logger.add(log_file, level=level, filter=_filter)
    
    _LOGURU_RANK0_INIT = True 



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
            logger.error(f"Error calling vLLM: {e}")
            return ["Error"] * len(batch_token_ids)


class SPDRollout(BaseRollout):
    def __init__(self, config: RolloutConfig, model_config, device_mesh):
        super().__init__(config, model_config, device_mesh)
        
        # 初始化 loguru rank0-only 配置（确保只在 rank0 打印 DEBUG/INFO）
        _setup_loguru_rank0()
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.path, trust_remote_code=True)

        # ==================== 多 vLLM 服务器支持 ====================
        # 从环境变量读取多个 vLLM 服务器 URL（逗号分隔）
        # 例如: SPD_VLLM_URLS="http://localhost:8000/v1/completions,http://localhost:8001/v1/completions,..."
        # 如果只有一个 URL，所有 rank 共用同一个服务器（兼容旧配置）
        vllm_urls_str = os.getenv("SPD_VLLM_URLS", "http://localhost:8000/v1/completions")
        vllm_urls = [url.strip() for url in vllm_urls_str.split(",")]
        
        # 获取当前进程的 rank
        # 注意: verl 框架使用 Ray 管理分布式，不会设置 LOCAL_RANK 环境变量
        # 优先级: 
        #   1. torch.distributed.get_rank() - 如果 dist 已初始化
        #   2. 从 CUDA_VISIBLE_DEVICES 解析实际 GPU ID - Ray 会为每个 Actor 设置不同的值
        #   3. 默认使用 0
        if dist.is_initialized():
            global_rank = dist.get_rank()
        else:
            # Ray 场景: 每个 Actor 的 CUDA_VISIBLE_DEVICES 被设置为单个 GPU
            # 例如: CUDA_VISIBLE_DEVICES=2 表示该 Actor 使用 GPU 2
            cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "0")
            # 取第一个 GPU ID 作为 rank (Ray 通常只给每个 Actor 分配一个 GPU)
            global_rank = int(cuda_visible.split(",")[0])
        
        # 根据 rank 选择对应的 vLLM 服务器（取模以支持服务器数 < GPU 数的情况）
        selected_url = vllm_urls[global_rank % len(vllm_urls)]
        
        logger.info(f"[SPDRollout] global_rank={global_rank}, CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}, "
                    f"using vLLM server: {selected_url} (from {len(vllm_urls)} servers)")
        
        self.vllm_engine = VllmEngineServer(url=selected_url, server_name="model-b")

        # 从环境变量获取 SPD 特定配置
        lora_rank = int(os.getenv("SPD_LORA_RANK", "16"))
        lora_alpha = int(os.getenv("SPD_LORA_ALPHA", "32"))
        sep_token_id = int(os.getenv("SPD_SEP_TOKEN_ID"))
        adapter_path = os.getenv("SPD_ADAPTER_PATH") # 获取 adapter_path
        
        spd_config = ScoringModelConfig(
            model_name_or_path=model_config.path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            sep_token_id=sep_token_id,
            adapter_path=adapter_path
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
        
        # ==================== DEBUG: Rollout 输入信息 ====================
        logger.debug(f"[SPD Rollout 输入] batch_size={batch_size}, seq_len={context_draft_target_ids.shape[1]}")
        logger.debug(f"[SPD Rollout 输入] input_ids shape: {context_draft_target_ids.shape}")
        logger.debug(f"[SPD Rollout 输入] attention_mask shape: {attention_mask.shape}")
        logger.debug(f"[SPD Rollout 输入] non_tensor_batch keys: {list(prompts.non_tensor_batch.keys())}")
        
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
        
        # ==================== DEBUG: Draft/Target 信息 ====================
        logger.debug(f"[SPD Rollout Draft/Target] draft_len={draft_len}")
        logger.debug(f"[SPD Rollout Draft/Target] draft_tokens shape: {draft_tokens.shape}")
        logger.debug(f"[SPD Rollout Draft/Target] target_tokens shape: {target_tokens.shape}")
        # 打印第一个样本的 draft 和 target tokens (用于调试)
        if batch_size > 0:
            logger.debug(f"[SPD Rollout Sample 0] draft_tokens: {draft_tokens[0].tolist()[:10]}... (first 10)")
            logger.debug(f"[SPD Rollout Sample 0] target_tokens: {target_tokens[0].tolist()[:10]}... (first 10)")
        
        # 获取位置索引
        draft_start_idx = torch.tensor([info['draft_start_idx'] for info in rollout_infos], device=self.device, dtype=torch.long)
        draft_end_idx = torch.tensor([info['draft_end_idx'] for info in rollout_infos], device=self.device, dtype=torch.long)
        
        # 计算 Match Mask: [Batch, Draft_Len]
        # 无需处理 Padding
        match_mask = (draft_tokens == target_tokens).to(self.device)
        
        # 2. 模型前向传播
        # 传入解析出的 tokens 以应用 mask (Rollout 阶段必须应用 mask 以强制 accept)
        # 注意：ScoringActor 内部会再次计算 match_mask 并应用 logits mask
        
        # [Fix] OOM 优化: 使用 Micro-Batch 进行推理
        # 从 config 中读取 micro_batch_size_per_gpu, 默认为 32
        micro_batch_size = self.config.log_prob_micro_batch_size_per_gpu
        if micro_batch_size is None:
            micro_batch_size = 32
        micro_batch_size = min(micro_batch_size, batch_size)
            
        logger.info(f"[DEBUG SPDRollout] Using micro_batch_size: {micro_batch_size}, batch_size: {batch_size}")
        
        probs_list = []
        
        for i in range(0, batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, batch_size)
            
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
        
        # ==================== DEBUG: 模型输出信息 ====================
        logger.debug(f"[SPD Rollout 模型输出] probs shape: {probs.shape}")
        logger.debug(f"[SPD Rollout 模型输出] probs 统计: min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}")
        # 打印第一个样本的概率分布
        if batch_size > 0:
            logger.debug(f"[SPD Rollout Sample 0] probs: {[f'{p:.3f}' for p in probs[0].tolist()]}")
        
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
        
        # [New] 温度采样：增加 mismatch 位置的采样多样性
        # 温度 > 1.0 时，概率会趋向 0.5，增加随机性
        # 温度 = 1.0 时，保持原始概率
        # 温度 < 1.0 时，概率会更极端（接近 0 或 1）
        sample_temperature = float(os.getenv("SPD_SAMPLE_TEMPERATURE", "2.0"))
        
        if sample_temperature != 1.0:
            # 将 prob 转换为 logit，应用温度，再转回 prob
            # 使用 clamp 避免 log(0) 或 log(inf)
            probs_clamped = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
            logits = torch.log(probs_clamped / (1.0 - probs_clamped))
            logits_tempered = logits / sample_temperature
            probs_tempered = torch.sigmoid(logits_tempered)
            
            # 保持 match 位置的 prob 不变（强制接受）
            probs_for_sample = torch.where(match_mask, probs, probs_tempered)
            
            logger.debug(f"[Temperature Sampling] temp={sample_temperature}, "
                        f"original_probs_range=({probs.min().item():.4f}, {probs.max().item():.4f}), "
                        f"tempered_probs_range=({probs_tempered.min().item():.4f}, {probs_tempered.max().item():.4f})")
        else:
            probs_for_sample = probs
        
        actions = torch.bernoulli(probs_for_sample)
        sequences = actions.long()
        
        # ==================== DEBUG: 采样结果 ====================
        logger.debug(f"[SPD Rollout 采样] actions shape: {actions.shape}")
        accept_rate = actions.mean().item()
        logger.debug(f"[SPD Rollout 采样] 整体 Accept Rate: {accept_rate:.4f}")
        # 打印第一个样本的采样结果
        if batch_size > 0:
            logger.debug(f"[SPD Rollout Sample 0] actions: {actions[0].long().tolist()}")
            logger.debug(f"[SPD Rollout Sample 0] match_mask: {match_mask[0].tolist()}")
        
        # 3.2 构造 log_probs
        # log_probs: [Batch, Draft_Len]
        # P(Accept) = prob, P(Reject) = 1 - prob
        # 如果 action=1, log_prob = log(prob)
        # 如果 action=0, log_prob = log(1 - prob)
        log_probs = sequences * torch.log(probs + 1e-10) + (1 - sequences) * torch.log(1 - probs + 1e-10)
        
        # ==================== DEBUG: Log Probs 信息 ====================
        logger.debug(f"[SPD Rollout Log Probs] log_probs 统计: min={log_probs.min().item():.4f}, max={log_probs.max().item():.4f}, mean={log_probs.mean().item():.4f}")
        if batch_size > 0:
            logger.debug(f"[SPD Rollout Sample 0] log_probs: {[f'{lp:.3f}' for lp in log_probs[0].tolist()]}")
        
        # 3.3 构造 response_mask (Loss Mask)
        # 核心逻辑: Loss Mask = ~Match Mask
        # Match 的位置不计算 Loss (因为是被强制 Accept 的)
        loss_mask = (~match_mask).float() # [Batch, Draft_Len]
        
        # ==================== DEBUG: Loss Mask 信息 ====================
        mismatch_ratio = loss_mask.mean().item()
        logger.debug(f"[SPD Rollout Loss Mask] mismatch (需要学习) 比例: {mismatch_ratio:.4f}")
        if batch_size > 0:
            sample_mismatch_count = loss_mask[0].sum().item()
            logger.debug(f"[SPD Rollout Sample 0] mismatch 位置数量: {int(sample_mismatch_count)}/{draft_len}")
        
        # === Heavy Rollout (vLLM 补全) ===
        
        # 3.4 计算有效长度 L (Tensor)
        # actions 是 [Batch, Draft_Len] 的 0/1 序列
        # valid_mask = cumprod(actions) -> L = sum(valid_mask)
        accept_decisions = actions.float()
        valid_mask_tensor = torch.cumprod(accept_decisions, dim=1)
        L_tensor = valid_mask_tensor.sum(dim=1).long() # [Batch]
        L_list = L_tensor.tolist()
        
        # ==================== DEBUG: 有效长度 L 统计 ====================
        logger.debug(f"[SPD Rollout 有效长度] L 统计: min={min(L_list)}, max={max(L_list)}, mean={sum(L_list)/len(L_list):.2f}")
        L_zero_count = sum(1 for l in L_list if l == 0)
        logger.debug(f"[SPD Rollout 有效长度] L=0 的样本数: {L_zero_count}/{batch_size}")
        # 打印 L 分布
        L_distribution = {}
        for l in L_list:
            L_distribution[l] = L_distribution.get(l, 0) + 1
        logger.debug(f"[SPD Rollout 有效长度] L 分布: {dict(sorted(L_distribution.items()))}")
        
        # [Fix] L=0 的样本不参与学习，将其 loss_mask 设为全 0
        # 原因: 如果没有接受任何 draft token，无法从中学到任何东西
        L_zero_mask = (L_tensor == 0).unsqueeze(1).expand_as(loss_mask)  # [Batch, Draft_Len]
        loss_mask = loss_mask * (~L_zero_mask).float()
        
        # 初始化 is_correct_hybrid_list
        # [Fix] 当 L=0 时，退化为 Baseline，正确性应等于 is_correct_baseline
        # 从 extra_info 中获取 is_correct_baseline（而不是 rollout_info，因为这个字段只在 extra_info 中）
        extra_infos_for_baseline = prompts.non_tensor_batch['extra_info']
        is_correct_baseline_list = [info.get('is_correct_baseline', False) for info in extra_infos_for_baseline]
        is_correct_hybrid_list = [False] * batch_size
        
        # 3.5 构造 Hybrid Context 并调用 vLLM
        # 需要 vLLM 配置和 Tokenizer (用于解码检查)
        
        
        
        # 提取相关 Tokens
        context_ids_list = [info['context_ids'] for info in rollout_infos]
        draft_tokens_list = [info['draft_tokens'] for info in rollout_infos]
        target_tokens_list = [info['target_tokens'] for info in rollout_infos]
        
        prompts_for_vllm = []
        indices_requiring_vllm = []
        statistics_for_print = []
        
        for i in range(batch_size):
            l_val = L_list[i]
            
            # L=0 时退化为 Baseline，无需调用 vLLM
            # [Fix] 此时正确性应等于 is_correct_baseline
            if l_val == 0:
                prompts_for_vllm.append(None)
                is_correct_hybrid_list[i] = is_correct_baseline_list[i]
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
            _max_tokens = int(os.getenv("SPD_VLLM_MAX_TOKENS", "1024"))
            
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


                # ================= only for debug ==========================
                context_text = self.tokenizer.decode(context_ids_list[i], skip_special_tokens=True)
                draft_text = self.tokenizer.decode(draft_tokens_list[i], skip_special_tokens=True)
                target_text = self.tokenizer.decode(target_tokens_list[i], skip_special_tokens=True)
                accept_len = L_list[i]

                logger.debug(f">>> [{i}]context_text: {context_text!r}")
                logger.debug(f">>> [{i}]draft_text: {draft_text!r}")
                logger.debug(f">>> [{i}]target_text: {target_text!r}")
                logger.debug(f">>> [{i}]match_mask: {match_mask[i].tolist()}")
                logger.debug(f">>> [{i}]probs: {probs[i].tolist()}")
                logger.debug(f">>> [{i}]actions: {actions[i].tolist()}")
                logger.debug(f">>> [{i}]accept_len: {accept_len}")
                logger.debug(f">>> [{i}]prompt_text: {prompt_text!r}")
                logger.debug(f">>> [{i}]completion_text: {completion_text!r}")
                # ===============================================================
                
                
                gt = ground_truths[i]
                if "\\boxed" in gt:
                    is_correct = gt in final_answer
                else:
                    is_correct = gt.strip() in final_answer.strip()
                    
                is_correct_hybrid_list[i] = is_correct
                logger.debug(f">>> [{i}] gt: {gt}  |  is_correct: {is_correct}")
                logger.debug(f" ================================================ ")
    
        is_correct_hybrid_tensor = torch.tensor(is_correct_hybrid_list, dtype=torch.float32, device=self.device)
        
        # ==================== DEBUG: 正确性统计 ====================
        correct_baseline_count = sum(1 for c in is_correct_baseline_list if c)
        correct_hybrid_count = sum(is_correct_hybrid_list)
        logger.debug(f"[SPD Rollout 正确性] Baseline 正确数: {correct_baseline_count}/{batch_size}")
        logger.debug(f"[SPD Rollout 正确性] Hybrid 正确数: {int(correct_hybrid_count)}/{batch_size}")
        
        # 统计四种场景
        scenario_A = sum(1 for i in range(batch_size) if is_correct_baseline_list[i] and is_correct_hybrid_list[i])  # 原本正确，Hybrid也正确
        scenario_B = sum(1 for i in range(batch_size) if is_correct_baseline_list[i] and not is_correct_hybrid_list[i])  # 原本正确，Hybrid错误
        scenario_C = sum(1 for i in range(batch_size) if not is_correct_baseline_list[i] and not is_correct_hybrid_list[i])  # 原本错误，Hybrid也错误
        scenario_D = sum(1 for i in range(batch_size) if not is_correct_baseline_list[i] and is_correct_hybrid_list[i])  # 原本错误，Hybrid正确
        logger.debug(f"[SPD Rollout 场景统计] A(加速成功)={scenario_A}, B(破坏正确)={scenario_B}, C(无用尝试)={scenario_C}, D(纠正错误)={scenario_D}")
        
        # ==================== INFO: Rollout 关键指标汇总 ====================
        avg_L = sum(L_list) / len(L_list) if L_list else 0
        logger.info(f"[Rollout] BS={batch_size} | AvgL={avg_L:.1f} | Accept={accept_rate:.2f} | Mismatch={mismatch_ratio:.2f} | 场景: A={scenario_A} B={scenario_B} C={scenario_C} D={scenario_D}")
        
        # 构造 position_ids (verl 框架需要此字段用于 compute_log_prob)
        # position_ids 需要根据 attention_mask 计算，考虑 left padding
        # 对于 left padding: position_ids = cumsum(attention_mask) - 1，且 padding 位置设为 0
        seq_len = attention_mask.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        # 处理 left padding: 根据 attention_mask 调整 position_ids
        # attention_mask: [0,0,0,1,1,1,1,1] (左侧 padding)
        # position_ids:   [0,0,0,0,1,2,3,4] (从第一个有效 token 开始计数)
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)
        
        # 构造输出 DataProto
        # 注意: 使用 'rollout_log_probs' 而不是 'old_log_probs'
        # 因为 verl 框架会在 compute_log_prob 中生成 'old_log_probs'
        # 如果这里也用 'old_log_probs'，union 时会因为值不同而报错
        batch = TensorDict(
            {
                'prompts': context_draft_target_ids,     # 原始 Prompt (用于 BatchRewardManager)
                'input_ids': context_draft_target_ids,   # Prompt (Context+Draft+Target)
                'attention_mask': attention_mask,
                'position_ids': position_ids,            # verl 框架需要此字段
                'responses': sequences,                  # Rollout 结果 (0/1 序列)
                'rollout_log_probs': log_probs,          # 对应的 log 概率 (使用 rollout_log_probs 避免与 verl 冲突)
                'response_mask': loss_mask,              # 关键: 只对 Mismatch 位置计算 Loss
            },
            batch_size=batch_size,
        )

        # [New] 将关键的 Tensor 结果注入到 non_tensor_batch['extra_info'] 中
        # 由于修改了 ray_trainer._get_gen_batch，extra_info 现在会传给 rollout
        # 这样 Reward Manager 可以直接从 extra_info 读取
        non_tensor_batch = prompts.non_tensor_batch
        extra_infos = non_tensor_batch['extra_info']

        # 转换 Tensor 为 Python list
        L_list_final = L_tensor.tolist()
        hybrid_list_final = is_correct_hybrid_tensor.tolist()

        # [Fix] 深拷贝每个 extra_info 字典，避免因 repeat() 浅拷贝导致的引用共享问题
        # 问题：DataProto.repeat() 使用 np.repeat() 只复制引用，不深拷贝字典
        # 导致同一个原始样本的 8 个 rollout 共享同一个 extra_info 字典
        for i in range(batch_size):
            # 先深拷贝，确保每个 rollout 有独立的 extra_info
            extra_infos[i] = copy.deepcopy(extra_infos[i])
            extra_infos[i]['effective_len'] = L_list_final[i]
            extra_infos[i]['is_correct_hybrid'] = hybrid_list_final[i]
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={'rollout_n': rollout_n})


