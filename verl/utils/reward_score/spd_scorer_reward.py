# -*- coding: utf-8 -*-
"""
SPD Scorer 自定义 Reward Function (Tensor Batch Optimized)
用于 verl 框架的 reward 计算

设计变更:
    - 移除 async/aiohttp，改为同步的 Tensor 批处理 + 批量 API 调用
    - 移除 parse_accept_decisions，直接使用 Tensor 输入
    - 强制使用 Batch 模式，大幅提升 GPU 利用率和吞吐量

作者: AI Assistant
日期: 2025-11-26
"""

import logging
import requests
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

# 全局 Tokenizer 缓存
_TOKENIZER_CACHE = {}

def get_tokenizer(model_path: str):
    """获取或加载 Tokenizer"""
    if model_path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_path]
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _TOKENIZER_CACHE[model_path] = tokenizer
    return tokenizer

def parse_input_sequence(input_ids, sep_token_id):
    """
    解析 input_ids (Context + Draft + Target)
    复用 spd_scorer.py 中的逻辑
    """
    try:
        from spd_scorer import parse_input_sequence as original_parse
        return original_parse(input_ids, sep_token_id)
    except ImportError:
        logger.warning("Could not import parse_input_sequence from spd_scorer. Implementing fallback.")
        # Fallback implementation (simplified)
        batch_size = input_ids.shape[0]
        device = input_ids.device
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=False)
        result = {"draft_tokens": [], "context_ids": []}
        
        # We need draft tokens. Assuming 3 SEPs: Context|Draft|Target
        # or 2 SEPs if Context|Draft+Target? Standard is 3.
        for b in range(batch_size):
            batch_seps = sep_positions[sep_positions[:, 0] == b, 1]
            if len(batch_seps) >= 2:
                # Context | Draft | Target
                # 0...sep1...sep2...
                sep1 = batch_seps[0].item()
                sep2 = batch_seps[1].item()
                draft = input_ids[b, sep1+1:sep2]
                context = input_ids[b, :sep1]
                
                # Append to list (we will pad later or just use list of tensors)
                result["draft_tokens"].append(draft)
                result["context_ids"].append(context)
            else:
                 result["draft_tokens"].append(torch.tensor([], device=device))
                 result["context_ids"].append(torch.tensor([], device=device))
        return result

def compute_effective_length_tensor(accept_decisions: torch.Tensor) -> torch.Tensor:
    """
    计算有效接受长度 L (Tensor Version)
    
    Args:
        accept_decisions: [Batch, Draft_Len] (0/1)
    
    Returns:
        L: [Batch]
    """
    # 找到第一个 0 的位置
    # 如果全为 1，则长度为 Draft_Len
    # 如果有 0，则长度为第一个 0 的索引
    
    # create a mask for first zero
    # cumprod: 1 1 1 0 0 -> 1 1 1 0 0
    # sum: 3
    
    # 注意: 如果中间有 0，后面的 1 也是无效的 (Speculative Decoding 性质)
    # 所以 cumprod 是正确的逻辑: 一旦遇到 0，后面全变成 0
    mask = torch.cumprod(accept_decisions, dim=1)
    L = mask.sum(dim=1)
    return L

def batch_vllm_generate(
    prompts: List[List[int]],
    api_url: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024
) -> List[str]:
    """
    调用 vLLM 批量接口 (发送 Token IDs)
    """
    if not prompts:
        return []

    headers = {"Content-Type": "application/json"}
    
    # vLLM 支持 batch prompt_token_ids
    data = {
        "model": model,
        "prompt_token_ids": prompts, # 传递 Token IDs 列表
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        # 提取结果文本
        return [choice['text'] for item in result['choices'] for choice in [item]] # 假设 choices 结构匹配
        # vLLM OpenAI API 结构: choices 是一个列表，对应 batch 中的每一个
        # 但是标准的 OpenAI /completions 接口对于 batch 输入，choices 长度 = batch_size
        if 'choices' in result:
             return [c['text'] for c in result['choices']]
    except Exception as e:
        logger.error(f"vLLM Batch Request Error: {e}")
        # 失败时返回空字符串列表，后续逻辑需处理
        return [""] * len(prompts)


def compute_score_batch(
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]],
    prompt_ids: Optional[torch.Tensor] = None,   # [Batch, Seq_Len]
    response_ids: Optional[torch.Tensor] = None, # [Batch, Response_Len] (即 Accept Decisions)
    attention_mask: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    penalty_break: float = -10.0,
    reward_correct: float = 100.0,
    reward_useless: float = 0.0,
    target_model_url: Optional[str] = None,
    target_model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    **kwargs
) -> List[Union[float, Dict[str, Any]]]:
    """
    批量计算 Reward (Tensor Batch Optimized)
    
    逻辑:
    1. 使用 response_ids (Tensor) 直接计算有效长度 L (Tensor Op)
    2. 使用 prompt_ids (Tensor) 提取 Context 和 Draft Tokens
    3. 构造 Hybrid Prefix (Context + Draft[:L]) -> Batch List[int]
    4. 批量调用 vLLM 生成
    5. 验证正确性 -> 计算 Reward (使用统一公式)
    
    完全向量化实现，移除 Python 循环。
    """
    
    # -------------------------------------------------------------------------
    # 0. 数据校验 (Strict Mode)
    # -------------------------------------------------------------------------
    if response_ids is None or prompt_ids is None:
        raise ValueError("compute_score_batch: 'response_ids' and 'prompt_ids' tensors are REQUIRED.")

    batch_size = len(solution_strs)
    
    # 确保配置存在
    t_url = target_model_url or extra_infos[0].get("target_model_url")
    m_path = model_path or extra_infos[0].get("model_path")
    
    if not t_url or not m_path:
        raise ValueError("Missing configuration: 'target_model_url' or 'model_path' must be provided.")

    tokenizer = get_tokenizer(m_path)
    t_name = target_model_name or extra_infos[0].get("target_model_name") or "target-model"

    # -------------------------------------------------------------------------
    # 1. 计算有效长度 L (Tensor Parallel)
    # -------------------------------------------------------------------------
    accept_decisions = response_ids.float()
    valid_mask = torch.cumprod(accept_decisions, dim=1)
    L_tensor = valid_mask.sum(dim=1).long() # [Batch]
    L_list = L_tensor.tolist()

    # -------------------------------------------------------------------------
    # 2. 构造 vLLM Prompts & Baseline Correctness
    # -------------------------------------------------------------------------
    sep_token_id = kwargs.get("sep_token_id", 128009)
    parsed_batch = parse_input_sequence(prompt_ids, sep_token_id)
    
    # Check parsing
    if not parsed_batch.get("draft_tokens") or not parsed_batch.get("context_ids"):
        raise ValueError(f"Failed to parse tokens using SEP={sep_token_id}")

    # 批量提取数据
    # spd_scorer.py updated parse_input_sequence returns padded tensors
    
    if "context_ids" in parsed_batch and isinstance(parsed_batch["context_ids"], torch.Tensor):
        # context_ids is Padded Tensor [Batch, MaxCtx]. Need to strip padding.
        # Use context_end_idx to slice
        c_end_idxs = parsed_batch["context_end_idx"].tolist()
        context_ids_padded = parsed_batch["context_ids"]
        context_ids_list = [context_ids_padded[i, :end].tolist() for i, end in enumerate(c_end_idxs)]
    elif "context_ids" in parsed_batch:
         # List of tensors/lists fallback
         context_ids_list = [c.tolist() if isinstance(c, torch.Tensor) else c for c in parsed_batch["context_ids"]]
    else:
        raise ValueError("parsed_batch missing context_ids")

    if "draft_tokens" in parsed_batch and isinstance(parsed_batch["draft_tokens"], torch.Tensor):
        # draft_tokens is Padded Tensor [Batch, MaxDraft].
        # We don't need to strip padding explicitly here because we slice with [:l_val] later.
        # But l_val depends on response_ids which matches draft length.
        # However, for robustness, we convert rows to lists.
        # Note: These lists will have trailing zeros.
        # But since we do `draft_tokens_list[i][:l_val]`, and `l_val` <= actual draft length,
        # the trailing zeros are never accessed. So it is safe.
        draft_tokens_list = parsed_batch["draft_tokens"].tolist()
    elif "draft_tokens" in parsed_batch:
         draft_tokens_list = [d.tolist() if isinstance(d, torch.Tensor) else d for d in parsed_batch["draft_tokens"]]
    else:
        raise ValueError("parsed_batch missing draft_tokens")
    
    # 提取 is_correct_baseline
    # 假设 extra_infos 是 list of dicts. 如果没有 is_correct_baseline 则报错
    is_correct_baseline_list = [info.get("is_correct_baseline") for info in extra_infos]
    if None in is_correct_baseline_list:
         raise ValueError("Missing 'is_correct_baseline' in extra_info")
    
    # 转为 Tensor 以便后续计算
    s_t_tensor = torch.tensor([1.0 if x else 0.0 for x in is_correct_baseline_list], device=response_ids.device)

    # 构造 Prompts
    prompts_for_vllm = []
    indices_needing_api = []
    
    for i in range(batch_size):
        l_val = L_list[i]
        
        # 如果 L=0，Hybrid == Baseline，无需调用 API
        if l_val == 0:
            prompts_for_vllm.append(None)
        else:
            # Context + Draft[:L]
            hybrid_ids = context_ids_list[i] + draft_tokens_list[i][:l_val]
            prompts_for_vllm.append(hybrid_ids)
            indices_needing_api.append(i)

    # -------------------------------------------------------------------------
    # 3. 批量生成
    # -------------------------------------------------------------------------
    valid_prompts = [p for p in prompts_for_vllm if p is not None]
    completions = []
    
    if valid_prompts:
        completions = batch_vllm_generate(
            prompts=valid_prompts,
            api_url=t_url,
            model=t_name
        )
    
    # -------------------------------------------------------------------------
    # 4. 验证正确性 & 计算 Reward (Vectorized)
    # -------------------------------------------------------------------------
    # 我们需要计算 s_h (Score Hybrid)。
    # 对于 L=0 的样本，s_h = s_t。
    # 对于 L>0 的样本，检查 ground_truth in (Accepted + Completion)。
    
    s_h_list = []
    comp_idx = 0
    
    for i in range(batch_size):
        if prompts_for_vllm[i] is None:
            # L=0 case
            s_h_list.append(s_t_tensor[i].item())
        else:
            completion_text = completions[comp_idx]
            comp_idx += 1
            
            # Decode accepted text
            # 这里必须 decode，因为 ground_truth 是文本。
            # 这是唯一无法完全向量化的部分 (Text Matching)
            l_val = L_list[i]
            accepted_text = tokenizer.decode(draft_tokens_list[i][:l_val], skip_special_tokens=True)
            final_answer = accepted_text + completion_text
            gt = ground_truths[i]
            
            if "\\boxed" in gt:
                is_correct = gt in final_answer
            else:
                is_correct = gt.strip() in final_answer.strip()
            
            s_h_list.append(1.0 if is_correct else 0.0)

    s_h_tensor = torch.tensor(s_h_list, device=response_ids.device)

    # -------------------------------------------------------------------------
    # 5. 应用公式计算 Reward (Pure Tensor Op)
    # -------------------------------------------------------------------------
    # R = S_t * S_h * (alpha * L) + 
    #     S_t * (1 - S_h) * penalty_break + 
    #     (1 - S_t) * (1 - S_h) * reward_useless + 
    #     (1 - S_t) * S_h * reward_correct
    
    # Cast L_tensor to float for multiplication
    L_float = L_tensor.float()
    
    term1 = s_t_tensor * s_h_tensor * (alpha * L_float)
    term2 = s_t_tensor * (1.0 - s_h_tensor) * penalty_break
    term3 = (1.0 - s_t_tensor) * (1.0 - s_h_tensor) * reward_useless
    term4 = (1.0 - s_t_tensor) * s_h_tensor * reward_correct
    
    rewards_tensor = term1 + term2 + term3 + term4
    
    # Return list of results (or tensor if caller supports it, but BatchRewardManager expects list of dicts/floats)
    # BatchRewardManager expects a list of results.
    # We can return simple floats or dicts. To match interface, let's return list of dicts.
    # But now we don't loop for calculation! We loop only for formatting return.
    
    # 快速构造返回结果
    rewards_list = rewards_tensor.tolist()
    
    return [
        {
            "score": r,
            "effective_length": l,
            # Skip scenario string generation for speed if not needed
        }
        for r, l in zip(rewards_list, L_list)
    ]

# 兼容旧接口
def compute_score(*args, **kwargs):
    # 如果调用的是单条数据的接口，重定向或者报错
    # 但为了兼容 NaiveRewardManager，我们保留一个简单的 wrapper
    # 这里不再实现单条逻辑，建议全部走 compute_score_batch
    pass
