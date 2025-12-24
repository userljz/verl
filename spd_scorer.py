# -*- coding: utf-8 -*-
"""
Speculative Decoding Scoring Model (SPD Scorer)
用于 Speculative Decoding 场景的评分模型实现

核心思想:
    - 模型接收 [Context] + [SEP] + [Draft] + [SEP] + [Target] + [SEP] 格式的输入
    - 对于 Draft Tokens 的每个位置，模型输出一个 Accept/Reject 的概率 (二分类)
    - Mismatch 聚焦策略: 对于 Draft == Target 的位置，强制输出接受 (logit -> +inf)
    - 只训练 LoRA Adapters 和 Score Head，冻结 Backbone

作者: AI Assistant
日期: 2025-11-25
"""

import os
import math
from loguru import logger
from typing import Optional, Tuple, Dict, List, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel


@dataclass
class ScoringModelConfig:
    """
    评分模型配置类
    
    Attributes:
        model_name_or_path: 基础模型路径 (如 meta-llama/Llama-3-8B)
        adapter_path: Peft Adapter 路径 (包含 adapter_model.safetensors 和 adapter_config.json)
        hidden_size: 隐藏层维度 (Llama-3-8B 默认 4096)
        sep_token_id: 分隔符 Token ID
        mismatch_logit_value: 用于 Match 位置的强制 logit 值
    """
    model_name_or_path: str = "meta-llama/Llama-3-8B"
    adapter_path: Optional[str] = None
    hidden_size: int = 4096
    sep_token_id: int = 128009  # Llama-3 的 <|eot_id|> token
    mismatch_logit_value: float = 50.0  # 用于 Match 位置的强制 logit 值
    
    # 兼容旧代码的字段 (虽然可能不再直接使用)
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0
    target_modules: str = "all"
    score_head_hidden_size: Optional[int] = None
    
    def __post_init__(self):
        # 允许从环境变量覆盖配置
        if self.score_head_hidden_size is None:
            self.score_head_hidden_size = self.hidden_size // 4


@dataclass
class SPDInputData:
    """
    SPD 模型的输入数据结构
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: Optional[torch.Tensor] = None
    draft_start_idx: Optional[torch.Tensor] = None
    draft_end_idx: Optional[torch.Tensor] = None
    target_start_idx: Optional[torch.Tensor] = None
    target_end_idx: Optional[torch.Tensor] = None
    draft_tokens: Optional[torch.Tensor] = None
    target_tokens: Optional[torch.Tensor] = None


# ==============================================================================
# 2. Score Head 定义 (与 AcceptHead 结构一致)
# ==============================================================================

class ScoreHead(nn.Module):
    """
    轻量级回归头，将隐藏状态映射为接受概率的 logits
    结构：LayerNorm → Linear(H→H/4) → GELU → Linear(H/4→1)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size // 4, 1)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [Batch, Seq_Len, Hidden_Size]
        Returns:
            logits: [Batch, Seq_Len, 1]
        """
        x = self.layer_norm(hidden_states)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x # 保持 [Batch, Seq_Len, 1] 形状，ScoringActor 会处理 squeeze


# ==============================================================================
# 3. ScoringActor 主模型
# ==============================================================================

class ScoringActor(nn.Module):
    """
    Speculative Decoding 评分模型 (ScoringActor)
    
    核心功能:
        1. 加载 Backbone (Llama-3)
        2. 替换 lm_head 为 ScoreHead
        3. 加载 Peft Adapter (包含 LoRA 和 ScoreHead 权重)
    """
    
    def __init__(self, config: ScoringModelConfig, backbone: nn.Module = None):
        super().__init__()
        self.config = config
        
        # 1. 加载 Backbone 并应用 Adapter
        if backbone is not None:
            self.backbone = backbone
            logger.info("使用传入的 backbone 模型")
        else:
            self._init_model()
            
        # 设置 transformer 引用 (用于获取 hidden states)
        # self.backbone 是 PeftModel
        # self.backbone.base_model 是 LlamaForCausalLM (或 wrapped)
        # 我们需要访问底层的 model (LlamaModel)
        if hasattr(self.backbone, "get_base_model"):
             base_model = self.backbone.get_base_model()
        else:
             base_model = self.backbone
             
        if hasattr(base_model, "model"):
            self.transformer = base_model.model
        else:
            self.transformer = base_model
            
        # 设置 score_head 引用 (指向替换后的 lm_head)
        # 注意: PeftModel 可能会包装 modules_to_save
        # 我们尝试获取有效的 head
        self.score_head = self._get_active_head()
        
        # 冻结参数 (PeftModel 默认已经冻结了非 LoRA/ModulesToSave 参数，这里再次确认)
        self.score_head.requires_grad_(True) 
        
        logger.info(f"ScoringActor 初始化完成:")
        logger.info(f"  - Backbone: {config.model_name_or_path}")
        logger.info(f"  - Adapter: {config.adapter_path}")

    def _init_model(self):
        """加载模型、替换 Head、加载 Adapter"""
        logger.info(f"正在加载 Backbone: {self.config.model_name_or_path}")
        
        # 1. 加载 Base Model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # 更新 hidden_size
        if hasattr(model.config, 'hidden_size'):
            self.config.hidden_size = model.config.hidden_size
            
        # 2. 替换 lm_head 为 ScoreHead
        self._replace_lm_head(model)
        
        # 3. 加载 Peft Adapter
        if self.config.adapter_path:
            logger.info(f"Loading adapter from {self.config.adapter_path}...")
            # is_trainable=True 确保 modules_to_save (ScoreHead) 被正确加载为可训练状态
            model = PeftModel.from_pretrained(
                model,
                self.config.adapter_path,
                is_trainable=True
            )
        else:
            logger.warning("未提供 adapter_path，使用未初始化的 ScoreHead (仅供测试)")
            
        self.backbone = model

    def _replace_lm_head(self, model):
        """将模型的 lm_head 替换为 ScoreHead"""
        hidden_size = self.config.hidden_size
        score_head = ScoreHead(hidden_size=hidden_size)
        
        replaced = False
        if hasattr(model, "lm_head"):
            device = next(model.lm_head.parameters()).device
            dtype = next(model.lm_head.parameters()).dtype
            model.lm_head = score_head.to(device=device, dtype=dtype)
            replaced = True
        elif hasattr(model, "model") and hasattr(model.model, "lm_head"):
            # 有些模型封装在 .model 下
            device = next(model.model.lm_head.parameters()).device
            dtype = next(model.model.lm_head.parameters()).dtype
            model.model.lm_head = score_head.to(device=device, dtype=dtype)
            replaced = True
            
        if not replaced:
            logger.warning("Could not find 'lm_head' to replace automatically.")
        else:
            logger.info(f"Successfully replaced lm_head with ScoreHead (hidden_size={hidden_size})")

    def _get_active_head(self):
        """获取当前的 ScoreHead 引用"""
        # 尝试从 PeftModel 中找到 lm_head
        # PeftModel -> Base Model -> lm_head
        if hasattr(self.backbone, "lm_head"):
            return self.backbone.lm_head
            
        # 如果 Peft 包装了
        base = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
        if hasattr(base, "lm_head"):
            return base.lm_head
            
        if hasattr(base, "model") and hasattr(base.model, "lm_head"):
            return base.model.lm_head
            
        raise ValueError("Cannot locate lm_head (ScoreHead) in the model structure")

    def _create_match_mask(
        self,
        draft_tokens: torch.Tensor,
        target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        创建 Mismatch Mask
        match_mask[i] = True 表示 draft == target (Match)
        """
        match_mask = (draft_tokens == target_tokens)
        return match_mask
    
    def _apply_match_mask(
        self,
        logits: torch.Tensor,
        match_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        应用 Mismatch 聚焦策略
        """
        match_mask_expanded = match_mask.unsqueeze(-1)
        force_accept_logit = torch.full_like(logits, self.config.mismatch_logit_value)
        masked_logits = torch.where(match_mask_expanded, force_accept_logit, logits)
        return masked_logits
    
    def _extract_draft_hidden_states(
        self,
        hidden_states: torch.Tensor,
        draft_start_idx: torch.Tensor,
        draft_end_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        从完整的 hidden states 中提取 Draft Token 位置的隐藏状态
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        draft_len = (draft_end_idx[0] - draft_start_idx[0]).item()
        if draft_len <= 0:
             # Fallback just in case
             return hidden_states
             
        grid = torch.arange(draft_len, device=device).unsqueeze(0) # [1, L]
        gather_indices = draft_start_idx.unsqueeze(1) + grid # [B, L]
        
        flat_hidden = hidden_states.view(-1, hidden_size)
        batch_offsets = (torch.arange(batch_size, device=device) * seq_len).unsqueeze(1)
        flat_indices = (gather_indices + batch_offsets).view(-1)
        
        draft_hidden = flat_hidden[flat_indices].view(batch_size, draft_len, hidden_size)
        return draft_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        draft_start_idx: Optional[torch.Tensor] = None,
        draft_end_idx: Optional[torch.Tensor] = None,
        draft_tokens: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        batch_size = input_ids.size(0)
        
        # ==================== DEBUG: ScoringActor Forward 开始 ====================
        logger.debug("=" * 50)
        logger.debug("[ScoringActor Forward] 开始前向传播")
        logger.debug(f"[ScoringActor Forward] batch_size: {batch_size}")
        logger.debug(f"[ScoringActor Forward] input_ids shape: {input_ids.shape}")
        logger.debug(f"[ScoringActor Forward] attention_mask shape: {attention_mask.shape}")
        
        # 1. Transformer Forward (Backbone Only)
        # 处理 Attention Mask
        if attention_mask is not None and attention_mask.dim() == 2:
            if draft_start_idx is not None:
                seq_lens = attention_mask.sum(dim=-1)
                max_seq_len = input_ids.size(1)
                pad_lens = max_seq_len - seq_lens
                context_lens = draft_start_idx - pad_lens
                context_lens = torch.clamp(context_lens, min=0)
                
                attention_mask = create_spd_attention_mask(
                    context_lens=context_lens,
                    seq_lens=seq_lens,
                    padding_mask=attention_mask,
                    dtype=self.score_head.fc1.weight.dtype # Use head dtype
                )

        # 获取 hidden states (不经过 Head)
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        
        full_hidden_states = transformer_outputs.last_hidden_state
        
        # 2. 提取 Draft Hidden States
        if draft_start_idx is not None and draft_end_idx is not None:
            draft_hidden_states = self._extract_draft_hidden_states(
                full_hidden_states, draft_start_idx, draft_end_idx
            )
            # ==================== DEBUG: Draft Hidden States ====================
            logger.debug(f"[ScoringActor Forward] draft_start_idx: {draft_start_idx.tolist()[:3]}... (first 3)")
            logger.debug(f"[ScoringActor Forward] draft_end_idx: {draft_end_idx.tolist()[:3]}... (first 3)")
            logger.debug(f"[ScoringActor Forward] draft_hidden_states shape: {draft_hidden_states.shape}")
        else:
            draft_hidden_states = full_hidden_states
            logger.debug(f"[ScoringActor Forward] 使用完整 hidden states, shape: {draft_hidden_states.shape}")
        
        # 3. Score Head Forward
        # [Batch, Draft_Len, Hidden_Size] -> [Batch, Draft_Len, 1]
        raw_logits = self.score_head(draft_hidden_states)
        
        # ==================== DEBUG: Raw Logits ====================
        logger.debug(f"[ScoringActor Forward] raw_logits shape: {raw_logits.shape}")
        logger.debug(f"[ScoringActor Forward] raw_logits 统计: min={raw_logits.min().item():.4f}, max={raw_logits.max().item():.4f}, mean={raw_logits.mean().item():.4f}")
        
        # 4. Mismatch Mask
        match_mask = self._create_match_mask(draft_tokens, target_tokens)
        masked_logits = self._apply_match_mask(raw_logits, match_mask)
        
        # ==================== DEBUG: Match Mask ====================
        match_count = match_mask.sum().item()
        total_count = match_mask.numel()
        logger.debug(f"[ScoringActor Forward] match_mask 统计: match={int(match_count)}/{total_count} ({match_count/total_count*100:.1f}%)")
        logger.debug(f"[ScoringActor Forward] masked_logits 统计: min={masked_logits.min().item():.4f}, max={masked_logits.max().item():.4f}")
        
        # 5. Probabilities
        probs = torch.sigmoid(masked_logits).squeeze(-1)
        
        # ==================== DEBUG: Probabilities ====================
        logger.debug(f"[ScoringActor Forward] probs shape: {probs.shape}")
        logger.debug(f"[ScoringActor Forward] probs 统计: min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}")
        # 打印第一个样本的详细信息
        if batch_size > 0 and probs.dim() > 1 and probs.shape[1] > 0:
            logger.debug(f"[ScoringActor Forward Sample 0] probs: {[f'{p:.3f}' for p in probs[0].tolist()[:10]]}... (first 10)")
            logger.debug(f"[ScoringActor Forward Sample 0] match_mask: {match_mask[0].tolist()[:10]}... (first 10)")
        logger.debug("=" * 50)
        
        if return_dict:
            return {
                "raw_logits": raw_logits,
                "masked_logits": masked_logits,
                "probs": probs,
                "match_mask": match_mask,
                "hidden_states": draft_hidden_states
            }
        else:
            return masked_logits
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


class AutoModelForSPDScoring:
    """
    模拟 AutoModel 的工厂类，用于 verl 的加载流程适配
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return cls.from_config(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        config: 在这里通常是传入的 pretrained_model_name_or_path 字符串，或者是 HuggingFace Config
        """
        # 1. 确定 Base Model Path
        # 优先使用环境变量，否则使用参数
        env_model_path = os.getenv("SPD_MODEL_PATH")
        if env_model_path:
            model_path = env_model_path
        elif isinstance(config, str):
            model_path = config
        elif hasattr(config, "_name_or_path"):
            model_path = config._name_or_path
        else:
            model_path = "meta-llama/Llama-3-8B"
            
        # 2. 确定 Adapter Path
        adapter_path = os.getenv("SPD_ADAPTER_PATH")
        
        # 3. 其他配置
        hidden_size = 4096
        if hasattr(config, "hidden_size"):
             hidden_size = config.hidden_size
        
        logger.info(f"[AutoModelForSPDScoring] Initializing:")
        logger.info(f"  - Model Path: {model_path}")
        logger.info(f"  - Adapter Path: {adapter_path}")
        
        # 创建 Config
        spd_config = ScoringModelConfig(
            model_name_or_path=model_path,
            adapter_path=adapter_path,
            hidden_size=hidden_size
        )
        
        # 实例化 Actor
        model = ScoringActor(spd_config)
        return model

# ==============================================================================
# 5. 辅助工具函数
# ==============================================================================

def create_hybrid_attention_mask(
    seq_len: int,
    context_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    创建单样本的混合 Attention Mask
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask[:context_len, :context_len] = torch.tril(
        torch.ones(context_len, context_len, device=device, dtype=dtype)
    )
    mask[context_len:, :] = 1.0
    return mask


def create_hybrid_attention_mask_batch(
    context_lens: List[int],
    seq_lens: List[int],
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    is_left_padding: bool = True,
) -> torch.Tensor:
    """
    批量创建混合 Attention Mask
    """
    batch_size = len(context_lens)
    batch_mask = torch.zeros(batch_size, 1, max_seq_len, max_seq_len, device=device, dtype=dtype)
    
    for i in range(batch_size):
        c_len = context_lens[i]
        s_len = seq_lens[i]
        single_mask = create_hybrid_attention_mask(s_len, c_len, device, dtype)
        
        if is_left_padding:
            offset = max_seq_len - s_len
            batch_mask[i, 0, offset:, offset:] = single_mask
        else:
            batch_mask[i, 0, :s_len, :s_len] = single_mask
        
    return batch_mask


def combine_hybrid_and_padding_mask(
    hybrid_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    将 Hybrid Mask 与 Padding Mask 结合
    """
    batch_size, _, seq_len, _ = hybrid_mask.shape
    device = hybrid_mask.device
    dtype = hybrid_mask.dtype
    
    padding_mask_4d = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
    padding_mask_4d = padding_mask_4d.to(dtype)
    
    combined_mask = hybrid_mask * padding_mask_4d
    return combined_mask


def convert_mask_to_4d_attention_mask(
    combined_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    将 0/1 Mask 转换为 Hugging Face 模型可用的 4D Attention Mask
    """
    inverted_mask = 1.0 - combined_mask.to(dtype)
    attention_mask = inverted_mask * torch.finfo(dtype).min
    return attention_mask


def create_spd_attention_mask(
    context_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    padding_mask: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    一站式创建 SPD 场景的完整 4D Attention Mask
    """
    batch_size = padding_mask.shape[0]
    max_seq_len = padding_mask.shape[1]
    device = padding_mask.device
    
    c_lens = context_lens.tolist()
    s_lens = seq_lens.tolist()
    
    hybrid_mask = create_hybrid_attention_mask_batch(
        context_lens=c_lens,
        seq_lens=s_lens,
        max_seq_len=max_seq_len, 
        device=device,
        dtype=dtype,
        is_left_padding=True 
    )
    
    combined_mask = combine_hybrid_and_padding_mask(
        hybrid_mask=hybrid_mask,
        padding_mask=padding_mask,
    )
    
    attention_mask = convert_mask_to_4d_attention_mask(
        combined_mask=combined_mask,
        dtype=dtype,
    )
    
    return attention_mask


def create_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    根据 attention_mask 创建 position_ids
    """
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids
