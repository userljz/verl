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
import logging
from typing import Optional, Tuple, Dict, List, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

# ==============================================================================
# 1. 数据结构定义
# ==============================================================================

@dataclass
class ScoringModelConfig:
    """
    评分模型配置类
    
    Attributes:
        model_name_or_path: 基础模型路径 (如 meta-llama/Llama-3-8B)
        hidden_size: 隐藏层维度 (Llama-3-8B 默认 4096)
        score_head_hidden_size: Score Head 中间层维度 (默认 hidden_size // 4)
        sep_token_id: 分隔符 Token ID
        lora_rank: LoRA 秩
        lora_alpha: LoRA alpha 系数
        lora_dropout: LoRA Dropout
        target_modules: LoRA 目标模块列表
        mismatch_logit_value: 用于 Match 位置的强制 logit 值
    """
    model_name_or_path: str = "meta-llama/Llama-3-8B"
    hidden_size: int = 4096
    score_head_hidden_size: Optional[int] = None  # 默认 hidden_size // 4
    sep_token_id: int = 128009  # Llama-3 的 <|eot_id|> token，可根据实际情况调整
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # 默认 ["q_proj", "k_proj", "v_proj", "o_proj"]
    mismatch_logit_value: float = 50.0  # 用于 Match 位置的强制 logit 值 (不用 inf 避免数值问题)
    
    def __post_init__(self):
        if self.score_head_hidden_size is None:
            self.score_head_hidden_size = self.hidden_size // 4
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class SPDInputData:
    """
    SPD 模型的输入数据结构
    
    Attributes:
        input_ids: [Batch, Seq_Len] - 完整输入序列
        attention_mask: 注意力掩码，支持两种格式:
            - [Batch, Seq_Len]: 简单 Padding Mask (模型内部会自动生成 Causal Mask)
            - [Batch, 1, Seq_Len, Seq_Len]: 完整 4D Mask (用于自定义 Attention 模式)
              使用 create_spd_attention_mask() 生成混合 Mask (Context Causal + Draft/Target Bidirectional)
        position_ids: [Batch, Seq_Len] - 位置编码
        draft_start_idx: 每个样本中 Draft Tokens 的起始索引
        draft_end_idx: 每个样本中 Draft Tokens 的结束索引
        target_start_idx: 每个样本中 Target Tokens 的起始索引
        target_end_idx: 每个样本中 Target Tokens 的结束索引
        draft_tokens: [Batch, Draft_Len] - Draft Tokens (用于构建 Mismatch Mask)
        target_tokens: [Batch, Draft_Len] - Target Tokens (与 Draft 对齐)
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
# 2. Score Head 定义
# ==============================================================================

class ScoreHead(nn.Module):
    """
    评分头模块
    
    结构: Linear(hidden_size -> hidden_size/4) -> ReLU -> Linear(hidden_size/4 -> 1)
    
    输出: 每个 Token 位置的评分 logit (用于 Accept/Reject 二分类)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        """
        初始化 Score Head
        
        Args:
            hidden_size: 输入隐藏层维度
            intermediate_size: 中间层维度，默认 hidden_size // 4
        """
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size // 4
        
        self.intermediate_size = intermediate_size
        
        # 两层 MLP: hidden_size -> intermediate_size -> 1
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_size, 1, bias=True)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化 (Xavier 初始化)"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [Batch, Seq_Len, Hidden_Size] - Transformer 输出的隐藏状态
        
        Returns:
            logits: [Batch, Seq_Len, 1] - 每个位置的评分 logit
        """
        # [Batch, Seq_Len, Hidden_Size] -> [Batch, Seq_Len, Intermediate_Size]
        x = self.fc1(hidden_states)
        x = self.activation(x)
        # [Batch, Seq_Len, Intermediate_Size] -> [Batch, Seq_Len, 1]
        logits = self.fc2(x)
        
        return logits


# ==============================================================================
# 3. ScoringActor 主模型
# ==============================================================================

class ScoringActor(nn.Module):
    """
    Speculative Decoding 评分模型 (ScoringActor)
    
    基于 Llama-3-8B + LoRA + Score Head 的自定义模型
    
    输入格式: [Context Tokens] + [SEP] + [Draft Tokens] + [SEP] + [Target Tokens] + [SEP]
    
    核心功能:
        1. 使用 LoRA 微调 Backbone
        2. 替换 lm_head 为 score_head (二分类输出)
        3. Mismatch 聚焦策略: Draft == Target 时强制接受
    
    Forward 输出:
        - logits: [Batch, Draft_Seq_Len, 1] - 每个 Draft Token 位置的 Accept/Reject logit
        - probs: [Batch, Draft_Seq_Len] - 经过 sigmoid 和 mismatch mask 修正后的接受概率
        - match_mask: [Batch, Draft_Seq_Len] - Draft == Target 的掩码
    """
    
    def __init__(self, config: ScoringModelConfig, backbone: nn.Module = None):
        """
        初始化 ScoringActor
        
        Args:
            config: 模型配置
            backbone: 可选的预加载 backbone 模型
        """
        super().__init__()
        
        self.config = config
        
        # ----------------------------------------------------------------------
        # 步骤 1: 加载 Backbone 模型
        # ----------------------------------------------------------------------
        if backbone is not None:
            self.backbone = backbone
            logging.info("使用传入的 backbone 模型")
        else:
            self.backbone = self._load_backbone()
        
        # ----------------------------------------------------------------------
        # 步骤 2: 创建 Score Head (可训练)
        # ----------------------------------------------------------------------
        self.score_head = ScoreHead(
            hidden_size=config.hidden_size,
            intermediate_size=config.score_head_hidden_size
        )
        
        # ----------------------------------------------------------------------
        # 步骤 3: 应用 LoRA 适配器 (仅微调部分参数) 并设置 self.transformer
        # ----------------------------------------------------------------------
        self._apply_lora()
        
        # ----------------------------------------------------------------------
        # 步骤 5: 冻结非训练参数
        # ----------------------------------------------------------------------
        self._freeze_parameters()
        
        logging.info(f"ScoringActor 初始化完成:")
        logging.info(f"  - Backbone: {config.model_name_or_path}")
        logging.info(f"  - LoRA Rank: {config.lora_rank}")
        logging.info(f"  - Score Head: {config.hidden_size} -> {config.score_head_hidden_size} -> 1")
    
    def _load_backbone(self) -> nn.Module:
        """
        加载预训练的 Backbone 模型
        
        Returns:
            backbone: 预训练的 Transformer 模型
        """
        from transformers import AutoModelForCausalLM, AutoConfig
        
        logging.info(f"正在加载 Backbone: {self.config.model_name_or_path}")
        
        # 加载模型配置
        model_config = AutoConfig.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True
        )
        
        # 更新 hidden_size 配置
        if hasattr(model_config, 'hidden_size'):
            self.config.hidden_size = model_config.hidden_size
            self.config.score_head_hidden_size = model_config.hidden_size // 4
        
        # 加载预训练模型
        backbone = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            config=model_config,
            torch_dtype=torch.bfloat16,  # 使用 bf16 节省显存
            trust_remote_code=True,
        )
        
        logging.info(f"Backbone 加载完成, hidden_size={self.config.hidden_size}")
        return backbone
    
    def _apply_lora(self):
        """
        应用 LoRA 适配器到 Backbone
        
        LoRA 只在指定的 target_modules 上添加低秩分解的适配器
        """
        from peft import LoraConfig, TaskType, get_peft_model
        
        # 配置 LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # 我们做特征提取而非语言建模
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",  # 不训练 bias
        )
        
        # 应用 LoRA 到 backbone
        # 注意: 我们在 transformer (内部 model) 上应用 LoRA
        if hasattr(self.backbone, 'model'):
            self.backbone.model = get_peft_model(self.backbone.model, lora_config)
            self.transformer = self.backbone.model
        else:
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.transformer = self.backbone
        
        logging.info(f"LoRA 适配器已应用: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
        logging.info(f"目标模块: {self.config.target_modules}")
    
    def _freeze_parameters(self):
        """
        冻结参数
        
        策略:
            - 冻结: Backbone 的所有原始参数 (peft 库会自动处理)
            - 可训练: Score Head 参数 (需要手动设置)
        """
        # Score Head 的参数始终可训练
        for param in self.score_head.parameters():
            param.requires_grad = True
        
        # 打印所有可训练参数名称，以供检查
        trainable_param_names = [n for n, p in self.named_parameters() if p.requires_grad]
        
        logging.info("可训练参数列表:")
        for name in trainable_param_names:
            logging.info(f"  - {name}")
        
        # 统计可训练参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info(f"参数冻结完成:")
        logging.info(f"  - 总参数量: {total_params:,}")
        logging.info(f"  - 可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def _create_match_mask(
        self,
        draft_tokens: torch.Tensor,
        target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        创建 Mismatch Mask
        
        逻辑:
            - match_mask[i] = True: 表示 draft_tokens[i] == target_tokens[i] (Match)
            - match_mask[i] = False: 表示 draft_tokens[i] != target_tokens[i] (Mismatch)
        
        Args:
            draft_tokens: [Batch, Draft_Len] - Draft 模型生成的 Token IDs
            target_tokens: [Batch, Draft_Len] - Target 模型生成的 Token IDs
        
        Returns:
            match_mask: [Batch, Draft_Len] - 布尔掩码
        """
        # 逐位置比较 Draft 和 Target Token IDs
        match_mask = (draft_tokens == target_tokens)
        return match_mask
    
    def _apply_match_mask(
        self,
        logits: torch.Tensor,
        match_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        应用 Mismatch 聚焦策略
        
        核心逻辑:
            - 对于 Match 位置 (draft == target): 强制 logit = +large_value (确保 100% 接受)
            - 对于 Mismatch 位置: 保持原始 logit (让模型学习判断)
        
        目的:
            - 在推测解码中，相同的 Token 必须被接受
            - 模型应聚焦于学习 "Mismatch 情况下是否应该接受 Draft Token"
        
        Args:
            logits: [Batch, Draft_Len, 1] - 原始评分 logits
            match_mask: [Batch, Draft_Len] - Match 位置为 True
        
        Returns:
            masked_logits: [Batch, Draft_Len, 1] - 修正后的 logits
        """
        # 扩展 match_mask 维度以匹配 logits: [Batch, Draft_Len] -> [Batch, Draft_Len, 1]
        match_mask_expanded = match_mask.unsqueeze(-1)
        
        # 创建强制接受的 logit 值
        force_accept_logit = torch.full_like(logits, self.config.mismatch_logit_value)
        
        # 使用 where 进行条件替换:
        # - match_mask=True (Match): 使用 force_accept_logit
        # - match_mask=False (Mismatch): 保持原始 logit
        masked_logits = torch.where(match_mask_expanded, force_accept_logit, logits)
        
        return masked_logits
    
    def _extract_draft_hidden_states(
        self,
        hidden_states: torch.Tensor,
        draft_start_idx: torch.Tensor,
        draft_end_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        从完整的 hidden states 中提取 Draft Token 位置的隐藏状态 (Fixed Length Optimization)
        
        假设: 所有样本的 Draft Length 相同
        
        Args:
            hidden_states: [Batch, Seq_Len, Hidden_Size]
            draft_start_idx: [Batch]
            draft_end_idx: [Batch] - 在此假设下，end - start 是常数
        
        Returns:
            draft_hidden: [Batch, Draft_Len, Hidden_Size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # 验证假设 (仅调试用，可移除)
        # draft_len = (draft_end_idx - draft_start_idx)[0].item()
        
        # 只需要计算第一个样本的 draft_len
        draft_len = (draft_end_idx[0] - draft_start_idx[0]).item()
        
        if draft_len <= 0:
             raise ValueError(f"draft_len <= 0: {draft_len}")

        # 1. 构建 gather 索引 [Batch, Draft_Len]
        # 因为长度固定，不需要 Mask 处理 Padding
        # grid: [0, 1, ..., draft_len-1]
        grid = torch.arange(draft_len, device=device).unsqueeze(0) # [1, L]
        
        # indices[b, i] = draft_start_idx[b] + i
        # draft_start_idx: [B] -> [B, 1]
        # gather_indices: [B, 1] + [1, L] -> [B, L]
        gather_indices = draft_start_idx.unsqueeze(1) + grid # [B, L]
        
        # 2. 提取 (Flat Indexing)
        # hidden_states: [B, S, H] -> [B*S, H]
        flat_hidden = hidden_states.view(-1, hidden_size)
        
        # batch_offsets: [0, S, 2S, ...]
        # [B] -> [B, 1]
        batch_offsets = (torch.arange(batch_size, device=device) * seq_len).unsqueeze(1)
        
        # flat_indices: [B, L] + [B, 1] -> [B, L] -> view(-1) -> [B*L]
        # 利用广播机制相加
        flat_indices = (gather_indices + batch_offsets).view(-1)
        
        # Gather & Reshape
        # flat_hidden[flat_indices]: [B*L, H]
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
        
        完整流程:
            1. 通过 Transformer Backbone 获取 hidden states
            2. 提取 Draft Token 位置的 hidden states
            3. 通过 Score Head 计算 logits
            4. 应用 Mismatch Mask (Match 位置强制接受)
            5. 计算接受概率
        
        Args:
            input_ids: [Batch, Seq_Len] - 完整输入 [Context + SEP + Draft + SEP + Target + SEP]
            attention_mask: [Batch, Seq_Len] - 注意力掩码
            position_ids: [Batch, Seq_Len] - 位置编码 (可选)
            draft_start_idx: [Batch] - Draft Tokens 起始位置
            draft_end_idx: [Batch] - Draft Tokens 结束位置
            draft_tokens: [Batch, Draft_Len] - Draft Token IDs
            target_tokens: [Batch, Draft_Len] - Target Token IDs
            return_dict: 是否返回字典格式
        
        Returns:
            Dict containing:
                - raw_logits: [Batch, Draft_Len, 1] - 原始评分 logits (未 mask)
                - masked_logits: [Batch, Draft_Len, 1] - 经 Mismatch Mask 修正后的 logits
                - probs: [Batch, Draft_Len] - 接受概率 (sigmoid(masked_logits))
                - match_mask: [Batch, Draft_Len] - Match 位置掩码
                - hidden_states: [Batch, Draft_Len, Hidden_Size] - Draft 位置的隐藏状态
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # ----------------------------------------------------------------------
        # Step 1: Transformer Forward
        # 通过冻结的 Backbone + LoRA 获取 hidden states
        # ----------------------------------------------------------------------
        # 处理 Attention Mask (如果是 2D Padding Mask，则升级为 4D SPD Mask)
        if attention_mask is not None and attention_mask.dim() == 2:
            if draft_start_idx is not None:
                # 使用 draft_start_idx 作为分界点 (context_len)
                # 这意味着 [0, draft_start_idx) 是 Causal 的
                # [draft_start_idx, end) 是 Bidirectional 的
                context_lens = draft_start_idx
                
                # 计算 seq_lens (Batch 中每个样本的真实长度)
                seq_lens = attention_mask.sum(dim=-1)
                
                # 获取 max_seq_len (用于 4D mask 尺寸)
                max_seq_len = input_ids.size(1)
                
                # 创建 4D Mask
                # 注意: 需要确保 draft_len/target_len 逻辑与 mask 创建一致
                # 这里主要依赖 context_lens 和 seq_lens
                attention_mask = create_spd_attention_mask(
                    context_lens=context_lens,
                    seq_lens=seq_lens,
                    padding_mask=attention_mask,
                    dtype=self.backbone.dtype if hasattr(self.backbone, "dtype") else torch.bfloat16
                )
                logging.debug("已自动将 2D attention_mask 升级为 4D SPD Mask")

        # 既然强制使用了 LoRA，self.transformer 就是 PeftModel
        # 我们直接调用 base_model 以获取原始 Transformer 的输出
        transformer_outputs = self.transformer.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,  # 训练时不使用 cache
        )
        
        # 获取最后一层的 hidden states: [Batch, Seq_Len, Hidden_Size]
        # LlamaModel 的输出一定包含 last_hidden_state
        full_hidden_states = transformer_outputs.last_hidden_state
        
        # ----------------------------------------------------------------------
        # Step 2: 提取 Draft Token 位置的 Hidden States
        # ----------------------------------------------------------------------
        if draft_start_idx is not None and draft_end_idx is not None:
            # 使用提供的位置信息提取
            draft_hidden_states = self._extract_draft_hidden_states(
                full_hidden_states, draft_start_idx, draft_end_idx
            )
        else:
            # 如果未提供位置信息，假设整个序列都是 Draft (用于测试)
            draft_hidden_states = full_hidden_states
            logging.warning("未提供 draft_start_idx/draft_end_idx，使用完整序列")
        
        # ----------------------------------------------------------------------
        # Step 3: Score Head Forward
        # 计算每个 Draft Token 的评分 logit
        # ----------------------------------------------------------------------
        # [Batch, Draft_Len, Hidden_Size] -> [Batch, Draft_Len, 1]
        raw_logits = self.score_head(draft_hidden_states)
        
        # ----------------------------------------------------------------------
        # Step 4: 创建并应用 Mismatch Mask
        # ----------------------------------------------------------------------
        
        # 创建 Match Mask: draft == target 为 True
        match_mask = self._create_match_mask(draft_tokens, target_tokens)
        
        # 应用 Mask: Match 位置强制 logit = +large_value
        masked_logits = self._apply_match_mask(raw_logits, match_mask)
        
        # ----------------------------------------------------------------------
        # Step 5: 计算接受概率
        # ----------------------------------------------------------------------
        # Sigmoid: logit -> [0, 1] 概率
        # [Batch, Draft_Len, 1] -> [Batch, Draft_Len]
        probs = torch.sigmoid(masked_logits).squeeze(-1)
        
        # ----------------------------------------------------------------------
        # 返回结果
        # ----------------------------------------------------------------------
        if return_dict:
            return {
                "raw_logits": raw_logits,           # 原始 logits (用于监督学习 loss)
                "masked_logits": masked_logits,     # 修正后的 logits (用于采样)
                "probs": probs,                      # 接受概率
                "match_mask": match_mask,            # Match 掩码 (用于 reward 计算)
                "hidden_states": draft_hidden_states # 隐藏状态 (用于调试)
            }
        else:
            return masked_logits
    
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        获取所有可训练参数
        
        Returns:
            trainable_params: 可训练参数列表
        """
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
        Args:
            config: HuggingFace PretrainedConfig
            kwargs: 其他参数
        """
        # 优先从环境变量读取配置 (由 train_spd_scorer.py 注入)
        # 这避免了在 verl 复杂的调用栈中透传参数的困难
        model_path = os.getenv("SPD_MODEL_PATH", getattr(config, "_name_or_path", "meta-llama/Llama-3-8B"))
        
        # LoRA 配置
        lora_rank = int(os.getenv("SPD_LORA_RANK", "16"))
        lora_alpha = int(os.getenv("SPD_LORA_ALPHA", "32"))
        
        # 记录一下实际使用的配置，方便调试
        logging.info(f"[AutoModelForSPDScoring] Initializing with env vars:")
        logging.info(f"  - Model Path: {model_path}")
        logging.info(f"  - LoRA Rank: {lora_rank}")
        logging.info(f"  - LoRA Alpha: {lora_alpha}")
        
        # 创建 SPD Config
        spd_config = ScoringModelConfig(
            model_name_or_path=model_path,
            hidden_size=getattr(config, 'hidden_size', 4096),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            # 其他参数...
        )
        
        # 2. 实例化 Actor
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
    
    Attention 策略:
        - [0, context_len): Causal Mask
        - [context_len, seq_len): Full Bidirectional Mask
    
    Args:
        seq_len: 当前样本的总长度
        context_len: 上下文分界点
    
    Returns:
        mask: [Seq_Len, Seq_Len] (2D)
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    # 1. Context 部分: Causal
    mask[:context_len, :context_len] = torch.tril(
        torch.ones(context_len, context_len, device=device, dtype=dtype)
    )
    
    # 2. 后半部分: Full Bidirectional
    mask[context_len:, :] = 1.0
        
    return mask


def create_hybrid_attention_mask_batch(
    context_lens: List[int],
    seq_lens: List[int],
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    is_left_padding: bool = True, # 默认为 Left Padding
) -> torch.Tensor:
    """
    批量创建混合 Attention Mask (处理变长序列)
    
    Args:
        context_lens: 每个样本的 Context 长度列表
        seq_lens: 每个样本的总长度列表
        max_seq_len: Batch 内的最大长度
        is_left_padding: 是否为 Left Padding (verl 默认 True)
    
    Returns:
        batch_mask: [Batch, 1, Max_Seq_Len, Max_Seq_Len]
    """
    batch_size = len(context_lens)
    batch_mask = torch.zeros(batch_size, 1, max_seq_len, max_seq_len, device=device, dtype=dtype)
    
    for i in range(batch_size):
        c_len = context_lens[i]
        s_len = seq_lens[i]
        
        # 生成单样本 Mask [s_len, s_len]
        single_mask = create_hybrid_attention_mask(s_len, c_len, device, dtype)
        
        if is_left_padding:
            # Left Padding: 数据靠右对齐
            # 有效数据区域: [max_seq_len - s_len : max_seq_len]
            offset = max_seq_len - s_len
            batch_mask[i, 0, offset:, offset:] = single_mask
        else:
            # Right Padding: 数据靠左对齐 (默认)
            # 有效数据区域: [0 : s_len]
            batch_mask[i, 0, :s_len, :s_len] = single_mask
        
    return batch_mask


def combine_hybrid_and_padding_mask(
    hybrid_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    将 Hybrid Mask 与 Padding Mask 结合
    
    Args:
        hybrid_mask: [Batch, 1, Seq_Len, Seq_Len] - 混合注意力 Mask (1=可见, 0=不可见)
        padding_mask: [Batch, Seq_Len] - Padding Mask (1=真实Token, 0=Padding)
    
    Returns:
        combined_mask: [Batch, 1, Seq_Len, Seq_Len] - 结合后的 Mask (1=可见, 0=不可见)
    """
    batch_size, _, seq_len, _ = hybrid_mask.shape
    device = hybrid_mask.device
    dtype = hybrid_mask.dtype
    
    # 扩展 padding_mask 为 4D
    # [Batch, Seq_Len] -> [Batch, 1, 1, Seq_Len] (Key 维度的 Mask)
    # 表示: 哪些 Key 位置是有效的 (可以被 attend 到)
    padding_mask_4d = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
    padding_mask_4d = padding_mask_4d.to(dtype)
    
    # 结合: 两者都为 1 时才能 attend
    # hybrid_mask: [B, 1, S, S] - 结构性 Mask (Causal/Bidirectional)
    # padding_mask_4d: [B, 1, 1, S] - 广播到 [B, 1, S, S]
    combined_mask = hybrid_mask * padding_mask_4d
    
    return combined_mask


def convert_mask_to_4d_attention_mask(
    combined_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    将 0/1 Mask 转换为 Hugging Face 模型可用的 4D Attention Mask
    
    Hugging Face 的 Attention 实现期望:
        - 可以 attend 的位置: 0
        - 不能 attend 的位置: 一个很大的负数 (如 -10000 或 -inf)
    
    Args:
        combined_mask: [Batch, 1, Seq_Len, Seq_Len], 1 = 可以看到, 0 = 不能看到
        dtype: 输出数据类型
    
    Returns:
        attention_mask: [Batch, 1, Seq_Len, Seq_Len], 0 = 可以看到, -large = 不能看到
    """
    # 将 1 变成 0, 0 变成 -inf (或一个很大的负数)
    # 使用 torch.finfo(dtype).min 可能导致数值问题，使用 -10000 更安全
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
    一站式创建 SPD 场景的完整 4D Attention Mask (支持 Batch 变长 Context)
    
    Args:
        context_lens: [Batch] 每个样本的 Context 长度 (分界点)
        seq_lens: [Batch] 每个样本的总长度
        padding_mask: [Batch, Max_Seq_Len] - Padding Mask
        dtype: 数据类型
    
    Returns:
        attention_mask: [Batch, 1, Max_Seq_Len, Max_Seq_Len]
    """
    batch_size = padding_mask.shape[0]
    max_seq_len = padding_mask.shape[1]
    device = padding_mask.device
    
    # 转换为 List 以便循环
    c_lens = context_lens.tolist()
    s_lens = seq_lens.tolist()
    
    # Step 1: 创建 Batch Hybrid Mask
    hybrid_mask = create_hybrid_attention_mask_batch(
        context_lens=c_lens,
        seq_lens=s_lens,
        max_seq_len=max_seq_len, # 显式传入画布大小
        device=device,
        dtype=torch.float32,
        is_left_padding=True # 显式指定 Left Padding
    )
    
    # Step 2: 与 Padding Mask 结合
    combined_mask = combine_hybrid_and_padding_mask(
        hybrid_mask=hybrid_mask,
        padding_mask=padding_mask,
    )
    
    # Step 3: 转换为 Attention Mask 格式
    attention_mask = convert_mask_to_4d_attention_mask(
        combined_mask=combined_mask,
        dtype=dtype,
    )
    
    return attention_mask


def create_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    根据 attention_mask 创建 position_ids
    
    Args:
        attention_mask: [Batch, Seq_Len]
    
    Returns:
        position_ids: [Batch, Seq_Len]
    """
    # 累积求和得到位置编码
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


