# GRPO训练参数设置与自定义模型指南

本指南详细说明如何在GRPO训练中设置LoRA参数，以及如何使用自定义模型。

## 目录

1. [LoRA参数设置](#1-lora参数设置)
2. [自定义模型使用](#2-自定义模型使用)
3. [自定义模型接口要求](#3-自定义模型接口要求)

---

## 1. LoRA参数设置

### 1.0 默认训练模式

**重要**：verl默认使用**全量训练（Full Fine-tuning）**，而不是LoRA训练。

- **默认配置**：`actor_rollout_ref.model.lora_rank=0`（配置文件中的默认值）
- **全量训练**：当`lora_rank=0`时，训练所有模型参数
- **LoRA训练**：只有当`lora_rank > 0`时才会启用LoRA训练

如果你不设置`lora_rank`参数，系统会使用默认值0，进行全量训练。

### 1.1 LoRA简介

LoRA (Low-Rank Adaptation) 是一种参数高效的微调技术，通过在预训练权重中注入可训练的低秩矩阵来减少内存占用和计算成本。在GRPO训练中使用LoRA可以：

- 使用有限的硬件（如8x80GB GPU）训练超大规模模型（如70B+）
- 由于内存使用减少，可以使用更大的batch size
- 简化模型传输和部署，只需保存LoRA适配器

### 1.2 LoRA配置参数

#### 最小配置（最少参数）

从全量训练切换到LoRA训练，**最少需要设置2个参数**：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    # LoRA最小配置（只需这2个参数）
    actor_rollout_ref.model.lora_rank=64 \              # 必需：必须>0才能启用LoRA
    actor_rollout_ref.rollout.load_format=safetensors \ # 必需：使vLLM能加载基础模型
    # ... 其他配置保持不变
```

**说明**：
- `lora_rank`：必须设置且>0，这是启用LoRA的关键参数
- `load_format=safetensors`：必须设置，否则vLLM无法加载模型
- `lora_alpha`：有默认值16，可以不设置（但推荐显式设置）
- `target_modules`：有默认值"all-linear"，可以不设置（但推荐显式设置）

#### 推荐配置（完整参数）

虽然最小配置就能工作，但**推荐显式设置所有参数**以获得更好的控制和可读性：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    # LoRA推荐配置
    actor_rollout_ref.model.lora_rank=64 \              # LoRA rank（必须>0）
    actor_rollout_ref.model.lora_alpha=32 \            # LoRA alpha参数（推荐显式设置）
    actor_rollout_ref.model.target_modules=all-linear \ # LoRA目标模块（推荐显式设置）
    actor_rollout_ref.rollout.load_format=safetensors \ # 必需：使vLLM能加载基础模型
    # ... 其他配置
```

**参数详细说明**：

1. **`actor_rollout_ref.model.lora_rank`** (int): 
   - **必需**：必须设置为大于0的值才能启用LoRA
   - **默认值**：0（全量训练）
   - **推荐值**：
     - 小模型（0.5B）: `lora_rank >= 32`
     - 中等模型（7B）: `lora_rank >= 64`
     - 大模型（32B+）: `lora_rank >= 128`
   - 太小的rank会影响收敛速度和性能

2. **`actor_rollout_ref.rollout.load_format`** (str):
   - **必需**：必须设置为`"safetensors"`，使vLLM能够加载基础模型
   - **默认值**：无（如果不设置会导致错误）
   - 这是LoRA训练的关键配置，不能省略

3. **`actor_rollout_ref.model.lora_alpha`** (int/float):
   - **可选**：有默认值16，但推荐显式设置
   - **推荐值**：通常设置为`lora_rank`的一半或相等
     - `lora_alpha = lora_rank`（常见）
     - `lora_alpha = lora_rank / 2`（也可以）

4. **`actor_rollout_ref.model.target_modules`** (str):
   - **可选**：有默认值`"all-linear"`，但推荐显式设置
   - **选项**：
     - `"all-linear"`: 应用到所有线性层（推荐，默认值）
     - 也可以指定特定模块，如`["q_proj", "v_proj", "k_proj", "o_proj"]`

#### 可选参数

```bash
# 从预训练的LoRA适配器继续训练
actor_rollout_ref.model.lora_adapter_path=/path/to/lora_adapter \

# 排除某些模块（可选）
actor_rollout_ref.model.exclude_modules=null \

# 推荐配置（提升性能）
actor_rollout_ref.model.use_shm=True \                    # 将模型预加载到/dev/shm
actor_rollout_ref.rollout.layered_summon=True \          # 大模型推荐：逐层同步减少内存峰值
```

**`lora_adapter_path`说明**：

如果要从之前保存的LoRA适配器继续训练，设置此参数：

```bash
actor_rollout_ref.model.lora_adapter_path=/path/to/lora_adapter
```

适配器目录需要包含：
- `adapter_model.safetensors`: LoRA权重文件
- `adapter_config.json`: LoRA配置文件

**注意**：如果设置了`lora_adapter_path`，系统会自动从适配器配置中读取`lora_rank`，无需再手动设置。

### 1.3 完整LoRA训练示例

#### 示例1：从头开始训练LoRA

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    
    # 模型配置
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    
    # LoRA配置
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_shm=True \
    
    # Actor训练配置（LoRA训练建议提高学习率）
    actor_rollout_ref.actor.optim.lr=3e-6 \              # 比全量微调高一个数量级
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    
    # Rollout配置
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \  # 必需
    actor_rollout_ref.rollout.layered_summon=True \      # 大模型推荐
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    
    # Trainer配置
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=15
```

#### 示例2：从LoRA适配器继续训练

```bash
lora_adapter_path=/path/to/saved/lora_adapter

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    # ... 其他配置 ...
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.lora_adapter_path=${lora_adapter_path} \  # 从适配器加载
    # 注意：如果设置了lora_adapter_path，会自动读取rank，无需再设置lora_rank
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.rollout.load_format=safetensors \
    # ... 其他配置 ...
```

### 1.4 LoRA训练最佳实践

1. **学习率调整**：
   - LoRA训练建议将学习率提高一个数量级
   - 全量微调：`lr=1e-6` → LoRA训练：`lr=3e-6` 或 `lr=1e-5`

2. **LoRA Rank选择**：
   - 太小（<32）会导致收敛慢或性能差
   - 推荐：小模型≥32，大模型≥128
   - 测试表明：合适的rank下，LoRA训练的性能和收敛速度与全量微调接近

3. **内存优化**：
   - 大模型（70B+）或GPU内存有限（<48GB）时，启用`layered_summon=True`
   - 启用`use_shm=True`可以加速模型加载

4. **保存和部署**：
   - LoRA训练只保存适配器权重，大大减少存储空间
   - 部署时只需加载基础模型+LoRA适配器

---

## 2. 自定义模型使用

### 2.1 使用HuggingFace自定义模型

如果你的模型已经在HuggingFace上，或者符合HuggingFace的模型接口，可以直接使用：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    # 直接指定模型路径
    actor_rollout_ref.model.path=your-org/your-custom-model \
    actor_rollout_ref.model.trust_remote_code=True \    # 如果模型需要自定义代码
    # ... 其他配置 ...
```

**要求**：
- 模型必须继承自`PreTrainedModel`（通常是`PreTrainedModelForCausalLM`）
- 模型需要支持标准的HuggingFace接口
- 如果模型有自定义代码，需要设置`trust_remote_code=True`

### 2.2 FSDP后端自定义模型扩展

如果使用FSDP后端，verl支持任何符合HuggingFace接口的模型。但为了与vLLM同步权重，可能需要实现`dtensor_weight_loader`。

#### 支持的模型（已实现dtensor_weight_loader）

以下模型已经支持，可以直接使用：
- `GPT2LMHeadModel`
- `LlamaForCausalLM`
- `MistralForCausalLM`
- `Qwen2ForCausalLM`
- `DeepseekV2ForCausalLM`
- `GemmaForCausalLM`
- `Gemma2ForCausalLM`
- 等等...

#### 为自定义模型实现dtensor_weight_loader

如果你的模型不在上述列表中，需要实现`dtensor_weight_loader`以便与vLLM同步权重。

**步骤**：

1. **找到vLLM中的模型类**，复制`load_weights`方法

2. **修改为dtensor_weight_loader函数**，位置：`verl/third_party/vllm/dtensor_weight_loader.py`

```python
def your_model_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    """
    将FSDP分片的actor权重加载到vLLM模型中
    
    Args:
        actor_weights: FSDP分片的actor模型权重字典
        vllm_model: vLLM模型实例
    
    Returns:
        加载权重后的vLLM模型
    """
    from verl.third_party.vllm.dtensor_weight_loader import redistribute_dtensor
    
    params_dict = dict(vllm_model.named_parameters())
    loaded_params = set()
    
    for name, loaded_weight in actor_weights.items():
        # 对于需要特殊处理的参数（如qkv_proj）
        # 添加参数映射逻辑
        
        # 关键：使用redistribute_dtensor处理分片权重
        local_loaded_weight = redistribute_dtensor(
            param_name=name, 
            loaded_weights=loaded_weight
        )
        
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, local_loaded_weight.to(dtype=param.dtype))
        
        loaded_params.add(name)
    
    # 检查未加载的参数
    unloaded_params = params_dict.keys() - loaded_params
    if unloaded_params:
        raise RuntimeError(f"Some weights not initialized: {unloaded_params}")
    
    return vllm_model
```

3. **注册到注册表**：

```python
# 在 verl/third_party/vllm/dtensor_weight_loader.py 中
__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__ = {
    "YourModelForCausalLM": your_model_dtensor_weight_loader,
    # ... 其他模型
}
```

**参考示例**：查看`verl/third_party/vllm/dtensor_weight_loader.py`中`gemma_dtensor_weight_loader`的实现。

### 2.3 Megatron后端自定义模型扩展

如果使用Megatron后端，需要：

1. **实现ModelLayerSpec和ModelLayer**（如果模型不能用`TransformerLayerSpec`配置）

2. **在`verl/models/mcore/model_initializer.py`中注册**：

```python
# 如果模型可以用TransformerLayerSpec配置，直接使用GPTModel
# 否则需要实现自定义的ModelLayerSpec和ModelLayer
```

3. **注册模型类型**（在`verl/models/mcore/registry.py`中）：

```python
class SupportedModel(Enum):
    # ... 现有模型
    YOUR_MODEL = "YourModelForCausalLM"

# 注册配置转换器
MODEL_CONFIG_CONVERTER_REGISTRY[SupportedModel.YOUR_MODEL] = your_hf_to_mcore_config

# 注册模型初始化器
MODEL_INITIALIZER_REGISTRY[SupportedModel.YOUR_MODEL] = YourModelInitializer

# 注册forward函数
MODEL_FORWARD_REGISTRY[SupportedModel.YOUR_MODEL] = model_forward_gen()
```

---

## 3. 自定义模型接口要求

### 3.1 HuggingFace模型接口要求

如果你的模型要用于GRPO训练，需要满足以下接口要求：

#### 必需接口

1. **标准forward方法**：

```python
class YourCustomModel(PreTrainedModelForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        返回类型必须是 CausalLMOutputWithPast 或包含 logits 的元组
        """
        # 实现前向传播
        # ...
        return CausalLMOutputWithPast(
            logits=logits,  # (batch_size, seq_len, vocab_size)
            past_key_values=past_key_values,
            # ... 其他字段
        )
```

2. **生成配置**：

```python
# 模型需要支持generation_config
model.generation_config = GenerationConfig(...)
```

3. **Tokenizer兼容**：

```python
# 模型需要与tokenizer兼容
# tokenizer的vocab_size应该与模型的vocab_size匹配
assert model.config.vocab_size == tokenizer.vocab_size
```

#### 输出格式要求

模型的forward方法必须返回包含`logits`的输出：

- **返回类型1**：`CausalLMOutputWithPast`
  ```python
  return CausalLMOutputWithPast(
      logits=logits,  # (batch_size, seq_len, vocab_size)
      past_key_values=past_key_values,
  )
  ```

- **返回类型2**：元组 `(logits,)` 或 `(logits, past_key_values, ...)`
  ```python
  return (logits,)  # logits: (batch_size, seq_len, vocab_size)
  ```

#### 可选但推荐的接口

1. **支持gradient checkpointing**：

```python
# 如果模型支持gradient checkpointing，可以启用以节省内存
if self.config.enable_gradient_checkpointing:
    self.gradient_checkpointing_enable()
```

2. **支持remove padding**（如果使用`use_remove_padding=True`）：

```python
# 模型需要能够处理不规则的序列长度
# 通常通过attention_mask来处理
```

### 3.2 模型配置要求

模型的`config`类需要包含以下字段：

```python
@dataclass
class YourModelConfig(PretrainedConfig):
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None  # 如果使用GQA
    num_hidden_layers: int
    intermediate_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool = False
    
    # 其他模型特定配置
    # ...
```

### 3.3 多模态模型接口

如果你的模型是多模态模型（如视觉语言模型），需要额外支持：

```python
def forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,  # 图像输入
    image_grid_thw: Optional[torch.Tensor] = None,  # 图像网格信息
    # ... 其他参数
) -> CausalLMOutputWithPast:
    # 处理多模态输入
    # ...
```

**在verl中使用多模态模型**：

```bash
python3 -m verl.trainer.main_ppo \
    # ... 配置 ...
    actor_rollout_ref.model.path=your-multimodal-model \
    # 多模态数据会在DataProto的non_tensor_batch中传递
```

### 3.4 检查清单

在实现自定义模型时，确保：

- [ ] 模型继承自`PreTrainedModelForCausalLM`或兼容的基类
- [ ] `forward`方法返回包含`logits`的输出
- [ ] `logits`的形状是`(batch_size, seq_len, vocab_size)`
- [ ] 模型配置包含必需的字段（vocab_size, hidden_size等）
- [ ] 如果使用FSDP+vLLM，实现了`dtensor_weight_loader`（如果需要）
- [ ] 如果使用Megatron，注册了模型类型和转换器
- [ ] 模型支持`generation_config`
- [ ] Tokenizer与模型兼容

### 3.5 测试自定义模型

在正式训练前，建议先测试模型是否能正常加载和运行：

```python
from transformers import AutoModel, AutoTokenizer

# 测试模型加载
model = AutoModel.from_pretrained("your-model-path", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("your-model-path")

# 测试forward
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
assert outputs.logits.shape[-1] == model.config.vocab_size

# 测试生成
generated = model.generate(**inputs, max_length=50)
print(tokenizer.decode(generated[0]))
```

---

## 4. 完整示例：LoRA + 自定义模型

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    
    # 自定义模型配置
    actor_rollout_ref.model.path=your-org/your-custom-model \
    actor_rollout_ref.model.trust_remote_code=True \
    
    # LoRA配置
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.load_format=safetensors \
    
    # 数据配置
    data.train_files=/path/to/train.parquet \
    data.train_batch_size=16 \
    
    # Actor配置
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    
    # Rollout配置
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.name=vllm \
    
    # Trainer配置
    trainer.n_gpus_per_node=2 \
    trainer.total_epochs=15
```

---

## 5. 常见问题

### Q1: 从全量训练切换到LoRA训练，是不是只需要加一个lora参数就行了？

**A**: **不是**。最少需要设置**2个必需参数**：

1. **`actor_rollout_ref.model.lora_rank=64`**（必须>0）
2. **`actor_rollout_ref.rollout.load_format=safetensors`**（必需）

虽然`lora_alpha`和`target_modules`有默认值，但**强烈推荐显式设置**所有4个参数以获得更好的控制和可读性。

**最小配置示例**：
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.lora_rank=64 \              # 必需
    actor_rollout_ref.rollout.load_format=safetensors \ # 必需
    # ... 其他配置保持不变
```

**推荐配置示例**：
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.load_format=safetensors \
    # ... 其他配置保持不变
```

### Q2: LoRA训练时学习率应该设置多少？

**A**: LoRA训练建议将学习率提高一个数量级。如果全量微调用`1e-6`，LoRA训练可以用`3e-6`到`1e-5`。

### Q3: 如何选择LoRA rank？

**A**: 
- 小模型（<1B）: rank >= 32
- 中等模型（1B-10B）: rank >= 64  
- 大模型（>10B）: rank >= 128

太小的rank会影响性能，建议从64开始尝试。

### Q4: 自定义模型必须实现哪些接口？

**A**: 最基本的要求是：
1. 继承`PreTrainedModelForCausalLM`
2. `forward`方法返回包含`logits`的输出
3. `logits`形状为`(batch_size, seq_len, vocab_size)`

### Q5: 如何知道我的模型是否需要实现dtensor_weight_loader？

**A**: 
- 如果你的模型在`verl/third_party/vllm/dtensor_weight_loader.py`的`__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__`中已注册，则不需要
- 如果使用FSDP+vLLM且模型未注册，训练时可能会报错，需要实现

### Q6: LoRA适配器保存在哪里？

**A**: LoRA适配器会保存在checkpoint目录下的`lora_adapter`子目录中：
```
checkpoints/your_project/your_experiment/global_step_X/actor/lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

---

## 参考资源

- **LoRA文档**：`docs/advance/ppo_lora.rst`
- **FSDP扩展文档**：`docs/advance/fsdp_extension.rst`
- **Megatron扩展文档**：`docs/advance/megatron_extension.rst`
- **LoRA示例脚本**：
  - `examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`
  - `examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora_from_adapter.sh`

