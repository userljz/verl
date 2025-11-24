import os
import sys
import subprocess
import datasets
import pandas as pd

# 确保当前目录在sys.path中，以便导入verl库
sys.path.append(os.getcwd())

def extract_solution(solution_str):
    """
    从数据集中提取标准答案。
    对于数学推理任务（如GSM8k, OpenR1-Math），模型通常会在 \boxed{} 中输出最终答案。
    这个函数用于从 ground_truth 字符串中提取这个 boxed 内容，以便后续 Reward Function 进行匹配打分。
    """
    if not solution_str:
        return ""
    
    # 寻找最后一个 \boxed{...} 的起始位置
    idx = solution_str.rfind("\\boxed")
    if idx < 0:
        return solution_str # 如果找不到，直接返回原字符串作为答案
    
    # 简单的括号匹配逻辑，提取 {} 内部的内容
    content = solution_str[idx:]
    if content.startswith("\\boxed{"):
        count = 0
        start = 7 # len("\\boxed{")
        for i, char in enumerate(content[start:], start=start):
            if char == '{':
                count += 1
            elif char == '}':
                if count == 0:
                    return content[start:i]
                count -= 1
    
    return solution_str

def prepare_data(output_dir="data/openr1"):
    """
    准备训练数据：下载、预处理并保存为 Parquet 格式。
    Verl 框架要求数据格式为 Parquet，并且包含特定的字段结构（prompt, reward_model等）。
    """
    print(f"正在准备数据，目标目录: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = "open-r1/OpenR1-Math-220k"
    
    try:
        print(f"正在加载完整数据集: {dataset_name}")
        # 使用 HuggingFace datasets 库加载数据
        dataset = datasets.load_dataset(dataset_name, split="train")
        print(f"成功加载 {len(dataset)} 条样本。")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return None, None

    # 系统提示词：引导模型进行思维链（Chain-of-Thought）推理，并规范输出格式
    system_prompt = "Please reason step by step and put your final answer within \\boxed{}."

    # 开始数据预处理
    print(f"开始处理 {len(dataset)} 条数据...")
    
    processed_data = []
    for i, item in enumerate(dataset):
        if i % 10000 == 0:
            print(f"已处理 {i} 条...")
        
        # 适配不同的数据集字段名
        q = item.get('problem', item.get('question'))
        a = item.get('solution', item.get('response'))
        
        if not q or not a: continue

        # 构造符合 Verl 协议的数据结构
        processed_data.append({
            # data_source 指定了使用哪个 Reward Function。
            # 'lighteval/MATH' 对应 verl/utils/reward_score/math_reward.py，支持 latex 格式的数学公式匹配
            "data_source": "lighteval/MATH", 
            
            # Prompt 必须是 Chat 格式的 list
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ],
            
            "ability": "math",
            
            # reward_model 字段包含用于计算奖励的真值（Ground Truth）
            "reward_model": {
                "style": "rule", 
                "ground_truth": extract_solution(a) # 提取出的标准答案
            },
            
            # 额外信息，用于调试或日志
            "extra_info": {"split": "train", "index": i}
        })

    # 转换为 Pandas DataFrame
    df = pd.DataFrame(processed_data)
    
    # 划分训练集和测试集 (95% 训练, 5% 验证)
    train_size = int(len(df) * 0.95)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # 保存为 Parquet 文件
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    print(f"数据已保存: 训练集 {len(train_df)} 条, 验证集 {len(test_df)} 条")
    return train_path, test_path

def main():
    # 1. 准备数据
    train_file, test_file = prepare_data()
    if not train_file:
        print("数据准备失败，退出")
        return

    # 2. 指定基础模型路径
    model_path = "meta-llama/Llama-3.1-8B-Instruct"  # Qwen/Qwen2.5-0.5B-Instruct  Qwen/Qwen3-4B-Instruct-2507  meta-llama/Llama-3.1-8B-Instruct
    
    # 关键超参数配置
    n_gpus = 8
    train_batch_size = 128
    ppo_mini_batch_size = 64
    micro_batch_size_per_gpu = 8
    rollout_n = 16
    offload = False
    vllm_gpu_memory_utilization = 0.4 # 给 vLLM 分配 40% 显存，避免初始化 OOM
    
    # 3. 构造启动命令
    # 我们通过调用 verl.trainer.main_ppo 模块来启动训练。
    # 所有的配置参数都通过 Hydra 格式传递（key=value）。
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        
        # =================================================================
        # 算法核心配置 (GRPO)
        # =================================================================
        "algorithm.adv_estimator=grpo",       # 指定使用 GRPO (Group Relative Policy Optimization) 算法
        "algorithm.use_kl_in_reward=False",   # GRPO 特性：不把 KL 散度惩罚直接加在 Reward 里，而是作为 Loss 的一部分
        "algorithm.kl_ctrl.kl_coef=0.001",    # KL 散度系数，防止模型偏离基座模型太远
        
        # =================================================================
        # 数据配置
        # =================================================================
        f"data.train_files={train_file}",     # 训练数据路径
        f"data.val_files={test_file}",       # 验证数据路径
        f"data.train_batch_size={train_batch_size}",         # 全局 Batch Size：每次更新参数时使用的数据量（Prompt数量）。越大越稳。
        "data.max_prompt_length=4096",        # 最大输入长度（问题长度），设大一点防止截断
        "data.max_response_length=4096",      # 最大输出长度（思维链长度），GRPO 需要模型输出很长的思考过程
        
        # =================================================================
        # 模型配置
        # =================================================================
        f"actor_rollout_ref.model.path={model_path}", # 模型路径
        "actor_rollout_ref.model.use_remove_padding=True", # 开启去 Padding 优化，极大提升训练效率
        
        # =================================================================
        # Rollout (推理/生成) 配置
        # GRPO 的核心在于：对于同一个问题，生成一组（Group）不同的回答
        # =================================================================
        f"actor_rollout_ref.rollout.n={rollout_n}",     # 关键参数：每个 Prompt 采样 {rollout_n} 个回答。GRPO 会对比这 {rollout_n} 个回答来计算优势。
        "actor_rollout_ref.rollout.name=vllm",# 使用 vLLM 作为推理引擎，速度极快
        f"actor_rollout_ref.rollout.gpu_memory_utilization={vllm_gpu_memory_utilization}", # 限制 vLLM 占用 80% 显存，剩下的留给训练
        "actor_rollout_ref.rollout.free_cache_engine=False",    # 关闭 vLLM 显存卸载（避免 AMD 环境下的 sleep/wake_up 死锁），显存充足时建议关闭

        # 【新增】强制设置数据并行度为 8，确保 8 张卡都参与推理
        "actor_rollout_ref.rollout.data_parallel_size=8",

        "actor_rollout_ref.rollout.enforce_eager=True",         # AMD ROCm 环境特定优化：关闭 CUDA Graph 避免兼容性问题
        
        # 推理时的并行设置
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1", # 单个模型不做张量并行（0.5B太小了，不需要切分）
        # vLLM 特定优化参数
        "actor_rollout_ref.rollout.enable_chunked_prefill=False", # 关闭 Chunked Prefill 以避免上下文长度检查报错
        "actor_rollout_ref.rollout.max_num_batched_tokens=16384", # 允许 vLLM 一次处理更多的 Token
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro_batch_size_per_gpu}", # 计算生成文本 LogProb 时的 Batch Size
        
        # =================================================================
        # Actor (策略模型) 训练配置
        # 负责执行反向传播和参数更新
        # =================================================================
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",        # PPO 更新时的 Mini Batch。必须 <= train_batch_size ({train_batch_size})
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch_size_per_gpu}",# 每张卡每次前向传播处理的数据量（梯度累积）
        "actor_rollout_ref.actor.use_kl_loss=True",               # 开启 KL Loss 计算
        "actor_rollout_ref.actor.kl_loss_coef=0.001",             # KL Loss 的权重
        
        # FSDP (Fully Sharded Data Parallel) 优化配置
        # 因为显存足够大 (256GB)，我们关闭所有 Offload，让参数常驻显存，速度最快
        f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
        
        # =================================================================
        # Reference (参考模型) 配置
        # 用于计算 KL 散度，确保新模型不“忘本”
        # =================================================================
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro_batch_size_per_gpu}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={offload}",  # 同样关闭 Offload，计算 KL 飞快
        
        # =================================================================
        # Trainer (训练器) 全局配置
        # =================================================================
        "trainer.total_epochs=3",             # 训练 3 个 Epoch
        f"trainer.n_gpus_per_node={n_gpus}",          # 使用 {n_gpus} 张 GPU
        "trainer.nnodes=1",                   # 单机训练
        "trainer.project_name=verl_grpo_full_scale", # Wandb 项目名
        "trainer.experiment_name=qwen_05b_math_8gpu",# 实验名
        "trainer.logger=['console','wandb']", # 日志输出到控制台和wandb
        "trainer.test_freq=10",               # 每 10 个 Step 就在验证集上测一次，方便观察效果
        "trainer.save_freq=-1",               # 不保存 Checkpoint (设为 -1)
    ]
    
    offload_status = "ON" if offload else "OFF"
    print("\n" + "="*50)
    print(f"🚀 开始运行 {n_gpus}卡 MI325 高性能 GRPO 训练...")
    print(f"配置: 全量数据 | Batch={train_batch_size} | Rollout N={rollout_n} | Offload={offload_status}")
    print("="*50 + "\n")
    
    try:
        # 复制当前环境变量
        env = os.environ.copy()
        # 开启完整错误栈打印，方便调试
        env["HYDRA_FULL_ERROR"] = "1"
        # AMD 环境常见优化：禁用 NCCL P2P（有时会导致死锁），改用共享内存或 Ring 模式
        env["NCCL_P2P_DISABLE"] = "1" 
        
        # 执行训练命令
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n训练过程中出错: {e}")
    except KeyboardInterrupt:
        print("\n训练被用户中断。")

if __name__ == "__main__":
    main()
