import os
import sys
import subprocess
import datasets
import pandas as pd

# 确保当前目录在sys.path中，以便导入verl
sys.path.append(os.getcwd())

def extract_solution(solution_str):
    """从OpenR1-Math数据中提取boxed答案。
    OpenR1通常遵循DeepSeek R1格式，答案在 \\boxed{} 中。
    """
    if not solution_str:
        return ""
    
    idx = solution_str.rfind("\\boxed")
    if idx < 0:
        return solution_str
    
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
    下载并预处理 OpenR1-Math-220k 完整数据集。
    """
    print(f"正在准备数据，目标目录: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = "open-r1/OpenR1-Math-220k"
    
    try:
        print(f"正在加载完整数据集: {dataset_name}")
        # 加载完整数据集
        dataset = datasets.load_dataset(dataset_name, split="train")
        print(f"成功加载 {len(dataset)} 条样本。")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return None, None

    system_prompt = "Please reason step by step and put your final answer within \\boxed{}."

    # 处理数据
    processed_data = []
    # 即使是全量数据，为了演示效率，我们这里也可以只取一部分，或者全量
    # 考虑到 Qwen 0.5B 训练很快，我们取 20000 条做演示，让你能在一个小时内看到明显变化
    # 如果你想跑全量 220k，注释掉下面这行切片即可
    # dataset = dataset.select(range(20000)) 
    
    print(f"开始处理 {len(dataset)} 条数据...")
    
    # 批量处理以提高速度
    def process_batch(batch):
        new_data = {
            "data_source": [],
            "prompt": [],
            "ability": [],
            "reward_model": [],
            "extra_info": []
        }
        for prob, sol in zip(batch['problem'], batch['solution']): # OpenR1 字段名
            ground_truth = extract_solution(sol)
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prob}
            ]
            new_data["data_source"].append("lighteval/MATH")
            new_data["prompt"].append(prompt_messages)
            new_data["ability"].append("math")
            new_data["reward_model"].append({"style": "rule", "ground_truth": ground_truth})
            new_data["extra_info"].append({"split": "train"})
        return new_data

    # 使用 map 进行并行处理 (如果数据量很大)
    # 这里为了简单直接转 pandas
    # 注意：OpenR1-Math-220k 数据字段可能是 'problem' 和 'solution' 或者 'question' 'response'
    # 我们做一个简单的适配
    data_list = []
    for i, item in enumerate(dataset):
        if i % 10000 == 0:
            print(f"已处理 {i} 条...")
        
        q = item.get('problem', item.get('question'))
        a = item.get('solution', item.get('response'))
        
        if not q or not a: continue

        processed_data.append({
            "data_source": "lighteval/MATH",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule", 
                "ground_truth": extract_solution(a)
            },
            "extra_info": {"split": "train", "index": i}
        })

    df = pd.DataFrame(processed_data)
    
    # 95% 训练，5% 验证
    train_size = int(len(df) * 0.95)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
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

    # 2. 设置训练参数
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 3. 构造启动命令
    # 针对 8x MI325 (256GB) 的豪华配置
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        
        # --- 算法核心 ---
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
        
        # --- 数据配置 ---
        f"data.train_files={train_file}",
        f"data.val_files={test_file}",
        "data.train_batch_size=2048", # 显存巨大，可以开超大 Batch Size 加速训练
        "data.max_prompt_length=2048", # 增加上下文长度
        "data.max_response_length=2048", # 允许更长的思维链
        
        # --- 模型配置 ---
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        
        # --- Rollout (推理) 配置 ---
        # 采样数 N=16 (GRPO 推荐值，显存足够大可以更大，基线更稳)
        "actor_rollout_ref.rollout.n=16", 
        "actor_rollout_ref.rollout.name=vllm",
        # MI325 256G 显存极大，不需要太吝啬，给 vLLM 0.4 足够了，剩下的给训练
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4", 
        "actor_rollout_ref.rollout.enforce_eager=True",
        # 0.5B 模型极小，单卡推理绰绰有余，TP=1 效率最高
        # 8 卡环境下，Verl 会自动开启 Data Parallel 推理 (8路并发)
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        
        # --- Actor (训练) 配置 ---
        # FSDP 训练
        "actor_rollout_ref.actor.ppo_mini_batch_size=512", # 增大 mini batch
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64", # 256G 显存可以随便开
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        # 显存足够，关闭 Offload 以获得极致速度
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        
        # --- Reference (参考模型) 配置 ---
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64",
        # 同样关闭 Ref 模型的 Offload，让它常驻显存，计算 KL 散度飞快
        "actor_rollout_ref.ref.fsdp_config.param_offload=False",
        
        # --- Rollout Log Prob 配置 ---
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64",
        
        # --- Trainer 配置 ---
        "trainer.total_epochs=3", # 跑 3 个 Epoch 观察效果
        "trainer.n_gpus_per_node=8", # 满血 8 卡
        "trainer.nnodes=1",
        "trainer.project_name=verl_grpo_full_scale",
        "trainer.experiment_name=qwen_05b_math_8gpu",
        "trainer.logger=['console']",
        # 每 10 个 step 验证一次，让你能频繁看到效果变化
        "trainer.test_freq=10",
        "trainer.save_freq=-1", # 不保存 checkpoint
    ]
    
    print("\n" + "="*50)
    print("🚀 开始运行 8卡 MI325 高性能 GRPO 训练...")
    print(f"配置: 全量数据 | Batch=2048 | Rollout N=16 | Offload=OFF")
    print("="*50 + "\n")
    
    try:
        env = os.environ.copy()
        env["HYDRA_FULL_ERROR"] = "1"
        # AMD 环境通常需要设置这个，防止多进程死锁
        env["NCCL_P2P_DISABLE"] = "1" 
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n训练过程中出错: {e}")
    except KeyboardInterrupt:
        print("\n训练被用户中断。")

if __name__ == "__main__":
    main()
