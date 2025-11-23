import os
import re
import sys
import subprocess
import datasets
import pandas as pd
import torch

# 确保当前目录在sys.path中，以便导入verl
sys.path.append(os.getcwd())

def extract_solution(solution_str):
    """从OpenR1-Math数据中提取boxed答案。
    OpenR1通常遵循DeepSeek R1格式，答案在 \\boxed{} 中。
    """
    if not solution_str:
        return ""
    
    # 简单的boxed提取逻辑
    # 寻找最后一个 \boxed{...}
    idx = solution_str.rfind("\\boxed")
    if idx < 0:
        return solution_str # 如果没有boxed，返回原字符串（可能会导致reward低，但在最小例子中可以接受）
    
    # 提取 boxed 内容
    # 这是一个简化版本，处理简单的括号匹配
    content = solution_str[idx:]
    if content.startswith("\\boxed{"):
        # 寻找匹配的 }
        count = 0
        start = 7 # len("\\boxed{")
        for i, char in enumerate(content[start:], start=start):
            if char == '{':
                count += 1
            elif char == '}':
                if count == 0:
                    return content[start:i]
                count -= 1
    
    return solution_str # Fallback

def prepare_data(output_dir="data/openr1", sample_size=200):
    """
    下载并预处理 OpenR1-Math-220k 数据集的一小部分。
    """
    print(f"正在准备数据，目标目录: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = "open-r1/OpenR1-Math-220k"
    
    try:
        # 尝试下载数据集，如果网络不通或数据集不存在，使用 fallback
        print(f"正在加载数据集: {dataset_name}")
        # 使用 streaming=True 可以快速获取前几条数据而不用下载整个数据集
        dataset = datasets.load_dataset(dataset_name, split="train", streaming=True)
        # 获取前 sample_size 条数据
        data_list = list(dataset.take(sample_size))
        print(f"成功加载 {len(data_list)} 条样本。")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("将生成一些模拟的数学数据用于演示。")
        data_list = [
            {"problem": "What is 1 + 1?", "solution": "The answer is \\boxed{2}."},
            {"problem": "Calculate 2 * 3.", "solution": "2 * 3 = 6. \\boxed{6}"},
            {"problem": "Solve for x: x + 5 = 10", "solution": "x = 10 - 5 = 5. \\boxed{5}"},
        ] * (sample_size // 3 + 1)
        data_list = data_list[:sample_size]

    # 系统提示词，引导模型进行思考并输出 boxed 答案
    system_prompt = "Please reason step by step and put your final answer within \\boxed{}."

    processed_data = []
    for idx, item in enumerate(data_list):
        # 字段名适配：OpenR1-Math 通常有 'problem' 和 'solution'
        question = item.get("problem", item.get("question", ""))
        solution = item.get("solution", item.get("response", ""))
        
        # 提取答案用于 Reward 计算
        ground_truth = extract_solution(solution)
        
        # 构造 prompt (Message list 格式)
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        processed_data.append({
            "data_source": "lighteval/MATH", # 使用 MATH 的 reward 逻辑（支持 \\boxed{}）
            "prompt": prompt_messages,
            "ability": "math",
            "reward_model": {
                "style": "rule", 
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": "train",
                "index": idx
            }
        })
    
    # 转换为 DataFrame 并保存为 Parquet
    df = pd.DataFrame(processed_data)
    
    # 划分训练集和测试集
    train_df = df.iloc[:int(len(df)*0.8)]
    test_df = df.iloc[int(len(df)*0.8):]
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    print(f"数据已保存到 {train_path} 和 {test_path}")
    return train_path, test_path

def main():
    # 1. 准备数据
    train_file, test_file = prepare_data()
    
    # 2. 设置训练参数
    # 使用 Qwen2.5-0.5B-Instruct 作为基础模型
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 3. 构造启动命令
    # 我们调用 verl.trainer.main_ppo 模块
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        
        # 算法配置：使用 GRPO
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False", # GRPO 通常不把 KL 放在 reward 里
        
        # 数据路径
        f"data.train_files={train_file}",
        f"data.val_files={test_file}",
        "data.train_batch_size=128", # 全局 batch size
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        
        # 模型配置
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        
        # Rollout 配置 (GRPO 核心)
        "actor_rollout_ref.rollout.n=5", # 每个 prompt 采样 5 个回复
        "actor_rollout_ref.rollout.name=vllm", # 使用 vllm 进行推理
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5", # 限制显存占用
        
        # Actor 训练配置
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.use_kl_loss=True", # GRPO 使用 KL loss
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        
        # Trainer 配置
        "trainer.total_epochs=1", # 演示只跑 1 个 epoch
        "trainer.n_gpus_per_node=1", # 单卡
        "trainer.nnodes=1",
        "trainer.project_name=verl_grpo_minimal_example",
        "trainer.experiment_name=qwen_05b_math",
        "trainer.logger=['console']", # 只输出到控制台，不使用 wandb
    ]
    
    print("\n" + "="*50)
    print("开始运行 GRPO 训练...")
    print("执行命令:", " ".join(cmd))
    print("="*50 + "\n")
    
    # 4. 执行训练
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n训练过程中出错: {e}")
    except KeyboardInterrupt:
        print("\n训练被用户中断。")

if __name__ == "__main__":
    # 确保 verl 依赖已安装 (transformers, torch, vllm 等)
    main()

