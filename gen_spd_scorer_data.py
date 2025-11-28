import os
import json
import random
import argparse
import torch
from tqdm import tqdm
from typing import List, Dict, Any
import numpy as np

# Try to import vllm, but handle if it's not available (though for this script it's highly recommended)
from vllm import LLM, SamplingParams
VLLM_AVAILABLE = True


from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SPDDataGenerator:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
        
        # Determine SEP token ID
        if args.sep_token_id is not None:
            self.sep_token_id = args.sep_token_id
        elif hasattr(self.tokenizer, "eot_id"): # Llama 3
             self.sep_token_id = self.tokenizer.eot_id
        elif self.tokenizer.eos_token_id is not None:
            self.sep_token_id = self.tokenizer.eos_token_id
        else:
            self.sep_token_id = 0 # Fallback
        
        print(f"Using SEP Token ID: {self.sep_token_id}")

    def load_prompts(self) -> List[Dict]:
        """
        Load prompts from file.
        Expected format: JSONL with 'id' and 'prompt' (or 'instruction', 'input') fields.
        """
        prompts = []
        if not os.path.exists(self.args.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {self.args.prompt_file}")
            
        with open(self.args.prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                # Normalize prompt field
                text = data.get('prompt') or data.get('instruction') or data.get('input')
                if not text:
                    continue
                
                # Ensure ID
                if 'id' not in data:
                    data['id'] = len(prompts)
                
                prompts.append({
                    "id": data['id'],
                    "prompt": text,
                    "original_data": data
                })
        
        if self.args.num_samples > 0:
            prompts = prompts[:self.args.num_samples]
            
        print(f"Loaded {len(prompts)} prompts.")
        return prompts

    def step1_generate_original_answers(self, prompts: List[Dict]):
        """
        Use Target Model to generate original answers for all prompts.
        Saves to target_original_answer.jsonl
        """
        output_file = os.path.join(self.args.output_dir, "target_original_answer.jsonl")
        
        # Check if already exists
        if os.path.exists(output_file) and not self.args.overwrite:
            print(f"Step 1 output exists ({output_file}). Skipping generation.")
            return output_file

        print("Step 1: Generating original answers with Target Model (vLLM)...")
        
        llm = LLM(
            model=self.args.target_model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=self.args.max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        )
        
        prompt_texts = [p['prompt'] for p in prompts]
        outputs = llm.generate(prompt_texts, sampling_params)
        
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            prompt_ids = output.prompt_token_ids
            
            results.append({
                "id": prompts[i]['id'],
                "prompt": prompts[i]['prompt'],
                "prompt_ids": prompt_ids,
                "answer_text": generated_text,
                "answer_ids": token_ids
            })
            
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                
        print(f"Saved {len(results)} original answers to {output_file}")
        
        # Clean up vLLM to free memory
        import gc
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_file

    def step2_generate_drafts(self, original_answers_file: str):
        """
        Generate SPD Drafts.
        1. Read original answers.
        2. Cut at random position.
        3. Use Draft Model to generate spd_k tokens.
        """
        output_file = os.path.join(self.args.output_dir, "step2_drafts.jsonl")
        
        if os.path.exists(output_file) and not self.args.overwrite:
            print(f"Step 2 output exists ({output_file}). Skipping.")
            return output_file

        print("Step 2: Generating drafts with Draft Model (vLLM)...")
        
        # Load Original Answers
        samples = []
        with open(original_answers_file, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        
        # Prepare inputs for Draft Model
        draft_inputs = [] # (prompt + partial_answer)
        metadata = [] # Stores cut_idx, original_ids, etc.
        
        for sample in samples:
            answer_ids = sample['answer_ids']
            if len(answer_ids) < 1: # Skip empty
                continue
            
            # Use pre-computed prompt_ids if available
            if 'prompt_ids' in sample:
                prompt_ids = sample['prompt_ids']
            else:
                prompt_ids = self.tokenizer.encode(sample['prompt'], add_special_tokens=True)
            
            # Generate cuts with stride spd_k
            # We iterate through the answer to create multiple training samples from one generation trajectory.
            # Starting from 1 ensures we have at least 1 token prefix, matching the previous logic.
            for cut_idx in range(1, len(answer_ids), self.args.spd_k):
                context_ids = prompt_ids + answer_ids[:cut_idx]
                
                draft_inputs.append(context_ids)
                metadata.append({
                    "sample_id": sample['id'],
                    "original_answer_ids": answer_ids,
                    "cut_idx": cut_idx,
                    "prompt_ids": prompt_ids,
                    "context_ids": context_ids # Full context
                })
            
        # Load Draft Model
        llm = LLM(
            model=self.args.draft_model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        
        # Generate exactly spd_k tokens
        sampling_params = SamplingParams(
            temperature=1.0, # Sampling for diversity? Or 0 for deterministic draft? Usually draft is greedy or low temp. 
                             # But spd usually uses sampling. Let's use arg defaults or 1.0.
            max_tokens=self.args.spd_k,
            min_tokens=self.args.spd_k, # Force length if possible, but stop tokens might trigger
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        outputs = llm.generate(prompt_token_ids=draft_inputs, sampling_params=sampling_params)
        
        results = []
        for i, output in enumerate(outputs):
            draft_out_ids = output.outputs[0].token_ids
            
            # Validation: Check for EOS
            # User requirement: "check cut_idx + spd_k == len(draft_out)" (Wait, len of total?)
            # User said: "Check if cut_idx + spd_k == len(draft_out)" - this phrasing is slightly ambiguous.
            # Usually it means: did we successfully generate spd_k tokens without hitting EOS?
            # draft_out_ids contains ONLY the new tokens. So len(draft_out_ids) should be spd_k.
            
            if len(draft_out_ids) != self.args.spd_k:
                # EOS was hit early
                continue
                
            meta = metadata[i]
            results.append({
                "sample_id": meta['sample_id'],
                # "context_ids": meta['context_ids'], # Optimization: Don't save full context
                # "original_answer_head": meta['original_answer_ids'][:meta['cut_idx']], 
                "draft_out_ids": draft_out_ids,
                "cut_idx": meta['cut_idx']
            })
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
                
        print(f"Generated {len(results)} valid drafts. Saved to {output_file}")
        
        import gc
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_file

    def step3_target_verification(self, drafts_file: str, original_answers_file: str):
        """
        Step 3: Target Model Verification.
        Feed context + draft to Target Model.
        Get top-1 token for each draft position + 1 bonus position.
        """
        output_file = os.path.join(self.args.output_dir, "spd_training_data.jsonl")
        
        if os.path.exists(output_file) and not self.args.overwrite:
            print(f"Step 3 output exists ({output_file}). Skipping.")
            return

        print("Step 3: Verifying with Target Model (Transformers)...")
        
        # Load Original Answers for context reconstruction
        print(f"Loading original answers from {original_answers_file}...")
        original_answers = {}
        with open(original_answers_file, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                original_answers[d['id']] = d

        # Load Data
        samples = []
        with open(drafts_file, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
                
        # Load Target Model with Transformers (better control for logits)
        print(f"Loading model: {self.args.target_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.args.target_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.eval()
        
        final_data = []
        batch_size = self.args.batch_size
        
        for i in tqdm(range(0, len(samples), batch_size), desc="Verifying"):
            batch = samples[i:i+batch_size]
            
            # Prepare batch inputs
            # Input = Context + Draft
            batch_input_ids = []
            for item in batch:
                # Reconstruct context
                orig = original_answers.get(item['sample_id'])
                if not orig:
                    # Should not happen if data consistency is maintained
                    print(f"Error: Sample ID {item['sample_id']} not found.")
                    continue

                if 'prompt_ids' not in orig:
                    orig['prompt_ids'] = self.tokenizer.encode(orig['prompt'], add_special_tokens=True)
                
                context_ids = orig['prompt_ids'] + orig['answer_ids'][:item['cut_idx']]
                item['context_ids'] = context_ids # Temporarily store for logit extraction logic

                inp = context_ids + item['draft_out_ids']
                batch_input_ids.append(torch.tensor(inp))
            
            # Pad batch
            from torch.nn.utils.rnn import pad_sequence
            input_tensor = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = (input_tensor != self.tokenizer.pad_token_id).long()
            
            input_tensor = input_tensor.to(model.device)
            attention_mask = attention_mask.to(model.device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
                logits = outputs.logits # [B, Seq_Len, Vocab]
            
            # Extract target tokens
            # We need predictions for the positions corresponding to Draft Tokens + Bonus.
            # Input sequence: [C_0, ..., C_m, D_0, ..., D_{k-1}]
            # Logit at index j predicts token at j+1.
            # We want to verify D_0. Predictor is at index m (last context token).
            # Wait: index m corresponds to token C_m. Logit at m predicts C_{m+1} which is D_0.
            # So, for D_0, we look at logit[len(context)-1].
            # For D_{k-1}, we look at logit[len(context)+k-2].
            # For Bonus (D_k), we look at logit[len(context)+k-1].
            
            # So we need logits from [len(context)-1] to [len(context)+k-1] inclusive.
            # Total count: (len(context)+k-1) - (len(context)-1) + 1 = k + 1.
            
            preds = torch.argmax(logits, dim=-1) # [B, Seq_Len]
            
            for b_idx, item in enumerate(batch):
                ctx_len = len(item['context_ids'])
                draft_len = len(item['draft_out_ids']) # spd_k
                total_len = ctx_len + draft_len
                
                # Verify indices
                # We want predictions for tokens at: ctx_len (first draft) to total_len (bonus)
                # The logits predicting these are at indices: ctx_len-1 to total_len-1
                
                start_logit_idx = ctx_len - 1
                end_logit_idx = start_logit_idx + draft_len # Inclusive
                
                # Check bounds
                if end_logit_idx >= preds.shape[1]:
                    # Should not happen if input was passed correctly
                    print(f"Error: index out of bounds for sample {item['sample_id']}")
                    continue
                    
                target_out_ids = preds[b_idx, start_logit_idx : end_logit_idx + 1].tolist()
                
                # Ensure length is spd_k + 1
                if len(target_out_ids) != draft_len + 1:
                    print(f"Warning: Unexpected target length {len(target_out_ids)} for sample {item['sample_id']}")
                    continue
                    
                # Save simplified data
                final_data.append({
                    "sample_id": item['sample_id'],
                    "cut_idx": item['cut_idx'],
                    "draft_ids": item['draft_out_ids'],
                    "target_ids": target_out_ids,
                    "sep_token_id": self.sep_token_id
                })
                
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in final_data:
                f.write(json.dumps(item) + "\n")
                
        print(f"Done. Saved {len(final_data)} SPD training samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate data for SPD Scorer training")
    parser.add_argument("--target_model_path", type=str, required=True, help="Path to Target Model (Oracle)")
    parser.add_argument("--draft_model_path", type=str, required=True, help="Path to Draft Model")
    parser.add_argument("--prompt_file", type=str, required=True, help="JSONL file with prompts")
    parser.add_argument("--output_dir", type=str, default="data/spd_gen", help="Output directory")
    parser.add_argument("--spd_k", type=int, default=5, help="Gamma (number of draft tokens)")
    parser.add_argument("--num_samples", type=int, default=-1, help="Max samples to process")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for original answer generation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for verification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sep_token_id", type=int, default=None, help="Separator token ID (optional, auto-detect)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing intermediate files")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    generator = SPDDataGenerator(args)
    
    # 1. Load Prompts
    prompts = generator.load_prompts()
    
    # 2. Step 1: Target Model -> Original Answers
    answers_file = generator.step1_generate_original_answers(prompts)
    
    # 3. Step 2: Draft Model -> Drafts
    drafts_file = generator.step2_generate_drafts(answers_file)
    
    # 4. Step 3: Target Model -> Verification (Target Tokens)
    generator.step3_target_verification(drafts_file, answers_file)

if __name__ == "__main__":
    main()

