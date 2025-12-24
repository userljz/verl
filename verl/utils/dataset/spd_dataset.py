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

import torch
from verl.utils.dataset.rl_dataset import RLHFDataset
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

class SPDRLHFDataset(RLHFDataset):
    """
    Custom Dataset for SPD (Speculative Decoding) Scorer training.
    
    This dataset expects pre-computed 'input_ids' in the parquet data,
    which includes both the context and the answer prefix.
    It skips the standard tokenizer.apply_chat_template step used in RLHFDataset.
    """

    def __getitem__(self, item):
        """
        Retrieves a sample from the dataset.
        If 'input_ids' is present in the sample, it uses it directly (with padding).
        Otherwise, it falls back to the default RLHFDataset behavior (tokenizing 'prompt').
        """
        # Get the raw dictionary from the dataframe/dataset
        row_dict = self.dataframe[item]

        # Check if we have pre-computed input_ids (as prepared in train_spd_scorer.py)
        if "input_ids" in row_dict and row_dict["input_ids"] is not None:
            # 1. Convert input_ids to list
            input_ids_list = row_dict["input_ids"]
            # Ensure it's a list of ints
            if hasattr(input_ids_list, 'tolist'):
                 input_ids_list = input_ids_list.tolist()
            
            original_len = len(input_ids_list)
            
            # 2. 只保留最后 max_prompt_length 个 token (类似 SFT 训练时的 input_ids[-1024:])
            # 这会截掉前面的 context，保留 draft 和 target
            if original_len > self.max_prompt_length:
                truncate_len = original_len - self.max_prompt_length
                input_ids_list = input_ids_list[-self.max_prompt_length:]
            else:
                truncate_len = 0
            
            current_len = len(input_ids_list)
            
            # 3. Convert to tensor
            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            
            # 4. Apply left padding if needed (pad to max_prompt_length)
            if current_len < self.max_prompt_length:
                pad_len = self.max_prompt_length - current_len
                input_ids, attention_mask = verl_F.postprocess_data(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,  # Left padding is standard for decoder-only models in verl
                    truncation="error",  # Should not truncate since we already handled it
                )
                # Remove batch dimension -> (seq_len,)
                input_ids = input_ids[0]
                attention_mask = attention_mask[0]
            else:
                pad_len = 0
            
            # 5. Compute position_ids
            # Based on the attention mask (handling left padding correctly)
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
            
            # 6. Update row_dict with processed tensors
            row_dict["input_ids"] = input_ids
            row_dict["attention_mask"] = attention_mask
            row_dict["position_ids"] = position_ids
            
            # 7. Update extra_info indices
            # Index adjustment: 先截掉左边 truncate_len，再 left pad pad_len
            # 最终调整 = -truncate_len + pad_len
            index_adjustment = -truncate_len + pad_len
            
            if "extra_info" in row_dict and row_dict["extra_info"] is not None:
                # Adjust indices if they exist
                for key in ["draft_start_idx", "draft_end_idx", "target_start_idx", "target_end_idx"]:
                    if key in row_dict["extra_info"]:
                         row_dict["extra_info"][key] += index_adjustment

            if "rollout_info" in row_dict and row_dict["rollout_info"] is not None:
                # Adjust indices in rollout_info as well
                for key in ["draft_start_idx", "draft_end_idx", "target_start_idx", "target_end_idx"]:
                    if key in row_dict["rollout_info"]:
                        row_dict["rollout_info"][key] += index_adjustment

            # 7. Handle standard metadata fields required by Trainer/Worker
            if "extra_info" not in row_dict or row_dict["extra_info"] is None:
                row_dict["extra_info"] = dict()
            
            # Ensure 'index' exists for tracking
            index = row_dict.get("extra_info", {}).get("index", 0)
            row_dict["index"] = index
            
            # Pass through other fields like 'draft_tokens', 'target_tokens', 'is_correct_baseline'
            # These are already in row_dict and will be handled as non-tensor data (numpy objects) in collate_fn
            
            return row_dict
        
        # Fallback to default RLHFDataset behavior if input_ids is missing
        return super().__getitem__(item)

