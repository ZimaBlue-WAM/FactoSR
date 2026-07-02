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

from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_utils import _to_info_dict, normalize_reward_extra_infos
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""
        # import time
        # t_dapo_call_start = time.time()
        batch_size = len(data)
        # print(f"[TIMING-dapo-rm] Starting reward computation for batch_size={batch_size}")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        scores = []
        overlong_reward_list = []
        overlong_list = []
        already_print_data_sources = {}

        for i in range(len(data)):
            # t_sample_start = time.time()
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            # Extract inverse_answer: decode from inverse_responses tensor (same as forward) or use pre-decoded inverse_answer
            # data_item is per-sample (DataProto[i]) so batch values are already indexed
            inverse_answer = data_item.non_tensor_batch.get("inverse_answer", None)
            if inverse_answer is None:
                # Decode from batch.batch tensors (same storage location as forward responses)
                if "inverse_responses" in data_item.batch:
                    inverse_response_ids = data_item.batch["inverse_responses"]
                    inverse_prompt_length = data_item.batch.get("inverse_prompt_lengths", torch.tensor(0)).item()
                    
                    # Use attention_mask to filter valid tokens (exactly same as forward response)
                    if "inverse_attention_mask" in data_item.batch:
                        inverse_attn_mask = data_item.batch["inverse_attention_mask"]
                        valid_response_length = inverse_attn_mask[inverse_prompt_length:].sum().item()
                        valid_response_ids = inverse_response_ids[:valid_response_length]
                    else:
                        # Fallback: no mask available, use full ids
                        valid_response_ids = inverse_response_ids
                    
                    inverse_answer = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    eos_token = self.tokenizer.eos_token
                    if eos_token and inverse_answer.endswith(eos_token):
                        inverse_answer = inverse_answer[: -len(eos_token)]

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                inverse_answer=inverse_answer,  # Pass inverse_answer as kwarg
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
            else:
                score = result
            scores.append(result)

            reward = score

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    overlong_reward_list.append(overlong_reward)
                    overlong_list.append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward
            
            # t_sample_end = time.time()
            # sample_times.append(t_sample_end - t_sample_start)
            # if (i + 1) % 100 == 0:
            #     avg_time = sum(sample_times[-100:]) / len(sample_times[-100:])
            #     print(f"[TIMING-dapo-rm] Processed {i+1}/{batch_size} samples, avg_time={avg_time*1000:.1f}ms/sample")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        
        # t_loop_end = time.time()
        # loop_duration = t_loop_end - t_loop_start
        # if sample_times:
        #     avg_sample_time = sum(sample_times) / len(sample_times)
        #     min_sample_time = min(sample_times)
        #     max_sample_time = max(sample_times)
        #     print(f"[TIMING-dapo-rm] Loop completed: {batch_size} samples in {loop_duration:.2f}s")
        #     print(f"[TIMING-dapo-rm] Per-sample stats: avg={avg_sample_time*1000:.1f}ms, min={min_sample_time*1000:.1f}ms, max={max_sample_time*1000:.1f}ms")

        # t_dapo_call_end = time.time()
        # total_duration = t_dapo_call_end - t_dapo_call_start
        # print(f"[TIMING-dapo-rm] Total DAPO reward manager call: {total_duration:.2f}s")

        # Build reward_extra_info from scores via normalize (ensures all keys have len batch_len)
        info_dicts = [_to_info_dict(s) for s in scores]
        reward_extra_info = normalize_reward_extra_infos(info_dicts)
        if overlong_reward_list:
            reward_extra_info["overlong_reward"] = np.array(overlong_reward_list, dtype=np.float32)
            reward_extra_info["overlong"] = np.array(overlong_list, dtype=np.float32)
        if "accuracy_reward" not in reward_extra_info and "score" in reward_extra_info:
            reward_extra_info["accuracy_reward"] = reward_extra_info["score"]

        # Ensure spatial_t keys exist for mixed training (for metric logging)
        # Add r_temporal_consistency, r_dynamic_camera whenever we have any spatial data (xy/z/t),
        # so the metric is always logged even when the batch has no spatial_t samples (zeros)
        reward_fn_key_vals = data.non_tensor_batch.get(self.reward_fn_key)
        if reward_fn_key_vals is not None:
            spatial_t_patterns = ("spatial_t", "temporal", "camera_motion", "cameramotion", "route_plan", "route", "trajectory", "ego_motion")
            spatial_xy_patterns = ("spatial_xy", "correspondence", "point_correspondence", "multi_view_correspondence")
            spatial_z_patterns = ("spatial_z", "spatial_3d", "3d", "scannet", "3rscan", "multi_view_3d", "grounding")
            try:
                iter_ds = reward_fn_key_vals if hasattr(reward_fn_key_vals, "__iter__") and not isinstance(reward_fn_key_vals, str) else [reward_fn_key_vals]
                has_spatial_t = any(any(p in str(ds).lower() for p in spatial_t_patterns) for ds in iter_ds)
                has_spatial_xy = any(any(p in str(ds).lower() for p in spatial_xy_patterns) for ds in iter_ds)
                has_spatial_z = any(any(p in str(ds).lower() for p in spatial_z_patterns) for ds in iter_ds)
                has_any_spatial = has_spatial_t or has_spatial_xy or has_spatial_z
            except (TypeError, AttributeError):
                has_any_spatial = False
            if has_any_spatial:
                for key in ("r_temporal_consistency", "r_dynamic_camera"):
                    if key not in reward_extra_info:
                        reward_extra_info[key] = np.zeros(batch_size, dtype=np.float32)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
