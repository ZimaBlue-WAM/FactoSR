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

"""
Combined Reward Function

This module provides a flexible mechanism to combine multiple reward functions.
Supports weighted combination of any number of reward functions.

Note: All accuracy/format rewards are handled by spatial_reward.py.
This module is mainly for combining spatial_reward with other specialized rewards.

Usage:
    - Set data_source to "accuracy+format" or "accuracy+format+other"
    - Configure weights via extra_info["weights"] = {"accuracy": 0.8, "format": 0.2}
    - Or use default_compute_score with reward_list in extra_info
"""

from typing import Any, Dict, List, Optional, Union


def compute_score(
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[float, Dict[str, Any]]:
    """
    Compute combined reward from multiple reward functions.

    Args:
        solution_str: Model output string
        ground_truth: Ground truth answer
        extra_info: Additional information containing:
            - "reward_list": List[str], list of reward names to combine (e.g., ["accuracy", "format"])
            - "weights": Dict[str, float], weights for each reward (e.g., {"accuracy": 0.8, "format": 0.2})
                         If not provided, uses equal weights
            - "return_dict": bool, whether to return detailed breakdown (default: False)
            - Other parameters will be passed to individual reward functions
        **kwargs: Additional keyword arguments

    Returns:
        float: Combined reward score or dict with detailed breakdown
    """
    if extra_info is None:
        extra_info = {}
    
    return_dict = extra_info.get("return_dict", True)  # Default True for RL training logging
    
    # Get list of rewards to combine
    # Can be specified via reward_list or inferred from data_source (e.g., "accuracy+format")
    reward_list = extra_info.get("reward_list", [])
    
    # If reward_list is empty, try to infer from data_source in kwargs or extra_info
    if not reward_list:
        data_source = kwargs.get("data_source") or extra_info.get("data_source", "")
        if "+" in str(data_source):
            reward_list = [r.strip() for r in str(data_source).split("+")]
            # Normalize short names to full names (consistent with default_compute_score)
            reward_list = [
                "spatial_xy" if r.lower() == "xy" else
                "spatial_t" if r.lower() == "t" else
                "spatial_z" if r.lower() == "z" else
                r
                for r in reward_list
            ]
        else:
            # Default to accuracy + format if nothing specified
            reward_list = ["accuracy", "format"]
    
    # Get weights for each reward (format is excluded, it's a hard gate 0/1)
    # Weights control relative contribution of each reward within a sample's combined score.
    # combined_score = format * (w_acc * acc + w_xy * spatial_xy + w_z * spatial_z + w_t * spatial_t)
    # Adjust these to control the emphasis on each spatial task during training.
    default_weights = {
        "accuracy":   1.0,   # MCQ / open-ended accuracy
        "spatial_xy": 1.0,   # Point correspondence (discrete: 0/0.5/1)
    }
    weights = extra_info.get("weights", default_weights)
    # Fill in any missing keys from default (so partial overrides still work)

    # Normalize weights to sum to 1.0 (only for weighted rewards, format excluded)
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    for k, v in default_weights.items():
        weights.setdefault(k, v)
    # Compute each reward
    reward_scores = {}
    reward_details = {}
    computed_rewards = set()
    accuracy_reward_value = None
    
    # Map reward names to their modules
    # Use spatial_reward as unified entry point for all accuracy/format/spatial rewards
    reward_modules = {}
    for reward_name in reward_list:
        reward_name_lower = reward_name.lower().strip()
        # Map short names to full names for routing
        if reward_name_lower == "z":
            reward_name_lower = "spatial_z"
        elif reward_name_lower == "xy":
            reward_name_lower = "spatial_xy"
        elif reward_name_lower == "t":
            reward_name_lower = "spatial_t"
        
        if reward_name_lower in ["accuracy", "format", "accuracy+format", "spatial_z", "spatial_xy", "spatial_t"]:
            # Use spatial_reward as unified entry point
            from . import spatial_reward
            reward_modules[reward_name] = spatial_reward
        else:
            # For other rewards, use default_compute_score to route to the correct module
            # This allows extending to any reward registered in __init__.py
            reward_modules[reward_name] = None  # Will use default_compute_score
    
    # Cascading logic: check format first if it's in the reward list
    format_reward_value = None
    if "format" in reward_list:
        # Compute format first
        format_extra_info = extra_info.copy()
        format_extra_info["return_dict"] = True
        format_extra_info["reward_type"] = "format"
        
        from . import spatial_reward
        format_result = spatial_reward.compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=format_extra_info,
            format_score=kwargs.get("format_score", 1.0),
            **kwargs,
        )
        
        if isinstance(format_result, dict):
            format_reward_value = format_result.get("format_reward", 0.0)
            reward_scores["format"] = format_result.get("score", format_reward_value)
            reward_details["format"] = format_result
        else:
            format_reward_value = float(format_result)
            reward_scores["format"] = format_reward_value
            reward_details["format"] = {"score": format_reward_value, "format_reward": format_reward_value}
        
        # If format is wrong (0.0), return 0 immediately
        if format_reward_value == 0.0:
            combined_score = 0.0
            
            if return_dict:
                result_dict = {
                    "score": 0.0,
                    "combined_score": 0.0,
                    "reward_scores": {"format": 0.0},
                    "reward_details": {"format": reward_details["format"]},
                    "weights": weights,
                    "reward_list": reward_list,
                }
                # Add format_reward to top level for logging
                result_dict["format_reward"] = 0.0
                result_dict["accuracy_reward"] = 0.0
                
                # If spatial_xy is in reward_list, set default values for r_reprojection and r_contrastive
                # to avoid NaN in batch-level aggregation when format fails
                reward_list_lower = [r.lower().strip() for r in reward_list]
                if "spatial_xy" in reward_list_lower:
                    result_dict["r_reprojection"] = 0.0
                    result_dict["r_contrastive"] = 0.0
                
                # If spatial_t is in reward_list, set default values for r_temporal_consistency and r_dynamic_camera
                if "spatial_t" in reward_list_lower:
                    result_dict["r_temporal_consistency"] = 0.0
                    result_dict["r_dynamic_camera"] = 0.0
                
                # If spatial_z is in reward_list, set default values for r_3d_iou, r_depth_consistency, etc.
                if "spatial_z" in reward_list_lower:
                    result_dict["r_3d_iou"] = 0.0
                    result_dict["r_3d_iou_f1"] = 0.0
                    result_dict["r_depth_consistency"] = 0.0
                    result_dict["r_proj"] = 0.0
                    result_dict["tp_count"] = 0
                    result_dict["precision"] = 0.0
                    result_dict["recall"] = 0.0
                
                return result_dict
            
            return 0.0
    
    # Helper: compute a single reward with caching-friendly return
    def _compute_single_reward(reward_name: str):
        reward_extra_info = extra_info.copy()
        reward_extra_info["return_dict"] = True
        reward_module = reward_modules.get(reward_name)
        reward_name_lower = reward_name.lower().strip()
        
        # Map short names to full names for reward_type
        if reward_name_lower == "z":
            reward_name_lower = "spatial_z"
        elif reward_name_lower == "xy":
            reward_name_lower = "spatial_xy"
        elif reward_name_lower == "t":
            reward_name_lower = "spatial_t"

        if reward_module is not None:
            if reward_name_lower in ["accuracy", "format", "accuracy+format", "spatial_z", "spatial_xy", "spatial_t"]:
                reward_extra_info["reward_type"] = reward_name_lower
                format_score = kwargs.get("format_score", reward_extra_info.get("format_score", 1.0))
                score = kwargs.get("score", reward_extra_info.get("score", 1.0))
                return reward_module.compute_score(
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=reward_extra_info,
                    format_score=format_score,
                    score=score,
                    **kwargs,
                )
            return reward_module.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=reward_extra_info,
                **kwargs,
            )

        from verl.utils.reward_score import default_compute_score
        return default_compute_score(
            data_source=reward_name,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=reward_extra_info,
            **kwargs,
        )

    reward_list_lower = [r.lower().strip() for r in reward_list]
    has_spatial_xy = "spatial_xy" in reward_list_lower
    has_accuracy = "accuracy" in reward_list_lower

    # When spatial_xy is present, pre-compute it first so its score can gate accuracy
    if has_spatial_xy and format_reward_value == 1.0:
        xy_name = next(r for r in reward_list if r.lower().strip() == "spatial_xy")
        try:
            xy_result = _compute_single_reward(xy_name)
            if isinstance(xy_result, dict):
                reward_scores[xy_name] = xy_result.get("score", 0.0)
                reward_details[xy_name] = xy_result
            else:
                reward_scores[xy_name] = float(xy_result)
                reward_details[xy_name] = {"score": float(xy_result)}
        except Exception as e:
            print(f"Warning: Reward function 'spatial_xy' failed: {e}")
            reward_scores[xy_name] = 0.0
            reward_details[xy_name] = {"score": 0.0, "error": str(e)}
        computed_rewards.add(xy_name)

        # Use spatial_xy_score as a hard gate for accuracy:
        # Only compute (and count) accuracy reward when spatial_xy_score > 0
        # (spatial_xy=0 means label is wrong, so accuracy also meaningless)
        spatial_xy_score = reward_scores[xy_name]
        spatial_xy_gate = extra_info.get("spatial_xy_accuracy_gate", 0.5)  # default: any nonzero passes
        if has_accuracy:
            acc_name = next(r for r in reward_list if r.lower().strip() == "accuracy")
            if spatial_xy_score > spatial_xy_gate:
                try:
                    acc_result = _compute_single_reward(acc_name)
                    if isinstance(acc_result, dict):
                        accuracy_reward_value = acc_result.get("accuracy_reward", 0.0)
                        reward_scores[acc_name] = acc_result.get("score", accuracy_reward_value)
                        reward_details[acc_name] = acc_result
                    else:
                        accuracy_reward_value = float(acc_result)
                        reward_scores[acc_name] = accuracy_reward_value
                        reward_details[acc_name] = {"score": accuracy_reward_value, "accuracy_reward": accuracy_reward_value}
                except Exception as e:
                    print(f"Warning: Reward function 'accuracy' failed: {e}")
                    reward_scores[acc_name] = 0.0
                    reward_details[acc_name] = {"score": 0.0, "error": str(e)}
            else:
                # spatial_xy gated accuracy out
                reward_scores[acc_name] = 0.0
                reward_details[acc_name] = {
                    "score": 0.0,
                    "accuracy_reward": 0.0,
                    "gated_by_spatial_xy": True,
                    "spatial_xy_score": float(spatial_xy_score),
                }
            computed_rewards.add(acc_name)

    elif has_accuracy and accuracy_reward_value is None:
        acc_name = next(r for r in reward_list if r.lower().strip() == "accuracy")
        acc_result = _compute_single_reward(acc_name)
        if isinstance(acc_result, dict):
            accuracy_reward_value = acc_result.get("accuracy_reward", 0.0)
            reward_scores[acc_name] = acc_result.get("score", accuracy_reward_value)
            reward_details[acc_name] = acc_result
        else:
            accuracy_reward_value = float(acc_result)
            reward_scores[acc_name] = accuracy_reward_value
            reward_details[acc_name] = {"score": accuracy_reward_value, "accuracy_reward": accuracy_reward_value}
        computed_rewards.add(acc_name)

    # Format is correct or not in reward list, compute remaining rewards
    for reward_name in reward_list:
        # Skip format if already computed
        if reward_name == "format":
            continue
        if reward_name in computed_rewards:
            continue
        # Create a copy of extra_info for this reward
        reward_extra_info = extra_info.copy()
        reward_extra_info["return_dict"] = True  # Always get dict for combination
        
        # Call the individual reward function
        try:
            reward_module = reward_modules.get(reward_name)
            reward_name_lower = reward_name.lower().strip()
            
            if reward_name_lower == "spatial_xy":
                # Hard gating on format: format=0 → spatial_xy=0
                if format_reward_value != 1.0:
                    reward_scores[reward_name] = 0.0
                    reward_details[reward_name] = {
                        "score": 0.0,
                        "r_contrastive": 0.0,
                        "r_reprojection": 0.0,
                        "gated_by_format": True,
                    }
                    computed_rewards.add(reward_name)
                    continue

            reward_result = _compute_single_reward(reward_name)
            
            # Handle both dict and float returns
            if isinstance(reward_result, dict):
                reward_scores[reward_name] = reward_result.get("score", reward_result.get("accuracy_reward", reward_result.get("format_reward", 0.0)))
                reward_details[reward_name] = reward_result
                if reward_name_lower == "accuracy":
                    accuracy_reward_value = reward_result.get("accuracy_reward", reward_scores[reward_name])
            else:
                reward_scores[reward_name] = float(reward_result)
                reward_details[reward_name] = {"score": float(reward_result)}
                if reward_name_lower == "accuracy":
                    accuracy_reward_value = float(reward_result)
            computed_rewards.add(reward_name)
        
        except Exception as e:
            # If a reward function fails, assign 0.0
            print(f"Warning: Reward function '{reward_name}' failed: {e}")
            reward_scores[reward_name] = 0.0
            reward_details[reward_name] = {"score": 0.0, "error": str(e)}
            computed_rewards.add(reward_name)
    
    # Compute combined score
    # Format is a multiplier (0 or 1), not weighted. Only other rewards have weights.
    if "format" in reward_list and format_reward_value is not None:
        # Format is a multiplier: format_reward * (weighted sum of other rewards)
        other_rewards = [r for r in reward_list if r != "format"]
        if other_rewards:
            # Only use weights for non-format rewards
            other_rewards_weighted_sum = sum(weights.get(r, 0.0) * reward_scores[r] for r in other_rewards)
            combined_score = format_reward_value * other_rewards_weighted_sum
        else:
            # Only format in reward_list (no other rewards)
            combined_score = format_reward_value
    else:
        # No format in reward_list, use weighted sum for all rewards
        combined_score = sum(weights.get(reward, 0.0) * reward_scores[reward] for reward in reward_list)
    
    if return_dict:
        result_dict = {
            "score": combined_score,
            "combined_score": combined_score,
            "reward_scores": reward_scores,
            "reward_details": reward_details,
            "weights": weights,
            "reward_list": reward_list,
        }
        
        # Flatten reward_details to enable logging of specific sub-rewards
        # Reference: spatial_z, spatial_xy return rich dicts; extract key metrics for training logs
        target_keys = {
            "accuracy": ["accuracy_reward"],
            "format": ["format_reward"],
            "spatial_xy": ["r_contrastive", "r_reprojection"],
            "spatial_z": ["r_3d_iou", "r_3d_iou_f1", "r_depth_consistency", "r_proj", "tp_count", "precision", "recall"],
            "spatial_t": ["r_temporal_consistency", "r_dynamic_camera"],
        }
        
        for reward_name, details in reward_details.items():
            if isinstance(details, dict):
                # Get the keys we want to extract for this reward
                reward_name_lower = reward_name.lower()
                keys_to_extract = target_keys.get(reward_name_lower, [])
                
                # Also handle "accuracy+format" case which contains both accuracy_reward and format_reward
                if reward_name_lower == "accuracy+format" or reward_name_lower == "format+accuracy":
                    keys_to_extract = ["accuracy_reward", "format_reward"]
                
                # Extract only the target keys
                for key in keys_to_extract:
                    if key in details:
                        value = details[key]
                        # Only extract numeric values
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            # Use simple key name (e.g., "accuracy_reward" instead of "accuracy_accuracy_reward")
                            result_dict[key] = value
        return result_dict

    return float(combined_score)

