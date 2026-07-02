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

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if (
        isinstance(data_source, str)
        and (
            "spatial" in data_source.lower()
            or data_source in ["accuracy", "format", "accuracy+format", "format+accuracy"]
            or data_source.startswith("spatial_")
            or data_source in ["3d_spatial_reasoning", "scannet", "3rscan", "multi_view_3d"]
            or data_source in ["point_correspondence", "correspondence", "multi_view_correspondence"]
            or data_source in ["temporal_spatial", "camera_motion", "trajectory_caption", "ego_motion"]
            or data_source in ["accuracy_format", "format_accuracy"]
            or ("accuracy" in data_source.lower() and "format" in data_source.lower())
        )
    ):
        # Unified spatial reward function - ALL accuracy/format/spatial rewards go through spatial_reward.py
        # This is the single source of truth for all these reward types
        # Similar to gsm8k.py, provides a unified interface
        from . import spatial_reward

        # Determine reward_type from data_source
        if extra_info is None:
            extra_info = {}
        
        # Map data_source to reward_type
        data_source_lower = data_source.lower()
        if data_source_lower == "accuracy":
            reward_type = "accuracy"
        elif data_source_lower == "format":
            reward_type = "format"
        elif data_source_lower in ["accuracy+format", "format+accuracy", "accuracy_format", "format_accuracy"]:
            reward_type = "accuracy+format"
        elif "z" in data_source_lower:
            # Handle "z" as spatial_z reward (standalone or in combinations)
            if "spatial_z" in data_source_lower:
                # Already has spatial_z (e.g. from rl_tar_dataset mapping), avoid double-replace
                reward_type = data_source_lower
            elif "+" in data_source_lower:
                # For combinations with "z", replace "z" with "spatial_z"
                # e.g., "accuracy+format+z" -> "accuracy+format+spatial_z"
                reward_type = data_source_lower.replace("z", "spatial_z")
            else:
                reward_type = "spatial_z"
        elif "spatial_3d" in data_source_lower or ("3d" in data_source_lower and "spatial" in data_source_lower) or data_source in ["scannet", "3rscan", "multi_view_3d"]:
            reward_type = "spatial_z"
        elif "xy" in data_source_lower:
            # Handle "xy" as spatial_xy (standalone or in combinations like "accuracy+format+xy")
            if "spatial_xy" in data_source_lower:
                # Already has spatial_xy (e.g. from rl_tar_dataset mapping), avoid double-replace
                reward_type = data_source_lower
            elif "+" in data_source_lower:
                # For combinations with "xy", replace "xy" with "spatial_xy"
                # e.g., "accuracy+format+xy" -> "accuracy+format+spatial_xy"
                reward_type = data_source_lower.replace("xy", "spatial_xy")
            else:
                reward_type = "spatial_xy"
        elif "correspondence" in data_source_lower or "spatial_xy" in data_source_lower or data_source in ["point_correspondence", "multi_view_correspondence"]:
            reward_type = "spatial_xy"
        elif "t" in data_source_lower and not "temporal" in data_source_lower:
            # Handle "t" as spatial_t reward (standalone or in combinations, but not "temporal")
            if "+" in data_source_lower:
                # For combinations with "t", replace standalone "t" with "spatial_t"
                # Need to be careful not to replace "t" in other words
                parts = data_source_lower.split("+")
                mapped_parts = []
                for part in parts:
                    part_stripped = part.strip()
                    if part_stripped == "t":
                        mapped_parts.append("spatial_t")
                    else:
                        mapped_parts.append(part_stripped)
                reward_type = "+".join(mapped_parts)
            else:
                reward_type = "spatial_t"
        elif "temporal" in data_source_lower or ("camera" in data_source_lower and "motion" in data_source_lower) or data_source in ["trajectory_caption", "ego_motion"]:
            reward_type = "spatial_t"
        else:
            # Default: use reward_type from extra_info or default to accuracy+format
            reward_type = extra_info.get("reward_type", "accuracy+format")
        
        extra_info["reward_type"] = reward_type
        
        # Check if reward_type is a combination that includes spatial_xy/z
        # spatial_xy and spatial_z use combined_reward for complex gating logic
        # spatial_t uses spatial_reward directly (simpler accuracy check on inverse)
        if "+" in reward_type and ("spatial_xy" in reward_type or "spatial_z" in reward_type):
            from . import combined_reward
            if extra_info is None:
                extra_info = {}
            extra_info["data_source"] = reward_type
            return combined_reward.compute_score(solution_str, ground_truth, extra_info=extra_info, **kwargs)
        
        # All other reward types (including spatial_t combinations) use spatial_reward
        # Get format_score and score from kwargs or extra_info (similar to gsm8k.py)
        # Default format_score=1.0 to give equal weight to format and accuracy
        # Both format and accuracy are important for RL training
        format_score = kwargs.get("format_score", extra_info.get("format_score", 1.0))
        score = kwargs.get("score", extra_info.get("score", 1.0))
        
        res = spatial_reward.compute_score(
            solution_str, ground_truth, extra_info=extra_info, format_score=format_score, score=score, **kwargs
        )
    elif (
        isinstance(data_source, str)
        and ("+" in data_source or data_source.lower() in ["combined"])
        and (
            # Non-spatial/non-accuracy-format tasks
            ("spatial" not in data_source.lower() and "accuracy" not in data_source.lower() and "format" not in data_source.lower())
            # OR combinations that include spatial_xy/z (which need combined_reward)
            # Note: spatial_t is handled by spatial_reward directly
            or ("spatial_xy" in data_source.lower() or "spatial_z" in data_source.lower())
        )
    ):
        # Combined rewards for specialized tasks or spatial combinations
        # This handles:
        # 1. Non-spatial/non-accuracy-format tasks (original behavior)
        # 2. Combinations that include spatial_xy/z (e.g., "accuracy+format+spatial_xy")
        # Note: spatial_t combinations use spatial_reward directly
        from . import combined_reward

        # Pass data_source to combined_reward for parsing
        if extra_info is None:
            extra_info = {}
        extra_info["data_source"] = data_source
        res = combined_reward.compute_score(solution_str, ground_truth, extra_info=extra_info, **kwargs)
    
    else:
        from . import spatial_reward
        extra_info = {
        "reward_type": "accuracy+format",
        "format_pattern": r'<think>.*?</think>.*<answer>.*?</answer>',
        "required_tags": ["<think>", "<answer>"],
        "answer_key": "answer",
        }
        res = spatial_reward.compute_score(
            solution_str, ground_truth, extra_info=extra_info,
        )

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
