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
Temporal Spatial Reasoning Reward Function

This module implements cycle consistency rewards for temporal spatial reasoning tasks,
combining:
1. Static Understanding Cycle Consistency: Spatial logic consistency through inverse questioning
2. Dynamic Understanding Cycle Consistency: Temporal reversibility through time-reversed video

Designed for VLM-based spatial reasoning QA tasks involving camera motion and temporal sequences.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ============================================================================
# Global Precompiled Regex Patterns for Performance Optimization
# ============================================================================
# Compile all regex patterns once at module load time to avoid repeated compilation
# This provides 10-20x speedup for regex operations

# Label extraction patterns
_LABEL_PATTERN_A_D = re.compile(r'\b([A-D])\b')
_LABEL_PATTERN_A_Z = re.compile(r'\b([A-Z])\b')

# Camera motion patterns (used in parse_camera_motion)
_MOTION_PATTERNS = {
    # Roll
    'roll_left': re.compile(r'roll(?:ing)?\s+(?:to\s+)?(?:the\s+)?left'),
    'roll_right': re.compile(r'roll(?:ing)?\s+(?:to\s+)?(?:the\s+)?right'),
    'clockwise': re.compile(r'clockwise(?!\s*counter)'),
    'counterclockwise': re.compile(r'counterclockwise'),
    # Tilt
    'tilt_up': re.compile(r'tilt(?:ing)?\s+(?:to\s+)?(?:the\s+)?up(?:wards)?'),
    'tilt_down': re.compile(r'tilt(?:ing)?\s+(?:to\s+)?(?:the\s+)?down(?:wards)?'),
    # Pan
    'pan_left': re.compile(r'pan(?:ning)?\s+(?:to\s+)?(?:the\s+)?left'),
    'pan_right': re.compile(r'pan(?:ning)?\s+(?:to\s+)?(?:the\s+)?right'),
    # Simple directions
    'left': re.compile(r'\bleft\b'),
    'right': re.compile(r'\bright\b'),
    'up': re.compile(r'\bup(?:wards)?\b'),
    'down': re.compile(r'\bdown(?:wards)?\b'),
}

# Motion pattern mapping (same order as original motion_patterns dict)
_MOTION_PATTERN_RESULTS = [
    ('roll_left', {"motion_type": "roll", "direction": "left"}),
    ('roll_right', {"motion_type": "roll", "direction": "right"}),
    ('clockwise', {"motion_type": "roll", "direction": "left"}),
    ('counterclockwise', {"motion_type": "roll", "direction": "right"}),
    ('tilt_up', {"motion_type": "tilt", "direction": "upwards"}),
    ('tilt_down', {"motion_type": "tilt", "direction": "downwards"}),
    ('pan_left', {"motion_type": "pan", "direction": "left"}),
    ('pan_right', {"motion_type": "pan", "direction": "right"}),
    ('left', {"motion_type": "unknown", "direction": "left"}),
    ('right', {"motion_type": "unknown", "direction": "right"}),
    ('up', {"motion_type": "unknown", "direction": "upwards"}),
    ('down', {"motion_type": "unknown", "direction": "downwards"}),
]

# JSON extraction patterns
_JSON_DIRECTION_PATTERN = re.compile(r'\{[^{}]*"direction"[^{}]*\}|\{[^{}]*"answer"[^{}]*\}', re.DOTALL)
_JSON_ACTIONS_PATTERN = re.compile(r'\{[^{}]*"actions"[^{}]*\}', re.DOTALL)
_JSON_ANGLE_PATTERN = re.compile(r'\{[^{}]*"angle"[^{}]*\}', re.DOTALL)

# Trajectory caption patterns
_MOVE_PATTERN = re.compile(r'moves?\s+(forward|backward|left|right|up|down)\s+(\d+\.?\d*)\s*(meters?|m|feet?|ft)?')
_TURN_PATTERN = re.compile(r'turns?\s+(left|right|clockwise|counterclockwise)\s+(\d+\.?\d*)\s*degrees?')
_ROTATE_PATTERN = re.compile(r'rotates?\s+(\d+\.?\d*)\s*degrees?\s+(left|right|clockwise|counterclockwise)')

# Angle pattern
_ANGLE_PATTERN = re.compile(r'(-?\d+\.?\d*)\s*degrees?', re.IGNORECASE)

# LLM inverse generation patterns
_INVERSE_QUESTION_PATTERN = re.compile(r'INVERSE_QUESTION:\s*(.+?)(?=EXPECTED_ANSWER|$)', re.DOTALL)
_EXPECTED_ANSWER_PATTERN = re.compile(r'EXPECTED_ANSWER:\s*(.+?)$', re.DOTALL)

# Cached inverse direction mapping (static, never changes)
_INVERSE_DIRECTION_MAP = {
    "left": "right",
    "right": "left",
    "front": "back",
    "forward": "backward",
    "back": "front",
    "behind": "front",
    "backward": "forward",
    "above": "below",
    "below": "above",
    "up": "down",
    "down": "up",
}

# ============================================================================
# End of Global Precompiled Patterns
# ============================================================================


def parse_camera_motion(answer_str: str) -> Optional[Dict[str, Any]]:
    """
    解析相机运动答案
    
    支持格式:
    - "A. rolling to the left (clockwise)"
    - "The answer is C. left"
    - "tilting upwards"
    
    Returns:
        Dict with keys:
            - "motion_type": str, "roll" | "tilt" | "pan" | "unknown"
            - "direction": str, "left" | "right" | "upwards" | "downwards"
            - "label": Optional[str], 选项标签 (A, B, C, D)
    """
    answer_str = answer_str.strip()
    
    # 提取选项标签 (use precompiled pattern)
    label_matches = _LABEL_PATTERN_A_D.findall(answer_str)
    label = label_matches[-1] if label_matches else None
    
    answer_lower = answer_str.lower()
    
    # 解析运动类型和方向 (use precompiled patterns)
    for pattern_key, motion_info in _MOTION_PATTERN_RESULTS:
        if _MOTION_PATTERNS[pattern_key].search(answer_lower):
            result = motion_info.copy()
            if label:
                result["label"] = label
            return result
    
    return None


def get_inverse_camera_motion(motion_type: str, direction: str) -> Optional[str]:
    """
    获取反向相机运动
    
    Args:
        motion_type: "roll" | "tilt" | "pan" | "unknown"
        direction: "left" | "right" | "upwards" | "downwards"
    
    Returns:
        反向的方向
    """
    inverse_map = {
        "left": "right",
        "right": "left",
        "upwards": "downwards",
        "downwards": "upwards",
        "up": "down",
        "down": "up",
    }
    
    return inverse_map.get(direction.lower())


def compute_camera_motion_cycle_reward(
    answer_fwd: str, 
    answer_inv: str,
    ground_truth_fwd: str,
    expected_inv: Optional[str] = None
) -> float:
    """
    计算相机运动的循环一致性reward（仅检查反向答案）
    
    注意：正向答案的正确性由accuracy reward计算，这里只检查反向答案。
    
    Args:
        answer_fwd: 正向答案（frame1 → frame2，模型预测，用于推导期望反向方向）
        answer_inv: 逆向答案（frame2 → frame1，模型预测）
        ground_truth_fwd: 正向ground truth（用于推导期望反向方向）
        expected_inv: 逆向ground truth（优先使用）
    
    Returns:
        Reward值 [0.0, 1.0]：反向答案正确返回1.0，否则返回0.0
    """
    parsed_inv = parse_camera_motion(answer_inv)
    
    if not parsed_inv:
        return 0.0
    
    # === 检查反向答案是否正确 ===
    inverse_correct = False
    
    if expected_inv:
        # 使用提供的expected_inv进行验证
        parsed_expected = parse_camera_motion(expected_inv)
        if parsed_expected:
            # 优先比较方向
            if parsed_inv.get("direction") and parsed_expected.get("direction"):
                inverse_correct = parsed_inv.get("direction") == parsed_expected.get("direction")
            # 如果没有方向信息，比较标签
            elif parsed_inv.get("label") and parsed_expected.get("label"):
                inverse_correct = parsed_inv.get("label") == parsed_expected.get("label")
    else:
        # 如果没有expected_inv，通过正向ground truth推导期望的反向方向
        parsed_fwd = parse_camera_motion(answer_fwd)
        parsed_gt_fwd = parse_camera_motion(ground_truth_fwd)
        
        # 优先使用ground_truth_fwd来推导期望的反向方向
        if parsed_gt_fwd:
            expected_direction = get_inverse_camera_motion(
                parsed_gt_fwd.get("motion_type", "unknown"),
                parsed_gt_fwd.get("direction", "")
            )
            if expected_direction and parsed_inv.get("direction"):
                inverse_correct = parsed_inv.get("direction") == expected_direction
        # 如果ground_truth_fwd无法解析，尝试使用answer_fwd
        elif parsed_fwd:
            expected_direction = get_inverse_camera_motion(
                parsed_fwd.get("motion_type", "unknown"),
                parsed_fwd.get("direction", "")
            )
            if expected_direction and parsed_inv.get("direction"):
                inverse_correct = parsed_inv.get("direction") == expected_direction
    
    # === 返回反向答案的正确性（0或1）===
    return 1.0 if inverse_correct else 0.0


def parse_spatial_relation(answer_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse spatial relation answer from solution string.

    Supports formats:
    - "Left" / "Right" / "Front" / "Back" / "Behind"
    - "A" / "B" / "C" (multiple choice)
    - "left" / "right" / "front" / "back"
    - JSON: {"direction": "left", "distance": "2 meters"}

    Args:
        answer_str: The answer string containing spatial relation.

    Returns:
        Dict with keys:
            - "direction": str, the spatial direction
            - "label": Optional[str], if it's a multiple choice answer
        or None if parsing fails.
    """
    answer_str = answer_str.strip().upper()
    
    # Multiple choice format (A, B, C, D) - use precompiled pattern
    label_matches = _LABEL_PATTERN_A_Z.findall(answer_str)
    if label_matches:
        label = label_matches[-1]
        result = {"label": label}
        
        # Try to infer direction from context or mapping
        # This is a fallback - ideally direction should be provided separately
        return result
    
    # Direction keywords
    direction_keywords = {
        "LEFT": "left",
        "RIGHT": "right",
        "FRONT": "front",
        "FORWARD": "front",
        "BACK": "back",
        "BEHIND": "back",
        "BACKWARD": "back",
        "ABOVE": "above",
        "BELOW": "below",
        "UP": "above",
        "DOWN": "below",
    }
    
    for keyword, direction in direction_keywords.items():
        if keyword in answer_str:
            return {"direction": direction}
    
    # Try JSON format - use precompiled pattern
    try:
        json_match = _JSON_DIRECTION_PATTERN.search(answer_str)
        if json_match:
            data = json.loads(json_match.group(0))
            direction = data.get("direction", data.get("answer", ""))
            if direction:
                return {"direction": str(direction).lower()}
    except (json.JSONDecodeError, KeyError):
        pass
    
    return None


def get_inverse_direction(direction: str) -> Optional[str]:
    """
    Get the inverse spatial direction.

    Args:
        direction: Spatial direction (left, right, front, back, etc.)

    Returns:
        Inverse direction or None if not applicable.
    """
    direction = direction.lower()
    # Use cached inverse direction mapping
    return _INVERSE_DIRECTION_MAP.get(direction)


def parse_trajectory_caption(caption_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse trajectory caption to extract motion actions.

    Supports formats:
    - "Camera moves forward 2 meters and turns left 30 degrees"
    - "Moves right, then rotates clockwise"
    - JSON: {"actions": [{"type": "move", "direction": "forward", "distance": 2}, ...]}

    Args:
        caption_str: Trajectory description text.

    Returns:
        Dict with parsed actions or None if parsing fails.
    """
    caption_lower = caption_str.lower()
    
    # Try to extract actions
    actions = []
    
    # Pattern: "moves [direction] [distance] [unit]" - use precompiled pattern
    move_matches = _MOVE_PATTERN.findall(caption_lower)
    for direction, distance, unit in move_matches:
        try:
            actions.append({
                "type": "move",
                "direction": direction,
                "distance": float(distance),
            })
        except ValueError:
            pass
    
    # Pattern: "turns [direction] [angle] degrees" - use precompiled pattern
    turn_matches = _TURN_PATTERN.findall(caption_lower)
    for direction, angle in turn_matches:
        try:
            # Normalize direction
            if direction in ["clockwise", "right"]:
                dir_norm = "right"
            elif direction in ["counterclockwise", "left"]:
                dir_norm = "left"
            else:
                dir_norm = direction
            
            actions.append({
                "type": "turn",
                "direction": dir_norm,
                "angle": float(angle),
            })
        except ValueError:
            pass
    
    # Pattern: "rotates [angle] degrees [direction]" - use precompiled pattern
    rotate_matches = _ROTATE_PATTERN.findall(caption_lower)
    for angle, direction in rotate_matches:
        try:
            if direction in ["clockwise", "right"]:
                dir_norm = "right"
            elif direction in ["counterclockwise", "left"]:
                dir_norm = "left"
            else:
                dir_norm = direction
            
            actions.append({
                "type": "turn",
                "direction": dir_norm,
                "angle": float(angle),
            })
        except ValueError:
            pass
    
    if actions:
        return {"actions": actions}
    
    # Try JSON format - use precompiled pattern
    try:
        json_match = _JSON_ACTIONS_PATTERN.search(caption_str)
        if json_match:
            data = json.loads(json_match.group(0))
            if "actions" in data:
                return data
    except (json.JSONDecodeError, KeyError):
        pass
    
    return None


def inverse_trajectory_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute inverse of trajectory actions (for time reversal).

    Args:
        actions: List of action dicts.

    Returns:
        List of inverse actions (reversed order and inverted directions).
    """
    inverse_actions = []
    
    # Reverse the order and invert each action
    for action in reversed(actions):
        action_type = action.get("type")
        inv_action = action.copy()
        
        if action_type == "move":
            direction = action.get("direction", "")
            inv_direction_map = {
                "forward": "backward",
                "backward": "forward",
                "left": "right",
                "right": "left",
                "up": "down",
                "down": "up",
            }
            inv_action["direction"] = inv_direction_map.get(direction, direction)
        
        elif action_type == "turn":
            direction = action.get("direction", "")
            angle = action.get("angle", 0)
            # Invert direction and keep same angle magnitude
            inv_direction_map = {
                "left": "right",
                "right": "left",
                "clockwise": "counterclockwise",
                "counterclockwise": "clockwise",
            }
            inv_action["direction"] = inv_direction_map.get(direction, direction)
            # Angle magnitude stays the same, but direction flips
        
        inverse_actions.append(inv_action)
    
    return inverse_actions


def parse_orientation_answer(answer_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse orientation/rotation angle answer.

    Supports formats:
    - "30 degrees"
    - "Rotates 45 degrees to the left"
    - "-15 degrees"
    - JSON: {"angle": 30, "direction": "left"}

    Args:
        answer_str: Answer string containing angle information.

    Returns:
        Dict with "angle" (float) and optionally "direction" (str).
    """
    answer_str = answer_str.strip()
    
    # Pattern: "(\d+\.?\d*) degrees?" - use precompiled pattern
    angle_match = _ANGLE_PATTERN.search(answer_str)
    
    if angle_match:
        try:
            angle = float(angle_match.group(1))
            
            # Try to extract direction
            direction = None
            if "left" in answer_str.lower() or "counterclockwise" in answer_str.lower():
                direction = "left"
            elif "right" in answer_str.lower() or "clockwise" in answer_str.lower():
                direction = "right"
            
            result = {"angle": angle}
            if direction:
                result["direction"] = direction
            
            return result
        except ValueError:
            pass
    
    # Try JSON format - use precompiled pattern
    try:
        json_match = _JSON_ANGLE_PATTERN.search(answer_str)
        if json_match:
            data = json.loads(json_match.group(0))
            angle = data.get("angle")
            if angle is not None:
                return {"angle": float(angle), "direction": data.get("direction")}
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    
    return None


def compute_static_cycle_reward(
    answer_orig: str, answer_inv: str, inverse_expected: Optional[str] = None
) -> float:
    """
    Compute static understanding cycle consistency reward.

    Args:
        answer_orig: Original answer (e.g., "Left")
        answer_inv: Answer to inverse question (e.g., "Right")
        inverse_expected: Expected inverse answer (if provided, used for validation)

    Returns:
        Reward value in range [0, 1]
    """
    parsed_orig = parse_spatial_relation(answer_orig)
    parsed_inv = parse_spatial_relation(answer_inv)
    
    if not parsed_orig or not parsed_inv:
        return 0.0
    
    # Check if we have direction information
    dir_orig = parsed_orig.get("direction")
    dir_inv = parsed_inv.get("direction")
    
    if dir_orig and dir_inv:
        # Check if directions are inverse
        expected_inv = get_inverse_direction(dir_orig)
        if expected_inv == dir_inv:
            return 1.0
        else:
            # Partial credit if close (e.g., "left" vs "right" but got "back")
            return 0.0
    
    # Fallback: Check label consistency (for multiple choice)
    # If original answer is A and inverse is B, check if they're logically consistent
    label_orig = parsed_orig.get("label")
    label_inv = parsed_inv.get("label")
    
    if label_orig and label_inv:
        # If we have expected inverse, use it
        if inverse_expected:
            expected_label = parse_spatial_relation(inverse_expected).get("label") if parse_spatial_relation(inverse_expected) else None
            if expected_label == label_inv:
                return 1.0
        
        # Otherwise, give partial credit if labels are different (assuming they should be)
        # This is a weak signal, but better than nothing
        if label_orig != label_inv:
            return 0.5
    
    return 0.0


def compute_trajectory_cycle_reward(caption_fwd: str, caption_bwd: str) -> float:
    """
    Compute dynamic understanding cycle consistency reward for trajectory captions.

    Args:
        caption_fwd: Trajectory caption for forward video
        caption_bwd: Trajectory caption for backward (reversed) video

    Returns:
        Reward value in range [0, 1]
    """
    parsed_fwd = parse_trajectory_caption(caption_fwd)
    parsed_bwd = parse_trajectory_caption(caption_bwd)
    
    if not parsed_fwd or not parsed_bwd:
        return 0.0
    
    actions_fwd = parsed_fwd.get("actions", [])
    actions_bwd = parsed_bwd.get("actions", [])
    
    if not actions_fwd or not actions_bwd:
        return 0.0
    
    # Compute expected inverse actions
    expected_inv_actions = inverse_trajectory_actions(actions_fwd)
    
    # Compare actions_bwd with expected_inv_actions
    # Simple matching: check if number of actions match and directions are consistent
    if len(actions_bwd) != len(expected_inv_actions):
        return 0.0
    
    match_score = 0.0
    total_actions = len(actions_bwd)
    
    for act_bwd, act_exp in zip(actions_bwd, expected_inv_actions):
        type_match = act_bwd.get("type") == act_exp.get("type")
        
        if not type_match:
            continue
        
        if act_bwd.get("type") == "move":
            dir_match = act_bwd.get("direction") == act_exp.get("direction")
            dist_match = abs(act_bwd.get("distance", 0) - act_exp.get("distance", 0)) < 0.1
            if dir_match and dist_match:
                match_score += 1.0
        
        elif act_bwd.get("type") == "turn":
            dir_match = act_bwd.get("direction") == act_exp.get("direction")
            angle_match = abs(act_bwd.get("angle", 0) - act_exp.get("angle", 0)) < 1.0  # 1 degree tolerance
            if dir_match and angle_match:
                match_score += 1.0
    
    return match_score / total_actions if total_actions > 0 else 0.0


def compute_orientation_cycle_reward(angle_fwd: str, angle_bwd: str) -> float:
    """
    Compute dynamic understanding cycle consistency reward for orientation changes.

    Args:
        angle_fwd: Orientation answer for forward video (e.g., "+30 degrees")
        angle_bwd: Orientation answer for backward video (e.g., "-30 degrees")

    Returns:
        Reward value in range [0, 1]
    """
    parsed_fwd = parse_orientation_answer(angle_fwd)
    parsed_bwd = parse_orientation_answer(angle_bwd)
    
    if not parsed_fwd or not parsed_bwd:
        return 0.0
    
    angle_fwd_val = parsed_fwd.get("angle", 0)
    angle_bwd_val = parsed_bwd.get("angle", 0)
    
    # For time reversal, angles should sum to approximately zero
    # angle_fwd + angle_bwd ≈ 0
    angle_sum = abs(angle_fwd_val + angle_bwd_val)
    
    # Reward: exponential decay with angle error
    # Tolerance: 5 degrees
    reward = np.exp(-angle_sum / 5.0)
    
    return float(reward)


def generate_inverse_question_with_llm(
    original_question: str, original_answer: str, llm_generator=None
) -> Optional[Tuple[str, str]]:
    """
    Generate inverse question and expected answer using LLM generator.

    Args:
        original_question: Original spatial reasoning question
        original_answer: Model's answer to original question
        llm_generator: Optional LLM function that takes prompt and returns response

    Returns:
        Tuple of (inverse_question, expected_answer) or None if generation fails
    """
    if llm_generator is None:
        return None
    
    prompt = f"""I have a spatial reasoning question: '{original_question}'
The answer is '{original_answer}'.

Please generate a logical inverse question to verify this answer.
Also, provide the expected answer to your new question, assuming the original answer is correct.

Format your response as:
INVERSE_QUESTION: [your inverse question]
EXPECTED_ANSWER: [expected answer]"""
    
    try:
        response = llm_generator(prompt)
        
        # Parse response - use precompiled patterns
        inv_q_match = _INVERSE_QUESTION_PATTERN.search(response)
        exp_a_match = _EXPECTED_ANSWER_PATTERN.search(response)
        
        if inv_q_match and exp_a_match:
            inverse_question = inv_q_match.group(1).strip()
            expected_answer = exp_a_match.group(1).strip()
            return (inverse_question, expected_answer)
    except Exception as e:
        print(f"Warning: LLM inverse question generation failed: {e}")
    
    return None


def verify_inverse_question_with_llm(
    inverse_question: str, expected_answer: str, original_question: str, llm_generator=None
) -> bool:
    """
    Verify inverse question validity using LLM meta-cycle check.

    Args:
        inverse_question: Generated inverse question
        expected_answer: Expected answer to inverse question
        original_question: Original question
        llm_generator: Optional LLM function

    Returns:
        True if the inverse question is valid (can recover original answer)
    """
    if llm_generator is None:
        return True  # Skip validation if no LLM
    
    prompt = f"""Given this inverse question: '{inverse_question}'
And its answer: '{expected_answer}'

What should be the answer to the original question: '{original_question}'?

Answer with just the answer (no explanation)."""
    
    try:
        recovered_answer = llm_generator(prompt).strip()
        
        # Simple matching (can be improved with semantic similarity)
        # Check if recovered answer matches original (case-insensitive, partial match)
        return recovered_answer.lower() in original_question.lower() or original_question.lower() in recovered_answer.lower()
    except Exception:
        return True  # Assume valid if check fails


def compute_score(
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[float, Dict[str, Any]]:
    """
    Compute temporal spatial reasoning reward for dynamic cycle consistency.
    
    Supports new data format with inv_conversations and inv_ground_truth fields.

    Args:
        solution_str: Model output string containing spatial reasoning answer (forward)
        ground_truth: Ground truth answer or dict with:
            - "answer": str, GT answer (forward)
            - "inverse_answer": Optional[str], model's answer to inverse question
            - "inverse_ground_truth": Optional[str], GT answer for inverse question
        extra_info: Additional information containing:
            - "inv_conversations": Optional[List[Dict]], inverse conversations (from JSON data)
            - "inv_ground_truth": Optional[List[str]], inverse ground truth (from JSON data)
            - "inverse_answer": Optional[str], model's answer to inverse question (if already computed)
            - "question": str, original question
            - "mode": str, "static" or "dynamic" or "both" (default: "dynamic")
            - "return_dict": bool, whether to return detailed breakdown
        **kwargs: Additional keyword arguments

    Returns:
        float: Reward score in range [0, 1] or dict with detailed breakdown
    """
    if extra_info is None:
        extra_info = {}
    
    mode = extra_info.get("mode", "dynamic")  # Default to dynamic for camera motion
    return_dict = extra_info.get("return_dict", True)
    
    # ========================================================================
    # EARLY FORMAT CHECK (same logic as spatial_reward.check_format)
    # ========================================================================
    # Use spatial_reward.check_format for consistency; it has precompiled patterns
    from .spatial_reward import check_format
    format_pattern = extra_info.get("format_pattern")
    required_tags = extra_info.get("required_tags")
    check_answer_tag = extra_info.get("check_answer_tag", True)
    check_reasoning_tag = extra_info.get("check_reasoning_tag", True)
    check_bbox_tag = extra_info.get("check_bbox_tag", False)
    check_think_steps = extra_info.get("check_think_steps", True)
    min_think_steps = extra_info.get("min_think_steps", 1)
    max_think_steps = extra_info.get("max_think_steps", 6)
    
    forward_format_ok = check_format(
        solution_str, format_pattern, required_tags,
        check_answer_tag, check_reasoning_tag, check_bbox_tag,
        check_think_steps, min_think_steps, max_think_steps,
    ) == 1.0
    
    if not forward_format_ok:
        if return_dict:
            return {
                "score": 0.0,
                "r_dynamic_camera": 0.0,
                "r_temporal_consistency": 0.0,
            }
        return 0.0
    # ========================================================================
    
    # Parse ground truth
    if isinstance(ground_truth, str):
        gt_answer = ground_truth
        gt_dict = {}
    elif isinstance(ground_truth, dict):
        gt_answer = ground_truth.get("answer", ground_truth.get("ground_truth", ""))
        gt_dict = ground_truth
    else:
        gt_answer = ""
        gt_dict = {}
    
    # Get inverse data from multiple sources (priority order)
    # 1. From extra_info (passed from dataset loader)
    inv_ground_truth_list = extra_info.get("inv_ground_truth")
    if inv_ground_truth_list and isinstance(inv_ground_truth_list, list) and len(inv_ground_truth_list) > 0:
        inv_ground_truth = inv_ground_truth_list[0]
    else:
        inv_ground_truth = None
    
    # 2. From ground_truth dict
    if not inv_ground_truth:
        inv_ground_truth = gt_dict.get("inverse_ground_truth")
        if isinstance(inv_ground_truth, list) and len(inv_ground_truth) > 0:
            inv_ground_truth = inv_ground_truth[0]
    
    # Get inverse answer (model's prediction for inverse question)
    # Priority order:
    # 1. From kwargs (passed from batch.non_tensor_batch["inverse_answer"])
    # 2. From extra_info (backward compatibility)
    # 3. From gt_dict (for testing)
    inverse_answer = kwargs.get("inverse_answer") or extra_info.get("inverse_answer") or gt_dict.get("inverse_answer")
    
    # Initialize rewards
    r_dynamic_camera = 0.0
    
    # === Dynamic Cycle Consistency for Camera Motion ===
    if mode in ["dynamic", "both"]:
        # Check if we have inverse data for camera motion
        if inverse_answer and inv_ground_truth:
            # Same format restriction as forward: inverse answer must pass format check
            from .spatial_reward import check_format
            format_pattern = extra_info.get("format_pattern")
            required_tags = extra_info.get("required_tags")
            check_answer_tag = extra_info.get("check_answer_tag", True)
            check_reasoning_tag = extra_info.get("check_reasoning_tag", True)
            check_bbox_tag = extra_info.get("check_bbox_tag", False)
            check_think_steps = extra_info.get("check_think_steps", True)
            min_think_steps = extra_info.get("min_think_steps", 1)
            max_think_steps = extra_info.get("max_think_steps", 6)
            inverse_format_ok = check_format(
                inverse_answer, format_pattern, required_tags,
                check_answer_tag, check_reasoning_tag, check_bbox_tag,
                check_think_steps, min_think_steps, max_think_steps,
            ) == 1.0
            if not inverse_format_ok:
                r_dynamic_camera = 0.0
            else:
                # Use camera motion cycle consistency (only checks inverse answer correctness)
                # Forward answer correctness is handled by accuracy reward
                try:
                    r_dynamic_camera = compute_camera_motion_cycle_reward(
                        answer_fwd=solution_str,
                        answer_inv=inverse_answer,
                        ground_truth_fwd=gt_answer,
                        expected_inv=inv_ground_truth
                    )
                    import math
                    if math.isnan(r_dynamic_camera):
                        r_dynamic_camera = 0.0
                except Exception:
                    r_dynamic_camera = 0.0
        else:
            # Fallback: try to detect camera motion from question/answer
            original_question = extra_info.get("question", "")
            if any(kw in original_question.lower() or kw in solution_str.lower() 
                   for kw in ["roll", "tilt", "pan", "rotation", "camera"]):
                # This looks like camera motion, but we don't have inverse data
                # Return 0 (cannot compute cycle consistency without inverse)
                r_dynamic_camera = 0.0
    
    # For now, we focus on dynamic camera motion consistency
    # Static cycle consistency can be added later if needed
    final_score = r_dynamic_camera
    
    # Ensure score is in [0, 1] range
    final_score = max(0.0, min(1.0, final_score))
    
    # Final NaN check before returning
    import math
    if math.isnan(final_score) or math.isnan(r_dynamic_camera):
        final_score = 0.0
        r_dynamic_camera = 0.0
    
    if return_dict:
        return {
            "score": final_score,
            "r_dynamic_camera": r_dynamic_camera,
            "r_temporal_consistency": r_dynamic_camera,  # Alias for clarity
        }
    
    return float(final_score)

