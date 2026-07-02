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
Spatial Reasoning Reward Function

This module implements comprehensive reward functions for spatial reasoning tasks,
including basic accuracy/format checks and advanced spatial reasoning rewards.

Reference: Similar to gsm8k.py, provides a unified interface for spatial rewards.
"""

import re
import string
import copy as cp
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

# Import spatial-specific reward modules
from verl.utils.reward_score import spatial_z, spatial_xy, spatial_t
from verl.utils.reward_score.accuracy_reward import compute_score as compute_accuracy_score

# Try to import LLM libraries (optional)
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ============================================================================
# Precompiled regex patterns for format check (performance optimization)
# ============================================================================
# Fixed patterns - compiled once at module load (used in check_format)
_BBOX_TAG_PATTERN = re.compile(r'<bbox>(.*?)</bbox>', re.DOTALL)

# Cache for dynamic format_pattern (compiled once per unique pattern string)
# format_pattern typically same across batch, so cache hit rate is high
_FORMAT_PATTERN_CACHE: Dict[str, re.Pattern] = {}
_FORMAT_PATTERN_CACHE_MAX_SIZE = 32  # Limit cache size to avoid unbounded growth


def _get_compiled_format_pattern(pattern_str: Optional[str]) -> Optional[re.Pattern]:
    """Get or create compiled regex for format_pattern. Same logic as re.search(pattern, str, re.DOTALL)."""
    if not pattern_str:
        return None
    if pattern_str in _FORMAT_PATTERN_CACHE:
        return _FORMAT_PATTERN_CACHE[pattern_str]
    compiled = re.compile(pattern_str, re.DOTALL)
    if len(_FORMAT_PATTERN_CACHE) < _FORMAT_PATTERN_CACHE_MAX_SIZE:
        _FORMAT_PATTERN_CACHE[pattern_str] = compiled
    return compiled


# ============================================================================


def extract_answer_with_vlm(
    solution_str: str,
    vlm_client: Optional[Any] = None,
    vlm_model: Optional[str] = None,
    vlm_kwargs: Optional[Dict[str, Any]] = None,
    is_multichoice: bool = False,
    question: Optional[str] = None,
) -> Optional[str]:
    """
    Extract answer from solution string using VLM/LLM.
    
    This is more robust than regex-based extraction, especially for multi-choice
    questions where the answer might be embedded in reasoning text.
    
    Args:
        solution_str: The solution text from VLM
        vlm_client: Optional VLM client (litellm or openai client)
        vlm_model: Optional model name (e.g., "gpt-4", "claude-3-opus")
        vlm_kwargs: Optional kwargs for VLM call
        is_multichoice: Whether this is a multi-choice question
        question: Optional question text for better context
    
    Returns:
        Extracted answer string or None
    """
    if not vlm_client and not vlm_model:
        return None
    
    # Build extraction prompt
    if is_multichoice:
        prompt = f"""Extract the final answer from the following model output. 
The answer should be a single letter (A, B, C, D, etc.) representing the chosen option.

Model Output:
{solution_str}

Extract only the final answer letter. If multiple letters appear, choose the one that represents the final answer.
Output format: Just the letter (e.g., "A" or "B"), nothing else."""
    else:
        # Fix f-string syntax: cannot use backslash in expression part
        question_part = f"Question: {question}\n" if question else ""
        prompt = f"""Extract the final answer from the following model output.
{question_part}Model Output:
{solution_str}

Extract only the final answer. Remove any reasoning, explanations, or intermediate steps.
Output format: Just the answer, nothing else."""
    
    try:
        # Try litellm first
        if LITELLM_AVAILABLE and vlm_model:
            if vlm_kwargs is None:
                vlm_kwargs = {}
            response = litellm.completion(
                model=vlm_model,
                messages=[{"role": "user", "content": prompt}],
                **vlm_kwargs,
            )
            extracted = response.choices[0].message.content.strip()
        
        # Fallback to OpenAI
        elif OPENAI_AVAILABLE and vlm_model:
            if vlm_client is None:
                vlm_client = openai.OpenAI()
            if vlm_kwargs is None:
                vlm_kwargs = {}
            response = vlm_client.chat.completions.create(
                model=vlm_model,
                messages=[{"role": "user", "content": prompt}],
                **vlm_kwargs,
            )
            extracted = response.choices[0].message.content.strip()
        
        # Fallback to custom client (if provided)
        elif vlm_client and hasattr(vlm_client, 'chat'):
            if vlm_kwargs is None:
                vlm_kwargs = {}
            response = vlm_client.chat.completions.create(
                model=vlm_model or "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                **vlm_kwargs,
            )
            extracted = response.choices[0].message.content.strip()
        
        else:
            return None
        
        # Clean extracted answer
        extracted = extracted.strip()
        
        # For multichoice, extract just the letter
        if is_multichoice:
            letter_match = re.search(r'\b([A-Z])\b', extracted.upper())
            if letter_match:
                return letter_match.group(1)
        
        return extracted if extracted else None
    
    except Exception as e:
        print(f"Warning: VLM extraction failed: {e}")
        return None


def extract_answer(solution_str: str, answer_key: str = "answer", 
                   is_multichoice: bool = False,
                   use_vlm: bool = False,
                   vlm_client: Optional[Any] = None,
                   vlm_model: Optional[str] = None,
                   vlm_kwargs: Optional[Dict[str, Any]] = None,
                   question: Optional[str] = None) -> Optional[str]:
    """
    Extract answer from solution string using common patterns.
    
    Similar to gsm8k.py's extract_solution, but more general.
    Supports multi-choice answers (A/B/C/D) extraction from reasoning text.
    Can use VLM for more robust extraction.
    
    Args:
        solution_str: The solution text
        answer_key: Key to extract from JSON (default: "answer")
        is_multichoice: Whether this is a multi-choice question (default: False)
        use_vlm: Whether to use VLM for extraction (default: False)
        vlm_client: Optional VLM client (litellm or openai client)
        vlm_model: Optional model name (e.g., "gpt-4", "claude-3-opus")
        vlm_kwargs: Optional kwargs for VLM call
        question: Optional question text for better context
    
    Returns:
        Extracted answer string or None
    """
    # Pattern 0: Extract from <answer>...</answer> tags (highest priority)
    # This matches the structured format: <answer>...</answer>
    # Reference: deepeyes.py and VLMEval2's structured output format
    # Supports both multi-choice (A/B/C/D) and open-ended answers (numbers, text, etc.)
    # We don't assume the answer type here - just extract what's in the tag
    # The answer can be anything: "A", "1", "left", "42", "The backpack is on the left", etc.
    answer_tag_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if answer_tag_match:
        answer = answer_tag_match.group(1).strip()
        if not answer:
            return None
        # Return the full content from <answer> tag without any assumptions about format
        # The comparison logic will handle multi-choice vs open-ended based on ground_truth
        return answer.strip()
    
    # Try VLM extraction if enabled (fallback if no <answer> tags)
    if use_vlm:
        vlm_answer = extract_answer_with_vlm(
            solution_str, vlm_client, vlm_model, vlm_kwargs, is_multichoice, question
        )
        if vlm_answer:
            return vlm_answer
        # Fallback to regex if VLM fails
    
    # Pattern 1: Extract from JSON-like structure
    json_match = re.search(rf'"{answer_key}"\s*:\s*"([^"]+)"', solution_str)
    if json_match:
        answer = json_match.group(1)
        # If multichoice, try to extract just the letter
        if is_multichoice:
            letter_match = re.search(r'\b([A-Z])\b', answer.upper())
            if letter_match:
                return letter_match.group(1)
        return answer
    
    # Pattern 2: Extract from markdown code blocks
    code_match = re.search(r'```.*?\n(.*?)\n```', solution_str, re.DOTALL)
    if code_match:
        answer = code_match.group(1).strip()
        if is_multichoice:
            letter_match = re.search(r'\b([A-Z])\b', answer.upper())
            if letter_match:
                return letter_match.group(1)
        return answer
    
    # Pattern 3: Multi-choice answer extraction (A/B/C/D)
    # This handles cases like "The answer is A" or "Answer: B" in reasoning text
    if is_multichoice:
        # Pattern 3a: "Answer: A" or "Answer is A" or "The answer is A"
        answer_patterns = [
            r'(?i)answer\s*[:\s]+\s*([A-Z])',  # "Answer: A" or "Answer is A"
            r'(?i)the\s+answer\s+is\s+([A-Z])',  # "The answer is A"
            r'(?i)option\s+([A-Z])',  # "Option A"
            r'(?i)choice\s+([A-Z])',  # "Choice A"
            r'(?i)correct\s+answer\s+is\s+([A-Z])',  # "Correct answer is A"
            r'(?i)select\s+([A-Z])',  # "Select A"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, solution_str)
            if match:
                return match.group(1).upper()
        
        # Pattern 3b: Find standalone letters (A-Z) that appear after reasoning
        # Look for patterns like "...therefore A" or "...so the answer is A"
        # Extract the last occurrence of a single letter (likely the answer)
        letter_matches = re.findall(r'\b([A-Z])\b', solution_str.upper())
        if letter_matches:
            # Filter out common words that might be capitalized
            common_words = {'I', 'A', 'THE', 'AN', 'AND', 'OR', 'BUT', 'SO', 'TO', 'OF', 'IN', 'ON', 'AT', 'FOR'}
            # Get the last few letters and pick the one that's likely the answer
            # Usually the answer appears near the end
            for letter in reversed(letter_matches[-5:]):  # Check last 5 letters
                if letter not in common_words or len(letter_matches) <= 3:
                    return letter
        
        # Pattern 3c: Check if the entire solution is just a single letter
        single_letter = solution_str.strip().upper()
        if len(single_letter) == 1 and single_letter.isalpha() and single_letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return single_letter
    
    # Pattern 4: Extract last line (fallback)
    lines = solution_str.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        # If multichoice, try to extract letter from last line
        if is_multichoice:
            letter_match = re.search(r'\b([A-Z])\b', last_line.upper())
            if letter_match:
                return letter_match.group(1)
        return last_line
    
    return None


def check_format(solution_str: str, format_pattern: Optional[str] = None, 
                required_tags: Optional[List[str]] = None,
                check_answer_tag: bool = True,
                check_reasoning_tag: bool = True,
                check_bbox_tag: bool = False,
                check_think_steps: bool = True,
                min_think_steps: int = 1,
                max_think_steps: int = 6) -> float:
    """
    Check if the solution string follows the expected format.
    
    Supports structured format checking based on prompt requirements:
    - <think>...</think> or <think>...</think> or <think>...</think> (thinking process, required)
    - <bbox>...</bbox> (bounding box coordinates, optional)
    - <answer>...</answer> (final answer, required)
    
    Args:
        solution_str: The solution string
        format_pattern: Optional regex pattern for format check
        required_tags: Optional list of required XML/HTML tags
        check_answer_tag: Whether to check for <answer>...</answer> tags (default: True)
        check_reasoning_tag: Whether to check for reasoning tags (default: True)
        check_bbox_tag: Whether to check for <bbox>...</bbox> tags (default: False, optional)
        check_think_steps: Whether to check "StepN" count inside <think> (default: True)
        min_think_steps: Minimum number of steps required (default: 1)
        max_think_steps: Maximum number of steps allowed (default: 6)
    
    Returns:
        Format reward: 1.0 if format is correct, 0.0 otherwise
    """
    # Check 0: Reasoning tag format (required)
    if check_reasoning_tag:
        # Check for <think> tags
        reasoning_tag_count_open = solution_str.count('<think>')
        reasoning_tag_count_close = solution_str.count('</think>')
        
        # Both opening and closing tags must exist and match in count
        if reasoning_tag_count_open == 0 or reasoning_tag_count_close == 0:
            return 0.0
        if reasoning_tag_count_open != reasoning_tag_count_close:
            return 0.0
        
        # Strict check: exactly one pair of <think> tags
        if reasoning_tag_count_open != 1:
            return 0.0
        
        # Check content is non-empty
        think_start = solution_str.find('<think>')
        think_end = solution_str.find('</think>')
        if think_end <= think_start + len('<think>'):
            return 0.0  # Empty content
        
        # Check step count inside <think> if required
        if check_think_steps:
            think_content = solution_str[think_start + len('<think>'):think_end]
            # Count occurrences of "Step" (case-insensitive) as step markers
            import re as _re
            step_count = len(_re.findall(r'\bstep\s*\d+', think_content, _re.IGNORECASE))
            if step_count < min_think_steps or step_count > max_think_steps:
                return 0.0
    
    # Check 1: <bbox> tag format (optional)
    # Format: <bbox>[[x1,y1,x2,y2], ...]</bbox>
    if check_bbox_tag:
        bbox_tag_count_open = solution_str.count('<bbox>')
        bbox_tag_count_close = solution_str.count('</bbox>')
        if bbox_tag_count_open != bbox_tag_count_close:
            return 0.0
        # If <bbox> tags exist, check if they contain valid coordinate format
        if bbox_tag_count_open > 0:
            bbox_match = _BBOX_TAG_PATTERN.search(solution_str)
            if not bbox_match or not bbox_match.group(1).strip():
                return 0.0
            # Try to validate bbox format (should contain brackets and numbers)
            bbox_content = bbox_match.group(1).strip()
            # Basic check: should contain brackets and numbers
            if not (('[' in bbox_content or '(' in bbox_content) and 
                    any(c.isdigit() for c in bbox_content)):
                return 0.0
    
    # Check 2: <answer> tag format (required)
    if check_answer_tag:
        answer_tag_count_open = solution_str.count('<answer>')
        answer_tag_count_close = solution_str.count('</answer>')
        
        # Both opening and closing tags must exist and match in count
        if answer_tag_count_open == 0 or answer_tag_count_close == 0:
            return 0.0
        if answer_tag_count_open != answer_tag_count_close:
            return 0.0
        
        # Strict check: exactly one pair of <answer> tags
        if answer_tag_count_open != 1:
            return 0.0
        
        # Check content is non-empty
        answer_start = solution_str.find('<answer>')
        answer_end = solution_str.find('</answer>')
        if answer_end <= answer_start + len('<answer>'):
            return 0.0  # Empty content
        
        # Check order: <think> must come before <answer> (if both are checked)
        if check_reasoning_tag:
            think_end = solution_str.find('</think>')
            if think_end > answer_start:
                return 0.0  # <think> ends after <answer> starts - wrong order
    
    # Check 3: Custom regex pattern (use precompiled/cached pattern)
    compiled_format = _get_compiled_format_pattern(format_pattern)
    if compiled_format is not None:
        if not compiled_format.search(solution_str):
            return 0.0
    
    # Check 4: Required tags (additional custom tags)
    if required_tags:
        for tag in required_tags:
            if tag not in solution_str:
                return 0.0
    
    # Check 5: Empty response
    if not solution_str or not solution_str.strip():
        return 0.0
    
    # Check 6: Balanced brackets (for JSON/bbox format)
    open_brackets = solution_str.count('[') + solution_str.count('(') + solution_str.count('{')
    close_brackets = solution_str.count(']') + solution_str.count(')') + solution_str.count('}')
    if abs(open_brackets - close_brackets) > 2:
        return 0.0
    
    # Check 7: JSON structure (if starts with JSON)
    if solution_str.strip().startswith('{') or solution_str.strip().startswith('['):
        if solution_str.count('{') != solution_str.count('}') or solution_str.count('[') != solution_str.count(']'):
            return 0.0
    
    return 1.0


def can_infer_option_vlmeval2(answer: str, choices: Dict[str, str], last_num: int = 5) -> Optional[str]:
    """
    Extract option from answer using VLMEval2's can_infer_option logic.
    
    Args:
        answer: The answer string to extract option from
        choices: Dictionary mapping option letters to option text (e.g., {'A': 'option A text', 'B': 'option B text'})
        last_num: Number of last words to check for option letter
    
    Returns:
        Option letter (A-Z) if found, False otherwise
    """
    if 'Failed to obtain answer via API' in answer:
        return None
    
    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'
    
    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt
    
    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')
    
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)
    
    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - last_num):
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return None


def can_infer_text_vlmeval2(answer: str, choices: Dict[str, str]) -> Optional[str]:
    """
    Extract option from answer using VLMEval2's can_infer_text logic.
    
    Args:
        answer: The answer string to extract option from
        choices: Dictionary mapping option letters to option text
    
    Returns:
        Option letter if found, None otherwise
    """
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return None
    
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return None


def can_infer_vlmeval2(answer: str, choices: Dict[str, str]) -> Optional[str]:
    """
    Extract option from answer using VLMEval2's can_infer logic.
    First tries can_infer_option, then falls back to can_infer_text.
    
    Args:
        answer: The answer string to extract option from
        choices: Dictionary mapping option letters to option text
    
    Returns:
        Option letter if found, None otherwise
    """
    answer = str(answer)
    copt = can_infer_option_vlmeval2(answer, choices)
    return copt if copt else can_infer_text_vlmeval2(answer, choices)


def build_option_str_vlmeval2(choices: Dict[str, str]) -> str:
    """
    Build option string from choices dict, matching VLMEval2's format.
    
    Args:
        choices: Dictionary mapping option letters to option text
    
    Returns:
        Formatted option string
    """
    s = 'There are several options: \n'
    for c, content in choices.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s


def build_prompt_vlmeval2(question: str, options: str, prediction: str) -> str:
    """
    Build prompt for judge model, matching VLMEval2's format.
    
    Args:
        question: Question text
        options: Formatted options string
        prediction: Model prediction/answer
    
    Returns:
        Formatted prompt string
    """
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)


def _is_numeric_string(s: str) -> bool:
    """
    Check if a string represents a number (integer or float).
    
    Args:
        s: String to check
    
    Returns:
        True if the string represents a number, False otherwise
    """
    if not s or not s.strip():
        return False
    s = s.strip()
    try:
        float(s)
        return True
    except ValueError:
        return False


def _compare_numeric_strings(answer: str, ground_truth: str, epsilon: float = 1e-6) -> bool:
    """
    Compare two numeric strings with floating point tolerance.
    
    Args:
        answer: Answer string (may be numeric or non-numeric)
        ground_truth: Ground truth string (may be numeric or non-numeric)
        epsilon: Tolerance for floating point comparison (default: 1e-6)
    
    Returns:
        True if both are numeric and within epsilon, or if both are non-numeric and equal
    """
    answer = answer.strip()
    ground_truth = ground_truth.strip()
    
    # If both are numeric, compare as floats
    if _is_numeric_string(answer) and _is_numeric_string(ground_truth):
        try:
            answer_float = float(answer)
            gt_float = float(ground_truth)
            return abs(answer_float - gt_float) < epsilon
        except (ValueError, OverflowError):
            # Fallback to string comparison if conversion fails
            return answer == ground_truth
    
    # If one is numeric and the other is not, they don't match
    if _is_numeric_string(answer) != _is_numeric_string(ground_truth):
        return False
    
    # Both are non-numeric, use exact string match
    return answer == ground_truth


def check_accuracy(solution_str: str, ground_truth: Union[str, Dict[str, Any]], 
                  answer_key: str = "answer",
                  extra_info: Optional[Dict[str, Any]] = None) -> float:
    """
    Check if the solution is accurate/correct using VLMEval2's matching logic.
    
    Model output format should be: <think>...</think><answer>...</answer>
    The answer is extracted from <answer>...</answer> tags and compared with ground truth.
    For multi-choice questions, uses VLMEval2's can_infer logic to extract option letters.
    If can_infer fails and judge_model is provided, calls judge_model to extract answer,
    then uses can_infer again on the judge model's response.
    This matches VLMEval2's behavior: only call judge when can_infer fails.
    
    For open-ended questions (non-multi-choice):
    - If both answer and ground_truth are numeric (integers or floats), compares them as floats
      with tolerance epsilon (default: 1e-6) to handle floating point precision errors.
    - If either is non-numeric, uses exact string match.
    
    Args:
        solution_str: Model output string (format: <think>...</think><answer>...</answer>)
        ground_truth: Ground truth answer (string or dict)
        answer_key: Key to extract answer (default: "answer")
        extra_info: Additional information containing:
            - "candidate_points": Dict mapping option letters to option text (for multi-choice questions)
            - "judge_model": Optional judge model (e.g., Gemini, GPT-4) for answer extraction
                            Only called when can_infer fails to extract option from answer
                            Should have a `generate(prompt)` method that returns a string
            - "question": Optional question text (used when building prompt for judge model)
            - "epsilon": Optional tolerance for floating point comparison (default: 1e-6)
                         Only used for numeric answers in open-ended questions
    
    Returns:
        Accuracy reward: 1.0 if correct, 0.0 otherwise
    """
    # Extract ground truth value
    if isinstance(ground_truth, dict):
        gt_value = ground_truth.get("answer", ground_truth.get("ground_truth", ""))
    else:
        gt_value = str(ground_truth)
    if isinstance(gt_value, list) and gt_value:
        gt_value = gt_value[0]
    gt_value = str(gt_value) if gt_value is not None else ""
    
    def _extract_option_letter(text: str) -> Optional[str]:
        """从文本中提取选项字母。如 'The answer is C' 或 'C. description' -> 'C'"""
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        if len(text) == 1 and text.upper() in string.ascii_uppercase:
            return text.upper()
        patterns = [
            r'(?:answer|option)\s+is\s+([A-Z])',
            r'(?:answer|option)[:：]\s*([A-Z])',
            r'^([A-Z])[\.\)]\s',
            r'\b([A-Z])\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in string.ascii_uppercase:
                    return letter
        return None
    
    test_reward = True
    if test_reward:
        answer = extract_answer(
            solution_str, 
            answer_key, 
            is_multichoice=False,
            use_vlm=False,
            vlm_client=None,
            vlm_model=None,
            vlm_kwargs=None,
            question=None
        )
        if answer is None:
            answer = solution_str.strip()

        # 仅针对选择题：规则能判断则直接返回（正确/错误均不调用 API）；无法判断才调用 API
        gt_option_letter = _extract_option_letter(gt_value)
        if gt_option_letter is not None:
            answer_option_letter = _extract_option_letter(answer)
            # if answer_option_letter is not None:
            # 规则能判断选择题：正确返回 1，错误返回 0，均不调用 API
            return 1.0 if answer_option_letter == gt_option_letter else 0.0
            # answer 无法提取选项，尝试 candidate_points
            # if extra_info:
            #     candidate_points = extra_info.get("candidate_points", {})
            #     if candidate_points:
            #         choices = {k: str(v) if isinstance(v, (list, tuple)) else str(v) for k, v in candidate_points.items()}
            #         extracted_option = can_infer_vlmeval2(answer, choices)
            #         if extracted_option is not None:
            #             # 规则能判断：正确返回 1，错误返回 0，均不调用 API
            #             return 1.0 if extracted_option.upper() == gt_option_letter else 0.0
        
        # 规则无法判断 → 调用 API
        res = compute_accuracy_score(model_output=answer, ground_truths=gt_value)
        return res['score']
    # Extract answer from <answer>...</answer> tags (highest priority)
    # Model output format: <think>...</think><answer>...</answer>
    else:
        answer = extract_answer(
            solution_str, 
            answer_key, 
            is_multichoice=False,  # Only used for fallback regex patterns when <answer> tag is not present
            use_vlm=False,
            vlm_client=None,
            vlm_model=None,
            vlm_kwargs=None,
            question=None
        )
        if answer is None:
            # If no answer found in tags, try to extract from the whole string as fallback
            answer = solution_str.strip()
        
        # Get extra_info if not provided
        if extra_info is None:
            extra_info = {}
        
        # Check if ground truth is a single letter (A-Z), indicating multi-choice question
        gt_stripped = gt_value.strip().upper()
        is_gt_multichoice = (
            len(gt_stripped) == 1 and 
            gt_stripped in string.ascii_uppercase
        )
        
        # For multi-choice questions, use VLMEval2's can_infer logic
        if is_gt_multichoice:
            candidate_points = extra_info.get("candidate_points", {})
            if candidate_points:
                # Build choices dict from candidate_points
                choices = {k: str(v) if isinstance(v, (list, tuple)) else str(v) for k, v in candidate_points.items()}
                
                # First try: Extract option from answer using VLMEval2 logic
                extracted_option = can_infer_vlmeval2(answer, choices)
                
                if extracted_option:
                    # Compare extracted option with ground truth (case-insensitive)
                    if extracted_option.upper() == gt_stripped:
                        return 1.0
                    else:
                        return 0.0
                
                # Second try: If can_infer failed, try judge model (if provided)
                # This matches VLMEval2's behavior: only call judge when can_infer fails
                judge_model = extra_info.get("judge_model")
                question_text = extra_info.get("question", "")
                
                if judge_model is not None:
                    try:
                        # Build prompt and option string
                        option_str = build_option_str_vlmeval2(choices)
                        prompt = build_prompt_vlmeval2(question_text, option_str, answer)
                        
                        # Call judge model
                        if hasattr(judge_model, 'generate'):
                            judge_answer = judge_model.generate(prompt)
                            
                            # Extract option from judge model's answer
                            extracted_option = can_infer_vlmeval2(judge_answer, choices)
                            
                            if extracted_option:
                                # Compare extracted option with ground truth
                                if extracted_option.upper() == gt_stripped:
                                    return 1.0
                                else:
                                    return 0.0
                    except Exception as e:
                        # If judge model fails, fall through to direct comparison
                        pass
                
                # Fallback: If both can_infer and judge model failed, try direct comparison
                answer_upper = answer.strip().upper()
                if answer_upper == gt_stripped:
                    return 1.0
                else:
                    return 0.0
            else:
                # No candidate_points provided, use direct comparison
                answer_upper = answer.strip().upper()
                if answer_upper == gt_stripped:
                    return 1.0
                else:
                    return 0.0
        else:
            # For open-ended questions, use numeric comparison with tolerance for numbers
            # or exact match for non-numeric answers
            answer_stripped = answer.strip()
            gt_stripped = gt_value.strip()
            
            # Get epsilon tolerance from extra_info (default: 1e-6)
            epsilon = extra_info.get("epsilon", 1-1e-6)
            
            if _compare_numeric_strings(answer_stripped, gt_stripped, epsilon):
                return 1.0
            else:
                return 0.0


def compute_score(
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    format_score: float = 1.0,  # Format weight (decoupled from accuracy)
    score: float = 1.0,  # Accuracy weight (decoupled from format)
    inverse_answer: Optional[str] = None,  # Model's answer to inverse question (for temporal consistency)
    **kwargs,
) -> Union[float, Dict[str, Any]]:
    """
    Compute spatial reasoning reward.
    
    Similar to gsm8k.py's compute_score, but supports multiple spatial reward types.
    
    Args:
        solution_str: The solution text (forward)
        ground_truth: The ground truth answer (forward), or dict with "answer" and "inv_ground_truth"
        extra_info: Additional information containing:
            - "reward_type": str, type of reward to use:
                * "accuracy" - basic accuracy check
                * "format" - basic format check
                * "accuracy+format" - combined (default, decoupled: format and accuracy computed independently, then weighted sum)
                * "accuracy+format+spatial_t" - forward accuracy+format + inverse temporal consistency
                * "spatial_z" - 3D spatial reasoning (z-axis)
                * "spatial_xy" - XY correspondence
                * "spatial_t" - temporal spatial reasoning
            - "format_pattern": Optional[str], regex pattern for format check
            - "required_tags": Optional[List[str]], required XML/HTML tags
            - "answer_key": str, key to extract answer (default: "answer")
            - "inv_ground_truth": Optional[str or List[str]], inverse ground truth for temporal consistency
            - "candidate_points": Dict mapping option letters to option text (for multi-choice questions)
                                 e.g., {"A": "option A text", "B": "option B text"}
            - Other parameters passed to specific spatial reward functions
        format_score: Weight for format reward (default: 1.0)
        score: Weight for accuracy reward (default: 1.0)
        inverse_answer: Model's answer to inverse question (for temporal consistency check)
        **kwargs: Additional keyword arguments
            
    Note:
        Model output format should be: <think>...</think><answer>...</answer>
        The answer is extracted from <answer>...</answer> tags and compared with ground truth.
        For multi-choice questions, uses VLMEval2's can_infer logic to extract option letters,
        which matches the evaluation logic in /pfs/sunhaoze/code/VLMEval2/run_lmdeploy_mgpu.py.
        For "accuracy+format", final_score = format_score * format_reward + score * accuracy_reward
        With default weights (format_score=1.0, score=1.0), max reward is 2.0 when both format and accuracy are correct
        For "accuracy+format+spatial_t", temporal consistency is checked using inverse_answer and inv_ground_truth
    
    Returns:
        Reward score (float) or dict with detailed breakdown if return_dict=True
    """
    if extra_info is None:
        extra_info = {}
    
    return_dict = extra_info.get("return_dict", True)  # Default True for RL training logging
    reward_type = extra_info.get("reward_type", "accuracy+format")
    
    # Route to specific spatial reward functions
    if reward_type == "spatial_z" or reward_type == "spatial_3d" or reward_type == "z":
        # 3D spatial reasoning (z-axis)
        # Map "z" to spatial_z for backward compatibility
        if reward_type == "z":
            extra_info = extra_info.copy() if extra_info else {}
            extra_info["reward_type"] = "spatial_z"
        return spatial_z.compute_score(solution_str, ground_truth, extra_info=extra_info, **kwargs)
    
    elif reward_type == "spatial_xy" or reward_type == "correspondence":
        # XY correspondence
        return spatial_xy.compute_score(solution_str, ground_truth, extra_info=extra_info, **kwargs)
    
    elif reward_type == "format":
        # Format only
        format_pattern = extra_info.get("format_pattern")
        required_tags = extra_info.get("required_tags")
        check_answer_tag = extra_info.get("check_answer_tag", True)
        check_reasoning_tag = extra_info.get("check_reasoning_tag", True)
        check_bbox_tag = extra_info.get("check_bbox_tag", False)
        check_think_steps = extra_info.get("check_think_steps", True)
        min_think_steps = extra_info.get("min_think_steps", 1)
        max_think_steps = extra_info.get("max_think_steps", 6)
        format_reward = check_format(
            solution_str, format_pattern, required_tags,
            check_answer_tag, check_reasoning_tag, check_bbox_tag,
            check_think_steps, min_think_steps, max_think_steps,
        )
        
        if return_dict:
            return {
                "score": format_reward * format_score if format_score > 0 else format_reward,
                "format_reward": format_reward,
            }
        return format_reward * format_score if format_score > 0 else format_reward
    
    elif reward_type == "accuracy":
        # Accuracy only
        answer_key = extra_info.get("answer_key", "answer")
        
        accuracy_reward = check_accuracy(
            solution_str, ground_truth, answer_key, extra_info
        )
        
        if return_dict:
            return {
                "score": accuracy_reward * score,
                "accuracy_reward": accuracy_reward,
            }
        return accuracy_reward * score
    
    else:
        # Default: accuracy + format, or accuracy+format+spatial_t
        # Cascading design: format is a prerequisite - if format is wrong, reward is 0
        # Only when format is correct do we compute accuracy and other rewards

        format_pattern = extra_info.get("format_pattern")
        required_tags = extra_info.get("required_tags")
        check_answer_tag = extra_info.get("check_answer_tag", True)
        check_reasoning_tag = extra_info.get("check_reasoning_tag", True)
        check_bbox_tag = extra_info.get("check_bbox_tag", False)
        check_think_steps = extra_info.get("check_think_steps", True)
        min_think_steps = extra_info.get("min_think_steps", 1)
        max_think_steps = extra_info.get("max_think_steps", 6)
        
        # First check forward format (0.0 or 1.0)
        format_reward = check_format(
            solution_str, format_pattern, required_tags,
            check_answer_tag, check_reasoning_tag, check_bbox_tag,
            check_think_steps, min_think_steps, max_think_steps,
        )

        # answer_key must be defined before spatial_t block
        answer_key = extra_info.get("answer_key", "answer")
        
        # Cascading logic: only compute accuracy if format is correct
        if format_reward == 0.0:
            # Format is wrong -> reward = 0, skip accuracy check
            accuracy_reward = 0.0
            final_score = 0.0
        else:
            # Format is correct -> compute accuracy
            answer_key = extra_info.get("answer_key", "answer")
            accuracy_reward = check_accuracy(
                solution_str, ground_truth, answer_key, extra_info
            )

            # Cascading multiplication: final_score = format_reward * accuracy_reward * score
            # - If format is wrong (format_reward=0): final_score = 0 * accuracy_reward = 0
            # - If format is correct (format_reward=1) and accuracy is correct: final_score = 1 * 1 * score = score
            # - If format is correct (format_reward=1) but accuracy is wrong: final_score = 1 * 0 * score = 0
            # This ensures format is a prerequisite: only when format is correct does accuracy matter
            final_score = format_reward * accuracy_reward * score
        
        # Check if temporal consistency (spatial_t) is requested
        # Only trigger if reward_type explicitly contains "spatial_t" in the combination

        r_temporal_consistency = 0.0
        if "spatial_t" in reward_type:
            # Get inverse data
            inv_ground_truth = None
            if isinstance(ground_truth, dict):
                inv_ground_truth = ground_truth.get("inv_ground_truth")
            if not inv_ground_truth:
                inv_ground_truth = extra_info.get("inv_ground_truth")
            
            # Handle list format
            if isinstance(inv_ground_truth, list) and len(inv_ground_truth) > 0:
                inv_ground_truth = inv_ground_truth[0]
            
            # Get inverse_answer (from parameter or kwargs)
            inv_answer = inverse_answer or kwargs.get("inverse_answer")
            
            # Compute temporal consistency if we have both inverse answer and ground truth
            inverse_accuracy_reward = None
            if inv_answer and inv_ground_truth:
                # Check inverse format
                inverse_format_reward = check_format(
                    inv_answer, format_pattern, required_tags,
                    check_answer_tag, check_reasoning_tag, check_bbox_tag
                )
                
                if inverse_format_reward == 0.0:
                    inverse_accuracy_reward = 0.0
                else:
                    # Check inverse accuracy
                    inverse_accuracy_reward = check_accuracy(
                        inv_answer, inv_ground_truth, answer_key, extra_info
                    )
                
                # r_temporal_consistency = inverse_format * inverse_accuracy (for logging)
                r_temporal_consistency = inverse_format_reward * inverse_accuracy_reward * score

            # Combine forward and inverse scores
            forward_weight = extra_info.get("forward_weight", 1.0)
            inverse_weight = extra_info.get("inverse_weight", 1.0)
            
            if r_temporal_consistency > 0.0 and final_score > 0.0:
                # Both forward and inverse available: weighted average
                # final_score = format_reward * (forward_weight * accuracy + inverse_weight * inverse_accuracy)
                final_score = 1.0
            else:
                final_score = 0.0

        if return_dict:
            result = {
                "score": final_score,
                "format_reward": format_reward,
                "accuracy_reward": accuracy_reward,
            }
            if "spatial_t" in reward_type:
                # Add temporal consistency score (inverse_format * inverse_accuracy) for logging
                result["r_temporal_consistency"] = r_temporal_consistency
            return result
        
        return final_score


if __name__ == "__main__":
    # Test cases for spatial reward functions
    solution_str = "<think>Some reasoning here.</think><answer>The answer is 42.</answer>"
    ground_truth = "42"
    extra_info = {
        "reward_type": "accuracy+format",
        "format_pattern": r'<think>.*?</think>.*<answer>.*?</answer>',
        "required_tags": ["<think>", "<answer>"],
        "answer_key": "answer",
    }
    score = compute_score(solution_str, ground_truth, extra_info)
    print(f"Computed score: {score}")