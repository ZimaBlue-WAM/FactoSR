import re
import signal
import functools
import time
import traceback
import threading
from openai import OpenAI
from dataclasses import dataclass
from functools import wraps, partial
from typing import Optional, Union, List
from verl.utils.reward_score.math_reward.judge import Judger

import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

judger = Judger()

# 全局信号量，限制API并发调用数，防止触发速率限制
# 每个进程会有自己的信号量实例，限制每个进程内的并发
_api_call_semaphore = threading.Semaphore(3)  # 限制每个进程最多3个并发API调用
_api_call_lock = threading.Lock()  # 用于保护时间戳记录
_last_api_call_time = {}  # 记录每个API key的最后调用时间

unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "p . m",
    "lb",
    "tile",
    "per",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "month",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
    "kilogram",
    "second",
    "ampere",
    "A",
    "K",
    "mol",
    "cd",
    "N",
    "J",
    "W",
    "Pa",
    "Hz",
    "C",
    "V",
    "Ω",
    "F",
    "T",
    "H",
    "eV",
    "kW·h",
    "atm",
    "bar",
    "°C",
    "m"
]
unit_texts.extend([t + "s" for t in unit_texts])

def last_n_boxed_strings(string, n):
    boxed_list = []

    work_str = string[:]
    while work_str and len(boxed_list) < n:
        idx = work_str.rfind("\\boxed")
        if idx < 0:
            idx = work_str.rfind("\\fbox")

        if idx < 0:
            break

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(work_str):
            if work_str[i] == "{":
                num_left_braces_open += 1
            elif work_str[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is not None:
            boxed_expr = work_str[idx: right_brace_idx + 1]
            boxed_list.append(boxed_expr)
            work_str = work_str[:idx]
        else:
            work_str = work_str[:idx]

    boxed_list.reverse()
    return boxed_list


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left): -1]
    except Exception:
        return None


def get_answer_str(s: str, return_origin=False, num_answers=1):
    boxed_list = last_n_boxed_strings(s, num_answers)
    answer_list = [remove_boxed(b) if b else "" for b in boxed_list]

    missing = num_answers - len(answer_list)
    fill_str = s if return_origin else ""
    answer_list = [fill_str] * missing + answer_list
    if answer_list==['']:
        return [s]
    return answer_list

def solution2answer(solution: str, math_mode="eval_peeking", return_origin=False, num_answers=1) -> tuple[bool, list | str]:
    answer = solution
    if math_mode == "eval_peeking":
        answer = get_answer_str(solution, return_origin, num_answers)
    else:
        raise ValueError(f"Invalid math_mode: {math_mode}")
    return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)
    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return 1, given_answer_normalized_mathd, ground_truth_normalized_mathd
    return 0, given_answer_normalized_mathd, ground_truth_normalized_mathd

def timeout_handler(signum, frame):
    raise TimeoutError("reward_fn execution timed out")


def with_timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except FutureTimeoutError:
                    future.cancel()
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
        return wrapper
    return decorator

def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func

def retry(max_attempts:int=3, delay:int=1, print_trace_back=False, return_error_info=False):
    assert isinstance(max_attempts, int) and isinstance(delay, int), '参数必须是整数'

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if print_trace_back:
                        e = traceback.format_exc()
                        error_info = f">>>函数{func.__name__}第{attempts + 1}次尝试失败，报错信息为: {e}"
                        print(error_info)
                    time.sleep(delay)
                    attempts += 1
            if return_error_info:
                return error_info
            else:
                return None
        
        @attach_wrapper(wrapper)
        def set_max_attempts(new_max_attempts):
            nonlocal max_attempts
            max_attempts = new_max_attempts

        @attach_wrapper(wrapper)
        def set_delay(new_delay):
            nonlocal delay
            delay = new_delay

        wrapper.get_attempts = lambda: max_attempts
        wrapper.get_delay = lambda: delay
        return wrapper
    return decorator

@dataclass
class ModelArgs:
    use_model: bool = False
    # model_name: str = 'Gemini-2.5-flash'  # Model name
    # api_key: str = "pk-9861dc90-7089-43e4-8237-8ffb7401f291" #"a269910f-c0fd-45c6-9511-9410e8404905"
    # base_url: str = "http://modelservice.jdcloud.com/v1" #"https://sd092klhlef3ta500u6ug.apigateway-cn-beijing.volceapi.com/mlp/s-20250430183800-df4h4/v1/"  # Anonymized model path or URL
    model_name: str = 'qwen3-30b'  # Model name
    api_key: str = "EMPTY"
    base_url: str = "http://172.20.66.1:8000/v1"
    max_tokens: int = 1024
    temperature: float = 0.0

@with_timeout(30)
def grade_answer_xverify(given_answer: str, ground_truth: str, problem: str, model_args: ModelArgs) -> bool:
    @retry(max_attempts=4, delay=1, print_trace_back=True, return_error_info=True)
    def call_api(prompt:str, 
                system_prompt:Optional[str]=None,
                client=None,
                base_url:Optional[str]=None,
                model:str="gpt-3.5-turbo", 
                api_key:Union[None,str]=None, 
                max_tokens:int=None, 
                temperature:float=0.7,
                logprobs:bool=False,
                top_logprobs:int=1,
                **kwargs) -> str:
        if not client:
            assert api_key is not None,'Please input your api key'
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
                )
        if not logprobs:
            top_logprobs = None

        messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
        if prompt:
            messages.append({"role": "user", "content": prompt}) 

        # 使用信号量限制并发，防止触发速率限制
        with _api_call_semaphore:
            # 添加延迟，避免请求过于频繁
            # 检查并控制调用频率（至少间隔0.5秒）
            current_key = api_key or (client.api_key if hasattr(client, 'api_key') else 'default')
            current_time = time.time()
            
            with _api_call_lock:
                if current_key in _last_api_call_time:
                    time_since_last = current_time - _last_api_call_time[current_key]
                    if time_since_last < 0.5:  # 至少间隔0.5秒
                        time.sleep(0.5 - time_since_last)
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    **kwargs
                )
                result = response.choices[0].message.content

            finally:
                # 无论成功还是失败，都更新最后调用时间
                with _api_call_lock:
                    _last_api_call_time[current_key] = time.time()
        
        return result
    
    client = OpenAI(api_key=model_args.api_key, base_url=model_args.base_url)

#     prompt=f"""As a grading reward model, your task is to evaluate whether the candidate's final answer matches the provided standard answer. 
# You must first output a detailed step-by-step analysis (your reasoning process), then give a final structured judgment. 
# Do not regenerate or improve answers, only compare.

# Evaluation Protocol:
# 1. Reference Standard:
#    - The standard (gold) answer is definitive and always correct.
#    - The question is always valid — never challenge it.
#    - Do not regenerate answers; only compare candidate's final answer with the gold answer.

# 2. Comparison Method:
#    - Analyze the question's requirements and the gold answer's structure.
#    - Determine if the question requires exact matching or allows equivalence.
#    - Compare ONLY the candidate's final answer. Ignore reasoning errors.
#    - Ignore differences in formatting or style.
#    - For math expressions: check algebraic equivalence step by step; if uncertain, test numerically at multiple points.
#    - For multiple-choice: only compare the final choice and its content.

# 3. Multi-part Answers:
#    - All parts must match the gold answer exactly.
#    - Partial matches are incorrect.
#    - If not specified, answer order may vary. For example, \\frac{{27}}{{7}}, -\\frac{{8}}{{7}} and -\\frac{{8}}{{7}}, \\frac{{27}}{{7}} are equivalent.

# 4. Validity Check:
#    - If incomplete (cut off, unfinished sentence) → Label as INCOMPLETE.
#    - If repetitive (looping words/phrases) → Label as REPETITIVE.
#    - If explicit refusal (e.g., "I cannot answer...") → Label as REFUSAL.
#    - Gives an answer but then negates it at the end. → Label as REFUSAL.
#    - Any of the above → classify as C with the correct error type.

# Grading Scale:
# \\boxed{{A}} - CORRECT:
#    - Matches gold exactly or equivalent (including algebraic/numeric equivalence).
#    - For numerical values: equivalent if equal under rounding tolerance.
#    - Semantic equivalence allowed.

# \\boxed{{B}} - INCORRECT:
#    - Any deviation from gold.
#    - Partial matches for multi-part answers.

# \\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
#    - Invalid answers (must specify error type).

# Execution Steps and Output:

# Analysis step by step:
# [ 
# 1. First check validity (INCOMPLETE, REPETITIVE, REFUSAL). 
# 2. Compare candidate’s final answer vs. gold answer in detail. The most important thing to note is not to try solving the problem yourself, but only to compare the similarity between the final answer and gold answer.
#    - Identify strict requirements (e.g., exact match, order, completeness).
#    - Allow tolerances (format differences, equivalent math forms, unsimplified fraction, provide the full answer for completion-type questions). Note: Unsimplified fractions are allowed.
#    - Check for equivalences (e.g., \\frac{{2x-7}}{{(x+1)(x-2)}} and \\frac{{3}}{{x+1}} - \\frac{{1}}{{x-2}} are equivalent).
#       - Consider following situation:
#          - Factoring or expansion: x^2+2x+1 → (x+1)^2  
#          - Fraction simplification: (x^2−1)/(x+1) → x−1  
#          - Leaving fraction unsimplified: (x^2−1)/(x+1) (unchanged)  
#          - Partial fraction decomposition: 1/(x(x+1)) → 1/x − 1/(x+1)  
#          - Fraction to decimal conversion: 1/2 → 0.5  
#          - Trigonometric identities: sin^2x+cos^2x=1  
#          - Trigonometric transformations: sin 2x = 2 sin x cos x  
#          - Taylor expansion: sin x ≈ x − x^3/3!  
#          - Exponential/logarithm rules: ln(ab)=ln a + ln b  
#          - Substitution: let y=x+1, then x^2+2x+1 = y^2  
#          - Approximating special constants: π ≈ 3.14159, e ≈ 2.718  
#          - Angle-radian conversion: π/6 = 30°   
#          - Dimensional conversion (e.g., F = ma, m=1000 g, a=2 m/s² → F = 2 N)
#    - For multiple-choice questions, the answer is considered correct only if the selected option exactly matches the standard answer, or if the answer content is fully equivalent to the correct option.
#       - If both the option label and the option content appear in the answer, both must match the standard answer for it to be considered correct.
# 3. Provide a thorough reasoning chain, highlighting subtle equivalences or deviations. 
# ]

# Final Judgment:
# \\boxed{{A/B/C}}

# Here is your task.
# <Original Question Begin>
# {problem}
# <Original Question End>

# <Standard Answer Begin>
# {ground_truth}
# <Standard Answer End>

# <Candidate's Answer Begin>
# {given_answer}
# <Candidate's Answer End>

# Only output the final judgment in the format \\boxed{{A}}, \\boxed{{B}}, or \\boxed{{C}}.
# """
    prompt=f"""As a grading reward model, your task is to evaluate whether the candidate's final answer matches the provided standard answer. 
    You must only give a final structured judgment. 
    Do not regenerate or improve answers, only compare.

    Evaluation Protocol:
    1. Reference Standard:
    - The standard (gold) answer is definitive and always correct.
    - The question is always valid — never challenge it.
    - Do not regenerate answers; only compare candidate's final answer with the gold answer.

    2. Comparison Method:
    - Analyze the question's requirements and the gold answer's structure.
    - Determine if the question requires exact matching or allows equivalence.
    - Compare ONLY the candidate's final answer. Ignore reasoning errors.
    - Ignore differences in formatting or style.
    - For math expressions: check algebraic equivalence step by step; if uncertain, test numerically at multiple points.
    - For multiple-choice: only compare the final choice and its content.
    - Reasonable unit conversions and missing conversions are acceptable.

    3. Multi-part Answers:
    - All parts must match the gold answer exactly.
    - Partial matches are incorrect.
    - If not specified, answer order may vary. For example, \\frac{{27}}{{7}}, -\\frac{{8}}{{7}} and -\\frac{{8}}{{7}}, \\frac{{27}}{{7}} are equivalent.

    4. Validity Check:
    - If incomplete (cut off, unfinished sentence) → Label as INCOMPLETE.
    - If repetitive (looping words/phrases) → Label as REPETITIVE.
    - If explicit refusal (e.g., "I cannot answer...") → Label as REFUSAL.
    - Gives an answer but then negates it at the end. → Label as REFUSAL.
    - Any of the above → classify as C with the correct error type.

    Grading Scale:
    \\boxed{{A}} - CORRECT:
    - Matches gold exactly or equivalent (including algebraic/numeric equivalence).
    - For numerical values: equivalent if equal under rounding tolerance.
    - Semantic equivalence allowed.

    \\boxed{{B}} - INCORRECT:
    - Any deviation from gold.
    - Partial matches for multi-part answers.

    Here is your task.
    <Original Question Begin>
    {problem}
    <Original Question End>

    <Standard Answer Begin>
    {ground_truth}
    <Standard Answer End>

    <Candidate's Answer Begin>
    {given_answer}
    <Candidate's Answer End>

    Only output the final judgment in the format \\boxed{{A}}, or \\boxed{{B}}.
    """
    for  _attempt in range(3):
        try:
            correct = call_api(prompt=prompt,
                            client=client, 
                            max_tokens=model_args.max_tokens,
                            model=model_args.model_name,
                            temperature=model_args.temperature)
            # call_api 返回字符串，直接使用
            if isinstance(correct, dict):
                correct_text = correct.get('text', '')
            else:
                correct_text = str(correct)
            return 1.0 if "A" in correct_text.strip() else 0.0
        except:
            continue
    
    return 0.0

    
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

@with_timeout(5)
def physics_verify(answer_type, extracted_pred, extracted_gt):
    if answer_type is not None:
        return judger.judge(answer_type, extracted_pred, extracted_gt)
    else:
        return judger.auto_judge(extracted_pred, extracted_gt)
    
"""
按 answer_type 依次调用 mathd→physics_verify→gemini 打分，返回 (平均分, 逐题分, 预测, 答案, 打分器)。
"""
@with_timeout(30)
def grade(extracted_answers: List[str], 
          gt_answers: List[str],
          answer_types: List[str], 
          problem=None, 
          model_args: ModelArgs = ModelArgs()):
    
    total_score = 0
    extracted_preds, extracted_gts, scored_by = [], [], []
    score_list = []
    for extracted_answer, gt_answer, answer_type in zip(extracted_answers, gt_answers, answer_types):
        score, extracted_pred, extracted_gt = grade_answer_mathd(extracted_answer, gt_answer)
        scored_by.append('not_scored')
        if not score:
            try:
                score, extracted_pred, extracted_gt = physics_verify(answer_type, extracted_answer, extracted_gt)
            except OverflowError as e:
                print(f"OverflowError: {e}, extracted_pred: {extracted_pred}, extracted_gt: {extracted_gt}")
                score, extracted_pred, extracted_gt = 0, "", ""
            except:
                pass
        else:
            scored_by[-1] = "mathd"

        if not score and extracted_answer and gt_answer:
            try:
                score = grade_answer_xverify(extracted_answer, gt_answer, problem, model_args)    
            except:
                pass
        else:
            scored_by[-1] = "rules"
        
        if score and scored_by[-1]=="not_scored":
            scored_by[-1] = "gemini"

        score_list.append(score)
        total_score += score
        extracted_preds.append(extracted_answer)
        extracted_gts.append(extracted_gt)
    total_score /= len(gt_answers)
    return total_score, score_list, extracted_preds, extracted_gts, scored_by


def answer_tag_reward_fn_for_r1(model_output: str, 
                                ground_truths: Union[str, List[str]], 
                                problem=None, 
                                answer_types: List[str]=[],
                                points: Union[float, List[float]]=[],
                                use_xverify=False):
    model_args = ModelArgs(use_model=use_xverify)
    # Make sure ground_truths is a list
    ground_truths = [ground_truths] if isinstance(ground_truths, str) else ground_truths
    num_questions_to_answer = len(ground_truths)
    # Allow answer types not provieded
    try:
        if (answer_types is None) or (len(answer_types) != num_questions_to_answer) or answer_types[0] == "":
            answer_types = [None] * num_questions_to_answer
    except:
        answer_types = [None] * num_questions_to_answer
    
    # Extract \boxed envirnoment.
    extracted_answers = solution2answer(str(model_output), num_answers=num_questions_to_answer)
    ground_truths = [solution2answer(str(gt), return_origin=True)[0] for gt in ground_truths]
    if not any(extracted_answers):
        return 0.0, 0.0, extracted_answers, ground_truths, ["not_scored"] * num_questions_to_answer
    try:
        score, score_list, extracted_pred, extracted_gt, scored_by = grade(extracted_answers,
                                                                ground_truths,
                                                                answer_types,
                                                                problem,
                                                                model_args)
    except:
        return 0.0, 0.0, extracted_answers, ground_truths, ["not_scored"] * num_questions_to_answer
    if points is None or len(points) == 0:
        points = [1.0] * num_questions_to_answer
    if len(points) == num_questions_to_answer:
        point = sum([s * p for s, p in zip(score_list, points)])
    else:
        point = score

    return score, point, extracted_pred, extracted_gt, scored_by
    
def compute_score(model_output: str, 
                  ground_truths: Union[str, List[str]], 
                  problem=None, 
                  answer_types: Union[str, List[str]]=[], 
                  points: Union[float, List[float]]=[],
                  use_xverify=False):

    score, point, extracted_pred, extracted_gt, scored_by = answer_tag_reward_fn_for_r1(model_output, 
                                                                                ground_truths, 
                                                                                problem, 
                                                                                answer_types,
                                                                                points,
                                                                                use_xverify)
    # print(f"Model Output: {model_output}, Extracted_pred: {extracted_pred}, Ground Truths: {ground_truths}, Score: {score}, Point: {point}, Scored By: {scored_by}")

    return {
        "score": score,
        "point": point,
        "acc": abs(score - 1.0) < 1e-5,
        "extracted_gt": str(extracted_gt),
        "extracted_pred": str(extracted_pred),
        "scored_by": str(scored_by)
    }

if __name__ == "__main__":
    model_output = "<think>\nStep1: Observe: The image displays two computer monitors on a desk. The monitor inside the blue box is on the left side of the image, and the monitor inside the red box is on the right side of the image. The blue box's monitor appears larger and is positioned more towards the foreground.\nStep2: Analyze: The desk space between the two monitors serves as a primary scale reference. The blue box's monitor is situated on the left, taking up a significant portion of the desk space. The red box's monitor is on the right, and a noticeable gap separates it from the blue box's monitor, suggesting a substantial horizontal distance between them.\nStep3: Verify: By synthesizing the cues, the placement of the monitors on the desk provides a clear left-right context. The left monitor (blue box) and right monitor (red box) are arranged in a separated, horizontal line. This analysis confirms their relative positions can be determined along the left-right axis.\n</think>\n<answer>monitor-(blue box) is more to the left.</answer>"
    ground_truth = "Positioned to the left is monitor-(blue box)."
    problem = """Which is more to the left, the monitor-(red box) or the monitor-(blue box)?\nassistant\n"""
    result = compute_score(model_output, ground_truth,problem)
    print(result)