# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import re
from typing import Callable, Optional

import numpy as np
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

# initialize math_verify_func once
math_verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(),
    ),
)

boxed = lambda x: "\\boxed{" + x + "}" if not x.startswith("\\boxed{") else x


def math_expression_reward(
    ground_truth: str, response: str, tag: str = "answer"
) -> tuple[float, bool]:
    """Reward the agent when the answer within the <{tag}> tags is the same expression as the ground truth.

    The `tag` is customizable and must be specified as part of the user COT prompt text file.
    """
    match = re.search(rf"<{tag}>([\s\S]*)</{tag}>", response)
    if match:
        answer = match.group(1)
        try:
            score, _ = math_verify_func([boxed(ground_truth)], [boxed(answer)])
            return float(score), score > 0.1
        except (Exception, TimeoutException) as e:
            return 0.0, False
    return 0.0, False


def format_reward(
    ground_truth: str,
    response: str,
    think_tag: str = "think",
    answer_tag: str = "answer",
) -> tuple[float, Optional[bool]]:
    """Reward the agent when the response follows the format: (.*) <think> (.*) </think> <answer> (.*) </answer>.

    The `think_tag` and `answer_tag` are customizable and must be specified as part of the user COT prompt text file.
    """
    rew = 0.0
    if re.search(rf"<{think_tag}>[\s\S]*</{think_tag}>", response):
        rew += 0.25  # 0.25 points for having think tags
    if re.search(rf"<{answer_tag}>[\s\S]*</{answer_tag}>", response):
        rew += 0.75  # 0.75 points for having answer tags
    return rew, None


def exact_answer_alphanumeric_reward(
    ground_truth: str, response: str, answer_tag: str = "answer"
) -> tuple[float, bool]:
    """Reward the agent when the answer within the <{answer_tag}> tags is the same as the ground truth (case-insensitive).

    The `answer_tag` is customizable and must be specified as part of the user COT prompt text file.
    """
    match = re.search(rf"<{answer_tag}>([\s\S]*)</{answer_tag}>", response)
    if match:
        answer = match.group(1)
        # Remove all non-alphanumeric characters (including whitespace, punctuation, etc.)
        answer_clean = "".join(c for c in answer if c.isalnum()).lower()
        ground_truth_clean = "".join(c for c in ground_truth if c.isalnum()).lower()
        if answer_clean == ground_truth_clean:
            return 1.0, True
    return 0.0, False


def bbox_giou_reward(
    ground_truth: str,
    response: str,
    giou_penalty_thres: float = 10.0,
    answer_tag: str = "answer",
) -> tuple[float, bool]:
    """Given [x1, y1, x2, y2] normalized bounding box coordinates within the <{answer_tag}> tags, compute the GIoU between the ground truth and the response.

    The `answer_tag` is customizable and must be specified as part of the user COT prompt text file.
    """
    match = re.search(rf"<{answer_tag}>([\s\S]*)</{answer_tag}>", response)
    if match:
        answer = match.group(1)
    else:
        return 0.0, False

    try:
        x1g, y1g, x2g, y2g = [
            float(x) for x in ground_truth.replace("[", "").replace("]", "").split(",")
        ]
        x1r, y1r, x2r, y2r = [
            float(x) for x in answer.replace("[", "").replace("]", "").split(",")
        ]
    except ValueError:
        return 0.0, False

    # compute iou function
    # compute the area of the ground truth and response bounding boxes
    area_g = (x2g - x1g) * (y2g - y1g)
    area_r = (x2r - x1r) * (y2r - y1r)
    # compute the intersection of the ground truth and response bounding boxes
    x1i = max(x1g, x1r)
    y1i = max(y1g, y1r)
    x2i = min(x2g, x2r)
    y2i = min(y2g, y2r)
    # compute the area of the intersection
    area_i = max(0.0, x2i - x1i) * max(0.0, y2i - y1i)
    # compute the area of the union
    area_u = max(1e-3, area_g + area_r - area_i)
    # compute the iou
    iou = area_i / area_u
    # if iou is too low, introduce a giou term to compensate
    if iou < giou_penalty_thres:
        # compute convex hull as min
        x1c = min(x1g, x1r)
        y1c = min(y1g, y1r)
        x2c = max(x2g, x2r)
        y2c = max(y2g, y2r)
        # compute the area of the convex hull
        area_c = max(1e-3, (x2c - x1c) * (y2c - y1c))
        # compute the giou
        giou = iou - (area_c - area_u) / area_c
    else:
        giou = iou
    return giou, giou > 0.5


def extract_all_boxed(text: str) -> list[str]:
    r"""Extract every ``\boxed{...}`` payload from ``text`` (innermost,
    brace-balanced).

    Mirrors Omni's helper of the same name; needed by the
    ``verl_geo3k`` reward to score multi-boxed responses.
    """
    results: list[str] = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx < 0:
            break
        start = idx + len("\\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1])
            i = j
        else:
            break
    return results


def verl_geo3k_reward(
    ground_truth: str,
    response: str,
    format_score: float = 0.1,
) -> tuple[float, bool]:
    r"""Reward function for MMPR-Tiny / Geometry3K following verl's
    geo3k implementation.

    Exact replication of:
    https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/geo3k.py
    (relaxed format regex to accept missing opening ``<think>`` tag and
    use ``re.search`` so nested braces inside ``\boxed{...}`` are
    matched).

    Returns ``(reward, is_correct)`` where:
        ``reward = (1 - format_score) * accuracy + format_score * format``

    ``mathruler`` is imported lazily so this module stays importable
    in environments without it; ``ImportError`` only triggers when this
    reward function is actually instantiated by the env.
    """
    try:
        from mathruler.grader import extract_boxed_content, grade_answer
    except ImportError as exc:
        raise ImportError(
            "verl_geo3k_reward requires the `mathruler` package. "
            "Install it via `uv add mathruler` (or in your env's "
            "constraints) before enabling the `verl_geo3k` reward."
        ) from exc

    format_pattern = re.compile(r"</think>.*\\boxed\{.*\}", re.DOTALL)
    has_format = bool(re.search(format_pattern, response))
    format_reward_value = 1.0 if has_format else 0.0

    try:
        answer = extract_boxed_content(response)
        is_correct = grade_answer(answer, ground_truth)
        acc_reward_value = 1.0 if is_correct else 0.0
    except Exception as e:  # noqa: BLE001 -- mathruler can raise broad errors on weird inputs
        print(
            f"verl_geo3k_reward: mathruler grade failed ({type(e).__name__}: {e}); "
            f"gt={ground_truth!r} response_head={response[:200]!r}",
            flush=True,
        )
        acc_reward_value = 0.0
        is_correct = False

    final_reward = (
        (1.0 - format_score) * acc_reward_value + format_score * format_reward_value
    )
    return final_reward, is_correct


def combine_reward_functions(
    reward_functions: list[tuple[Callable[[str, str], tuple[float, bool]], float]],
) -> Callable[[str, str], tuple[float, bool]]:
    """Returns a callable function that takes (ground_truth, response) and collects multiple reward functions in sequence.

    The reward functions are weighted by the second element of the tuple.
    This information can be provided in the YAML config file and resolved in the VLMEnvironment class.

    Args:
        reward_functions: list[tuple[Callable[[str, str], tuple[float, bool]], float]]. A list of reward functions and their weights.

    Returns:
        Callable[[str, str], tuple[float, bool]]: A callable function that takes (ground_truth, response) and collects multiple reward functions in sequence
    """
    weights = [weight for _, weight in reward_functions]
    weights = np.array(weights) / np.sum(weights)  # renormalize weights to 1

    def combined_reward_func(ground_truth: str, response: str) -> tuple[float, bool]:
        reward_env_score = [
            reward_func(ground_truth, response) for reward_func, _ in reward_functions
        ]
        rewards = [x[0] for x in reward_env_score]
        is_correct = [
            x[1] for x in reward_env_score if x[1] is not None
        ]  # skip None values, because they do not contribute to the "correctness" of the response (e.g. format_reward, because the answer can still be correct without <think> tags)
        is_correct = all(is_correct)
        return np.sum(np.array(rewards) * weights), is_correct

    return combined_reward_func
