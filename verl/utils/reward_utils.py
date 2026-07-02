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

"""Reward utilities: normalization and sanitization for mixed spatial_xy/spatial_z/spatial_t batches."""

from typing import Any

import numpy as np
import torch


def _to_info_dict(score: dict | float) -> dict[str, Any]:
    """Convert score to dict (float -> {"score": float})."""
    return score if isinstance(score, dict) else {"score": float(score)}


def normalize_reward_extra_infos(
    infos: list[dict[str, Any]],
    default_for_missing: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Merge per-sample reward_extra_info into a consistent dict.
    Missing numeric keys get default_for_missing (avoids NaN in mixed batches).
    """
    all_keys = set()
    for info in infos:
        if isinstance(info, dict):
            all_keys.update(info.keys())

    result = {}
    for key in all_keys:
        values = [info.get(key) if isinstance(info, dict) else None for info in infos]
        try:
            numeric = [v for v in values if v is not None]
            if numeric and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in numeric):
                arr = np.array(
                    [float(v) if v is not None else default_for_missing for v in values],
                    dtype=np.float32,
                )
            else:
                arr = np.array(values, dtype=object)
        except (ValueError, TypeError):
            arr = np.array(values, dtype=object)
        result[key] = arr
    return result


def sanitize_reward_extra_infos(reward_extra_infos_dict: dict[str, Any]) -> None:
    """
    In-place sanitization: replace NaN/Inf/None with 0.0 in reward_extra_infos_dict.
    Supports mixed spatial_xy/spatial_z/spatial_t batches.
    """

    def _is_numeric_or_none(x):
        return x is None or isinstance(x, (int, np.integer, float, np.floating))

    for key, value in reward_extra_infos_dict.items():
        if torch.is_tensor(value):
            reward_extra_infos_dict[key] = torch.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
            )
            continue

        try:
            arr = np.array(value, dtype=object)
        except Exception:
            continue
        if arr.size == 0 or not all(_is_numeric_or_none(x) for x in arr.ravel()):
            continue

        flat = arr.ravel()
        bad = np.array(
            [
                (x is None) or (isinstance(x, (float, np.floating)) and not np.isfinite(x))
                for x in flat
            ],
            dtype=bool,
        )
        if bad.any():
            reward_extra_infos_dict[key] = np.array(
                [0.0 if bad[i] else float(flat[i]) for i in range(len(flat))],
                dtype=np.float32,
            ).reshape(arr.shape)
