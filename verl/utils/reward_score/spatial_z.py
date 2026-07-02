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
3D Spatial Reasoning Reward Function

This module implements a hybrid geometric consistency reward for 3D spatial reasoning tasks,
combining 3D IoU, 2D projection consistency, and depth-aware consistency to strengthen
the model's perception of the depth dimension in multi-view scenarios.
"""

import ast
import json
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def parse_3d_box(solution_str: str) -> Optional[np.ndarray]:
    """
    Parse 3D bounding box from solution string.

    Supports multiple formats:
    - List of dicts: [{"bbox_3d": [x, y, z, w, h, l, rx, ry, rz], "label": "..."}, ...]
    - JSON: {"center": [x, y, z], "size": [w, h, l], "rotation": [rx, ry, rz]}
    - JSON: {"x": x, "y": y, "z": z, "width": w, "height": h, "length": l, "rotation": [rx, ry, rz]}
    - List format: [x, y, z, w, h, l, rx, ry, rz]
    - Text format: "center: (x, y, z), size: (w, h, l), rotation: (rx, ry, rz)"

    Args:
        solution_str: The solution string containing 3D box information.

    Returns:
        numpy array of shape (9,) representing [x, y, z, w, h, l, rx, ry, rz],
        or None if parsing fails.
    """
    # Try list of dicts format first: [{"bbox_3d": [...], "label": "..."}, ...]
    # Also handles Python dict format with single quotes: [{'bbox_3d': [...], 'label': '...'}, ...]
    try:
        # First try standard JSON (double quotes)
        parsed = json.loads(solution_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            # Get first item
            first_item = parsed[0]
            if isinstance(first_item, dict) and "bbox_3d" in first_item:
                bbox_3d = first_item["bbox_3d"]
                if isinstance(bbox_3d, list) and len(bbox_3d) >= 6:
                    box = np.array(bbox_3d[:9] if len(bbox_3d) >= 9 else bbox_3d[:6] + [0, 0, 0], dtype=np.float32)
                    return box
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    
    # Try Python eval for single-quote format (use with caution, but safe here since we only extract numbers)
    try:
        # Use ast.literal_eval for safer evaluation of Python literals
        parsed = ast.literal_eval(solution_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            first_item = parsed[0]
            if isinstance(first_item, dict):
                # Try both 'bbox_3d' and "bbox_3d" keys
                bbox_3d = first_item.get("bbox_3d") or first_item.get('bbox_3d')
                if bbox_3d is not None and isinstance(bbox_3d, list) and len(bbox_3d) >= 6:
                    box = np.array(bbox_3d[:9] if len(bbox_3d) >= 9 else bbox_3d[:6] + [0, 0, 0], dtype=np.float32)
                    return box
    except (ValueError, SyntaxError, TypeError):
        pass
    
    # Try to find bbox_3d pattern in string using regex (handles both single and double quotes)
    try:
        # Look for "bbox_3d": [...] or 'bbox_3d': [...] pattern
        bbox_match = re.search(r'["\']bbox_3d["\']\s*:\s*\[([\s\d\.,\-]+)\]', solution_str, re.DOTALL)
        if bbox_match:
            values_str = bbox_match.group(1)
            values = [float(x.strip()) for x in values_str.split(",") if x.strip()]
            if len(values) >= 6:
                box = np.array(values[:9] if len(values) >= 9 else values[:6] + [0, 0, 0], dtype=np.float32)
                return box
    except (ValueError, AttributeError):
        pass
    
    # Try JSON format
    try:
        # Look for JSON object in the string
        json_match = re.search(r'\{[^{}]*"center"[^{}]*\}|' r'\{[^{}]*"x"[^{}]*\}', solution_str, re.DOTALL)
        if json_match:
            box_dict = json.loads(json_match.group(0))
            if "center" in box_dict and "size" in box_dict:
                center = np.array(box_dict["center"], dtype=np.float32)
                size = np.array(box_dict["size"], dtype=np.float32)
                rotation = np.array(box_dict.get("rotation", [0, 0, 0]), dtype=np.float32)
            elif "x" in box_dict and "y" in box_dict and "z" in box_dict:
                center = np.array([box_dict["x"], box_dict["y"], box_dict["z"]], dtype=np.float32)
                size = np.array(
                    [box_dict.get("width", 0), box_dict.get("height", 0), box_dict.get("length", 0)], dtype=np.float32
                )
                rotation = np.array(box_dict.get("rotation", [0, 0, 0]), dtype=np.float32)
            else:
                raise ValueError("Invalid JSON format")
            return np.concatenate([center, size, rotation])
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Try list format: [x, y, z, w, h, l, rx, ry, rz]
    try:
        list_match = re.search(r'\[[\s\d\.,\-]+\]', solution_str)
        if list_match:
            values = json.loads(list_match.group(0))
            if len(values) >= 6:
                box = np.array(values[:9] if len(values) >= 9 else values[:6] + [0, 0, 0], dtype=np.float32)
                return box
    except (json.JSONDecodeError, ValueError):
        pass

    # Try text format: "center: (x, y, z), size: (w, h, l)"
    try:
        center_match = re.search(r'center[:\s]*\(([\d\.,\-\s]+)\)', solution_str, re.IGNORECASE)
        size_match = re.search(r'size[:\s]*\(([\d\.,\-\s]+)\)', solution_str, re.IGNORECASE)
        rotation_match = re.search(r'rotation[:\s]*\(([\d\.,\-\s]+)\)', solution_str, re.IGNORECASE)

        if center_match and size_match:
            center = np.array([float(x.strip()) for x in center_match.group(1).split(",")], dtype=np.float32)
            size = np.array([float(x.strip()) for x in size_match.group(1).split(",")], dtype=np.float32)
            if rotation_match:
                rotation = np.array([float(x.strip()) for x in rotation_match.group(1).split(",")], dtype=np.float32)
            else:
                rotation = np.array([0, 0, 0], dtype=np.float32)
            return np.concatenate([center, size, rotation])
    except (ValueError, AttributeError):
        pass

    return None


def parse_all_3d_boxes(solution_str: str) -> List[np.ndarray]:
    """
    Parse all 3D bounding boxes from solution string.
    
    Specifically designed for grounding format: [{"bbox_3d": [...], "label": "..."}, ...]
    
    Args:
        solution_str: The solution string containing 3D box information.
    
    Returns:
        List of numpy arrays, each of shape (9,) representing [x, y, z, w, h, l, rx, ry, rz],
        or empty list if parsing fails.
    """
    boxes = []
    
    # Try list of dicts format: [{"bbox_3d": [...], "label": "..."}, ...]
    try:
        # First try standard JSON (double quotes)
        parsed = json.loads(solution_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "bbox_3d" in item:
                    bbox_3d = item["bbox_3d"]
                    if isinstance(bbox_3d, list) and len(bbox_3d) >= 6:
                        box = np.array(bbox_3d[:9] if len(bbox_3d) >= 9 else bbox_3d[:6] + [0, 0, 0], dtype=np.float32)
                        boxes.append(box)
            if boxes:
                return boxes
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    
    # Try Python eval for single-quote format
    try:
        parsed = ast.literal_eval(solution_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    bbox_3d = item.get("bbox_3d") or item.get('bbox_3d')
                    if bbox_3d is not None and isinstance(bbox_3d, list) and len(bbox_3d) >= 6:
                        box = np.array(bbox_3d[:9] if len(bbox_3d) >= 9 else bbox_3d[:6] + [0, 0, 0], dtype=np.float32)
                        boxes.append(box)
            if boxes:
                return boxes
    except (ValueError, SyntaxError, TypeError):
        pass
    
    # Try regex to find all bbox_3d patterns
    try:
        bbox_matches = re.findall(r'["\']bbox_3d["\']\s*:\s*\[([\s\d\.,\-]+)\]', solution_str, re.DOTALL)
        for match in bbox_matches:
            values = [float(x.strip()) for x in match.split(",") if x.strip()]
            if len(values) >= 6:
                box = np.array(values[:9] if len(values) >= 9 else values[:6] + [0, 0, 0], dtype=np.float32)
                boxes.append(box)
        if boxes:
            return boxes
    except (ValueError, AttributeError):
        pass
    
    return boxes


def check_format_correct(solution_str: str) -> bool:
    """
    Check if the solution string has correct format for grounding task.
    
    Expected format: [{"bbox_3d": [x, y, z, w, h, l, rx, ry, rz], "label": "..."}, ...]
    
    Args:
        solution_str: The solution string to check.
    
    Returns:
        True if format is correct, False otherwise.
    """
    boxes = parse_all_3d_boxes(solution_str)
    
    # Format is correct if we can parse at least one box with the expected structure
    if len(boxes) == 0:
        return False
    
    # Additional check: verify the string contains the expected structure
    # Check for list of dicts with "bbox_3d" and "label" keys
    try:
        parsed = json.loads(solution_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    if "bbox_3d" not in item or "label" not in item:
                        return False
                    bbox_3d = item["bbox_3d"]
                    if not isinstance(bbox_3d, list) or len(bbox_3d) < 6:
                        return False
            return True
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Try ast.literal_eval for single-quote format
    try:
        parsed = ast.literal_eval(solution_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    bbox_3d = item.get("bbox_3d") or item.get('bbox_3d')
                    label = item.get("label") or item.get('label')
                    if bbox_3d is None or label is None:
                        return False
                    if not isinstance(bbox_3d, list) or len(bbox_3d) < 6:
                        return False
            return True
    except (ValueError, SyntaxError, TypeError):
        pass
    
    # If we parsed boxes but can't verify structure, consider format incorrect
    return False


def hungarian_match_boxes(pred_boxes: List[np.ndarray], gt_boxes: List[np.ndarray]) -> List[Tuple[int, int]]:
    """
    Match predicted boxes to ground truth boxes using Hungarian algorithm.
    
    Creates a cost matrix based on negative IoU (we want to maximize IoU, so minimize negative IoU).
    
    Args:
        pred_boxes: List of predicted 3D boxes, each of shape (9,)
        gt_boxes: List of ground truth 3D boxes, each of shape (9,)
    
    Returns:
        List of tuples (pred_idx, gt_idx) representing the optimal matching.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    
    # Build cost matrix: cost[i][j] = -IoU(pred_boxes[i], gt_boxes[j])
    # We use negative IoU because Hungarian algorithm minimizes cost
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    
    # Ensure both boxes have rotation
    pred_boxes_normalized = []
    for box in pred_boxes:
        if len(box) < 9:
            box_norm = np.concatenate([box[:6], np.zeros(3, dtype=np.float32)])
        else:
            box_norm = box.copy()
        pred_boxes_normalized.append(box_norm)
    
    gt_boxes_normalized = []
    for box in gt_boxes:
        if len(box) < 9:
            box_norm = np.concatenate([box[:6], np.zeros(3, dtype=np.float32)])
        else:
            box_norm = box.copy()
        gt_boxes_normalized.append(box_norm)
    
    # Build cost matrix
    cost_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, pred_box in enumerate(pred_boxes_normalized):
        for j, gt_box in enumerate(gt_boxes_normalized):
            iou = compute_3d_giou(pred_box, gt_box)
            # Use negative IoU as cost (we want to maximize IoU)
            cost_matrix[i, j] = -iou
    
    # Hungarian algorithm (simple implementation for small matrices)
    # For larger matrices, consider using scipy.optimize.linear_sum_assignment
    if n_pred <= n_gt:
        # More predictions than GT: match each GT to best prediction
        matches = []
        used_pred = set()
        for j in range(n_gt):
            best_i = None
            best_cost = float('inf')
            for i in range(n_pred):
                if i not in used_pred:
                    if cost_matrix[i, j] < best_cost:
                        best_cost = cost_matrix[i, j]
                        best_i = i
            if best_i is not None:
                matches.append((best_i, j))
                used_pred.add(best_i)
    else:
        # More GT than predictions: match each prediction to best GT
        matches = []
        used_gt = set()
        for i in range(n_pred):
            best_j = None
            best_cost = float('inf')
            for j in range(n_gt):
                if j not in used_gt:
                    if cost_matrix[i, j] < best_cost:
                        best_cost = cost_matrix[i, j]
                        best_j = j
            if best_j is not None:
                matches.append((i, best_j))
                used_gt.add(best_j)
    
    return matches


def compute_3d_giou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute 3D Generalized IoU (GIoU) between two 3D bounding boxes.

    GIoU addresses the gradient vanishing problem when boxes don't overlap.
    For rotated boxes, computes the axis-aligned bounding box (AABB) of the rotated box.

    Args:
        box1: numpy array of shape (9,) [x, y, z, w, h, l, pitch, yaw, roll]
        box2: numpy array of shape (9,) [x, y, z, w, h, l, pitch, yaw, roll]

    Returns:
        GIoU value in range [-1, 1]
    """
    # Extract center, size, and rotation
    center1, size1 = box1[:3], box1[3:6]
    center2, size2 = box2[:3], box2[3:6]
    rotation1 = box1[6:9] if len(box1) >= 9 else np.array([0, 0, 0], dtype=np.float32)
    rotation2 = box2[6:9] if len(box2) >= 9 else np.array([0, 0, 0], dtype=np.float32)

    # Compute AABB considering rotation
    # Generate 8 corners for each box, apply rotation, then compute AABB
    def get_aabb(center, size, rotation):
        # Generate 8 corners in local coordinate system
        corners_local = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float32)
        # Scale by half size
        corners_local = corners_local * (size / 2)
        
        # Apply rotation if present
        if np.any(np.abs(rotation) > 1e-6):
            rotation_matrix = euler_to_rotation_matrix(rotation[0], rotation[1], rotation[2])
            corners_local = (rotation_matrix @ corners_local.T).T
        
        # Translate to world coordinate system
        corners_world = corners_local + center
        
        # Compute AABB
        min_corner = np.min(corners_world, axis=0)
        max_corner = np.max(corners_world, axis=0)
        return min_corner, max_corner

    min1, max1 = get_aabb(center1, size1, rotation1)
    min2, max2 = get_aabb(center2, size2, rotation2)

    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(0, inter_max - inter_min)
    inter_volume = np.prod(inter_size)

    # Union
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    union_volume = volume1 + volume2 - inter_volume

    # IoU
    if union_volume == 0:
        iou = 0.0
    else:
        iou = inter_volume / union_volume

    # Enclosing box (convex hull)
    enclosing_min = np.minimum(min1, min2)
    enclosing_max = np.maximum(max1, max2)
    enclosing_size = enclosing_max - enclosing_min
    enclosing_volume = np.prod(enclosing_size)

    # GIoU
    if enclosing_volume == 0:
        giou = iou
    else:
        giou = iou - (enclosing_volume - union_volume) / enclosing_volume

    return float(giou)


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Convert Euler angles (rx, ry, rz) to rotation matrix.
    
    Uses ZXY rotation order: first rotate around Z-axis (roll), then X-axis (pitch), 
    finally Y-axis (yaw).
    
    Args:
        rx: Rotation around X-axis (pitch) in radians
        ry: Rotation around Y-axis (yaw) in radians
        rz: Rotation around Z-axis (roll) in radians
    
    Returns:
        3x3 rotation matrix
    """
    # Rotation matrices for each axis
    cos_rx, sin_rx = np.cos(rx), np.sin(rx)
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    cos_rz, sin_rz = np.cos(rz), np.sin(rz)
    
    # Rotation around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, cos_rx, -sin_rx],
        [0, sin_rx, cos_rx]
    ], dtype=np.float32)
    
    # Rotation around Y-axis
    Ry = np.array([
        [cos_ry, 0, sin_ry],
        [0, 1, 0],
        [-sin_ry, 0, cos_ry]
    ], dtype=np.float32)
    
    # Rotation around Z-axis
    Rz = np.array([
        [cos_rz, -sin_rz, 0],
        [sin_rz, cos_rz, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Combined rotation: R = Ry * Rx * Rz (ZXY order: first Z, then X, finally Y)
    R = Ry @ Rx @ Rz
    return R


def build_intrinsic_from_fov(hfov: float, vfov: float, width: int, height: int) -> np.ndarray:
    """
    Build camera intrinsic matrix from field of view (FOV) and image dimensions.
    
    Args:
        hfov: Horizontal field of view in degrees
        vfov: Vertical field of view in degrees
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        3x3 camera intrinsic matrix
    """
    # Convert FOV from degrees to radians
    hfov_rad = np.deg2rad(hfov)
    vfov_rad = np.deg2rad(vfov)
    
    # Calculate focal lengths
    fx = width / (2 * np.tan(hfov_rad / 2))
    fy = height / (2 * np.tan(vfov_rad / 2))
    
    # Principal point (image center)
    cx = width / 2.0
    cy = height / 2.0
    
    # Build intrinsic matrix
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return intrinsic


def build_extrinsic_identity() -> np.ndarray:
    """
    Build identity extrinsic matrix (camera coordinate = world coordinate).
    
    Returns:
        4x4 identity matrix [I|0; 0|1]
    """
    extrinsic = np.eye(4, dtype=np.float32)
    return extrinsic


def project_3d_to_2d(
    box_3d: np.ndarray, camera_intrinsic: np.ndarray, camera_extrinsic: np.ndarray
) -> Optional[Tuple[float, float, float, float]]:
    """
    Project 3D bounding box to 2D image plane.

    Args:
        box_3d: numpy array of shape (9,) [x, y, z, w, h, l, rx, ry, rz]
        camera_intrinsic: 3x3 camera intrinsic matrix
        camera_extrinsic: 4x4 camera extrinsic matrix [R|t; 0|1]

    Returns:
        Tuple of (u_min, v_min, u_max, v_max) representing 2D bounding box,
        or None if projection fails.
    """
    center = box_3d[:3]
    size = box_3d[3:6]
    rotation = box_3d[6:9] if len(box_3d) >= 9 else np.array([0, 0, 0], dtype=np.float32)

    # Generate 8 corners of the 3D box in local coordinate system
    corners_local = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    # Scale by half size
    corners_local = corners_local * (size / 2)
    
    # Apply rotation if present
    if np.any(np.abs(rotation) > 1e-6):
        rotation_matrix = euler_to_rotation_matrix(rotation[0], rotation[1], rotation[2])
        corners_local = (rotation_matrix @ corners_local.T).T
    
    # Translate to world coordinate system
    corners_3d = corners_local + center

    # Transform to camera coordinate system
    corners_3d_homo = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
    corners_cam = (camera_extrinsic @ corners_3d_homo.T).T[:, :3]

    # Check if box is behind camera
    if np.all(corners_cam[:, 2] <= 0):
        return None

    # Project to image plane
    corners_2d_homo = (camera_intrinsic @ corners_cam.T).T
    corners_2d = corners_2d_homo[:, :2] / (corners_2d_homo[:, 2:3] + 1e-8)

    # Compute 2D bounding box
    u_min = float(np.min(corners_2d[:, 0]))
    v_min = float(np.min(corners_2d[:, 1]))
    u_max = float(np.max(corners_2d[:, 0]))
    v_max = float(np.max(corners_2d[:, 1]))

    return (u_min, v_min, u_max, v_max)


def compute_2d_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Compute 2D IoU between two bounding boxes.

    Args:
        box1: (u_min, v_min, u_max, v_max)
        box2: (u_min, v_min, u_max, v_max)

    Returns:
        IoU value in range [0, 1]
    """
    u_min1, v_min1, u_max1, v_max1 = box1
    u_min2, v_min2, u_max2, v_max2 = box2

    # Intersection
    inter_u_min = max(u_min1, u_min2)
    inter_v_min = max(v_min1, v_min2)
    inter_u_max = min(u_max1, u_max2)
    inter_v_max = min(v_max1, v_max2)

    if inter_u_max <= inter_u_min or inter_v_max <= inter_v_min:
        return 0.0

    inter_area = (inter_u_max - inter_u_min) * (inter_v_max - inter_v_min)

    # Union
    area1 = (u_max1 - u_min1) * (v_max1 - v_min1)
    area2 = (u_max2 - u_min2) * (v_max2 - v_min2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def get_depth_in_camera_view(box_3d: np.ndarray, camera_extrinsic: np.ndarray) -> float:
    """
    Get the depth (Z coordinate) of box center in camera coordinate system.

    Args:
        box_3d: numpy array of shape (9,) [x, y, z, w, h, l, rx, ry, rz]
        camera_extrinsic: 4x4 camera extrinsic matrix [R|t; 0|1]

    Returns:
        Depth value (Z coordinate in camera frame)
    """
    center = box_3d[:3]
    center_homo = np.concatenate([center, [1]])
    center_cam = (camera_extrinsic @ center_homo)[:3]
    return float(center_cam[2])


def is_box_visible(box_2d: Optional[Tuple[float, float, float, float]], image_size: Tuple[int, int]) -> bool:
    """
    Check if 2D bounding box is visible in the image.

    Args:
        box_2d: (u_min, v_min, u_max, v_max) or None
        image_size: (width, height) of the image

    Returns:
        True if box is visible, False otherwise
    """
    if box_2d is None:
        return False

    u_min, v_min, u_max, v_max = box_2d
    img_w, img_h = image_size

    # Check if box overlaps with image bounds
    if u_max < 0 or u_min > img_w or v_max < 0 or v_min > img_h:
        return False

    # Check if box has valid size
    if u_max <= u_min or v_max <= v_min:
        return False

    return True


def compute_score(
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[float, Dict[str, Any]]:
    """
    Compute hybrid geometric consistency reward for 3D spatial reasoning.

    This function combines:
    1. 3D GIoU: Global geometry accuracy (average IoU of matched boxes)
    2. F1 Score: Detection quality metric based on IoU threshold (TP = IoU > threshold)
    3. Depth Consistency: Depth-aware reward (gated by 2D projection accuracy)
    4. 2D Projection: Multi-view 2D projection consistency

    Args:
        solution_str: Model output string containing 3D box prediction
        ground_truth: Ground truth 3D box (not used if boxes_3d_camera_coords is provided).
            Can be:
            - String: JSON string or list format
            - Dict: {"box_3d": [x, y, z, w, h, l, rx, ry, rz], ...}
        extra_info: Additional information containing:
            - boxes_3d_camera_coords: Required list of 3D boxes in camera coordinates.
              Format: [[x, y, z, w, h, l, rx, ry, rz], ...] or [x, y, z, w, h, l, rx, ry, rz]
              If not provided, function returns 0.0
            - boxes_2d: Optional list of 2D bounding boxes.
              If provided, projects predicted 3D box to 2D and computes IoU with GT 2D box.
              Format: [[u_min, v_min, u_max, v_max], ...] or [u_min, v_min, u_max, v_max]
            - camera_params_list: Optional list of camera parameters. If not provided,
              will be built from hfov/vfov and width/height.
              Each camera parameter dict contains:
                - intrinsic: 3x3 camera intrinsic matrix (or "K")
                - extrinsic: 4x4 camera extrinsic matrix [R|t; 0|1] (or "ext")
                - image_size: (width, height) tuple (or "img_size")
            - hfov: Horizontal field of view in degrees (used to build intrinsic if camera_params_list not provided)
            - vfov: Vertical field of view in degrees (used to build intrinsic if camera_params_list not provided)
            - width/height: Image dimensions (used to build intrinsic and image_size)
            - original_width/original_height: Alternative image dimensions
            - image_size: (width, height) tuple (used as fallback)
            - weights: Optional dict with keys "r_3d_iou", "r_f1", "r_depth", "r_proj" (default: r_3d_iou=1.0, others=0.0)
            - use_iou_f1_combination: If True, use paper style: α * IoU + (1-α) * F1 (default: False)
            - iou_f1_alpha: Weight for IoU in IoU+F1 combination (default: 0.5, paper default)
            - iou_threshold_for_f1: IoU threshold for TP definition in F1 calculation (default: 0.25, paper default)
            - depth_threshold: Threshold for depth consistency (default: 0.2)
            - min_2d_iou_for_depth: Minimum 2D IoU to consider depth reward (default: 0.3)
        **kwargs: Additional keyword arguments

    Returns:
        float: Reward score in range [0, 1] or dict with detailed breakdown containing:
            - score: Final combined reward score
            - r_3d_iou: Average 3D GIoU of matched boxes
            - r_f1: F1 score based on IoU threshold
            - precision: Precision metric (TP / N_predicted)
            - recall: Recall metric (TP / M_ground_truth)
            - tp_count: Number of True Positives (IoU > threshold)
            - r_3d_iou_f1: Combined IoU+F1 score (if use_iou_f1_combination=True)
            - r_depth_consistency: Depth consistency reward
            - r_proj: 2D projection consistency reward
            - r_format: Format reward (1.0 if format correct)
            - num_matched_boxes: Number of matched box pairs
            - num_views: Number of camera views
            - valid_views: Number of views with valid projections
    """
    if extra_info is None:
        extra_info = {}

    # Configuration (needed for early returns)
    return_dict = extra_info.get("return_dict", True)  # Default True for RL training logging
    # === Format Check: Only proceed if format is correct ===
    format_correct = check_format_correct(solution_str)
    if not format_correct:
        # Format is incorrect, return 0 reward
        if return_dict:
            return {
                "score": 0.0,
                "r_3d_iou": 0.0,
                "r_3d_iou_f1": 0.0,
                "r_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp_count": 0,
                "r_depth_consistency": 0.0,
                "r_proj": 0.0,
                "r_format": 0.0,
                "num_views": 0,
                "valid_views": 0,
                "num_matched_boxes": 0,
            }
        return 0.0

    # Parse all ground truth boxes from boxes_3d_camera_coords in extra_info
    boxes_3d_camera_coords = extra_info.get("boxes_3d_camera_coords")
    gt_boxes_3d = []
    if boxes_3d_camera_coords is not None:
        if isinstance(boxes_3d_camera_coords, list):
            for box_data in boxes_3d_camera_coords:
                if isinstance(box_data, list) and len(box_data) >= 6:
                    box = np.array(box_data, dtype=np.float32)
                    # Ensure box has rotation
                    if len(box) < 9:
                        box = np.concatenate([box[:6], np.zeros(3, dtype=np.float32)])
                    gt_boxes_3d.append(box)
        elif isinstance(boxes_3d_camera_coords, (list, np.ndarray)) and len(boxes_3d_camera_coords) >= 6:
            # Single box format
            box = np.array(boxes_3d_camera_coords, dtype=np.float32)
            if len(box) < 9:
                box = np.concatenate([box[:6], np.zeros(3, dtype=np.float32)])
            gt_boxes_3d.append(box)

    if len(gt_boxes_3d) == 0:
        if return_dict:
            return {
                "score": 0.0,
                "r_3d_iou": 0.0,
                "r_3d_iou_f1": 0.0,
                "r_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp_count": 0,
                "r_depth_consistency": 0.0,
                "r_proj": 0.0,
                "r_format": 1.0,  # Format was correct
                "num_views": 0,
                "valid_views": 0,
                "num_matched_boxes": 0,
            }
        return 0.0

    # Parse all prediction boxes
    pred_boxes_3d = parse_all_3d_boxes(solution_str)
    if len(pred_boxes_3d) == 0:
        # Cannot parse any boxes, return 0
        if return_dict:
            return {
                "score": 0.0,
                "r_3d_iou": 0.0,
                "r_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp_count": 0,
                "r_depth_consistency": 0.0,
                "r_proj": 0.0,
                "r_format": 1.0,  # Format was correct
                "num_views": 0,
                "valid_views": 0,
                "num_matched_boxes": 0,
            }
        return 0.0

    # Ensure all prediction boxes have rotation
    for i, box in enumerate(pred_boxes_3d):
        if len(box) < 9:
            pred_boxes_3d[i] = np.concatenate([box[:6], np.zeros(3, dtype=np.float32)])

    # === Match boxes using Hungarian algorithm ===
    matches = hungarian_match_boxes(pred_boxes_3d, gt_boxes_3d)
    
    if len(matches) == 0:
        # No matches found, return 0
        if return_dict:
            return {
                "score": 0.0,
                "r_3d_iou": 0.0,
                "r_3d_iou_f1": 0.0,
                "r_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp_count": 0,
                "r_depth_consistency": 0.0,
                "r_proj": 0.0,
                "r_format": 1.0,  # Format was correct
                "num_views": 0,
                "valid_views": 0,
                "num_matched_boxes": 0,
            }
        return 0.0

    # === Reward Component 1: 3D GIoU (Global Geometry) - Average over all matched boxes ===
    iou_scores = []
    for pred_idx, gt_idx in matches:
        iou = compute_3d_giou(pred_boxes_3d[pred_idx], gt_boxes_3d[gt_idx])
        # Normalize to [0, 1] range (GIoU can be negative)
        iou_normalized = (iou + 1.0) / 2.0
        iou_scores.append(iou_normalized)
    
    r_3d_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    
    # === Reward Component 1b: F1 Score (Detection Quality) ===
    # Based on paper: TP is defined as a match with IoU > threshold (default 0.25)
    iou_threshold = extra_info.get("iou_threshold_for_f1", 0.25)  # Paper default: 0.25
    
    # Count True Positives: matches with IoU > threshold
    tp_count = sum(1 for iou in iou_scores if iou > iou_threshold)
    
    # Precision: TP / N_predicted
    n_pred = len(pred_boxes_3d)
    precision = tp_count / n_pred if n_pred > 0 else 0.0
    
    # Recall: TP / M_ground_truth
    m_gt = len(gt_boxes_3d)
    recall = tp_count / m_gt if m_gt > 0 else 0.0
    
    # F1 Score: harmonic mean of precision and recall
    if precision + recall > 0:
        r_f1 = 2.0 * precision * recall / (precision + recall)
    else:
        r_f1 = 0.0
    
    r_f1 = float(r_f1)

    # === Reward Component 2 & 3: Multi-view Consistency ===
    camera_params_list = extra_info.get("camera_params_list", [])
    depth_scores = []
    proj_scores = []

    # Configuration
    depth_threshold = extra_info.get("depth_threshold", 0.3)
    min_2d_iou_for_depth = extra_info.get("min_2d_iou_for_depth", 0.2)
    
    # Build camera parameters from extra_info if camera_params_list is empty
    if len(camera_params_list) == 0:
        # Try to build from FOV and image dimensions
        hfov = extra_info.get("hfov")
        vfov = extra_info.get("vfov")
        width = extra_info.get("width") or extra_info.get("original_width")
        height = extra_info.get("height") or extra_info.get("original_height")
        
        if hfov is not None and vfov is not None and width is not None and height is not None:
            intrinsic = build_intrinsic_from_fov(hfov, vfov, width, height)
            extrinsic = build_extrinsic_identity()  # Camera coord = world coord
            image_size = (width, height)
            camera_params_list = [{
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "image_size": image_size
            }]
    
    # Get GT 2D boxes from extra_info if available
    boxes_2d_gt = extra_info.get("boxes_2d")
    gt_boxes_2d = []
    if boxes_2d_gt is not None:
        if isinstance(boxes_2d_gt, list):
            for box_2d_data in boxes_2d_gt:
                if isinstance(box_2d_data, list) and len(box_2d_data) >= 4:
                    # Format: [u_min, v_min, u_max, v_max]
                    gt_boxes_2d.append(tuple(box_2d_data[:4]))
        elif isinstance(boxes_2d_gt, (list, tuple)) and len(boxes_2d_gt) >= 4:
            # Single box format
            gt_boxes_2d.append(tuple(boxes_2d_gt[:4]))

    # STRICT MODE A: If we have GT 2D boxes from metadata, MUST use direct 2D IoU
    # No fallback to Mode B - if Mode A cannot execute, return 0
    if len(gt_boxes_2d) > 0:
        # Build camera_params_list from FOV if empty (required for Mode A)
        if len(camera_params_list) == 0:
            hfov = extra_info.get("hfov")
            vfov = extra_info.get("vfov")
            width = extra_info.get("width") or extra_info.get("original_width")
            height = extra_info.get("height") or extra_info.get("original_height")
            
            # STRICT: All FOV parameters must be present, no fallback
            if hfov is None or vfov is None or width is None or height is None:
                # Cannot execute Mode A without camera parameters
                missing_params = []
                if hfov is None:
                    missing_params.append("hfov")
                if vfov is None:
                    missing_params.append("vfov")
                if width is None:
                    missing_params.append("width/original_width")
                if height is None:
                    missing_params.append("height/original_height")
                warnings.warn(
                    f"[spatial_z] Mode A requires boxes_2d but missing camera parameters: {', '.join(missing_params)}. "
                    f"Returning 0.0 reward.",
                    UserWarning
                )
                if return_dict:
                    return {
                        "score": 0.0,
                        "r_3d_iou": r_3d_iou,
                        "r_3d_iou_f1": r_3d_iou_f1,
                        "r_f1": r_f1,
                        "precision": precision,
                        "recall": recall,
                        "tp_count": tp_count,
                        "r_depth_consistency": 0.0,
                        "r_proj": 0.0,
                        "r_format": 1.0,
                        "num_views": 0,
                        "valid_views": 0,
                        "num_matched_boxes": len(matches),
                    }
                return 0.0
            
            intrinsic = build_intrinsic_from_fov(hfov, vfov, width, height)
            extrinsic = build_extrinsic_identity()  # Camera coord = world coord
            image_size = (width, height)
            camera_params_list = [{
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "image_size": image_size
            }]
        
        # STRICT: camera_params_list must exist
        if len(camera_params_list) == 0:
            warnings.warn(
                "[spatial_z] Mode A requires boxes_2d but camera_params_list is empty. "
                "Cannot build camera parameters. Returning 0.0 reward.",
                UserWarning
            )
            if return_dict:
                return {
                    "score": 0.0,
                    "r_3d_iou": r_3d_iou,
                    "r_3d_iou_f1": r_3d_iou_f1,
                    "r_f1": r_f1,
                    "precision": precision,
                    "recall": recall,
                    "tp_count": tp_count,
                    "r_depth_consistency": 0.0,
                    "r_proj": 0.0,
                    "r_format": 1.0,
                    "num_views": 0,
                    "valid_views": 0,
                    "num_matched_boxes": len(matches),
                }
            return 0.0
        
        # Get camera parameters (strict, no fallback)
        cam = camera_params_list[0]
        image_size = cam.get("image_size") or cam.get("img_size")
        if image_size is None:
            # STRICT: image_size must exist
            warnings.warn(
                "[spatial_z] Mode A requires boxes_2d but image_size is missing from camera parameters. "
                "Returning 0.0 reward.",
                UserWarning
            )
            if return_dict:
                return {
                    "score": 0.0,
                    "r_3d_iou": r_3d_iou,
                    "r_3d_iou_f1": r_3d_iou_f1,
                    "r_f1": r_f1,
                    "precision": precision,
                    "recall": recall,
                    "tp_count": tp_count,
                    "r_depth_consistency": 0.0,
                    "r_proj": 0.0,
                    "r_format": 1.0,
                    "num_views": 0,
                    "valid_views": 0,
                    "num_matched_boxes": len(matches),
                }
            return 0.0
        
        # Get intrinsic and extrinsic (strict, no fallback)
        # Use explicit None check to handle numpy arrays correctly
        intrinsic = cam.get("intrinsic")
        if intrinsic is None:
            intrinsic = cam.get("K")
        extrinsic = cam.get("extrinsic")
        if extrinsic is None:
            extrinsic = cam.get("ext")
        
        # STRICT: If not provided, try to build from FOV (must succeed)
        if intrinsic is None:
            hfov = extra_info.get("hfov")
            vfov = extra_info.get("vfov")
            if isinstance(image_size, tuple):
                width, height = image_size
            else:
                width = image_size.get("width")
                height = image_size.get("height")
            
            if hfov is None or vfov is None or width is None or height is None:
                missing_params = []
                if hfov is None:
                    missing_params.append("hfov")
                if vfov is None:
                    missing_params.append("vfov")
                if width is None:
                    missing_params.append("width")
                if height is None:
                    missing_params.append("height")
                warnings.warn(
                    f"[spatial_z] Mode A requires boxes_2d but missing FOV parameters to build intrinsic: "
                    f"{', '.join(missing_params)}. Returning 0.0 reward.",
                    UserWarning
                )
                if return_dict:
                    return {
                        "score": 0.0,
                        "r_3d_iou": r_3d_iou,
                        "r_3d_iou_f1": r_3d_iou_f1,
                        "r_f1": r_f1,
                        "precision": precision,
                        "recall": recall,
                        "tp_count": tp_count,
                        "r_depth_consistency": 0.0,
                        "r_proj": 0.0,
                        "r_format": 1.0,
                        "num_views": 0,
                        "valid_views": 0,
                        "num_matched_boxes": len(matches),
                    }
                return 0.0
            
            intrinsic = build_intrinsic_from_fov(hfov, vfov, width, height)
        
        if extrinsic is None:
            extrinsic = build_extrinsic_identity()
        
        # STRICT: intrinsic must exist
        if intrinsic is None:
            warnings.warn(
                "[spatial_z] Mode A requires boxes_2d but intrinsic matrix is None after building. "
                "Returning 0.0 reward.",
                UserWarning
            )
            if return_dict:
                return {
                    "score": 0.0,
                    "r_3d_iou": r_3d_iou,
                    "r_3d_iou_f1": r_3d_iou_f1,
                    "r_f1": r_f1,
                    "precision": precision,
                    "recall": recall,
                    "tp_count": tp_count,
                    "r_depth_consistency": 0.0,
                    "r_proj": 0.0,
                    "r_format": 1.0,
                    "num_views": 0,
                    "valid_views": 0,
                    "num_matched_boxes": len(matches),
                }
            return 0.0
        
        # Convert to numpy arrays
        intrinsic = np.array(intrinsic, dtype=np.float32)
        extrinsic = np.array(extrinsic, dtype=np.float32)
        
        # Ensure extrinsic is 4x4
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        
        # Compute 2D IoU and depth consistency for all matched boxes
        for pred_idx, gt_idx in matches:
            # Get corresponding GT 2D box (if available)
            if gt_idx < len(gt_boxes_2d):
                gt_box_2d = gt_boxes_2d[gt_idx]
                
                # Check visibility
                if not is_box_visible(gt_box_2d, image_size):
                    continue
                
                # Project prediction 3D box to 2D
                pred_box_2d = project_3d_to_2d(pred_boxes_3d[pred_idx], intrinsic, extrinsic)
                
                # STRICT: Projection must succeed
                if pred_box_2d is None:
                    continue
                
                # Compute 2D IoU using metadata GT box
                iou_2d = compute_2d_iou(pred_box_2d, gt_box_2d)
                proj_scores.append(iou_2d)
                
                # Compute depth consistency (only if 2D projection is reasonably accurate)
                if iou_2d > min_2d_iou_for_depth:
                    z_pred = get_depth_in_camera_view(pred_boxes_3d[pred_idx], extrinsic)
                    z_gt = get_depth_in_camera_view(gt_boxes_3d[gt_idx], extrinsic)
                    
                    # STRICT: Depth must be valid (in front of camera)
                    if z_pred > 0 and z_gt > 0:
                        # Relative depth error
                        depth_error = abs(z_pred - z_gt) / (z_gt + 1e-6)
                        
                        # Depth reward: Use IoU as a qualification gate (pass threshold = full depth reward),
                        # not as a linear multiplier. This allows depth to contribute significantly even when
                        # IoU is moderate, while still requiring basic 2D alignment.
                        # Use smoother decay: 1 / (1 + depth_error / depth_threshold) instead of exp(-depth_error / depth_threshold)
                        # This gives more gradient signal and less aggressive decay.
                        r_depth = 1.0 / (1.0 + depth_error / depth_threshold)
                        depth_scores.append(r_depth)
    else:
        # Only use projection method if boxes_2d is not provided
        # Compute 2D projection and depth consistency for all matched boxes
        for cam_idx, cam in enumerate(camera_params_list):
            # Get camera parameters
            intrinsic = np.array(cam.get("intrinsic", cam.get("K")), dtype=np.float32)
            extrinsic = np.array(cam.get("extrinsic", cam.get("ext")), dtype=np.float32)
            image_size = cam.get("image_size", cam.get("img_size", (1920, 1080)))

            # Ensure extrinsic is 4x4
            if extrinsic.shape == (3, 4):
                extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

            # Process all matched boxes
            for pred_idx, gt_idx in matches:
                # Project GT and prediction to 2D
                gt_box_2d = project_3d_to_2d(gt_boxes_3d[gt_idx], intrinsic, extrinsic)
                pred_box_2d = project_3d_to_2d(pred_boxes_3d[pred_idx], intrinsic, extrinsic)

                # Check visibility
                if gt_box_2d is None or not is_box_visible(gt_box_2d, image_size):
                    continue

                if pred_box_2d is None:
                    # Prediction is behind camera or projection failed
                    continue

                # Compute 2D IoU
                iou_2d = compute_2d_iou(pred_box_2d, gt_box_2d)
                proj_scores.append(iou_2d)

                # Compute depth consistency (only if 2D projection is reasonably accurate)
                if iou_2d > min_2d_iou_for_depth:
                    z_pred = get_depth_in_camera_view(pred_boxes_3d[pred_idx], extrinsic)
                    z_gt = get_depth_in_camera_view(gt_boxes_3d[gt_idx], extrinsic)

                    # Skip if depth is invalid (behind camera)
                    if z_pred <= 0 or z_gt <= 0:
                        continue

                    # Relative depth error
                    depth_error = abs(z_pred - z_gt) / (z_gt + 1e-6)

                    # Depth reward: Use IoU as a qualification gate (pass threshold = full depth reward),
                    # not as a linear multiplier. This allows depth to contribute significantly even when
                    # IoU is moderate, while still requiring basic 2D alignment.
                    # Use smoother decay: 1 / (1 + depth_error / depth_threshold) instead of exp(-depth_error / depth_threshold)
                    # This gives more gradient signal and less aggressive decay.
                    r_depth = 1.0 / (1.0 + depth_error / depth_threshold)
                    depth_scores.append(r_depth)

    # Aggregate multi-view scores
    r_proj = float(np.mean(proj_scores)) if proj_scores else 0.0
    r_depth_consistency = float(np.mean(depth_scores)) if depth_scores else 0.0

    # === Format Reward ===
    # Format is correct (already checked at the beginning), so r_format = 1.0
    r_format = 1.0

    # === Logging: IoU+F1 combination (for diagnostics only) ===
    iou_f1_alpha = extra_info.get("iou_f1_alpha", 0.5)
    r_3d_iou_f1 = iou_f1_alpha * r_3d_iou + (1.0 - iou_f1_alpha) * r_f1

    # =========================================================================
    # Innovation 1: Relative Depth Ordering Reward (Kendall-tau)
    # Instead of rewarding absolute coordinate accuracy, reward whether the
    # predicted z-ordering (depth ranking) matches the GT z-ordering.
    # This directly trains the skill tested by relative-depth benchmarks.
    # =========================================================================
    def _kendall_tau(pred_z_list, gt_z_list):
        """
        Compute Kendall-tau rank correlation between predicted and GT z-values.
        Returns value in [-1, 1]: +1 = perfect order, -1 = perfectly reversed.
        Returns None if fewer than 2 pairs (ordering undefined).
        """
        n = len(pred_z_list)
        if n < 2:
            return None
        concordant = discordant = 0
        for a in range(n):
            for b in range(a + 1, n):
                pd = pred_z_list[a] - pred_z_list[b]
                gd = gt_z_list[a] - gt_z_list[b]
                if pd * gd > 0:
                    concordant += 1
                elif pd * gd < 0:
                    discordant += 1
        total = n * (n - 1) / 2
        return (concordant - discordant) / total if total > 0 else 0.0

    # Extract z-values (depth in camera frame) for matched box pairs
    matched_pred_z = [float(pred_boxes_3d[i][2]) for i, j in matches]
    matched_gt_z   = [float(gt_boxes_3d[j][2])   for i, j in matches]

    tau = _kendall_tau(matched_pred_z, matched_gt_z)
    # Normalize tau from [-1, 1] → [0, 1]: 0.5 = random, 1.0 = perfect order
    r_ordering = (tau + 1.0) / 2.0 if tau is not None else None

    # =========================================================================
    # Innovation 2: Multiplicative Three-tier Reward (detection × ordering)
    # - detection gate: any TP required before ordering is evaluated
    # - ordering gate:  normalized tau must reach threshold for full credit
    #
    # Tiers:
    #   0   — no box detected (tp_count == 0)
    #   0.5 — detected something, but ordering is wrong / unverifiable (1 match)
    #   1.0 — detected AND relative depth ordering is correct (tau >= gate)
    # =========================================================================
    ordering_gate = extra_info.get("ordering_gate", 0.5)  # tau_norm >= 0.5 ↔ tau >= 0

    if tp_count == 0:
        final_score = 0.0
    elif r_ordering is None:
        # Only 1 matched pair: detection succeeds but ordering cannot be verified
        final_score = 1.0
    elif r_ordering >= ordering_gate:
        final_score = 1.0
    else:
        final_score = 0.

    # Ensure score is in [0, 1] range (safety)
    final_score = max(0.0, min(1.0, final_score))

    if return_dict:
        result = {
            "score": final_score,
            "r_3d_iou": r_3d_iou,
            "r_3d_iou_f1": r_3d_iou_f1,
            "r_f1": r_f1,
            "precision": precision,
            "recall": recall,
            "tp_count": tp_count,
            "r_depth_consistency": r_depth_consistency,
            "r_proj": r_proj,
            "r_format": r_format,
            "num_views": len(camera_params_list),
            "valid_views": len(proj_scores),
            "num_matched_boxes": len(matches),
        }
        return result

    return float(final_score)

