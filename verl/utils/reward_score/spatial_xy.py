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
XY Spatial Reasoning Reward Function - Full Projection Version

This is an optimized version that performs full-image reprojection first,
then samples the corresponding points for reward calculation.

Key differences from spatial_xy.py:
1. Performs full-depth-map reprojection (View 1 -> View 2) once
2. Samples projected coordinates directly from the reprojection map
3. More efficient for multiple candidate points

Designed for point correspondence tasks where we need to find matching points across views.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DINOv2 features will be disabled.")


def parse_point_correspondence(solution_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse point correspondence answer from solution string.

    Supports formats:
    - "The corresponding point is A"
    - "Answer: B"
    - "A" (single letter)
    - JSON: {"answer": "A", "point": [u, v]}

    Args:
        solution_str: The solution string containing the answer.

    Returns:
        Dict with keys:
            - "label": str, the label (A, B, C, D, etc.)
            - "point": Optional[Tuple[float, float]], if coordinates are provided
        or None if parsing fails.
    """
    # Try to extract label (A, B, C, D, etc.)
    label_pattern = r'\b([A-Z])\b'
    matches = re.findall(label_pattern, solution_str.upper())
    
    if matches:
        # Take the last match (usually the answer)
        label = matches[-1]
        result = {"label": label}
        
        # Try to extract coordinates if present
        coord_pattern = r'\[([\d\.,\-\s]+)\]|\(([\d\.,\-\s]+)\)'
        coord_match = re.search(coord_pattern, solution_str)
        if coord_match:
            coords_str = coord_match.group(1) or coord_match.group(2)
            try:
                coords = [float(x.strip()) for x in coords_str.split(",")]
                if len(coords) >= 2:
                    result["point"] = (coords[0], coords[1])
            except ValueError:
                pass
        
        return result
    
    # Try JSON format
    try:
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}|' r'\{[^{}]*"label"[^{}]*\}', solution_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            result = {"label": str(data.get("answer", data.get("label", ""))).upper()}
            if "point" in data:
                result["point"] = tuple(data["point"][:2])
            return result
    except (json.JSONDecodeError, KeyError):
        pass
    
    return None


def reproject_full_depth_map(
    depth1: np.ndarray,
    K1: np.ndarray,
    pose1: np.ndarray,
    K2: np.ndarray,
    pose2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproject entire depth map from View 1 to View 2 pixel coordinates.
    
    This performs a full-image reprojection following the standard computer vision pipeline:
    View 1 pixel → View 1 camera → World → View 2 camera → View 2 pixel
    
    Args:
        depth1: Depth map for view 1, shape (H, W)
        K1: Camera intrinsic matrix for view 1, shape (3, 3) or (4, 4)
        pose1: Camera-to-World matrix for view 1, shape (4, 4)
        K2: Camera intrinsic matrix for view 2, shape (3, 3) or (4, 4)
        pose2: Camera-to-World matrix for view 2, shape (4, 4)
    
    Returns:
        Tuple of (u_proj, v_proj, valid_mask, z_proj):
            - u_proj: Projected u coordinates in View 2, shape (H, W)
            - v_proj: Projected v coordinates in View 2, shape (H, W)
            - valid_mask: Boolean mask for valid projections, shape (H, W)
            - z_proj: Projected depth in View 2 camera coordinates, shape (H, W)
    """
    H, W = depth1.shape
    
    # Ensure K1 and K2 are properly formatted
    if K1.shape == (3, 3):
        K1_4x4 = np.eye(4)
        K1_4x4[:3, :3] = K1
        K1_3x3 = K1
    else:
        K1_4x4 = K1
        K1_3x3 = K1[:3, :3]
    
    if K2.shape == (3, 3):
        K2_4x4 = np.eye(4)
        K2_4x4[:3, :3] = K2
        K2_3x3 = K2
    else:
        K2_4x4 = K2
        K2_3x3 = K2[:3, :3]
    
    # Step 1: Create pixel coordinate grids (following debug_2d_3d_mapping.py)
    # xmap: u coordinates (columns), shape (H, W)
    # ymap: v coordinates (rows), shape (H, W)
    xmap = np.array([[i for i in range(W)] for j in range(H)])  # (H, W), u coordinates
    ymap = np.array([[j for i in range(W)] for j in range(H)])  # (H, W), v coordinates
    
    # Step 2: Unproject View 1 pixels to View 1 camera coordinates
    # X_cam = (u - cx) * depth / fx
    # Y_cam = (v - cy) * depth / fy
    # Z_cam = depth
    pts0 = (xmap - K1_4x4[0, 2]) * depth1 / K1_4x4[0, 0]  # X_cam
    pts1 = (ymap - K1_4x4[1, 2]) * depth1 / K1_4x4[1, 1]  # Y_cam
    pts2 = depth1  # Z_cam
    
    # Stack to form camera coordinates: (H, W, 3)
    pts_cam = np.transpose(np.stack([pts0, pts1, pts2]), (1, 2, 0))
    
    # Add homogeneous coordinate: (H, W, 4)
    pts_cam_homo = np.concatenate([pts_cam, np.ones([pts_cam.shape[0], pts_cam.shape[1], 1])], axis=2)
    
    # Reshape to (HW, 4) and transpose to (4, HW) for matrix multiplication
    pts_cam_homo_flat = pts_cam_homo.reshape(-1, 4).T  # (4, HW)
    
    # Step 3: Transform to world coordinates using Camera-to-World matrix
    camera_to_world1 = pose1
    pts_world_homo = np.matmul(camera_to_world1, pts_cam_homo_flat)  # (4, HW)
    pts_world = pts_world_homo[:3, :]  # (3, HW)
    
    # Step 4: Transform to View 2 camera coordinates
    # Get World-to-Camera matrix for view 2 (inverse of Camera-to-World)
    world_to_camera2 = np.linalg.inv(pose2)

    
    # Convert to homogeneous: (4, HW)
    pts_world_homo_2 = np.vstack([pts_world, np.ones((1, pts_world.shape[1]))])
    pts_cam2_homo = np.matmul(world_to_camera2, pts_world_homo_2)  # (4, HW)
    pts_cam2 = pts_cam2_homo[:3, :]  # (3, HW)
    
    # Step 5: Check if points are behind camera
    z2 = pts_cam2[2, :]
    valid_depth = z2 > 0
    
    # Step 6: Project to View 2 pixel coordinates using K2
    uv2_homo = K2_3x3 @ pts_cam2  # (3, HW)
    
    # Avoid division by zero
    z_safe = np.where(np.abs(uv2_homo[2, :]) < 1e-8, 1e-8, uv2_homo[2, :])
    uv2 = uv2_homo[:2, :] / z_safe  # (2, HW)
    u2, v2 = uv2[0, :], uv2[1, :]
    
    # Step 7: Check image bounds
    valid_bounds = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
    
    # Combine validity checks
    valid_mask_flat = valid_depth & valid_bounds & np.isfinite(u2) & np.isfinite(v2)
    
    # Reshape to (H, W)
    u_proj = u2.reshape(H, W)
    v_proj = v2.reshape(H, W)
    valid_mask = valid_mask_flat.reshape(H, W)
    z_proj = z2.reshape(H, W)
    
    # Note: We don't set invalid pixels to NaN here because:
    # 1. Performance: Setting NaN adds ~4ms overhead per call (11x slower)
    # 2. Redundant: valid_mask already tracks which pixels are valid
    # 3. Safe: compute_reprojection_reward_from_map checks valid_mask before accessing u_proj/v_proj
    
    return u_proj, v_proj, valid_mask, z_proj


def compute_overlap_mask_from_map(
    depth2: np.ndarray,
    u_proj_map: np.ndarray,
    v_proj_map: np.ndarray,
    z_proj_map: np.ndarray,
    valid_mask: np.ndarray,
    depth_threshold: float = 0.05,
) -> np.ndarray:
    """
    Compute overlap mask for view 2 using reprojection map + depth consistency.

    Args:
        depth2: Depth map for view 2, shape (H2, W2)
        u_proj_map: Projected u coordinates in View 2, shape (H1, W1)
        v_proj_map: Projected v coordinates in View 2, shape (H1, W1)
        z_proj_map: Projected depth in View 2 camera coordinates, shape (H1, W1)
        valid_mask: Valid projection mask, shape (H1, W1)
        depth_threshold: Absolute depth consistency threshold (in same units as depth maps)

    Returns:
        mask2: Boolean mask for view 2, True where pixels are visible from view 1
    """
    H2, W2 = depth2.shape
    mask2 = np.zeros((H2, W2), dtype=bool)

    if depth2.size == 0:
        return mask2

    valid_idx = np.where(valid_mask)
    if valid_idx[0].size == 0:
        return mask2

    u_proj = u_proj_map[valid_idx]
    v_proj = v_proj_map[valid_idx]
    z_proj = z_proj_map[valid_idx]

    u_int = np.round(u_proj).astype(int)
    v_int = np.round(v_proj).astype(int)

    in_bounds = (u_int >= 0) & (u_int < W2) & (v_int >= 0) & (v_int < H2)
    if not in_bounds.any():
        return mask2

    u_int = u_int[in_bounds]
    v_int = v_int[in_bounds]
    z_proj = z_proj[in_bounds]

    depth2_vals = depth2[v_int, u_int]
    depth_valid = (
        np.isfinite(depth2_vals)
        & (depth2_vals > 0)
        & np.isfinite(z_proj)
        & (z_proj > 0)
    )
    if not depth_valid.any():
        return mask2

    depth_diff = np.abs(depth2_vals - z_proj)
    consistent = depth_valid & (depth_diff <= depth_threshold)
    # consistent = depth_valid

    if consistent.any():
        mask2[v_int[consistent], u_int[consistent]] = True

    return mask2


def compute_reprojection_reward_from_map(
    pt_ref: Tuple[float, float],
    u_proj_map: np.ndarray,
    v_proj_map: np.ndarray,
    valid_mask: np.ndarray,
    pt_pred: Tuple[float, float],
    img_size2: Tuple[int, int],
    sigma: float = 10.0,
) -> float:
    """
    Compute reprojection reward using precomputed projection map.
    
    Args:
        pt_ref: Reference point (u, v) in view 1 (pixel coordinates)
        u_proj_map: Precomputed u projection map from View 1 to View 2, shape (H, W)
        v_proj_map: Precomputed v projection map from View 1 to View 2, shape (H, W)
        valid_mask: Valid projection mask, shape (H, W)
        pt_pred: Predicted point (u, v) in view 2 (pixel coordinates)
        img_size2: Image size (H, W) for view 2
        sigma: Tolerance radius in normalized units [0, 1]
    
    Returns:
        Reward value in range [0, 1]
    """
    # Extract reference point coordinates
    u_ref, v_ref = pt_ref
    u_ref_int = int(round(u_ref))
    v_ref_int = int(round(v_ref))
    
    H1, W1 = u_proj_map.shape
    H2, W2 = img_size2
    
    # Check if reference point is within bounds
    if not (0 <= v_ref_int < H1 and 0 <= u_ref_int < W1):
        return 0.0
    
    # Check if projection is valid
    if not valid_mask[v_ref_int, u_ref_int]:
        return 0.0
    
    # Get projected coordinates in View 2
    # Note: No need to check np.isfinite() here because valid_mask already ensures validity
    u_proj = u_proj_map[v_ref_int, u_ref_int]
    v_proj = v_proj_map[v_ref_int, u_ref_int]
    
    # Extract predicted point coordinates
    u_pred, v_pred = pt_pred
    
    # Normalize coordinates to [0, 1] for comparison
    u_proj_norm = u_proj / W2
    v_proj_norm = v_proj / H2
    u_pred_norm = u_pred / W2
    v_pred_norm = v_pred / H2
    
    # Compute distance in normalized space
    dist = np.linalg.norm(np.array([u_pred_norm, v_pred_norm]) - np.array([u_proj_norm, v_proj_norm]))
    
    # Check if distance is valid
    if not np.isfinite(dist):
        return 0.0
    
    # Check for valid sigma
    if sigma <= 0 or not np.isfinite(sigma):
        return 0.0
    
    # Soft reward with gaussian kernel
    reward = np.exp(-dist / sigma)
    
    # Check if reward is valid
    if not np.isfinite(reward):
        return 0.0
    
    # Hard penalty if too far
    if dist > 3 * sigma:
        return 0.0
    
    return float(reward)


def compute_score(
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[float, Dict[str, Any]]:
    """
    Compute XY spatial reasoning reward using full projection map.
    
    This optimized version:
    1. Performs full depth map reprojection once (View 1 -> View 2)
    2. Samples projected coordinates for the reference point
    3. Compares with predicted point to compute reward
    
    Args:
        solution_str: Model output string containing point correspondence answer
        ground_truth: Ground truth answer. Can be:
            - String: Label like "A" or dict with "label" and "point"
            - Dict: {"label": "A", "point": (u, v), "ref_point": (u, v)}
        extra_info: Additional information containing:
            - image1, image2: RGB images for two views (numpy arrays)
            - depth1, depth2: Depth maps for two views (numpy arrays)
            - K1, K2: Camera intrinsic matrices (3x3 or 4x4)
            - pose1, pose2: Camera-to-World matrices (4x4)
            - candidate_points: Dict mapping labels to (u, v) coordinates
            - correspondence_points: List of correspondence points
            - ref_point: Reference point (u, v) in view 1
            - weights: Dict with "reprojection" weight (default: 1.0)
            - return_dict: Whether to return detailed breakdown
        **kwargs: Additional keyword arguments
    
    Returns:
        float: Reward score in range [0, 1] or dict with detailed breakdown
    """
    if extra_info is None:
        extra_info = {}

    return_dict = extra_info.get("return_dict", True)

    def _return_zero_dict(pred_label_val: str = "", gt_label_val: str = ""):
        if return_dict:
            weights = extra_info.get("weights", {})
            w_reprojection = weights.get("reprojection", 1.0)
            w_contrastive = weights.get("contrastive", 0.0)
            return {
                "score": 0.0,
                "r_contrastive": 0.0,
                "r_reprojection": 0.0,
                "label_correct": 0.0,
                "pred_label": pred_label_val,
                "gt_label": gt_label_val,
                "w_contrastive": float(w_contrastive),
                "w_reprojection": float(w_reprojection),
            }
        return 0.0
    
    # Parse solution
    solution = parse_point_correspondence(solution_str)
    if solution is None:
        return _return_zero_dict()
    
    pred_label = solution["label"]
    
    # Parse ground truth
    if isinstance(ground_truth, str):
        gt_label = ground_truth.upper().strip()
    elif isinstance(ground_truth, dict):
        gt_label = str(ground_truth.get("label", ground_truth.get("answer", ""))).upper()
    else:
        gt_label = None
    
    # Get camera parameters and images
    K1 = extra_info.get("K1")
    pose1 = extra_info.get("pose1")
    K2 = extra_info.get("K2")
    pose2 = extra_info.get("pose2")
    image1 = extra_info.get("image1")
    depth1 = extra_info.get("depth1")
    image2 = extra_info.get("image2")
    depth2 = extra_info.get("depth2")
    
    # Check if all required data is available
    if depth1 is None or K1 is None or pose1 is None or K2 is None or pose2 is None:
        print(f"[spatial_xy] Missing required spatial data")
        return _return_zero_dict(pred_label, gt_label or "")
    
    # Get image sizes
    if image1 is not None:
        if isinstance(image1, np.ndarray):
            if image1.ndim == 3:
                img_size1 = (image1.shape[0], image1.shape[1])  # (H, W)
            else:
                img_size1 = (image1.shape[1], image1.shape[2])  # (H, W)
        else:
            img_size1 = None
    else:
        img_size1 = None
    
    if image2 is not None:
        if isinstance(image2, np.ndarray):
            if image2.ndim == 3:
                img_size2 = (image2.shape[0], image2.shape[1])  # (H, W)
            else:
                img_size2 = (image2.shape[1], image2.shape[2])  # (H, W)
        else:
            img_size2 = None
    else:
        img_size2 = None
    
    # Get candidate points
    candidate_points = extra_info.get("candidate_points", {})
    
    # If candidate_points not provided, extract from correspondence_points
    if not candidate_points:
        correspondence_points = extra_info.get("correspondence_points", [])
        if len(correspondence_points) >= 5:
            labels = ["A", "B", "C", "D"]
            candidate_points = {}
            for i, label in enumerate(labels):
                if i + 1 < len(correspondence_points):
                    pt = correspondence_points[i + 1]
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        candidate_points[label] = tuple(pt[:2])
    
    # Get predicted point
    if pred_label not in candidate_points:
        return _return_zero_dict(pred_label, gt_label or "")
    
    pt_pred = candidate_points[pred_label]
    
    # Get reference point
    ref_point = extra_info.get("ref_point")
    if ref_point is None:
        correspondence_points = extra_info.get("correspondence_points", [])
        if len(correspondence_points) > 0:
            ref_pt = correspondence_points[0]
            if isinstance(ref_pt, (list, tuple)) and len(ref_pt) >= 2:
                ref_point = tuple(ref_pt[:2])
    
    if ref_point is None:
        return _return_zero_dict(pred_label, gt_label or "")
    
    # Perform full depth map reprojection (View 1 -> View 2)
    try:
        u_proj_map, v_proj_map, valid_mask, z_proj_map = reproject_full_depth_map(
            depth1, K1, pose1, K2, pose2
        )
    except Exception as e:
        print(f"[spatial_xy] Reprojection failed: {e}")
        return _return_zero_dict(pred_label, gt_label or "")
    
    # Compute reprojection reward using the projection map
    r_reprojection = compute_reprojection_reward_from_map(
        ref_point,
        u_proj_map,
        v_proj_map,
        valid_mask,
        pt_pred,
        img_size2,
        sigma=extra_info.get("reprojection_sigma", 0.2)
    )
    
    # Check if r_reprojection is NaN/inf
    if not np.isfinite(r_reprojection):
        r_reprojection = 0.0

    # Additional: overlap mask check with depth consistency (scheme B)
    if r_reprojection > 0 and depth2 is not None:
        depth_threshold = extra_info.get("overlap_depth_threshold", 0.05)
        try:
            mask2 = compute_overlap_mask_from_map(
                depth2, u_proj_map, v_proj_map, z_proj_map, valid_mask, depth_threshold=depth_threshold
            )
            u_pred_px, v_pred_px = int(round(pt_pred[0])), int(round(pt_pred[1]))
            if 0 <= v_pred_px < mask2.shape[0] and 0 <= u_pred_px < mask2.shape[1]:
                if not mask2[v_pred_px, u_pred_px]:
                    r_reprojection = 0.0
            else:
                print(
                    f"[spatial_xy] Predicted point out of bounds for mask2: "
                    f"u={u_pred_px}, v={v_pred_px}, mask2_shape={mask2.shape}"
                )
        except Exception as e:
            print(f"[spatial_xy] Overlap mask check failed: {e}")
    
    # Get weights
    weights = extra_info.get("weights", {})
    w_reprojection = weights.get("reprojection", 1.0)
    w_contrastive = weights.get("contrastive", 0.0)  # v2 doesn't use contrastive, but keep for compatibility
    
    # Compute final score
    final_score = w_reprojection * r_reprojection
    
    # Check if final_score is NaN/inf
    if not np.isfinite(final_score):
        final_score = 0.0
    
    # Ensure score is in [0, 1] range
    final_score = max(0.0, min(1.0, final_score))
    
    # Check label correctness
    label_correct = 1.0 if (gt_label and pred_label == gt_label) else 0.0
    
    # Return dict format for logging
    if return_dict:
        result_dict = {
            "score": float(final_score),
            "r_contrastive": 0.0,  # v2 doesn't compute contrastive reward, set to 0.0 for compatibility
            "r_reprojection": float(r_reprojection),
            "label_correct": float(label_correct),
            "pred_label": pred_label,
            "gt_label": gt_label,
            "w_contrastive": float(w_contrastive),  # Keep for compatibility with v1
            "w_reprojection": float(w_reprojection),
        }
        return result_dict
    
    return float(final_score)
