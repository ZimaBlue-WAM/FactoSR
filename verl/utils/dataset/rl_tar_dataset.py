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
RL TarDataset for Reinforcement Learning Training

This module provides a TarDataset implementation specifically designed for RL training.
It extends the base TarDataset functionality to support RL-specific fields like
reward_model, data_source, and extra_info that are required for reward computation.

Key differences from SFT TarDataset:
- Extracts reward_model.ground_truth from JSON for accuracy checking
- Extracts data_source from JSON for reward function routing
- Extracts extra_info from JSON for additional reward configuration
- Returns data in format compatible with RLHFDataset (for collate_fn compatibility)
"""

import json
import math
import os
import random
from io import BytesIO
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.utils.data
import webdataset as wds
from PIL import Image
from omegaconf import DictConfig, ListConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

# Increase PIL Image size limit to handle large images (default is ~89M pixels)
# Set to a reasonable limit (e.g., 500M pixels) to avoid decompression bomb warnings
# while still preventing truly malicious images
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500M pixels

from verl.utils.dataset.tar_dataset import (
    TarDataset,
    convert_intern_format_to_hf_format,
    get_tar_urls_under_dir,
    tar_collate_fn,
    convert_intern_format_to_verl_format,
)
import numpy as np
from collections import defaultdict


class RLTarDataset(TarDataset):
    """
    TarDataset for RL training that supports reward_model, data_source, and extra_info fields.
    
    This class extends TarDataset to extract RL-specific fields from JSON data in tar files.
    It ensures compatibility with RL training pipeline by returning data in the same format
    as RLHFDataset (with reward_model, data_source fields in the output dict).
    
    Args:
        tar_files: list of tar file paths or directory containing tar files
        tokenizer: HuggingFace tokenizer
        config: data config (same as TarDataset)
        processor: HuggingFace processor for multimodal (optional)
        max_samples: maximum samples (not used for WebDataset)
    """

    def __init__(
        self,
        tar_files=None,
        data_files=None,
        tokenizer: PreTrainedTokenizer = None,
        config: DictConfig = None,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Support both 'tar_files' and 'data_files' parameter names for compatibility
        # with create_rl_dataset which uses 'data_files'
        if tar_files is None:
            if data_files is not None:
                tar_files = data_files
            else:
                raise ValueError("RLTarDataset requires either 'tar_files' or 'data_files' parameter")
        
        # Ensure config is not None and has a seed value
        if config is None:
            from omegaconf import OmegaConf
            config = OmegaConf.create({})
        
        # Ensure seed has a default value if None
        if config.get("seed") is None:
            config.seed = 42
        
        # Initialize parent TarDataset (but we'll override dataset_pipelines)
        super().__init__(tar_files, tokenizer, config, processor, max_samples)
        
        # RL-specific configuration
        self.reward_fn_key = config.get("reward_fn_key", "data_source")
        self.default_data_source = config.get("default_data_source", "accuracy+format")
        
        # Override dataset_pipelines to use raw WebDataset (without decode_sample)
        # RLTarDataset needs raw samples with 'json' key, not decoded samples
        self._create_raw_webdataset_pipelines()

    def _create_raw_webdataset_pipelines(self):
        """Create WebDataset pipelines without decode_sample for RLTarDataset.
        
        This creates raw WebDataset pipelines that return samples with 'json' key,
        instead of decoded samples with 'images' and 'messages'.
        
        Reference: tar_dataset.py create_webdataset function structure
        """
        import random
        import webdataset as wds
        import torch.distributed as dist
        
        self.dataset_pipelines = []
        for idx, info in enumerate(self.dataset_infos):
            dataset_seed = self.base_seed + idx * 997
            rank = dist.get_rank() if dist.is_initialized() else 0
            seed = dataset_seed + rank
            
            # Create raw WebDataset pipeline matching tar_dataset.py structure
            # but WITHOUT decode_sample and select - we want raw samples
            pipeline = wds.DataPipeline(
                # Same structure as create_webdataset in tar_dataset.py
                wds.ResampledShards(info["tar_urls"]),
                # at this point we have an iterator over all the shards
                # this shuffles the shards
                wds.shuffle(1000, rng=random.Random(seed)),
                # add wds.split_by_node here if you are using multiple nodes
                wds.split_by_node,
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                # this shuffles the samples in memory
                wds.shuffle(1000, rng=random.Random(seed)),
                # NOTE: We skip wds.map(decode_sample) and wds.select() here
                # because we want raw samples with 'json' key for _decode_sample_for_rl
            )
            self.dataset_pipelines.append(pipeline)

    def __len__(self):
        """Return the number of samples per cycle.
        
        For IterableDataset, this returns samples_per_cycle which represents
        the number of samples in one sampling cycle. This allows DataLoader
        to calculate batch count properly.
        """
        return self.samples_per_cycle

    def _load_pose_matrix(self, pose_path: str):
        """
        Load 4x4 pose matrix (Camera-to-World) from text file.
        
        Args:
            pose_path: Path to pose file (text format, 4x4 matrix)
        
        Returns:
            4x4 numpy array or None if loading fails
        """
        try:
            import numpy as np
            if not os.path.exists(pose_path):
                return None
            
            # Try to load as space-separated or tab-separated matrix
            matrix = np.loadtxt(pose_path)
            
            # Ensure it's 4x4
            if matrix.shape == (4, 4):
                return matrix.astype(np.float32)
            elif matrix.shape == (16,):
                # Reshape if flattened
                return matrix.reshape(4, 4).astype(np.float32)
            else:
                return None
        except Exception:
            return None

    def _load_intrinsic_matrix(self, intrinsic_path: str):
        """
        Load 3x3 or 4x4 intrinsic matrix from text file.
        
        Args:
            intrinsic_path: Path to intrinsic file (text format)
        
        Returns:
            3x3 or 4x4 numpy array or None if loading fails
        """
        try:
            import numpy as np
            if not os.path.exists(intrinsic_path):
                return None
            
            # Try to load as space-separated or tab-separated matrix
            matrix = np.loadtxt(intrinsic_path)
            
            # Handle different shapes
            if matrix.shape == (3, 3):
                return matrix.astype(np.float32)
            elif matrix.shape == (4, 4):
                # Extract 3x3 from top-left
                return matrix[:3, :3].astype(np.float32)
            elif matrix.shape == (9,):
                # Reshape if flattened 3x3
                return matrix.reshape(3, 3).astype(np.float32)
            elif matrix.shape == (16,):
                # Reshape if flattened 4x4, extract 3x3
                matrix_4x4 = matrix.reshape(4, 4)
                return matrix_4x4[:3, :3].astype(np.float32)
            else:
                return None
        except Exception:
            return None

    def _load_depth_map(self, depth_path: str, depth_scale: float = 1000.0, target_size: Optional[Tuple[int, int]] = None):
        """
        Load depth map from image file and optionally resize to match RGB image size.
        
        Args:
            depth_path: Path to depth image file (PNG format)
            depth_scale: Scale factor to convert depth values (default: 1000.0)
            target_size: Optional target size (width, height) to resize depth map to match RGB image
        
        Returns:
            Depth map as numpy array (H, W) or None if loading fails
        """
        try:
            import numpy as np
            from PIL import Image as PILImage
            if not os.path.exists(depth_path):
                return None
            
            # Load depth image (usually 16-bit PNG)
            depth_img = Image.open(depth_path)
            depth_array = np.array(depth_img, dtype=np.float32)
            
            # Resize depth map to match RGB image size if target_size is provided
            if target_size is not None:
                width, height = target_size
                # Use PIL to resize (nearest neighbor interpolation to preserve depth values)
                depth_img_resized = depth_img.resize((width, height), PILImage.NEAREST)
                depth_array = np.array(depth_img_resized, dtype=np.float32)
            
            # Scale depth values
            depth_array = depth_array / depth_scale
            
            return depth_array
        except Exception:
            return None

    def _decode_sample_for_rl(self, sample, default_view_type=None):
        """
        Decode sample from tar file with RL-specific field extraction.
        
        This is similar to TarDataset's decode_sample but extracts additional
        RL training fields (reward_model, data_source, extra_info) from JSON.
        
        Args:
            sample: dict from WebDataset, e.g. {
                "__key__": "sample1",
                "0.jpg": bytes,
                "1.jpg": bytes,
                "json": bytes,
            }
            default_view_type: Optional default view_type for this dataset
        
        Returns:
            dict with images, messages, has_view_type, view_type,
            reward_model, data_source, extra_info
            or None if invalid
        """
        # Track decode failures for debugging (store in instance variable)
        self._last_decode_failure_reason = None
        
        # === 1. Decode images ===
        # Load images directly (same as tar_dataset.py)
        # Images will be resized later by process_image() and processor
        images = []
        extensions = ("jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG", "WEBP")
        MAX_IMAGE_PIXELS = 100_000_000  # Skip images larger than 100M pixels to avoid slow processing
        image_load_errors = []  # Track image loading errors for debugging
        
        for k, v in sample.items():
            if any(k.endswith(ext) for ext in extensions):
                img = None
                try:
                    # Open and load image (same as tar_dataset.py)
                    img = Image.open(BytesIO(v))
                    # Check image size before processing
                    if img.width > 0 and img.height > 0:
                        total_pixels = img.width * img.height
                        # Skip extremely large images that would be too slow to process
                        if total_pixels > MAX_IMAGE_PIXELS:
                            if img:
                                img.close()
                            image_load_errors.append(f"{k}:too_large_{total_pixels}")
                            
                            return None
                        # Validate aspect ratio
                        image_ratio = img.width / img.height
                        if image_ratio > 200 or image_ratio < 0.005:
                            if img:
                                img.close()
                            print(f"Skipping image with invalid aspect ratio: {k} ({image_ratio:.2f})")
                            self._last_decode_failure_reason = f"invalid_aspect_ratio_{image_ratio:.2f}_key={k}"
                            return None
                        # Keep image open (don't convert to RGB here, processor will handle it)
                        images.append(img)
                    else:
                        print(f"Skipping image with zero dimensions: {k}")
                        if img:
                            try:
                                w, h = img.width, img.height
                                img.close()
                                image_load_errors.append(f"{k}:invalid_size_{w}x{h}")
                            except:
                                img.close()
                                image_load_errors.append(f"{k}:invalid_size_unknown")
                        else:
                            image_load_errors.append(f"{k}:invalid_size_no_image")
                except Exception as e:
                    # Track why image loading failed
                    print(f"Failed to load image {k}: {str(e)}")    
                    if img:
                        try:
                            img.close()
                        except:
                            pass
                    error_type = type(e).__name__
                    error_msg = str(e)[:100] if str(e) else ""
                    image_load_errors.append(f"{k}:{error_type}_{error_msg}")
                    continue
        
        # === 2. Decode json ===
        json_data = None
        json_key = None
        # Try multiple patterns for JSON key
        json_patterns = ["json", ".json", "json.txt", "data.json"]
        
        for k, v in sample.items():
            # Try exact match first
            if k == "json" or k.endswith("json") or k.endswith(".json"):
                json_key = k
                try:
                    json_data = json.loads(v.decode("utf-8"))
                except Exception as e:
                    self._last_decode_failure_reason = f"json_decode_error_{type(e).__name__}"
                    return None
                break
        
        if json_data is None:
            # Log sample keys for debugging
            sample_keys = [k for k in sample.keys() if not k.startswith("__")]
            raise ValueError(f"No JSON key found in sample. Sample keys: {sample_keys[:5]}")

        # Convert string to dict if needed (handle nested JSON strings)
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse json_data as JSON string: {e}")
        
        # Ensure json_data is a dict
        if not isinstance(json_data, dict):
            raise TypeError(f"json_data must be a dict, but got {type(json_data).__name__}: {json_data}")

        # === 3. Convert to HF format ===
        messages = None
        image_num = 0
        
        if "messages" in json_data and json_data["messages"]:
            messages = json_data["messages"]
            # Convert string to list if needed (handle JSON string)
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse messages as JSON string: {e}")
            
            # Ensure messages is a list
            if not isinstance(messages, list):
                raise TypeError(f"messages must be a list, but got {type(messages).__name__}: {messages}")
            
            # Count images in messages
            for idx, msg in enumerate(messages):
                # Convert string to dict if needed
                if isinstance(msg, str):
                    try:
                        msg = json.loads(msg)
                        messages[idx] = msg  # Update the list
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse message[{idx}] as JSON string: {e}")
                
                # Ensure each message is a dict
                if not isinstance(msg, dict):
                    raise TypeError(f"message[{idx}] must be a dict, but got {type(msg).__name__}: {msg}")
                
                content = msg.get("content", [])
                # Convert string to list if needed
                if isinstance(content, str):
                    content_stripped = content.strip()
                    if content_stripped.startswith('['):
                        # Looks like JSON array, try to parse
                        try:
                            content = json.loads(content)
                            msg["content"] = content  # Update the message dict
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Failed to parse content in message[{idx}] as JSON array: {e}")
                    else:
                        # Plain text string, convert to standard format
                        content = [{"type": "text", "text": content}]
                        msg["content"] = content  # Update the message dict
                
                # Ensure content is a list
                if not isinstance(content, list):
                    raise TypeError(f"content in message[{idx}] must be a list, but got {type(content).__name__}: {content}")
                
                for item_idx, item in enumerate(content):
                    # Convert string to dict if needed
                    if isinstance(item, str):
                        try:
                            item = json.loads(item)
                            content[item_idx] = item  # Update the content list
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Failed to parse content item[{item_idx}] in message[{idx}] as JSON string: {e}")
                    
                    # Ensure each item is a dict
                    if not isinstance(item, dict):
                        raise TypeError(f"content item[{item_idx}] in message[{idx}] must be a dict, but got {type(item).__name__}: {item}")
                    
                    if item.get("type") == "image":
                        image_num += 1
        elif "conversations" in json_data and json_data["conversations"]:
            conversations = json_data["conversations"]
            # Convert string to list/dict if needed
            if isinstance(conversations, str):
                try:
                    conversations = json.loads(conversations)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse conversations as JSON string: {e}")
            try:
                messages, image_num = convert_intern_format_to_verl_format(conversations,images,remove_think=False)
            except Exception as e:
                raise ValueError(f"Failed to convert conversations format: {type(e).__name__}: {e}")
        else:
            raise ValueError("No 'messages' or 'conversations' field found in JSON data")
        
        # === 8. Filter samples based on inference_correct_count ===
        # Skip samples where inference_correct_count is 0 (all wrong) or 4 (all correct)
        inference_correct_count = json_data.get("inference_correct_count",None)
        if inference_correct_count is not None:
            try:
                # Handle string or numeric tyspes
                count = int(inference_correct_count)
                if count == 0 or count==1 or count == 4:
                    # Skip this sample: all wrong (0) or all correct (4)
                    return None
            except (ValueError, TypeError):
                # If inference_correct_count is not a valid number, continue processing
                pass
        else:
            pass
        
        # === 4. Validate messages ===
        if not messages or len(messages) == 0:
            raise ValueError("messages is empty or None")
        
        # Ensure all messages are dicts (convert strings if needed)
        for idx, msg in enumerate(messages):
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                    messages[idx] = msg
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse message[{idx}] as JSON string: {e}")
            if not isinstance(msg, dict):
                raise TypeError(f"message[{idx}] must be a dict, but got {type(msg).__name__}: {msg}")
        
        # Check if there's any text content
        has_text = False
        for msg in messages:
            content = msg.get("content", [])
            # Convert string to list if needed
            if isinstance(content, str):
                content_stripped = content.strip()
                if content_stripped.startswith('['):
                    # Looks like JSON array, try to parse
                    try:
                        content = json.loads(content)
                        msg["content"] = content
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse content as JSON array: {e}")
                else:
                    # Plain text string, convert to standard format
                    content = [{"type": "text", "text": content}]
                    msg["content"] = content
            if not isinstance(content, list):
                raise TypeError(f"content must be a list, but got {type(content).__name__}: {content}")
            for item in content:
                # Convert string to dict if needed
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                        # Update the item in content list
                        item_idx = content.index(item) if item in content else None
                        if item_idx is not None:
                            content[item_idx] = item
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse content item as JSON string: {e}")
                if not isinstance(item, dict):
                    raise TypeError(f"content item must be a dict, but got {type(item).__name__}: {item}")
                if item.get("type") == "text" and item.get("text", "").strip():
                    has_text = True
                    break
            if has_text:
                break
        if not has_text:
            raise ValueError("No text content found in messages")

        # === 3.5. Add think system prompt for RL training ===
        # For RL training, add think system prompt to guide model to reason
        # and output in <think>...</think><answer>...</answer> format
        # BUT skip if metadata has box_idx (grounding tasks don't need think prompt)
        
        # Check if metadata has box_idx
        metadata = json_data.get("metadata")
        has_box_idx = False
        if isinstance(metadata, dict):
            has_box_idx = "box_idx" in metadata and metadata.get("box_idx") is not None
        elif isinstance(metadata, str):
            try:
                metadata_parsed = json.loads(metadata)
                if isinstance(metadata_parsed, dict):
                    has_box_idx = "box_idx" in metadata_parsed and metadata_parsed.get("box_idx") is not None
            except:
                pass
        
        # Only add think system prompt if metadata doesn't have box_idx
        # if not has_box_idx:
        think_system_prompt = (
            "Think step by step to solve the problem, And then give your final answer consistently with your reasoning."
            "Use <think>...</think> tags to show your reasoning process, "
            "then provide your final answer in <answer>...</answer> tags. "
            "Format: <think>your reasoning here</think><answer>your answer here</answer>"
        )
        # Use standard format: content should be a list with dict items
        messages.insert(0, {
            "role": "system",
            "content": [{"type": "text", "text": think_system_prompt}]
        })

        # === 5. Validate image count ===
        # Get all image keys from sample for debugging
        sample_image_keys = [k for k in sample.keys() if any(k.endswith(ext) for ext in extensions) and not k.startswith("__")]
        
        if image_num > 0 and len(images) == 0:
            # Include detailed error information about why images failed to load
            error_details = image_load_errors[:5] if image_load_errors else ["no_errors_logged"]
            raise ValueError(f"Images referenced but missing: image_num={image_num}, sample_keys={sample_image_keys}, loaded_images=0, errors={error_details}")
        if len(images) != image_num:
            error_details = image_load_errors[:5] if image_load_errors else []
            raise ValueError(f"Image count mismatch: images={len(images)}, image_num={image_num}, sample_keys={sample_image_keys}, errors={error_details}")




        # === 6. Extract view_type (same as TarDataset) ===
        view_type = "OpenData"
        metadata = json_data.get("metadata")
        # Convert string to dict if needed
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
                json_data["metadata"] = metadata  # Update json_data with converted value
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse metadata as JSON string: {e}")
        if isinstance(metadata, dict):
            if "question_tags" in metadata:
                question_tags = metadata["question_tags"]
                if isinstance(question_tags, list):
                    view_type = question_tags[0]
                else:
                    view_type = "Ours-unknown"
            elif "question_type" in metadata:
                view_type_value = metadata["question_type"]
                view_type = view_type_value if isinstance(view_type_value, str) else str(view_type_value)
                view_type = "Ours-" + view_type
            elif "view_type" in metadata:
                view_type_value = metadata["view_type"]
                view_type = view_type_value if isinstance(view_type_value, str) else str(view_type_value)
                view_type = "Ours-" + view_type
            else:
                view_type = "Ours-unknown"
        else:
            if default_view_type is not None:
                view_type = default_view_type
        
        if default_view_type in ('scannet', 'benchmark_tar_long', 'vsi'):
            has_view_type = True
        elif default_view_type == 'llava_ov':
            has_view_type = False
        else:
            has_view_type = False
        
        # === 7. Extract RL training fields ===
        # Extract ground_truth (may be a list or string)
        ground_truth_raw = json_data.get("ground_truth")
        ground_truth = None
        if ground_truth_raw is not None:
            # Handle list format: ["72"] -> "72"
            if isinstance(ground_truth_raw, list):
                if len(ground_truth_raw) > 0:
                    ground_truth = str(ground_truth_raw[0])  # Take first element
            else:
                ground_truth = str(ground_truth_raw)
        
        # Extract reward_model (may be dict or list)
        reward_model_raw = json_data.get("reward_model", {})
        # Convert string to dict/list if needed
        if isinstance(reward_model_raw, str):
            try:
                reward_model_raw = json.loads(reward_model_raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse reward_model as JSON string: {e}")
        reward_type = None
        
        if isinstance(reward_model_raw, list):
            # Format: ["accuracy", "format"] -> convert to reward_type string for routing
            reward_type_list = [str(r).strip().lower() for r in reward_model_raw if r]
            if reward_type_list:
                # Map "z" to "spatial_z" for proper routing
                reward_type_list_mapped = []
                for r in reward_type_list:
                    if r == "z":
                        reward_type_list_mapped.append("spatial_z")
                        # pass
                    elif r == "xy":
                        reward_type_list_mapped.append("spatial_xy")
                        # pass
                    elif r == "t":
                        reward_type_list_mapped.append("spatial_t")
                        # pass
                    else:
                        reward_type_list_mapped.append(r)
                reward_type = "+".join(reward_type_list_mapped)  # e.g., "accuracy+format" or "spatial_z"
            else:
                reward_type = self.default_data_source  # Fallback to default
            
            # Construct reward_model dict with ground_truth
            reward_model = {}
            if ground_truth:
                reward_model["ground_truth"] = ground_truth
                
        elif isinstance(reward_model_raw, dict):
            # Format: {"ground_truth": "72"} (existing format)
            reward_model = reward_model_raw.copy()
            # Ensure ground_truth is set
            if ground_truth and "ground_truth" not in reward_model:
                reward_model["ground_truth"] = ground_truth
            elif not reward_model.get("ground_truth") and ground_truth:
                reward_model["ground_truth"] = ground_truth
        else:
            # Fallback: construct from ground_truth
            reward_model = {}
            if ground_truth:
                reward_model["ground_truth"] = ground_truth
        
        # Extract data_source (dataset identifier, e.g., "openai/gsm8k")
        original_data_source = json_data.get("data_source", self.default_data_source)
        
        # Extract extra_info (optional, for additional reward configuration)
        extra_info = json_data.get("extra_info", {})
        # Convert string to dict if needed
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse extra_info as JSON string: {e}")
        if not isinstance(extra_info, dict):
            raise TypeError(f"extra_info must be a dict, but got {type(extra_info).__name__}: {extra_info}")
        
        # Extract inverse data for temporal cycle consistency (if present)
        # These fields are used by spatial_t reward for camera motion consistency
        if "inv_conversations" in json_data:
            extra_info["inv_conversations"] = json_data["inv_conversations"]
        if "inv_ground_truth" in json_data:
            extra_info["inv_ground_truth"] = json_data["inv_ground_truth"]
        if "temporal_metadata" in json_data:
            extra_info["temporal_metadata"] = json_data["temporal_metadata"]
        
        # If reward_model was a list, use reward_type for routing instead of data_source
        if reward_type:
            # Store original data_source in extra_info for reference
            extra_info["original_data_source"] = original_data_source
            # Use reward_type (e.g., "accuracy+format") for reward function routing
            data_source = reward_type
        else:
            # If reward_model is dict or not present, use original data_source
            data_source = original_data_source
            # Try to infer reward_type from data_source if it matches known patterns
            data_source_lower = str(data_source).lower()
            if data_source_lower in ["accuracy", "format", "accuracy+format", "format+accuracy"]:
                reward_type = data_source_lower
            elif "accuracy" in data_source_lower and "format" in data_source_lower:
                reward_type = "accuracy+format"
            else:
                # Default reward type
                reward_type = self.default_data_source
        
        # Store reward_type in extra_info for reward function
        extra_info["reward_type"] = reward_type
        
        # Skip spatial_t samples that lack required data (more efficient than computing r_dynamic_camera=0)
        # spatial_t needs: inv_ground_truth (for cycle consistency) + >=2 images (for inverse rollout)
        if reward_type and ("spatial_t" in str(reward_type).lower() or str(reward_type).lower() == "t"):
            inv_gt = extra_info.get("inv_ground_truth")
            has_inv_gt = inv_gt is not None and (
                (isinstance(inv_gt, list) and len(inv_gt) > 0)
                or (isinstance(inv_gt, str) and bool(inv_gt.strip()))
            )
            if not has_inv_gt or image_num < 2:
                self._last_decode_failure_reason = "spatial_t_incomplete"
                return None
        
        # Ensure ground_truth is in reward_model (required for RL training)
        if not reward_model or "ground_truth" not in reward_model:
            # Try to extract from assistant message as fallback
            if messages and messages[-1].get("role") == "assistant":
                assistant_content = messages[-1].get("content", [])
                for item in assistant_content:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        # Try to extract answer from <answer> tags
                        import re
                        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
                        if answer_match:
                            if not reward_model:
                                reward_model = {}
                            reward_model["ground_truth"] = answer_match.group(1).strip()
                            break
        
        # Add index if not present (for compatibility with RLHFDataset)
        if "index" not in extra_info:
            extra_info["index"] = 0
        
        # If spatial_xy reward is needed, load metadata information (pose, intrinsics, depth, etc.)
        if reward_type and ("spatial_xy" in reward_type.lower() or "xy" in reward_type.lower()):
            metadata = json_data.get("metadata", {})
            # Convert string to dict if needed (in case it wasn't converted earlier)
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                    json_data["metadata"] = metadata  # Update json_data
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse metadata as JSON string: {e}")
            if isinstance(metadata, dict):
                # Add correspondence_points from metadata
                if "correspondence_points" in metadata:
                    extra_info["correspondence_points"] = metadata["correspondence_points"]
                
                # Add original image sizes for coordinate normalization
                # correspondence_points coordinates are based on original image sizes
                # Store as (H, W) format to match spatial_xy.py expectations
                original_width = json_data.get("original_width", json_data.get("width", []))
                original_height = json_data.get("original_height", json_data.get("height", []))
                if isinstance(original_width, list) and len(original_width) >= 2:
                    if isinstance(original_height, list) and len(original_height) >= 2:
                        # (H, W) format: (height, width)
                        extra_info["original_img_size1"] = (int(original_height[0]), int(original_width[0]))
                        extra_info["original_img_size2"] = (int(original_height[1]), int(original_width[1]))
                elif isinstance(original_width, (int, float)) and isinstance(original_height, (int, float)):
                    # Single size for both images, (H, W) format
                    extra_info["original_img_size1"] = (int(original_height), int(original_width))
                    extra_info["original_img_size2"] = (int(original_height), int(original_width))
                
                # Load pose, intrinsic, depth data from paths
                try:
                    import numpy as np
                    import os
                    
                    pose_paths = metadata.get("pose_paths", [])
                    intrinsic_paths = metadata.get("intrinsic_paths", [])
                    depth_paths = metadata.get("depth_paths", [])
                    depth_scales = metadata.get("depth_scales", [1000, 1000])  # Default scale
                    
                    # Load pose matrices (4x4 Camera-to-World matrices)
                    if len(pose_paths) >= 2:
                        pose1 = self._load_pose_matrix(pose_paths[0])
                        pose2 = self._load_pose_matrix(pose_paths[1])
                        if pose1 is not None:
                            extra_info["pose1"] = pose1
                        if pose2 is not None:
                            extra_info["pose2"] = pose2
                    
                    # Load intrinsic matrices (3x3 or 4x4)
                    if len(intrinsic_paths) >= 2:
                        K1 = self._load_intrinsic_matrix(intrinsic_paths[0])
                        K2 = self._load_intrinsic_matrix(intrinsic_paths[1])
                        if K1 is not None:
                            extra_info["K1"] = K1
                        if K2 is not None:
                            extra_info["K2"] = K2
                    
                    # Load depth maps - resize to match RGB image sizes
                    if len(depth_paths) >= 2 and len(images) >= 2:
                        # Get RGB image sizes for resizing depth maps
                        # Use original image sizes (before processor resize)
                        # Processor will resize both RGB and depth maps consistently
                        img1_size = (images[0].width, images[0].height) if hasattr(images[0], 'width') else None
                        img2_size = (images[1].width, images[1].height) if hasattr(images[1], 'width') else None
                        
                        depth1 = self._load_depth_map(
                            depth_paths[0], 
                            depth_scales[0] if len(depth_scales) > 0 else 1000,
                            target_size=img1_size
                        )
                        depth2 = self._load_depth_map(
                            depth_paths[1], 
                            depth_scales[1] if len(depth_scales) > 1 else 1000,
                            target_size=img2_size
                        )
                        if depth1 is not None:
                            extra_info["depth1"] = depth1
                        if depth2 is not None:
                            extra_info["depth2"] = depth2
                    
                    # Add images to extra_info (for spatial_xy reward)
                    if len(images) >= 2:
                        # Convert PIL Images to numpy arrays if needed
                        if hasattr(images[0], 'numpy'):
                            extra_info["image1"] = images[0].numpy()
                        elif isinstance(images[0], np.ndarray):
                            extra_info["image1"] = images[0]
                        else:
                            # PIL Image, convert to numpy
                            extra_info["image1"] = np.array(images[0])
                        
                        if hasattr(images[1], 'numpy'):
                            extra_info["image2"] = images[1].numpy()
                        elif isinstance(images[1], np.ndarray):
                            extra_info["image2"] = images[1]
                        else:
                            # PIL Image, convert to numpy
                            extra_info["image2"] = np.array(images[1])
                    
                    # Extract ref_point from correspondence_points if available
                    # NOTE: ref_depth will be sampled in spatial_xy.py after all resizing is complete
                    # to ensure coordinates and depth are in the same space
                    if "correspondence_points" in extra_info and len(extra_info["correspondence_points"]) > 0:
                        ref_point_original = extra_info["correspondence_points"][0]
                        if isinstance(ref_point_original, (list, tuple)) and len(ref_point_original) >= 2:
                            # Store original ref_point coordinates (will be scaled in spatial_xy.py)
                            extra_info["ref_point"] = tuple(ref_point_original[:2])
                            # Do NOT sample ref_depth here - it will be sampled in spatial_xy.py
                            # after depth is resized to match the final image size
                
                except Exception as e:
                    # If loading fails, log warning but don't fail the entire sample
                    import warnings
                    warnings.warn(f"Failed to load spatial_xy metadata for sample: {e}")
        
        # If spatial_z reward is needed, load 3D grounding metadata
        if reward_type and ("spatial_z" in reward_type.lower() or reward_type.lower() == "z"):
            metadata = json_data.get("metadata", {})
            # Convert string to dict if needed (in case it wasn't converted earlier)
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                    json_data["metadata"] = metadata  # Update json_data
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse metadata as JSON string: {e}")
            
            if isinstance(metadata, dict):
                try:
                    import re
                    
                    # Extract box_idx (which boxes to use for reward calculation)
                    box_idx = metadata.get("box_idx", [])
                    
                    # Extract all 3D and 2D bounding boxes
                    boxes_3d_camera_coords = metadata.get("boxes_3d_camera_coords", [])
                    boxes_2d = metadata.get("boxes_2d", [])
                    
                    # Filter boxes based on box_idx (only keep the boxes we need)
                    if box_idx and boxes_3d_camera_coords:
                        # Use box_idx to select specific boxes
                        selected_boxes_3d = [boxes_3d_camera_coords[i] for i in box_idx if i < len(boxes_3d_camera_coords)]
                        extra_info["boxes_3d_camera_coords"] = selected_boxes_3d
                        
                        # Also filter 2D boxes if available
                        if boxes_2d:
                            selected_boxes_2d = [boxes_2d[i] for i in box_idx if i < len(boxes_2d)]
                            extra_info["boxes_2d"] = selected_boxes_2d
                    elif boxes_3d_camera_coords:
                        # If no box_idx provided, use all boxes
                        extra_info["boxes_3d_camera_coords"] = boxes_3d_camera_coords
                        if boxes_2d:
                            extra_info["boxes_2d"] = boxes_2d
                    
                    # Extract camera FOV parameters from question text
                    # The question contains: "Horizontal fov, hfov=57.95, and vertical fov, vfov=45.11"
                    hfov = metadata.get("hfov")
                    vfov = metadata.get("vfov")
                    
                    # If not in metadata, try to parse from question text in messages
                    if (hfov is None or vfov is None) and messages:
                        for msg in messages:
                            if msg.get("role") == "human" or msg.get("role") == "user":
                                content = msg.get("content", [])
                                for item in content:
                                    if item.get("type") == "text":
                                        text = item.get("text", "")
                                        # Parse: "hfov=57.95" and "vfov=45.11"
                                        hfov_match = re.search(r'hfov\s*[=:]\s*([\d.]+)', text, re.IGNORECASE)
                                        vfov_match = re.search(r'vfov\s*[=:]\s*([\d.]+)', text, re.IGNORECASE)
                                        if hfov_match and hfov is None:
                                            # Remove trailing punctuation like . , ;
                                            hfov_str = hfov_match.group(1).rstrip('.,;')
                                            hfov = float(hfov_str)
                                        if vfov_match and vfov is None:
                                            # Remove trailing punctuation like . , ;
                                            vfov_str = vfov_match.group(1).rstrip('.,;')
                                            vfov = float(vfov_str)
                                        if hfov is not None and vfov is not None:
                                            break
                            if hfov is not None and vfov is not None:
                                break
                    
                    # Add camera parameters to extra_info
                    if hfov is not None:
                        extra_info["hfov"] = float(hfov)
                    if vfov is not None:
                        extra_info["vfov"] = float(vfov)
                    
                    # Extract image dimensions from JSON root level
                    width = json_data.get("width") or json_data.get("original_width")
                    height = json_data.get("height") or json_data.get("original_height")
                    
                    if width is not None:
                        extra_info["width"] = int(width)
                    if height is not None:
                        extra_info["height"] = int(height)
                    
                    # Optionally load intrinsic matrix if path is provided
                    intrinsic_paths = metadata.get("intrinsic_paths")
                    if intrinsic_paths:
                        if isinstance(intrinsic_paths, str):
                            # Single path
                            intrinsic = self._load_intrinsic_matrix(intrinsic_paths)
                            if intrinsic is not None:
                                extra_info["intrinsic"] = intrinsic
                        elif isinstance(intrinsic_paths, list) and len(intrinsic_paths) > 0:
                            # List of paths, use the first one
                            intrinsic = self._load_intrinsic_matrix(intrinsic_paths[0])
                            if intrinsic is not None:
                                extra_info["intrinsic"] = intrinsic
                    
                    # Optionally load pose matrix if path is provided (Camera-to-World)
                    pose_paths = metadata.get("pose_paths")
                    if pose_paths:
                        if isinstance(pose_paths, str):
                            # Single path
                            pose = self._load_pose_matrix(pose_paths)
                            if pose is not None:
                                extra_info["pose"] = pose
                        elif isinstance(pose_paths, list) and len(pose_paths) > 0:
                            # List of paths, use the first one
                            pose = self._load_pose_matrix(pose_paths[0])
                            if pose is not None:
                                extra_info["pose"] = pose
                
                except Exception as e:
                    # If loading fails, log warning but don't fail the entire sample
                    import warnings
                    warnings.warn(f"Failed to load spatial_z metadata for sample: {e}")
        
        return {
            "images": images,
            "messages": messages,
            "has_view_type": has_view_type,
            "view_type": view_type,
            "reward_model": reward_model,  # For RL training
            "data_source": data_source,  # For RL training
            "extra_info": extra_info,  # For RL training
        }

    def __iter__(self):
        """Iterate over samples and process them for RL training."""
        import time

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[RLTarDataset] [PROGRESS] __iter__: Starting data iteration (rank={rank}, worker_id={worker_id})")
        
        processed_count = 0
        error_count = 0
        skipped_assistant_with_image = 0
        last_progress_time = time.time()
        PROGRESS_INTERVAL = 30  # Print progress every 30 seconds
        
        # Error tracking by category
        error_stats = {
            "decode_failed": 0,
            "no_ground_truth": 0,
            "no_messages": 0,
            "image_count_mismatch": 0,
            "image_processing_failed": 0,
            "no_assistant_message": 0,
            "no_assistant_tokens": 0,
            "truncation_removed_all": 0,
            "other": 0
        }

        rng = random.Random(self.base_seed + rank * 1000 + worker_id)

        print(f"[RLTarDataset] [PROGRESS] __iter__: Initializing dataset iterators...")
        dataset_iters = [iter(pipeline) for pipeline in self.dataset_pipelines]
        remaining_counts = list(self.target_counts)
        total_remaining = sum(remaining_counts)
        print(f"[RLTarDataset] [PROGRESS] __iter__: Dataset iterators initialized, total_remaining={total_remaining}")

        while True:
            if total_remaining == 0:
                remaining_counts = list(self.target_counts)
                total_remaining = sum(remaining_counts)

            available_indices = [i for i, count in enumerate(remaining_counts) if count > 0]
            if not available_indices:
                remaining_counts = list(self.target_counts)
                total_remaining = sum(remaining_counts)
                available_indices = [i for i, count in enumerate(remaining_counts) if count > 0]

            if len(available_indices) == 1:
                dataset_idx = available_indices[0]
            else:
                weights = [remaining_counts[i] for i in available_indices]
                dataset_idx = rng.choices(available_indices, weights=weights, k=1)[0]

            remaining_counts[dataset_idx] -= 1
            total_remaining -= 1

            iterator = dataset_iters[dataset_idx]
            sample = None
            try:
                sample = next(iterator)
            except StopIteration:
                dataset_iters[dataset_idx] = iter(self.dataset_pipelines[dataset_idx])
                iterator = dataset_iters[dataset_idx]
                try:
                    sample = next(iterator)
                except StopIteration:
                    continue

            if sample is None:
                continue

            sample_start_time = time.time()
            
            # Print progress periodically
            current_time = time.time()
            if current_time - last_progress_time >= PROGRESS_INTERVAL:
                print(f"[RLTarDataset] [PROGRESS] __iter__: Processed={processed_count}, Errors={error_count}, "
                      f"Skipped(assistant_image)={skipped_assistant_with_image}, "
                      f"Time elapsed={current_time - last_progress_time:.1f}s")
                if error_count > 0:
                    print(f"[RLTarDataset] [PROGRESS] Error breakdown: {error_stats}")
                last_progress_time = current_time

            # === Step 1: Decode sample with RL fields ===
            dataset_info = self.dataset_infos[dataset_idx]
            default_view_type = dataset_info.get("view_type")
            decode_failure_reason = None
            try:
                decoded = self._decode_sample_for_rl(sample, default_view_type=default_view_type)
                # Get failure reason from instance variable if decode failed
                if decoded is None:
                    decode_failure_reason = getattr(self, '_last_decode_failure_reason', 'unknown_decode_failure')
            except Exception as e:
                error_count += 1
                error_stats["decode_failed"] += 1
                decode_failure_reason = f"exception_{type(e).__name__}"
                if error_count <= 5:
                    sample_keys = [k for k in sample.keys() if not k.startswith("__")][:10]
                    print(f"[RLTarDataset] [ERROR] Sample decode exception: {type(e).__name__}: {str(e)[:200]}")
                    print(f"[RLTarDataset] [ERROR] Sample keys: {sample_keys}")
                # Track failure reasons
                if decode_failure_reason not in error_stats:
                    error_stats[decode_failure_reason] = 0
                error_stats[decode_failure_reason] += 1
                continue
            
            if decoded is None:
                error_count += 1
                error_stats["decode_failed"] += 1
                if error_count <= 10:  # Show more samples for debugging
                    sample_keys = [k for k in sample.keys() if not k.startswith("__")][:20]
                    print(f"[RLTarDataset] [ERROR] Sample decode returned None (error #{error_count})")
                    print(f"[RLTarDataset] [ERROR] Sample keys: {sample_keys}")
                    print(f"[RLTarDataset] [ERROR] Failure reason: {decode_failure_reason}")
                    # Show sample key patterns
                    json_like_keys = [k for k in sample_keys if "json" in k.lower() or k.endswith(".txt")]
                    if json_like_keys:
                        print(f"[RLTarDataset] [ERROR] Found JSON-like keys: {json_like_keys}")
                # Track failure reasons (simplify key names in stats)
                if decode_failure_reason:
                    # Extract main reason without full key list
                    if decode_failure_reason.startswith("no_json_key_found"):
                        reason_key = "no_json_key_found"
                    else:
                        reason_key = decode_failure_reason.split("_keys=")[0] if "_keys=" in decode_failure_reason else decode_failure_reason
                    if reason_key not in error_stats:
                        error_stats[reason_key] = 0
                    error_stats[reason_key] += 1
                continue

            images = decoded.get("images", [])
            has_view_type = decoded.get("has_view_type", False)
            view_type = decoded.get("view_type", "unknown")
            messages = decoded.get("messages", [])
            reward_model = decoded.get("reward_model", {})
            data_source = decoded.get("data_source", self.default_data_source)
            extra_info = decoded.get("extra_info", {})

            # === Step 2: Validate reward_model has ground_truth ===
            if not reward_model or "ground_truth" not in reward_model:
                # Try to extract from assistant message as fallback
                if messages and messages[-1].get("role") == "assistant":
                    assistant_content = messages[-1].get("content", [])
                    for item in assistant_content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            import re
                            answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
                            if answer_match:
                                reward_model = {"ground_truth": answer_match.group(1).strip()}
                                break
                
                # If still no ground_truth, skip this sample
                if not reward_model or "ground_truth" not in reward_model:
                    error_count += 1
                    error_stats["no_ground_truth"] += 1
                    if error_count <= 5:
                        print(f"[RLTarDataset] [ERROR] Skipping sample: No ground_truth found (error #{error_count})")
                        print(f"[RLTarDataset] [ERROR] Sample keys: {list(sample.keys())[:10] if isinstance(sample, dict) else 'N/A'}")
                        print(f"[RLTarDataset] [ERROR] Messages: {len(messages)} messages, last role: {messages[-1].get('role') if messages else 'N/A'}")
                    continue

            # === Step 3: Process messages (same as TarDataset) ===
            if self.multiturn_enabled:
                # Multi-turn mode
                if not messages:
                    error_count += 1
                    error_stats["no_messages"] += 1
                    if error_count <= 5:
                        print(f"[RLTarDataset] [ERROR] No messages in multiturn mode (error #{error_count})")
                    continue

                # Filter: Skip samples with images in assistant messages
                has_image_in_assistant = False
                for msg in messages:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", [])
                        for item in content:
                            if item.get("type") == "image":
                                has_image_in_assistant = True
                                break
                        if has_image_in_assistant:
                            break

                if has_image_in_assistant:
                    skipped_assistant_with_image += 1
                    if skipped_assistant_with_image <= 5:
                        print(f"⚠️ Skipping sample: assistant contains image (sample #{skipped_assistant_with_image})")
                    continue

                MAX_MESSAGES = 10
                if len(messages) > MAX_MESSAGES:
                    error_count += 1
                    continue
            else:
                # Single-turn mode: ensure we have user and assistant messages
                if len(messages) < 2:
                    error_count += 1
                    continue

            # === Step 4: Process with processor/tokenizer (same as TarDataset) ===
            if self.processor is not None:
                # Multimodal processing (same as TarDataset)
                from verl.utils.dataset.vision_utils import process_image

                expected_image_count = 0
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        for item in content:
                            if item.get("type") == "image":
                                expected_image_count += 1

                actual_image_count = len(images)
                if expected_image_count != actual_image_count:
                    error_count += 1
                    error_stats["image_count_mismatch"] += 1
                    if error_count <= 5:
                        print(f"[RLTarDataset] [ERROR] Image count mismatch: expected {expected_image_count}, got {actual_image_count} (error #{error_count})")
                    continue

                processed_images = None
                if images:
                    # Use fetch_image directly to resize images before convert("RGB")
                    # This avoids loading very large images into memory before resize
                    if processed_count == 0:
                        print(f"[RLTarDataset] [PROGRESS] __iter__: Processing first image sample (this may take a while for large images)...")
                    from qwen_vl_utils import fetch_image
                    processed_images = []
                    for img_idx, img in enumerate(images):
                        try:
                            # Create a dict with image and resize parameters
                            # fetch_image will resize the image before convert("RGB")
                            image_dict = {"image": img}
                            if self.min_pixels is not None:
                                image_dict["min_pixels"] = self.min_pixels
                            if self.max_pixels is not None:
                                image_dict["max_pixels"] = self.max_pixels
                            if processed_count == 0 and img_idx == 0:
                                print(f"[RLTarDataset] [PROGRESS] __iter__: Processing image {img_idx+1}/{len(images)} with min_pixels={self.min_pixels}, max_pixels={self.max_pixels}")
                            processed_img = fetch_image(image_dict, image_patch_size=self.image_patch_size)
                            processed_images.append(processed_img)
                        except Exception as e:
                            error_count += 1
                            error_stats["image_processing_failed"] += 1
                            if error_count <= 5:
                                print(f"[RLTarDataset] [ERROR] Image processing failed: {type(e).__name__}: {str(e)[:200]}")
                            break
                    
                    if len(processed_images) != len(images):
                        error_count += 1
                        error_stats["image_processing_failed"] += 1
                        if error_count <= 5:
                            print(f"[RLTarDataset] [ERROR] Image processing incomplete: {len(processed_images)}/{len(images)} processed (error #{error_count})")
                        continue
                    
                    if processed_count == 0:
                        print(f"[RLTarDataset] [PROGRESS] __iter__: Image processing completed for first sample")
                    
                    # ===== CRITICAL: Keep original images in extra_info (do NOT update with fetch_image processed images) =====
                    # spatial_xy.py will use original images and original coordinates directly, without any scaling
                    # Note: processed_images are used for model input, but extra_info keeps original images for reward computation
                    # This ensures coordinates, depths, and images are all in the same (original) coordinate space
                    if processed_count == 0 and ("image1" in extra_info or "image2" in extra_info):
                        print(f"[RLTarDataset] Keeping original images in extra_info for spatial_xy reward computation")
                        print(f"[RLTarDataset] Original image1 shape={extra_info.get('image1', 'None').shape if extra_info.get('image1') is not None else 'None'}, "
                              f"image2 shape={extra_info.get('image2', 'None').shape if extra_info.get('image2') is not None else 'None'}")
                    # ===== END CRITICAL UPDATE =====

                if processed_count == 0:
                    print(f"[RLTarDataset] [PROGRESS] __iter__: Applying chat template and processing with processor...")
                # CRITICAL: Only use prompt_messages (exclude assistant messages)
                # This ensures input_ids only contains prompt, not GPT response
                prompt_messages = [msg for msg in messages if msg.get("role") != "assistant"]
                
                # DEBUG: Print first sample's messages and config
                if processed_count == 0:
                    print(f"[RL_TAR_DEBUG] prompt_messages (roles only): {[m.get('role') for m in prompt_messages]}")
                    print(f"[RL_TAR_DEBUG] apply_chat_template_kwargs: {self.apply_chat_template_kwargs}")
                
                # Always add generation prompt since we're excluding assistant messages
                # CRITICAL: add_generation_prompt=True must come AFTER **kwargs to override any config value
                raw_prompt = self.processor.apply_chat_template(
                    prompt_messages, tokenize=False, **self.apply_chat_template_kwargs, add_generation_prompt=True
                )
                
                # DEBUG: Print generated prompt
                if processed_count == 0:
                    print(f"[RL_TAR_DEBUG] raw_prompt (last 200 chars): ...{raw_prompt[-200:]}")
                    # Tokenize to check the ending
                    prompt_tokens = self.processor.tokenizer.encode(raw_prompt)
                    print(f"[RL_TAR_DEBUG] raw_prompt token_ids (last 10): {prompt_tokens[-10:]}")

                processor_kwargs = {"text": [raw_prompt], "images": processed_images, "return_tensors": "pt"}
                if self.min_pixels is not None:
                    processor_kwargs["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    processor_kwargs["max_pixels"] = self.max_pixels

                if processed_count == 0:
                    print(f"[RLTarDataset] [PROGRESS] __iter__: Calling processor (this may take a while for first sample)...")
                model_inputs = self.processor(**processor_kwargs)
                if processed_count == 0:
                    print(f"[RLTarDataset] [PROGRESS] __iter__: Processor completed, sequence_length={model_inputs['input_ids'][0].shape[0]}")

                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]
                sequence_length = input_ids.shape[0]

                # Handle truncation (same as TarDataset)
                if sequence_length > self.max_length:
                    image_token_id = 151655
                    image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(image_token_positions) > 0:
                        last_image_pos = image_token_positions[-1].item()
                        safe_truncation_start = last_image_pos + 10
                        
                        if self.max_length < safe_truncation_start:
                            if error_count < 5:
                                print(f"⚠️ Skipping sample: Images take {safe_truncation_start} tokens, but max_length={self.max_length}")
                            error_count += 1
                            continue
                        
                        input_ids = input_ids[:self.max_length]
                        attention_mask = attention_mask[:self.max_length]
                        sequence_length = self.max_length
                    else:
                        input_ids = input_ids[:self.max_length]
                        attention_mask = attention_mask[:self.max_length]
                        sequence_length = self.max_length

                # Handle position_ids (same as TarDataset)
                if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                    if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                        from verl.models.transformers.qwen3_vl import get_rope_index
                    else:
                        from verl.models.transformers.qwen2_vl import get_rope_index

                    vision_position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids,
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask,
                    )

                    valid_mask = attention_mask.bool()
                    text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
                    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                    position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
                elif "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
                    from verl.models.transformers.glm4v import get_rope_index

                    vision_position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids,
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        attention_mask=attention_mask,
                    )

                    valid_mask = attention_mask.bool()
                    text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
                    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                    position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
                else:
                    from verl.utils.model import compute_position_id_with_mask
                    
                    if self.processor and "Qwen2VL" in self.processor.__class__.__name__:
                        text_position_ids_1d = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
                        text_position_ids = text_position_ids_1d.unsqueeze(0)
                        vision_position_ids = torch.zeros((3, len(input_ids)), dtype=torch.long)
                        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
                    else:
                        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

                # Create loss_mask: all zeros since we only have prompt (no response)
                # loss_mask is still required by protocol.py, but functionally unused
                loss_mask = torch.zeros_like(attention_mask)
                sequence_length = input_ids.shape[0]

                # Prepare output dict
                # CRITICAL: prompt_messages already extracted above (exclude assistant messages)
                # This is used for rollout to ensure correct format
                
                # CRITICAL: For multi-modal data, also save multi_modal_data (processed images)
                # This is needed for vLLM rollout to correctly handle images
                multi_modal_data = {}
                if processed_images:
                    # Convert processed_images to format expected by vLLM
                    # vLLM expects: {"image": [PIL.Image, ...]} or {"image": [np.ndarray, ...]}
                    multi_modal_data["image"] = processed_images
                
                output = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "loss_mask": loss_mask,
                    "has_view_type": has_view_type,
                    "view_type": view_type,
                    "reward_model": reward_model,  # RL field
                    "data_source": data_source,  # RL field
                    "extra_info": extra_info,  # RL field
                    "raw_prompt": prompt_messages,  # Original prompt messages for rollout
                    "multi_modal_data": multi_modal_data,  # Processed images for vLLM
                }

                if model_inputs:
                    output["multi_modal_inputs"] = dict(model_inputs)
                else:
                    output["multi_modal_inputs"] = {}

                processed_count += 1
                if processed_count == 1:
                    print(f"[RLTarDataset] [PROGRESS] __iter__: First sample processed successfully! (sequence_length={sequence_length})")
                if processed_count % 1000 == 0:
                    print(f"[RLTarDataset] [PROGRESS] __iter__: Processed: {processed_count}, Errors: {error_count}, Skipped (assistant has image): {skipped_assistant_with_image}")

                yield output

            else:
                # Text-only mode: only use prompt_messages (exclude assistant messages)
                # CRITICAL: Only use prompt_messages (exclude assistant messages)
                # This ensures input_ids only contains prompt, not GPT response
                prompt_messages = [msg for msg in messages if msg.get("role") != "assistant"]
                # Always add generation prompt since we're excluding assistant messages
                # CRITICAL: add_generation_prompt=True must come AFTER **kwargs to override any config value
                full_text = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, **self.apply_chat_template_kwargs, add_generation_prompt=True
                )

                encoded = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"][0]
                attention_mask = encoded["attention_mask"][0]
                sequence_length = input_ids.shape[0]

                # Create loss_mask: all zeros since we only have prompt (no response)
                # loss_mask is still required by protocol.py, but functionally unused
                loss_mask = torch.zeros_like(attention_mask)

                from verl.utils.model import compute_position_id_with_mask

                position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

                processed_count += 1

                # CRITICAL: prompt_messages already extracted above (exclude assistant messages)
                # This is used for rollout to ensure correct format
                
                # Text-only mode: no multi_modal_data
                multi_modal_data = {}

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "loss_mask": loss_mask,
                    "has_view_type": has_view_type,
                    "view_type": view_type,
                    "reward_model": reward_model,  # RL field
                    "data_source": data_source,  # RL field
                    "extra_info": extra_info,  # RL field
                    "raw_prompt": prompt_messages,  # Original prompt messages for rollout
                    "multi_modal_data": multi_modal_data,  # Empty for text-only
                }

    def get_collate_fn(self):
        """Return a collate function that pads to batch max length and handles RL fields.
        
        This is similar to TarDataset.get_collate_fn() but also handles RL-specific
        fields: reward_model, data_source, extra_info.
        """
        from functools import partial
        return partial(rl_tar_collate_fn, pad_token_id=self.tokenizer.pad_token_id)
    
    async def process_vision_info(
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        """Extract images and videos from messages.

        This method is called by AgentLoop (e.g SingleTurnAgentLoop) before apply_chat_template to
        the `raw_prompt` from dataset. User may customize RLHFDataset and override this method to
        support custom vision extraction.

        >>> messages = kwargs["raw_prompt"]
        >>> images, videos = RLHFDataset.process_vision_info(messages, image_patch_size)
        >>> videos, video_metadatas = zip(*videos)
        >>> raw_prompt = processor.apply_chat_template(messages, tokenize=False)
        >>> inputs = processor(text=[raw_prompt], images=images, videos=videos,
        ...                    video_metadata=video_metadatas, do_sample_frames=False)

        Args:
            messages: List of messages from dataset `raw_prompt`.
            image_patch_size: Image patch size for processor.
            config: Config for dataset.

        Returns:
            images: List of images.
            videos: List of videos, each video is a tuple of (video_tensor, video_metadata).
        """
        from qwen_vl_utils import process_vision_info
        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        return images, videos


def rl_tar_collate_fn(batch, pad_token_id=0):
    """Custom collate function for RLTarDataset that pads to batch max length and handles RL fields.
    
    This extends tar_collate_fn to also handle RL-specific fields:
    - reward_model: dict with ground_truth
    - data_source: str for reward function routing
    - extra_info: dict with additional reward configuration
    - prompts: pure prompt token ids for rollout
    
    Args:
        batch: list of dicts, each containing input_ids, attention_mask, reward_model, etc.
        pad_token_id: token id for padding
    
    Returns:
        dict with batched and padded tensors, plus RL fields as numpy arrays
    """
    import numpy as np
    
    # Use tar_collate_fn for standard fields (input_ids, attention_mask, position_ids, loss_mask, etc.)
    # Note: No need to extract prompts here - rollout will do it on-the-fly from input_ids + loss_mask
    # This reduces memory overhead by avoiding duplicate storage
    result = tar_collate_fn(batch, pad_token_id=pad_token_id)
    
    # === Collect RL-specific fields (non-tensor fields) ===
    batch_reward_model = []
    batch_data_source = []
    batch_extra_info = []
    batch_raw_prompt = []
    batch_multi_modal_data = []
    
    for sample in batch:
        batch_reward_model.append(sample.get("reward_model", {}))
        batch_data_source.append(sample.get("data_source", ""))
        batch_extra_info.append(sample.get("extra_info", {}))
        batch_raw_prompt.append(sample.get("raw_prompt", []))
        batch_multi_modal_data.append(sample.get("multi_modal_data", {}))
    
    # Convert to numpy arrays of dtype object (similar to rl_dataset.collate_fn)
    result["reward_model"] = np.fromiter(batch_reward_model, dtype=object, count=len(batch_reward_model))
    result["data_source"] = np.fromiter(batch_data_source, dtype=object, count=len(batch_data_source))
    result["extra_info"] = np.fromiter(batch_extra_info, dtype=object, count=len(batch_extra_info))
    result["raw_prompt"] = np.fromiter(batch_raw_prompt, dtype=object, count=len(batch_raw_prompt))
    result["multi_modal_data"] = np.fromiter(batch_multi_modal_data, dtype=object, count=len(batch_multi_modal_data))
    
    # Convert multi_modal_inputs from list to numpy array if present
    # tar_collate_fn returns multi_modal_inputs as a list, but DataProto.from_single_dict requires np.ndarray
    if "multi_modal_inputs" in result and isinstance(result["multi_modal_inputs"], list):
        result["multi_modal_inputs"] = np.array(result["multi_modal_inputs"], dtype=object)
    
    # Convert view_type from list to numpy array if present
    if "view_type" in result and isinstance(result["view_type"], list):
        result["view_type"] = np.array(result["view_type"], dtype=object)
    
    # Fix position_ids shape for DataProto.from_single_dict compatibility
    # tar_collate_fn returns position_ids as [4, batch, seq_len] for Qwen2-VL models
    # DataProto.from_single_dict requires all tensors to have batch_size as first dimension
    # So we need to permute [4, batch, seq_len] → [batch, 4, seq_len]
    if "position_ids" in result and isinstance(result["position_ids"], torch.Tensor):
        pos_ids = result["position_ids"]
        if pos_ids.dim() == 3 and pos_ids.shape[0] == 4:
            # Qwen2-VL format: [4, batch, seq_len] → permute to [batch, 4, seq_len]
            # This is required for DataProto.from_single_dict which checks batch_size consistency
            result["position_ids"] = pos_ids.permute(1, 0, 2).contiguous()  # [batch, 4, seq_len]
    
    return result

