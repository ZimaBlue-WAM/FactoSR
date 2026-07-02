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
import json
import math
import os
import random
import re
from io import BytesIO

# from webdataset.handlers import ignore_and_continue
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils.data
import webdataset as wds
from PIL import Image

from omegaconf import DictConfig, OmegaConf

# Try to import s3fs for S3 support
try:
    import s3fs

    S3FS_AVAILABLE = True
except ImportError:
    S3FS_AVAILABLE = False
    print("Warning: s3fs not available. Install with: pip install s3fs")


def get_tar_urls_under_dir(data_dir):
    """Recursively get all tar files under a directory

    Supports local paths, HDFS paths, and S3 paths
    """
    import subprocess

    # Check if it's an S3 path
    if data_dir.startswith("s3://"):
        print(f"Listing S3 directory: {data_dir}")

        # Method 1: Try using s3fs (Python library, more reliable)
        if S3FS_AVAILABLE:
            try:
                # Create s3fs filesystem
                # Credentials can come from environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
                # or from ~/.aws/credentials, or from IAM role
                fs = s3fs.S3FileSystem(anon=False)  # anon=False means use credentials

                # Parse S3 URL: s3://bucket/path -> bucket, path
                s3_path = data_dir[5:]  # Remove 's3://'
                if "/" in s3_path:
                    bucket, prefix = s3_path.split("/", 1)
                    prefix = prefix.rstrip("/")  # Remove trailing slash
                else:
                    bucket = s3_path
                    prefix = ""

                # List all .tar files recursively
                # s3fs glob pattern: bucket/prefix/**/*.tar
                if prefix:
                    pattern = f"{bucket}/{prefix}/**/*.tar"
                else:
                    pattern = f"{bucket}/**/*.tar"

                tar_paths = fs.glob(pattern)

                # Convert to s3:// URLs
                tar_urls = [f"s3://{path}" for path in tar_paths]
                tar_urls = sorted(tar_urls)

                print(f"Found {len(tar_urls)} tar files using s3fs")
                if len(tar_urls) > 0:
                    print(f"  Example: {tar_urls[0]}")
                return tar_urls
            except Exception as e:
                print(f"s3fs failed: {e}")
                import traceback

                traceback.print_exc()
                print("Falling back to rclone...")

        # Method 2: Try using rclone (most versatile, supports many backends)
        try:
            # rclone ls lists files recursively
            # Format: "size path/to/file.tar"
            cmd = f"rclone ls {data_dir} --include '*.tar'"
            print(f"Trying rclone: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                tar_files = []
                for line in result.stdout.strip().split("\n"):
                    parts = line.split(maxsplit=1)  # Split into size and path
                    if len(parts) == 2 and parts[1].endswith(".tar"):
                        # parts[1] is the relative path
                        tar_files.append(parts[1])

                # Reconstruct full S3 URLs
                tar_urls = [f"{data_dir.rstrip('/')}/{f}" for f in tar_files]
                tar_urls = sorted(tar_urls)
                print(f"Found {len(tar_urls)} tar files using rclone")
                return tar_urls
            else:
                print(f"rclone not available or failed: returncode={result.returncode}")
                if result.stderr:
                    print(f"stderr: {result.stderr[:300]}")
        except Exception as e:
            print(f"Failed to use rclone: {e}")

        # All S3 methods failed
        print("ERROR: Failed to list S3 directory!")
        print("Please ensure rclone is configured: rclone config")
        return []

    # Check if it's an HDFS path
    elif data_dir.startswith("hdfs://"):
        try:
            cmd = f"hdfs dfs -ls -R {data_dir} | grep '\\.tar$' | awk '{{print $8}}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            tar_urls = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            tar_urls = sorted(tar_urls)
            return tar_urls
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Failed to list HDFS directory {data_dir}: {e}")
            return []
    else:
        # Local path: use pathlib
        print(f"Searching local directory: {data_dir}")
        tar_urls = Path(data_dir).rglob("*.tar")
        tar_urls = list(tar_urls)
        tar_urls = [str(tar_url) for tar_url in tar_urls]
        tar_urls = sorted(tar_urls)
        print(f"Found {len(tar_urls)} tar files in local directory")
        return tar_urls


def has_redacted_reasoning(messages):
    """
    Check if messages contain <think> tags.
    
    Args:
        messages: List of message dicts in HF format
        
    Returns:
        True if any message contains <think> tags, False otherwise
    """
    if not messages or not isinstance(messages, list):
        return False
    
    pattern = r'<think>.*?</think>'
    
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            # If content is a string, check directly
            if re.search(pattern, content, flags=re.DOTALL):
                return True
        elif isinstance(content, list):
            # If content is a list, check text items
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text and re.search(pattern, text, flags=re.DOTALL):
                        return True
    
    return False


def remove_redacted_reasoning(text, remove_think=True):
    """
    Remove <think>...</think> tags and their content from text.
    
    Args:
        text: Input text string that may contain <think> tags
        remove_think: If True, remove <think> tags; if False, keep them
        
    Returns:
        Cleaned text without redacted_reasoning tags and content (if remove_think=True)
    """
    if not text or not isinstance(text, str):
        return text
    
    if not remove_think:
        return text
    
    # Use regex to match and remove <think>...</think> tags and their content
    # Note: using non-greedy match (.*?) and DOTALL flag to handle multi-line content
    # Match <think>...</think> tags and remove them (user requested)
    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return cleaned_text.strip()


def clean_assistant_messages(messages, remove_think=True):
    """
    Clean redacted_reasoning tags from assistant messages in HF format.
    
    Args:
        messages: List of message dicts in HF format
        remove_think: If True, remove <think> tags; if False, keep them
        
    Returns:
        Cleaned messages with redacted_reasoning removed from assistant content (if remove_think=True)
    """
    if not remove_think:
        return messages
    
    cleaned_messages = []
    for msg in messages:
        if msg.get("role") == "assistant":
            # Clean assistant messages
            content = msg.get("content", [])
            cleaned_content = []
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        # Remove redacted_reasoning tags from text
                        cleaned_text = remove_redacted_reasoning(text, remove_think=remove_think)
                        cleaned_content.append({
                            "index": item.get("index"),
                            "text": cleaned_text,
                            "type": item.get("type")
                        })
                else:
                    # Keep non-text items as is (e.g., images)
                    cleaned_content.append(item)
            cleaned_messages.append({
                "role": msg.get("role"),
                "content": cleaned_content
            })
        else:
            # Keep user messages unchanged
            cleaned_messages.append(msg)
    
    return cleaned_messages


def parse_utterance_content(value, remove_think=True):
    """
    Convert intern format conversation value into hf format content list
    
    Args:
        value: Input conversation value string
        remove_think: If True, remove <think> tags; if False, keep them
    """

    # FIXED_PROMPT = (
    # "Think step by step to solve the problem. "
    # "Use <think>...</think> tags to show your reasoning process, "
    # "then provide your final answer in <answer>...</answer> tags. "
    # "Format: <think>your reasoning here</think>"
    # "<answer>your answer here</answer>\n"
    # )


    content = []
    image_counter = 0

    parts = re.split(r"(<image>)", value)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part == "<image>":
            content.append({"index": image_counter, "text": None, "type": "image"})
            image_counter += 1
        else:
            # Clean text: remove redacted_reasoning tags before adding to content
            cleaned_text = remove_redacted_reasoning(part, remove_think=remove_think)
            content.append({"index": None, "text": cleaned_text, "type": "text"})

    return content, image_counter


def convert_intern_format_to_verl_format(conversations: list[dict], images, remove_think=True):
    """
    e.g., a InternVL format conversation [
        {'from': 'human', 'value': '<image>Describe the image.'},
        {'from': 'gpt', 'value': 'This image is NSFW.'}
    ]
    will be converted to [
        {
            "role": "user",
            "content": [
                {"index": 0, "text": None, "Image": image, "type": "image"},
                {"index": None, "text": "Describe the image.", "type": "text"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"index": None, "text": "This image is NSFW.", "type": "text"}
            ]
        }
    ]
    
    Args:
        conversations: List of conversation dicts in Intern format
        images: List of Imgs
        remove_think: If True, remove <think> tags; if False, keep them
    """

    res = []
    image_counter = 0
    img_index=0
    # print(conversations[:2])
    for utterance in conversations:
        # print(utterance)
        if utterance["from"] == "human":
            role = "user"
        elif utterance["from"] == "gpt":
            role = "assistant"
        content, image_num = parse_utterance_content(utterance["value"], remove_think=remove_think)
        for index in range(len(content)):
            if content[index].get("type",'text')=='image':
                content[index]['image']=images[img_index]
                img_index+=1
        image_counter += image_num
        res.append({"role": role, "content": content})
    return res, image_counter

def convert_intern_format_to_hf_format(conversations: list[dict], remove_think=True):
    """
    e.g., a InternVL format conversation [
        {'from': 'human', 'value': '<image>Describe the image.'},
        {'from': 'gpt', 'value': 'This image is NSFW.'}
    ]
    will be converted to [
        {
            "role": "user",
            "content": [
                {"index": 0, "text": None, "type": "image"},
                {"index": None, "text": "Describe the image.", "type": "text"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"index": None, "text": "This image is NSFW.", "type": "text"}
            ]
        }
    ]
    
    Args:
        conversations: List of conversation dicts in Intern format
        remove_think: If True, remove <think> tags; if False, keep them
    """
    res = []
    image_counter = 0
    # print(conversations[:2])
    for utterance in conversations:
        # print(utterance)
        if utterance["from"] == "human":
            role = "user"
        elif utterance["from"] == "gpt":
            role = "assistant"
        content, image_num = parse_utterance_content(utterance["value"], remove_think=remove_think)
        image_counter += image_num
        res.append({"role": role, "content": content})
    return res, image_counter

def create_webdataset(
    urls: list[str],
    # num_samples_per_epoch: int = 100_000,
    base_seed: int = 42,
    extensions=("jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG", "WEBP"),
    default_view_type: str = None,
    remove_think: bool = True,
):
    """
    Create a WebDataset supporting multi-image + json structure.
    This implementation does not maintain a dataset cache.
    It randomly samples `num_samples_per_epoch` data samples from the whole dataset every epoch.

    Args:
        urls: List of tar file URLs
        base_seed: Base seed for random sampling
        extensions: Image file extensions to look for
        default_view_type: Default view type for samples
        remove_think: If True, remove <think>...</think> tags from assistant messages; 
                     if False, keep them. Default is True.

    Note: For S3 URLs (s3://...), WebDataset uses fsspec which automatically uses s3fs
    if available. Make sure s3fs is installed: pip install s3fs
    Credentials can be provided via:
    - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    - ~/.aws/credentials file
    - IAM role (if running on EC2)
    """

    rank = dist.get_rank() if dist.is_initialized() else 0
    # world_size = dist.get_world_size() if dist.is_initialized() else 1

    seed = base_seed + rank
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # rng = random.Random(seed)
    # rng.shuffle(urls)

    def decode_sample(sample):
        """
        Decode and validate sample from tar file

        sample: dict, e.g. {
            "__key__": "sample1",
            "0.jpg": bytes,
            "1.jpg": bytes,
            "json": bytes,
        }

        Returns:
            {"images": [PIL.Image, ...], "messages": [...]} or None if invalid
        """
        # === 1. Decode images ===
        images = []
        corrupted_images = 0
        for k, v in sample.items():
            if any(k.endswith(ext) for ext in extensions):
                try:
                    img = Image.open(BytesIO(v)).convert("RGB")
                    # Validate image has reasonable size
                    if img.width > 0 and img.height > 0:
                        image_ratio = img.width / img.height
                        if image_ratio > 200 or image_ratio < 0.005:
                            return None
                        images.append(img)
                    else:
                        corrupted_images += 1
                except Exception:
                    corrupted_images += 1
                    continue
        # === 2. Decode json ===
        json_data = None
        for k, v in sample.items():
            if k.endswith("json"):
                try:
                    json_data = json.loads(v.decode("utf-8"))
                except Exception:
                    # print(f"Failed to parse json: {e}")
                    return None
                break
        # === 3. Check if data is complete ===
        if json_data is None:
            # print('No json data found.')
            return None

        # === 4. Convert to HF format ===
        # Support both "conversations" (Intern format) and "messages" (HF format) fields
        messages = None
        image_num = 0
        
        if "messages" in json_data and json_data["messages"]:
            # Already in HF format, use directly
            messages = json_data["messages"]
            # Clean redacted_reasoning tags from assistant messages
            messages = clean_assistant_messages(messages, remove_think=remove_think)
            # Count images in messages
            for msg in messages:
                content = msg.get("content", [])
                for item in content:
                    if item.get("type") == "image":
                        image_num += 1
        elif "conversations" in json_data and json_data["conversations"]:
            # Intern format, convert to HF format
            try:
                messages, image_num = convert_intern_format_to_hf_format(json_data["conversations"], remove_think=remove_think)
                # Clean redacted_reasoning tags from assistant messages (already cleaned in parse_utterance_content, but double-check)
                messages = clean_assistant_messages(messages, remove_think=remove_think)
            except Exception:
                # print(f"Failed to convert format: {e}")
                return None
        else:
            # Neither format found
            # print('No conversations or messages in json.')
            return None
        # === 5. Validate messages have content ===
        if not messages or len(messages) == 0:
            # print('Empty messages after conversion.')
            return None
        # Check if there's any text content
        has_text = False
        for msg in messages:
            content = msg.get("content", [])
            for item in content:
                if item.get("type") == "text" and item.get("text", "").strip():
                    has_text = True
                    break
            if has_text:
                break
        if not has_text:
            # print('No text content in messages.')
            return None
        
        # === 5.5. Add think system prompt if needed ===
        # When remove_think=False and data contains <think> tags, add think system prompt
        if not remove_think:
            # Check if messages contain <think> tags
            if has_redacted_reasoning(messages):
                # Add system message with think prompt at the beginning
                # Check if there's already a system message
                
                # Add think system prompt
                
                think_system_prompt = (
                    "Think step by step to solve the problem. "
                    "Use <think>...</think> tags to show your reasoning process, "
                    "then provide your final answer in <answer>...</answer> tags. "
                    "Format: <think>your reasoning here</think><answer>your answer here</answer>"
                )
                # Use standard format: content should be a list with dict items
                messages.insert(0, {
                    "role": "system",
                    "content": [{"type": "text", "text": think_system_prompt}]
                })

        # === 6. Validate image count ===
        # If images are expected but missing
        if image_num > 0 and len(images) == 0:
            # print(f'Expected {image_num} images but got 0.')
            return None

        # Check if the <image> token num in text equals actual image num
        if len(images) != image_num:
            # print(f'Image count mismatch: expected {image_num}, got {len(images)}.')
            return None

        # === 7. Filter samples with too many images (prevent extreme cases) ===
        # More images → more visual tokens → higher memory
        # if len(images) > 3:
        #     # print(f'Too many images: {len(images)}, skipping for stability.')
        #     return None
        
        # === 8. Check if view_type field exists ===
        # If view_type exists, mark this sample to skip KL loss
        # if "metadata" in json_data:
        #     has_view_type = "view_type" in json_data["metadata"]
        # else:
        view_type = "OpenData"
        metadata = json_data.get("metadata")
        # to do: question_tags
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
        # NOTE: keep has_view_type False to ensure KL loss is enabled for all samples later
        
        # if default_view_type in ('scannet', 'benchmark_tar_long', 'vsi'):
        #     has_view_type = True
        # elif default_view_type == 'llava_ov':
        #     has_view_type = False
        # else:
        #     has_view_type = False
        if default_view_type == 'llava_ov':
            has_view_type = False
        else:
            has_view_type = True
        # === 9. All checks passed ===
        return {"images": images, "messages": messages, "has_view_type": has_view_type, "view_type": view_type}

    # Create WebDataset pipeline with proper shuffling
    # Using ResampledShards for true cross-epoch randomness
    dataset = wds.DataPipeline(
        # wds.SimpleShardList(urls),
        wds.ResampledShards(urls),
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
        wds.map(decode_sample),
        wds.select(lambda x: x is not None),
        # wds.with_epoch(num_samples_per_epoch)
        # wds.batched(1)
    )

    return dataset


def tar_collate_fn(batch, pad_token_id=0):
    """Custom collate function for TarDataset that pads to batch max length

    Args:
        batch: list of dicts, each containing input_ids, attention_mask, etc.
        pad_token_id: token id for padding

    Returns:
        dict with batched and padded tensors

    IMPORTANT: For VL models, padding input_ids will NOT affect image features
    because image_grid_thw is stored separately in multi_modal_inputs and won't be modified.
    The model will use image_grid_thw to determine how many image features to expect.
    """
    # Find max length in this batch
    max_len_in_batch = max(sample["input_ids"].shape[0] for sample in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_position_ids = []
    batch_loss_mask = []
    batch_multi_modal_inputs = []
    batch_has_view_type = []  # Collect has_view_type flags
    batch_view_type = []  # Collect view_type values

    for sample in batch:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        position_ids = sample["position_ids"]
        loss_mask = sample["loss_mask"]
        has_view_type = sample.get("has_view_type", False)  # Default to False if not present
        view_type = sample.get("view_type", "Ours-unknown")

        seq_len = input_ids.shape[0]

        # Pad to max_len_in_batch
        if seq_len < max_len_in_batch:
            pad_len = max_len_in_batch - seq_len

            # Pad input_ids
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])

            # Pad attention_mask
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])

            # Pad loss_mask
            loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype)])

            # Pad position_ids
            if position_ids.dim() == 1:
                # Standard position_ids: [seq_len]
                position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=position_ids.dtype)])
            elif position_ids.dim() == 2:
                # Multi-dimensional position_ids (e.g., Qwen2-VL): [4, seq_len]
                # Pad the last dimension (seq_len)
                position_ids = torch.cat(
                    [position_ids, torch.zeros((position_ids.shape[0], pad_len), dtype=position_ids.dtype)], dim=-1
                )
            else:
                # Unexpected shape, just keep as is
                pass

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_position_ids.append(position_ids)
        batch_loss_mask.append(loss_mask)
        batch_has_view_type.append(has_view_type)  # Collect view_type flag
        batch_view_type.append(view_type)

        # Collect multi_modal_inputs (MUST add for every sample to maintain batch consistency)
        # Even if sample has no images, add empty dict as placeholder
        if "multi_modal_inputs" in sample:
            batch_multi_modal_inputs.append(sample["multi_modal_inputs"])
        else:
            # Text-only sample, add empty dict to maintain list length
            batch_multi_modal_inputs.append({})

    # Stack into batch
    # Check if all position_ids have the same shape
    first_pos_shape = batch_position_ids[0].shape
    all_same_shape = all(pos.shape == first_pos_shape for pos in batch_position_ids)

    # Debug: print first sample's position_ids shape
    if len(batch_position_ids) > 0:
        first_pos_dim = batch_position_ids[0].dim()

    if all_same_shape:
        # All position_ids have same shape, can stack directly
        if first_pos_dim == 1:
            # Standard case: [seq_len] → stack to [batch, seq_len]
            stacked_position_ids = torch.stack(batch_position_ids, dim=0)
        elif first_pos_dim == 2:
            # Qwen2-VL case: [4, seq_len] (text + vision position_ids)
            # Expected output: [4, batch, seq_len]
            # Stack along dim=1 to insert batch dimension
            stacked_position_ids = torch.stack(batch_position_ids, dim=1)  # [4, batch, seq_len]
            # DO NOT permute! Qwen2-VL expects [4, batch, seq_len] format
        else:
            # Unexpected dimension
            print(f"Warning: Unexpected position_ids dimension: {first_pos_dim}, shape: {first_pos_shape}")
            stacked_position_ids = torch.stack(batch_position_ids, dim=0)
    else:
        # Mixed shapes (shouldn't happen, but handle it)
        print(f"Warning: Mixed position_ids shapes in batch: {[pos.shape for pos in batch_position_ids]}")
        # Convert all to same shape (use the most common one or first one)
        target_dim = first_pos_shape[0] if len(first_pos_shape) > 1 else None
        normalized_position_ids = []
        for pos in batch_position_ids:
            if pos.dim() == 1 and target_dim is not None:
                # Expand 1D to 2D by repeating
                pos = pos.unsqueeze(0).repeat(target_dim, 1)
            normalized_position_ids.append(pos)
        
        # Check if normalized position_ids are 2D [4, seq_len] format
        if normalized_position_ids[0].dim() == 2 and normalized_position_ids[0].shape[0] == 4:
            # Qwen2-VL format: stack along dim=1 to get [4, batch, seq_len]
            stacked_position_ids = torch.stack(normalized_position_ids, dim=1)  # [4, batch, seq_len]
        else:
            # Fallback: standard stack
            stacked_position_ids = torch.stack(normalized_position_ids, dim=0)

    result = {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "attention_mask": torch.stack(batch_attention_mask, dim=0),
        "position_ids": stacked_position_ids,
        "loss_mask": torch.stack(batch_loss_mask, dim=0),
    }

    # Debug: print shapes occasionally (every 100 batches)
    # import random
    # if random.random() < 0.01:  # 1% chance to print
    #     print(f"[Collate Debug] Batch shapes - input_ids: {result['input_ids'].shape}, "
    #           f"position_ids: {result['position_ids'].shape}, "
    #           f"attention_mask: {result['attention_mask'].shape}")

    # Handle multi_modal_inputs (don't stack, just pass as list)
    if batch_multi_modal_inputs:
        result["multi_modal_inputs"] = batch_multi_modal_inputs
    
    # Add has_view_type as tensor for sample-level KL loss control
    if batch_has_view_type:
        result["has_view_type"] = torch.tensor(batch_has_view_type, dtype=torch.bool)
    
    # Always add view_type to result (even if empty list, for consistency)
    result["view_type"] = batch_view_type if batch_view_type else []

    # Debug: Verify multi_modal_inputs matches batch size
    batch_size = result["input_ids"].shape[0]
    mm_count = len(batch_multi_modal_inputs)
    if batch_size != mm_count:
        print("\n❌ CRITICAL ERROR: Batch size mismatch!")
        print(f"   input_ids batch size: {batch_size}")
        print(f"   multi_modal_inputs count: {mm_count}")
        print("   This WILL cause features/tokens mismatch!")
        # This should never happen after our fix

    return result


def packed_collate_fn(batch, pad_token_id=0, max_length=2048, image_token_id=151655):
    """Sequence packing collate function that packs short sequences together
    
    This significantly reduces padding waste by packing multiple short sequences
    into one, up to max_length. Each sub-sequence maintains its own attention
    boundaries (block-diagonal attention mask).
    
    Args:
        batch: list of dicts, each containing input_ids, attention_mask, etc.
        pad_token_id: token id for padding
        max_length: maximum sequence length
        image_token_id: token id for image tokens (to avoid splitting images)
    
    Returns:
        dict with packed and batched tensors, including cu_seqlens for Flash Attention
    """
    # 1. Prepare samples with metadata
    samples = []
    for sample in batch:
        seq_len = sample["input_ids"].shape[0]
        
        # Check if sample has images (VL model)
        has_images = False
        multi_modal_inputs = sample.get("multi_modal_inputs", {})
        if multi_modal_inputs and "pixel_values" in multi_modal_inputs:
            has_images = True
            # Count image tokens
            n_image_tokens = (sample["input_ids"] == image_token_id).sum().item()
        else:
            n_image_tokens = 0
        
        samples.append({
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "position_ids": sample["position_ids"],
            "loss_mask": sample["loss_mask"],
            "has_view_type": sample.get("has_view_type", False),
            "view_type": sample.get("view_type", "Ours-unknown"),
            "multi_modal_inputs": multi_modal_inputs,
            "seq_len": seq_len,
            "has_images": has_images,
            "n_image_tokens": n_image_tokens,
        })
    
    # 2. Sort by sequence length (short first for better packing)
    samples.sort(key=lambda x: x["seq_len"])
    
    # 3. Greedy packing algorithm
    packed_sequences = []
    current_pack = []
    current_length = 0
    current_has_images = False
    
    for sample in samples:
        can_pack = True
        
        # Rule 1: Don't exceed max_length
        if current_length + sample["seq_len"] > max_length:
            can_pack = False
        
        # Rule 2: Limit total number of images in a pack (avoid GPU OOM)
        current_n_images = sum(1 for s in current_pack if s["has_images"])
        max_images_per_pack = 3  # Allow up to 3 images per pack (tunable)
        if sample["has_images"] and current_n_images >= max_images_per_pack:
            can_pack = False
        
        # Rule 3: Limit total image tokens in a pack (more precise memory control)
        current_image_tokens = sum(s["n_image_tokens"] for s in current_pack)
        max_image_tokens_per_pack = 10000  # Tunable based on GPU memory
        if current_image_tokens + sample["n_image_tokens"] > max_image_tokens_per_pack:
            can_pack = False
        
        if can_pack:
            # Pack into current pack
            current_pack.append(sample)
            current_length += sample["seq_len"]
            if sample["has_images"]:
                current_has_images = True
        else:
            # Current pack is full, start new pack
            if current_pack:
                packed_sequences.append(current_pack)
            current_pack = [sample]
            current_length = sample["seq_len"]
            current_has_images = sample["has_images"]
    
    # Add last pack
    if current_pack:
        packed_sequences.append(current_pack)

    if dist.get_rank() == 0:
        # Debug: 打印打包统计
        total_samples = len(batch)
        total_packs = len(packed_sequences)
        samples_per_pack = [len(pack) for pack in packed_sequences]
    # 4. Build packed batch
    batch_input_ids = []
    batch_attention_mask = []
    batch_position_ids = []
    batch_loss_mask = []
    batch_has_view_type = []
    batch_view_type = []
    batch_multi_modal_inputs = []
    batch_cu_seqlens = []  # Cumulative sequence lengths for Flash Attention
    batch_max_seqlen = []  # Max sequence length in each pack
    
    for pack_idx, pack in enumerate(packed_sequences):
        # Concatenate all samples in this pack
        pack_input_ids = []
        pack_loss_mask = []
        pack_position_ids = []
        cu_seqlens = [0]
        current_pos = 0
        pack_has_view_type = []
        pack_view_type = []
        pack_multi_modal = []
        
        for sample in pack:
            seq_len = sample["seq_len"]
            
            # Append input_ids
            pack_input_ids.append(sample["input_ids"])
            
            # Append loss_mask
            pack_loss_mask.append(sample["loss_mask"])
            
            # Position IDs: reset for each sub-sequence (CRITICAL for Flash Attention varlen)
            if sample["position_ids"].dim() == 1:
                # Standard: [seq_len]
                # Each sub-sequence starts from 0
                sub_position_ids = torch.arange(seq_len, dtype=sample["position_ids"].dtype)
            elif sample["position_ids"].dim() == 2:
                # VL model (Qwen2-VL): [4, seq_len]
                # IMPORTANT: For sequence packing, we need to reset position_ids for each sub-sequence
                # This makes the packed position_ids non-monotonic, triggering Flash Attention varlen path
                # 
                # Original position_ids might be: [0,1,2,...,seq_len-1] for each dimension
                # We need to reset to start from 0 for this sub-sequence
                #
                # For Qwen2-VL, the 4 dimensions are: [text_pos, h_pos, w_pos, t_pos]
                # Reset all dimensions to range [0, seq_len-1]
                sub_position_ids = torch.zeros((4, seq_len), dtype=sample["position_ids"].dtype)
                for dim in range(4):
                    sub_position_ids[dim] = torch.arange(seq_len, dtype=sample["position_ids"].dtype)
            else:
                sub_position_ids = sample["position_ids"]
            
            pack_position_ids.append(sub_position_ids)
            
            # Track has_view_type for each sub-sequence
            pack_has_view_type.append(sample["has_view_type"])
            
            # Track view_type for each sub-sequence
            pack_view_type.append(sample["view_type"])
            
            # Track multi_modal_inputs
            pack_multi_modal.append(sample["multi_modal_inputs"])
            
            # Update cumulative sequence lengths
            current_pos += seq_len
            cu_seqlens.append(current_pos)
        
        # Concatenate sequences in this pack
        packed_input_ids = torch.cat(pack_input_ids)
        packed_loss_mask = torch.cat(pack_loss_mask)
        
        # Position IDs: handle different dimensions
        if pack_position_ids[0].dim() == 1:
            # Standard: concatenate and keep sequential
            packed_position_ids = torch.cat(pack_position_ids)
        elif pack_position_ids[0].dim() == 2:
            # VL model: [4, seq_len] - concatenate along seq dimension
            packed_position_ids = torch.cat(pack_position_ids, dim=-1)
        else:
            packed_position_ids = torch.cat(pack_position_ids)
        
        # Build block-diagonal attention mask
        total_len = current_pos
        # For packed sequences, use 1D mask (Flash Attention will handle causal + doc boundaries)
        packed_attention_mask = torch.ones(total_len, dtype=torch.long)
        
        # Pad packed sequence to max_length
        if total_len < max_length:
            pad_len = max_length - total_len
            
            # Pad input_ids
            packed_input_ids = torch.cat([
                packed_input_ids,
                torch.full((pad_len,), pad_token_id, dtype=packed_input_ids.dtype)
            ])
            
            # Pad attention_mask
            packed_attention_mask = torch.cat([
                packed_attention_mask,
                torch.zeros(pad_len, dtype=packed_attention_mask.dtype)
            ])
            
            # Pad loss_mask
            packed_loss_mask = torch.cat([
                packed_loss_mask,
                torch.zeros(pad_len, dtype=packed_loss_mask.dtype)
            ])
            
            # Pad position_ids
            if packed_position_ids.dim() == 1:
                packed_position_ids = torch.cat([
                    packed_position_ids,
                    torch.zeros(pad_len, dtype=packed_position_ids.dtype)
                ])
            elif packed_position_ids.dim() == 2:
                # VL model: [4, total_len]
                packed_position_ids = torch.cat([
                    packed_position_ids,
                    torch.zeros((packed_position_ids.shape[0], pad_len), dtype=packed_position_ids.dtype)
                ], dim=-1)
        
        batch_input_ids.append(packed_input_ids)
        batch_attention_mask.append(packed_attention_mask)
        batch_position_ids.append(packed_position_ids)
        batch_loss_mask.append(packed_loss_mask)
        
        # For has_view_type: save per-subsequence info (not pack-level)
        # This allows subsequence-level KL loss control
        batch_has_view_type.append(pack_has_view_type)  # List of bools, one per subsequence

        # For view_type: extend with all subsequence view_types
        batch_view_type.extend(pack_view_type)
        
        # For multi_modal_inputs: merge all in this pack
        # CRITICAL: This is complex for VL models - need to merge pixel_values and image_grid_thw
        merged_mm = {}
        if any(mm for mm in pack_multi_modal if mm):
            # Collect all pixel_values and image_grid_thw
            all_pixel_values = []
            all_image_grid_thw = []
            
            for mm in pack_multi_modal:
                if mm and "pixel_values" in mm:
                    all_pixel_values.append(mm["pixel_values"])
                    if "image_grid_thw" in mm:
                        all_image_grid_thw.append(mm["image_grid_thw"])
            
            if all_pixel_values:
                # Concatenate along image dimension
                merged_mm["pixel_values"] = torch.cat(all_pixel_values, dim=0)
                if all_image_grid_thw:
                    merged_mm["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        
        batch_multi_modal_inputs.append(merged_mm if merged_mm else {})
        
        # Store cu_seqlens for Flash Attention varlen
        batch_cu_seqlens.append(torch.tensor(cu_seqlens, dtype=torch.int32))
        batch_max_seqlen.append(max(s["seq_len"] for s in pack))
    
    # 5. Stack into final batch
    first_pos_shape = batch_position_ids[0].shape
    first_pos_dim = batch_position_ids[0].dim()
    
    if first_pos_dim == 1:
        stacked_position_ids = torch.stack(batch_position_ids, dim=0)
    elif first_pos_dim == 2:
        # VL model: stack along dim=1
        stacked_position_ids = torch.stack(batch_position_ids, dim=1)
    else:
        stacked_position_ids = torch.stack(batch_position_ids, dim=0)
    
    result = {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "attention_mask": torch.stack(batch_attention_mask, dim=0),
        "position_ids": stacked_position_ids,
        "loss_mask": torch.stack(batch_loss_mask, dim=0),
        "cu_seqlens": batch_cu_seqlens,  # List of tensors
        "max_seqlen_in_batch": batch_max_seqlen,  # List of ints
        "is_packed": True,  # Flag to indicate packed batch
    }
    
    if batch_has_view_type:
        # For packed sequences, has_view_type is a list of lists (one list per pack, each element is a subsequence)
        # Store as list of lists to preserve subsequence-level information
        result["has_view_type"] = batch_has_view_type  # List of lists: [[bool, bool, ...], ...]
    
    # Always add view_type to result (even if empty list, for consistency)
    result["view_type"] = batch_view_type if batch_view_type else []
    
    if batch_multi_modal_inputs:
        result["multi_modal_inputs"] = batch_multi_modal_inputs
    
    return result


class TarDataset(torch.utils.data.IterableDataset):
    """Wrapper for WebDataset to handle tokenization and image processing for VL models"""

    def __init__(self, tar_files, tokenizer, config, processor=None, max_samples=-1):
        """
        Args:
            tar_files: list of tar file paths or directory
            tokenizer: HuggingFace tokenizer
            config: data config
            processor: HuggingFace processor for multimodal (optional)
            max_samples: maximum samples (not used for WebDataset)
        """
        from omegaconf.listconfig import ListConfig
        
        if isinstance(tar_files, str):
            dataset_entries = [tar_files]
        elif isinstance(tar_files, ListConfig):
            dataset_entries = list(tar_files)
        else:
            dataset_entries = list(tar_files)

        if len(dataset_entries) == 0:
            raise ValueError("TarDataset: no tar files or directories provided.")

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_length = config.get("max_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # Data format configuration (critical!)
        self.multiturn_enabled = config.get("multiturn", {}).get("enable", False)
        self.prompt_key = config.get("prompt_key", "question")
        self.response_key = config.get("response_key", "answer")
        self.messages_key = config.get("multiturn", {}).get("messages_key", "messages")

        # Image resolution control (critical for memory!)
        self.min_pixels = config.get("min_pixels", None)
        self.max_pixels = config.get("max_pixels", None)
        
        # Sequence packing configuration
        self.use_sequence_packing = config.get("use_sequence_packing", False)

        # Control whether to remove <think> tags from assistant messages
        # Default to True (remove think tags) for backward compatibility
        self.remove_think = config.get("remove_think", True)

        self.base_seed = config.get("seed", 42)
        self.global_tar_shuffle = config.get("global_tar_shuffle", True)
        self.dataset_infos: list[dict[str, object]] = []

        # Read dataset_view_types from config (similar to train_sampling_ratios)
        dataset_view_types = config.get("dataset_view_types", None)
        if dataset_view_types is not None:
            if isinstance(dataset_view_types, (list, ListConfig)):
                dataset_view_types = list(dataset_view_types)
            else:
                # If it's a dict mapping path to view_type
                dataset_view_types = dict(dataset_view_types) if isinstance(dataset_view_types, dict) else None

        for idx, entry in enumerate(dataset_entries):
            path = str(entry)
            is_remote = path.startswith("hdfs://") or path.startswith("s3://")
            is_file = path.endswith(".tar")

            if is_file:
                tar_urls = [path]
            elif is_remote or (not is_remote and os.path.isdir(path)):
                tar_urls = get_tar_urls_under_dir(path)
            else:
                tar_urls = [path]

            if len(tar_urls) == 0:
                raise ValueError(f"TarDataset: no tar files found under {path}")
            
            # Determine view_type for this dataset
            view_type = None
            if dataset_view_types is not None:
                if isinstance(dataset_view_types, list):
                    # List format: [view_type1, view_type2, ...] corresponding to dataset_entries
                    if idx < len(dataset_view_types):
                        view_type = str(dataset_view_types[idx])
                elif isinstance(dataset_view_types, dict):
                    # Dict format: {path: view_type, ...}
                    # Try exact match first
                    if path in dataset_view_types:
                        view_type = str(dataset_view_types[path])
                    else:
                        # Try matching by path prefix (for directories)
                        for config_path, config_view_type in dataset_view_types.items():
                            if path.startswith(str(config_path)) or str(config_path).startswith(path):
                                view_type = str(config_view_type)
                                break


            self.dataset_infos.append(
                {
                    "path": path,
                    "tar_urls": tar_urls,
                    "view_type": view_type,  # Store dataset-level view_type
                }
            )
            
        if self.global_tar_shuffle:
            rng = random.Random(self.base_seed)
            shard_assignments: list[tuple[int, str]] = []
            for dataset_idx, info in enumerate(self.dataset_infos):
                shard_assignments.extend((dataset_idx, url) for url in info["tar_urls"])
                info["tar_urls"] = []

            rng.shuffle(shard_assignments)

            for dataset_idx, url in shard_assignments:
                self.dataset_infos[dataset_idx]["tar_urls"].append(url)

        ratios = (
            config.get("train_sampling_ratios")
            or config.get("sampling_ratios")
            or config.get("train_sampling_ratio")
        )
        if ratios is None:
            ratios = [1.0] * len(self.dataset_infos)
        else:
            if isinstance(ratios, ListConfig):
                ratios = list(ratios)
            ratios = [float(r) for r in ratios]

        if len(ratios) < len(self.dataset_infos):
            if len(ratios) == 1:
                ratios = [ratios[0]] * len(self.dataset_infos)
            else:
                raise ValueError(
                    "Length of train_sampling_ratios must be >= number of train_files, "
                    "or provide a single value to broadcast."
                )
        elif len(ratios) > len(self.dataset_infos):
            ratios = ratios[: len(self.dataset_infos)]

        if any(r <= 0 for r in ratios):
            raise ValueError("Sampling ratios must be positive numbers.")

        self.desired_ratios = ratios
        ratio_sum = sum(self.desired_ratios)
        self.dataset_sampling_probs = [r / ratio_sum for r in self.desired_ratios]

        base_cycle = config.get("max_sampling_cycle")
        if base_cycle is None or base_cycle <= 0:
            base_cycle = config.get("train_max_samples", -1)
        if base_cycle is None or base_cycle <= 0:
            base_cycle = max(len(self.dataset_infos), 1024)
        else:
            base_cycle = max(len(self.dataset_infos), int(math.ceil(base_cycle)))

        raw_targets: list[int] = [
            max(1, int(round(base_cycle * prob))) for prob in self.dataset_sampling_probs
        ]

        
        total_targets = sum(raw_targets)

        # Further reduce counts by greatest common divisor to keep schedule compact
        # Only apply GCD optimization if there are multiple datasets AND it doesn't reduce samples too much
        if len(raw_targets) > 1:
            gcd_value = raw_targets[0]
            for target in raw_targets[1:]:
                gcd_value = math.gcd(gcd_value, target)
            if gcd_value > 1:
                # Only apply GCD if the result is still >= batch_size (to avoid empty dataloader)
                # Estimate batch_size from config if available
                estimated_batch_size = self.config.get("train_batch_size", self.config.get("gen_batch_size", 16))
                reduced_targets = [max(1, target // gcd_value) for target in raw_targets]
                reduced_total = sum(reduced_targets)
                # Only apply GCD if reduced_total is still >= estimated_batch_size
                if reduced_total >= estimated_batch_size:
                    raw_targets = reduced_targets
                    total_targets = reduced_total

        self.target_counts = raw_targets
        self.samples_per_cycle = total_targets
        
        # Debug: print samples_per_cycle information
        print(f"[TarDataset] samples_per_cycle: {self.samples_per_cycle}, base_cycle: {base_cycle}, "
              f"dataset_infos count: {len(self.dataset_infos)}, target_counts: {raw_targets}")

        self.dataset_pipelines = []
        for idx, info in enumerate(self.dataset_infos):
            dataset_seed = self.base_seed + idx * 997
            default_view_type = info.get("view_type", None)  # Get dataset-level view_type
            pipeline = create_webdataset(
                urls=info["tar_urls"], 
                base_seed=dataset_seed,
                default_view_type=default_view_type,
                remove_think=self.remove_think
            )
            self.dataset_pipelines.append(pipeline)

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            self._log_dataset_mix()

    def _log_dataset_mix(self):
        print("TarDataset: sampling summary")
        print(f"  Num datasets: {len(self.dataset_infos)}")
        print(f"  Samples per cycle: {self.samples_per_cycle}")
        print("  Dataset details:")
        for idx, (info, ratio, target, prob) in enumerate(
            zip(
                self.dataset_infos,
                self.desired_ratios,
                self.target_counts,
                self.dataset_sampling_probs,
            )
        ):
            tar_count = len(info.get("tar_urls", []))
            path = info.get("path")
            print(
                f"    [{idx}] {path} | shards={tar_count} | "
                f"target/cycle={target} | desired_ratio={ratio} | prob={prob:.4f}"
            )

    def get_collate_fn(self):
        """Return a collate function (packed or standard based on config)"""
        from functools import partial

        if self.use_sequence_packing:
            # Use sequence packing for better efficiency
            if dist.get_rank() == 0:
                print("📦 Using Sequence Packing collate function")
            return partial(
                packed_collate_fn,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=self.max_length,
                image_token_id=151655,  # Qwen2-VL image_pad token
            )
        else:
            # Use standard padding collate function
            return partial(tar_collate_fn, pad_token_id=self.tokenizer.pad_token_id)

    def _validate_sample(self, output, images, messages, sample_idx):
        """Validate that image, question, answer are correctly paired and loss_mask is correct

        This is called for the first few samples to ensure data quality
        """
        print("\n" + "=" * 100)
        print(f"🔍 DATA VALIDATION CHECKPOINT - Sample #{sample_idx}")
        print("=" * 100)

        # Extract data
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]
        loss_mask = output["loss_mask"]
        position_ids = output.get("position_ids")
        multi_modal_inputs = output.get("multi_modal_inputs", {})

        seq_len = input_ids.shape[0]

        print("\n【基本信息】")
        print(f"  序列长度: {seq_len}")
        print(f"  图像数量: {len(images)}")
        print(f"  消息数量: {len(messages)}")
        print(f"  Position IDs shape: {position_ids.shape if position_ids is not None else 'None'}")

        # Decode full text
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        print("\n【完整文本（带特殊标记）】")
        print(f"  {repr(full_text[:500])}...")  # 只显示前 500 字符

        # Check messages structure and match with loss_mask
        print("\n【消息结构与 Loss Mask 对应】")
        is_multiturn = len(messages) > 2  # More than user+assistant = multi-turn

        if is_multiturn:
            print(f"  ⚠️  检测到多轮对话（{len(messages)} 条消息）")
            print("  期望：所有 assistant 消息都应计算 loss (mask=1)")

        assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
        user_count = sum(1 for msg in messages if msg.get("role") == "user")
        print(f"  User 消息数: {user_count}, Assistant 消息数: {assistant_count}")
        print()

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", [])

            text_parts = []
            image_count = 0
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    image_count += 1

            full_text = " ".join(text_parts)
            expected_mask = "1 (计算 loss)" if role == "assistant" else "0 (不计算)"

            print(f"  消息 {i} ({role}) - 期望 loss_mask={expected_mask}:")
            print(f"    图像数量: {image_count}")
            if len(full_text) > 0:
                print(f"    文本长度: {len(full_text)} 字符")
                print(f"    文本: {repr(full_text[:100])}...")

        # Warn if multi-turn but loss_mask ratio is too low
        if is_multiturn:
            loss_ratio = loss_mask.sum().item() / seq_len
            expected_ratio = 0.3  # Rough estimate, assistant messages should be significant portion
            if loss_ratio < 0.15:
                print(f"\n  ⚠️⚠️⚠️  WARNING: 多轮对话但 loss_mask 占比很低 ({loss_ratio * 100:.1f}%)")
                print("  可能只有最后一轮 assistant 计算了 loss，前面的轮次被浪费了！")

        # === NEW: Detailed per-message loss_mask analysis for multi-turn ===
        if is_multiturn and assistant_count > 1:
            print("\n【每条 Assistant 消息的 Loss Mask 详情】")
            # Try to find where each assistant message is in the token sequence
            # by checking loss_mask=1 segments

            # Find all segments where loss_mask=1
            loss_segments = []
            in_segment = False
            segment_start = None

            for i in range(seq_len):
                if loss_mask[i].item() == 1 and not in_segment:
                    segment_start = i
                    in_segment = True
                elif loss_mask[i].item() == 0 and in_segment:
                    loss_segments.append((segment_start, i))
                    in_segment = False

            # Handle last segment
            if in_segment:
                loss_segments.append((segment_start, seq_len))

            print(f"  发现 {len(loss_segments)} 个 loss_mask=1 的连续片段")
            print(f"  期望 {assistant_count} 个片段（每个 assistant 消息一个）")

            if len(loss_segments) != assistant_count:
                print("  ⚠️⚠️⚠️  WARNING: 片段数量不匹配！")
                print("  这说明多轮 loss_mask 计算可能失败了，可能使用了 fallback 逻辑")
            else:
                print("  ✅ 片段数量匹配！多轮 loss_mask 计算成功")

            # Show each segment
            for idx, (start, end) in enumerate(loss_segments):
                segment_tokens = input_ids[start:end]
                segment_text = self.tokenizer.decode(segment_tokens, skip_special_tokens=False)
                print(f"\n  片段 {idx + 1}: tokens [{start}:{end}] ({end - start} tokens)")
                print(f"    文本: {repr(segment_text[:150])}...")

        # Check loss_mask distribution
        loss_mask_sum = loss_mask.sum().item()
        loss_mask_zeros = (loss_mask == 0).sum().item()
        loss_mask_ones = (loss_mask == 1).sum().item()

        print("\n【Loss Mask 分析】")
        print(f"  总长度: {seq_len}")
        print(f"  Mask=0 (prompt): {loss_mask_zeros} ({loss_mask_zeros / seq_len * 100:.1f}%)")
        print(f"  Mask=1 (answer): {loss_mask_ones} ({loss_mask_ones / seq_len * 100:.1f}%)")
        print(f"  Loss 计算的 token 数: {loss_mask_sum}")

        # Find boundary between prompt and answer
        # First token with loss_mask=1 indicates answer start
        answer_start_idx = None
        for i in range(seq_len):
            if loss_mask[i].item() == 1:
                answer_start_idx = i
                break

        if answer_start_idx is not None:
            print("\n【Prompt/Answer 边界】")
            print(f"  Answer 开始位置: {answer_start_idx}")

            # Decode prompt part
            prompt_tokens = input_ids[:answer_start_idx]
            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            print(f"  Prompt 长度: {len(prompt_tokens)} tokens, {len(prompt_text)} 字符")
            print(f"    前 300 字符: {repr(prompt_text[:300])}...")
            if len(prompt_text) > 300:
                print(f"    后 200 字符: ...{repr(prompt_text[-200:])}")

            # Decode answer part
            answer_tokens = input_ids[answer_start_idx:]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=False)
            print(f"  Answer 长度: {len(answer_tokens)} tokens, {len(answer_text)} 字符")
            print(f"    前 300 字符: {repr(answer_text[:300])}...")
            if len(answer_text) > 300:
                print(f"    后 200 字符: ...{repr(answer_text[-200:])}")

        # Visualize loss_mask pattern (first 50 tokens)
        print("\n【Loss Mask 可视化（前 50 tokens）】")
        print(f"  {'Pos':<6} {'Token ID':<10} {'Attn':<6} {'Loss':<6} {'Token Text':<30}")
        print(f"  {'-' * 6} {'-' * 10} {'-' * 6} {'-' * 6} {'-' * 30}")

        for i in range(min(50, seq_len)):
            token_id = input_ids[i].item()
            attn = attention_mask[i].item()
            loss = loss_mask[i].item()
            token_text = self.tokenizer.decode([token_id])

            # Mark prompt vs answer
            if loss == 0:
                marker = "PROMPT" if i < seq_len - 1 else "LAST"
            else:
                marker = "ANSWER"

            print(f"  {i:<6} {token_id:<10} {attn:<6} {loss:<6} {repr(token_text):<30} [{marker}]")

        if seq_len > 50:
            print(f"  ... (省略 {seq_len - 50} 个 tokens)")

        # Check multi_modal_inputs
        if multi_modal_inputs:
            print("\n【Multi-modal Inputs】")
            for key, value in multi_modal_inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: type={type(value)}")

        # Validation checks
        print("\n【验证检查】")
        checks_passed = True

        # Check 1: Loss mask should have both 0s and 1s
        if loss_mask_ones == 0:
            print("  ❌ FAIL: Loss mask 全是 0，没有 answer 部分！")
            checks_passed = False
        elif loss_mask_zeros == 0:
            print("  ⚠️  WARNING: Loss mask 全是 1，prompt 没有被 mask！")
            checks_passed = False
        else:
            print("  ✅ PASS: Loss mask 同时包含 prompt (0) 和 answer (1)")

        # Check 2: Last token should be masked
        if loss_mask[seq_len - 1].item() != 0:
            print("  ⚠️  WARNING: 最后一个 token 的 loss_mask 不是 0")
        else:
            print("  ✅ PASS: 最后一个 token 被正确 mask")

        # Check 3: Answer portion should be reasonable (not too small)
        answer_ratio = loss_mask_ones / seq_len
        if answer_ratio < 0.05:
            print(f"  ⚠️  WARNING: Answer 部分太小 ({answer_ratio * 100:.1f}%)")
        elif answer_ratio > 0.95:
            print(f"  ⚠️  WARNING: Answer 部分太大 ({answer_ratio * 100:.1f}%)")
        else:
            print(f"  ✅ PASS: Answer 部分占比合理 ({answer_ratio * 100:.1f}%)")

        # Check 4: Image count matches
        if self.processor is not None and len(images) > 0:
            if "pixel_values" in multi_modal_inputs:
                print("  ✅ PASS: 图像已处理，pixel_values 存在")
            else:
                print("  ⚠️  WARNING: 有图像但 pixel_values 不存在")

        if checks_passed:
            print(f"\n✅ 样本 #{sample_idx} 验证通过")
        else:
            print(f"\n❌ 样本 #{sample_idx} 验证失败，请检查")

        print("=" * 100 + "\n")

    def __iter__(self):
        """Iterate over samples and process them (supports multimodal)"""
        import time

        processed_count = 0
        error_count = 0
        skipped_assistant_with_image = 0  # Track samples with images in assistant

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        rng = random.Random(self.base_seed + rank * 1000 + worker_id)

        dataset_iters = [iter(pipeline) for pipeline in self.dataset_pipelines]
        remaining_counts = list(self.target_counts)
        total_remaining = sum(remaining_counts)

        while True:
            if total_remaining == 0:
                remaining_counts = list(self.target_counts)
                total_remaining = sum(remaining_counts)

            available_indices = [i for i, count in enumerate(remaining_counts) if count > 0]
            if not available_indices:
                # Should not happen, but reset just in case.
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
                # Reset iterator and try again
                dataset_iters[dataset_idx] = iter(self.dataset_pipelines[dataset_idx])
                iterator = dataset_iters[dataset_idx]
                try:
                    sample = next(iterator)
                except StopIteration:
                    # If still no data after reset, skip this iteration
                    # This should not happen in normal cases
                    continue
            

            if sample is None:
                continue

            

            sample_start_time = time.time()

            # === Step 1: Extract data based on configuration ===
            images = sample.get("images", [])
            has_view_type = sample.get("has_view_type", False)  # Extract view_type flag
            view_type = sample.get("view_type", "unknown")
            if view_type is None:
                view_type = "unknown"
            view_type = view_type if isinstance(view_type, str) else str(view_type)

            if self.multiturn_enabled:
                # Multi-turn mode: use messages format
                messages = sample.get(self.messages_key, [])
                if not messages:
                    error_count += 1
                    continue

                # Filter: Skip samples with images in assistant messages
                # Assistant should only output text, not images
                # CRITICAL: Check this BEFORE counting expected images
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

                # Filter: Skip samples with too many messages (too slow to process)
                MAX_MESSAGES = 10  # Conservative limit to avoid extremely slow samples
                if len(messages) > MAX_MESSAGES:
                    error_count += 1
                    continue
            else:
                # Single-turn mode: extract question/answer from messages or direct fields
                # First try to get from direct fields (for backward compatibility)
                question = sample.get(self.prompt_key)
                answer = sample.get(self.response_key)
                
                # If not found, extract from messages format (decode_sample always returns messages)
                if question is None or answer is None:
                    messages_from_sample = sample.get("messages", [])
                    if messages_from_sample and len(messages_from_sample) >= 2:
                        # Extract first user message as question
                        first_user_msg = None
                        first_assistant_msg = None
                        for msg in messages_from_sample:
                            if msg.get("role") == "user" and first_user_msg is None:
                                first_user_msg = msg
                            elif msg.get("role") == "assistant" and first_assistant_msg is None:
                                first_assistant_msg = msg
                            if first_user_msg and first_assistant_msg:
                                break
                        
                        if first_user_msg and first_assistant_msg:
                            # Extract text from user message
                            user_content = first_user_msg.get("content", [])
                            question_parts = []
                            for item in user_content:
                                if item.get("type") == "text" and item.get("text"):
                                    question_parts.append(item["text"])
                            question = " ".join(question_parts) if question_parts else None
                            
                            # Extract text from assistant message
                            assistant_content = first_assistant_msg.get("content", [])
                            answer_parts = []
                            for item in assistant_content:
                                if item.get("type") == "text" and item.get("text"):
                                    answer_parts.append(item["text"])
                            answer = " ".join(answer_parts) if answer_parts else None
                
                if question is None or answer is None:
                    error_count += 1
                    continue

                # Build user content: text + images
                user_content = [{"type": "text", "text": question}]
                # Add images to user content if present
                for i in range(len(images)):
                    user_content.append({"type": "image", "index": i})

                # Convert to messages format for consistent processing
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]

            if self.processor is not None:
                # Use processor for multimodal models (VL models)
                from verl.utils.dataset.vision_utils import process_image

                # CRITICAL CHECK 1: Count expected images BEFORE processing
                # Count images referenced in messages (user messages only, assistant shouldn't have images)
                expected_image_count = 0
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        for item in content:
                            if item.get("type") == "image":
                                expected_image_count += 1

                # Validate image count matches
                actual_image_count = len(images)
                if expected_image_count != actual_image_count:
                    # Image count mismatch - skip this sample
                    if error_count < 5:
                        print(
                            f"⚠️ Skipping sample: Image count mismatch (expected {expected_image_count}, got {actual_image_count})"
                        )
                    error_count += 1
                    continue

                # === Step 1: Process images ===
                # CRITICAL: For multi-image samples, all images will be resized to same size (min_pixels/max_pixels)
                # This ensures consistent visual token count per image
                processed_images = None
                if images:
                    processed_images = [process_image(img, image_patch_size=self.image_patch_size) for img in images]

                    # Validate all images were processed successfully
                    if len(processed_images) != len(images):
                        if error_count < 5:
                            print(
                                f"⚠️ Skipping sample: Image processing failed ({len(processed_images)}/{len(images)} processed)"
                            )
                        error_count += 1
                        continue

                # === Step 2: Get full text and process with images ===
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )

                # === Step 3: Use processor to encode text + images ===
                # IMPORTANT: Set min_pixels and max_pixels to control image resolution
                # DO NOT use truncation=True to avoid breaking image tokens
                processor_kwargs = {"text": [raw_prompt], "images": processed_images, "return_tensors": "pt"}
                # Override min_pixels and max_pixels if specified
                if self.min_pixels is not None:
                    processor_kwargs["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    processor_kwargs["max_pixels"] = self.max_pixels

                model_inputs = self.processor(**processor_kwargs)

                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]

                # Get sequence length first (needed for checks)
                sequence_length = input_ids.shape[0]

                # CRITICAL CHECK 2: Verify image tokens are NOT truncated
                # Count image_pad tokens in input_ids (Qwen2-VL uses token 151655)
                image_token_id = 151655  # <|image_pad|> token for Qwen2-VL
                n_image_tokens = (input_ids == image_token_id).sum().item()

                # Calculate expected image tokens from image_grid_thw
                expected_image_tokens = 0
                if "image_grid_thw" in model_inputs and processed_images:
                    image_grid_thw = model_inputs["image_grid_thw"]
                    # image_grid_thw shape: [num_images, 3] (height, width, time)
                    # Each image token represents merge_size^2 features
                    merge_size = getattr(self.processor.image_processor, "spatial_merge_size", 2)
                    for i in range(image_grid_thw.shape[0]):
                        h, w, t = image_grid_thw[i]
                        features_per_image = h * w * t
                        tokens_per_image = features_per_image // (merge_size**2)
                        expected_image_tokens += tokens_per_image

                # If we have images, verify tokens match
                if processed_images and expected_image_tokens > 0:
                    if n_image_tokens != expected_image_tokens:
                        # Image tokens mismatch - likely truncated
                        if error_count < 5:
                            print(
                                f"⚠️ Skipping sample: Image tokens mismatch (found {n_image_tokens}, expected {expected_image_tokens})"
                            )
                            print("   This likely means <image> tokens were truncated!")
                        error_count += 1
                        continue

                    # CRITICAL CHECK 3: Verify image tokens are in valid positions
                    # Find positions of image tokens
                    # image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
                    # if len(image_token_positions) > 0:
                    #     first_image_pos = image_token_positions[0].item()
                    #     last_image_pos = image_token_positions[-1].item()

                    #     # If images are at the very end, they might be truncated
                    #     # Images should typically be in the first half of the sequence (user message)
                    #     if last_image_pos > sequence_length * 0.8:
                    #         if error_count < 5:
                    #             print(f"⚠️ Skipping sample: Image tokens at end of sequence (pos {last_image_pos}/{sequence_length})")
                    #             print(f"   This might indicate truncation!")
                    #         error_count += 1
                    #         continue

                # Check sequence length and intelligently truncate if too long
                # CRITICAL: Must NOT truncate image tokens - only truncate text at the end
                if sequence_length > self.max_length:
                    # Find the last image token position to ensure we don't truncate images
                    image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(image_token_positions) > 0:
                        last_image_pos = image_token_positions[-1].item()
                        
                        # Safe truncation point: must be after all image tokens
                        # Add a small buffer (e.g., 10 tokens) after the last image token
                        safe_truncation_start = last_image_pos + 10
                        
                        # If max_length doesn't leave room for images + buffer, skip
                        if self.max_length < safe_truncation_start:
                            if error_count < 5:
                                print(f"⚠️ Skipping sample: Images take {safe_truncation_start} tokens, "
                                      f"but max_length={self.max_length}")
                            error_count += 1
                            continue
                        
                        # Truncate from the end (remove text, keep images)
                        input_ids = input_ids[:self.max_length]
                        attention_mask = attention_mask[:self.max_length]
                        
                        # Verify image tokens are still intact after truncation
                        n_image_tokens_after = (input_ids == image_token_id).sum().item()
                        if n_image_tokens_after != n_image_tokens:
                            # Image tokens were truncated - this should not happen with our logic
                            if error_count < 5:
                                print(f"⚠️ ERROR: Image tokens truncated ({n_image_tokens} → {n_image_tokens_after})")
                            error_count += 1
                            continue
                        
                        # Update sequence_length
                        sequence_length = self.max_length
                        
                        # Log truncation occasionally
                        if processed_count < 10 and processed_images:
                            print(f"✂️ Truncated sample: {input_ids.shape[0]} → {self.max_length} tokens "
                                  f"(images intact: {n_image_tokens} tokens)")
                    else:
                        # No images, can truncate freely from the end
                        input_ids = input_ids[:self.max_length]
                        attention_mask = attention_mask[:self.max_length]
                        sequence_length = self.max_length

                # Handle position_ids for different VL models
                # Important: compute position_ids AFTER truncation for Qwen2-VL
                if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                    # Qwen2-VL uses special rope for images
                    if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                        from verl.models.transformers.qwen3_vl import get_rope_index
                    else:
                        from verl.models.transformers.qwen2_vl import get_rope_index

                    # get_rope_index returns vision position_ids (3, seq_len)
                    # We need to add text_position_ids to get (4, seq_len)
                    vision_position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids,
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask,
                    )  # (3, seq_length)

                    # Add text_position_ids (standard sequential IDs)
                    valid_mask = attention_mask.bool()
                    text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
                    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())

                    # Combine: (1, seq) + (3, seq) → (4, seq)
                    position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)

                elif "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
                    # GLM4V uses special rope
                    from verl.models.transformers.glm4v import get_rope_index

                    vision_position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids,
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        attention_mask=attention_mask,
                    )  # (3, seq_length)

                    # Add text_position_ids
                    valid_mask = attention_mask.bool()
                    text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
                    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())

                    # Combine: (1, seq) + (3, seq) → (4, seq)
                    position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
                else:
                    # Standard position_ids - but for Qwen2-VL, we still need [4, seq_len] format
                    from verl.utils.model import compute_position_id_with_mask
                    
                    # Check if this is a Qwen2-VL model (processor has Qwen2VL in name)
                    if self.processor and "Qwen2VL" in self.processor.__class__.__name__:
                        # Generate standard 1D position_ids
                        text_position_ids_1d = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
                        # Expand to [4, seq_len] format for Qwen2-VL
                        # First dim (text): use actual position_ids
                        # Last 3 dims (vision): use zeros (no images)
                        text_position_ids = text_position_ids_1d.unsqueeze(0)  # [1, seq_len]
                        vision_position_ids = torch.zeros((3, len(input_ids)), dtype=torch.long)  # [3, seq_len]
                        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # [4, seq_len]
                    else:
                        # Non-VL model: use standard 1D position_ids
                        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

                # === Step 4: Prepare output dict ===
                output = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "has_view_type": has_view_type,  # Pass view_type flag for KL loss control
                    "view_type": view_type,
                }

                # Add other multi_modal_inputs (pixel_values, image_grid_thw, etc.)
                # CRITICAL: Always add multi_modal_inputs (even if empty) to maintain batch consistency
                if model_inputs:
                    output["multi_modal_inputs"] = dict(model_inputs)
                else:
                    # No images in this sample, use empty dict
                    output["multi_modal_inputs"] = {}

                # === Step 5: Create loss_mask for dialogue ===
                # Strategy: Use special tokens in input_ids to find boundaries
                # This avoids both: (1) reprocessing images, (2) inaccurate ratio mapping
                loss_mask = torch.zeros_like(attention_mask)
                sequence_length = input_ids.shape[0]

                # Use token-based search to find assistant boundaries
                # This is accurate and doesn't reprocess images
                # Works for both single-turn and multi-turn

                # Check if last message is assistant (required for training)
                if not messages or messages[-1]["role"] != "assistant":
                    # Data quality issue: no assistant response to train on
                    if error_count < 5:
                        print(f"⚠️ Skipping sample: No assistant message in conversation (error #{error_count + 1})")
                    error_count += 1
                    continue

                if messages[-1]["role"] == "assistant":
                    # Find assistant marker in input_ids
                    # Qwen2-VL format: <|im_start|> (151644) + "assistant" (77091) + \n (198)
                    im_start_token = 151644  # <|im_start|>
                    assistant_token = 77091  # "assistant" (NOT 872 which is "user"!)

                    # Find all occurrences of <|im_start|>assistant
                    assistant_positions = []
                    debug_this_sample = False  # Disable debug output (verified working)

                    for i in range(len(input_ids) - 1):
                        if input_ids[i].item() == im_start_token:
                            next_token = input_ids[i + 1].item() if i + 1 < len(input_ids) else -1
                            if next_token == assistant_token:
                                # Found <|im_start|>assistant, skip to next token after \n
                                # Typical: <|im_start|>assistant\n
                                pos = i + 2  # After <|im_start|> and "assistant"
                                # Skip \n if present
                                if pos < len(input_ids) and input_ids[pos].item() == 198:
                                    pos += 1
                                assistant_positions.append((i, pos))  # Save both marker pos and content start

                                if debug_this_sample:
                                    print(
                                        f"[Sample {processed_count + 1}] Found assistant at position {i}, content starts at {pos}"
                                    )

                    # For multi-turn support: mark ALL assistant segments
                    # IMPORTANT: Exclude the last assistant if it's the generation prompt (no content after it)
                    if not assistant_positions:
                        # Cannot find assistant marker tokens in input_ids
                        # This is a data/tokenization issue - skip this sample
                        if error_count < 5:
                            print(f"⚠️ Skipping sample: Cannot find assistant marker tokens (error #{error_count + 1})")
                        error_count += 1
                        continue

                    if assistant_positions:
                        im_end_token = 151645  # <|im_end|>

                        # Filter out generation prompt (last assistant with no <|im_end|> after it)
                        valid_assistants = []

                        for idx, (marker_pos, content_start) in enumerate(assistant_positions):
                            # Find where this assistant message ends
                            end_pos = None

                            # Look for <|im_end|> after this assistant
                            for j in range(content_start, len(input_ids)):
                                if input_ids[j].item() == im_end_token:
                                    end_pos = j  # End at <|im_end|> (don't include it)
                                    break

                            # Only include assistants with <|im_end|> (has actual content)
                            # Generation prompt at the end won't have <|im_end|>
                            if end_pos is not None and end_pos > content_start:
                                segment_length = end_pos - content_start
                                if segment_length > 0:
                                    valid_assistants.append((content_start, end_pos))

                                    if debug_this_sample:
                                        # Decode this segment to see content
                                        segment_tokens = input_ids[content_start:end_pos]
                                        segment_text = self.tokenizer.decode(segment_tokens)
                                        print(
                                            f"  Assistant #{idx + 1}: [{content_start}:{end_pos}] ({segment_length} tokens)"
                                        )
                                        print(f"    Content: {repr(segment_text[:100])}")
                            elif debug_this_sample:
                                print(
                                    f"  Assistant #{idx + 1}: [{marker_pos}] - Filtered (no <|im_end|>, likely generation prompt)"
                                )

                        # Debug: Log if we filtered out positions (disabled for clean output)
                        # if len(valid_assistants) < len(assistant_positions):
                        #     filtered = len(assistant_positions) - len(valid_assistants)
                        #     if debug_this_sample or processed_count <= 10:
                        #         print(f"[Sample {processed_count+1}] Filtered {filtered} generation prompts from {len(assistant_positions)} positions")
                        #         print(f"  Valid assistants: {len(valid_assistants)}")

                        # Set loss_mask for all valid assistant segments
                        if valid_assistants:
                            for start_pos, end_pos in valid_assistants:
                                # CRITICAL FIX: Include <|im_end|> token (at end_pos) in loss_mask
                                # This teaches the model WHEN to stop generating
                                # Without this, model never learns to predict EOS → infinite repetition
                                actual_end = min(end_pos + 1, sequence_length)
                                loss_mask[start_pos:actual_end] = 1
                                
                                # DEBUG: Print for first few samples
                                # if processed_count <= 3:
                                #     print(f"[DEBUG LOSS_MASK] Sample {processed_count}: "
                                #           f"assistant segment [{start_pos}:{actual_end}], "
                                #           f"end_pos(im_end)={end_pos}, seq_len={sequence_length}")
                        else:
                            # All assistants were filtered out (only generation prompts)
                            # This is a data quality issue - skip this sample
                            if error_count < 5:
                                print(
                                    f"⚠️ Skipping sample: All assistant messages filtered (likely generation prompts only, error #{error_count + 1})"
                                )
                            error_count += 1
                            continue

                output["loss_mask"] = loss_mask

                # Quick sanity check on loss_mask
                mask_ones = (loss_mask == 1).sum().item()
                
                # CRITICAL: If truncation removed all answer tokens, skip this sample
                if mask_ones == 0:
                    if error_count < 5:
                        print(f"⚠️ Skipping sample: Truncation removed all answer tokens "
                              f"(sequence was truncated to {sequence_length} tokens)")
                    error_count += 1
                    continue
                
                # mask_ratio = mask_ones / sequence_length if sequence_length > 0 else 0

                processed_count += 1

                # === Debug checkpoint: Validate first few samples ===
                # if processed_count <= 5:
                #     self._validate_sample(output, images, messages, processed_count)

                # Calculate processing time
                sample_time = time.time() - sample_start_time

                # Print statistics - only every 1000 samples
                # if processed_count <= 10:
                #     print(f"[TarDataset] ✅ Sample #{processed_count}: len={sequence_length}, images={len(images)}, messages={len(messages)}, time={sample_time:.2f}s")
                # elif processed_count <= 100 and processed_count % 10 == 0:
                #     print(f"[TarDataset] Processed: {processed_count} samples, Errors: {error_count}")
                if processed_count % 1000 == 0:
                    print(
                        f"[TarDataset] Processed: {processed_count}, Errors: {error_count}, "
                        f"Skipped (assistant has image): {skipped_assistant_with_image}"
                    )

                # Warn if sample takes too long
                # if sample_time > 10.0:
                #     print(f"[TarDataset] ⚠️ SLOW SAMPLE: #{processed_count} took {sample_time:.2f}s (messages={len(messages)}, len={sequence_length})")

                yield output

            else:
                # Text-only mode: use tokenizer with chat template
                # Apply chat template to get full text
                full_text = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )

                # Tokenize full text with truncation enabled
                # For text-only samples, tokenizer truncation is safe (no images to protect)
                encoded = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    truncation=True,  # Always enable truncation for text-only
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"][0]
                attention_mask = encoded["attention_mask"][0]
                sequence_length = input_ids.shape[0]
                
                # Log truncation for text-only samples
                if sequence_length == self.max_length and processed_count < 10:
                    print(f"✂️ Truncated text-only sample to {self.max_length} tokens")

                # === Create loss_mask for multi-turn dialogue (text-only) ===
                loss_mask = torch.zeros_like(attention_mask)

                # Process messages incrementally to find exact token boundaries
                prev_length = 0
                message_boundaries = []

                for msg_idx, msg in enumerate(messages):
                    # Get messages up to current one (inclusive)
                    partial_messages = messages[: msg_idx + 1]
                    partial_text = self.tokenizer.apply_chat_template(
                        partial_messages,
                        add_generation_prompt=(msg_idx == len(messages) - 1),
                        tokenize=False,
                        **self.apply_chat_template_kwargs,
                    )

                    # Tokenize to get length (use same settings as full encoding)
                    partial_encoded = self.tokenizer(
                        partial_text,
                        add_special_tokens=True,  # Match full encoding
                        return_tensors="pt",
                    )
                    current_length = partial_encoded["input_ids"][0].shape[0]

                    # This message's tokens span from prev_length to current_length
                    msg_start = prev_length
                    msg_end = current_length

                    # Store boundary for this message
                    message_boundaries.append((msg_start, msg_end, msg["role"]))

                    # Update for next iteration
                    prev_length = current_length

                # Verify: last message's end should match total sequence length
                if len(message_boundaries) > 0:
                    last_start, last_end, last_role = message_boundaries[-1]
                    if abs(last_end - sequence_length) > 2:  # Allow small tolerance
                        raise ValueError(
                            f"Text-only length mismatch: last_end={last_end}, sequence_length={sequence_length}"
                        )

                # Set loss_mask for all assistant messages
                # CRITICAL: Skip <|im_start|>assistant\n tokens (same as multi-modal mode)
                # Only mark assistant content and <|im_end|> for loss calculation
                im_start_token = 151644  # <|im_start|>
                assistant_token = 77091   # "assistant"
                im_end_token = 151645     # <|im_end|>
                
                for msg_start, msg_end, role in message_boundaries:
                    if role == "assistant" and msg_end > msg_start:
                        # Find where assistant content actually starts (after <|im_start|>assistant\n)
                        content_start = msg_start
                        
                        # Search for <|im_start|>assistant pattern in this message segment
                        for i in range(msg_start, min(msg_start + 10, msg_end - 1)):
                            if input_ids[i].item() == im_start_token:
                                next_token = input_ids[i + 1].item() if i + 1 < msg_end else -1
                                if next_token == assistant_token:
                                    # Found <|im_start|>assistant, skip to content start
                                    pos = i + 2  # After <|im_start|> and "assistant"
                                    # Skip \n if present
                                    if pos < msg_end and input_ids[pos].item() == 198:
                                        pos += 1
                                    content_start = pos
                                    break
                        
                        # Ensure indices are within sequence bounds
                        actual_start = max(0, min(content_start, sequence_length - 1))
                        actual_end = max(actual_start + 1, min(msg_end, sequence_length))

                        if actual_end > actual_start:
                            loss_mask[actual_start:actual_end] = 1

                # Do not mask out the last token - let model learn EOS token
                # if sequence_length > 0:
                #     loss_mask[sequence_length - 1] = 0

                from verl.utils.model import compute_position_id_with_mask

                position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
                
                # CRITICAL: Check if truncation removed all answer tokens
                mask_ones = (loss_mask == 1).sum().item()
                if mask_ones == 0:
                    if error_count < 5:
                        print(f"⚠️ Skipping text-only sample: No answer tokens after truncation "
                              f"(sequence length: {sequence_length})")
                    error_count += 1
                    continue

                processed_count += 1

                # Print statistics every 1000 samples
                # if processed_count % 1000 == 0:
                #     print(f"[TarDataset] Processed: {processed_count} samples, Errors: {error_count}")

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "loss_mask": loss_mask,
                    "has_view_type": has_view_type,  # Pass view_type flag for KL loss control
                    "view_type": view_type,
                }


class DynamicPackingDataset(torch.utils.data.IterableDataset):
    """
    Dynamically pack samples until reaching target batch size.
    
    This wrapper continuously fetches samples from the underlying dataset
    and packs them into sequences. It yields a batch only when the number
    of PACKED sequences reaches the target batch size.
    
    Benefits:
    - Safe: won't OOM even with long sequences (packs fewer samples)
    - Efficient: automatically uses more samples when they're short
    - Dynamic: adapts to actual data distribution
    
    Example:
        If target_batch_size = 256 and max_length = 2048:
        - Short samples (avg 500 tokens): may fetch ~1000 samples → pack into 256 batches
        - Long samples (avg 1800 tokens): may fetch ~256 samples → pack into 256 batches
    """
    
    def __init__(
        self,
        dataset,
        target_batch_size,
        max_length=2048,
        pad_token_id=0,
        image_token_id=151655,
        max_images_per_pack=3,
        max_image_tokens_per_pack=10000,
        mini_batch_size=64,  # Fetch mini_batch_size samples at a time for incremental packing
    ):
        """
        Args:
            dataset: underlying IterableDataset (e.g., TarDataset)
            target_batch_size: target number of PACKED sequences per batch
            max_length: maximum sequence length for packing
            pad_token_id: token id for padding
            image_token_id: token id for image tokens
            max_images_per_pack: max images allowed in one packed sequence
            max_image_tokens_per_pack: max image tokens in one packed sequence
            mini_batch_size: number of samples to fetch per round (controls memory usage)
        """
        self.dataset = dataset
        self.target_batch_size = target_batch_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self.max_images_per_pack = max_images_per_pack
        self.max_image_tokens_per_pack = max_image_tokens_per_pack
        self.mini_batch_size = mini_batch_size
        
        # Statistics (per worker/rank)
        self._total_samples_fetched = 0
        self._total_batches_yielded = 0
        self._pack_stats_list = []  # Track pack utilization statistics
        
    def get_stats(self):
        """Return statistics about samples fetched vs batches yielded"""
        return {
            "samples_fetched": self._total_samples_fetched,
            "batches_yielded": self._total_batches_yielded,
            "avg_samples_per_batch": (
                self._total_samples_fetched / self._total_batches_yielded 
                if self._total_batches_yielded > 0 else 0
            ),
        }
        
    def __iter__(self):
        """Iterate and yield dynamically packed batches using pure incremental packing
        
        Strategy (Most Memory Efficient):
        - Fetch samples one at a time
        - Try to add to existing packs using First-Fit algorithm
        - If no pack can fit, create new pack
        - Continue until we have target_batch_size packs
        
        Advantages:
        - Minimal memory usage: only ~target_batch_size packs in memory
        - No batch accumulation
        - Simplest implementation
        
        Trade-offs:
        - Lower packing quality than batch greedy (70-80% vs 90-95%)
        - No global optimization
        
        Memory usage: ~target_batch_size packs × avg_pack_size × 4MB
                     = 256 packs × 1.5 samples × 4MB = ~1.5GB (acceptable)
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        dataset_iter = iter(self.dataset)
        # print('max_length:', self.max_length)
        while True:
            try:
                # Build packed sequences incrementally
                packed_sequences = []  # List of packs, each pack is a list of prepared samples
                total_samples_fetched = 0
                
                # Phase 1: Create target_batch_size packs
                while len(packed_sequences) < self.target_batch_size:
                    # Fetch one sample
                    sample = next(dataset_iter)
                    total_samples_fetched += 1
                    
                    # Prepare sample metadata
                    seq_len = sample["input_ids"].shape[0]
                    has_images = False
                    multi_modal_inputs = sample.get("multi_modal_inputs", {})
                    if multi_modal_inputs and "pixel_values" in multi_modal_inputs:
                        has_images = True
                        n_image_tokens = (sample["input_ids"] == self.image_token_id).sum().item()
                    else:
                        n_image_tokens = 0
                    
                    prep_sample = {
                        "sample": sample,
                        "seq_len": seq_len,
                        "has_images": has_images,
                        "n_image_tokens": n_image_tokens,
                    }
                    
                    # Try to add to existing packs (First-Fit strategy)
                    added = False
                    for pack in packed_sequences:
                        # Calculate current pack length
                        # Note: Image tokens are already included in seq_len, so we only need to check total length
                        pack_len = sum(s["seq_len"] for s in pack)
                        can_fit = pack_len + seq_len <= self.max_length
                        
                        if can_fit:
                            pack.append(prep_sample)
                            added = True
                            break  # First-Fit: use first pack that fits
                    
                    # If couldn't add to any existing pack, create new pack
                    if not added:
                        packed_sequences.append([prep_sample])
                
                # Phase 2: Continue filling existing packs until they're full
                # Simple strategy: if a sample can't fit into any pack, stop and output current packs
                while True:
                    try:
                        sample = next(dataset_iter)
                    except StopIteration:
                        # Dataset exhausted, break
                        break
                    
                    # Prepare sample metadata
                    seq_len = sample["input_ids"].shape[0]
                    has_images = False
                    multi_modal_inputs = sample.get("multi_modal_inputs", {})
                    if multi_modal_inputs and "pixel_values" in multi_modal_inputs:
                        has_images = True
                        n_image_tokens = (sample["input_ids"] == self.image_token_id).sum().item()
                    else:
                        n_image_tokens = 0
                    
                    prep_sample = {
                        "sample": sample,
                        "seq_len": seq_len,
                        "has_images": has_images,
                        "n_image_tokens": n_image_tokens,
                    }
                    
                    # Try to add to existing packs
                    added = False
                    for pack in packed_sequences:
                        pack_len = sum(s["seq_len"] for s in pack)
                        can_fit = pack_len + prep_sample["seq_len"] <= self.max_length
                        
                        if can_fit:
                            pack.append(prep_sample)
                            added = True
                            break  # First-Fit: use first pack that fits
                    
                    # If current sample can't fit into any pack, stop and output current packs
                    if not added:
                        break
                
                # We now have target_batch_size packs (potentially better filled)
                # Sanity check
                if len(packed_sequences) == 0:
                    if rank == 0:
                        print(f"⚠️ Warning: No packed sequences after fetching {total_samples_fetched} samples, retrying...")
                    continue
                
                # Build batch dict from packed sequences
                batch = self._build_batch_from_packs(packed_sequences)
                
                # Track statistics
                samples_in_batch = sum(len(pack) for pack in packed_sequences)
                self._total_samples_fetched += samples_in_batch
                self._total_batches_yielded += 1
                
                # Add metadata for tracking
                batch["_dynamic_packing_stats"] = {
                    "samples_in_this_batch": samples_in_batch,
                    "packed_sequences": len(packed_sequences),
                }
                
                # Log statistics periodically
                if rank == 0 and self._total_batches_yielded % 100 == 0:
                    avg_samples_per_batch = self._total_samples_fetched / self._total_batches_yielded
                    packing_ratio = avg_samples_per_batch / self.target_batch_size
                    
                    # Calculate pack utilization statistics
                    if hasattr(self, '_pack_stats_list') and len(self._pack_stats_list) > 0:
                        recent_stats = self._pack_stats_list[-len(packed_sequences):]  # Stats for this batch
                        avg_pack_length = sum(s["pack_length"] for s in recent_stats) / len(recent_stats)
                        avg_utilization = sum(s["utilization"] for s in recent_stats) / len(recent_stats)
                        min_utilization = min(s["utilization"] for s in recent_stats)
                        max_utilization = max(s["utilization"] for s in recent_stats)
                        avg_samples_per_pack = sum(s["num_samples"] for s in recent_stats) / len(recent_stats)
                        
                        # Count utilization distribution
                        high_util = sum(1 for s in recent_stats if s["utilization"] > 0.9)
                        medium_util = sum(1 for s in recent_stats if 0.7 <= s["utilization"] <= 0.9)
                        low_util = sum(1 for s in recent_stats if s["utilization"] < 0.7)
                        
                        print(f"📊 Pure Incremental Packing Stats (batch {self._total_batches_yielded}):")
                        print(f"   Total samples fetched: {self._total_samples_fetched}")
                        print(f"   Avg samples per batch: {avg_samples_per_batch:.1f}")
                        print(f"   Packing ratio: {packing_ratio:.2f}x")
                        print(f"\n   📦 Pack Utilization (this batch):")
                        print(f"      Avg pack length: {avg_pack_length:.0f} / {self.max_length} tokens")
                        print(f"      Avg utilization: {avg_utilization:.1%}")
                        print(f"      Min/Max utilization: {min_utilization:.1%} / {max_utilization:.1%}")
                        print(f"      Avg samples per pack: {avg_samples_per_pack:.1f}")
                        print(f"      Utilization distribution:")
                        print(f"        High (>90%): {high_util}/{len(recent_stats)} ({high_util/len(recent_stats)*100:.1f}%)")
                        print(f"        Medium (70-90%): {medium_util}/{len(recent_stats)} ({medium_util/len(recent_stats)*100:.1f}%)")
                        print(f"        Low (<70%): {low_util}/{len(recent_stats)} ({low_util/len(recent_stats)*100:.1f}%)")
                        
                        # Show details for first few packs in this batch
                        if len(recent_stats) <= 10:
                            print(f"\n   📋 Pack Details:")
                            for idx, stats in enumerate(recent_stats):
                                print(f"      Pack {idx+1}: {stats['pack_length']:4d} tokens ({stats['utilization']:.1%}), "
                                      f"{stats['num_samples']} samples")
                        else:
                            # Show first 5 and last 5
                            print(f"\n   📋 Pack Details (first 5 and last 5):")
                            for idx, stats in enumerate(recent_stats[:5]):
                                print(f"      Pack {idx+1}: {stats['pack_length']:4d} tokens ({stats['utilization']:.1%}), "
                                      f"{stats['num_samples']} samples")
                            print(f"      ... ({len(recent_stats)-10} packs omitted) ...")
                            for idx, stats in enumerate(recent_stats[-5:], start=len(recent_stats)-4):
                                print(f"      Pack {idx+1}: {stats['pack_length']:4d} tokens ({stats['utilization']:.1%}), "
                                      f"{stats['num_samples']} samples")
                        
                        # Clear old stats to avoid memory buildup (keep last 1000 packs)
                        if len(self._pack_stats_list) > 1000:
                            self._pack_stats_list = self._pack_stats_list[-500:]
                    else:
                        print(f"📊 Pure Incremental Packing Stats (batch {self._total_batches_yielded}):")
                        print(f"   Total samples fetched: {self._total_samples_fetched}")
                        print(f"   Avg samples per batch: {avg_samples_per_batch:.1f}")
                        print(f"   Packing ratio: {packing_ratio:.2f}x")
                    
                    print(f"   Memory optimized: Using 1D attention mask (saves ~3.8GB)")
                    print(f"   (ratio > 1.0 means we're packing multiple samples together)")
                    if packing_ratio < 1.5:
                        print(f"   ⚠️  Low packing ratio - consider using mini-batch mode for better quality")
                    if hasattr(self, '_pack_stats_list') and len(self._pack_stats_list) > 0:
                        recent_stats = self._pack_stats_list[-len(packed_sequences):]
                        avg_util = sum(s["utilization"] for s in recent_stats) / len(recent_stats)
                        if avg_util < 0.7:
                            print(f"   ⚠️  Low pack utilization ({avg_util:.1%}) - packs are not full enough")
                
                yield batch
                
            except Exception as e:
                if rank == 0:
                    print(f"❌ Error in DynamicPackingDataset: {e}")
                    import traceback
                    traceback.print_exc()
                raise
    
    def _pack_samples(self, samples):
        """
        Pack samples using greedy algorithm.
        Returns list of packs, where each pack is a list of samples.
        """
        if len(samples) == 0:
            return []
        
        # Prepare samples with metadata
        prepared_samples = []
        for sample in samples:
            seq_len = sample["input_ids"].shape[0]
            
            # Check if sample has images
            has_images = False
            multi_modal_inputs = sample.get("multi_modal_inputs", {})
            if multi_modal_inputs and "pixel_values" in multi_modal_inputs:
                has_images = True
                n_image_tokens = (sample["input_ids"] == self.image_token_id).sum().item()
            else:
                n_image_tokens = 0
            
            prepared_samples.append({
                "sample": sample,
                "seq_len": seq_len,
                "has_images": has_images,
                "n_image_tokens": n_image_tokens,
            })
        
        # Sort by sequence length (short first for better packing)
        prepared_samples.sort(key=lambda x: x["seq_len"])
        
        # Greedy packing
        packed_sequences = []
        current_pack = []
        current_length = 0
        
        for prep_sample in prepared_samples:
            can_pack = True
            
            # Rule 1: Don't exceed max_length
            if current_length + prep_sample["seq_len"] > self.max_length:
                can_pack = False
            
            # Rule 2: Limit images per pack
            # if prep_sample["has_images"]:
            #     current_n_images = sum(1 for s in current_pack if s["has_images"])
            #     if current_n_images >= self.max_images_per_pack:
            #         can_pack = False
            
            # # Rule 3: Limit image tokens per pack
            # current_image_tokens = sum(s["n_image_tokens"] for s in current_pack)
            # if current_image_tokens + prep_sample["n_image_tokens"] > self.max_image_tokens_per_pack:
            #     can_pack = False
            
            if can_pack:
                current_pack.append(prep_sample)
                current_length += prep_sample["seq_len"]
            else:
                # Current pack is full, start new pack
                if current_pack:
                    packed_sequences.append(current_pack)
                current_pack = [prep_sample]
                current_length = prep_sample["seq_len"]
        
        # Add last pack
        if current_pack:
            packed_sequences.append(current_pack)
        
        return packed_sequences
    
    def _build_batch_from_packs(self, packed_sequences):
        """
        Build a batch dict from packed sequences.
        Similar to packed_collate_fn but operates on already-packed sequences.
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_loss_mask = []
        batch_has_view_type = []
        batch_view_type = []
        batch_multi_modal_inputs = []
        batch_cu_seqlens = []
        batch_max_seqlen = []
        
        for pack in packed_sequences:
            # Concatenate all samples in this pack
            pack_input_ids = []
            pack_loss_mask = []
            pack_position_ids = []
            cu_seqlens = [0]
            pack_has_view_type = []
            pack_view_type = []
            pack_multi_modal = []
            
            for prep_sample in pack:
                sample = prep_sample["sample"]
                seq_len = prep_sample["seq_len"]
                
                pack_input_ids.append(sample["input_ids"])
                pack_loss_mask.append(sample["loss_mask"])
                
                # Reset position IDs for each sub-sequence
                if sample["position_ids"].dim() == 1:
                    sub_position_ids = torch.arange(seq_len, dtype=sample["position_ids"].dtype)
                elif sample["position_ids"].dim() == 2:
                    # VL model: [4, seq_len]
                    sub_position_ids = torch.zeros((4, seq_len), dtype=sample["position_ids"].dtype)
                    for dim in range(4):
                        sub_position_ids[dim] = torch.arange(seq_len, dtype=sample["position_ids"].dtype)
                else:
                    sub_position_ids = sample["position_ids"]
                
                pack_position_ids.append(sub_position_ids)
                pack_has_view_type.append(sample.get("has_view_type", False))
                pack_view_type.append(sample.get("view_type", "unknown"))
                
                # Multi-modal inputs
                if sample.get("multi_modal_inputs"):
                    pack_multi_modal.append(sample["multi_modal_inputs"])
                
                cu_seqlens.append(cu_seqlens[-1] + seq_len)
            
            # Concatenate
            pack_input_ids = torch.cat(pack_input_ids, dim=0)
            pack_loss_mask = torch.cat(pack_loss_mask, dim=0)
            
            if pack_position_ids[0].dim() == 1:
                pack_position_ids = torch.cat(pack_position_ids, dim=0)
            else:
                pack_position_ids = torch.cat(pack_position_ids, dim=1)
            
            # Build 1D attention mask for Flash Attention varlen mode
            # We don't need 4D block-diagonal mask because Flash Attention varlen
            # uses cu_seqlens to handle sub-sequence boundaries automatically
            pack_seq_len = pack_input_ids.shape[0]
            pack_attention_mask = torch.ones(pack_seq_len, dtype=torch.long)
            # All tokens are valid (1), padding will be added later when batching
            
            # Merge multi_modal_inputs for this pack
            # CRITICAL: When packing multiple samples with images, need to merge their features
            merged_mm = {}
            if pack_multi_modal:
                # Merge pixel_values (concatenate along batch dim)
                all_pixel_values = [mm.get("pixel_values") for mm in pack_multi_modal if mm.get("pixel_values") is not None]
                if all_pixel_values:
                    merged_mm["pixel_values"] = torch.cat(all_pixel_values, dim=0)
                
                # Merge image_grid_thw (concatenate along batch dim)
                all_image_grid_thw = [mm.get("image_grid_thw") for mm in pack_multi_modal if mm.get("image_grid_thw") is not None]
                if all_image_grid_thw:
                    merged_mm["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
            
            # Add to batch
            batch_input_ids.append(pack_input_ids)
            batch_attention_mask.append(pack_attention_mask)
            batch_position_ids.append(pack_position_ids)
            batch_loss_mask.append(pack_loss_mask)
            
            # For has_view_type: save per-subsequence info (not pack-level)
            # This allows subsequence-level KL loss control
            batch_has_view_type.append(pack_has_view_type)  # List of bools, one per subsequence
            batch_view_type.extend(pack_view_type)
            
            # Add merged multi_modal_inputs (one per pack, not per sample)
            batch_multi_modal_inputs.append(merged_mm if merged_mm else {})
            
            batch_cu_seqlens.append(torch.tensor(cu_seqlens, dtype=torch.long))
            batch_max_seqlen.append(max(cu_seqlens[i+1] - cu_seqlens[i] for i in range(len(cu_seqlens)-1)))
            
            # Track pack statistics for utilization analysis
            pack_stats = {
                "pack_length": pack_seq_len,
                "num_samples": len(pack),
                "utilization": pack_seq_len / self.max_length if self.max_length > 0 else 0.0,
            }
            if not hasattr(self, '_pack_stats_list'):
                self._pack_stats_list = []
            self._pack_stats_list.append(pack_stats)
        
        # Pad to same length
        max_seq_len = max(seq.shape[0] for seq in batch_input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_position_ids = []
        padded_loss_masks = []
        
        for i in range(len(batch_input_ids)):
            seq_len = batch_input_ids[i].shape[0]
            pad_len = max_seq_len - seq_len
            
            # Pad input_ids
            padded_input_ids.append(
                torch.cat([batch_input_ids[i], torch.full((pad_len,), self.pad_token_id, dtype=batch_input_ids[i].dtype)])
            )
            
            # Pad attention_mask (1D for varlen mode - saves 3.8GB memory!)
            # Flash Attention varlen uses cu_seqlens for block-diagonal, so we only need
            # a simple 1D mask to mark valid vs padding tokens
            padded_mask = torch.cat([
                batch_attention_mask[i],
                torch.zeros(pad_len, dtype=batch_attention_mask[i].dtype)
            ])
            padded_attention_masks.append(padded_mask)
            
            # Pad position_ids
            if batch_position_ids[i].dim() == 1:
                padded_position_ids.append(
                    torch.cat([batch_position_ids[i], torch.zeros(pad_len, dtype=batch_position_ids[i].dtype)])
                )
            else:
                # VL model: [4, seq_len]
                pad_tensor = torch.zeros((4, pad_len), dtype=batch_position_ids[i].dtype)
                padded_position_ids.append(torch.cat([batch_position_ids[i], pad_tensor], dim=1))
            
            # Pad loss_mask
            padded_loss_masks.append(
                torch.cat([batch_loss_mask[i], torch.zeros(pad_len, dtype=batch_loss_mask[i].dtype)])
            )
        
        # Stack into batch
        result = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),  # [batch, seq] - 1D mask for varlen mode!
            "loss_mask": torch.stack(padded_loss_masks),
            "is_packed": True,
            "cu_seqlens": batch_cu_seqlens,
            "max_seqlen_in_batch": batch_max_seqlen,
        }
        
        # Handle position_ids
        if padded_position_ids[0].dim() == 1:
            result["position_ids"] = torch.stack(padded_position_ids)
        else:
            # VL model: [batch, 4, seq_len] → [4, batch, seq_len]
            stacked = torch.stack(padded_position_ids)  # [batch, 4, seq_len]
            result["position_ids"] = stacked.permute(1, 0, 2).contiguous()  # [4, batch, seq_len] + make contiguous
        
        # Multi-modal inputs
        if batch_multi_modal_inputs:
            result["multi_modal_inputs"] = batch_multi_modal_inputs
        
        # has_view_type
        if batch_has_view_type:
            # For packed sequences, has_view_type is a list of lists (one list per pack, each element is a subsequence)
            # Store as list of lists to preserve subsequence-level information
            result["has_view_type"] = batch_has_view_type  # List of lists: [[bool, bool, ...], ...]

        # Always add view_type to result (even if empty list, for consistency)
        result['view_type'] = batch_view_type if batch_view_type else []
        
        return result
