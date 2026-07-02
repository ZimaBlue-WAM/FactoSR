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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import time
from contextlib import nullcontext
from typing import Optional

import hydra
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.tar_dataset import create_webdataset
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.config.optimizer import build_optimizer
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))
torch_distributed_logger = logging.getLogger("torch.distributed")
torch_distributed_logger.setLevel(logging.ERROR) 


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        processor=None,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        self.processor = processor  # Store processor for VL models
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)

        self.lora = self.config.model.get("lora_adapter_path") is not None or self.config.model.lora_rank > 0

        # Initialize resume-related variables
        self.resume_global_step = 0
        
        # Initialize dynamic packing statistics
        self._total_samples_processed = 0  # Total samples actually fetched and processed
        self._total_batches_processed = 0  # Total batches (iterations)
        self._use_dynamic_packing = False  # Will be set in _build_dataloader

        # Initialize device_name early (needed for teacher model loading)
        self.device_name = self.config.trainer.device

        # Check if distillation is enabled
        self.distillation_config = self.config.model.get("distillation", {})
        self.enable_distillation = self.distillation_config.get("enable", False)
        self.kl_loss_weight = self.distillation_config.get("kl_loss_weight", 0.5)
        self.distillation_temperature = self.distillation_config.get("temperature", 4.0)
        self.use_hidden_states_kl = self.distillation_config.get("use_hidden_states_kl", True)
        self.teacher_model = None
        self._last_kl_loss = None  # For storing KL loss during training step

        # build model
        self._build_model_optimizer()
        
        # Setup attention capture if distillation enabled
        if self.enable_distillation:
            (
                self.student_attn_outputs,
                self.student_attn_handles,
                self.student_num_attn_layers,
            ) = self._register_attention_hooks(self.model, is_teacher=False)
            self.teacher_attn_outputs = None
            self.teacher_attn_handles = []
            self.teacher_num_attn_layers = 0

        # Load teacher model if distillation is enabled
        if self.enable_distillation:
            self._load_teacher_model()

        # Initialize checkpoint manager
        self._init_checkpoint_manager()

        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0, (
            f"train_batch_size ({self.config.data.train_batch_size}) must be divisible by "
            f"micro_batch_size_per_gpu ({self.config.data.micro_batch_size_per_gpu}) to avoid "
            f"gradient accumulation inconsistency across ranks"
        )
        
        # Verify configuration consistency across all ranks
        self._verify_config_consistency()
    
    def _verify_config_consistency(self):
        """Verify critical config values are consistent across all ranks
        
        Inconsistent configs can cause deadlocks in distributed training
        """
        import torch.distributed as dist
        
        rank = self.device_mesh.get_rank()
        
        # Get device name (device_name not initialized yet at this point)
        device_name = get_device_name()
        
        # Check critical config values
        configs_to_check = {
            "train_batch_size": self.config.data.train_batch_size,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "balance_dp_token": int(self.config.data.balance_dp_token),
        }
        
        for config_name, config_value in configs_to_check.items():
            # Convert to tensor
            local_value = torch.tensor(config_value, device=device_name)
            
            # Gather from all ranks
            all_values = [torch.zeros_like(local_value) for _ in range(dist.get_world_size())]
            dist.all_gather(all_values, local_value)
            
            # Check if all values are the same
            all_values_cpu = [v.item() for v in all_values]
            if len(set(all_values_cpu)) > 1:
                if rank == 0:
                    print(f"❌ ERROR: Config '{config_name}' is inconsistent across ranks!")
                    print(f"   Values: {all_values_cpu}")
                    print(f"   This WILL cause deadlock!")
                raise ValueError(f"Config '{config_name}' inconsistent across ranks: {all_values_cpu}")
            elif rank == 0:
                print(f"Config '{config_name}' consistent across all ranks: {config_value}")

    def _is_webdataset(self, dataset):
        """Check if the dataset is a WebDataset (DataPipeline)
        
        The DataPipeline object returned by WebDataset does not have a __len__ method
        """
        return not hasattr(dataset, '__len__')
    
    def _manual_split_batch(self, batch, micro_batch_size):
        """Manually split a batch dict into micro batches
        
        Args:
            batch: dict with tensors of shape [batch_size, ...] or special shapes
            micro_batch_size: size of each micro batch
            
        Returns:
            list of micro batch dicts
        """
        # Get batch size from a standard tensor (input_ids)
        batch_size = batch["input_ids"].shape[0] if "input_ids" in batch else len(next(iter(batch.values())))
        
        # Handle special fields for packed sequences
        is_packed = batch.get("is_packed", False)
        cu_seqlens = batch.get("cu_seqlens", None)
        max_seqlen_in_batch = batch.get("max_seqlen_in_batch", None)
        view_type = batch.get("view_type", None)
        has_view_type = batch.get("has_view_type", None)  # Can be list of lists for packed sequences

        micro_batches = []
        for i in range(0, batch_size, micro_batch_size):
            micro_batch = {}
            for key, value in batch.items():
                # Skip special packed fields (will be handled separately)
                if key in ["is_packed", "cu_seqlens", "max_seqlen_in_batch", "view_type", "has_view_type"]:
                    continue
                    
                if isinstance(value, torch.Tensor):
                    # Special handling for position_ids (Qwen2-VL format)
                    if key == "position_ids" and value.dim() == 3:
                        # Check if it's [4, batch, seq] or [batch, 4, seq]
                        if value.shape[0] == 4 and value.shape[1] == batch_size:
                            # [4, batch, seq] format - slice along dim 1
                            micro_batch[key] = value[:, i:i+micro_batch_size]
                        elif value.shape[1] == 4 and value.shape[0] == batch_size:
                            # [batch, 4, seq] format - slice along dim 0, then permute to [4, micro_batch_size, seq]
                            micro_batch[key] = value[i:i+micro_batch_size].permute(1, 0, 2).contiguous()
                        else:
                            # Fallback: try standard slicing
                            if value.shape[0] == batch_size:
                                micro_batch[key] = value[i:i+micro_batch_size]
                            elif value.shape[1] == batch_size:
                                micro_batch[key] = value[:, i:i+micro_batch_size]
                            else:
                                micro_batch[key] = value
                    # Handle different tensor shapes
                    elif value.dim() == 1 and value.shape[0] == batch_size:
                        # 1D tensor with batch dimension (e.g., has_view_type): [batch] → slice
                        micro_batch[key] = value[i:i+micro_batch_size]
                    elif value.dim() >= 2 and value.shape[0] == batch_size:
                        # Standard: [batch, ...] → slice along dim 0
                        micro_batch[key] = value[i:i+micro_batch_size]
                    elif value.dim() >= 2 and value.shape[1] == batch_size:
                        # Special (e.g., Qwen2-VL position_ids): [4, batch, seq] → slice along dim 1
                        micro_batch[key] = value[:, i:i+micro_batch_size]
                    else:
                        # Other cases, keep as is or slice if possible
                        micro_batch[key] = value
                elif isinstance(value, list):
                    # List (e.g., multi_modal_inputs)
                    micro_batch[key] = value[i:i+micro_batch_size]
                else:
                    micro_batch[key] = value
            
            # Add packed sequence metadata if present
            if is_packed:
                micro_batch["is_packed"] = True
                if cu_seqlens is not None:
                    micro_batch["cu_seqlens"] = cu_seqlens[i:i+micro_batch_size]
                if max_seqlen_in_batch is not None:
                    micro_batch["max_seqlen_in_batch"] = max_seqlen_in_batch[i:i+micro_batch_size]
            
            # Handle has_view_type (can be list of lists for packed sequences)
            if has_view_type is not None:
                if isinstance(has_view_type, list):
                    # For packed sequences: has_view_type is list of lists
                    # Each element is a list of bools (one per subsequence in that pack)
                    micro_batch["has_view_type"] = has_view_type[i:i+micro_batch_size]
                elif isinstance(has_view_type, torch.Tensor):
                    # For non-packed sequences: tensor of bools
                    micro_batch["has_view_type"] = has_view_type[i:i+micro_batch_size]
                else:
                    # Single value or other type
                    micro_batch["has_view_type"] = has_view_type
            
            if view_type is not None:
                if isinstance(view_type, list):
                    micro_batch["view_type"] = view_type[i:i+micro_batch_size]
                else:
                    micro_batch["view_type"] = view_type[i:i+micro_batch_size].cpu().tolist()
            micro_batches.append(micro_batch)
        
        return micro_batches

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        is_webdataset = self._is_webdataset(self.train_dataset)
        
        if is_webdataset:
            
            from torch.utils.data import DataLoader
            self.train_sampler = None
            
            # Check if we should use dynamic packing
            use_packing = getattr(self.train_dataset, 'use_sequence_packing', False)
            
            if use_packing:
                # Wrap with DynamicPackingDataset for optimal throughput
                from verl.utils.dataset.tar_dataset import DynamicPackingDataset
                
                self._use_dynamic_packing = True  # Set flag for statistics tracking
                
                if self.device_mesh.get_rank() == 0:
                    print("🚀 Using Dynamic Packing DataLoader (pure incremental mode)")
                    print(f"   Target batch size per rank: {config.data.train_batch_size}")
                    print(f"   Max length: {config.data.max_length}")
                    print(f"   Mode: One-by-one fetch with First-Fit packing")
                    print(f"   Max image tokens per pack: 6000")
                    print(f"   Memory usage: Minimal (~1.5GB for packs only)")
                    print(f"   Note: Packing quality may be lower than batch mode")
                
                # Store original dataset for validation
                original_train_dataset = self.train_dataset
                
                # Use train_batch_size as target_batch_size for DynamicPackingDataset
                # target_batch_size is the number of packed sequences per GPU per iteration
                # The packs will be split into micro_batches by micro_batch_size_per_gpu
                target_batch_size = config.data.train_batch_size
                
                # Wrap the dataset
                self.train_dataset = DynamicPackingDataset(
                    dataset=original_train_dataset,
                    target_batch_size=target_batch_size,
                    max_length=config.data.max_length,
                    pad_token_id=original_train_dataset.tokenizer.pad_token_id,
                    image_token_id=151655,  # Qwen2-VL image token
                    # max_images_per_pack=2,  # Max 2 images per pack to save memory
                    max_image_tokens_per_pack=6000,  # Max 6000 image tokens per pack
                    # Note: mini_batch_size is not used in pure incremental mode
                )
                
                # DataLoader with batch_size=1 (each iteration returns one pre-packed batch)
                self.train_dataloader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=1,  # Each yield from dataset is already a full batch
                    num_workers=12,  # Use 0 because IterableDataset already uses workers internally
                    pin_memory=True,
                    prefetch_factor=None,  # Only valid when num_workers > 0
                    persistent_workers=False,
                    drop_last=False,
                    pin_memory_device=device_name,
                    collate_fn=lambda x: x[0],  # Just unwrap the single item
                )
            else:
                # Original path: use standard collate_fn
                train_collate_fn = None
                if hasattr(self.train_dataset, 'get_collate_fn'):
                    train_collate_fn = self.train_dataset.get_collate_fn()
                    if self.device_mesh.get_rank() == 0:
                        print("Using custom collate_fn for dynamic padding")
                
                self.train_dataloader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=config.data.train_batch_size,
                    num_workers=8,
                    pin_memory=True,
                    prefetch_factor=4,  # Prefetch 4 batches per worker (trade CPU RAM for speed)
                    persistent_workers=True,  # Keep workers alive between epochs (faster)
                    drop_last=True,
                    pin_memory_device=device_name,
                    collate_fn=train_collate_fn,
                )
            
            # Validation dataloader (always use standard packing, no dynamic packing for val)
            val_collate_fn = None
            if hasattr(self.val_dataset, 'get_collate_fn'):
                val_collate_fn = self.val_dataset.get_collate_fn()
            
            self.val_sampler = None
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=config.data.micro_batch_size_per_gpu,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=2,  # Smaller prefetch for validation
                persistent_workers=True,
                drop_last=True,
                pin_memory_device=device_name,
                collate_fn=val_collate_fn,
            )
        else:
            # Normal dataset: use DistributedSampler for shuffle and distribution
            
            self.train_sampler = DistributedSampler(
                self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
            )
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=config.data.train_batch_size,
                sampler=self.train_sampler,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
                pin_memory_device=device_name,
            )

            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=config.data.micro_batch_size_per_gpu,
                sampler=self.val_sampler,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
                pin_memory_device=device_name,
            )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            # Load model based on model type
            model_type = config.model_type if hasattr(config, 'model_type') else 'unknown'
            if self.device_mesh.get_rank() == 0:
                print(f"Loading model type: {model_type}")
            
            # Check if it's a VL model that needs special handling
            if 'qwen2_5_vl' in model_type.lower() or 'qwen2_vl' in model_type.lower():
                # Qwen2.5-VL or Qwen2-VL
                from transformers import Qwen2_5_VLForConditionalGeneration
                if self.device_mesh.get_rank() == 0:
                    print(f"Loading Qwen2.5-VL using Qwen2_5_VLForConditionalGeneration")
                self.model: PreTrainedModel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                )
            elif 'qwen3_vl' in model_type.lower():
                # Qwen3-VL
                from transformers import Qwen3VLForConditionalGeneration
                if self.device_mesh.get_rank() == 0:
                    print(f"Loading Qwen3-VL using Qwen3VLForConditionalGeneration")
                self.model: PreTrainedModel = Qwen3VLForConditionalGeneration.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                )
            else:
                # Standard CausalLM models
                self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                )

            # Apply monkey patch for:
            # 1. Sequence parallel support (if ulysses > 1)
            # 2. Remove padding support (if use_remove_padding = True)
            # 3. VL models: ALWAYS apply to support sequence packing and mixed batches
            model_type = config.model_type if hasattr(config, 'model_type') else 'unknown'
            is_vl_model = 'qwen2_5_vl' in model_type.lower() or 'qwen2_vl' in model_type.lower() or 'qwen3_vl' in model_type.lower()
            
            # For VL models, always enable use_remove_padding to get the full monkey patch
            # This ensures our modified qwen2_vl_attn_forward is applied (supports sequence packing)
            effective_use_remove_padding = self.use_remove_padding or is_vl_model
            
            if effective_use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(
                    model=self.model, 
                    ulysses_sp_size=self.config.ulysses_sequence_parallel_size,
                    use_remove_padding=effective_use_remove_padding  # Pass to monkey_patch
                )
                
                if is_vl_model and self.device_mesh.get_rank() == 0:
                    print("✅ Applied monkey patch for VL model (sequence packing + mixed batches support)")

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.lora:
                self.model.enable_input_require_grads()

                lora_adapter_path = self.config.model.get("lora_adapter_path")
                if lora_adapter_path is not None:
                    from peft import PeftModel

                    print(f"Loading pre-trained LoRA adapter for sft from: {lora_adapter_path}")

                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.use_shm)

                    self.model = PeftModel.from_pretrained(self.model, local_adapter_path, is_trainable=True)
                    peft_config = self.model.peft_config["default"]
                    # Ensure task_type is TaskType enum, not string
                    if isinstance(peft_config.task_type, str):
                        peft_config.task_type = TaskType.CAUSAL_LM
                else:
                    # Convert config to regular Python types before creating PEFT model
                    lora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "bias": "none",
                    }
                    self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.lora,
        )

        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = self._build_optimizer_with_different_lr()

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        # Calculate total_steps and num_warmup_steps
        # First read from configuration, if not then calculate from dataloader
        if hasattr(self.config.trainer, 'total_training_steps') and self.config.trainer.total_training_steps is not None:
            # Read total_training_steps directly from configuration
            self.total_steps = self.config.trainer.total_training_steps
            if self.device_mesh.get_rank() == 0:
                print(f"Using total_training_steps from configuration = {self.total_steps}")
        else:
            # 从 dataloader 计算
            is_webdataset = self._is_webdataset(self.train_dataset)
            if is_webdataset:
                raise ValueError(
                    "When using WebDataset (tar files), 'trainer.total_training_steps' must be specified in the configuration.\n"
                    "Estimated formula: total_training_steps = (total number of samples / train_batch_size) * total_epochs / world_size"
                )
            self.steps_per_epoch = len(self.train_dataloader)
            self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
            if self.device_mesh.get_rank() == 0:
                print(
                    f"Calculate from dataloader: steps_per_epoch={self.steps_per_epoch}, "
                    f"total_epochs={self.config.trainer.total_epochs}, "
                    f"total_steps={self.total_steps}"
                )

        # Calculate warmup steps: first read from configuration, if not then calculate from ratio
        if hasattr(self.config.optim, 'num_warmup_steps') and self.config.optim.num_warmup_steps is not None:
            num_warmup_steps = self.config.optim.num_warmup_steps
            if self.device_mesh.get_rank() == 0:
                print(f"Using num_warmup_steps from configuration = {num_warmup_steps}")
        else:
            num_warmup_steps = int(self.total_steps * self.config.optim.lr_warmup_steps_ratio)
            if self.device_mesh.get_rank() == 0:
                print(f"Calculate warmup steps from ratio: {num_warmup_steps} ({self.config.optim.lr_warmup_steps_ratio} * {self.total_steps})")

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _build_optimizer_with_different_lr(self):
        """Build optimizer with different learning rates for ViT and LLM components.
        
        For VL models (Qwen2.5-VL, Qwen2-VL, Qwen3-VL), ViT parameters use a lower learning rate
        (typically 1/10 of LLM learning rate) to stabilize training.
        
        Returns:
            Optimizer instance with parameter groups
        """
        # Check if this is a VL model
        model_type = self.model_config.model_type if hasattr(self.model_config, 'model_type') else 'unknown'
        is_vl_model = 'qwen2_5_vl' in model_type.lower() or 'qwen2_vl' in model_type.lower() or 'qwen3_vl' in model_type.lower()
        
        # Get base learning rate from config
        base_lr = self.config.optim.lr
        
        # Get ViT learning rate from config, or use 1/10 of base LR as default
        vit_lr_ratio = self.config.optim.get("vit_lr_ratio", 0.1)  # Default: 1/10
        vit_lr = base_lr * vit_lr_ratio
        
        if is_vl_model:
            # Separate ViT and LLM parameters
            vit_params = []
            llm_params = []
            
            # Get all model parameters from FSDP model
            # FSDP wraps the model, so we need to iterate through named_parameters
            # which handles FSDP's parameter naming (may include _fsdp_wrapped_module prefix)
            all_params_dict = {}
            for name, param in self.fsdp_model.named_parameters():
                if param.requires_grad:
                    all_params_dict[name] = param
                    # Check if parameter belongs to visual module
                    # FSDP may add prefixes like '_fsdp_wrapped_module.', so we check for 'visual' anywhere in the name
                    if 'visual' in name.lower():
                        vit_params.append(param)
                    else:
                        llm_params.append(param)
            
            # If we didn't find ViT params by name pattern, try to access the underlying model
            if len(vit_params) == 0:
                if self.device_mesh.get_rank() == 0:
                    print("⚠️  Warning: Could not find ViT parameters by name pattern, trying direct model access...")
                
                # Try to access underlying model (for FSDP1, it's wrapped in _fsdp_wrapped_module)
                model = self.fsdp_model
                if hasattr(model, '_fsdp_wrapped_module'):
                    model = model._fsdp_wrapped_module
                elif hasattr(model, 'module'):
                    model = model.module
                
                # For VL models, visual encoder is typically at model.visual or model.model.visual
                visual_module = None
                if hasattr(model, 'visual'):
                    visual_module = model.visual
                elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
                    visual_module = model.model.visual
                
                if visual_module is not None:
                    try:
                        # Get ViT parameters directly from visual module
                        vit_param_ids = set()
                        for param in visual_module.parameters():
                            if param.requires_grad:
                                vit_param_ids.add(id(param))
                        
                        # Re-separate all parameters using parameter IDs
                        vit_params = []
                        llm_params = []
                        for name, param in self.fsdp_model.named_parameters():
                            if param.requires_grad:
                                if id(param) in vit_param_ids:
                                    vit_params.append(param)
                                else:
                                    llm_params.append(param)
                    except Exception as e:
                        if self.device_mesh.get_rank() == 0:
                            print(f"⚠️  Warning: Failed to separate ViT params: {e}, using uniform LR")
                        # Fallback: use uniform learning rate
                        return build_optimizer(self.fsdp_model.parameters(), self.config.optim)
                else:
                    if self.device_mesh.get_rank() == 0:
                        print("⚠️  Warning: Visual module not found, using uniform LR")
                    # Fallback: use uniform learning rate
                    return build_optimizer(self.fsdp_model.parameters(), self.config.optim)
            
            if len(vit_params) > 0 and len(llm_params) > 0:
                
                if self.device_mesh.get_rank() == 0:
                    num_vit_params = sum(p.numel() for p in vit_params)
                    num_llm_params = sum(p.numel() for p in llm_params)
                    total_params = num_vit_params + num_llm_params
                    print(f"📊 Parameter separation for VL model:")
                    print(f"   ViT parameters: {num_vit_params:,} ({num_vit_params/total_params*100:.1f}%)")
                    print(f"   LLM parameters: {num_llm_params:,} ({num_llm_params/total_params*100:.1f}%)")
                    print(f"   ViT learning rate: {vit_lr:.2e} (ratio: {vit_lr_ratio})")
                    print(f"   LLM learning rate: {base_lr:.2e}")
                
                # Create parameter groups with different learning rates
                param_groups = [
                    {
                        "params": vit_params,
                        "lr": vit_lr,
                        "weight_decay": self.config.optim.weight_decay,
                    },
                    {
                        "params": llm_params,
                        "lr": base_lr,
                        "weight_decay": self.config.optim.weight_decay,
                    },
                ]
                
                # Build optimizer with parameter groups
                import importlib
                optimizer_args = {
                    "weight_decay": self.config.optim.weight_decay,
                }
                
                optimizer_name_lower = self.config.optim.optimizer.lower()
                if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
                    optimizer_args["betas"] = self.config.optim.betas
                
                if self.config.optim.get("override_optimizer_config") is not None:
                    optimizer_args.update(self.config.optim.override_optimizer_config)
                
                try:
                    module = importlib.import_module(self.config.optim.optimizer_impl)
                    optimizer_cls = getattr(module, self.config.optim.optimizer)
                except ImportError as e:
                    raise ImportError(
                        f"Failed to import module '{self.config.optim.optimizer_impl}'. Make sure the package is installed. Error: {e}"
                    ) from e
                except AttributeError as e:
                    raise AttributeError(
                        f"Optimizer '{self.config.optim.optimizer}' not found in module '{self.config.optim.optimizer_impl}'. "
                        f"Available optimizers: {dir(module)}"
                    ) from e
                
                return optimizer_cls(param_groups, **optimizer_args)
            else:
                if self.device_mesh.get_rank() == 0:
                    print("⚠️  Warning: Could not separate ViT and LLM parameters, using uniform LR")
                # Fallback: use uniform learning rate
                return build_optimizer(self.fsdp_model.parameters(), self.config.optim)
        else:
            # Not a VL model, use standard optimizer
            return build_optimizer(self.fsdp_model.parameters(), self.config.optim)
    def _get_attention_modules(self, model: nn.Module) -> list[nn.Module]:
        modules: list[nn.Module] = []

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for layer in model.model.layers:
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
                if attn is not None:
                    modules.append(attn)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            for layer in model.transformer.layers:
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
                if attn is not None:
                    modules.append(attn)
        else:
            # Fallback: collect modules whose class name includes "attention"
            for module in model.modules():
                name = module.__class__.__name__.lower()
                if name.endswith("attention") or "attention" in name:
                    modules.append(module)
            # Keep insertion order but ensure uniqueness
            seen = set()
            unique_modules = []
            for module in modules:
                ident = id(module)
                if ident not in seen:
                    unique_modules.append(module)
                    seen.add(ident)
            modules = unique_modules

        if not modules:
            raise ValueError("Could not locate attention modules for distillation hooks")

        return modules

    def _register_attention_hooks(
        self,
        model: nn.Module,
        is_teacher: bool,
    ) -> tuple[list[Optional[torch.Tensor]], list[torch.utils.hooks.RemovableHandle], int]:
        if not self.enable_distillation:
            return [], [], 0

        attention_modules = self._get_attention_modules(model)
        outputs: list[Optional[torch.Tensor]] = [None] * len(attention_modules)
        handles: list[torch.utils.hooks.RemovableHandle] = []

        for idx, module in enumerate(attention_modules):

            def hook(_, __, output, idx=idx, is_teacher=is_teacher):
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output
                if attn_output is None:
                    outputs[idx] = None
                else:
                    outputs[idx] = attn_output.detach() if is_teacher else attn_output

            handles.append(module.register_forward_hook(hook))

        return outputs, handles, len(attention_modules)

    def _reset_attn_cache(self, is_teacher: bool) -> None:
        if not self.enable_distillation:
            return
        storage = self.teacher_attn_outputs if is_teacher else self.student_attn_outputs
        if storage is None:
            return
        for idx in range(len(storage)):
            storage[idx] = None

    def _collect_attention_outputs(self, is_teacher: bool) -> list[Optional[torch.Tensor]]:
        if not self.enable_distillation:
            return []
        storage = self.teacher_attn_outputs if is_teacher else self.student_attn_outputs
        if storage is None:
            return []
        outputs: list[Optional[torch.Tensor]] = []
        for idx, value in enumerate(storage):
            outputs.append(value)
            storage[idx] = None
        return outputs

    def _load_teacher_model(self):
        """Load teacher model for knowledge distillation"""
        teacher_model_path = self.distillation_config.get("teacher_model_path")
        if teacher_model_path is None:
            raise ValueError("teacher_model_path must be specified when distillation is enabled")
        
        if self.device_mesh.get_rank() == 0:
            print(f"Loading teacher model from: {teacher_model_path}")
        
        from verl.utils.fs import copy_to_local
        from transformers import AutoConfig, AutoModelForCausalLM
        
        local_teacher_path = copy_to_local(src=teacher_model_path, verbose=True)
        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        # Load teacher config
        teacher_config = AutoConfig.from_pretrained(local_teacher_path, trust_remote_code=trust_remote_code)
        model_type = teacher_config.model_type if hasattr(teacher_config, 'model_type') else 'unknown'
        
        # Load teacher model (same logic as student model)
        if 'qwen2_5_vl' in model_type.lower() or 'qwen2_vl' in model_type.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration
            if self.device_mesh.get_rank() == 0:
                print(f"Loading teacher Qwen2.5-VL model")
            self.teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local_teacher_path,
                config=teacher_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
        elif 'qwen3_vl' in model_type.lower():
            from transformers import Qwen3VLForConditionalGeneration
            if self.device_mesh.get_rank() == 0:
                print(f"Loading teacher Qwen3-VL model")
            self.teacher_model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_teacher_path,
                config=teacher_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
        else:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                local_teacher_path,
                config=teacher_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
        
        # Set teacher model to eval mode and disable gradients
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Move teacher model to device
        self.teacher_model = self.teacher_model.to(self.device_name)

        (
            self.teacher_attn_outputs,
            self.teacher_attn_handles,
            self.teacher_num_attn_layers,
        ) = self._register_attention_hooks(self.teacher_model, is_teacher=True)
        
        if self.device_mesh.get_rank() == 0:
            kl_mode = "Hidden States KL (All Layers)" if self.use_hidden_states_kl else "Attention KL"
            print(f"Teacher model loaded successfully.")
            print(f"   Distillation Mode: {kl_mode}")
            print(f"   Loss Weight: {self.kl_loss_weight}")
            print(f"   Temperature: {self.distillation_temperature}")
            if self.use_hidden_states_kl:
                print(f"   Computing KL directly on hidden features (no lm_head)")

    def _compute_hidden_states_kl_loss(self, student_hidden_states, teacher_hidden_states, attention_mask, loss_mask=None, sample_mask=None, token_mask=None, is_packed=False, cu_seqlens=None):
        """Compute KL divergence directly on ALL hidden states layers (treating hidden_dim as distribution)
        
        Strategy: Treat each token's hidden_dim as a distribution over features,
                  compute KL divergence between student and teacher distributions.
                  Normalize per sample/subsequence first, then average (consistent with CE loss).
        
        Args:
            student_hidden_states: Student model's hidden states (tuple of all layers)
            teacher_hidden_states: Teacher model's hidden states (tuple of all layers)
            attention_mask: Attention mask for valid tokens
            loss_mask: Optional mask for answer-only tokens
            sample_mask: Sample-level mask to skip certain samples (will be expanded to token-level)
            token_mask: Token-level mask to skip certain tokens (takes precedence over sample_mask)
            is_packed: Whether sequences are packed
            cu_seqlens: Cumulative sequence lengths for packed sequences (list of tensors)
        
        Returns:
            kl_loss: Scalar KL divergence loss (averaged across all layers, normalized per sample/subsequence)
        """
        import torch.nn.functional as F
        
        # Convert to list if needed
        if isinstance(student_hidden_states, (list, tuple)):
            student_layers = list(student_hidden_states)
        else:
            student_layers = [student_hidden_states]
        
        if isinstance(teacher_hidden_states, (list, tuple)):
            teacher_layers = list(teacher_hidden_states)
        else:
            teacher_layers = [teacher_hidden_states]
        
        # Ensure same number of layers
        num_layers = min(len(student_layers), len(teacher_layers))
        if num_layers == 0:
            raise ValueError("No hidden states layers found")
        
        if self.device_mesh.get_rank() == 0:
            # Only print once on first call
            if not hasattr(self, '_printed_kl_layers'):
                print(f"Computing Hidden States KL on ALL {num_layers} layers (direct on features)")
                self._printed_kl_layers = True
        
        # Build token mask once (used for all layers)
        # Priority: token_mask > sample_mask > attention_mask
        final_token_mask = None
        
        if token_mask is not None:
            # Direct token-level mask provided (for packed sequences with subsequence-level control)
            final_token_mask = token_mask.float()
        else:
            # Build from attention_mask and sample_mask
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    # Packed sequences: extract diagonal
                    token_mask_from_attn = []
                    for i in range(attention_mask.shape[0]):
                        diagonal = attention_mask[i, 0].diagonal()
                        mask_1d = (diagonal > float('-inf')).float()
                        token_mask_from_attn.append(mask_1d)
                    final_token_mask = torch.stack(token_mask_from_attn, dim=0)
                elif attention_mask.dim() == 2:
                    # 1D mask: [batch, seq]
                    final_token_mask = attention_mask.float()
                else:
                    final_token_mask = attention_mask.float()
            
            # Apply sample_mask (expand to token-level)
            if sample_mask is not None and final_token_mask is not None:
                sample_mask_expanded = sample_mask.unsqueeze(1).float()
                final_token_mask = final_token_mask * sample_mask_expanded
        
        token_mask = final_token_mask  # Use final_token_mask for consistency with rest of code
        
        # Temperature for scaling
        temperature = max(self.distillation_temperature or 1.0, 1e-6)
        
        # Compute KL for each layer and average
        layer_losses = []
        
        for layer_idx in range(num_layers):
            student_h = student_layers[layer_idx]
            teacher_h = teacher_layers[layer_idx]
            
            if student_h is None or teacher_h is None:
                continue
            
            # Ensure float for numerical stability
            student_h = student_h.float()
            teacher_h = teacher_h.float()
            
            # Treat hidden_dim as a distribution dimension
            # Apply temperature scaling and softmax over hidden_dim
            # Shape: [batch, seq, hidden_dim]
            student_log_probs = F.log_softmax(student_h / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_h / temperature, dim=-1)
            
            # Compute KL divergence: KL(teacher || student)
            # kl_div with log_target=False expects: (log(student), teacher)
            kl = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='none'
            ).sum(dim=-1)  # Sum over hidden_dim → [batch, seq]
            
            # Apply temperature scaling
            kl = kl * (temperature ** 2)
            
            # === Sample-level loss normalization (consistent with CE loss) ===
            batch_size = kl.shape[0]
            
            if token_mask is not None:
                # Adjust mask shape if needed
                mask_to_use = token_mask
                if mask_to_use.shape[1] != kl.shape[1]:
                    if mask_to_use.shape[1] > kl.shape[1]:
                        mask_to_use = mask_to_use[:, :kl.shape[1]]
                    else:
                        pad_len = kl.shape[1] - mask_to_use.shape[1]
                        mask_to_use = F.pad(mask_to_use, (0, pad_len))
                
                kl_masked = kl * mask_to_use  # [batch, seq]
                
                if is_packed and cu_seqlens is not None:
                    # Pack模式：按subsequence normalize（每个subsequence是一个sample）
                    subsequence_losses = []
                    
                    for pack_idx in range(batch_size):
                        pack_kl = kl_masked[pack_idx]  # [seq_len]
                        pack_mask = mask_to_use[pack_idx]  # [seq_len]
                        pack_cu_seqlens = cu_seqlens[pack_idx]  # Tensor: [0, seq1_len, seq1_len+seq2_len, ...]
                        pack_cu_seqlens_list = pack_cu_seqlens.cpu().tolist()
                        
                        # 对每个subsequence normalize
                        for subseq_idx in range(len(pack_cu_seqlens_list) - 1):
                            subseq_start = pack_cu_seqlens_list[subseq_idx]
                            subseq_end = pack_cu_seqlens_list[subseq_idx + 1]
                            
                            # 提取该subsequence的KL loss和mask
                            subseq_kl = pack_kl[subseq_start:subseq_end]
                            subseq_mask = pack_mask[subseq_start:subseq_end]
                            
                            # 计算该subsequence的KL sum和token count
                            subseq_kl_sum = subseq_kl.sum()
                            subseq_token_count = subseq_mask.sum()
                            
                            # 只处理有有效token的subsequence
                            if subseq_token_count > 0:
                                subseq_kl_normalized = subseq_kl_sum / (subseq_token_count + 1e-8)
                                subsequence_losses.append(subseq_kl_normalized)
                    
                    # 对所有subsequence的normalized loss求平均
                    if subsequence_losses:
                        layer_loss = torch.stack(subsequence_losses).mean()
                    else:
                        layer_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device_name)
                else:
                    # 非Pack模式：按sample normalize
                    # kl_masked shape: [batch_size, seq_len]
                    # mask_to_use shape: [batch_size, seq_len]
                    per_sample_kl_sum = kl_masked.sum(dim=1)  # [batch_size]
                    valid_token_counts = mask_to_use.sum(dim=1)  # [batch_size]
                    
                    per_sample_kl_normalized = per_sample_kl_sum / (valid_token_counts + 1e-8)
                    # 只对有效samples求平均
                    valid_samples_mask = valid_token_counts > 0
                    if valid_samples_mask.any():
                        layer_loss = per_sample_kl_normalized[valid_samples_mask].mean()
                    else:
                        layer_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device_name)
            else:
                # No mask: fallback to token-level average
                layer_loss = kl.mean()
            
            layer_losses.append(layer_loss)
        
        # Average across all layers
        if len(layer_losses) == 0:
            return torch.tensor(0.0, device=self.device_name, requires_grad=True)
        
        return sum(layer_losses) / len(layer_losses)
    
    def _compute_attention_kl_loss(self, student_attentions, teacher_attentions, attention_mask, loss_mask=None, sample_mask=None, token_mask=None, cu_seqlens=None, is_packed=False):
        """Compute temperature-scaled KL divergence between student and teacher attention features.
        
        Args:
            student_attentions: List of attention outputs from student model
            teacher_attentions: List of attention outputs from teacher model
            attention_mask: Attention mask for valid tokens
            loss_mask: (unused) Loss mask for answer tokens
            sample_mask: [batch_size] Boolean tensor indicating which samples should compute KL loss
                        True = compute KL, False = skip KL (will be expanded to token-level)
            token_mask: [batch_size, seq_len] Boolean tensor indicating which tokens should compute KL loss
                        True = compute KL, False = skip KL (takes precedence over sample_mask)
            cu_seqlens: For packed sequences, cumulative sequence lengths [batch_size, num_seqs_per_pack]
            is_packed: Whether the sequences are packed
        """
        import torch.nn.functional as F

        if isinstance(student_attentions, (list, tuple)):
            student_attn_layers = student_attentions
        else:
            student_attn_layers = [student_attentions]

        if isinstance(teacher_attentions, (list, tuple)):
            teacher_attn_layers = teacher_attentions
        else:
            teacher_attn_layers = [teacher_attentions]

        num_layers = min(len(student_attn_layers), len(teacher_attn_layers))
        if num_layers == 0:
            raise ValueError("No attention layers found")

        temperature = max(self.distillation_temperature or 1.0, 1e-6)

        # Build token mask: Priority: token_mask > sample_mask > attention_mask
        final_token_mask = None
        
        if token_mask is not None:
            # Direct token-level mask provided (for packed sequences with subsequence-level control)
            final_token_mask = token_mask.float()
        else:
            # Build from attention_mask and sample_mask
            if attention_mask is not None:
                # Handle different attention mask formats
                if attention_mask.dim() == 4:
                    # Packed sequences: [batch, 1, seq_len, seq_len]
                    # Extract diagonal (valid tokens): values > -inf are valid
                    token_mask_from_attn = []
                    for i in range(attention_mask.shape[0]):
                        diagonal = attention_mask[i, 0].diagonal()
                        # Convert: -inf → 0 (invalid), 0.0 → 1 (valid)
                        mask_1d = (diagonal > float('-inf')).float()
                        token_mask_from_attn.append(mask_1d)
                    final_token_mask = torch.stack(token_mask_from_attn, dim=0)  # [batch, seq_len]
                else:
                    # Standard: [batch, seq_len]
                    final_token_mask = attention_mask.float()
            
            # Apply sample_mask if provided (sample-level KL loss control)
            if sample_mask is not None and final_token_mask is not None:
                # sample_mask: [batch_size], token_mask: [batch_size, seq_len]
                # Expand sample_mask to match token_mask shape
                sample_mask_expanded = sample_mask.unsqueeze(1).float()  # [batch_size, 1]
                final_token_mask = final_token_mask * sample_mask_expanded  # [batch_size, seq_len]
        
        token_mask = final_token_mask  # Use final_token_mask for consistency with rest of code

        def normalize(tensor: torch.Tensor) -> torch.Tensor:
            mean = tensor.mean(dim=-1, keepdim=True)
            std = tensor.std(dim=-1, keepdim=True) + 1e-6
            return (tensor - mean) / std

        layer_losses = []
        for layer_idx in range(num_layers):
            student_attn = student_attn_layers[layer_idx]
            teacher_attn = teacher_attn_layers[layer_idx]

            if student_attn is None or teacher_attn is None:
                continue

            student_attn = student_attn.float()
            teacher_attn = teacher_attn.float()

            # Get attention weights shape: should be [batch, num_heads, seq_len, seq_len]
            # Handle different possible shapes
            if student_attn.dim() == 4:
                batch_size = student_attn.shape[0]
                num_heads = student_attn.shape[1]
                seq_len = student_attn.shape[2]
            elif student_attn.dim() == 3:
                # Fallback: [batch, seq_len, seq_len] - assume single head
                batch_size = student_attn.shape[0]
                num_heads = 1
                seq_len = student_attn.shape[1]
                student_attn = student_attn.unsqueeze(1)  # Add head dimension
                teacher_attn = teacher_attn.unsqueeze(1)
            else:
                # Unsupported shape, fall back to original implementation
                student_norm = normalize(student_attn)
                teacher_norm = normalize(teacher_attn)
                student_flat = student_norm.reshape(-1, student_norm.shape[-1])
                teacher_flat = teacher_norm.reshape(-1, teacher_norm.shape[-1])
                
                if token_mask is not None:
                    mask = token_mask.reshape(-1) > 0
                    if mask.sum() > 0 and mask.numel() <= student_flat.size(0):
                        if mask.numel() < student_flat.size(0):
                            pad_len = student_flat.size(0) - mask.numel()
                            mask = torch.nn.functional.pad(mask, (0, pad_len))
                        student_flat = student_flat[mask]
                        teacher_flat = teacher_flat[mask]
                
                if student_flat.numel() > 0:
                    student_log_probs = F.log_softmax(student_flat / temperature, dim=-1)
                    teacher_probs = F.softmax(teacher_flat / temperature, dim=-1)
                    layer_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                    layer_losses.append(layer_loss)
                continue

            student_norm = normalize(student_attn)
            teacher_norm = normalize(teacher_attn)

            # For packed sequences, compute KL per subsequence to avoid cross-subsequence comparisons
            if is_packed and cu_seqlens is not None:
                # Packed sequences: compute KL per subsequence
                subsequence_losses = []
                
                for pack_idx in range(batch_size):
                    pack_cu_seqlens = cu_seqlens[pack_idx]  # [0, seq1_len, seq1_len+seq2_len, ...]
                    pack_cu_seqlens_list = pack_cu_seqlens.cpu().tolist()
                    
                    # Extract attention weights for this pack: [num_heads, seq_len, seq_len]
                    pack_student = student_norm[pack_idx]  # [num_heads, seq_len, seq_len]
                    pack_teacher = teacher_norm[pack_idx]  # [num_heads, seq_len, seq_len]
                    
                    # Get token mask for this pack
                    pack_token_mask = None
                    if token_mask is not None:
                        pack_token_mask = token_mask[pack_idx]  # [seq_len]
                    
                    # Process each subsequence in this pack
                    for subseq_idx in range(len(pack_cu_seqlens_list) - 1):
                        subseq_start = pack_cu_seqlens_list[subseq_idx]
                        subseq_end = pack_cu_seqlens_list[subseq_idx + 1]
                        subseq_len = subseq_end - subseq_start
                        
                        if subseq_len == 0:
                            continue
                        
                        # Extract subsequence attention weights: [num_heads, subseq_len, subseq_len]
                        # For block-diagonal attention, only extract the relevant block
                        # Ensure we don't exceed the actual sequence length
                        actual_seq_len = pack_student.shape[2]
                        subseq_end_clamped = min(subseq_end, actual_seq_len)
                        subseq_start_clamped = min(subseq_start, actual_seq_len)
                        
                        if subseq_start_clamped >= subseq_end_clamped:
                            continue
                        
                        subseq_student = pack_student[:, subseq_start_clamped:subseq_end_clamped, subseq_start_clamped:subseq_end_clamped]  # [num_heads, subseq_len, subseq_len]
                        subseq_teacher = pack_teacher[:, subseq_start_clamped:subseq_end_clamped, subseq_start_clamped:subseq_end_clamped]  # [num_heads, subseq_len, subseq_len]
                        
                        # Get actual shapes after extraction (must match between attention weights and mask)
                        actual_q_len = subseq_student.shape[1]  # Query length
                        actual_k_len = subseq_student.shape[2]  # Key length (may differ from query length)
                        actual_subseq_len = actual_q_len  # Use query length as reference
                        
                        # Apply token mask if provided (mask query tokens)
                        if pack_token_mask is not None:
                            # CRITICAL: Use the same clamped indices for mask extraction to ensure 1-to-1 correspondence
                            # Extract mask using the same indices as attention weights
                            mask_end_clamped = min(subseq_end_clamped, pack_token_mask.shape[0])
                            mask_start_clamped = min(subseq_start_clamped, pack_token_mask.shape[0])
                            
                            if mask_start_clamped < mask_end_clamped:
                                subseq_token_mask = pack_token_mask[mask_start_clamped:mask_end_clamped]  # Extract using clamped indices
                                
                                # Ensure mask length exactly matches actual_q_len (query length)
                                if subseq_token_mask.shape[0] < actual_q_len:
                                    # Pad with False if mask is shorter (shouldn't happen if indices match)
                                    pad_len = actual_q_len - subseq_token_mask.shape[0]
                                    subseq_token_mask = torch.cat([
                                        subseq_token_mask, 
                                        torch.zeros(pad_len, dtype=subseq_token_mask.dtype, device=subseq_token_mask.device)
                                    ])
                                elif subseq_token_mask.shape[0] > actual_q_len:
                                    # Truncate if mask is longer (shouldn't happen if indices match)
                                    subseq_token_mask = subseq_token_mask[:actual_q_len]
                            else:
                                continue
                            
                            # Convert to bool if needed
                            if subseq_token_mask.dtype != torch.bool:
                                subseq_token_mask = subseq_token_mask.bool()
                            if not subseq_token_mask.any():
                                continue  # Skip if no valid tokens
                            
                            # Flatten attention weights: [num_heads, actual_q_len, actual_k_len] -> [num_heads, actual_q_len*actual_k_len]
                            subseq_student_flat = subseq_student.reshape(num_heads, -1)  # [num_heads, actual_q_len*actual_k_len]
                            subseq_teacher_flat = subseq_teacher.reshape(num_heads, -1)  # [num_heads, actual_q_len*actual_k_len]
                            
                            # Verify mask length matches query length (should always be true now)
                            assert subseq_token_mask.shape[0] == actual_q_len, \
                                f"Mask length {subseq_token_mask.shape[0]} != query length {actual_q_len}"
                            
                            # Create a 2D mask: [actual_q_len, actual_k_len] where both query and key are valid
                            # The mask ensures 1-to-1 correspondence: attn_mask_2d[i, j] corresponds to 
                            # subseq_student[:, i, j] (attention weight from query i to key j)
                            if actual_k_len == actual_q_len:
                                # Symmetric attention: use same mask for query and key
                                query_mask = subseq_token_mask.unsqueeze(1)  # [actual_q_len, 1]
                                key_mask = subseq_token_mask.unsqueeze(0)  # [1, actual_k_len]
                                attn_mask_2d = (query_mask & key_mask)  # [actual_q_len, actual_k_len]
                            else:
                                # Asymmetric attention: query and key lengths differ
                                # For key positions, we need to extract corresponding mask positions
                                # Since we extracted attention weights using [subseq_start_clamped:subseq_end_clamped] for both dimensions,
                                # we should use the same range for key mask
                                query_mask = subseq_token_mask.unsqueeze(1)  # [actual_q_len, 1]
                                # Extract key mask using the same clamped range (keys are in the same range as queries for packed sequences)
                                key_mask_extracted = pack_token_mask[mask_start_clamped:mask_end_clamped] if pack_token_mask is not None else None
                                if key_mask_extracted is not None and key_mask_extracted.shape[0] >= actual_k_len:
                                    # Use extracted key mask
                                    key_mask = key_mask_extracted[:actual_k_len].unsqueeze(0)  # [1, actual_k_len]
                                    if key_mask.dtype != torch.bool:
                                        key_mask = key_mask.bool()
                                else:
                                    # Fallback: assume all keys in the subsequence range are valid
                                    key_mask = torch.ones(actual_k_len, dtype=torch.bool, device=subseq_token_mask.device).unsqueeze(0)  # [1, actual_k_len]
                                attn_mask_2d = (query_mask & key_mask)  # [actual_q_len, actual_k_len]
                            
                            # Flatten mask: reshape in the same order as attention weights (row-major)
                            # attn_mask_2d[i, j] -> attn_mask_flat[i * actual_k_len + j]
                            # This matches subseq_student.reshape(num_heads, -1) which also uses row-major order
                            attn_mask_flat = attn_mask_2d.reshape(-1)  # [actual_q_len*actual_k_len]
                            
                            # Verify mask size matches flattened attention weights size (should always be true)
                            flat_size = subseq_student_flat.shape[1]  # Should be actual_q_len * actual_k_len
                            assert attn_mask_flat.shape[0] == flat_size, \
                                f"Mask flat size {attn_mask_flat.shape[0]} != attention weights flat size {flat_size} " \
                                f"(q_len={actual_q_len}, k_len={actual_k_len})"
                            
                            # Process each head separately
                            head_losses = []
                            for head_idx in range(num_heads):
                                head_student = subseq_student_flat[head_idx][attn_mask_flat]  # [num_valid_attn]
                                head_teacher = subseq_teacher_flat[head_idx][attn_mask_flat]  # [num_valid_attn]
                                
                                if head_student.numel() == 0:
                                    continue
                                
                                # Compute KL for this head
                                head_student_log_probs = F.log_softmax(head_student / temperature, dim=-1)
                                head_teacher_probs = F.softmax(head_teacher / temperature, dim=-1)
                                head_kl = F.kl_div(head_student_log_probs, head_teacher_probs, reduction='sum')
                                head_losses.append(head_kl)
                            
                            if head_losses:
                                # Average across heads for this subsequence
                                subsequence_losses.append(sum(head_losses) / len(head_losses))
                        else:
                            # No token mask: use all attention weights in this subsequence
                            # Flatten: [num_heads, subseq_len, subseq_len] -> [num_heads, subseq_len*subseq_len]
                            subseq_student_flat = subseq_student.reshape(num_heads, -1)  # [num_heads, subseq_len*subseq_len]
                            subseq_teacher_flat = subseq_teacher.reshape(num_heads, -1)  # [num_heads, subseq_len*subseq_len]
                            
                            # Process each head separately
                            head_losses = []
                            for head_idx in range(num_heads):
                                head_student = subseq_student_flat[head_idx]  # [subseq_len*subseq_len]
                                head_teacher = subseq_teacher_flat[head_idx]  # [subseq_len*subseq_len]
                                
                                # Compute KL for this head
                                head_student_log_probs = F.log_softmax(head_student / temperature, dim=-1)
                                head_teacher_probs = F.softmax(head_teacher / temperature, dim=-1)
                                head_kl = F.kl_div(head_student_log_probs, head_teacher_probs, reduction='sum')
                                head_losses.append(head_kl)
                            
                            if head_losses:
                                # Average across heads for this subsequence
                                subsequence_losses.append(sum(head_losses) / len(head_losses))
                
                if subsequence_losses:
                    # Average across all subsequences
                    layer_loss = sum(subsequence_losses) / len(subsequence_losses)
                    layer_losses.append(layer_loss)
            else:
                # Non-packed sequences: use original token-level approach
                student_flat = student_norm.reshape(-1, student_norm.shape[-1])
                teacher_flat = teacher_norm.reshape(-1, teacher_norm.shape[-1])

                if token_mask is not None:
                    mask = token_mask
                    if mask.shape[1] != student_norm.shape[1]:
                        if mask.shape[1] > student_norm.shape[1]:
                            mask = mask[:, : student_norm.shape[1]]
                        else:
                            pad_len = student_norm.shape[1] - mask.shape[1]
                            mask = torch.nn.functional.pad(mask, (0, pad_len))
                    mask = mask.reshape(-1) > 0
                    if mask.sum() == 0:
                        continue
                    if mask.numel() > student_flat.size(0):
                        mask = mask[: student_flat.size(0)]
                    elif mask.numel() < student_flat.size(0):
                        pad_len = student_flat.size(0) - mask.numel()
                        mask = torch.nn.functional.pad(mask, (0, pad_len))
                    student_flat = student_flat[mask]
                    teacher_flat = teacher_flat[mask]

                student_log_probs = F.log_softmax(student_flat / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_flat / temperature, dim=-1)

                layer_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                layer_losses.append(layer_loss)

        if not layer_losses:
            if attention_mask is not None:
                device = attention_mask.device
            else:
                device = None
                for attn in student_attn_layers:
                    if attn is not None:
                        device = attn.device
                        break
                if device is None:
                    device = torch.device(self.device_name)
            return torch.tensor(0.0, device=device, requires_grad=True)

        return sum(layer_losses) / len(layer_losses)


    def _validate_packed_eos_supervision(self, input_ids, loss_mask, cu_seqlens, attention_mask):
        """验证packed sequences中每个sample的EOS token是否有正常监督
        
        Args:
            input_ids: [batch_size, seq_len] token ids
            loss_mask: [batch_size, seq_len] loss mask (1表示需要计算loss)
            cu_seqlens: List/Tensor of tensors, each tensor is [0, seq1_len, seq1_len+seq2_len, ...] for one pack
            attention_mask: [batch_size, seq_len] attention mask (1表示有效token)
        
        Returns:
            dict: 验证结果统计
        """
        batch_size = input_ids.shape[0]
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        if eos_token_id is None:
            # 尝试从tokenizer获取
            try:
                eos_token_id = self.tokenizer.eos_token_id
            except:
                eos_token_id = None
        
        if eos_token_id is None:
            if self.device_mesh.get_rank() == 0:
                print("⚠️ WARNING: Cannot find eos_token_id, skipping EOS supervision validation")
            return {"skipped": True}
        
        validation_results = {
            "total_packs": batch_size,
            "total_subsequences": 0,
            "eos_found": 0,  # Total number of EOS tokens found across all subsequences
            "eos_supervised": 0,  # Number of EOS tokens that are supervised
            "eos_not_supervised": 0,  # Number of EOS tokens that are NOT supervised
            "no_eos_found": 0,  # Number of subsequences with no EOS
            "subsequences_with_eos": 0,  # Number of subsequences that contain at least one EOS
            "subsequences_with_all_eos_supervised": 0,  # Number of subsequences where ALL EOS are supervised
            "subsequences_with_unsupervised_eos": 0,  # Number of subsequences with at least one unsupervised EOS
            "issues": []  # List of (pack_idx, subseq_idx, issue_description)
        }
        
        # 处理cu_seqlens的不同格式
        # cu_seqlens可能是list of tensors，也可能是单个tensor（需要按batch拆分）
        if isinstance(cu_seqlens, torch.Tensor):
            # 如果是单个tensor，需要按batch拆分
            # 假设cu_seqlens是[batch_size, max_subseqs_per_pack]格式
            if cu_seqlens.dim() == 2:
                cu_seqlens_list = [cu_seqlens[i] for i in range(batch_size)]
            else:
                # 如果是一维tensor，假设所有pack共享相同的结构（不太可能，但处理一下）
                cu_seqlens_list = [cu_seqlens] * batch_size
        elif isinstance(cu_seqlens, (list, tuple)):
            cu_seqlens_list = list(cu_seqlens)
        else:
            if self.device_mesh.get_rank() == 0:
                print(f"⚠️ WARNING: Unknown cu_seqlens type: {type(cu_seqlens)}, skipping validation")
            return {"skipped": True}
        
        for pack_idx in range(batch_size):
            pack_input_ids = input_ids[pack_idx]  # [seq_len]
            pack_loss_mask = loss_mask[pack_idx]  # [seq_len]
            pack_attention_mask = attention_mask[pack_idx]  # [seq_len]
            
            # 获取这个pack的cu_seqlens
            if pack_idx < len(cu_seqlens_list):
                pack_cu_seqlens = cu_seqlens_list[pack_idx]
                # 转换为list
                if isinstance(pack_cu_seqlens, torch.Tensor):
                    pack_cu_seqlens_list = pack_cu_seqlens.cpu().tolist()
                else:
                    pack_cu_seqlens_list = list(pack_cu_seqlens)
            else:
                # 如果cu_seqlens数量不足，跳过
                validation_results["issues"].append(
                    (pack_idx, -1, f"No cu_seqlens for pack {pack_idx}")
                )
                continue
            
            num_subseqs = len(pack_cu_seqlens_list) - 1
            if num_subseqs <= 0:
                continue
            
            validation_results["total_subsequences"] += num_subseqs
            
            # 遍历每个subsequence
            for subseq_idx in range(num_subseqs):
                subseq_start = pack_cu_seqlens_list[subseq_idx]
                subseq_end = pack_cu_seqlens_list[subseq_idx + 1]
                
                if subseq_end <= subseq_start:
                    continue
                
                # 确保索引不越界
                subseq_start = max(0, min(subseq_start, pack_input_ids.shape[0] - 1))
                subseq_end = max(subseq_start + 1, min(subseq_end, pack_input_ids.shape[0]))
                
                # 提取这个subsequence的tokens和masks
                subseq_input_ids = pack_input_ids[subseq_start:subseq_end]  # [subseq_len]
                subseq_loss_mask = pack_loss_mask[subseq_start:subseq_end]  # [subseq_len]
                subseq_attention_mask = pack_attention_mask[subseq_start:subseq_end]  # [subseq_len]
                
                # 找到所有有效token的位置（考虑attention_mask）
                valid_positions = (subseq_attention_mask == 1).nonzero(as_tuple=True)[0]
                if len(valid_positions) == 0:
                    validation_results["issues"].append(
                        (pack_idx, subseq_idx, "No valid tokens in subsequence")
                    )
                    continue
                
                # 检查subsequence中是否包含EOS token（不一定是最后一个token）
                # 在packed sequences中，EOS可能在subsequence中间，后面还有padding或其他内容
                # 对于多轮对话，一个subsequence可能包含多个EOS token（每个turn一个）
                # 重要：检查所有EOS token，区分assistant的EOS和user的EOS
                # 方法：通过loss_mask判断
                #   - 如果EOS的loss_mask=1，说明它是assistant的EOS且被监督（正常）
                #   - 如果EOS的loss_mask=0，需要进一步判断：
                #     * 如果EOS前面有loss_mask=1的区域，说明这是assistant回复的EOS但没被监督（问题）
                #     * 如果EOS前后都是loss_mask=0，可能是user的EOS（正常，不应该被监督）
                assistant_eos_positions = []  # 记录所有assistant的EOS（包括被监督和未被监督的）
                
                for valid_pos in valid_positions:
                    absolute_pos = subseq_start + valid_pos.item()
                    if absolute_pos < pack_input_ids.shape[0]:
                        token_id = pack_input_ids[absolute_pos].item()
                        if token_id == eos_token_id:
                            # 检查loss_mask
                            eos_loss_mask = pack_loss_mask[absolute_pos].item()
                            
                            if eos_loss_mask == 1:
                                # loss_mask=1说明是assistant的EOS且被监督（正常）
                                assistant_eos_positions.append(absolute_pos)
                            else:
                                # loss_mask=0，需要判断是user的EOS还是assistant的EOS但没被监督
                                # 方法：检查EOS前面是否有loss_mask=1的区域
                                # 如果有，说明这是assistant回复的EOS，应该被监督但没被监督（问题）
                                is_likely_assistant_eos = False
                                
                                # 向前查找，检查EOS前面是否有loss_mask=1的区域
                                # 查找范围：EOS前面最多100个token（通常足够覆盖一个回复）
                                lookback_start = max(subseq_start, absolute_pos - 100)
                                for lookback_pos in range(absolute_pos - 1, lookback_start - 1, -1):
                                    if lookback_pos < pack_input_ids.shape[0] and lookback_pos >= subseq_start:
                                        lookback_loss_mask = pack_loss_mask[lookback_pos].item()
                                        if lookback_loss_mask == 1:
                                            # 找到了loss_mask=1的区域，说明这是assistant回复的EOS
                                            # 但它的loss_mask=0，说明没有被监督（问题）
                                            is_likely_assistant_eos = True
                                            break
                                
                                if is_likely_assistant_eos:
                                    # 这是assistant的EOS但没被监督（问题）
                                    assistant_eos_positions.append(absolute_pos)
                                # 如果is_likely_assistant_eos=False，可能是user的EOS（正常，不记录）
                
                # 使用assistant的EOS位置进行验证
                # assistant_eos_positions包含所有assistant的EOS（包括被监督和未被监督的）
                eos_positions_in_subseq = assistant_eos_positions
                
                if len(eos_positions_in_subseq) > 0:
                    # 找到了assistant EOS token（可能有多个，multi-turn对话中每个turn一个）
                    validation_results["subsequences_with_eos"] += 1
                    validation_results["eos_found"] += len(eos_positions_in_subseq)
                    
                    # 检查每个EOS token的监督情况
                    supervised_count = 0
                    unsupervised_count = 0
                    unsupervised_positions = []
                    
                    for eos_pos in eos_positions_in_subseq:
                        # 检查EOS位置的loss_mask
                        eos_loss_mask = pack_loss_mask[eos_pos].item()
                        
                        if eos_loss_mask == 1:
                            supervised_count += 1
                            validation_results["eos_supervised"] += 1
                        else:
                            # loss_mask=0，说明是assistant的EOS但没被监督（问题）
                            unsupervised_count += 1
                            validation_results["eos_not_supervised"] += 1
                            unsupervised_positions.append(eos_pos)
                    
                    # 记录统计信息
                    if unsupervised_count == 0:
                        # 所有assistant EOS都被监督（正常情况）
                        validation_results["subsequences_with_all_eos_supervised"] += 1
                    else:
                        # 至少有一个assistant EOS未被监督（问题）
                        validation_results["subsequences_with_unsupervised_eos"] += 1
                        # 记录问题：列出所有未被监督的assistant EOS位置
                        if len(unsupervised_positions) == 1:
                            validation_results["issues"].append(
                                (pack_idx, subseq_idx, 
                                 f"Assistant EOS at pos {unsupervised_positions[0]} not supervised (loss_mask=0)")
                            )
                        else:
                            # 多个assistant EOS未被监督
                            positions_str = ", ".join(map(str, unsupervised_positions))
                            validation_results["issues"].append(
                                (pack_idx, subseq_idx, 
                                 f"{len(unsupervised_positions)} assistant EOS tokens not supervised at positions: {positions_str}")
                            )
                    
                    # 如果找到多个EOS，记录信息（用于调试多轮对话）
                    if len(eos_positions_in_subseq) > 1:
                        # 这是一个多轮对话的subsequence，有多个EOS
                        positions_str = ", ".join(map(str, eos_positions_in_subseq))
                        if unsupervised_count > 0:
                            # 只在有问题时记录详细信息
                            pass  # 已经在issues中记录了
                        # 可选：记录所有EOS的位置（用于调试）
                        # validation_results["issues"].append(
                        #     (pack_idx, subseq_idx, 
                        #      f"Multi-turn subsequence with {len(eos_positions_in_subseq)} EOS tokens at positions: {positions_str}")
                        # )
                else:
                    # 没有找到assistant EOS token
                    # 获取最后一个有效token用于调试信息
                    last_valid_pos = valid_positions[-1].item()
                    absolute_pos = subseq_start + last_valid_pos
                    if absolute_pos < pack_input_ids.shape[0]:
                        last_valid_token_id = pack_input_ids[absolute_pos].item()
                        validation_results["no_eos_found"] += 1
                        validation_results["issues"].append(
                            (pack_idx, subseq_idx,
                             f"No assistant EOS found, last token is {last_valid_token_id} at pos {absolute_pos}")
                        )
                    else:
                        validation_results["no_eos_found"] += 1
                        validation_results["issues"].append(
                            (pack_idx, subseq_idx, "No assistant EOS found, position out of bounds")
                        )
        
        return validation_results
    
    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        self._last_view_type_loss = {}
        self._last_loss_wo_normalize = None
        view_types = batch.pop("view_type", None)
        
        # Check if this is a packed batch
        is_packed = batch.get("is_packed", False)
        # 保存cu_seqlens用于验证（在pop之前）
        cu_seqlens_for_validation = batch.get("cu_seqlens", None) if is_packed else None
        cu_seqlens = batch.pop("cu_seqlens", None) if is_packed else None
        max_seqlen_in_batch = batch.pop("max_seqlen_in_batch", None) if is_packed else None

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        
        # Debug: Validate batch for potential Flash Attention issues
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
        
        # Check for extremely small batch sizes that might cause issues
        if batch_size < 1:
            raise ValueError(f"Invalid batch_size: {batch_size}")
        
        # Check attention_mask validity
        if attention_mask.dim() == 2:
            # Standard 2D mask: should have at least some valid tokens
            valid_tokens = attention_mask.sum().item()
            if valid_tokens == 0:
                raise ValueError(f"Attention mask has no valid tokens! Shape: {attention_mask.shape}")
            
            # Check if any sample has zero valid tokens (would cause indexing issues)
            per_sample_valid = attention_mask.sum(dim=1)
            if (per_sample_valid == 0).any():
                zero_samples = (per_sample_valid == 0).nonzero(as_tuple=True)[0].tolist()
                raise ValueError(f"Samples with zero valid tokens: {zero_samples}")
            
            # Additional check: ensure attention_mask values are only 0 or 1
            unique_values = attention_mask.unique()
            if not all(v in [0, 1] for v in unique_values.tolist()):
                raise ValueError(f"Attention mask has invalid values (should be 0 or 1): {unique_values.tolist()}")
            
            # CRITICAL: Check for potential Flash Attention issues
            # Flash Attention's _upad_input creates indices based on attention_mask
            # If indices are out of bounds, it will cause CUDA kernel assertion failure
            # This usually happens when:
            # 1. attention_mask doesn't match actual sequence length
            # 2. There are gaps or inconsistencies in the mask pattern
            
            # Check each sample for suspicious patterns
            for sample_idx in range(batch_size):
                sample_mask = attention_mask[sample_idx]
                sample_ids = input_ids[sample_idx]
                
                # Find first and last valid token positions
                valid_positions = (sample_mask == 1).nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    first_valid = valid_positions[0].item()
                    last_valid = valid_positions[-1].item()
                    n_valid = len(valid_positions)
                    
                    # Check if there are gaps in the mask (non-contiguous valid tokens)
                    expected_consecutive = last_valid - first_valid + 1
                    if n_valid != expected_consecutive:
                        # There are gaps! This might cause issues
                        print(f"⚠️ WARNING: Sample {sample_idx} has non-contiguous attention mask!")
                        print(f"   First valid: {first_valid}, Last valid: {last_valid}")
                        print(f"   Expected consecutive: {expected_consecutive}, Actual valid: {n_valid}")
                        print(f"   This may cause Flash Attention index issues!")
                    
                    # Check if valid tokens extend to sequence end (common pattern issue)
                    if last_valid < seq_len - 1:
                        # This is normal (padding at the end)
                        pass
                    
                    # CRITICAL: Check if any tokens after last_valid position are NOT pad tokens
                    if last_valid < seq_len - 1:
                        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
                        tokens_after_valid = sample_ids[last_valid + 1:]
                        non_pad_after = (tokens_after_valid != pad_token_id).sum().item()
                        if non_pad_after > 0:
                            print(f"⚠️ WARNING: Sample {sample_idx} has {non_pad_after} non-pad tokens after last valid position!")
                            print(f"   Last valid position: {last_valid}, but non-pad tokens found after it")
                            print(f"   This indicates attention_mask/input_ids mismatch!")
                            # Print the problematic tokens
                            problematic_tokens = tokens_after_valid[tokens_after_valid != pad_token_id][:10]
                            print(f"   Problematic token IDs: {problematic_tokens.tolist()}")
        elif attention_mask.dim() == 4:
            # 4D mask from packing: check diagonal
            for i in range(batch_size):
                diagonal = attention_mask[i, 0].diagonal()
                valid_tokens = (diagonal > float('-inf')).sum().item()
                if valid_tokens == 0:
                    raise ValueError(f"Packed sequence sample {i} has no valid tokens!")
        
        # For packed sequences: keep attention_mask as 2D
        # Flash Attention varlen will be triggered by non-monotonic position_ids
        # The modified qwen2_vl.py will handle this correctly
        if is_packed and cu_seqlens is not None:
            # Keep attention_mask as 2D [batch, seq] with 0/1 values
            # The verl qwen2_vl attention implementation will:
            # 1. Detect non-monotonic position_ids
            # 2. Call prepare_fa2_from_position_ids to extract cu_seqlens
            # 3. Use flash_attn_varlen_func for efficient packed attention
            pass  # attention_mask stays as-is (2D from collate_fn)
        
        # Check loss_mask before reshaping
        original_loss_mask = batch.pop("loss_mask")
        
        # Extract has_view_type flag (if present) to control KL loss
        # If has_view_type is True, skip KL loss for this sample
        has_view_type = batch.pop("has_view_type", None)
        
        # Save for debugging
        self._last_loss_mask = original_loss_mask
        
        # 验证packed sequences中EOS token的监督情况
        # 每个step都进行验证（只在rank 0执行，避免重复输出）
        if is_packed and cu_seqlens_for_validation is not None:
            # 只在rank 0执行验证
            should_validate = (self.device_mesh.get_rank() == 0)
            
            if should_validate:
                # 更新step计数（用于显示）
                if not hasattr(self, '_validation_step_count'):
                    self._validation_step_count = 0
                self._validation_step_count += 1
                
                # 确保loss_mask在CPU上用于验证（避免GPU内存问题）
                original_loss_mask_cpu = original_loss_mask.cpu() if original_loss_mask.is_cuda else original_loss_mask
                input_ids_cpu = input_ids.cpu() if input_ids.is_cuda else input_ids
                attention_mask_cpu = attention_mask.cpu() if attention_mask.is_cuda else attention_mask
                
                validation_results = self._validate_packed_eos_supervision(
                    input_ids_cpu, 
                    original_loss_mask_cpu, 
                    cu_seqlens_for_validation, 
                    attention_mask_cpu
                )
                
                # Validation is performed silently (no output)
                # Results are stored in validation_results but not printed
        
        loss_mask_shifted = original_loss_mask[:, 1:]
        loss_mask_shifted = loss_mask_shifted.to(self.device_name)
        loss_mask = loss_mask_shifted.reshape(-1)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        
        # Debug: print shapes occasionally
        # import random
        # if random.random() < 0.01:  # 1% chance
        #     print(f"[Forward Debug] input_ids: {input_ids.shape}, position_ids: {position_ids.shape}, "
        #           f"attention_mask: {attention_mask.shape}")
        
        # Handle multi_modal_inputs if present (for VL models)
        multi_modal_inputs = batch.pop("multi_modal_inputs", None)
        model_kwargs = {}
        if multi_modal_inputs is not None and len(multi_modal_inputs) > 0:
            # multi_modal_inputs is a list of dicts (one per sample in batch)
            # For Qwen2-VL: {"pixel_values": tensor, "image_grid_thw": tensor, ...}
            
            # Collect all keys from first sample
            all_keys = set()
            for mm_input in multi_modal_inputs:
                all_keys.update(mm_input.keys())
            
            # Merge each key across batch
            # Skip empty multi_modal_inputs (text-only samples)
            merged_mm_inputs = {}
            for key in all_keys:
                values = []
                for mm_input in multi_modal_inputs:
                    # Skip empty dicts (text-only samples)
                    if not mm_input:
                        continue
                    if key in mm_input:
                        val = mm_input[key]
                        # Move to device if tensor
                        if isinstance(val, torch.Tensor):
                            val = val.to(self.device_name)
                        values.append(val)
                
                if len(values) > 0:
                    # Merge tensors
                    if isinstance(values[0], torch.Tensor):
                        # For most cases, cat along dim 0 (batch dimension)
                        merged_mm_inputs[key] = torch.cat(values, dim=0)
                    else:
                        # Non-tensor values, keep as list
                        merged_mm_inputs[key] = values
            
            # Debug: Check if merged values match input_ids
            # if "pixel_values" in merged_mm_inputs and "image_grid_thw" in merged_mm_inputs:
            #     import random
            #     if random.random() < 0.01:  # 1% chance to debug print
            #         pixel_values = merged_mm_inputs["pixel_values"]
            #         image_grid_thw = merged_mm_inputs["image_grid_thw"]
                    
            #         # Calculate expected features from image_grid_thw
            #         total_features = 0
            #         for i in range(image_grid_thw.shape[0]):
            #             t, h, w = image_grid_thw[i]
            #             total_features += t * h * w
                    
            #         # Count image_pad tokens in input_ids
            #         image_token_id = 151655  # <|image_pad|>
            #         n_image_tokens = (input_ids == image_token_id).sum().item()
                    
            #         # Check merge ratio (should be 4:1)
            #         expected_tokens = total_features.item() // 4 if isinstance(total_features, torch.Tensor) else total_features // 4
                    
            #         print(f"\n[DEBUG Multi-modal Merge]")
            #         print(f"  Micro batch size: {input_ids.shape[0]}")
            #         print(f"  pixel_values shape: {pixel_values.shape}")
            #         print(f"  image_grid_thw shape: {image_grid_thw.shape}")
            #         print(f"  image_grid_thw values: {image_grid_thw}")
            #         print(f"  Total features (from grid_thw): {total_features}")
            #         print(f"  <|image_pad|> tokens in input_ids: {n_image_tokens}")
            #         print(f"  Expected tokens (features/4): {expected_tokens}")
            #         print(f"  Match: {'✅' if n_image_tokens == expected_tokens else f'❌ {n_image_tokens} != {expected_tokens}'}")
            
            model_kwargs.update(merged_mm_inputs)

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                
                if self.enable_distillation:
                    self._reset_attn_cache(is_teacher=False)
                    self._reset_attn_cache(is_teacher=True)

                output = self.fsdp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=self.use_hidden_states_kl if self.enable_distillation else False,
                    **model_kwargs,
                )

                # Extract logits and hidden_states before deleting output
                if hasattr(output, 'logits'):
                    logits = output.logits
                elif isinstance(output, tuple):
                    hidden_states = output[0]
                    logits = self.model.lm_head(hidden_states) if hasattr(self.model, 'lm_head') else hidden_states
                elif hasattr(output, 'last_hidden_state'):
                    hidden_states = output.last_hidden_state
                    if hasattr(self.model, 'lm_head'):
                        logits = self.model.lm_head(hidden_states)
                    else:
                        raise ValueError("Model has no lm_head to compute logits")
                elif hasattr(output, '__getitem__'):
                    hidden_states = output[0]
                    if hasattr(self.model, 'lm_head'):
                        logits = self.model.lm_head(hidden_states)
                    else:
                        logits = hidden_states
                else:
                    raise ValueError(
                        f"Cannot extract logits from output type: {type(output)}, available attributes: {dir(output)}"
                    )

                # Save student hidden states for KL computation if needed
                student_hidden_states = None
                if self.enable_distillation and self.use_hidden_states_kl and hasattr(output, 'hidden_states'):
                    student_hidden_states = output.hidden_states

                # Now safe to delete output
                del output

                student_attentions = (
                    self._collect_attention_outputs(is_teacher=False)
                    if self.enable_distillation
                    else []
                )

                teacher_attentions = []
                teacher_hidden_states = None
                if self.enable_distillation and self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_output = self.teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=self.use_hidden_states_kl,
                            **model_kwargs,
                        )
                        
                        # Store hidden states if using hidden states KL
                        if self.use_hidden_states_kl and hasattr(teacher_output, 'hidden_states'):
                            teacher_hidden_states = teacher_output.hidden_states
                        
                        del teacher_output
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if not self.use_hidden_states_kl:
                        # Only collect attention outputs if using attention KL
                        teacher_attentions = self._collect_attention_outputs(is_teacher=True)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                # Get vocab_size from model config
                vocab_size = getattr(self.model.config, 'vocab_size', logits.shape[-1])
                shift_logits = shift_logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)


                # # === View-type aware loss statistics ===
                valid_token_counts = loss_mask_shifted.sum(dim=1)
                per_sample_loss_sum = loss.reshape(batch_size, -1).sum(dim=1)
                nonzero_mask = valid_token_counts > 0
                # view_types is NonTensorStack
                if isinstance(view_types, list):
                    view_types_list = view_types
                else:
                    view_types_list = view_types.cpu().tolist()
                # print(f"view_types : {view_types}")
                # view_type_values = self._prepare_view_type_list(view_types, batch_size)
                # use 
                view_type_loss_stats = {}
                for idx, vt in enumerate(view_types_list):
                    if idx >= per_sample_loss_sum.shape[0] or not nonzero_mask[idx]:
                        continue
                    stats = view_type_loss_stats.setdefault(vt, {"loss_sum": 0.0, "token_count": 0.0, "count": 0})
                    stats["loss_sum"] += per_sample_loss_sum[idx].detach().float().item()
                    stats["token_count"] += valid_token_counts[idx].detach().float().item()
                    stats["count"] += 1

                # === Sample-level loss normalization ===
                # Instead of token-level average, normalize per sample/subsequence first, then average
                if is_packed and cu_seqlens is not None:
                    # Pack模式：按subsequence normalize（每个subsequence是一个sample）
                    subsequence_losses = []
                    total_subsequences = 0
                    
                    # loss shape: [batch_size * seq_len] (flattened)
                    # loss_mask_shifted shape: [batch_size, seq_len]
                    loss_reshaped = loss.reshape(batch_size, -1)  # [batch_size, seq_len]
                    
                    for pack_idx in range(batch_size):
                        pack_loss = loss_reshaped[pack_idx]  # [seq_len]
                        pack_loss_mask = loss_mask_shifted[pack_idx]  # [seq_len]
                        pack_cu_seqlens = cu_seqlens[pack_idx]  # Tensor: [0, seq1_len, seq1_len+seq2_len, ...]
                        pack_cu_seqlens_list = pack_cu_seqlens.cpu().tolist()
                        
                        # 对每个subsequence normalize
                        for subseq_idx in range(len(pack_cu_seqlens_list) - 1):
                            subseq_start = pack_cu_seqlens_list[subseq_idx]
                            subseq_end = pack_cu_seqlens_list[subseq_idx + 1]
                            
                            # 提取该subsequence的loss和mask
                            subseq_loss = pack_loss[subseq_start:subseq_end]
                            subseq_mask = pack_loss_mask[subseq_start:subseq_end]
                            
                            # 计算该subsequence的loss sum和token count
                            subseq_loss_sum = subseq_loss.sum()
                            subseq_token_count = subseq_mask.sum()
                            
                            # 只处理有有效token的subsequence
                            if subseq_token_count > 0:
                                subseq_loss_normalized = subseq_loss_sum / (subseq_token_count + 1e-8)
                                subsequence_losses.append(subseq_loss_normalized)
                                total_subsequences += 1
                    
                    # 对所有subsequence的normalized loss求平均
                    if total_subsequences > 0:
                        # subsequence_losses中的元素都是tensor，使用torch.stack求平均
                        loss = torch.stack(subsequence_losses).mean()
                    else:
                        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device_name)
                else:
                    # 非Pack模式：按sample normalize
                    # valid_token_counts shape: [batch_size]
                    # per_sample_loss_sum shape: [batch_size]
                    per_sample_loss_normalized = per_sample_loss_sum / (valid_token_counts + 1e-8)
                    # 只对有效samples求平均
                    valid_samples_mask = valid_token_counts > 0
                    if valid_samples_mask.any():
                        loss = per_sample_loss_normalized[valid_samples_mask].mean()
                    else:
                        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device_name)
                
                tmp_loss = loss.detach()
                # Add KL distillation loss if enabled (using attention weights)
                # Create mask for KL loss computation
                # For packed sequences: create token-level mask based on subsequence has_view_type
                # For non-packed sequences: create sample-level mask
                sample_mask = None
                kl_token_mask = None  # Token-level mask for packed sequences
                
                if has_view_type is not None:
                    batch_size = input_ids.shape[0]
                    
                    # Check if this is a packed sequence
                    if is_packed and cu_seqlens is not None:
                        # Packed sequence: create token-level KL mask
                        # has_view_type is a list of lists: [[bool, bool, ...], ...]
                        # Each inner list corresponds to one pack, each bool corresponds to one subsequence
                        kl_token_mask = []
                        
                        for pack_idx in range(batch_size):
                            pack_seq_len = input_ids.shape[1]  # Full sequence length (including padding)
                            pack_mask = torch.zeros(pack_seq_len, dtype=torch.bool, device=self.device_name)
                            
                            # Get cu_seqlens for this pack
                            pack_cu_seqlens = cu_seqlens[pack_idx]  # Tensor: [0, seq1_len, seq1_len+seq2_len, ...]
                            pack_cu_seqlens_list = pack_cu_seqlens.cpu().tolist()
                            
                            # Get has_view_type for this pack (list of bools, one per subsequence)
                            if isinstance(has_view_type, list) and len(has_view_type) > pack_idx:
                                pack_has_view_type_list = has_view_type[pack_idx]
                                if not isinstance(pack_has_view_type_list, list):
                                    # Fallback: if it's a single bool, treat as single subsequence
                                    pack_has_view_type_list = [pack_has_view_type_list]
                            else:
                                # Fallback: assume all subsequences need KL
                                pack_has_view_type_list = [False] * (len(pack_cu_seqlens_list) - 1)
                            
                            # Build token-level mask: True = compute KL, False = skip KL
                            # For each subsequence, if it doesn't have view_type, mark its tokens as True
                            for subseq_idx in range(len(pack_cu_seqlens_list) - 1):
                                subseq_start = pack_cu_seqlens_list[subseq_idx]
                                subseq_end = pack_cu_seqlens_list[subseq_idx + 1]
                                
                                # Check if this subsequence needs KL loss
                                if subseq_idx < len(pack_has_view_type_list):
                                    subseq_has_view_type = pack_has_view_type_list[subseq_idx]
                                    # If subsequence doesn't have view_type, compute KL
                                    if not subseq_has_view_type:
                                        pack_mask[subseq_start:subseq_end] = True
                            
                            # Exclude padding tokens: only compute KL for valid tokens (attention_mask == 1)
                            pack_attention_mask = attention_mask[pack_idx]  # [seq_len]
                            pack_mask = pack_mask & (pack_attention_mask.bool())
                            
                            kl_token_mask.append(pack_mask)
                        
                        # Stack into [batch_size, seq_len] tensor
                        kl_token_mask = torch.stack(kl_token_mask, dim=0)  # [batch_size, seq_len]
                        
                    else:
                        # Non-packed sequence: use sample-level mask
                        if isinstance(has_view_type, torch.Tensor):
                            # Ensure it's on the right device and inverted (True = skip → False = compute)
                            sample_mask = ~has_view_type.bool().to(self.device_name)
                        elif isinstance(has_view_type, (list, tuple)):
                            # Convert list to tensor and invert
                            sample_mask = torch.tensor([not x for x in has_view_type], 
                                                       dtype=torch.bool, device=self.device_name)
                        else:
                            # Single value - apply to all samples in batch
                            sample_mask = torch.tensor([not has_view_type] * batch_size,
                                                       dtype=torch.bool, device=self.device_name)
                
                # Add KL distillation loss if enabled
                kl_loss_value = None
                # Check if we should compute KL loss
                should_compute_kl = False
                if kl_token_mask is not None:
                    should_compute_kl = kl_token_mask.any()
                elif sample_mask is not None:
                    should_compute_kl = sample_mask.any()
                else:
                    should_compute_kl = True  # No mask means compute for all
                
                if self.enable_distillation and should_compute_kl:
                    if self.use_hidden_states_kl:
                        # Use Hidden States KL (more stable, less memory)
                        # Use the saved student_hidden_states (extracted before deleting output)
                        if teacher_hidden_states is not None and student_hidden_states is not None:
                            kl_loss = self._compute_hidden_states_kl_loss(
                                student_hidden_states,
                                teacher_hidden_states,
                                attention_mask,
                                sample_mask=sample_mask,
                                token_mask=kl_token_mask,
                                is_packed=is_packed,
                                cu_seqlens=cu_seqlens,
                            )
                            kl_loss_value = kl_loss.detach()
                            loss = loss + self.kl_loss_weight * kl_loss
                            
                            # Store mask info for logging
                            if kl_token_mask is not None:
                                # Token-level mask: count tokens
                                self._kl_tokens_computed = kl_token_mask.sum().item()
                                self._kl_tokens_skipped = (~kl_token_mask).sum().item()
                                
                                # For packed sequences: count subsequences (not packs)
                                # A subsequence is considered if it has at least one token with KL
                                if is_packed and cu_seqlens is not None:
                                    total_subsequences = 0
                                    subsequences_with_kl = 0
                                    
                                    for pack_idx in range(input_ids.shape[0]):
                                        pack_cu_seqlens = cu_seqlens[pack_idx]  # [0, seq1_len, seq1_len+seq2_len, ...]
                                        pack_cu_seqlens_list = pack_cu_seqlens.cpu().tolist()
                                        num_subseqs_in_pack = len(pack_cu_seqlens_list) - 1
                                        total_subsequences += num_subseqs_in_pack
                                        
                                        # Check each subsequence in this pack
                                        pack_kl_mask = kl_token_mask[pack_idx]  # [seq_len]
                                        for subseq_idx in range(num_subseqs_in_pack):
                                            subseq_start = pack_cu_seqlens_list[subseq_idx]
                                            subseq_end = pack_cu_seqlens_list[subseq_idx + 1]
                                            # Check if this subsequence has any token with KL
                                            if pack_kl_mask[subseq_start:subseq_end].any():
                                                subsequences_with_kl += 1
                                    
                                    self._kl_samples_computed = subsequences_with_kl
                                    self._kl_samples_skipped = total_subsequences - subsequences_with_kl
                                else:
                                    # Non-packed: count packs (samples)
                                    # A pack is considered if it has at least one token with KL
                                    samples_with_kl = (kl_token_mask.sum(dim=1) > 0).sum().item()
                                    self._kl_samples_computed = samples_with_kl
                                    self._kl_samples_skipped = input_ids.shape[0] - samples_with_kl
                            elif sample_mask is not None:
                                # Sample-level mask: count samples
                                self._kl_samples_computed = sample_mask.sum().item()
                                self._kl_samples_skipped = (~sample_mask).sum().item()
                            else:
                                # No mask: all samples/tokens compute KL
                                self._kl_samples_computed = input_ids.shape[0]
                                self._kl_samples_skipped = 0
                            
                            # Free memory - delete both student and teacher hidden states
                            # del teacher_hidden_states
                            # del student_hidden_states
                            # if torch.cuda.is_available():
                            #     torch.cuda.empty_cache()
                    else:
                        # Use Attention KL (current implementation)
                        if (
                            teacher_attentions
                            and any(attn is not None for attn in student_attentions)
                            and any(attn is not None for attn in teacher_attentions)
                        ):
                            kl_loss = self._compute_attention_kl_loss(
                                student_attentions,
                                teacher_attentions,
                                attention_mask,
                                sample_mask=sample_mask,
                                token_mask=kl_token_mask,
                                cu_seqlens=cu_seqlens,
                                is_packed=is_packed,
                            )
                            kl_loss_value = kl_loss.detach()
                            loss = loss + self.kl_loss_weight * kl_loss
                            
                            # Store mask info for logging
                            if kl_token_mask is not None:
                                # Token-level mask: count tokens
                                self._kl_tokens_computed = kl_token_mask.sum().item()
                                self._kl_tokens_skipped = (~kl_token_mask).sum().item()
                                
                                # For packed sequences: count subsequences (not packs)
                                # A subsequence is considered if it has at least one token with KL
                                if is_packed and cu_seqlens is not None:
                                    total_subsequences = 0
                                    subsequences_with_kl = 0
                                    
                                    for pack_idx in range(input_ids.shape[0]):
                                        pack_cu_seqlens = cu_seqlens[pack_idx]  # [0, seq1_len, seq1_len+seq2_len, ...]
                                        pack_cu_seqlens_list = pack_cu_seqlens.cpu().tolist()
                                        num_subseqs_in_pack = len(pack_cu_seqlens_list) - 1
                                        total_subsequences += num_subseqs_in_pack
                                        
                                        # Check each subsequence in this pack
                                        pack_kl_mask = kl_token_mask[pack_idx]  # [seq_len]
                                        for subseq_idx in range(num_subseqs_in_pack):
                                            subseq_start = pack_cu_seqlens_list[subseq_idx]
                                            subseq_end = pack_cu_seqlens_list[subseq_idx + 1]
                                            # Check if this subsequence has any token with KL
                                            if pack_kl_mask[subseq_start:subseq_end].any():
                                                subsequences_with_kl += 1
                                    
                                    self._kl_samples_computed = subsequences_with_kl
                                    self._kl_samples_skipped = total_subsequences - subsequences_with_kl
                                else:
                                    # Non-packed: count packs (samples)
                                    # A pack is considered if it has at least one token with KL
                                    samples_with_kl = (kl_token_mask.sum(dim=1) > 0).sum().item()
                                    self._kl_samples_computed = samples_with_kl
                                    self._kl_samples_skipped = input_ids.shape[0] - samples_with_kl
                            elif sample_mask is not None:
                                # Sample-level mask: count samples
                                self._kl_samples_computed = sample_mask.sum().item()
                                self._kl_samples_skipped = (~sample_mask).sum().item()
                            else:
                                # No mask: all samples/tokens compute KL
                                self._kl_samples_computed = input_ids.shape[0]
                                self._kl_samples_skipped = 0
                            
                            # Delete attention weights
                            # del teacher_attentions
                            # del student_attentions
                            # if torch.cuda.is_available():
                            #     torch.cuda.empty_cache()
                
                # Store KL loss for logging (will be collected in training_step)
                self._last_kl_loss = kl_loss_value
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                    output_hidden_states=True,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            # IMPORTANT: balance_dp_token must be same across all ranks
            # All ranks execute same branch to avoid deadlock
            if self.config.data.balance_dp_token:
                # All ranks participate in all_reduce
                torch.distributed.all_reduce(valid_token_this_rank, op=torch.distributed.ReduceOp.SUM)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                # No all_reduce, but all ranks execute this branch
                dp_size = 1

            # loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            # avg_loss_wo_kl = tmp_loss / (valid_token_this_rank + 1e-8) * dp_size
            for vt, stats in view_type_loss_stats.items():
                stats["loss"] = stats["loss_sum"] / (stats["token_count"] + 1e-8)
            self._last_view_type_loss = view_type_loss_stats
            self._last_loss_wo_normalize = tmp_loss

            loss = loss / n_micro_batches  # normalize loss

            if do_backward:
                loss.backward()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            return loss

    def training_step(self, batch):
        start_time = time.time()

        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()
        
        # Timeout protection: if step takes too long, something is wrong
        # Note: For large-scale training (512+ GPUs), steps may take longer
        # Use configurable timeout or disable for large scale
        import signal
        timeout_seconds = getattr(self.config.trainer, 'step_timeout_seconds', None)
        
        if timeout_seconds is not None and timeout_seconds > 0:
            def timeout_handler(signum, frame):
                rank = self.device_mesh.get_rank()
                raise TimeoutError(
                    f"Training step timeout (>={timeout_seconds}s) on rank {rank}! "
                    f"Possible deadlock. Set trainer.step_timeout_seconds=0 to disable."
                )
            
            # Set timeout on all ranks (not just rank 0) to avoid deadlock
            # If only rank 0 times out, other ranks will hang in collective ops
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            except (ValueError, OSError) as e:
                # Some systems don't support SIGALRM, skip timeout
                if self.device_mesh.get_rank() == 0:
                    print(f"⚠️ Warning: Cannot set timeout signal: {e}")

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)
        # Handle both TensorDict and plain dict
        if isinstance(batch, TensorDict):
            micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        else:
            # Manual split for plain dict (for VL models with special position_ids)
            micro_batches = self._manual_split_batch(batch, self.config.data.micro_batch_size_per_gpu)
        
        n_micro_batches = len(micro_batches)
        
        # CRITICAL: Verify all ranks have same number of micro_batches
        # If not, some ranks will do extra backward() → all_reduce deadlock
        n_micro_batches_tensor = torch.tensor(n_micro_batches, device=self.device_name)
        torch.distributed.all_reduce(n_micro_batches_tensor, op=torch.distributed.ReduceOp.MAX)
        expected_n_micro_batches = n_micro_batches_tensor.item()
        
        if n_micro_batches != expected_n_micro_batches:
            print(f"⚠️ WARNING: Rank {self.device_mesh.get_rank()} has {n_micro_batches} micro_batches, "
                  f"but max across ranks is {expected_n_micro_batches}. "
                  f"This will cause deadlock!")
            # Pad with None to match (will skip these)
            while len(micro_batches) < expected_n_micro_batches:
                micro_batches.append(None)
        
        step_loss = 0
        step_kl_loss = 0.0  # Initialize KL loss accumulator
        step_loss_wo_normalize = 0.0  # Initialize loss without normalization accumulator
        view_type_stats_local = None
        # Initialize KL statistics accumulators
        step_kl_samples_computed = 0
        step_kl_samples_skipped = 0
        step_kl_tokens_computed = 0
        step_kl_tokens_skipped = 0
        # Calculate packing efficiency if using sequence packing
        is_packed = micro_batches[0].get("is_packed", False) if micro_batches and micro_batches[0] is not None else False
        if is_packed and micro_batches[0] is not None:
            # Calculate efficiency for this batch
            total_tokens = 0
            valid_tokens = 0
            num_packs = 0  # Total number of packed samples
            
            for micro_batch in micro_batches:
                if micro_batch is None:
                    continue
                # Count valid (non-padding) tokens
                if "attention_mask" in micro_batch:
                    attn_mask = micro_batch["attention_mask"]
                    if isinstance(attn_mask, torch.Tensor):
                        if attn_mask.dim() == 4:
                            # 4D mask from packing: [batch, 1, seq, seq]
                            # Use diagonal to count valid tokens
                            # diagonal value is 0.0 for valid, -inf for padding
                            for i in range(attn_mask.shape[0]):
                                diagonal = attn_mask[i, 0].diagonal()
                                # Count non-inf values (valid tokens)
                                valid_tokens += (diagonal > float('-inf')).sum().item()
                                total_tokens += attn_mask.shape[2]
                        else:
                            # 2D mask: [batch, seq]
                            valid_tokens += attn_mask.sum().item()
                            total_tokens += attn_mask.numel()
                
                # Count number of packs
                if "cu_seqlens" in micro_batch:
                    num_packs += len(micro_batch["cu_seqlens"])  # Number of packed samples
            
            if total_tokens > 0:
                self._packing_efficiency = valid_tokens / total_tokens
                self._packing_num_packs = num_packs / len(micro_batches) if len(micro_batches) > 0 else 0
        
        for micro_idx, micro_batch in enumerate(micro_batches):
            # Skip None micro_batches (padding for consistency)
            if micro_batch is None:
                # Still need to participate in gradient sync
                # Use a simple dummy loss that contributes 0 to the gradient
                if self.device_mesh.get_rank() == 0 and micro_idx == len(micro_batches) - 1:
                    print(f"⚠️ Rank has {n_micro_batches} micro_batches (padded to {expected_n_micro_batches})")
                
                # Create dummy loss: simple scalar with requires_grad=True
                dummy_loss = torch.tensor(0.0, device=self.device_name, requires_grad=True)
                loss = dummy_loss / expected_n_micro_batches
                loss.backward()
                self._last_kl_loss = None
                self._last_view_type_loss = {}
                self._last_loss_wo_normalize = None
            else:
                try:
                    loss = self._compute_loss_and_backward(batch=micro_batch, do_backward=True, n_micro_batches=expected_n_micro_batches)
                    loss_value = loss.item()
                    step_loss += loss_value
                    step_loss_wo_normalize += self._last_loss_wo_normalize if self._last_loss_wo_normalize is not None else 0.0
                
                    # Collect KL loss if available
                    if hasattr(self, '_last_kl_loss') and self._last_kl_loss is not None:
                        step_kl_loss += self._last_kl_loss.item()
                    
                    # Accumulate KL statistics from this micro_batch
                    if hasattr(self, '_kl_samples_computed'):
                        step_kl_samples_computed += self._kl_samples_computed
                    if hasattr(self, '_kl_samples_skipped'):
                        step_kl_samples_skipped += self._kl_samples_skipped
                    if hasattr(self, '_kl_tokens_computed'):
                        step_kl_tokens_computed += self._kl_tokens_computed
                    if hasattr(self, '_kl_tokens_skipped'):
                        step_kl_tokens_skipped += self._kl_tokens_skipped
                    
                    if getattr(self, "_last_view_type_loss", None):
                        if view_type_stats_local is None:
                            view_type_stats_local = self._last_view_type_loss.copy()
                            for vt, stats in view_type_stats_local.items():
                                stats["loss"] = float(stats.get("loss", 0.0))
                        else:
                            for vt, stats in self._last_view_type_loss.items():
                                try:
                                    view_type_stats_local[vt]["loss"] += float(stats.get("loss", 0.0))
                                except Exception as e:
                                    view_type_stats_local[vt] = {"loss": float(stats.get("loss", 0.0)), "token_count": float(stats.get("token_count", 0.0)), "count": int(stats.get("count", 0))}
                    
                    self._last_view_type_loss = {}
                    self._last_loss_wo_normalize = None
                except Exception as e:
                    # Debug: print batch info when error occurs
                    print(f"\n❌ Error in micro_batch {micro_idx} on rank {self.device_mesh.get_rank()}:")
                    print(f"  Batch size: {micro_batch['input_ids'].shape[0] if 'input_ids' in micro_batch else 'N/A'}")
                    print(f"  Seq len: {micro_batch['input_ids'].shape[1] if 'input_ids' in micro_batch and micro_batch['input_ids'].dim() > 1 else 'N/A'}")
                    print(f"  Attention mask shape: {micro_batch['attention_mask'].shape if 'attention_mask' in micro_batch else 'N/A'}")
                    if 'multi_modal_inputs' in micro_batch:
                        mm = micro_batch['multi_modal_inputs']
                        if isinstance(mm, list) and len(mm) > 0 and isinstance(mm[0], dict):
                            print(f"  Multi-modal inputs: {len(mm)} samples")
                            if 'pixel_values' in mm[0]:
                                print(f"  First sample pixel_values: {mm[0]['pixel_values'].shape}")
                            if 'image_grid_thw' in mm[0]:
                                print(f"  First sample image_grid_thw: {mm[0]['image_grid_thw'].shape}")
                    print(f"  Error: {type(e).__name__}: {str(e)}")
                    raise  # Re-raise to see full stack trace
            
            # Debug: Print if loss is abnormally high (disabled - loss variance is expected)
            # if loss_value > 2.0:
            #     print(f"\n⚠️ [Micro-batch {micro_idx}] 异常高 loss: {loss_value:.3f}")

        # Ensure all ranks complete gradient accumulation before clipping
        # This prevents some ranks from entering optimizer.step() before others finish backward()
        torch.distributed.barrier(device_ids=[get_device_id()] if is_cuda_available else None)
        
        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        # IMPORTANT: All ranks must execute same branch (step or skip)
        if not torch.isfinite(grad_norm):
            print(f"WARN: Rank {self.device_mesh.get_rank()} grad_norm is not finite: {grad_norm}")
            # Broadcast decision to all ranks to ensure consistency
            skip_update = torch.tensor(1, device=self.device_name)
            torch.distributed.all_reduce(skip_update, op=torch.distributed.ReduceOp.MAX)
            
            if skip_update.item() > 0:
                # All ranks skip together
                self.optimizer.zero_grad()
                if self.device_mesh.get_rank() == 0:
                    print(f"⚠️ Skipping optimizer step due to non-finite grad_norm on some ranks")
            else:
                self.optimizer.step()
        else:
            # Check if any rank has non-finite grad
            skip_update = torch.tensor(0, device=self.device_name)
            torch.distributed.all_reduce(skip_update, op=torch.distributed.ReduceOp.MAX)
            
            if skip_update.item() > 0:
                # Some other rank has non-finite grad, skip together
                self.optimizer.zero_grad()
                if self.device_mesh.get_rank() == 0:
                    print(f"⚠️ Skipping optimizer step due to non-finite grad_norm on other ranks")
            else:
                # All ranks have finite grad, step together
                self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)
        step_kl_loss_tensor = torch.tensor(step_kl_loss).to(self.device_name) if self.enable_distillation else None
        step_loss_wo_normalize = torch.tensor(step_loss_wo_normalize).to(self.device_name)
        if view_type_stats_local is None:
            view_type_stats_local = {}
        
        for vt, stats in view_type_stats_local.items():
            stats["is_valid"] = torch.tensor(1.0).to(self.device_name)

        # Collect all view_type keys across ranks so that every rank runs collectives in identical order.
        # Use try-except to handle potential deadlock/timeout issues
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_keys = [None for _ in range(world_size)]
            local_keys = tuple(view_type_stats_local.keys())
            try:
                # all_gather_object can be slow/hang on large scale (512+ GPUs)
                # Add timeout protection or use async version if available
                torch.distributed.all_gather_object(gathered_keys, local_keys)
            except Exception as e:
                # If all_gather_object fails, fall back to using only local keys
                # This prevents deadlock but may cause inconsistent metrics across ranks
                if self.device_mesh.get_rank() == 0:
                    print(f"⚠️ Warning: all_gather_object failed: {e}. Using local view_type keys only.")
                gathered_keys = [local_keys] * world_size
            
            all_keys = set()
            for keys in gathered_keys:
                if keys:
                    all_keys.update(keys)
            view_type_keys = sorted(all_keys)
            for vt in view_type_keys:
                if vt not in view_type_stats_local:
                    view_type_stats_local[vt] = {"loss_sum": 0.0, "token_count": 0.0, "count": 0, "loss": 0.0, "is_valid": torch.tensor(0.0).to(self.device_name)}
        else:
            view_type_keys = sorted(view_type_stats_local.keys())

        for vt in view_type_keys:
            stats = view_type_stats_local[vt]
            for key in ["loss_sum", "token_count", "count", "loss", "is_valid"]:
                value = stats.get(key, 0.0)
                if isinstance(value, torch.Tensor):
                    stats[key] = value.to(self.device_name, dtype=torch.float32)
                else:
                    stats[key] = torch.tensor(value, device=self.device_name, dtype=torch.float32)

        # compute time spent per step
        end_time = time.time()
        spend_time_per_step = end_time - start_time

        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(step_loss_wo_normalize, op=torch.distributed.ReduceOp.AVG)
            for vt in view_type_keys:
                stats = view_type_stats_local[vt]
                torch.distributed.all_reduce(stats["loss"], op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(stats["is_valid"], op=torch.distributed.ReduceOp.SUM)
            if step_kl_loss_tensor is not None:
                torch.distributed.all_reduce(step_kl_loss_tensor, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
            if step_kl_loss_tensor is not None:
                torch.distributed.all_reduce(step_kl_loss_tensor)
                step_kl_loss_tensor /= self.device_mesh.size(0)
        
        # Cancel timeout alarm (on all ranks, not just rank 0)
        timeout_seconds = getattr(self.config.trainer, 'step_timeout_seconds', None)
        if timeout_seconds is not None and timeout_seconds > 0:
            try:
                import signal
                signal.alarm(0)
            except (ValueError, OSError):
                pass  # Ignore if signal not supported
        
        metrics = {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/time(s)": spend_time_per_step,
            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "train/loss_wo_kl&norm": step_loss_wo_normalize.detach().item(),
        }

        for vt in view_type_keys:
            stats = view_type_stats_local[vt]
            is_valid_cnt = stats["is_valid"].detach()
            if is_valid_cnt.item() > 0:
                avg_loss = stats["loss"] / is_valid_cnt
                metrics[f"train/loss_by_{vt}"] = avg_loss.detach().item()
            else:
                metrics[f"train/loss_by_{vt}"] = 0.0
        
        # Add KL loss to metrics if distillation is enabled
        if self.enable_distillation and step_kl_loss_tensor is not None:
            metrics["train/kl_loss"] = step_kl_loss_tensor.detach().item()
            
            # Add KL statistics (accumulated across all micro_batches)
            total_samples = step_kl_samples_computed + step_kl_samples_skipped
            if total_samples > 0:
                metrics["train/kl_samples_computed"] = step_kl_samples_computed
                metrics["train/kl_samples_skipped"] = step_kl_samples_skipped
                metrics["train/kl_compute_ratio"] = step_kl_samples_computed / total_samples
            
            # Add token-level statistics if available (for packed sequences)
            total_tokens = step_kl_tokens_computed + step_kl_tokens_skipped
            if total_tokens > 0:
                metrics["train/kl_tokens_computed"] = step_kl_tokens_computed
                metrics["train/kl_tokens_skipped"] = step_kl_tokens_skipped
                metrics["train/kl_token_compute_ratio"] = step_kl_tokens_computed / total_tokens
        
        # Add sequence packing statistics if available
        if hasattr(self, '_packing_efficiency'):
            metrics["train/packing_efficiency"] = self._packing_efficiency
        if hasattr(self, '_packing_num_packs'):
            metrics["train/packing_num_packs"] = self._packing_num_packs
        
        # Add dynamic packing statistics if available
        if self._use_dynamic_packing and self._total_batches_processed > 0:
            avg_samples_per_batch = self._total_samples_processed / self._total_batches_processed
            metrics["train/samples_processed"] = self._total_samples_processed
            metrics["train/avg_samples_per_batch"] = avg_samples_per_batch
            # Packing ratio: how many more samples we process compared to batch size
            target_batch_size = self.config.data.train_batch_size
            packing_ratio = avg_samples_per_batch / target_batch_size if target_batch_size > 0 else 1.0
            metrics["train/packing_ratio"] = packing_ratio
        
        return metrics

    def validation_step(self, batch):
        """Validation step - accepts both TensorDict and plain dict"""
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # Use checkpoint manager to save
        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state (only for non-WebDataset)
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            
            # Only save state for StatefulDataLoader (WebDataset uses normal DataLoader)
            if not self._is_webdataset(self.train_dataset):
                dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
                # Use StatefulDataLoader's built-in state dict functionality
                dataloader_state_dict = self.train_dataloader.state_dict()
                torch.save(dataloader_state_dict, dataloader_local_path)
                print(f"Saved dataloader state to: {dataloader_local_path}")
            else:
                print("WebDataset: skip saving dataloader state")

            # Update latest checkpoint tracker (atomic write)
            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier(device_ids=[get_device_id()] if is_cuda_available else None)

    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager with proper configuration"""
        # Get checkpoint configuration from config, with defaults
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # Set default values if not specified
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # Create checkpoint config dict
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)

        # Get fsdp_size for HYBRID_SHARD checkpoint optimization
        fsdp_size = self.config.model.fsdp_config.get("fsdp_size", -1)
        
        # Initialize checkpoint manager
        # Use processor if available (for VL models), otherwise use tokenizer
        processing_class = self.processor if self.processor is not None else self.tokenizer
        
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config_dict,
            fsdp_size=fsdp_size,
        )

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint (only for non-WebDataset)"""
        # WebDataset skip loading dataloader state
        if self._is_webdataset(self.train_dataset):
            log_with_rank(
                "WebDataset: skip loading dataloader state",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )
            return
        
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _validate_first_batch(self, batch):
        """Validate the first batch to ensure data quality
        
        Checks:
        1. Batch shapes are correct
        2. Loss mask distribution is reasonable
        3. Data types are correct
        """
        print("\n" + "=" * 100)
        print("🔍 FIRST BATCH VALIDATION CHECKPOINT")
        print("=" * 100)
        
        # Get batch info
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        loss_mask = batch.get("loss_mask")
        position_ids = batch.get("position_ids")
        multi_modal_inputs = batch.get("multi_modal_inputs", [])
        
        batch_size = input_ids.shape[0] if input_ids is not None else 0
        seq_len = input_ids.shape[1] if input_ids is not None and input_ids.dim() > 1 else input_ids.shape[0]
        
        print(f"\n【Batch 基本信息】")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Loss mask shape: {loss_mask.shape}")
        print(f"  Position IDs shape: {position_ids.shape if position_ids is not None else 'None'}")
        
        # Check loss_mask distribution for each sample in batch
        print(f"\n【Loss Mask 分布（每个样本）】")
        print(f"  {'Sample':<10} {'Total':<10} {'Mask=0':<15} {'Mask=1':<15} {'Ratio':<10}")
        print(f"  {'-'*10} {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
        
        for i in range(min(batch_size, 5)):  # 只检查前 5 个样本
            sample_loss_mask = loss_mask[i] if loss_mask.dim() > 1 else loss_mask
            total_tokens = sample_loss_mask.shape[0]
            mask_zeros = (sample_loss_mask == 0).sum().item()
            mask_ones = (sample_loss_mask == 1).sum().item()
            ratio = mask_ones / total_tokens if total_tokens > 0 else 0
            
            print(f"  {i:<10} {total_tokens:<10} {mask_zeros:<15} {mask_ones:<15} {ratio*100:<9.1f}%")
        
        if batch_size > 5:
            print(f"  ... (省略 {batch_size - 5} 个样本)")
        
        # Overall statistics
        total_mask_zeros = (loss_mask == 0).sum().item()
        total_mask_ones = (loss_mask == 1).sum().item()
        total_tokens = loss_mask.numel()
        
        print(f"\n【整体 Loss Mask 统计】")
        print(f"  总 tokens: {total_tokens}")
        print(f"  Mask=0 (prompt): {total_mask_zeros} ({total_mask_zeros/total_tokens*100:.1f}%)")
        print(f"  Mask=1 (answer): {total_mask_ones} ({total_mask_ones/total_tokens*100:.1f}%)")
        
        # Check multi_modal_inputs
        if multi_modal_inputs:
            print(f"\n【Multi-modal Inputs】")
            if isinstance(multi_modal_inputs, list):
                print(f"  数量: {len(multi_modal_inputs)} (list)")
                if len(multi_modal_inputs) > 0:
                    first_mm = multi_modal_inputs[0]
                    if isinstance(first_mm, dict):
                        for key, value in first_mm.items():
                            if isinstance(value, torch.Tensor):
                                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(multi_modal_inputs, dict):
                for key, value in multi_modal_inputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Decode first sample for visual inspection
        print(f"\n【第一个样本解码】")
        first_input_ids = input_ids[0] if input_ids.dim() > 1 else input_ids
        first_loss_mask = loss_mask[0] if loss_mask.dim() > 1 else loss_mask
        
        # Find answer start
        answer_start = None
        for i in range(len(first_loss_mask)):
            if first_loss_mask[i].item() == 1:
                answer_start = i
                break
        
        if answer_start is not None:
            # Decode prompt part
            prompt_tokens = first_input_ids[:answer_start]
            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            print(f"  Prompt ({len(prompt_tokens)} tokens):")
            print(f"    前 300 字符: {repr(prompt_text[:300])}...")
            print(f"    后 200 字符: ...{repr(prompt_text[-200:])}")
            
            # Decode answer part
            answer_tokens = first_input_ids[answer_start:]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=False)
            print(f"  Answer ({len(answer_tokens)} tokens):")
            print(f"    前 300 字符: {repr(answer_text[:300])}...")
            if len(answer_text) > 300:
                print(f"    后 200 字符: ...{repr(answer_text[-200:])}")
        
        # Validation checks
        print(f"\n【验证检查】")
        checks_passed = True
        
        # Check 1: Shapes consistency
        if input_ids.shape[:-1] != attention_mask.shape[:-1]:
            print(f"  ❌ FAIL: input_ids 和 attention_mask shape 不一致")
            checks_passed = False
        else:
            print(f"  PASS: Batch shapes 一致")
        
        # Check 2: Loss mask should have both 0s and 1s
        if total_mask_ones == 0:
            print(f"  ❌ FAIL: Loss mask 全是 0，没有计算 loss 的 tokens！")
            checks_passed = False
        elif total_mask_zeros == 0:
            print(f"  ⚠️  WARNING: Loss mask 全是 1，prompt 没有被 mask")
        else:
            print(f"  PASS: Loss mask 同时包含 prompt (0) 和 answer (1)")
        
        # Check 3: Answer ratio should be reasonable
        answer_ratio = total_mask_ones / total_tokens
        if answer_ratio < 0.05:
            print(f"  ⚠️  WARNING: Answer 部分太小 ({answer_ratio*100:.1f}%)")
        elif answer_ratio > 0.95:
            print(f"  ⚠️  WARNING: Answer 部分太大 ({answer_ratio*100:.1f}%)")
        else:
            print(f"  PASS: Answer 部分占比合理 ({answer_ratio*100:.1f}%)")
        
        # Check 4: Position IDs
        if position_ids is not None:
            if position_ids.dim() == 3:
                print(f"  PASS: Position IDs 是 3D (VL 模型特殊格式)")
            elif position_ids.dim() == 2:
                print(f"  PASS: Position IDs 是 2D (标准格式)")
            else:
                print(f"  ⚠️  WARNING: Position IDs 维度异常: {position_ids.dim()}D")
        
        if checks_passed:
            print(f"\n第一个 Batch 验证通过")
        else:
            print(f"\n❌ 第一个 Batch 验证失败，请检查数据处理流程")
        
        print("=" * 100 + "\n")
    
    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None
        
        # total_training_steps 已经在 _build_model_optimizer 中设置了
        self.total_training_steps = self.total_steps
        log_with_rank(
            f"Total training steps: {self.total_training_steps}",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"Resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        # For WebDataset, steps_per_epoch may not exist, set start_epoch to 0
        if hasattr(self, 'steps_per_epoch'):
            start_epoch = global_step // self.steps_per_epoch
        else:
            start_epoch = 0

        train_time = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            # Only call set_epoch for DistributedSampler (WebDataset's sampler is None)
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch=epoch)

            # Build tqdm parameters
            tqdm_kwargs = {
                "desc": f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                "disable": rank != 0,
            }
            # If steps_per_epoch exists, add initial and total (normal dataset)
            if hasattr(self, 'steps_per_epoch'):
                tqdm_kwargs["initial"] = global_step % self.steps_per_epoch if epoch == start_epoch else 0
                tqdm_kwargs["total"] = self.steps_per_epoch
            
            for step_in_epoch, data in enumerate(tqdm(self.train_dataloader, **tqdm_kwargs)):
                # Heartbeat: Print every 10 steps to detect if stuck
                if global_step % 10 == 0 and rank == 0:
                    import datetime
                    now = datetime.datetime.now().strftime("%H:%M:%S")
                    
                    # Add dynamic packing info to heartbeat if enabled
                    heartbeat_msg = f"[{now}]  Heartbeat: step {global_step}"
                    if self._use_dynamic_packing and self._total_batches_processed > 0:
                        avg_samples = self._total_samples_processed / self._total_batches_processed
                        packing_ratio = avg_samples / self.config.data.train_batch_size
                        heartbeat_msg += f" | Samples: {self._total_samples_processed} ({avg_samples:.1f}/batch, {packing_ratio:.2f}x)"
                    print(heartbeat_msg)
                
                # Increment global_step
                global_step += 1
                
                # Extract dynamic packing statistics if available
                # Check directly for stats presence (don't rely on flag)
                if "_dynamic_packing_stats" in data:
                    # Enable dynamic packing flag if stats exist (auto-detection)
                    if not self._use_dynamic_packing:
                        self._use_dynamic_packing = True
                        if rank == 0:
                            print("Auto-detected dynamic packing from batch stats")
                    
                    packing_stats = data["_dynamic_packing_stats"]
                    samples_in_batch = packing_stats.get("samples_in_this_batch", 0)
                    self._total_samples_processed += samples_in_batch
                    self._total_batches_processed += 1
                    # Remove the stats from data to avoid passing it to the model
                    del data["_dynamic_packing_stats"]
                elif global_step <= 5 and rank == 0:
                    # Debug: stats key missing in first few steps
                    print(f"Step {global_step}: No dynamic packing stats (using standard dataloader)")
                
                # Log first batch for verification
                if global_step == 1 and rank == 0:
                    batch_size = data["input_ids"].shape[0] if "input_ids" in data else 0
                    seq_len = data["input_ids"].shape[1] if data["input_ids"].dim() == 2 else data["input_ids"].shape[0]
                    print(f"First batch: batch_size={batch_size}, seq_len={seq_len}")
                
                # For VL models, check if position_ids has special shape
                # Qwen2-VL: [4, batch, seq] where first dim is NOT batch_size
                position_ids_shape = data.get("position_ids").shape if "position_ids" in data else None
                use_tensordict = True
                
                # Get actual batch size from data
                actual_batch_size = data["input_ids"].shape[0] if "input_ids" in data else 0
                
                # Check if position_ids is multi-dimensional and first dim != batch_size
                if position_ids_shape is not None and len(position_ids_shape) == 3:
                    # Shape is [4, batch, seq] or similar - special VL format
                    if position_ids_shape[0] != actual_batch_size and position_ids_shape[1] == actual_batch_size:
                        # Special position_ids format (like Qwen2-VL), skip TensorDict
                        use_tensordict = False
                
                # Detect non-tensor fields (lists/dicts). TensorDict can't store them without shape info.
                has_non_tensor_fields = any(not isinstance(v, torch.Tensor) for v in data.values())
                if has_non_tensor_fields:
                    use_tensordict = False

                if not use_tensordict:
                    # Manually move to device
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data[key] = value.to(self.device_name)
                        elif key == "multi_modal_inputs" and isinstance(value, list):
                            # Keep multi_modal_inputs as is
                            pass
                else:
                    # Standard case: use TensorDict with actual batch size
                    # TensorDict only accepts tensor fields; keep non-tensor fields separately
                    tensor_fields = {}
                    non_tensor_fields = {}
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            tensor_fields[key] = value.to(self.device_name)
                        else:
                            non_tensor_fields[key] = value

                    data = TensorDict(tensor_fields, batch_size=actual_batch_size).to(self.device_name)

                    # Add back non-tensor fields (lists, bool flags, etc.)
                    for key, value in non_tensor_fields.items():
                        data[key] = value
                
                # continue
                metric = self.training_step(data)
                train_time += metric["train/time(s)"]
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                    # if global_step <= 5:
                    #     print(f"[DEBUG] Step {global_step} completed in {metric['train/time(s)']:.2f}s")

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                if (self.config.trainer.save_freq > 0 and is_save_step):
                    # Sync before checkpoint to ensure all ranks ready
                    if rank == 0:
                        print(f"[Step {global_step}] Starting checkpoint save...")
                    torch.distributed.barrier(device_ids=[get_device_id()] if is_cuda_available else None)
                    
                    self.save_checkpoint(step=global_step)
                    
                    # Sync after checkpoint
                    if rank == 0:
                        print(f"[Step {global_step}] Checkpoint saved successfully")
                    torch.distributed.barrier(device_ids=[get_device_id()] if is_cuda_available else None)


                # early exit or validation step
                # if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                #     # Sync before validation to ensure all ranks finish training step
                #     if rank == 0:
                #         print(f"[Step {global_step}] Starting validation...")
                #     torch.distributed.barrier(device_ids=[get_device_id()] if is_cuda_available else None)
                    
                #     # Perform validation
                #     val_losses = []
                #     for val_data in self.val_dataloader:
                #         # Check if position_ids has special shape (VL models)
                #         val_position_ids_shape = val_data.get("position_ids").shape if "position_ids" in val_data else None
                #         val_use_tensordict = True
                        
                #         # Get actual batch size from validation data
                #         val_actual_batch_size = val_data["input_ids"].shape[0] if "input_ids" in val_data else 0
                        
                #         if val_position_ids_shape is not None and len(val_position_ids_shape) == 3:
                #             # Shape is [4, batch, seq] - special VL format
                #             if val_position_ids_shape[0] != val_actual_batch_size and val_position_ids_shape[1] == val_actual_batch_size:
                #                 val_use_tensordict = False
                        
                #         has_non_tensor_fields = any(not isinstance(v, torch.Tensor) for v in val_data.values())
                #         if has_non_tensor_fields:
                #             val_use_tensordict = False

                #         if not val_use_tensordict:
                #             # Special position_ids, skip TensorDict
                #             for key, value in val_data.items():
                #                 if isinstance(value, torch.Tensor):
                #                     val_data[key] = value.to(self.device_name)
                #         else:
                #             # Standard case, use TensorDict with actual batch size
                #             tensor_fields = {}
                #             non_tensor_fields = {}
                #             for key, value in val_data.items():
                #                 if isinstance(value, torch.Tensor):
                #                     tensor_fields[key] = value.to(self.device_name)
                #                 else:
                #                     non_tensor_fields[key] = value

                #             val_data = TensorDict(tensor_fields, batch_size=val_actual_batch_size).to(
                #                 self.device_name
                #             )

                #             for key, value in non_tensor_fields.items():
                #                 val_data[key] = value
                        
                #         val_loss = self.validation_step(val_data)
                #         val_losses.append(val_loss)
                #     if rank == 0:
                #         val_loss = torch.mean(torch.stack(val_losses))
                #         metric = {"val/loss": val_loss.detach().item()}
                #         tracking.log(data=metric, step=global_step)
                #         last_valid_metric = metric
                #     torch.distributed.barrier(device_ids=[get_device_id()] if is_cuda_available else None)


                # if is_last_step:
                #     if rank == 0:
                #         print(f"Total time for train steps: {train_time:.2f}s")
                #         print(f"Final validation metrics: {last_valid_metric}")
                #     return


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    # Use create_device_mesh to support flexible FSDP configuration
    # If fsdp_size is not specified or >= world_size, use FULL_SHARD (all GPUs in one FSDP group)
    # Otherwise, use HYBRID_SHARD (multiple FSDP groups, each with fsdp_size GPUs)
    # This allows better performance for large-scale training by reducing cross-node communication
    from verl.workers.engine.fsdp.utils import create_device_mesh
    
    fsdp_size = config.model.fsdp_config.get("fsdp_size", -1)
    device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
    
    # Log device mesh configuration
    if rank == 0:
        if device_mesh.ndim == 1:
            print(f"Using FULL_SHARD: All {world_size} GPUs in one FSDP group")
        elif device_mesh.ndim == 2:
            num_fsdp_groups = world_size // fsdp_size
            print(f"Using HYBRID_SHARD: {num_fsdp_groups} FSDP groups, each with {fsdp_size} GPUs")
    
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer, hf_processor

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    trust_remote_code = config.model.trust_remote_code
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=trust_remote_code)
    # For multimodal models, we need a processor
    processor = hf_processor(local_model_path, trust_remote_code=trust_remote_code, use_fast=True)
    
    # Create dataset based on file type
    train_dataset = create_dataset_by_type(
        config.data.train_files, config.data, tokenizer, processor=processor, max_samples=config.data.get("train_max_samples", -1)
    )
    val_dataset = create_dataset_by_type(
        config.data.val_files, config.data, tokenizer, processor=processor, max_samples=config.data.get("val_max_samples", -1)
    )

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        processor=processor,  # Pass processor for VL models
    )

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def detect_file_type(data_paths):
    """Detect data file type
    
    Args:
        data_paths: single file path, directory path, or list of file/directory paths
        Supports local paths, HDFS paths (hdfs://), and S3 paths (s3://)
        
    Returns:
        'tar' or 'parquet'
    """
    from omegaconf.listconfig import ListConfig
    import os
    from pathlib import Path
    import subprocess
    
    # Convert to list
    if isinstance(data_paths, str):
        paths = [data_paths]
    elif isinstance(data_paths, ListConfig):
        paths = list(data_paths)
    else:
        paths = data_paths
    
    if len(paths) == 0:
        return 'parquet'
    
    first_path = str(paths[0])
    
    # Check if it's a remote path (HDFS/S3)
    is_remote = first_path.startswith('hdfs://') or first_path.startswith('s3://')
    
    if is_remote:
        # For remote paths, check if it ends with a file extension
        if first_path.endswith('.tar'):
            return 'tar'
        elif first_path.endswith('.parquet') or first_path.endswith('.arrow'):
            return 'parquet'
        else:
            # Assume it's a directory, list files
            # Try to find tar files first
            cmd = f"hdfs dfs -ls {first_path} | head -20 | grep '\\.tar'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return 'tar'
            
            # Try to find parquet/arrow files
            cmd = f"hdfs dfs -ls {first_path} | head -20 | grep -E '\\.(parquet|arrow)'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return 'parquet'
            
            # Default to tar for remote directories (common case)
            return 'tar'
    else:
        # Local path
        # If it's a directory, search for files inside (recursively)
        if os.path.isdir(first_path):
            # Search for tar files first (recursively with rglob)
            tar_files = list(Path(first_path).rglob("*.tar"))
            if tar_files:
                print(f"Detected tar dataset: found {len(tar_files)} tar files in {first_path}")
                return 'tar'
            
            # Then search for parquet/arrow files (recursively)
            parquet_files = list(Path(first_path).rglob("*.parquet")) + list(Path(first_path).rglob("*.arrow"))
            if parquet_files:
                print(f"Detected parquet/arrow dataset: found {len(parquet_files)} files in {first_path}")
                return 'parquet'
            
            # If no files found, default to parquet
            print(f"Warning: No tar/parquet/arrow files found in {first_path}, defaulting to parquet")
            return 'parquet'
        else:
            # If it's a file, check extension
            if first_path.endswith('.tar') or '.tar' in first_path:
                return 'tar'
            elif first_path.endswith('.parquet') or first_path.endswith('.arrow'):
                return 'parquet'
    
    # Default return parquet
    return 'parquet'


def create_dataset_by_type(data_paths, data_config, tokenizer, processor=None, max_samples=-1):
    """Create dataset based on file type
    
    Args:
        data_paths: data file path
        data_config: data configuration
        tokenizer: tokenizer
        processor: processor for multimodal models (optional)
        max_samples: maximum number of samples
        
    Returns:
        Dataset object
    """
    file_type = detect_file_type(data_paths)
    
    if file_type == 'tar':
        print(f"tar using create_webdataset")
        return create_tar_dataset(data_paths, data_config, tokenizer, processor, max_samples)
    else:
        print(f"parquet/arrow using create_sft_dataset")
        #TODO: add processor for parquet/arrow files
        return create_sft_dataset(data_paths, data_config, tokenizer, max_samples)


def create_sft_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    """Create a dataset for parquet/arrow files
    
    Args:
        data_paths: parquet/arrow file paths or directory containing parquet/arrow files
        data_config: data configuration
        tokenizer: tokenizer
        max_samples: maximum number of samples
        
    Returns:
        SFTDataset object
    """
    from omegaconf.listconfig import ListConfig
    from pathlib import Path
    import os
    
    # Convert to list
    if isinstance(data_paths, str):
        paths = [data_paths]
    elif isinstance(data_paths, ListConfig):
        paths = list(data_paths)
    else:
        paths = data_paths
    
    # Collect all parquet/arrow files
    file_list = []
    for path in paths:
        if os.path.isdir(path):
            # If it's a directory, search for all parquet/arrow files
            parquet_files = list(Path(path).glob("*.parquet"))
            arrow_files = list(Path(path).glob("*.arrow"))
            all_files = [str(f) for f in parquet_files + arrow_files]
            all_files = sorted(all_files)
            file_list.extend(all_files)
        else:
            # If it's a file, add directly
            file_list.append(path)
    
    if len(file_list) == 0:
        raise ValueError(f"No parquet/arrow files found in: {data_paths}")
    
    print(f"Found {len(file_list)} parquet/arrow files")
    
    # Convert to ListConfig to avoid double-wrapping in SFTDataset
    # SFTDataset expects either a string or ListConfig
    from omegaconf import OmegaConf
    if len(file_list) == 1:
        # Single file: pass as string
        files_to_pass = file_list[0]
    else:
        # Multiple files: convert to ListConfig
        files_to_pass = OmegaConf.create(file_list)
    
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=files_to_pass, tokenizer=tokenizer, config=data_config, max_samples=max_samples)
    return dataset


def create_tar_dataset(data_paths, data_config, tokenizer, processor=None, max_samples=-1):
    """create dataset based on tar files (using webdataset)
    
    Args:
        data_paths: tar files or directory containing tar files
        data_config: data config
        tokenizer: tokenizer
        processor: processor for multimodal models (optional)
        max_samples: max samples (not supported)
        
    Returns:
        TarDataset (wraps WebDataset with tokenization and image processing)
    """
    from verl.utils.dataset.tar_dataset import TarDataset
    
    # Use TarDataset class which handles WebDataset creation, tokenization, and image processing
    dataset = TarDataset(
        tar_files=data_paths, 
        tokenizer=tokenizer, 
        processor=processor,
        config=data_config, 
        max_samples=max_samples
    )
    
    return dataset


if __name__ == "__main__":
    main()
