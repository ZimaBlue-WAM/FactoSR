set -x
export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO

# Increase NCCL timeout to handle slow network synchronization
export NCCL_TIMEOUT=3600

# Network stability settings
# Note: Use TORCH_NCCL_* prefixed variables instead of NCCL_* for PyTorch compatibility
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0  # Disable async error handling to catch errors early
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=""  # Auto-detect network interface

export TORCH_DISTRIBUTED_BACKEND=nccl
export TORCH_NCCL_BLOCKING_WAIT=1  # Block on NCCL operations for better error reporting

export WANDB_API_KEY="f0c63d2a0c288fd5fcae0e0a6b16d29cb23fb454"

export VLLM_USE_V1=1

ENGINE=${1:-vllm}

source /opt/conda/etc/profile.d/conda.sh

conda activate /pfs/zhengshenghe/env/verl_shenghe

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/pfs/zhengshenghe/Spatial_Reason/data/general_data/raw/geometry3k/verl_format/train.parquet \
    data.val_files=/pfs/zhengshenghe/Spatial_Reason/data/general_data/raw/geometry3k/verl_format/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/pfs/shared_models/Qwen/Qwen3-VL-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="${WANDB_API_KEY}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen3-vl-grpo-roll-10' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
