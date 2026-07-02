set -e
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO

# Increase NCCL timeout to handle slow network synchronization
export NCCL_TIMEOUT=3600

# Network stability settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=""

export TORCH_DISTRIBUTED_BACKEND=nccl
export TORCH_NCCL_BLOCKING_WAIT=1

export WANDB_API_KEY="YOUR_WANDB_API_KEY"

export VLLM_USE_V1=1
cd /pfs/yijun/project/FactoSR

export PYTHONPATH=/pfs/yijun/project/FactoSR:$PYTHONPATH

source /opt/conda/etc/profile.d/conda.sh
conda activate /path/to/your/env

project_name='verl_qwen3-8b'
exp_name='DAPO-8b-vl-tar-factorized-rewards'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 10))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=score
max_num_gen_batches=32
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=8
train_prompt_mini_bsz=16

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"ray://localhost:10001"}

NNODES=${NNODES:-8}
# Paths
MODEL_PATH=${MODEL_PATH:-"/path/to/cot_sft_checkpoint"}

CKPTS_DIR=${CKPTS_DIR:-"/path/to/rl_checkpoints/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"['/path/to/dataset1','/path/to/dataset2','/path/to/dataset3']"}
TEST_FILE=${TEST_FILE:-"/path/to/test.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=2
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2 ))
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2 ))
offload=False
tp_size=4

max_length=$((max_prompt_length + max_response_length))

TRAIN_SAMPLING_RATIOS=${TRAIN_SAMPLING_RATIOS:-"[1,1,1]"}
SAMPLING_RATIOS_ARG=""
if [ -n "${TRAIN_SAMPLING_RATIOS}" ]; then
    SAMPLING_RATIOS_ARG="+data.train_sampling_ratios=${TRAIN_SAMPLING_RATIOS}"
fi

python3 -m recipe.dapo.main_dapo \
    ${SAMPLING_RATIOS_ARG} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.image_key=images \
    data.max_prompt_length=${max_prompt_length} \
    data.filter_overlong_prompts=True \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.custom_cls.path='pkg://verl.utils.dataset.rl_tar_dataset' \
    data.custom_cls.name='RLTarDataset' \
    +data.max_length=${max_length} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=bf16 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=15 \
    trainer.total_training_steps=3000 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="${WANDB_API_KEY}" \
    $@
