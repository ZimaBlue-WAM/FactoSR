cd /pfs/yijun/project/FactoSR/run_exp
export WANDB_API_KEY="YOUR_WANDB_API_KEY"

export PYTHONPATH=/pfs/yijun/project/FactoSR:$PYTHONPATH

GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

echo "=============================="
echo "Run started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Stage 1: Short-Answer SFT"
echo "=============================="

source /pfs/yijun/env/verl/bin/activate && torchrun \
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  -m verl.trainer.fsdp_sft_trainer \
    data.prompt_key=question \
    data.response_key=answer \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=8192 \
    model.partial_pretrain=/path/to/base_model \
    trainer.project_name=qa-sft-main-qwen3 \
    trainer.experiment_name=qa-sft-main-qwen3-128-full \
    trainer.total_epochs=4 \
    trainer.logger='["console","wandb"]' \
    optim.lr=5e-5 \
    optim.num_warmup_steps=400 \
    trainer.total_training_steps=4000 \
    trainer.save_freq=4000 \
    model.distillation.enable=True \
    model.distillation.teacher_model_path=/path/to/base_model \

sleep infinity
