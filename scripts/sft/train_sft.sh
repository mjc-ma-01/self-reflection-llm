#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

ngpu=${NGPU:-8}
ncpu=${NCPU:-16}
WANDB_PROJECT=${WANDB_PROJECT:-llm-cot-safety}
config_file=${CONFIG_FILE:-config/deepspeed_zero2.yaml}

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
model_name=${MODEL_NAME:-llama_8b}
model_path=${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}
sft_train_file=${SFT_TRAIN_FILE:-/mnt/shared-storage-user/majiachen/sft_source/sft/ready/train.jsonl}
sft_eval_file=${SFT_EVAL_FILE:-/mnt/shared-storage-user/majiachen/sft_source/sft/ready/test.jsonl}

train_task_names=${TRAIN_TASK_NAMES:-reflector_sft_ready_local_lr5e-6_epoch5_warmup0.05}
output_root=${OUTPUT_ROOT:-outputs/reflect_models}
base_dir=${output_root}/model:sft_mllm_${model_name}/train:${train_task_names}

echo "training..."
echo "run_name: $base_dir"
mkdir -p "${base_dir}"

per_device_bs=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
grad_accum_steps=${GRADIENT_ACCUMULATION_STEPS:-4}
if [[ -f "${sft_train_file}" ]]; then
  total_samples=${TOTAL_SAMPLES:-$(wc -l < "${sft_train_file}")}
else
  total_samples=${TOTAL_SAMPLES:-2800}
fi
batch_size=$((per_device_bs * grad_accum_steps * ngpu))
steps_per_epoch=$(( (total_samples + batch_size - 1) / batch_size ))
total_training_steps=$((steps_per_epoch * 5))

echo "Effective batch size: $batch_size"
echo "Steps per epoch: $steps_per_epoch"
echo "Total training steps: $total_training_steps"


accelerate launch --config_file "${config_file}" --num_processes "${ngpu}" --main_process_port $(( RANDOM % 1000 + 30000 )) \
    -m self_reflection_llm.training.sft \
    --model_path "${model_path}" \
    --train_file "${sft_train_file}" \
    --eval_file "${sft_eval_file}" \
    --output_dir "${base_dir}" \
    --num_train_epochs 5 \
    --per_device_train_batch_size=${per_device_bs} \
    --gradient_accumulation_steps=${grad_accum_steps} \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --lr_scheduler_kwargs '{"num_cycles": 0.5}' \
    --warmup_ratio 0.05 \
    --warmup_steps $((total_training_steps * 5 / 100)) \
    --weight_decay 0.05 \
    --max_grad_norm 0.3 \
    --bf16 \
    --logging_steps 10 \
    --eval_steps $((steps_per_epoch / 2)) \
    --save_steps $((steps_per_epoch / 2)) \
    --save_total_limit 5 \
    --load_best_model_at_end \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --logging_dir "${base_dir}/logs" \
    --report_to "none" \
    --dataloader_num_workers 4 \
    --gradient_checkpointing \
    --greater_is_better false \
    --group_by_length
