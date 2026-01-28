export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 限制使用这8个GPU

ngpu=8
ncpu=16
WANDB_PROJECT=llm-cot-safety
config_file=config/deepspeed_zero2.yaml

export TOKENIZERS_PARALLELISM=false
model_name=llama_8b

train_task_names=reflect_cot_1k_500harm_200help_300math_lr5e-6_epoch5_warmup0.05
base_dir=/models/reflect_results/model:sft_mllm_${model_name}/train:${train_task_names}

echo "training..."
echo "run_name: $base_dir"
mkdir -p $base_dir

per_device_bs=2  # 8B模型，每卡2个样本
grad_accum_steps=2  # 累计步数减少
total_samples=1000
batch_size=$((per_device_bs * grad_accum_steps * ngpu))
steps_per_epoch=$((total_samples / batch_size))
total_training_steps=$((steps_per_epoch * 4))  # 15个epoch

echo "Effective batch size: $batch_size"
echo "Steps per epoch: $steps_per_epoch"
echo "Total training steps: $total_training_steps"

accelerate launch --config_file ${config_file} --num_processes ${ngpu} --main_process_port $(( RANDOM % 1000 + 30000 )) \
    sft_llm_cot.py \
   --output_dir ${base_dir} \
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
    --logging_dir ${base_dir}/logs \
    --report_to "none" \
    --dataloader_num_workers 8 \
    --gradient_checkpointing \
    --greater_is_better false \
    --group_by_length \
    
