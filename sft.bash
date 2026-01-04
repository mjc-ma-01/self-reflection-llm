
ngpu=8
ncpu=16
WANDB_PROJECT=llm-cot-safety
config_file=config/deepspeed_zero2.yaml


model_name=llama_8b

train_task_names=reflect_cot_1k_w400_benign_lr3e-6_epoch5_warmup0.1
base_dir=/c23030/ckj/code/vlm/models/reflect_results/model:sft_mllm_${model_name}/train:${train_task_names}

echo "training..."
echo "run_name: $base_dir"
mkdir -p $base_dir

# # 计算合理的参数
# per_device_bs=2        # 增加每设备批量大小
# grad_accum_steps=2     # 减少梯度累积步数
# effective_bs=$((per_device_bs * grad_accum_steps * ngpu))  # 32
# steps_per_epoch=$((1200 / effective_bs))  # 37.5 ≈ 38

per_device_bs=1
grad_accum_steps=4  # 有效批量仍为32
steps_per_epoch=44


echo "Steps per epoch: $steps_per_epoch"

accelerate launch --config_file ${config_file} --num_processes ${ngpu} --main_process_port $(( RANDOM % 1000 + 30000 )) \
    sft_llm_cot.py \
   --output_dir ${base_dir} \
    --num_train_epochs 5 \
    --per_device_train_batch_size=${per_device_bs} \
    --gradient_accumulation_steps=${grad_accum_steps} \
    --learning_rate 3e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --bf16 \
    --logging_steps 10 \
    --eval_steps ${steps_per_epoch} \
    --save_steps ${steps_per_epoch} \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --logging_dir ${base_dir}/logs \
    --report_to "wandb" \
    --dataloader_num_workers 4 \
    --gradient_checkpointing \
    --greater_is_better false \
    --group_by_length \
    --max_steps=200 
    
                                      # 设置随机种子
    # --num_train_epochs 2 \
    # --per_device_train_batch_size=1 \
    # --gradient_accumulation_steps=1 \
    # --bf16 \
    # --logging_steps 10 \
    # --logging_strategy "steps" \
    # # --use_peft \
    # # --lora_target_modules  down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj 

    # --output_dir ${base_dir} \
    # --num_train_epochs 10 \
    # --per_device_train_batch_size=2 \
    # --gradient_accumulation_steps=2 \
    # --learning_rate 2e-5 \
    # --lr_scheduler_type "cosine" \
    # --warmup_ratio 0.1 \
    # --weight_decay 0.01 \
    # --max_grad_norm 1.0 \
    # --bf16 \
    # --logging_steps 5 \
    # --eval_steps 0.5 \
    # --save_steps 0.5 \
    # --save_strategy "steps" \
    # --load_best_model_at_end \
    # --metric_for_best_model "eval_loss" \
    # --greater_is_better false \



 