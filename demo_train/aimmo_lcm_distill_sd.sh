export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/sd"
export TRAIN_DIR="/data/noah/dataset/disney"
export TRAIN_CSV_PATH='/data/noah/dataset/disney/metadata.csv'
export OUTPUT_DIR="/data/noah/ckpt/finetuning/LCM_SD_DISNEY"
export HF_DATASETS_CACHE=$OUTPUT_DIR

accelerate launch examples/consistency_distillation/train_lcm_distill_sd.py \
  --pretrained_teacher_model=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --logging_dir=$HF_DATASETS_CACHE \
  --cache_dir=$HF_DATASETS_CACHE \
  --checkpointing_steps=5000 \
  --train_ann_path=$TRAIN_CSV_PATH \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=100000 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --loss_type="huber" \
  --ema_decay=0.95 \
  --adam_weight_decay=0.0 \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --validation_steps=1000 \
  --validation_prompts="a cartoon character" \
  --tracker_project_name="lcm-sc-desny-fine-tune" \
  --seed=453645634

    # --scale_lr \
