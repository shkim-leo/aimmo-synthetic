export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/rv"
export TRAIN_DIR="/data/noah/dataset/KOMATSU"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/LCM_LORA_KOMATSU"
export HF_DATASETS_CACHE=$TRAIN_DIR

accelerate launch examples/consistency_distillation/train_lcm_distill_lora_sd_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_shards_path_or_url=$TRAIN_DIR \
    --cache_dir=$TRAIN_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --lora_rank=64 \
    --learning_rate=1e-6 --loss_type="huber" --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --seed=453645634
