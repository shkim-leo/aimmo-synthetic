export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/modelscope_cerspense"
export TRAIN_DIR="/data/noah/dataset/AD_SEQ"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/TEXT2VID_AD"
export HF_DATASETS_CACHE=$TRAIN_DIR

accelerate launch examples/text_to_video/train_text2vid.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir=$TRAIN_DIR \
    --cache_dir=$TRAIN_DIR \
    --mixed_precision="fp16" \
    --center_crop \
    --learning_rate=1e-4 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --max_train_steps=50000 \
    --dataloader_num_workers=8 \
    --checkpointing_steps=5000 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --seed=42 \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention
    # --gradient_checkpointing \
