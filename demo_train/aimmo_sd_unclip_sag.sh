export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/unclip"
export TRAIN_DIR="/data/noah/dataset/AD_SD"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/SD_UNCLIP_SAG_MEAA"
export HF_DATASETS_CACHE=$TRAIN_DIR

accelerate launch examples/image_to_image/train_img2img_sd_unclip_sag.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --cache_dir=$TRAIN_DIR \
  --enable_xformers_memory_efficient_attention \
  --resolution=768 --center_crop --random_flip \
  --checkpointing_steps=5000 \
  --max_train_steps=100000 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR