export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/sdxl"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export TRAIN_DIR="/data/noah/dataset/AD"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/SDXL_AD"
export HF_DATASETS_CACHE=$TRAIN_DIR

accelerate launch examples/text_to_image/train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$TRAIN_DIR \
  --cache_dir=$TRAIN_DIR \
  --enable_xformers_memory_efficient_attention \
  --resolution=1024 --center_crop --random_flip \
  --checkpointing_steps=1 \
  --max_train_steps=1 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint="checkpoint-20000"