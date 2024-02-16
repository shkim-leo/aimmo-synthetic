export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/inpaint"
export TRAIN_DIR="/data/noah/dataset/AD_INPAINT"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/SD_INPAINT_AD"
export HF_DATASETS_CACHE=$OUTPUT_DIR

accelerate launch examples/inpainting/train_stablediffusion_inpainting.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --enable_xformers_memory_efficient_attention \
  --resolution=768 --center_crop \
  --checkpointing_steps=1000 \
  --max_train_steps=100000 \
  --validation_prompts="pedestrian on the driving road" \
  --validation_epochs=1 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR
  # --resume_from_checkpoint='checkpoint-5000'