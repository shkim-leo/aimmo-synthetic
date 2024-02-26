export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/inpaint"
export TRAIN_DIR="/data/noah/dataset/coco"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/controlnet_inpaint_coco"
export HF_DATASETS_CACHE=$OUTPUT_DIR

accelerate launch examples/controlnet/train_controlnet_sd_inpainting.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --enable_xformers_memory_efficient_attention \
  --resolution=768 --center_crop \
  --checkpointing_steps=2000 \
  --max_train_steps=200000 \
  --validation_prompts="person" \
  --validation_epochs=1 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR
  # --resume_from_checkpoint='checkpoint-2000'