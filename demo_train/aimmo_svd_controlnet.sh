export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/svd"
export TRAIN_PATH="/data/noah/dataset/AD_SEQ/metadata.csv"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/SVD_CON_AD_SEQ"
export HF_DATASETS_CACHE=$TRAIN_DIR

accelerate launch examples/stable_video_diffusion/train_svd_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_ann_path=$TRAIN_PATH \
  --max_train_steps=1 \
  --output_dir=$OUTPUT_DIR \
  --height=320 \
  --width=512 \
  --num_frames=8 \
  --sample_stride=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=3e-5 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_8bit_adam \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --checkpointing_steps=1 \
  --enable_xformers_memory_efficient_attention \
  --validation_steps=1250
  # --resume_from_checkpoint="checkpoint-10000"