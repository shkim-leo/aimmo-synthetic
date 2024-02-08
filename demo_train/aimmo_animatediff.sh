export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/sd"
export MODEL_ANI_NAME="/data/noah/ckpt/finetuning/ANI_AD" #64000
export TRAIN_PATH="/data/noah/dataset/AD_SEQ/metadata.csv"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/ANI_AD"
export HF_DATASETS_CACHE=$TRAIN_DIR

accelerate launch examples/animatediff/train_animatediff.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_animatediff_model_name_or_path=$MODEL_ANI_NAME \
  --train_ann_path=$TRAIN_PATH \
  --cache_dir=$TRAIN_PATH \
  --validation_prompt="cars are driving on the highway" \
  --max_train_steps=46000 \
  --output_dir=$OUTPUT_DIR \
  --resolution=320 \
  --sample_n_frames=16 \
  --sample_stride=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=3e-5 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_8bit_adam \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --checkpointing_steps=1000 \
  --enable_xformers_memory_efficient_attention
  # --resume_from_checkpoint="checkpoint-12500"