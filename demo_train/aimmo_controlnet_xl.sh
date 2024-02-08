export MODEL_NAME_PATH="/root/nfs/noah/ckpt/pretrain_ckpt/StableDiffusion/rvxl"
export TRAIN_DIR="/root/nfs/noah/dataset/AD_ControlNet"
export OUTPUT_DIR="/root/nfs/noah/ckpt/finetuning/Control_RVXL_AD"

accelerate launch examples/controlnet/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_NAME_PATH \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DIR \
 --enable_xformers_memory_efficient_attention \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=30000 \
 --checkpointing_steps=5000 \
 --use_8bit_adam \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --max_grad_norm=1 \
 --seed=42 \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0