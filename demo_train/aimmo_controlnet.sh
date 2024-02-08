export MODEL_NAME_PATH="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/sd"
export TRAIN_DIR="/data/noah/dataset/AD_CON"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/Control_MIDAS_SD_AD"

accelerate launch examples/controlnet/train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_NAME_PATH \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DIR \
 --enable_xformers_memory_efficient_attention \
 --mixed_precision="fp16" \
 --resolution=768 \
 --learning_rate=1e-5 \
 --gradient_accumulation_steps=4 \
 --mixed_precision="fp16" \
 --max_train_steps=100000 \
 --checkpointing_steps=5000 \
 --train_batch_size=4 \
 --use_8bit_adam \
 --gradient_checkpointing \
 --max_grad_norm=1 \
 --seed=42 \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0