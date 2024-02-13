export MODEL_NAME="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/svd"
export TRAIN_PATH="/data/noah/dataset/AD_SEQ/metadata.csv"
export OUTPUT_DIR="/data/noah/ckpt/finetuning/SVD_AD"
export VALIDATION_IMAGE=""

accelerate launch examples/stable_video_diffusion/train_svd.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_ann_path=$TRAIN_PATH \
    --output_dir=$OUTPUT_DIR \
    --validation_image=$VALIDATION_IMAGE \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=512 \
    --height=320 \
    --sample_n_frames=16 \
    --sample_stride=1 \
    --checkpointing_steps=2500 \
    --enable_xformers_memory_efficient_attention \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=42 \
    --mixed_precision="fp16" \
    --validation_steps=250 \
    --use_8bit_adam

