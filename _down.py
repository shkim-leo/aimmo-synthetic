import huggingface_hub

huggingface_hub.snapshot_download(
    repo_id="lllyasviel/control_v11f1p_sd15_depth",
    repo_type="model",
    local_dir="/data/noah/ckpt/pretrain_ckpt/StableDiffusion/controlnet_midas",
    local_dir_use_symlinks=False,
)
