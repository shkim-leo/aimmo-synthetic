import huggingface_hub

huggingface_hub.snapshot_download(
    repo_id="nateraw/background-remover-files",
    repo_type="dataset",
    local_dir="/data/noah/ckpt/pretrain_ckpt/matting",
    local_dir_use_symlinks=False,
)
