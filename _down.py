import huggingface_hub

huggingface_hub.snapshot_download(
    repo_id="Multimodal-Fatima/COCO_captions_train",
    repo_type="dataset",
    local_dir="/data/noah/dataset/coco",
    local_dir_use_symlinks=False,
)
