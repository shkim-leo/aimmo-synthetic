import os
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

device = "cuda:3"

torch.cuda.set_device(device)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/svd_xt", torch_dtype=torch.float16
)
pipe.to(device)

image_dir = "/data/noah/inference/ces_2024/ref"
video_dir = "/data/noah/inference/ces_2024/out"
frame_dir = "/data/noah/inference/ces_2024/out_frames"

for _ in os.listdir(image_dir):
    image_name = _[:-4]
    vid_path = os.path.join(video_dir, image_name + ".gif")
    frame_path = os.path.join(frame_dir, image_name)
    os.makedirs(frame_path, exist_ok=True)

    image = load_image(os.path.join(image_dir, _))
    image = image.resize((1024, 576))

    frames = pipe(
        image,
        num_frames=25,
        height=576,
        width=1024,
        decode_chunk_size=8,
        motion_bucket_id=100,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        # generator=torch.manual_seed(42),
    ).frames[0]

    for idx, frame in enumerate(frames):
        frame.save(os.path.join(frame_path, "{}_{}.jpg".format(image_name, idx)))

    export_to_gif(frames, output_gif_path=vid_path)
