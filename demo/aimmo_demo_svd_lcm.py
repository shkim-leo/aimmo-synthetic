import os
import torch
from diffusers import StableVideoDiffusionLCMPipeline
from diffusers.schedulers import LCMScheduler
from diffusers.utils import load_image, export_to_video
import cv2
import numpy as np
from PIL import Image

device = "cuda:3"

torch.cuda.set_device(device)
model_id = "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/svd_xt"
pipe = StableVideoDiffusionLCMPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

image_dir = "/data/noah/inference/incabin_sample/reference_img"
video_dir = "/data/noah/inference/incabin_sample/out"
# frame_dir = "/data/noah/inference/incabin_sample/frames"

for _ in os.listdir(image_dir):
    image_name = _[:-4]
    vid_path = os.path.join(video_dir, image_name + ".mp4")
    # frame_path = os.path.join(frame_dir, image_name)
    # os.makedirs(frame_path, exist_ok=True)

    image = load_image(os.path.join(image_dir, _))
    image = image.resize((1024, 576))
    frames = pipe(image, num_frames=25, decode_chunk_size=8, motion_bucket_id=127, noise_aug_strength=0).frames[0]

    # for idx, frame in enumerate(frames):
    #     frame.save(os.path.join(frame_path, "{}_{}.jpg".format(image_name, idx)))

    export_to_video(frames, vid_path, fps=2)
