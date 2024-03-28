import os

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from controlnet_aux import MidasDetector

from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    AutoencoderKL,
)

device = 'cuda:3'
input = "/data/noah/inference/depth2real/input"
output = "/data/noah/inference/depth2real/output"

model_id = "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/rv"
controlnet_id = "/data/noah/ckpt/finetuning/Control_Depth_SD_AD/checkpoint-30000/controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16,use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True).to(device)
pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
lora_id = "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/lora_detail"
pipe.load_lora_weights(lora_id, weight_name="add_detail.safetensors")
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(device)

prompt = "cars are driving in the highway ,best quality, extremely detailed, clearness, naturalness, film grain, crystal clear, photo with color, actuality,  <lora:add-detail:-1>"
negative_prompt = "cartoon, anime, painting, disfigured, immature, blur, picture, 3D, render, semi-realistic, drawing, poorly drawn, bad anatomy, wrong anatomy, gray scale, worst quality, low quality, sketch"

for name in os.listdir(input):
    if not name.endswith('.png'):
        continue
    
    image_path = os.path.join(input, name)
    image = Image.open(image_path).convert('RGB')
    height, width = image.height, image.width
    image.thumbnail((768,768))

    con_image = midas(image, image_resolution=image.height)
    con_image = con_image.resize((image.width, image.height))

    with torch.no_grad():
        result_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=0.75,
            controlnet_conditioning_scale=0.65,
            num_images_per_prompt=10,
            image = image,
            control_image = con_image,
            height = image.height,
            width = image.width,
            num_inference_steps=50,
            guidance_scale=7.5,
            added_cond_kwargs=None
        ).images
        # result_images = pipe(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     controlnet_conditioning_scale=0.65,
        #     num_images_per_prompt=10,
        #     image = con_image,
        #     height = image.height,
        #     width = image.width,
        #     num_inference_steps=50,
        #     guidance_scale=7.5,
        #     added_cond_kwargs=None
        # ).images
    
    for idx, result_image in enumerate(result_images):    
        result_image = result_image.resize((width, height))
        result_image.save(os.path.join(output, name[:-4]+'_{}'.format(idx)+name[-4:]))