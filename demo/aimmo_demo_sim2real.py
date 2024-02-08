import os

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from controlnet_aux import CannyDetector

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    AutoencoderKL,
)
from diffusers.blip.models.blip import blip_decoder


def load_demo_image(image_path, image_size, device):
    raw_image = Image.open(image_path).convert("RGB")

    w, h = raw_image.size
    # display(raw_image.resize((w//5,h//5)))

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def generate_base_pipeline(model_id, device, mode="SD", control_model_id=None):
    if mode == "SD":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        ).to(device)
    elif mode == "SDXL":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        ).to(device)
    elif mode == "Control_SD":
        if control_model_id is None:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True
            ).to(device)
        else:
            controlnet = ControlNetModel.from_pretrained(
                control_model_id, torch_dtype=torch.float16, use_safetensors=True
            ).to(device)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
        ).to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()

    elif mode == "Control_SDXL":
        if control_model_id is None:
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True
            ).to(device)
        else:
            controlnet = ControlNetModel.from_pretrained(
                control_model_id, torch_dtype=torch.float16, use_safetensors=True
            ).to(device)

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensoƒrs=True
        ).to(device)

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16, use_safetensors=True
        ).to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # pipe.enable_model_cpu_offload()
    else:
        raise ValueError('model must be in ["SD", "SDXL", "ControlNet_SD", "ControlNet_SDXL"]')

    return pipe


def generate_ref_pipeline(model_id, device):
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to(device)

    return refiner


def inference_sim2real(pipe, prompt, negative_prompt, image, mode, ref_image):
    if mode.startswith("Control"):
        controlnet_conditioning_scale = 0.75
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            image=image,
            guidance_scale=9.0,
            num_inference_steps=50,
            generator=torch.manual_seed(42),
        ).images[0]
    else:
        strength = 0.5
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=8.0,
            num_inference_steps=50,
            generator=torch.manual_seed(42),
            ip_adapter=ref_image,
        ).images[0]

    return image


sim2real_device = "cuda:2"

# clip_device = torch.device(sim2real_device if torch.cuda.is_available() else "cpu")
# clip_image_size = 512
# clip_model_path = "/data/noah/ckpt/pretrain_ckpt/BLIP/model_large_caption.pth"
# clip_model = blip_decoder(pretrained=clip_model_path, image_size=clip_image_size, vit="large")
# clip_model.eval()
# clip_model = clip_model.to(clip_device)

sim2real_mode = "SD"
sim2real_model_id = "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/sd"
# sim2real_model_id = '/data/noah/ckpt/pretrain_ckpt/StableDiffusion/rvxl_v2'
# sim2real_control_model_id = "/data/noah/ckpt/finetuning/Control_SD_AD/checkpoint-55000"
sim2real_pipe = generate_base_pipeline(sim2real_model_id, sim2real_device, mode=sim2real_mode, control_model_id=None)

# canny = CannyDetector()

ref_model_id = None
# ref_model_id = 'stabilityai/stable-diffusion-xl-refiner-1.0'

if ref_model_id:
    sim2real_ref_pipe = generate_ref_pipeline(ref_model_id, sim2real_device)

image_dir = "/data/noah/dataset/AD_SEQ/batch_5/20220624/2022-06-24_12-30-50_ADCV1-ADS-LC1/FR-View-CMR-Wide"
out_dir = "/data/noah/inference/ces_2024/out"
# out_line_dir = "/data/noah/inference/sim2real/output_line"
prompt = "{} ,best quality, extremely detailed, clearness, naturalness, film grain, crystal clear, photo with color, actuality"
negative_prompt = "cartoon, anime, painting, disfigured, immature, blur, picture, 3D, render, semi-realistic, drawing, poorly drawn, bad anatomy, wrong anatomy, gray scale, worst quality, low quality, sketch"

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

ref_image_path = "/workspace/diffusers-temp/sky-9-preview.jpg"
ref_image = Image.open(ref_image_path).convert("RGB")

for _ in os.listdir(image_dir):
    image_path = os.path.join(image_dir, _)
    # clip_image = load_demo_image(image_path=image_path, image_size=clip_image_size, device=clip_device)
    sim2real_image = Image.open(image_path).convert("RGB")
    sim2real_image = sim2real_image.resize((768, 512))
    # sim2real_image = canny(sim2real_image)
    # sim2real_image.save(os.path.join(out_line_dir, _))
    # sim2real_canny_image = cv2.Canny(np.array(sim2real_image), low_threshold, high_threshold)
    # sim2real_canny_image = sim2real_canny_image[:, :, None]
    # sim2real_canny_image = np.concatenate([sim2real_canny_image, sim2real_canny_image, sim2real_canny_image], axis=2)
    # sim2real_canny_image = Image.fromarray(sim2real_canny_image)

    with torch.no_grad():
        # caption = clip_model.generate(clip_image, sample=False, num_beams=3, max_length=40, min_length=5)[0]
        # caption = clip_model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)[0]
        # print(prompt.format(caption))
        sim2real_result_image = inference_sim2real(
            sim2real_pipe,
            prompt.format("cars are driving on the road with red sky"),
            negative_prompt,
            sim2real_image,
            sim2real_mode,
            ref_image,
        )

        if ref_model_id:
            sim2real_result_image = inference_sim2real(
                sim2real_ref_pipe,
                prompt.format("cars are driving on the road with red sky"),
                negative_prompt,
                [sim2real_result_image],
                "SDXL",
            )

    sim2real_result_image.save(os.path.join(out_dir, _))
