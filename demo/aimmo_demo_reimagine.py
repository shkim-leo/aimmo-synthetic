import os
import torch
from PIL import Image
import numpy as np

from diffusers import StableUnCLIPImg2ImgPipeline
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

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


# Start the StableUnCLIP Image variations pipeline
device = "cuda:2"
torch.cuda.set_device(torch.device(device))  # change allocation of current GPU


base_model_path = "/data/noah/ckpt/finetuning/SD_UNCLIP_AD"
image_dir = "/data/noah/inference/dire_test/origin"
out_dir = "/data/noah/inference/dire_test/output"

prompt = "{}, best quality, extremely detailed, clearness, naturalness, film grain, crystal clear, photo with color, actuality"
negative_prompt = "cartoon, anime, painting, disfigured, immature, blur, picture, 3D, render, semi-realistic, drawing, poorly drawn, bad anatomy, wrong anatomy, gray scale, worst quality, low quality, sketch"

clip_image_size = 512
clip_model_path = "/data/noah/ckpt/pretrain_ckpt/BLIP/model_large_caption.pth"
clip_model = blip_decoder(pretrained=clip_model_path, image_size=clip_image_size, vit="large")
clip_model.eval()
clip_model = clip_model.to(device)

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)

    init_image = Image.open(img_path).convert("RGB")
    result_size = (1024,1024)
    # init_image.thumbnail((768, 768), Image.ANTIALIAS)
    init_image.thumbnail((768, 768))

    with torch.no_grad():
        clip_image = load_demo_image(image_path=img_path, image_size=clip_image_size, device=device)
        caption = clip_model.generate(clip_image, sample=False, num_beams=3, max_length=40, min_length=5)[0]

        # Pipe to make the variation
        image = pipe(
            init_image,
            prompt=prompt.format(caption),
            negative_prompt=negative_prompt,
            guidance_scale=7.0,
            num_inference_steps=40,
            noise_level=0,
            generator=torch.manual_seed(42),
        ).images[0]
        
        image = image.resize(result_size)

    image.save(os.path.join(out_dir, img_name))
