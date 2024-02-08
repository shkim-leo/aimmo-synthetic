import cv2
import os
import csv
import json
import shutil
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from controlnet_aux.processor import Processor
from diffusers.blip.models.blip import blip_decoder
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = "cuda:3"
torch.cuda.set_device(device)


def load_demo_image(image, image_size, device):
    raw_image = image

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


base_path = "/data/noah/dataset/AD_CON"
image_path = os.path.join(base_path, "images")
con_image_path = os.path.join(base_path, "conditioning_images_2")
caption_path = os.path.join(base_path, "train_2.jsonl")
midas = Processor("depth_midas")

image_size = 512
model_path = "/data/noah/ckpt/pretrain_ckpt/BLIP/model_large_caption.pth"
model = blip_decoder(pretrained=model_path, image_size=image_size, vit="large")
model.eval()
model = model.to(device)

data = []
for image_name in os.listdir(image_path):
    image = Image.open(os.path.join(image_path, image_name)).convert("RGB")
    con_image = midas(image)
    con_image.save(os.path.join(con_image_path, image_name))

    image = load_demo_image(image=image, image_size=image_size, device=device)

    with torch.no_grad():
        caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)[0]
        data.append(
            {
                "text": caption,
                "image": "images/{}".format(image_name),
                "conditioning_images": "conditioning_images_2/{}".format(image_name),
            }
        )

with open(caption_path, encoding="utf-8", mode="w") as f:
    for i in data:
        f.write(json.dumps(i) + "\n")
