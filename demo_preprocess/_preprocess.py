# import os
# import numpy as np
# import torch
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForCausalLM
# import random

# device = "cuda:3"
# torch.cuda.set_device(device)

# from torchvision.transforms.functional import InterpolationMode
# from torchvision import transforms
# from diffusers.blip.models.blip import blip_decoder


# def load_demo_image(raw_image, image_size, device):
#     w, h = raw_image.size
#     # display(raw_image.resize((w//5,h//5)))

#     transform = transforms.Compose(
#         [
#             transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#         ]
#     )
#     image = transform(raw_image).unsqueeze(0).to(device)
#     return image


# image_size = 512
# model_path = "/data/noah/ckpt/pretrain_ckpt/BLIP/model_large_caption.pth"
# model = blip_decoder(pretrained=model_path, image_size=image_size, vit="large")
# model.eval()
# model = model.to(device)


# # set seed for reproducability


# def read_video_pyav(container, indices):
#     """
#     Decode the video with PyAV decoder.
#     Args:
#         container (`av.container.input.InputContainer`): PyAV container.
#         indices (`List[int]`): List of frame indices to decode.
#     Returns:
#         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
#     """
#     frames = []
#     container.seek(0)
#     start_index = indices[0]
#     end_index = indices[-1]
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > end_index:
#             break
#         if i >= start_index and i in indices:
#             frames.append(frame)
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
#     """
#     Sample a given number of frame indices from the video.
#     Args:
#         clip_len (`int`): Total number of frames to sample.
#         frame_sample_rate (`int`): Sample every n-th frame.
#         seg_len (`int`): Maximum allowed index of sample's last frame.
#     Returns:
#         indices (`List[int]`): List of sampled frame indices
#     """
#     converted_len = int(clip_len * frame_sample_rate)
#     end_idx = np.random.randint(converted_len, seg_len)
#     start_idx = end_idx - converted_len
#     indices = np.linspace(start_idx, end_idx, num=clip_len)
#     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
#     return indices


# # load video
# folder_path = "/data/noah/dataset/AD_SEQ"
# captions = []
# folder_paths = []

# for root, folders, files in os.walk(folder_path):
#     frames = []
#     for file in files:
#         if file.endswith(".png"):
#             if root not in folder_paths:
#                 folder_paths.append(root)
#             frame = Image.open(os.path.join(root, file)).convert("RGB")
#             frames.append(frame)

#     ###
#     if not len(frames):
#         continue

#     image = load_demo_image(raw_image=random.choice(frames), image_size=image_size, device=device)

#     with torch.no_grad():
#         # beam search
#         # caption = model.generate(image, sample=False, num_beams=3, max_length=40, min_length=5)
#         # nucleus sampling
#         caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)

#     captions.append(caption[0])

# import csv

# out_path = "/data/noah/dataset/AD_SEQ/metadata.csv"

# with open(out_path, mode="w", newline="") as file:
#     writer = csv.DictWriter(file, fieldnames=["folder", "caption"])

#     writer.writeheader()

#     for path, cap in zip(folder_paths, captions):
#         writer.writerow({"folder": path, "caption": cap})

import csv
from tqdm import tqdm
import os
from PIL import Image
import torch
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
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


image_size = 512
image_dir = "/data/noah/dataset/disney"
model_path = "/data/noah/ckpt/pretrain_ckpt/BLIP/model_large_caption.pth"
device = "cuda:2"
torch.cuda.set_device(torch.device(device))

model = blip_decoder(pretrained=model_path, image_size=image_size, vit="large")
model.eval()
model = model.to(device)

file_name = "/data/noah/dataset/disney/metadata.csv"
data = []

for img_name in tqdm(os.listdir(image_dir)):
    # if os.path.splitext(img_name)[-1] not in [".jpg", ".png"]:
    #     continue

    image_path = os.path.join(image_dir, img_name)

    image = load_demo_image(image_path=image_path, image_size=image_size, device=device)

    with torch.no_grad():
        # beam search
        # caption = model.generate(image, sample=False, num_beams=3, max_length=40, min_length=5)
        # nucleus sampling
        caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)

        data.append({"file_path": image_path, "text": caption[0]})

with open(file_name, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["file_path", "text"])

    writer.writeheader()

    for row in data:
        writer.writerow(row)

print(f"{file_name} 파일이 생성되었습니다.")
