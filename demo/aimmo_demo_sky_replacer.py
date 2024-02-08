import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
from PIL import Image
from huggingface_hub import hf_hub_download
from time import time

device = torch.device("cuda:2")
torch.cuda.set_device(device)

load_GR_time = 0
GR_time = 0
load_Inpating_time = 0
Inpating_time = 0
color_matching_time = 0

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

filenmae = "/data/noah/ckpt/pretrain_ckpt/Grounding_DINO/groundingdino_swinb_cogcoor.pth"
config_filename = "/workspace/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"

s_t = time()

model = load_model(config_filename, filenmae, device=device)

sam_checkpoint = "/data/noah/ckpt/pretrain_ckpt/SAM/sam_vit_h_4b8939.pth"
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

e_t = time()

load_GR_time = float(e_t) - float(s_t)

# load base and mask image
base_path = '/data/noah/inference/sky_replacer'
input_name = '1652255888589_FR-View-CMR-Wide.png'
input_path = os.path.join(base_path, 'input/{}'.format(input_name))
input_ref_path = os.path.join(base_path, 'reference')

output_path = os.path.join(base_path, 'output')
output_mask_path = os.path.join(base_path, 'output_mask')
output_background_path = os.path.join(base_path, 'output_background')
output_inpainting = os.path.join(base_path, 'output_inpainting')
init_image_source, init_image = load_image(input_path)

# detect object using grounding DINO
def detect(image, text_prompt, model, box_threshold=0.3, text_threshold=0.25):
    boxes, logits, phrases = predict(
        model=model, image=image, caption=text_prompt, box_threshold=box_threshold, text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=init_image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    return annotated_frame, boxes


annotated_frame, detected_boxes = detect(init_image, text_prompt="sky", model=model)


def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.cpu()


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

s_t = time()
segmented_frame_masks = segment(init_image_source, sam_predictor, boxes=detected_boxes)
annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)

mask = segmented_frame_masks[0][0].cpu().numpy()
inverted_mask = ((1 - mask) * 255).astype(np.uint8)

image_source_pil = Image.fromarray(init_image_source)
image_mask_pil = Image.fromarray(mask)
# image_mask_pil.save(os.path.join(output_mask_path, input_name))
e_t = time()

GR_time = float(e_t) - float(s_t)

from diffusers.utils import load_image

def partial_histogram_matching(source_image, target_image, alpha=0.15):
    corrected_channels = []
    
    for source_channel, target_channel in zip(cv2.split(source_image), cv2.split(target_image)):
        # 히스토그램 계산
        source_hist = cv2.calcHist([source_channel], [0], None, [256], [0, 256])
        target_hist = cv2.calcHist([target_channel], [0], None, [256], [0, 256])

        #전체 픽셀수로 히스토그램 정규화
        source_hist /= source_channel.size
        target_hist /= target_channel.size

        #누적 분포 계산
        source_cdf = np.cumsum(source_hist)
        target_cdf = np.cumsum(target_hist)

        #누적 분포 정규화
        source_cdf = source_cdf / source_cdf[-1]
        target_cdf = target_cdf / target_cdf[-1]

        #souce 이미지의 히스토그램 누적 분포를 target 이미지의 히스토그램 누적 분포로 매칭시키는 룩업 테이블 생성
        lut = np.interp(source_cdf, target_cdf, range(256))

        #alpha 값을 통한 룩업 테이블 수정
        partial_lut = alpha * lut + (1 - alpha) * np.arange(256)

        #룩업 데이블 기반 source 이미지 보정
        corrected_channel = cv2.LUT(source_channel, partial_lut.astype("uint8"))
        corrected_channels.append(corrected_channel)

    corrected_image = cv2.merge(corrected_channels)

    return corrected_image

corrected_images = []
ref_images = []
ref_names = os.listdir(input_ref_path)

s_t = time()

for r_name in ref_names:
    ref_image = load_image(os.path.join(input_ref_path, r_name))
    ref_image = ref_image.resize((image_source_pil.width, image_source_pil.height))
    ref_images.append(ref_image)
        
    corrected_imageA = partial_histogram_matching(init_image_source, np.array(ref_image))
    corrected_imageA = Image.fromarray(corrected_imageA)
    # corrected_imageA.save(os.path.join(output_background_path, '{}_{}'.format(input_name,r_name)))
    corrected_images.append(corrected_imageA)

e_t = time()

color_matching_time = (float(e_t) - float(s_t))/len(ref_names)
 
def color_clahe_equalization(image, clip_limit=1.0, tile_grid_size=(4, 4)):
    # Convert the image from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_channel = clahe.apply(l_channel)

    # Merge the CLAHE-enhanced L channel with the original A and B channels
    equalized_lab_image = cv2.merge([cl_channel, a_channel, b_channel])

    # Convert the image back to BGR color space
    equalized_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)

    return equalized_image

device = "cuda:2"
resolution = 1024
# init_image = image_source_pil.resize((resolution, resolution))
mask_image = image_mask_pil.resize((resolution, resolution))

s_t = time()
# model_id = "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/inpaint"
model_id = "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/sd_turbo"
pipeline = AutoPipelineForInpainting.from_pretrained(
    model_id, torch_dtype=torch.float16
)
pipeline.load_ip_adapter(
    "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/ip-adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)

pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1
)
e_t = time()

load_Inpating_time = float(e_t) - float(s_t)

pipeline = pipeline.to(device)
prompt = ""

s_t = time()
for corrected_image, ref_image, ref_name in zip(corrected_images, ref_images, ref_names):
    corrected_image = corrected_image.resize((resolution,resolution))
    ref_image = ref_image.resize((resolution, resolution))

    image = pipeline(
        prompt=prompt,
        image=corrected_image,
        mask_image=mask_image,
        strength=0.8,
        guidance_scale=7.5,
        num_inference_steps=5,
        ip_adapter_image=ref_image,
        height=1024,
        width=1024,
    ).images[0]
    
    image = image.resize((image_mask_pil.width, image_mask_pil.height))
    # image.save(os.path.join(output_inpainting,'{}_{}'.format(input_name, ref_name)))
    image = color_clahe_equalization(np.array(image))
    image = Image.fromarray(image)
    # image.save(os.path.join(output_path, '{}_{}'.format(input_name, ref_name)))
e_t = time()

Inpating_time = (float(e_t) - float(s_t))/len(ref_names)

print('load_GR_time : {}'.format(load_GR_time))
print('GR_time : {}'.format(GR_time))
print('load_Inpating_time : {}'.format(load_Inpating_time))
print('Inpating_time : {}'.format(Inpating_time))