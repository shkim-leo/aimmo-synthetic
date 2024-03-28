import os
import random
import json
import copy
from tqdm import tqdm

import numpy as np
from PIL import Image
import cv2
import torch

from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from controlnet_aux.processor import MidasDetector
import sys

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

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


# detect object using grounding DINO
def detect(image, image_source, text_prompt, model, box_threshold=0.5, text_threshold=0.5):
    boxes, logits, phrases = predict(
        model=model, image=image, caption=text_prompt, box_threshold=box_threshold, text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    return annotated_frame, boxes


def segment(image, sam_model, boxes, device):
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

def get_mask(image, dino, sam, device):
    result_mask = np.zeros((image.shape[0], image.shape[1]))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_torch, _ = transform(Image.fromarray(image).convert("RGB"), None)
    annotated_frame, detected_boxes = detect(
        image_torch, image, text_prompt="face", model=dino
    )
    if len(detected_boxes) == 0:
        return None

    seg_result = segment(image, sam, boxes=detected_boxes, device=device)
    result_masks = []
    for seg_map in seg_result:
        mask = seg_map[0].cpu().numpy().astype(np.uint8) * 255
        result_masks.append(mask)

    return result_masks

def closest_multiple_of_8(number):
    closest_multiple = (number // 8) * 8  # 가장 가까운 4의 배수
    return closest_multiple

base_path = "./"
os.makedirs(base_path,exist_ok=True)
device = "cuda:3"
device_2 = "cuda:2"
grounding_dino_ckpt_path = "/data/noah/ckpt/pretrain_ckpt/Grounding_DINO/groundingdino_swinb_cogcoor.pth"
grounding_dino_config_path = (
    "/workspace/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
)
grounding_dino = load_model(grounding_dino_config_path, grounding_dino_ckpt_path, device=device_2)
sam_ckpt_path = "/data/noah/ckpt/pretrain_ckpt/SAM/sam_vit_h_4b8939.pth"
sam_predictor = SamPredictor(build_sam(checkpoint=sam_ckpt_path).to(device_2))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/rv_inpaint_5.1", torch_dtype=torch.float16
).to(device)
pipe.load_lora_weights(
    "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/lora_detail", weight_name="add_detail.safetensors"
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
prompt = "woman, pretty, realistic, best quality, extremely detailed, clearness, crystal clear, <lora:add-detail:-1>"
negative_prompt = "cartoon, anime, painting, disfigured, immature, blur, picture, semi-realistic, gray scale, worst quality, low quality, out of frame, jpeg artifacts, ugly, poorly drawn eyes, wrong eyes"

image_path = "/data/noah/inference/ad_face_mask/얼굴_1653465190366_RR-Right-View-CMR-Narrow_Undistorted.png"
mask_paths = [
    "/data/noah/inference/ad_face_mask/mask_1.png",
    "/data/noah/inference/ad_face_mask/mask_2.png",
]

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, channel = image.shape
height, width = closest_multiple_of_8(height), closest_multiple_of_8(width)
    
if min(width, height) <=1000:
    height, width = closest_multiple_of_8(height*2), closest_multiple_of_8(width*2)

if width >= 3000 or height>=3000:
    height, width = closest_multiple_of_8(height//2), closest_multiple_of_8(width//2)

image = cv2.resize(image, (width, height))
masks = [cv2.imread(_) for _ in mask_paths]
# masks = get_mask(image, grounding_dino, sam_predictor, device_2)

overlay_image = np.copy(image)
sum_mask = np.zeros((image.shape[0], image.shape[1]))
for mask in masks:
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    sum_mask += mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, k, iterations=3)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=5)
    spot = np.argwhere(mask==255).tolist()
    blurred_mask = pipe.mask_processor.blur(Image.fromarray(mask).convert('L'), blur_factor=10)
    
    result_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=Image.fromarray(image),
        mask_image=blurred_mask,
        height=image.shape[0],
        width=image.shape[1],
        num_inference_steps=40,
        guidance_scale=8.5,
        # generator=generator
    ).images[0]
    
    result_image = np.array(result_image)

    for s in spot:
        overlay_image[s[0], s[1], :] = result_image[s[0], s[1], :]

image = Image.fromarray(image)
overlay_image = Image.fromarray(overlay_image)
sum_mask = Image.fromarray(sum_mask).convert('L')
overlay_image.save(os.path.join(base_path, '_out.png'))
sum_mask.save(os.path.join(base_path, '_mask.png'))