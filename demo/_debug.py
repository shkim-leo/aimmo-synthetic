import os
import random
import json
import copy
from tqdm import tqdm

import numpy as np
from PIL import Image
import cv2
import torch

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux.processor import MidasDetector
import sys

sys.path.insert(os.getcwd(), "../harmonization")
from harmonization import Harmonization
from gtgen.bpr import GtGenBPRInference


def make_inputs(image, annotation, target_indexs, harmonizer, midas, ann_idx):
    height, width = annotation["metadata"]["height"], annotation["metadata"]["width"]
    mask = np.zeros((height, width))
    spot = None
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Coefficients of the polynomial
    coefficients = [7.62995538e-14, -2.57068472e-10, 3.25925629e-07, -1.90207658e-04, 5.09169229e-02, -5.35772215e+00, 2.35348832e+02]
    # Create a polynomial object using poly1d
    height_poly = np.poly1d(coefficients)

    while True:
        target_index = random.choice(target_indexs)
        rb_spot = random_coordinate(annotation["annotations"], target_index, height, width)  # height, width 순

        if rb_spot is None:
            print("{} can not generate right bottom spot".format(annotation["filename"]))
            return None

        # rb_spot x값을 기준으로 height 선정 및 target_height 산출
        # target_height = random.randint(500, 500)
        target_height = int(height_poly(rb_spot[0]))
        mask_name = random.choice(mask_lists)

        image_pth = os.path.join(image_path, mask_name)
        mask_pth = os.path.join(mask_path, mask_name)

        paste_image = Image.open(image_pth)
        paste_mask = Image.open(mask_pth)

        ratio = float(target_height) / paste_mask.height
        paste_mask = paste_mask.resize((int(paste_mask.width * ratio), int(paste_mask.height * ratio)))
        paste_image = paste_image.resize((int(paste_image.width * ratio), int(paste_image.height * ratio)))

        paste_mask = np.array(paste_mask).astype("uint8")
        paste_image = np.array(paste_image).astype("uint8")

        if np.sum(paste_mask) == 0:
            continue

        paste_mask = cv2.morphologyEx(paste_mask, cv2.MORPH_OPEN, k, iterations=3)
        # paste_mask = cv2.dilate(paste_mask, k, iterations=2)
        paste_mask = np.where(paste_mask > 127, 255, 0).astype("uint8")
        sum_mask = add_mask(mask, paste_mask, rb_spot[1], rb_spot[0])

        if sum_mask is None:
            continue

        spot = np.argwhere(sum_mask == 255)
        sum_image = add_image(image, paste_image, paste_mask, rb_spot[1], rb_spot[0])
        break

    sum_image = harmonizer.harmonize(sum_image, sum_mask)
    sum_image = Image.fromarray(sum_image.astype("uint8")).convert("RGB")
    sum_mask = Image.fromarray(sum_mask.astype("uint8")).convert("L")
    con_image = midas(sum_image, image_resolution=height)

    output = {"image": sum_image, "mask": sum_mask, "con_image": con_image, "spot": spot}
    return output

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def find_outer_contour_coordinates(mask):
    # OpenCV의 findContours 함수를 사용하여 이진 이미지의 외곽선을 찾습니다.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 외곽선을 이루는 점들의 좌표를 반환합니다.
    outer_contour_coords = [[], []]
    for contour in contours:
        for point in contour:
            x, y = point[0]
            outer_contour_coords[0].append(y)
            outer_contour_coords[1].append(x)

    return outer_contour_coords


def euclidean_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def random_coordinate(annotation, target_index, height, width):
    mask = np.zeros((height, width))
    mask, state = polygon_to_mask(mask, annotation[target_index]["points"], color=255)

    if not state:
        return None

    for idx, ann in enumerate(annotation):
        if idx == target_index:
            continue

        mask, state = polygon_to_mask(mask, ann["points"], color=0)

    target_spots = np.argwhere(mask == 255).tolist()

    if len(target_spots) == 0:
        return None

    coordinates = find_outer_contour_coordinates(mask)
    threshold = 500

    # 랜덤으로 좌표 선택
    while True:
        target_spot = random.choice(target_spots)  # height,width 순
        distances = [
            euclidean_distance((coord[0], coord[1]), target_spot) for coord in zip(coordinates[0], coordinates[1])
        ]
        min_distance = int(min(distances))

        if min_distance >= threshold:
            return target_spot
        else:
            threshold = threshold // 2


def add_mask(mask, new_mask, right, bottom):
    mask_cp = mask.copy()

    # 새로운 마스크를 더할 위치 계산
    left = right - new_mask.shape[1]
    top = bottom - new_mask.shape[0]

    # 마스크 영역에 새로운 마스크 더하기
    if left < 0 or top < 0:
        return None

    mask_cp[top:bottom, left:right] += new_mask

    return mask_cp


def add_image(image, new_image, mask, right, bottom):
    image_cp = image.copy()

    # 새로운 마스크를 더할 위치 계산
    left = right - new_image.shape[1]
    top = bottom - new_image.shape[0]

    # 마스크 영역에 새로운 마스크 더하기
    if left < 0 or top < 0:
        return None

    for h in range(top, bottom):
        for w in range(left, right):
            if mask[h - top, w - left]:
                image_cp[h, w, :] = new_image[h - top, w - left, :]

    return image_cp


def make_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def make_result(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_mask_contours = np.copy(image)
    cv2.drawContours(image_with_mask_contours, contours, -1, (0, 0, 255), 2)
    return image_with_mask_contours


def mask_refinement(image, ann, bpr_inference):
    seg_result = bpr_inference.inference(
        img=image,
        seg=ann,
        img_scale=(256, 256),
        img_ratios=[1.0, 1.5, 2.0],
        nms_iou_threshold=0.5,
        point_density=0.25,
        patch_size=[32, 64, 96],
        padding=0,
    )

    height, width = image.shape[0], image.shape[1]
    result_map = np.zeros((height, width))

    for sr in seg_result["annotations"]:
        result_map, state = polygon_to_mask(result_map, sr["points"], 255)

    return result_map.astype("uint8")


def mask_to_polygon(mask):
    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선을 다각형으로 변환
    polygons = []
    for contour in contours:
        contour = contour.squeeze(axis=1)  # 차원 축소
        polygon = contour[:, [0, 1]].tolist()  # (y, x) 순서로 변환하여 리스트로 저장
        polygons.append(polygon)

    return polygons


def polygon_to_mask(mask, polygons, color=255):
    polygons = np.array(polygons, dtype=np.int32)
    state = False

    try:
        mask = cv2.fillPoly(mask.astype("uint8"), [polygons], color)
        state = True
    except:
        print("mask passed!")

    return mask, state


def modify_annotation(annotations, polygons, height, width):
    # draw generated mask
    generated_mask = np.zeros((height, width))
    generated_annotations = []
    original_annotations = []

    for polygon in polygons:
        generated_mask, state = polygon_to_mask(generated_mask, polygon, 255)

        if state:
            ann = {
                "id": "",
                "type": "poly_seg",
                "attributes": {},
                "points": polygon,
                "label": "person",
            }
            generated_annotations.append(ann)

    for annotation in annotations:
        # draw original mask
        original_mask = np.zeros((height, width))
        original_mask, state = polygon_to_mask(original_mask, annotation["points"], 255)

        if not state:
            continue

        # modify original mask
        original_mask = np.where((original_mask == 255) & (generated_mask == 255), 0, original_mask)
        original_polygons = mask_to_polygon(original_mask)

        for polygon in original_polygons:
            ann = copy.deepcopy(annotation)
            ann["points"] = polygon
            original_annotations.append(ann)

    original_annotations.extend(generated_annotations)
    return original_annotations


device = "cuda:2"

# 사전 정의된 Crop 이미지와 마스크
mask_path = "/data/noah/inference/magna_human_premask/masks"
image_path = "/data/noah/inference/magna_human_premask/images"
mask_lists = os.listdir(mask_path)

# 생성할 Annotation 정보
base_image_path = "/data/noah/dataset/magna_traffic_light/pre_images"
target_annotation_path = "/data/noah/dataset/magna_traffic_light/pre_anno"
target_class_name = "road"
target_height = None

save_base_path = "/data/noah/inference/magna_controlnet_inpainting_f"
save_result_path = os.path.join(save_base_path, "results")
save_draw_path = os.path.join(save_base_path, "draw_results")
save_refined_draw_path = os.path.join(save_base_path, "draw_results_refined")
save_annotation_draw_path = os.path.join(save_base_path, "annotation_draw")
save_modified_annotation_draw_path = os.path.join(save_base_path, "modified_annotation_draw")
save_mask_path = os.path.join(save_base_path, "masks")
make_dirs(
    [
        save_base_path,
        save_result_path,
        save_draw_path,
        save_refined_draw_path,
        save_modified_annotation_draw_path,
        save_annotation_draw_path,
        save_mask_path,
    ]
)

bpr_inference = GtGenBPRInference(devices=[2], batch_size=48)
bpr_model = bpr_inference.load_model("/data/noah/ckpt/finetuning/bpr.pth", img_scale=(256, 256))
assert bpr_model is not None, "model not loaded"

midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(device)
harmonizer = Harmonization("/data/noah/ckpt/pretrain_ckpt/duconet/duconet1024.pth", device=device)

model_id = "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/rv_inpaint_5.1"
controlnet_id = "/data/noah/ckpt/finetuning/controlnet_inpaint_coco/checkpoint-180000/controlnet"
lora_id = "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/lora_detail"

controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to(device)
pipe.load_lora_weights(lora_id, weight_name="add_detail.safetensors")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_freeu(s1=1.2, s2=0.5, b1=1.2, b2=1.4)
# generator = torch.Generator(device=device).manual_seed(42)

prompt = "a person is on the road, RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, <lora:add-detail:1>"
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), blurry, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream"
num_inference_steps = 25
guidance_scale = 7.5
strength = 1.0
sag_scale = 0.75
controlnet_conditioning_scale = 0.75

# write parameter
with open(os.path.join(save_base_path, "param.txt"), "w") as f:
    f.write("Description : 추론 결과\n")
    f.write("base model : {}\n".format(model_id))
    f.write("controlnet model : {}\n".format(controlnet_id))
    f.write("lora model : {}\n".format(lora_id))
    f.write("scheduler : {}\n".format(pipe.scheduler))

    f.write("prompt : {}\n".format(prompt))
    f.write("negative prompt : {}\n".format(negative_prompt))
    f.write("num_inference_steps : {}\n".format(num_inference_steps))
    f.write("guidance_scale : {}\n".format(guidance_scale))
    f.write("strength : {}\n".format(strength))
    f.write("controlnet_conditioning_scale : {}\n".format(controlnet_conditioning_scale))
    f.write("sag_scale : {}\n".format(sag_scale))

for ann_idx, ann_name in tqdm(enumerate(os.listdir(target_annotation_path)[:40])):
    annotation_path = os.path.join(target_annotation_path, ann_name)

    with open(annotation_path, "r") as f:
        annotation = json.load(f)

    target_indexs = []

    for idx, ann in enumerate(annotation["annotations"]):
        if ann["label"] == target_class_name:
            target_indexs.append(idx)

    if not len(target_indexs):
        continue

    height, width = annotation["metadata"]["height"], annotation["metadata"]["width"]
    image = Image.open(os.path.join(base_image_path, annotation["parent_path"][1:], annotation["filename"]))
    image = np.array(image).astype("uint8")

    sum_mask_image = np.zeros((height, width))
    sum_result_image = np.copy(image)
    sum_draw_image = None
    
    generate_cnt = random.randint(1, 4)
    inputs = None
    generated_spots = []

    for iter_cnt in range(generate_cnt):
        inputs = make_inputs(image, annotation, mask_lists, target_indexs, harmonizer, midas)

        if inputs is None:
            break

        if len(generated_spots) and np.any(np.all(np.isin(np.array(generated_spots), inputs["spot"]), axis=1)):
            continue
        else:
            generated_spots.extend(inputs["spot"].tolist())

        result_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=inputs["image"],
            control_image=inputs["con_image"],
            mask_image=inputs["mask"],
            height=inputs["image"].height,
            width=inputs["image"].width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            sag_scale=sag_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            padding_mask_crop=8,
        ).images[0]

        result_image = harmonizer.harmonize(np.array(result_image), np.array(inputs["mask"]))
        result_image = result_image.astype("uint8")

        for spot in inputs["spot"]:
            sum_result_image[spot[0], spot[1], :] = result_image[spot[0], spot[1], :]

        sum_mask_image = sum_mask_image + np.array(inputs["mask"])

    if inputs is not None:
        # mask refinement
        polygons = mask_to_polygon(sum_mask_image)
        generated_annotation = copy.deepcopy(annotation)
        anns = []

        for polygon in polygons:
            an = {
                "id": "",
                "type": "poly_seg",
                "attributes": {},
                "points": polygon,
                "label": "person",
            }
            anns.append(an)

        generated_annotation["annotations"] = anns
        refined_mask = mask_refinement(sum_result_image, generated_annotation, bpr_inference)
        polygons = mask_to_polygon(refined_mask)

        # draw result image with mask
        sum_draw_image = make_result(np.copy(sum_result_image), sum_mask_image.astype("uint8"))
        sum_draw_refined_image = make_result(np.copy(sum_result_image), refined_mask)

        # annotation 수정 작업 #
        modified_annotation = modify_annotation(
            annotation["annotations"],
            polygons,
            height,
            width,
        )

        # draw annotation
        original_mask = np.zeros((height, width, 3))
        modified_mask = np.zeros((height, width, 3))

        for ann in annotation["annotations"]:
            original_mask, state = polygon_to_mask(
                original_mask,
                ann["points"],
                color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            )

        for m_ann in modified_annotation:
            modified_mask, state = polygon_to_mask(
                modified_mask,
                m_ann["points"],
                color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            )

        Image.fromarray(sum_result_image.astype("uint8")).convert("RGB").save(
            os.path.join(save_result_path, annotation["filename"])
        )
        Image.fromarray(sum_draw_image.astype("uint8")).convert("RGB").save(
            os.path.join(save_draw_path, annotation["filename"])
        )
        Image.fromarray(sum_draw_refined_image.astype("uint8")).convert("RGB").save(
            os.path.join(save_refined_draw_path, annotation["filename"])
        )
        Image.fromarray(original_mask.astype("uint8")).convert("RGB").save(
            os.path.join(save_annotation_draw_path, annotation["filename"])
        )
        Image.fromarray(modified_mask.astype("uint8")).convert("RGB").save(
            os.path.join(save_modified_annotation_draw_path, annotation["filename"])
        )
        Image.fromarray(sum_mask_image.astype("uint8")).convert("L").save(
            os.path.join(save_mask_path, annotation["filename"])
        )