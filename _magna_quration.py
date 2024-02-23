# Format {"image": "", "conditioning_images": "", "text": "", "annotation": ""}

# annotation -> mask -> condition image 생성
# prompt
import json
import os
import cv2
from PIL import Image
import numpy as np
import random

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


annotation_path = "/data/noah/dataset/coco/anno"
condition_path = "/data/noah/dataset/coco/condition_images"
caption_path = os.path.join("/data/noah/dataset/coco", "train.jsonl")
gt_caption_path = "/data/noah/dataset/coco/tmp/captions_train2017.json"
target_classes = ["person"]

data = []

with open(gt_caption_path, "r") as f:
    gt_ann = json.load(f)

for annotation_name in tqdm(os.listdir(annotation_path)):
    anno_path = os.path.join(annotation_path, annotation_name)
    with open(anno_path, "r") as f:
        annotation = json.load(f)

    for img_info in gt_ann["images"]:
        if img_info["file_name"] == annotation["filename"]:
            image_id = img_info["id"]
            break

    captions = []
    for cap_info in gt_ann["annotations"]:
        if cap_info["image_id"] == cap_info["image_id"]:
            captions.append(cap_info["caption"])

    caption = ""
    for idx, cap in enumerate(captions):
        if idx == len(captions) - 1:
            caption += "{}".format(cap)
        else:
            caption += "{}, ".format(cap)

    # image_path
    image_path = os.path.join(annotation["parent_path"], annotation["filename"])
    con_path = os.path.join(condition_path, annotation["filename"])

    height, width = annotation["metadata"]["height"], annotation["metadata"]["width"]

    # generate condition image
    mask = np.zeros((height, width, 3))
    for ann in annotation["annotations"]:
        if ann["label"] in target_classes:
            pallete = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            points = np.array(ann["points"], dtype=np.int32)
            try:
                mask = cv2.fillPoly(mask, [points], color=pallete)
            except:
                continue
    mask = Image.fromarray(mask.astype("uint8"))
    mask.save(con_path)
    # display(Image.fromarray(mask))
    # print(image_path)

    data.append(
        {
            "text": caption,
            "image": "images/{}".format(annotation["filename"]),
            "conditioning_images": "conditioning_images/{}".format(annotation["filename"]),
            "annotation": "anno/{}".format(annotation_name),
        }
    )

with open(caption_path, encoding="utf-8", mode="w") as f:
    for i in data:
        f.write(json.dumps(i) + "\n")
