import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline
import os 
import json
import pandas as  pd
from PIL import Image
import shutil

device = 'cuda:2'
torch.cuda.set_device(torch.device(device))

def image_classifier(image, pipe):
    outputs = pipe(image)
    return outputs[0]['score']

threshold = 0.8
model = 'umm-maybe/AI-image-detector'
pipe = pipeline("image-classification", f"{model}", trust_remote_code=True, device= device)
fake_path = '/data/noah/inference/data_filtering_2/output'
out_path = '/data/noah/inference/data_filtering_2/{}'
out_path = out_path.format('umm-maybe')
os.makedirs(out_path, exist_ok=True)
out_json_path = os.path.join(out_path, 'result.json')
results = []
for idx, f in enumerate(os.listdir(fake_path)):
    img_path = os.path.join(fake_path, f)
    image = Image.open(img_path).convert("RGB")
    score = image_classifier(image, pipe)
    
    results.append({'img_name':f, 'score':score})

    if score <= threshold:
        shutil.copy(img_path, os.path.join(out_path, f))

results = sorted(results, key=lambda x:x["score"])

js_human_path = '/data/noah/inference/data_filtering_2/result.json'
with open(js_human_path,'rb') as f:
    human = json.load(f)

for result in results:
    img_name = result['img_name']
    
    for h in human:
        if img_name == h['image_name']:
            h['score'] = float(result['score'])
            if h['score']<=threshold:
                h['state']='good'
            else:
                h['state']='not good'
            break

with open(out_json_path, 'w') as f:
    json.dump(human, f)