import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import os 
import numpy as np
import pandas as  pd
from PIL import Image
import json
import shutil

threshold = 0.8
device = 'cuda:2'
torch.cuda.set_device(torch.device(device))

model='Nahrawy/AIorNot'
feature_extractor = AutoFeatureExtractor.from_pretrained(model)
model = AutoModelForImageClassification.from_pretrained(model).to(device)

fake_path = '/data/noah/inference/data_filtering_2/output'
out_path = '/data/noah/inference/data_filtering_2/{}'.format('Nahrawy')
os.makedirs(out_path, exist_ok=True)
out_json_path = os.path.join(out_path, 'result.json')
softmax = torch.nn.Softmax()
        
results = []
for idx, f in enumerate(os.listdir(fake_path)):
    img_path = os.path.join(fake_path, f)
    image = Image.open(img_path).convert("RGB")

    input = feature_extractor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input)
        logits = outputs.logits
        score = np.max(softmax(logits).detach().cpu().numpy())

    results.append({'img_name':f, 'score':str(score)})

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