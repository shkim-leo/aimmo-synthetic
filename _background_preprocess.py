import os
import csv
import shutil

dataset_folder = "/data/noah/dataset/background"
image_folder = os.path.join(dataset_folder, "image")
text_folder = os.path.join(dataset_folder, "text")
meta_path = os.path.join(dataset_folder, "metadata.csv")

text_names = sorted(os.listdir(text_folder))
image_names = sorted(os.listdir(image_folder))

data = []

for text_file_name, image_file_name in zip(text_names, image_names):
    text_file_path = os.path.join(text_folder, text_file_name)
    image_file_path = os.path.join(image_folder, image_file_name)
    caption = ""

    with open(text_file_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            caption = caption + line
        f.close()

    if "no human" in caption:
        file_path = os.path.join(dataset_folder, image_file_name)
        data.append({"file_path": file_path, "text": caption})
        shutil.move(image_file_path, file_path)

with open(meta_path, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["file_path", "text"])

    writer.writeheader()

    for row in data:
        writer.writerow(row)

print(f"{meta_path} 파일이 생성되었습니다.")
