import os
import torch
from iharm.inference.transforms import NormalizeTensor, PadToDivisor, ToTensor, AddFlippedTensor
from torchvision import transforms
import numpy as np
import cv2


class Predictor(object):
    def __init__(self, net, device, with_flip=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.device = device
        self.net = net.to(self.device)
        self.net.eval()

        if hasattr(net, "depth"):
            size_divisor = 2 ** (net.depth + 1)
        else:
            size_divisor = 1

        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        self.transforms = [
            PadToDivisor(divisor=size_divisor, border_mode=0),
            ToTensor(self.device),
            NormalizeTensor(mean, std, self.device),
        ]
        if with_flip:
            self.transforms.append(AddFlippedTensor())

    def predict(self, image, mask, image_lab=None, return_numpy=True):
        if image_lab is None:
            image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

        with torch.no_grad():
            for transform in self.transforms:
                image, mask = transform.transform(image, mask)

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            comp_image_lab = transform(image_lab).unsqueeze(0).to(self.device)
            predicted_output = self.net(image, mask, image_lab=comp_image_lab)
            predicted_image = predicted_output["images"]

            for transform in reversed(self.transforms):
                predicted_image = transform.inv_transform(predicted_image)

            predicted_image = torch.clamp(predicted_image, 0, 255)

        if return_numpy:
            return predicted_image.cpu().numpy()
        else:
            return predicted_image
