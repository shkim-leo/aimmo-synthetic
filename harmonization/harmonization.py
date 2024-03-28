import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model


class Harmonization:
    def __init__(self, ckpt_path, device="cuda:0", model_type="DucoNet"):
        self.device = device
        self.ckpt_path = ckpt_path
        self.predictor = Predictor(load_model(model_type, ckpt_path, verbose=True), torch.device(device))

    def harmonize(
        self,
        image,
        mask,
        resize_shape=(1024, 1024),
    ):
        ori_size = image.shape
        image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)
        mask = cv2.resize(mask, resize_shape, cv2.INTER_LINEAR)

        mask[mask <= 100] = 0
        mask[mask > 100] = 1
        mask = mask.astype(np.float32)

        pred = self.predictor.predict(image, mask)
        pred = cv2.resize(pred, ori_size[:-1][::-1])

        return pred
