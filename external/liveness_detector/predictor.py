# predictor.py

import torch
import numpy as np
from .anti_spoof_predict import AntiSpoofPredict

class LivenessDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.detector = AntiSpoofPredict(device_id=0)

    def predict(self, face_img: np.ndarray) -> float:
        """
        输入 BGR 彩色图像（单张人脸），返回活体置信度（越高越真实）
        """
        h, w, _ = face_img.shape
        fake_bbox = [0, 0, w, h]

        _, prediction = self.detector.predict(face_img, fake_bbox, self.model_path)
        prob = torch.softmax(torch.tensor(prediction), dim=0)[1].item()  # index 1 = real
        return prob
