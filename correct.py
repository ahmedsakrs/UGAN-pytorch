import torch
import cv2
from net import GeneratorNet
import numpy as np
from utils import img2tensor,tensor2img


class ColorCorrector:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path)
        with torch.no_grad():
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.model = GeneratorNet().to(device)
            self.model.load_state_dict(checkpoint)
    
    def correct(self, frame: np.ndarray, size=(512, 512)) -> np.ndarray:
        with torch.no_grad():
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, size)
            img_tensor = img2tensor(frame)
            output_tensor = self.model.forward(img_tensor)
            output_img = tensor2img(output_tensor)
            output_img = cv2.resize(output_img, (width, height))
            return output_img