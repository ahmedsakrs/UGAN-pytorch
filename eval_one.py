import torch
import torch.nn as nn 
import cv2,datetime,os
from time import time
from net import GeneratorNet
import argparse
import numpy as np
from utils import img2tensor,tensor2img

parser = argparse.ArgumentParser()
parser.add_argument('--img_path',type=str,default='n01496331_115.jpg',help='Input the image path')
parser.add_argument('--checkpoint',type=str,default='checkpoints/netG_299.pth',help='checkpoint for generator')
args = parser.parse_args()

if __name__ == "__main__":
    netG = GeneratorNet().cuda()
    with torch.no_grad():
        checkpoint = torch.load(args.checkpoint)
        netG.load_state_dict(checkpoint)
        img_path = args.img_path
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img = cv2.resize(img,(768,768))
        start = time()
        img_tensor = img2tensor(img)
        output_tensor = netG.forward(img_tensor)
        output_img = tensor2img(output_tensor)
        end = time()
        print(f"Done in {end - start} s")
        output_img = cv2.resize(output_img, (width, height))
        cv2.imwrite('output.jpg',output_img)

