import torch
import cv2
from time import time
from net import GeneratorNet
import argparse
from utils import img2tensor, tensor2img
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--vid_path", type=str, required=True, help="input video path")
parser.add_argument("--output_path", type=str, required=True, help="output video path")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/netG_299.pth",
    help="checkpoint for generator",
)
args = parser.parse_args()

if __name__ == "__main__":
    netG = GeneratorNet().cuda()
    with torch.no_grad():
        checkpoint = torch.load(args.checkpoint)
        netG.load_state_dict(checkpoint)

        input_path = args.vid_path
        output_path = args.output_path

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Could not open input video.")
            exit()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        start = time()

        for _ in tqdm(range(frame_count), desc="Processing Video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            # === Apply your operation ===
            img = cv2.resize(img, (768, 768))
            img_tensor = img2tensor(img)
            output_tensor = netG.forward(img_tensor)
            output_img = tensor2img(output_tensor)
            corrected_frame = cv2.resize(output_img, (frame_width, frame_height))

            out.write(corrected_frame)

        out.release()
        end = time()
        print(f"Video Corrected in {end - start} s")
