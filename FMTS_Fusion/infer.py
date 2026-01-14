import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import cv2
import numpy as np
import argparse
import os
from graduate_hitsz.FDRL_Seg.FMTS_Fusion.network import SGLMamba as net


def load_model(model_path, in_channel=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = net(in_chans=in_channel, embed_dim=192).to(device)
    pretrained = torch.load(model_path, map_location=device)

    param_key = "params"
    state_dict = pretrained[param_key] if param_key in pretrained else pretrained
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


def read_image_cv2(path):
    """
    Read image using cv2 (BGR), convert to RGB, 
    float32 normalize, CHW, and return tensor [1,3,H,W]
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC -> CHW

    tensor = torch.from_numpy(img).unsqueeze(0)  # [1,3,H,W]
    return tensor


@torch.no_grad()
def fuse_one_image(model, device, vi_path, ir_path, save_path):
    img_vi = read_image_cv2(vi_path).to(device)
    img_ir = read_image_cv2(ir_path).to(device)

    output = model(img_vi, img_ir)
    output = output.detach()[0].cpu().numpy()  # [C,H,W]

    output = np.clip(output, 0, 1) * 255
    output = output.astype(np.uint8).transpose(1, 2, 0)  # CHW -> HWC

    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, output)
    print(f"Saved fused image → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to WMamba checkpoint (.pth)')
    parser.add_argument('--vi_path', type=str, required=True,
                        help='Visible image path')
    parser.add_argument('--ir_path', type=str, required=True,
                        help='Infrared image path')
    parser.add_argument('--save_path', type=str, default='fused_out.png',
                        help='Save path for fused image')
    parser.add_argument('--in_channel', type=int, default=3,
                        help='Input channel count (default 3)')

    args = parser.parse_args()

    model, device = load_model(args.model_path, in_channel=args.in_channel)
    fuse_one_image(model, device, args.vi_path, args.ir_path, args.save_path)
