#!/usr/bin/env python3
"""
evasion attack v1

Generates a targeted L-inf PGD adversarial image (center square only),
saves the PNG resized back to the input image's original dimensions
to be uploaded and analyzed by external vlm

- No network or HuggingFace calls; manual upload only.
- Uses torchvision.models.efficientnet_b0
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import models
import warnings

# CLI / Defaults
parser = argparse.ArgumentParser(description="Generate adversarial image for manual upload")
parser.add_argument("--input", "-i", default="test_image.png", help="Input image path")
parser.add_argument("--output", "-o", default="test_image_evasion.png", help="Output adversarial image path")
parser.add_argument("--target", "-t", type=int, default=620, help="Target ImageNet class index")
parser.add_argument("--steps", type=int, default=60, help="PGD iterations")
parser.add_argument("--eps", type=float, default=12.0, help="L-inf epsilon (pixels, 0-255)")
parser.add_argument("--center", type=int, default=224, help="Side length of central square to perturb")
args = parser.parse_args()
"""
Allows for arguements to be added in the command line, such as --input
allowing the user to specify the file path of the input image
"""

input_image = args.input
output_image = args.output
target_class = args.target
num_steps = args.steps
eps_pix = args.eps
center_side = args.center
step_pix = max(1.0, eps_pix / max(1, num_steps))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

print(f"Device: {device}")
print(f"Input: {input_image} -> Output: {output_image}")
print("Generating adversarial image...")
# Prints basic info such as cuda or cpu depending on the platform used


# Load lightweight model (EfficientNet-B0)
attack_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)
attack_model.eval()

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

def pixels_to_model_input(pix: torch.Tensor) -> torch.Tensor:
    """pix: (1,3,H,W) float in [0,255] -> normalized input"""
    x = pix / 255.0
    return (x - mean) / std

# Load image, record original size, and prepare 256x256 copy for processing
img_orig = Image.open(input_image).convert("RGB")
orig_w, orig_h = img_orig.size
print(f"Original image size: {orig_w}x{orig_h}")

# create processing image at 256x256 (attack is performed on this)
proc_size = (256, 256)
img = img_orig.resize(proc_size)
w, h = img.size
orig_np = np.array(img, dtype=np.uint8)
orig_t = torch.from_numpy(orig_np).permute(2,0,1).unsqueeze(0).float().to(device)  # (1,3,H,W) 0-255

# Make mask for center square (center relative to processing size)
top = (h - center_side) // 2
left = (w - center_side) // 2
mask = torch.zeros_like(orig_t, device=device)
mask[:, :, top:top+center_side, left:left+center_side] = 1.0

# Initialize adv image (random start inside epsilon-ball)
adv = orig_t.clone()
rand = (torch.rand_like(adv, device=device) * 2.0 - 1.0) * eps_pix
adv = torch.clamp(adv + rand * mask, 0.0, 255.0)
adv.requires_grad = True

# Simple targeted L-inf PGD (no momentum)
for step in range(num_steps):
    if adv.grad is not None:
        adv.grad.detach_()
        adv.grad.zero_()

    inp = pixels_to_model_input(adv)
    logits = attack_model(inp)

    # Targeted objective: maximize target class logit -> minimize negative target logit
    loss = -logits[0, target_class]
    loss.backward()

    with torch.no_grad():
        grad_sign = adv.grad.data.sign()
        adv = adv - step_pix * grad_sign * mask
        # Project into L-inf ball and clamp to [0,255]
        delta = torch.clamp(adv - orig_t, -eps_pix, eps_pix)
        adv = torch.clamp(orig_t + delta, 0.0, 255.0)
        adv.requires_grad = True

    # light progress prints
    if (step + 1) % 10 == 0 or step == 0 or step == num_steps - 1:
        with torch.no_grad():
            crop = adv[:, :, top:top+center_side, left:left+center_side]
            out = attack_model(pixels_to_model_input(crop))
            pred_idx = int(out.argmax(dim=1).item())
            print(f"[iter {step+1}/{num_steps}] center-crop predicted idx: {pred_idx}")


# Save adversarial image at processing size (256x256)
final_arr = adv.detach().cpu().squeeze(0).permute(1,2,0).clamp(0,255).byte().numpy()
adv_pil_proc = Image.fromarray(final_arr, mode="RGB")


# Resize final image back to original input size before saving
if (orig_w, orig_h) != proc_size:
    adv_pil_final = adv_pil_proc.resize((orig_w, orig_h), resample=Image.BICUBIC)
else:
    adv_pil_final = adv_pil_proc

adv_pil_final.save(output_image)
print(f"Saved adversarial image to: {os.path.abspath(output_image)} (resized to {orig_w}x{orig_h})")
