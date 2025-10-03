import sys, platform, torch
from torch import nn
# nn is neural network toolbox

print(f"[INFO] Python: {sys.version.split()[0]} on {platform.system()} {platform.release()}")
print(f"[INFO] PyTorch: {torch.__version__}")

#can pytorch talk to the gpu
print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("[ERR] CUDA not available. If nvidia-smi works in WSL, you likely installed the CPU wheel; reinstall cu124.")

print(f"[INFO] CUDA devices: {torch.cuda.device_count()}")
print(f"[INFO] Current device: {torch.cuda.current_device()}")
print(f"[INFO] Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

#a tensor is an N-dimensional array
a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")

#matrix multiplication test
c = a @ b
print("[OK] GPU matmul:", c.shape)

# in_channels=3: input has 3 planes (RGB).
# out_channels=8: we’re learning 8 different filters (detectors).
# kernel_size=3: each filter is a small 3×3 window that slides over the image.
# padding=1: add a 1-pixel border of zeros so height/width stay the same after sliding.
m = nn.Conv2d(3, 8, kernel_size=3, padding=1).cuda()
# batch = 1: how many images we process together.
# channels = 3: red, green, blue (RGB). Each channel is like a grayscale image.
# height = 64, width = 64: image size.
x = torch.randn(1, 3, 64, 64, device="cuda")
y = m(x)
print("[OK] cuDNN conv:", y.shape)
