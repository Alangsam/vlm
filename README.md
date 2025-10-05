# VLM Playground

Quick GPU checks and simple scripts for running small vision-language models.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers pillow accelerate safetensors
```
## for jetson
```bash
pip3 install 'numpy<2'
wget https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/6cc/6ecfe8a5994fd/torch-2.6.0-cp310-cp310-linux_aarch64.whl#sha256=6cc6ecfe8a5994fd6d58fb6d6eb73ff2437428bb4953f3ebaa409f83a5f4db99 
wget https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/aa2/2da8dcf4c4c8d/torchvision-0.21.0-cp310-cp310-linux_aarch64.whl#sha256=aa22da8dcf4c4c8dc897e7922b1ef25cb0fe350e1a358168be87a854ad114531
wget https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/dda/ce98dc7d89263/torchaudio-2.6.0-cp310-cp310-linux_aarch64.whl#sha256=ddace98dc7d892634d2e5b08593436f3e3b1247a1cb11c0d5f4e5ccf64a9be8c 
pip3 install --force torch-2.6.0-cp310-cp310-linux_aarch64.whl 
pip3 install --force torchvision-0.21.0-cp310-cp310-linux_aarch64.whl 
pip3 install --force torchaudio-2.6.0-cp310-cp310-linux_aarch64.whl 
```
