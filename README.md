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
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/590/92ab729aee2b8/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=59092ab729aee2b8937d80cc1b35d1128275bd02a7e1bc911e7efa375bd97226 
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/1c0/3de08a69e9554/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=1c03de08a69e95542024477e0cde95fab3436804917133d3f00e67629d3fe902
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/de1/5388b8f70e4e1/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=de15388b8f70e4e17a05b23a4ae1f55a288c91449371bb8aeeb69184d40be17f
pip3 install --force torch-2.8.0-cp310-cp310-linux_aarch64.whl 
pip3 install --force torchvision-0.23.0-cp310-cp310-linux_aarch64.whl 
pip3 install --force torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl 
```
## sample img
```bash
wget https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?cs=srgb&dl=pexels-chevanon-1108099.jpg&fm=jpg
```
