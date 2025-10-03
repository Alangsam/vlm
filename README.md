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
