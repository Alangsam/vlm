# VLM Playground

Quick GPU checks and simple scripts for running small vision-language models.

## Setup (transformers method)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install transformers pillow accelerate safetensors
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
```
### for jetson
```bash
pip3 install 'numpy<2'
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/590/92ab729aee2b8/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=59092ab729aee2b8937d80cc1b35d1128275bd02a7e1bc911e7efa375bd97226 
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/1c0/3de08a69e9554/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=1c03de08a69e95542024477e0cde95fab3436804917133d3f00e67629d3fe902
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/de1/5388b8f70e4e1/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=de15388b8f70e4e17a05b23a4ae1f55a288c91449371bb8aeeb69184d40be17f
pip3 install torch-2.8.0-cp310-cp310-linux_aarch64.whl 
pip3 install torchvision-0.23.0-cp310-cp310-linux_aarch64.whl 
pip3 install --force torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl

link cuda tools:
$ export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### troubleshooting cpu only
```bash

python - <<'PY'
import os, ctypes, sys
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)              # None => CPU-only build
print("torch.backends.cuda.is_built():", torch.backends.cuda.is_built())
print("torch.cuda.is_available():", torch.cuda.is_available())
print("cudnn available:", torch.backends.cudnn.is_available())
print("env LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
for name in ["libcuda.so.1","libcudart.so.12","libcublas.so.12","libcudnn.so.9","libcudss.so.0"]:
    try:
        ctypes.CDLL(name); print("loaded:", name)
    except OSError as e:
        print("MISSING:", name, "->", e)
PY
```
### cusparselt-save to file and run
```bash
#!/bin/bash

set -ex

# cuSPARSELt license: https://docs.nvidia.com/cuda/cusparselt/license.html

CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64"
CUSPARSELT_VERSION="0.7.1.0"
CUSPARSELT_NAME="libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive"

# Create and enter temp working directory
mkdir -p tmp_cusparselt && cd tmp_cusparselt

# Download the archive
curl --retry 3 -OLs "${CUSPARSELT_URL}/${CUSPARSELT_NAME}.tar.xz"

# Extract the archive
tar xf "${CUSPARSELT_NAME}.tar.xz"

# Install headers and libraries
cp -a "${CUSPARSELT_NAME}/include/"* /usr/local/cuda/include/
cp -a "${CUSPARSELT_NAME}/lib/"* /usr/local/cuda/lib64/

# Clean up
cd ..
rm -rf tmp_cusparselt

# Update linker cache
ldconfig
```

### fix missing cudSS
```bash

# alternative download sources
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl



wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudss
```
### sample img
```bash
wget "https://media.istockphoto.com/id/636475496/photo/portrait-of-brown-puppy-with-bokeh-background.jpg?s=612x612&w=0&k=20&c=Ot63dQOYplm0kLJdlSVWbtKGwGkuZfnfdwH5ry9a6EQ="
```

## Example Run
```bash
python main.py --backend moondream --image ./samples/cute_dog.jpg
```
