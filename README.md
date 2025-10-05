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
##cusparselt
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

##test
```bash
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl


sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch release/0.20 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.20.0
python3 setup.py install --user # remove --user if installing in virtualenv


```
## sample img
```bash
wget "https://media.istockphoto.com/id/636475496/photo/portrait-of-brown-puppy-with-bokeh-background.jpg?s=612x612&w=0&k=20&c=Ot63dQOYplm0kLJdlSVWbtKGwGkuZfnfdwH5ry9a6EQ="
```
