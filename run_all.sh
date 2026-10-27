#!/usr/bin/env bash
set -e

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate

python -m pip install --upgrade pip

# CPU-only PyTorch versions compatible with Python 3.8.
# This project does not require CUDA/GPU packages.
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt

python python/train_export_mnist.py --epochs 3
python python/verify_quantized_model.py --all --debug

cd simulation
make clean || true
make SIM=icarus
BLUR_SIZES=1,3,5,7 make blur SIM=icarus
