#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SPLIT="${1:-train2017}"

echo "===== Step 0: Setup environment ====="
if command -v conda &>/dev/null; then
    if ! conda env list | grep -q "^clip "; then
        echo "Creating conda env 'clip'..."
        conda create -n clip python=3.10 -y
    fi
    eval "$(conda shell.bash hook)"
    conda activate clip
    echo "Installing PyTorch with CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "No conda found, using system Python $(python3 --version)"
fi

echo "===== Step 1: Install project ====="
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
pip install -e ".[dev,wandb]" -i "$MIRROR" --trusted-host pypi.tuna.tsinghua.edu.cn

echo "===== Step 2: Download COCO $SPLIT ====="
python scripts/prepare_data.py "$SPLIT" "./data/coco_$SPLIT"

echo "===== Step 3: Start training ====="
python -m src.clip.train \
    --data "./data/coco_$SPLIT/pairs.txt" \
    --batch_size 256 \
    --epochs 32 \
    --lr 1e-3 \
    --wd 0.2 \
    --warmup 2000 \
    --save_dir ./checkpoints \
    --wandb \
    --wandb_project clip-reproduction
