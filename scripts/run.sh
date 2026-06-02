#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${1:-flickr30k}"
ENV_NAME="clip"

echo "===== Step 0: Setup environment ====="
if command -v conda &>/dev/null; then
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        echo "Creating conda env '${ENV_NAME}'..."
        conda create -n "$ENV_NAME" python=3.10 -y
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    echo "Installing PyTorch with CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "No conda found, using system Python $(python3 --version)"
fi

echo "===== Step 1: Install project ====="
pip install -e ".[dev,wandb]"

echo "===== Step 2: Download dataset ($DATASET) ====="
python scripts/prepare_data.py "$DATASET" "./data/$DATASET"

echo "===== Step 3: Start training ====="
python -m src.clip.train \
    --data "./data/$DATASET/pairs.txt" \
    --batch_size 256 \
    --epochs 32 \
    --lr 1e-3 \
    --wd 0.2 \
    --warmup 2000 \
    --save_dir ./checkpoints \
    --wandb \
    --wandb_project clip-reproduction
