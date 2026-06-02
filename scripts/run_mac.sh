#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${1:-flickr30k}"

echo "===== Step 0: Setup environment ====="
if command -v conda &>/dev/null && ! conda env list | grep -q "^clip "; then
    echo "Creating conda env 'clip'..."
    conda create -n clip python=3.10 -y
fi
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate clip
fi
echo "Using Python: $(which python3)"

echo "===== Step 1: Install project ====="
pip install torch torchvision
pip install -e ".[dev]"

echo "===== Step 2: Download dataset ($DATASET) ====="
python scripts/prepare_data.py "$DATASET" "./data/$DATASET"

echo "===== Step 3: Start training ====="
python -m src.clip.train \
    --data "./data/$DATASET/pairs.txt" \
    --batch_size 64 \
    --epochs 32 \
    --lr 1e-3 \
    --wd 0.2 \
    --warmup 2000 \
    --save_dir ./checkpoints
