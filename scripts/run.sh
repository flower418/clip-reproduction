#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${1:-flickr30k}"

echo "===== Step 1: Install dependencies ====="
pip install -e ".[dev,wandb]" -q

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
