#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "===== Step 1: Install dependencies ====="
pip install -e ".[dev,wandb]" -q

echo "===== Step 2: Download COCO dataset ====="
python scripts/prepare_data.py ./data/coco

echo "===== Step 3: Start training ====="
python -m src.clip.train \
    --data ./data/coco/pairs.txt \
    --batch_size 256 \
    --epochs 32 \
    --lr 1e-3 \
    --wd 0.2 \
    --warmup 2000 \
    --save_dir ./checkpoints \
    --wandb \
    --wandb_project clip-reproduction \
    --wandb_name "clip-vit-b32-bs256"
