#!/bin/bash
# Prepare your data as: img_path<TAB>caption (one per line)
# Example:
#   /data/images/cat_001.jpg	A cat sitting on a sofa
#   /data/images/dog_042.jpg	A dog running in a park

python -m src.clip.train \
    --data ./data/pairs.txt \
    --batch_size 256 \
    --epochs 32 \
    --lr 1e-3 \
    --wd 0.2 \
    --warmup 2000 \
    --save_dir ./checkpoints
