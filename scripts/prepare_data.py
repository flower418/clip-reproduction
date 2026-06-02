"""Download COCO and generate pairs.txt for CLIP training."""
import os
import sys
from torchvision.datasets import CocoCaptions


def prepare_coco(data_dir, split="val2017"):
    """split: 'val2017' (~1GB, 5k images) or 'train2017' (~18GB, 118k images)"""
    os.makedirs(data_dir, exist_ok=True)
    pairs_path = os.path.join(data_dir, "pairs.txt")

    if os.path.exists(pairs_path):
        print(f"pairs.txt already exists at {pairs_path}, skipping")
        return pairs_path

    ann_file = os.path.join(data_dir, "annotations", f"captions_{split}.json")
    print(f"Downloading COCO {split}...")
    dataset = CocoCaptions(root=data_dir, annFile=ann_file, download=True)

    img_dir = os.path.join(data_dir, split)
    print(f"Writing {len(dataset)} image-caption pairs to {pairs_path}...")

    with open(pairs_path, "w") as f:
        for i in range(len(dataset)):
            img_id = dataset.ids[i]
            img_path = os.path.join(img_dir, f"{img_id:012d}.jpg")
            _, captions = dataset[i]
            for cap in captions:
                f.write(f"{img_path}\t{cap}\n")

    print(f"Done! {len(dataset)} images -> {pairs_path}")
    return pairs_path


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val2017"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else f"./data/coco_{split}"
    prepare_coco(data_dir, split)
