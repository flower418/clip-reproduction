"""Download Flickr30k / COCO and generate pairs.txt for CLIP training."""
import os
import sys
from pathlib import Path


def prepare_flickr30k(data_dir: str = "./data/flickr30k"):
    os.makedirs(data_dir, exist_ok=True)
    pairs_path = os.path.join(data_dir, "pairs.txt")

    if os.path.exists(pairs_path):
        print(f"pairs.txt already exists at {pairs_path}, skipping")
        return pairs_path

    from datasets import load_dataset

    print("Downloading Flickr30k (~5GB, 31k images)...")
    dataset = load_dataset("nlphuji/flickr30k", split="test", cache_dir=data_dir, trust_remote_code=True)

    img_dir = os.path.join(data_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Saving images to {img_dir} and writing {pairs_path}...")
    with open(pairs_path, "w") as f:
        for i, sample in enumerate(dataset):
            img = sample["image"]
            img_path = os.path.join(img_dir, f"{i:08d}.jpg")
            img.save(img_path)

            for caption in sample["caption"]:
                f.write(f"{img_path}\t{caption}\n")

    print(f"Done! {len(dataset)} images -> {pairs_path}")
    return pairs_path


def prepare_coco(data_dir: str = "./data/coco"):
    os.makedirs(data_dir, exist_ok=True)
    pairs_path = os.path.join(data_dir, "pairs.txt")

    if os.path.exists(pairs_path):
        print(f"pairs.txt already exists at {pairs_path}, skipping")
        return pairs_path

    from torchvision.datasets import CocoCaptions

    ann_file = os.path.join(data_dir, "annotations", "captions_train2017.json")
    print("Downloading COCO train2017 (~18GB, 118k images)...")

    dataset = CocoCaptions(root=data_dir, annFile=ann_file, download=True)

    img_dir = os.path.join(data_dir, "train2017")
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
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "flickr30k"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else f"./data/{dataset_name}"

    if dataset_name == "flickr30k":
        prepare_flickr30k(data_dir)
    elif dataset_name == "coco":
        prepare_coco(data_dir)
    else:
        print(f"Usage: python {__file__} [flickr30k|coco] [data_dir]")
        sys.exit(1)
