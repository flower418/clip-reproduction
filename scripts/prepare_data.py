"""Download COCO and generate pairs.txt for CLIP training."""
import os
import sys
import json
import urllib.request
import zipfile

from tqdm import tqdm


COCO_BASE = "http://images.cocodataset.org"
ANNOTATIONS_URL = f"{COCO_BASE}/annotations/annotations_trainval2017.zip"


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.join(dest_dir, url.split("/")[-1])
    if os.path.exists(fname):
        print(f"{fname} already exists, skipping download")
    else:
        print(f"Downloading {url} ...")
        with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(fname)) as t:
            urllib.request.urlretrieve(url, fname, reporthook=t.update_to)

    extract_dir = os.path.join(dest_dir, os.path.splitext(os.path.basename(fname))[0])
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting {fname} ...")
        with zipfile.ZipFile(fname, "r") as z:
            z.extractall(dest_dir)
    return dest_dir


def prepare_coco(data_dir, split="val2017"):
    os.makedirs(data_dir, exist_ok=True)
    pairs_path = os.path.join(data_dir, "pairs.txt")

    if os.path.exists(pairs_path):
        print(f"pairs.txt already exists at {pairs_path}, skipping")
        return pairs_path

    # 1. Download annotations
    download_and_extract(ANNOTATIONS_URL, data_dir)
    ann_file = os.path.join(data_dir, "annotations", f"captions_{split}.json")

    # 2. Download images
    images_url = f"{COCO_BASE}/zips/{split}.zip"
    download_and_extract(images_url, data_dir)

    # 3. Parse captions and write pairs.txt
    with open(ann_file, "r") as f:
        data = json.load(f)

    # Build image_id -> file_name map
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    # Group captions by image_id
    img_captions = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_captions:
            img_captions[img_id] = []
        img_captions[img_id].append(ann["caption"])

    img_dir = os.path.join(data_dir, split)
    n_images = len(img_captions)
    print(f"Writing {n_images} image-caption pairs to {pairs_path}...")

    with open(pairs_path, "w") as f:
        for img_id, captions in img_captions.items():
            img_path = os.path.join(img_dir, id_to_file[img_id])
            for cap in captions:
                f.write(f"{img_path}\t{cap}\n")

    print(f"Done! {n_images} images -> {pairs_path}")
    return pairs_path


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val2017"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else f"./data/coco_{split}"
    prepare_coco(data_dir, split)
