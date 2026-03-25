import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from pycocotools import mask as cocomask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a COCO-seg dataset and convert it to YOLO-seg format."
    )
    parser.add_argument("--coco-json", required=True, help="Path to _annotations.coco.json")
    parser.add_argument("--images-dir", required=True, help="Directory with source images")
    parser.add_argument("--out-dir", required=True, help="Output dataset root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument(
        "--ignore-class",
        action="append",
        default=["objects"],
        help="Class name to ignore (can be passed multiple times)",
    )
    return parser.parse_args()


def coco_segmentation_to_polygons(segmentation) -> List[List[float]]:
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return []
        if isinstance(segmentation[0], list):
            return segmentation
        return [segmentation]
    if isinstance(segmentation, dict):
        mask = cocomask.decode(segmentation)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary = (mask > 0).astype(np.uint8)

        polygons: List[List[float]] = []
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            poly = contour.reshape(-1, 2).astype(float).flatten().tolist()
            if len(poly) >= 6:
                polygons.append(poly)
        return polygons
    return []


def normalize_polygon(poly: List[float], width: int, height: int) -> List[float]:
    norm = []
    for i in range(0, len(poly), 2):
        x = min(max(poly[i] / width, 0.0), 1.0)
        y = min(max(poly[i + 1] / height, 0.0), 1.0)
        norm.extend([x, y])
    return norm


def split_images(image_ids: List[int], train_ratio: float, val_ratio: float, seed: int) -> Dict[int, str]:
    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    split_map: Dict[int, str] = {}
    for image_id in shuffled[:n_train]:
        split_map[image_id] = "train"
    for image_id in shuffled[n_train : n_train + n_val]:
        split_map[image_id] = "val"
    for image_id in shuffled[n_train + n_val :]:
        split_map[image_id] = "test"
    return split_map


def ensure_structure(out_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    ignore: Set[str] = set(args.ignore_class or [])

    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    id_to_name = {cat["id"]: cat["name"] for cat in categories}
    kept_categories = [cat for cat in sorted(categories, key=lambda x: x["id"]) if cat["name"] not in ignore]
    old_to_new_class = {cat["id"]: idx for idx, cat in enumerate(kept_categories)}
    class_names = [cat["name"] for cat in kept_categories]

    images = coco.get("images", [])
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    split_map = split_images(
        image_ids=[img["id"] for img in images],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    ensure_structure(out_dir)

    label_counts = defaultdict(int)
    copied_images = 0
    skipped_rle = 0
    skipped_classes = 0

    for img in images:
        image_id = img["id"]
        split = split_map[image_id]
        width = img["width"]
        height = img["height"]
        file_name = img["file_name"]

        src_image = images_dir / file_name
        if not src_image.exists():
            print(f"[WARN] Missing image file, skipping: {src_image}")
            continue

        dst_image = out_dir / "images" / split / file_name
        dst_label = out_dir / "labels" / split / f"{Path(file_name).stem}.txt"
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_image, dst_image)
        copied_images += 1

        lines: List[str] = []
        for ann in anns_by_image.get(image_id, []):
            old_class = ann["category_id"]
            class_name = id_to_name.get(old_class, "unknown")
            if old_class not in old_to_new_class:
                skipped_classes += 1
                continue

            polygons = coco_segmentation_to_polygons(ann.get("segmentation"))
            if not polygons:
                skipped_rle += 1
                continue

            class_idx = old_to_new_class[old_class]
            for poly in polygons:
                if len(poly) < 6 or len(poly) % 2 != 0:
                    continue
                norm_poly = normalize_polygon(poly, width, height)
                if len(norm_poly) < 6:
                    continue
                lines.append(f"{class_idx} " + " ".join(f"{v:.6f}" for v in norm_poly))
                label_counts[class_name] += 1

        with dst_label.open("w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines))
                f.write("\n")

    data_yaml = out_dir / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")

    print("\n=== DATASET PREP COMPLETE ===")
    print(f"Copied images: {copied_images}")
    print(f"Output dataset: {out_dir}")
    print(f"data.yaml: {data_yaml}")
    print(f"Skipped class labels (ignored classes): {skipped_classes}")
    print(f"Skipped labels (non-polygon segmentations): {skipped_rle}")
    print("Label count by class:")
    for class_name, count in sorted(label_counts.items(), key=lambda x: x[0]):
        print(f"  - {class_name}: {count}")


if __name__ == "__main__":
    main()
