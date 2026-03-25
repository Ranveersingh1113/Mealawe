import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantity_estimator import QuantityEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate per-food quantity thresholds from YOLO segmentation labels.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to use for calibration")
    parser.add_argument("--out", required=True, help="Output JSON file for recommended thresholds")
    parser.add_argument("--min-samples", type=int, default=8, help="Minimum samples required before emitting a rule")
    return parser.parse_args()


def load_data_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_masks(label_path: Path, image_shape: Tuple[int, int], class_names: Dict[int, str]) -> Dict[str, List[np.ndarray]]:
    height, width = image_shape
    masks_by_class: Dict[str, List[np.ndarray]] = {}
    if not label_path.exists():
        return masks_by_class

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        class_idx = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue

        polygon = []
        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * width))
            y = int(round(coords[i + 1] * height))
            polygon.append([x, y])

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
        class_name = class_names[class_idx]
        masks_by_class.setdefault(class_name, []).append(mask.astype(bool))

    return masks_by_class


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def main() -> None:
    args = parse_args()
    data = load_data_yaml(Path(args.data))
    dataset_root = Path(data["path"])
    class_names = {int(k): v for k, v in data["names"].items()}
    estimator = QuantityEstimator({})

    ratios_by_food: Dict[str, List[float]] = {}

    for split in args.splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue

        for image_path in sorted(image_dir.glob("*")):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            gt_masks = load_label_masks(label_dir / f"{image_path.stem}.txt", image.shape[:2], class_names)
            compartments = gt_masks.get("compartment", [])
            if not compartments:
                continue

            food_detections = []
            counter = 0
            for class_name, masks in gt_masks.items():
                if class_name == "compartment":
                    continue
                for mask in masks:
                    food_detections.append({"id": counter, "class_name": class_name, "mask": mask})
                    counter += 1

            mapped = estimator.map_food_to_compartments(food_detections, compartments)
            for comp in mapped:
                if not comp["foods"]:
                    continue
                fill_ratio = estimator.calculate_fill_ratio(comp["mask"], comp["total_food_mask"])
                primary_food = estimator.get_primary_food(comp["foods"])
                if primary_food is None or primary_food in estimator.PRESENCE_ONLY_ITEMS:
                    continue
                ratios_by_food.setdefault(primary_food, []).append(fill_ratio)

    recommended_rules: Dict[str, Dict[str, float]] = {}
    stats: Dict[str, Dict[str, Any]] = {}
    for food, ratios in sorted(ratios_by_food.items()):
        if len(ratios) < args.min_samples:
            stats[food] = {"count": len(ratios), "skipped": True}
            continue

        recommended_rules[food] = {
            "low": round(percentile(ratios, 10), 3),
            "target_min": round(percentile(ratios, 25), 3),
            "target_max": round(percentile(ratios, 75), 3),
            "high": round(percentile(ratios, 90), 3),
        }
        stats[food] = {
            "count": len(ratios),
            "min": round(min(ratios), 3),
            "median": round(percentile(ratios, 50), 3),
            "max": round(max(ratios), 3),
            **recommended_rules[food],
        }

    output = {
        "bulk_foods": sorted(estimator.DEFAULT_BULK_ITEMS),
        "expected_presence_items": ["roti", "salad", "sweet"],
        "food_quantity_rules": recommended_rules,
        "stats": stats,
        "notes": {
            "low": "Below this is clearly underfilled.",
            "target_min": "Lower edge of the healthy range.",
            "target_max": "Upper edge of the healthy range.",
            "high": "Above this is clearly overfilled.",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"Saved threshold recommendations to: {out_path}")


if __name__ == "__main__":
    main()
