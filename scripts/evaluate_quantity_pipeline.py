import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantity_estimator import QuantityEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO segmentation model on real thali images for quantity estimation."
    )
    parser.add_argument("--model", required=True, help="Path to trained best.pt checkpoint")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", required=True, help="Directory to write visuals and report")
    parser.add_argument("--sample-count", type=int, default=12, help="Number of sample visuals to save")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="Prediction NMS IoU threshold")
    return parser.parse_args()


def load_data_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


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


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def best_match_ious(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> List[float]:
    if not pred_masks or not gt_masks:
        return []

    used_gt = set()
    ious = []
    for pred in pred_masks:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_masks):
            if idx in used_gt:
                continue
            iou = mask_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            used_gt.add(best_idx)
        ious.append(best_iou)
    return ious


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def extract_predictions(result: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    compartment_detections: List[Dict[str, Any]] = []
    food_detections: List[Dict[str, Any]] = []

    if result.masks is None or result.boxes is None:
        return compartment_detections, food_detections

    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    names = result.names
    orig_h, orig_w = result.orig_shape

    resized_masks = []
    for mask in masks:
        resized = cv2.resize(mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5
        resized_masks.append(resized)

    for idx, (mask, cls_idx, conf, box) in enumerate(zip(resized_masks, classes, confidences, boxes)):
        class_name = names.get(int(cls_idx), str(cls_idx))
        bbox = tuple(int(round(v)) for v in box.tolist())
        detection = {
            "id": idx,
            "class_name": class_name,
            "mask": mask,
            "confidence": float(conf),
            "bbox": bbox,
            "centroid": mask_centroid(mask),
        }
        if class_name == "compartment":
            compartment_detections.append(detection)
        else:
            food_detections.append(detection)

    return compartment_detections, food_detections


def initialize_mapped(compartment_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapped = []
    for c_idx, comp in enumerate(compartment_detections):
        mapped.append(
            {
                "compartment_idx": c_idx,
                "mask": comp["mask"],
                "bbox": comp["bbox"],
                "foods": [],
                "total_food_mask": np.zeros_like(comp["mask"], dtype=bool),
            }
        )
    return mapped


def assign_food(mapped: List[Dict[str, Any]], comp_idx: int, food: Dict[str, Any]) -> None:
    mapped[comp_idx]["foods"].append(food)
    mapped[comp_idx]["total_food_mask"] = np.logical_or(mapped[comp_idx]["total_food_mask"], food["mask"] > 0)


def map_food_mask_ioa(
    estimator: QuantityEstimator, food_detections: List[Dict[str, Any]], compartment_detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return estimator.map_food_to_compartments(food_detections, [c["mask"] for c in compartment_detections])


def map_food_centroid(
    food_detections: List[Dict[str, Any]], compartment_detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    mapped = initialize_mapped(compartment_detections)
    for food in food_detections:
        centroid = food.get("centroid")
        if centroid is None:
            continue
        cx, cy = centroid
        candidates = []
        for idx, comp in enumerate(compartment_detections):
            comp_mask = comp["mask"]
            if 0 <= cy < comp_mask.shape[0] and 0 <= cx < comp_mask.shape[1] and comp_mask[cy, cx]:
                candidates.append((idx, np.sum(comp_mask)))
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda item: item[1])[0]
        assign_food(mapped, best_idx, food)
    return mapped


def map_food_box_iou(
    food_detections: List[Dict[str, Any]], compartment_detections: List[Dict[str, Any]], min_iou: float = 0.01
) -> List[Dict[str, Any]]:
    mapped = initialize_mapped(compartment_detections)
    for food in food_detections:
        food_box = food.get("bbox")
        if food_box is None:
            bbox = mask_bbox(food["mask"] > 0)
            if bbox is None:
                continue
            food_box = bbox
        best_iou = 0.0
        best_idx = -1
        for idx, comp in enumerate(compartment_detections):
            comp_box = comp.get("bbox")
            if comp_box is None:
                continue
            iou = bbox_iou(food_box, comp_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= min_iou:
            assign_food(mapped, best_idx, food)
    return mapped


def map_food_hybrid(
    estimator: QuantityEstimator, food_detections: List[Dict[str, Any]], compartment_detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    mapped = map_food_mask_ioa(estimator, food_detections, compartment_detections)
    assigned_ids = {food["id"] for comp in mapped for food in comp["foods"]}
    remaining = [food for food in food_detections if food["id"] not in assigned_ids]
    if not remaining:
        return mapped

    centroid_mapped = map_food_centroid(remaining, compartment_detections)
    centroid_ids = {food["id"] for comp in centroid_mapped for food in comp["foods"]}
    for comp_idx, comp in enumerate(centroid_mapped):
        for food in comp["foods"]:
            assign_food(mapped, comp_idx, food)

    remaining = [food for food in remaining if food["id"] not in centroid_ids]
    if not remaining:
        return mapped

    box_mapped = map_food_box_iou(remaining, compartment_detections)
    for comp_idx, comp in enumerate(box_mapped):
        for food in comp["foods"]:
            assign_food(mapped, comp_idx, food)
    return mapped


def visualize_prediction(
    image_path: Path, mapped: List[Dict[str, Any]], gt_compartments: List[np.ndarray], out_path: Path, title: str
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return

    pred_colors = [(255, 120, 60), (0, 200, 120), (120, 80, 255), (255, 200, 80), (255, 80, 180), (80, 220, 255)]

    # Ground-truth compartment outlines in white for quick comparison.
    for gt_mask in gt_compartments:
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), 1)

    cv2.rectangle(image, (8, 8), (430, 32), (0, 0, 0), -1)
    cv2.putText(image, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for idx, comp in enumerate(mapped):
        color = pred_colors[idx % len(pred_colors)]
        mask = comp["mask"].astype(bool)
        image[mask] = (image[mask] * 0.75 + np.array(color) * 0.25).astype(np.uint8)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)

        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"]:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 20, 30 + idx * 25

        foods = [f["class_name"] for f in comp["foods"]]
        fill = comp.get("fill_ratio", 0.0)
        text = f"C{idx} {fill:.2f}: {','.join(foods) if foods else 'EMPTY'}"
        cv2.rectangle(image, (cx - 2, cy - 18), (cx + 220, cy + 4), (0, 0, 0), -1)
        cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    data = load_data_yaml(data_path)
    dataset_root = Path(data["path"])
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    class_names = {int(k): v for k, v in data["names"].items()}

    model = YOLO(str(model_path))
    estimator = QuantityEstimator(
        thali_spec={
            "quantity_reject_below": 0.5,
            "quantity_pass_from": 0.6,
            "expected_presence_items": [],
        }
    )

    image_paths = sorted(image_dir.glob("*"))
    sample_paths = image_paths[: args.sample_count]

    report_rows = []
    compartment_iou_scores: List[float] = []
    fill_ratio_gaps: List[float] = []
    likely_bad_compartments = 0
    strategy_empty_counts = {"mask_ioa": 0, "centroid": 0, "box_iou": 0, "hybrid": 0}
    strategy_nonempty_counts = {"mask_ioa": 0, "centroid": 0, "box_iou": 0, "hybrid": 0}

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        results = model.predict(source=str(image_path), conf=args.conf, iou=args.iou, verbose=False)
        if not results:
            continue

        result = results[0]
        pred_compartments, food_detections = extract_predictions(result)
        mapped_variants = {
            "mask_ioa": map_food_mask_ioa(estimator, food_detections, pred_compartments),
            "centroid": map_food_centroid(food_detections, pred_compartments),
            "box_iou": map_food_box_iou(food_detections, pred_compartments),
            "hybrid": map_food_hybrid(estimator, food_detections, pred_compartments),
        }
        mapped = mapped_variants["hybrid"]
        fails, needs_vlm = estimator.quantity_pre_filter(mapped)

        label_path = label_dir / f"{image_path.stem}.txt"
        gt_masks = load_label_masks(label_path, image.shape[:2], class_names)
        gt_compartments = gt_masks.get("compartment", [])

        pred_comp_ious = best_match_ious([c["mask"] for c in pred_compartments], gt_compartments)
        compartment_iou_scores.extend(pred_comp_ious)

        gt_food_masks = []
        for class_name, masks in gt_masks.items():
            if class_name != "compartment":
                gt_food_masks.extend((class_name, m) for m in masks)

        for strategy_name, strategy_mapped in mapped_variants.items():
            empty_count = sum(1 for comp in strategy_mapped if not comp["foods"])
            strategy_empty_counts[strategy_name] += empty_count
            strategy_nonempty_counts[strategy_name] += max(0, len(strategy_mapped) - empty_count)

        for comp_idx, comp in enumerate(mapped):
            comp["fill_ratio"] = estimator.calculate_fill_ratio(comp["mask"], comp["total_food_mask"])

            if comp_idx < len(gt_compartments):
                gt_comp = gt_compartments[comp_idx]
                gt_food_union = np.zeros_like(gt_comp, dtype=bool)
                for _, gt_food_mask in gt_food_masks:
                    if np.logical_and(gt_food_mask, gt_comp).sum() > 0:
                        gt_food_union = np.logical_or(gt_food_union, gt_food_mask)
                gt_fill = estimator.calculate_fill_ratio(gt_comp, gt_food_union)
                fill_ratio_gaps.append(abs(comp["fill_ratio"] - gt_fill))

        low_quality_pred_count = sum(1 for iou in pred_comp_ious if iou < 0.5)
        likely_bad_compartments += low_quality_pred_count

        report_rows.append(
            {
                "image": image_path.name,
                "pred_compartments": len(pred_compartments),
                "gt_compartments": len(gt_compartments),
                "avg_pred_compartment_iou": round(float(np.mean(pred_comp_ious)), 4) if pred_comp_ious else 0.0,
                "empty_mask_ioa": sum(1 for comp in mapped_variants["mask_ioa"] if not comp["foods"]),
                "empty_centroid": sum(1 for comp in mapped_variants["centroid"] if not comp["foods"]),
                "empty_box_iou": sum(1 for comp in mapped_variants["box_iou"] if not comp["foods"]),
                "empty_hybrid": sum(1 for comp in mapped if not comp["foods"]),
                "auto_fail_count": len(fails),
                "needs_quality_count": len(needs_vlm),
                "issues": ";".join(f["issue"] for f in fails),
            }
        )

        if image_path in sample_paths:
            for strategy_name, strategy_mapped in mapped_variants.items():
                visualize_prediction(
                    image_path,
                    strategy_mapped,
                    gt_compartments,
                    visuals_dir / f"{image_path.stem}_{strategy_name}{image_path.suffix}",
                    f"{strategy_name} mapping",
                )

    avg_compartment_iou = float(np.mean(compartment_iou_scores)) if compartment_iou_scores else 0.0
    median_compartment_iou = float(np.median(compartment_iou_scores)) if compartment_iou_scores else 0.0
    avg_fill_gap = float(np.mean(fill_ratio_gaps)) if fill_ratio_gaps else 0.0

    summary = {
        "model": str(model_path),
        "split": args.split,
        "images_evaluated": len(report_rows),
        "sample_visuals_saved": len(sample_paths),
        "avg_compartment_iou": round(avg_compartment_iou, 4),
        "median_compartment_iou": round(median_compartment_iou, 4),
        "avg_fill_ratio_gap": round(avg_fill_gap, 4),
        "compartments_below_iou_0_5": int(likely_bad_compartments),
        "mapping_nonempty_compartments": strategy_nonempty_counts,
        "mapping_empty_compartments": strategy_empty_counts,
        "recommendation": (
            "Improve compartment annotations first"
            if avg_compartment_iou < 0.6 or avg_fill_gap > 0.12
            else "Current compartment masks look usable for quantity testing"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "per_image_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()) if report_rows else [])
        if report_rows:
            writer.writeheader()
            writer.writerows(report_rows)

    print(json.dumps(summary, indent=2))
    print(f"Visuals written to: {visuals_dir}")
    print(f"Per-image report written to: {out_dir / 'per_image_report.csv'}")


if __name__ == "__main__":
    main()
