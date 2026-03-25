from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.schemas import CompartmentSummary, DetectedFood, DetectionBox, QuantityStageResponse
from quantity_estimator import QuantityEstimator

VALID_COMPARTMENT_COUNTS = (3, 5, 7, 8)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def _mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(2.0, 0.01 * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(point[0][0]), int(point[0][1])] for point in approx]


def _bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def _clip_bbox_to_shape(bbox: Tuple[int, int, int, int], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    height, width = shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return x1, y1, x2, y2


def _clipped_food_mask_fill_ratio(
    compartment_bbox: Tuple[int, int, int, int], foods: List[Dict[str, Any]], shape: Tuple[int, int]
) -> float:
    if not foods:
        return 0.0
    x1, y1, x2, y2 = _clip_bbox_to_shape(compartment_bbox, shape)
    width = max(0, x2 - x1 + 1)
    height = max(0, y2 - y1 + 1)
    comp_area = width * height
    if comp_area == 0:
        return 0.0

    food_union = np.zeros(shape, dtype=bool)
    for food in foods:
        food_union = np.logical_or(food_union, food["mask"] > 0)
    food_area = int(np.sum(food_union[y1:y2 + 1, x1:x2 + 1]))
    return float(np.clip(food_area / comp_area, 0.0, 1.0))


def _compute_fill_ratios(estimator: QuantityEstimator, comp: Dict[str, Any]) -> Tuple[float, float, float, str]:
    mask_fill = estimator.calculate_fill_ratio(comp["mask"], comp["total_food_mask"])
    bbox = comp.get("bbox")
    if bbox is None:
        return mask_fill, mask_fill, mask_fill, "mask_only"

    bbox_fill = _clipped_food_mask_fill_ratio(bbox, comp["foods"], comp["mask"].shape)
    if comp["assignment_method"] == "mask_ioa":
        final_fill = max(mask_fill, bbox_fill)
        method = "mask_plus_bbox_clip"
    elif comp["assignment_method"] in {"center_bbox", "centroid", "box_iou"}:
        final_fill = bbox_fill
        method = "bbox_clipped_food_mask"
    else:
        final_fill = mask_fill
        method = "mask_only"

    return final_fill, mask_fill, bbox_fill, method


def _target_compartment_count(raw_count: int) -> int:
    if raw_count in VALID_COMPARTMENT_COUNTS:
        return raw_count
    lower_valid_counts = [count for count in VALID_COMPARTMENT_COUNTS if count <= raw_count]
    if not lower_valid_counts:
        return raw_count
    return max(lower_valid_counts)


def _max_compartment_overlap(comp: Dict[str, Any], others: List[Dict[str, Any]]) -> float:
    bbox = comp.get("bbox")
    if bbox is None:
        return 0.0
    overlaps = []
    for other in others:
        if other is comp:
            continue
        other_bbox = other.get("bbox")
        if other_bbox is None:
            continue
        overlaps.append(_bbox_iou(bbox, other_bbox))
    return max(overlaps, default=0.0)


def _cleanup_extra_compartments(mapped: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    raw_count = len(mapped)
    nonempty_count = sum(1 for comp in mapped if comp["foods"])
    if nonempty_count in VALID_COMPARTMENT_COUNTS and nonempty_count < raw_count:
        target_count = nonempty_count
    else:
        target_count = _target_compartment_count(raw_count)
    if target_count >= raw_count:
        for new_idx, comp in enumerate(mapped):
            comp["compartment_idx"] = new_idx
        return mapped, 0

    remove_count = raw_count - target_count
    scored = []
    for idx, comp in enumerate(mapped):
        area = int(np.sum(comp["mask"] > 0))
        overlap = _max_compartment_overlap(comp, mapped)
        confidence = float(comp.get("confidence", 0.0))
        empty_rank = 0 if not comp["foods"] else 1
        scored.append((empty_rank, -overlap, confidence, area, idx))

    remove_indices = {idx for *_, idx in sorted(scored)[:remove_count]}
    cleaned = [comp for idx, comp in enumerate(mapped) if idx not in remove_indices]
    for new_idx, comp in enumerate(cleaned):
        comp["compartment_idx"] = new_idx
    return cleaned, len(remove_indices)


def _extract_detections_from_yolo_result(result: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        detection = {
            "id": idx,
            "class_name": class_name,
            "confidence": float(conf),
            "mask": mask,
            "bbox": tuple(int(round(v)) for v in box.tolist()),
            "centroid": _mask_centroid(mask),
        }
        if class_name == "compartment":
            compartment_detections.append(detection)
        else:
            food_detections.append(detection)

    return compartment_detections, food_detections


def _init_compartment_mapping(compartment_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapped = []
    for idx, compartment in enumerate(compartment_detections):
        mapped.append(
            {
                "compartment_idx": idx,
                "mask": compartment["mask"],
                "bbox": compartment["bbox"],
                "confidence": compartment.get("confidence", 0.0),
                "foods": [],
                "total_food_mask": np.zeros_like(compartment["mask"], dtype=bool),
                "assignment_method": "none",
            }
        )
    return mapped


def _assign_food(mapped: List[Dict[str, Any]], compartment_idx: int, food: Dict[str, Any], method: str) -> None:
    mapped[compartment_idx]["foods"].append(food)
    mapped[compartment_idx]["total_food_mask"] = np.logical_or(
        mapped[compartment_idx]["total_food_mask"], food["mask"] > 0
    )
    if mapped[compartment_idx]["assignment_method"] == "none":
        mapped[compartment_idx]["assignment_method"] = method


def _map_food_mask_ioa(
    estimator: QuantityEstimator, food_detections: List[Dict[str, Any]], compartment_detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return estimator.map_food_to_compartments(food_detections, [comp["mask"] for comp in compartment_detections])


def _map_food_centroid(
    food_detections: List[Dict[str, Any]], compartment_detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    mapped = _init_compartment_mapping(compartment_detections)
    for food in food_detections:
        centroid = food.get("centroid")
        if centroid is None:
            continue
        cx, cy = centroid
        candidates = []
        for idx, comp in enumerate(compartment_detections):
            comp_mask = comp["mask"]
            if 0 <= cy < comp_mask.shape[0] and 0 <= cx < comp_mask.shape[1] and comp_mask[cy, cx]:
                candidates.append((idx, int(np.sum(comp_mask))))
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda item: item[1])[0]
        _assign_food(mapped, best_idx, food, "centroid")
    return mapped


def _point_in_bbox(point: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _map_food_center_bbox(
    food_detections: List[Dict[str, Any]], compartment_detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    mapped = _init_compartment_mapping(compartment_detections)
    for food in food_detections:
        centroid = food.get("centroid")
        if centroid is None:
            bbox = food.get("bbox")
            if bbox is None:
                continue
            centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

        candidates = []
        for idx, comp in enumerate(compartment_detections):
            comp_bbox = comp.get("bbox")
            if comp_bbox is None:
                continue
            if _point_in_bbox(centroid, comp_bbox):
                area = max(1, (comp_bbox[2] - comp_bbox[0] + 1) * (comp_bbox[3] - comp_bbox[1] + 1))
                candidates.append((idx, area))
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda item: item[1])[0]
        _assign_food(mapped, best_idx, food, "center_bbox")
    return mapped


def _map_food_box_iou(
    food_detections: List[Dict[str, Any]], compartment_detections: List[Dict[str, Any]], min_iou: float = 0.01
) -> List[Dict[str, Any]]:
    mapped = _init_compartment_mapping(compartment_detections)
    for food in food_detections:
        food_box = food.get("bbox") or _mask_bbox(food["mask"] > 0)
        if food_box is None:
            continue
        best_idx = -1
        best_iou = 0.0
        for idx, comp in enumerate(compartment_detections):
            comp_box = comp.get("bbox")
            if comp_box is None:
                continue
            iou = _bbox_iou(food_box, comp_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= min_iou:
            _assign_food(mapped, best_idx, food, "box_iou")
    return mapped


def _map_food_hybrid(
    estimator: QuantityEstimator, food_detections: List[Dict[str, Any]], compartment_detections: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    mapped = _init_compartment_mapping(compartment_detections)
    diagnostics = {"mask_ioa": 0, "center_bbox": 0, "centroid": 0, "box_iou": 0, "unassigned": 0}

    mask_mapped = _map_food_mask_ioa(estimator, food_detections, compartment_detections)
    assigned_ids = set()
    for comp_idx, comp in enumerate(mask_mapped):
        for food in comp["foods"]:
            assigned_ids.add(food["id"])
            diagnostics["mask_ioa"] += 1
            _assign_food(mapped, comp_idx, food, "mask_ioa")

    remaining = [food for food in food_detections if food["id"] not in assigned_ids]
    center_bbox_mapped = _map_food_center_bbox(remaining, compartment_detections)
    center_bbox_ids = set()
    for comp_idx, comp in enumerate(center_bbox_mapped):
        for food in comp["foods"]:
            center_bbox_ids.add(food["id"])
            diagnostics["center_bbox"] += 1
            _assign_food(mapped, comp_idx, food, "center_bbox")

    remaining = [food for food in remaining if food["id"] not in center_bbox_ids]
    centroid_mapped = _map_food_centroid(remaining, compartment_detections)
    centroid_ids = set()
    for comp_idx, comp in enumerate(centroid_mapped):
        for food in comp["foods"]:
            centroid_ids.add(food["id"])
            diagnostics["centroid"] += 1
            _assign_food(mapped, comp_idx, food, "centroid")

    remaining = [food for food in remaining if food["id"] not in centroid_ids]
    box_mapped = _map_food_box_iou(remaining, compartment_detections)
    box_ids = set()
    for comp_idx, comp in enumerate(box_mapped):
        for food in comp["foods"]:
            box_ids.add(food["id"])
            diagnostics["box_iou"] += 1
            _assign_food(mapped, comp_idx, food, "box_iou")

    diagnostics["unassigned"] = len(
        [food for food in food_detections if food["id"] not in assigned_ids | center_bbox_ids | centroid_ids | box_ids]
    )
    return mapped, diagnostics


def run_quantity_stage(result: Any, estimator: QuantityEstimator) -> QuantityStageResponse:
    compartment_detections, food_detections = _extract_detections_from_yolo_result(result)
    raw_compartment_count = len(compartment_detections)
    mapped, diagnostics = _map_food_hybrid(estimator, food_detections, compartment_detections)
    mapped, removed_count = _cleanup_extra_compartments(mapped)
    for comp in mapped:
        fill_ratio, mask_fill_ratio, bbox_fill_ratio, fill_ratio_method = _compute_fill_ratios(estimator, comp)
        comp["fill_ratio"] = fill_ratio
        comp["mask_fill_ratio"] = mask_fill_ratio
        comp["bbox_fill_ratio"] = bbox_fill_ratio
        comp["fill_ratio_method"] = fill_ratio_method
    fails, needs_vlm = estimator.quantity_pre_filter(mapped)

    mapped_summary: List[CompartmentSummary] = []
    for comp in mapped:
        bbox = comp.get("bbox") or _mask_bbox(comp["mask"])
        if bbox is None:
            bbox = (0, 0, 0, 0)
        fill_ratio = comp["fill_ratio"]
        quantity_result = estimator.classify_quantity(comp["foods"], fill_ratio)
        mapped_summary.append(
            CompartmentSummary(
                compartment_idx=comp["compartment_idx"],
                fill_ratio=fill_ratio,
                mask_fill_ratio=comp["mask_fill_ratio"],
                bbox_fill_ratio=comp["bbox_fill_ratio"],
                fill_ratio_method=comp["fill_ratio_method"],
                mask_polygon=_mask_to_polygon(comp["mask"]),
                foods=estimator.summarize_food_names(comp["foods"]),
                food_detections=[
                    DetectedFood(
                        detection_id=food["id"],
                        class_name=food["class_name"],
                        confidence=float(food["confidence"]),
                        bbox=DetectionBox(x1=food["bbox"][0], y1=food["bbox"][1], x2=food["bbox"][2], y2=food["bbox"][3]),
                        mask_polygon=_mask_to_polygon(food["mask"]),
                    )
                    for food in comp["foods"]
                ],
                assignment_method=comp["assignment_method"],
                quantity_status=quantity_result["status"],
                quantity_label=quantity_result["label"],
                primary_food=quantity_result["primary_food"],
                thresholds=quantity_result["thresholds"],
                bbox=DetectionBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
            )
        )

    return QuantityStageResponse(
        raw_compartments_detected=raw_compartment_count,
        compartments_detected=len(mapped),
        compartments_removed_by_cleanup=removed_count,
        foods_detected=len(food_detections),
        mapping_strategy="hybrid",
        mapping_diagnostics=diagnostics,
        mapped_summary=mapped_summary,
        auto_fails=fails,
        needs_quality_stage=needs_vlm,
    )
