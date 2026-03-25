import numpy as np
from typing import List, Dict, Tuple, Any, Optional


class QuantityEstimator:
    PRESENCE_ONLY_ITEMS = {"roti", "salad", "sweet"}
    DEFAULT_BULK_ITEMS = {"dal", "curry", "rice", "sabzi", "raita", "dahi"}

    def __init__(self, thali_spec: Dict[str, Any]):
        """
        thali_spec describes expected minimal criteria.
        e.g., {'quantity_reject_below': 0.5, 'quantity_pass_from': 0.6}
        """
        self.thali_spec = thali_spec

    def summarize_food_names(self, foods: List[Dict[str, Any]]) -> List[str]:
        """Collapse repeated presence-only detections while keeping bulk items visible."""
        counts: Dict[str, int] = {}
        ordered_names: List[str] = []
        for food in foods:
            name = food["class_name"]
            if name not in counts:
                ordered_names.append(name)
                counts[name] = 0
            counts[name] += 1

        summary: List[str] = []
        for name in ordered_names:
            if name in self.PRESENCE_ONLY_ITEMS:
                summary.append(name)
            else:
                count = counts[name]
                summary.append(f"{name} x{count}" if count > 1 else name)
        return summary

    def calculate_fill_ratio(self, compartment_mask: np.ndarray, food_mask: np.ndarray) -> float:
        """Calculate ratio of food pixels to compartment pixels."""
        comp_area = np.sum(compartment_mask > 0)
        food_inside_compartment = np.logical_and(compartment_mask > 0, food_mask > 0)
        food_area = np.sum(food_inside_compartment)
        if comp_area == 0:
            return 0.0
        return float(np.clip(food_area / comp_area, 0.0, 1.0))

    def get_food_quantity_rules(self) -> Dict[str, Dict[str, float]]:
        return self.thali_spec.get("food_quantity_rules", {})

    def get_quantity_reject_below(self) -> float:
        return float(self.thali_spec.get("quantity_reject_below", 0.5))

    def get_quantity_pass_from(self) -> float:
        return float(self.thali_spec.get("quantity_pass_from", 0.6))

    def get_bulk_foods(self) -> set:
        configured = self.thali_spec.get("bulk_foods")
        if configured:
            return set(configured)
        return set(self.DEFAULT_BULK_ITEMS)

    def get_primary_food(self, foods: List[Dict[str, Any]]) -> Optional[str]:
        if not foods:
            return None

        names = [food["class_name"] for food in foods]
        bulk_foods = self.get_bulk_foods()

        bulk_present = [name for name in names if name in bulk_foods]
        if bulk_present:
            counts: Dict[str, int] = {}
            for name in bulk_present:
                counts[name] = counts.get(name, 0) + 1
            return max(counts.items(), key=lambda item: item[1])[0]

        counts = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        return max(counts.items(), key=lambda item: item[1])[0]

    def classify_quantity(self, foods: List[Dict[str, Any]], fill_ratio: float) -> Dict[str, Any]:
        primary_food = self.get_primary_food(foods)
        if primary_food is None:
            return {
                "primary_food": None,
                "status": "EMPTY",
                "label": "Empty",
                "thresholds": None,
            }

        if primary_food in self.PRESENCE_ONLY_ITEMS:
            return {
                "primary_food": primary_food,
                "status": "PRESENCE_ONLY",
                "label": "Presence only",
                "thresholds": None,
            }

        rules = self.get_food_quantity_rules().get(primary_food)
        if not rules:
            rules = {
                "low": self.get_quantity_reject_below(),
                "target_min": self.get_quantity_pass_from(),
            }

        low = rules.get("low")
        target_min = rules.get("target_min", self.get_quantity_pass_from())
        if low is not None and fill_ratio < low:
            status = "LOW"
            label = "Low quantity"
        elif target_min is not None and fill_ratio < target_min:
            status = "BORDERLINE"
            label = "Borderline quantity"
        else:
            status = "OK"
            label = "Sufficient quantity"

        return {
            "primary_food": primary_food,
            "status": status,
            "label": label,
            "thresholds": {
                "low": low,
                "target_min": target_min,
            },
        }

    def map_food_to_compartments(
        self, 
        food_detections: List[Dict[str, Any]], 
        compartment_masks: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Assign each detected food item to a compartment based on 
        mask intersection over area (IoA).
        """
        mapped_compartments = []
        for c_idx, c_mask in enumerate(compartment_masks):
            comp_data = {
                'compartment_idx': c_idx,
                'mask': c_mask,
                'foods': [],
                'total_food_mask': np.zeros_like(c_mask, dtype=bool)
            }
            mapped_compartments.append(comp_data)

        # Match each food to the compartment with highest IoA
        for food in food_detections:
            f_mask = food['mask'] > 0
            best_iou = 0.0
            best_c_idx = -1
            
            f_area = np.sum(f_mask)
            if f_area == 0:
                continue
                
            for c_idx, c_mask in enumerate(compartment_masks):
                # Calculate Intersection over Area (of the food)
                intersection = np.logical_and(f_mask, c_mask > 0)
                inter_area = np.sum(intersection)
                ioa = inter_area / f_area
                
                if ioa > best_iou:
                    best_iou = ioa
                    best_c_idx = c_idx
            
            # If food item mostly falls inside a compartment (>30%)
            if best_c_idx >= 0 and best_iou > 0.3:
                mapped_compartments[best_c_idx]['foods'].append(food)
                # Accumulate food masks
                mapped_compartments[best_c_idx]['total_food_mask'] = np.logical_or(
                    mapped_compartments[best_c_idx]['total_food_mask'], 
                    f_mask
                )
                
        return mapped_compartments

    def quantity_pre_filter(self, mapped_compartments: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply deterministic rules:
        - Any compartment empty -> fail
        - Bulk/Spread foods fill_ratio < threshold -> fail
        - Roti/Salad/Sweet are checked only for PRESENCE, not quantity.
        """
        auto_fails = []
        needs_vlm = []
        
        detected_items = set()
        
        for comp in mapped_compartments:
            foods = comp['foods']
            c_mask = comp['mask']
            
            for f in foods:
                detected_items.add(f['class_name'])
            
            # Focus on quantity of this compartment
            if not foods:
                auto_fails.append({
                    'compartment_idx': comp['compartment_idx'],
                    'issue': 'EMPTY_COMPARTMENT',
                    'reason': 'No food detected in compartment.'
                })
                continue
                
            fill_ratio = comp.get('fill_ratio')
            if fill_ratio is None:
                fill_ratio = self.calculate_fill_ratio(c_mask, comp['total_food_mask'])
            
            quantity_result = self.classify_quantity(foods, fill_ratio)
            primary_food = quantity_result["primary_food"]
            status = quantity_result["status"]

            if status == "LOW":
                auto_fails.append({
                    'compartment_idx': comp['compartment_idx'],
                    'issue': 'UNDERFILLED',
                    'reason': f'{primary_food} fill ratio {fill_ratio:.2f} is below the reject threshold.',
                    'fill_ratio': fill_ratio,
                    'foods': self.summarize_food_names(foods),
                    'quantity_status': status,
                    'thresholds': quantity_result["thresholds"],
                })
            else:
                needs_vlm.append({
                    'compartment_idx': comp['compartment_idx'],
                    'fill_ratio': fill_ratio,
                    'foods': self.summarize_food_names(foods),
                    'quantity_status': status,
                    'thresholds': quantity_result["thresholds"],
                })

        # Check expected presence for items where we don't care about quantity
        expected_presence = self.thali_spec.get('expected_presence_items', [])
        for item in expected_presence:
            if item not in detected_items:
                auto_fails.append({
                    'compartment_idx': -1, # Global failure
                    'issue': f'MISSING_{item.upper()}',
                    'reason': f'Thali requires {item} but it was not detected.'
                })
            
        return auto_fails, needs_vlm
