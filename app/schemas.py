from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectedFood(BaseModel):
    detection_id: int
    class_name: str
    confidence: float
    bbox: DetectionBox
    mask_polygon: List[List[int]]


class CompartmentSummary(BaseModel):
    compartment_idx: int
    fill_ratio: float
    mask_fill_ratio: float
    bbox_fill_ratio: float
    fill_ratio_method: str
    mask_polygon: List[List[int]]
    foods: List[str]
    food_detections: List[DetectedFood]
    assignment_method: str
    quantity_status: str
    quantity_label: str
    primary_food: Optional[str] = None
    thresholds: Optional[Dict[str, Optional[float]]] = None
    bbox: DetectionBox


class QuantityIssue(BaseModel):
    compartment_idx: int
    issue: str
    reason: str
    fill_ratio: Optional[float] = None
    foods: Optional[List[str]] = None
    quantity_status: Optional[str] = None
    thresholds: Optional[Dict[str, Optional[float]]] = None


class QuantityStageResponse(BaseModel):
    raw_compartments_detected: int
    compartments_detected: int
    compartments_removed_by_cleanup: int
    foods_detected: int
    mapping_strategy: str
    mapping_diagnostics: Dict[str, int]
    mapped_summary: List[CompartmentSummary]
    auto_fails: List[QuantityIssue]
    needs_quality_stage: List[Dict[str, Any]]


class QualityStageResponse(BaseModel):
    status: str
    message: str
    compartment_candidates: List[Dict[str, Any]]


class InspectionResponse(BaseModel):
    overall_status: str
    quantity: QuantityStageResponse
    quality: Optional[QualityStageResponse]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_exists: bool
    model_loaded: bool
    mapping_strategy: str


class BatchInspectionResponse(BaseModel):
    results: List[InspectionResponse] = Field(default_factory=list)
