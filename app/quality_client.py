from typing import Any, Dict, List

from app.schemas import QualityStageResponse


def run_quality_stage_placeholder(image_bytes: bytes, compartments: List[Dict[str, Any]]) -> QualityStageResponse:
    """
    Placeholder for teammate-owned quality stage integration.
    Replace this with an internal service call when available.
    """
    _ = image_bytes
    return QualityStageResponse(
        status="PENDING_EXTERNAL_QUALITY_CHECK",
        message="Quantity stage passed. Forwarded to quality stage.",
        compartment_candidates=compartments,
    )
