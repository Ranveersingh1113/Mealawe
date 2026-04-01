import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

from app.pipeline import run_quantity_stage
from app.quality_client import run_quality_stage
from app.schemas import BatchInspectionResponse, HealthResponse, InspectionResponse
from quantity_estimator import QuantityEstimator


def _load_thali_spec() -> Dict[str, Any]:
    thali_spec_path = os.getenv("THALI_SPEC_PATH")
    if thali_spec_path and Path(thali_spec_path).exists():
        with open(thali_spec_path, "r", encoding="utf-8") as f:
            return json.load(f)
    default_path = Path("config/thali_spec.example.json")
    if default_path.exists():
        with default_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "quantity_reject_below": 0.5,
        "quantity_pass_from": 0.6,
        "expected_presence_items": [],
    }


def _load_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    return np.array(image)


def _run_single_inspection(image_bytes: bytes) -> InspectionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not available at YOLO_MODEL_PATH={MODEL_PATH}",
        )

    image_np = _load_image_bytes(image_bytes)
    results = model.predict(source=image_np, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
    if not results:
        raise HTTPException(status_code=500, detail="Model inference returned no result.")

    quantity_result = run_quantity_stage(results[0], estimator)
    if quantity_result.auto_fails:
        return InspectionResponse(
            overall_status="REJECTED_AT_QUANTITY_STAGE",
            quantity=quantity_result,
            quality=None,
        )

    quality_result = run_quality_stage(
        image_bytes=image_bytes,
        compartments=quantity_result.needs_quality_stage,
    )
    
    # Assess overall status combining CV and VLM
    if quality_result.status == "POOR":
        final_status = "REJECTED_AT_QUALITY_STAGE"
    elif quality_result.status == "ACCEPTABLE":
        final_status = "PASSED_WITH_WARNINGS"
    elif quality_result.status == "VLM_ERROR":
        final_status = "PASSED_QUANTITY_PENDING_QUALITY_RETRY" 
    else:
        final_status = "PASSED_ALL_STAGES"

    return InspectionResponse(
        overall_status=final_status,
        quantity=quantity_result,
        quality=quality_result,
    )


MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    "runs/segment/runs/mealawe/seg_s2/weights/best.pt",
)
CONF_THRES = float(os.getenv("YOLO_CONF", "0.25"))
IOU_THRES = float(os.getenv("YOLO_IOU", "0.60"))
MAX_BATCH_FILES = int(os.getenv("MEALAWE_MAX_BATCH_FILES", "8"))

model = None
if not Path(MODEL_PATH).exists():
    print(f"[WARN] YOLO model file not found at startup: {MODEL_PATH}")
else:
    model = YOLO(MODEL_PATH)
estimator = QuantityEstimator(_load_thali_spec())

app = FastAPI(title="Mealawe Inspection API", version="0.2.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_path=MODEL_PATH,
        model_exists=Path(MODEL_PATH).exists(),
        model_loaded=model is not None,
        mapping_strategy="hybrid",
    )


@app.post("/inspect", response_model=InspectionResponse)
async def inspect_thali(file: UploadFile = File(...)) -> InspectionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")
    image_bytes = await file.read()
    return _run_single_inspection(image_bytes)


@app.post("/inspect/batch", response_model=BatchInspectionResponse)
async def inspect_thali_batch(files: List[UploadFile] = File(...)) -> BatchInspectionResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed per batch is {MAX_BATCH_FILES}.",
        )

    results: List[InspectionResponse] = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Unsupported file type for {file.filename}.")
        image_bytes = await file.read()
        results.append(_run_single_inspection(image_bytes))
    return BatchInspectionResponse(results=results)
