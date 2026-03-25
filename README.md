# Mealawe Thali Inspection

This project is a working `YOLO + quantity gate + demo UI` prototype for Mealawe's thali inspection flow.

## What it does

1. A chef uploads a thali image.
2. YOLO segmentation detects compartments and food items.
3. A deterministic quantity gate filters obvious failures.
4. If the thali passes quantity, it is ready to be sent to a future VLM quality gate.

## Current quantity-gate rule

- `dal`, `curry`, `rice`, `sabzi`, `raita`, `dahi`
  - `fill_ratio < 0.5` -> reject
  - `0.5 <= fill_ratio < 0.6` -> borderline, pass onward
  - `fill_ratio >= 0.6` -> sufficient
- `roti`, `salad`, `sweet`
  - presence only
- extra empty compartments are cleaned up before the final decision when possible

## Main files

- [app/main.py](D:/OneDrive/Desktop/mealawe/app/main.py): FastAPI app and demo page routing
- [app/pipeline.py](D:/OneDrive/Desktop/mealawe/app/pipeline.py): YOLO parsing, mapping, cleanup, fill-ratio logic
- [app/static/index.html](D:/OneDrive/Desktop/mealawe/app/static/index.html): simple frontend demo
- [quantity_estimator.py](D:/OneDrive/Desktop/mealawe/quantity_estimator.py): quantity rules
- [config/thali_spec.example.json](D:/OneDrive/Desktop/mealawe/config/thali_spec.example.json): active quantity config
- [scripts/prepare_coco_for_yolo_seg.py](D:/OneDrive/Desktop/mealawe/scripts/prepare_coco_for_yolo_seg.py): dataset conversion
- [scripts/train_yolo_seg.py](D:/OneDrive/Desktop/mealawe/scripts/train_yolo_seg.py): training entrypoint
- [scripts/evaluate_quantity_pipeline.py](D:/OneDrive/Desktop/mealawe/scripts/evaluate_quantity_pipeline.py): offline evaluation helper

## Demo

The app serves a browser demo at `/`.

It shows:
- uploaded image
- predicted compartment masks
- predicted food masks
- compartment bounding boxes
- quantity status per compartment

## Status

What is working:
- dataset prep
- YOLO training flow
- FastAPI inference
- quantity exception filter
- frontend demo

What is still weak:
- `dal` vs `curry` vs `raita` confusion on real images
- compartment detection can still add false extras in some cases
- VLM quality gate is still a placeholder

## Next likely step

Connect the VLM gate after quantity pass:
- full thali image
- compartment crops
- predicted food labels
- fill ratios
- quantity status

For startup instructions, see [START.md](D:/OneDrive/Desktop/mealawe/START.md).
For a project progress summary, see [PROJECT_STATUS.md](D:/OneDrive/Desktop/mealawe/PROJECT_STATUS.md).
