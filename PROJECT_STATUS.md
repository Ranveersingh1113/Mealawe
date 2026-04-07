# Project Status

## Scope completed so far

We built a working prototype for the first half of the Mealawe inspection pipeline:

1. COCO segmentation export -> YOLO segmentation dataset
2. YOLO segmentation training flow
3. API for single-image and batch inspection
4. Deterministic quantity gate
5. Browser demo to visualize detections and quantity decisions

## Final working architecture today

```text
Uploaded image
-> YOLO segmentation
-> food-to-compartment mapping
-> extra-compartment cleanup
-> fill-ratio estimation
-> quantity exception filter
-> pass onward to future VLM gate
```

## Key implementation decisions frozen today

### Quantity gate

- Use the quantity stage as an `exception filter`, not a perfect judge.
- Bulk foods:
  - reject below `0.5`
  - borderline from `0.5` to `0.6`
  - sufficient from `0.6`
- Presence-only foods:
  - `roti`
  - `salad`
  - `sweet`
- Do not globally require `salad` or `sweet` anymore.

### Mapping

- Prefer `center_bbox` mapping for inference-time food-to-compartment assignment.
- Fall back to other geometric methods only when needed.

### Fill ratio

- Do not trust raw predicted compartment masks alone.
- Use clipped food-mask area inside the compartment bounding box for a more stable fill estimate.

### Extra compartments

- If extra empty compartments appear, remove them before final quantity decisions when the remaining count matches a valid tray size.

## What is working well

- Upload -> inspect flow
- Compartment and food overlays in the frontend
- Quantity gate is far more stable than earlier iterations
- Global false rejects for missing `salad`/`sweet` are removed

## Current known issues

### 1. Dal detection is weak on real images

The model still confuses:
- `dal`
- `curry`
- `raita`

This is likely a mix of:
- dataset labeling inconsistency
- insufficient real-world variation
- visual similarity from top view

### 2. Compartment detection is not perfect

False extra compartments can still happen in some images, though cleanup now removes some of them.

### 3. Quantity gate is intentionally simple

The current rules are for filtering obvious cases only.
The future VLM gate should make the final human-like quantity and quality judgment.

## Recommended next step

Build the VLM handoff after quantity pass.

Recommended VLM input:
- full thali image
- cropped compartment images
- predicted food label per compartment
- fill ratio
- quantity gate status

Recommended VLM output:
- per-compartment quality verdict
- per-compartment human-like quantity verdict
- confidence
- short reason
- overall approve / reject / human review

## Files that matter now

- [README.md](README.md)
- [START.md](START.md)
- [app/main.py](app/main.py)
- [app/pipeline.py](app/pipeline.py)
- [app/quality_client.py](app/quality_client.py)
- [app/schemas.py](app/schemas.py)
- [app/static/index.html](app/static/index.html)
- [quantity_estimator.py](quantity_estimator.py)
- [config/thali_spec.example.json](config/thali_spec.example.json)
- [quality-verification/](quality-verification/)

## Notes on cleanup done today

Removed:
- old exploratory analysis scripts
- old temporary logs
- old context files that were replaced by this summary
- stale cache folders where possible

Kept:
- trained model artifacts
- datasets
- training/evaluation scripts that are still useful
