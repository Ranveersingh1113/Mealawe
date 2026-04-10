# Mealawe Thali Inspection

A `YOLO + quantity gate + VLM quality gate + demo UI` pipeline for Mealawe's thali inspection flow.

## What it does

1. A chef uploads a thali image via the API or the browser demo.
2. YOLO segmentation detects compartments and food items.
3. A deterministic quantity gate filters obvious under-filled compartments.
4. Thalis that pass quantity are evaluated by a VLM quality gate (Google Gemini) for final approval.

## Pipeline architecture

```text
Uploaded image
-> YOLO segmentation
-> food-to-compartment mapping
-> extra-compartment cleanup
-> fill-ratio estimation
-> quantity exception filter
   -> REJECTED_AT_QUANTITY_STAGE   (stops here on failure)
-> VLM quality gate (Gemini)
   -> REJECTED_AT_QUALITY_STAGE    (poor quality)
   -> PASSED_WITH_WARNINGS         (acceptable quality)
   -> PASSED_ALL_STAGES            (good quality)
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Browser demo page |
| `GET`  | `/health` | Health check — model load status |
| `POST` | `/inspect` | Inspect a single thali image |
| `POST` | `/inspect/batch` | Inspect up to 8 thali images at once |

All endpoints return JSON. See [`app/schemas.py`](app/schemas.py) for the full response models.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | `runs/segment/runs/mealawe/seg_s2/weights/best.pt` | Path to trained YOLO weights |
| `THALI_SPEC_PATH` | `config/thali_spec.example.json` | Path to quantity config |
| `YOLO_CONF` | `0.25` | YOLO confidence threshold |
| `YOLO_IOU` | `0.60` | YOLO IoU threshold |
| `MEALAWE_MAX_BATCH_FILES` | `8` | Maximum images per batch request |
| `GEMINI_API_KEY` | *(required for quality stage)* | Google Gemini API key |

The `GEMINI_API_KEY` can also be placed in a `.env` file inside the `quality-verification/` directory.

## Quantity gate rules

- **Bulk foods** — `dal`, `curry`, `rice`, `sabzi`, `raita`, `dahi`:
  - `fill_ratio < 0.5` → reject
  - `0.5 <= fill_ratio < 0.6` → borderline, pass onward
  - `fill_ratio >= 0.6` → sufficient
- **Presence-only foods** — `roti`, `salad`, `sweet`: checked for presence only
- Extra empty compartments are cleaned up before the final decision when possible

## Main files

**API & pipeline**
- [`app/main.py`](app/main.py) — FastAPI app, endpoint routing
- [`app/pipeline.py`](app/pipeline.py) — YOLO parsing, mapping, cleanup, fill-ratio logic
- [`app/quality_client.py`](app/quality_client.py) — VLM quality gate (Gemini API integration)
- [`app/schemas.py`](app/schemas.py) — Pydantic response models
- [`app/static/index.html`](app/static/index.html) — browser demo UI
- [`quantity_estimator.py`](quantity_estimator.py) — quantity rules
- [`config/thali_spec.example.json`](config/thali_spec.example.json) — quantity config

**Quality verification pipeline**
- [`quality-verification/`](quality-verification/) — standalone VLM evaluation scripts
- [`quality-verification/evaluate.py`](quality-verification/evaluate.py) — batch VLM evaluation script
- [`quality-verification/prompt_template.py`](quality-verification/prompt_template.py) — Gemini system prompt

**Training & data**
- [`scripts/prepare_coco_for_yolo_seg.py`](scripts/prepare_coco_for_yolo_seg.py) — COCO → YOLO dataset conversion
- [`scripts/train_yolo_seg.py`](scripts/train_yolo_seg.py) — YOLO training entrypoint
- [`scripts/evaluate_quantity_pipeline.py`](scripts/evaluate_quantity_pipeline.py) — offline quantity evaluation

## Demo

The app serves a browser demo at `/`.

It shows:
- uploaded image
- predicted compartment masks and bounding boxes
- predicted food masks with labels
- quantity status per compartment
- final overall inspection status

## Status

What is working:
- dataset prep and YOLO training flow
- FastAPI inference (single and batch)
- quantity exception filter
- VLM quality gate via Google Gemini (primary: `gemini-3.1-pro-preview`, fallback: `gemini-2.5-flash`)
- frontend demo

What is still weak:
- `dal` vs `curry` vs `raita` confusion on real images
- compartment detection can still add false extras in some cases

## Quick start

See [START.md](START.md) for full setup and run instructions.
For a detailed progress summary, see [PROJECT_STATUS.md](PROJECT_STATUS.md).
