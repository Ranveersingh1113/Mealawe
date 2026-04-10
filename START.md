# Start Project

## 1. Open the project root

Navigate to the cloned repository directory in your terminal.

## 2. Install dependencies

If `.venv` already exists, you can skip this step.

**Linux / macOS (bash):**
```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Set the Gemini API key

The VLM quality stage requires a Google Gemini API key. Create a `.env` file inside `quality-verification/`:

```bash
echo "GEMINI_API_KEY=your_gemini_api_key_here" > quality-verification/.env
```

## 4. Start the app

**Linux / macOS (bash):**
```bash
export YOLO_MODEL_PATH="runs/segment/runs/mealawe/seg_s2/weights/best.pt"
export THALI_SPEC_PATH="config/thali_spec.example.json"
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Windows (PowerShell):**
```powershell
$env:YOLO_MODEL_PATH="runs/segment/runs/mealawe/seg_s2/weights/best.pt"
$env:THALI_SPEC_PATH="config/thali_spec.example.json"
$env:YOLO_CONFIG_DIR="$PWD\.yolo_config"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 5. Open the demo

In the browser:

```text
http://127.0.0.1:8000/
```

## 6. If the page acts weird

Hard refresh: **Ctrl + F5** (or **Cmd + Shift + R** on macOS).

## 7. If the model is not found

Verify the weights file exists:

**Linux / macOS:**
```bash
ls runs/segment/runs/mealawe/seg_s2/weights/best.pt
```

**Windows (PowerShell):**
```powershell
Test-Path "runs\segment\runs\mealawe\seg_s2\weights\best.pt"
```

## 8. If you want to train again later

**Linux / macOS:**
```bash
.venv/bin/python scripts/train_yolo_seg.py \
  --data datasets/mealawe_yolo_seg/data.yaml \
  --profile balanced --project runs/mealawe --name seg_m \
  --device 0 --batch 2 --workers 0
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_seg.py --data datasets\mealawe_yolo_seg\data.yaml --profile balanced --project runs/mealawe --name seg_m --device 0 --batch 2 --workers 0
```

## 9. If you want to rebuild the dataset later

**Linux / macOS:**
```bash
.venv/bin/python scripts/prepare_coco_for_yolo_seg.py \
  --coco-json "Mealawe 2.coco-segmentation/train/_annotations.coco.json" \
  --images-dir "Mealawe 2.coco-segmentation/train" \
  --out-dir "datasets/mealawe_yolo_seg"
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\python.exe scripts\prepare_coco_for_yolo_seg.py --coco-json "Mealawe 2.coco-segmentation/train/_annotations.coco.json" --images-dir "Mealawe 2.coco-segmentation/train" --out-dir "datasets/mealawe_yolo_seg"
```
