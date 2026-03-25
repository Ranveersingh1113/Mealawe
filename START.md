# Start Project

## 1. Open the project root

Use PowerShell in:

```powershell
D:\OneDrive\Desktop\mealawe
```

## 2. Install dependencies

If `.venv` already exists, you can skip this.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Start the app

```powershell
$env:YOLO_MODEL_PATH="runs/segment/runs/mealawe/seg_s2/weights/best.pt"
$env:THALI_SPEC_PATH="config/thali_spec.example.json"
$env:YOLO_CONFIG_DIR="$PWD\.yolo_config"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. Open the demo

In the browser:

```text
http://127.0.0.1:8000/
```

## 5. If the page acts weird

Hard refresh:

```text
Ctrl + F5
```

## 6. If the model is not found

Check this path exists:

```powershell
Test-Path "runs\segment\runs\mealawe\seg_s2\weights\best.pt"
```

## 7. If you want to train again later

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_seg.py --data datasets\mealawe_yolo_seg\data.yaml --profile balanced --project runs/mealawe --name seg_m --device 0 --batch 2 --workers 0
```

## 8. If you want to rebuild the dataset later

```powershell
.\.venv\Scripts\python.exe scripts\prepare_coco_for_yolo_seg.py --coco-json "Mealawe 2.coco-segmentation/train/_annotations.coco.json" --images-dir "Mealawe 2.coco-segmentation/train" --out-dir "datasets/mealawe_yolo_seg"
```
