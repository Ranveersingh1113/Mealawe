# Updated Dataset Training

This repo now includes the prepared YOLO segmentation dataset at:

```text
datasets/mealawe_yolo_seg_v2
```

It was built from:

```text
Mealawe 2.coco-segmentation (1)/train/_annotations.coco.json
```

Current split:

- `train`: 832 images
- `val`: 104 images
- `test`: 105 images

## On the other device

Clone the repo and fetch the dataset files tracked with Git LFS:

```powershell
git clone https://github.com/Ranveersingh1113/Mealawe.git
cd Mealawe
git lfs install
git lfs pull
```

Set up Python:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train YOLO26m

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_seg.py --data datasets\mealawe_yolo_seg_v2\data.yaml --profile balanced --project runs/segment/runs/mealawe --name seg_v2_yolo26m --device 0
```

## Train YOLO26l

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_seg.py --data datasets\mealawe_yolo_seg_v2\data.yaml --profile max --project runs/segment/runs/mealawe --name seg_v2_yolo26l --device 0
```

## Notes

- `balanced` uses `yolo26m-seg.pt`, `imgsz=960`, `epochs=140`, `batch=8`
- `max` uses `yolo26l-seg.pt`, `imgsz=1024`, `epochs=160`, `batch=6`
- If the new device runs out of GPU memory, lower `--batch` first
- If the base checkpoint is not already present locally, Ultralytics should download it on first run
