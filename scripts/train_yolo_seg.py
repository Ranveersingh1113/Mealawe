import argparse
from pathlib import Path

import albumentations as A
from ultralytics import YOLO


PROFILES = {
    "starter": {"model": "yolo26s-seg.pt", "imgsz": 832, "epochs": 120, "batch": 12},
    "balanced": {"model": "yolo26m-seg.pt", "imgsz": 960, "epochs": 140, "batch": 8},
    "max": {"model": "yolo26l-seg.pt", "imgsz": 1024, "epochs": 160, "batch": 6},
    "baseline": {"model": "yolo11m-seg.pt", "imgsz": 832, "epochs": 120, "batch": 12},
}


def build_image_only_augmentations() -> list:
    """
    Use YOLO-native transforms for horizontal flip and rotation.
    Use Albumentations for image-only effects that do not change labels:
    brightness, exposure-like gamma shifts, and blur.
    """
    return [
        A.RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),
            contrast_limit=(0.0, 0.0),
            p=0.5,
        ),
        A.RandomGamma(
            gamma_limit=(85, 115),
            p=0.4,
        ),
        A.Blur(
            blur_limit=(3, 5),
            p=0.2,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation for Mealawe.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="starter")
    parser.add_argument("--model", default=None, help="Override model checkpoint")
    parser.add_argument("--imgsz", type=int, default=None, help="Override input image size")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/mealawe")
    parser.add_argument("--name", default="seg")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]

    model_name = args.model or profile["model"]
    imgsz = args.imgsz or profile["imgsz"]
    epochs = args.epochs or profile["epochs"]
    batch = args.batch or profile["batch"]

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    print("=== TRAINING CONFIG ===")
    print(f"Profile: {args.profile}")
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Image Size: {imgsz}")
    print(f"Epochs: {epochs}")
    print(f"Batch: {batch}")
    print(f"Device: {args.device}")
    print("=======================")

    model = YOLO(model_name)
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cos_lr=True,
        amp=True,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.0,
        degrees=15.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.15,
        augmentations=build_image_only_augmentations(),
        overlap_mask=True,
        mask_ratio=4,
        val=True,
        plots=True,
    )

    print("\nRunning validation on best checkpoint...")
    best_ckpt = Path(args.project) / args.name / "weights" / "best.pt"
    best_model = YOLO(str(best_ckpt))
    metrics = best_model.val(data=str(data_path), device=args.device, plots=True)

    print("\n=== VALIDATION SUMMARY ===")
    print(metrics)
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
