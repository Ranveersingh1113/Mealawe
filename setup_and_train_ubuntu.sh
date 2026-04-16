#!/bin/bash
# Setup and Train YOLO Segmentation for Ubuntu Workstation (RTX A4000)
# Usage: 
#   ./setup_and_train_ubuntu.sh          # Train from scratch
#   ./setup_and_train_ubuntu.sh resume   # Resume from the last checkpoint

set -e

echo "=== System Update and Dependencies ==="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0

echo "=== Setting up Python Virtual Environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "=== Installing PyTorch with CUDA 12.1 for RTX A4000 ==="
# We install PyTorch first to ensure the correct CUDA version is pulled for the A4000.
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing Other Requirements ==="
pip install -r requirements.txt

# The RTX A4000 has 16GB of VRAM. You can afford a decent batch size (e.g., 8-16) depending on the model size.
# We will use the 'balanced' profile (yolo26m). If out of memory, try '--batch 4'.

if [ "$1" == "resume" ]; then
    echo "=== Resuming Training ==="
    # Provide the path to the model checkpoint you wish to resume from.
    # Usually it's runs/mealawe/seg_m/weights/last.pt
    CHECKPOINT_PATH="runs/mealawe/seg_m/weights/last.pt"
    
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "Error: Checkpoint $CHECKPOINT_PATH not found! Ensure you have copied the runs folder from your laptop if you want to resume."
        exit 1
    fi
    
    python scripts/train_yolo_seg.py \
        --data datasets/mealawe_yolo_seg/data.yaml \
        --profile balanced \
        --project runs/mealawe \
        --name seg_m \
        --device 0 \
        --batch 8 \
        --workers 8 \
        --model "$CHECKPOINT_PATH" \
        --resume
else
    echo "=== Starting Training from Scratch ==="
    python scripts/train_yolo_seg.py \
        --data datasets/mealawe_yolo_seg/data.yaml \
        --profile balanced \
        --project runs/mealawe \
        --name seg_m \
        --device 0 \
        --batch 8 \
        --workers 8
fi
