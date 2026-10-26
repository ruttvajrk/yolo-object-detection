"""Reproduce the fine-tuning run with the exact settings saved in runs/detect/runs/finetune/args.yaml.

    python train.py                       # needs a GPU; about 9 minutes on a Colab T4
    python train.py --data my_data.yaml   # same recipe on your own dataset

Seed 42 and deterministic=True are set, so a rerun on the same GPU type should land very close
to the reported numbers (GPU kernels can still add small differences).
"""

import argparse

import torch
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="african-wildlife.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: no GPU found - training will be very slow.")
    YOLO("yolov8n.pt").train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                             batch=args.batch, device=device, seed=42, deterministic=True,
                             project="runs", name="finetune", exist_ok=True)


if __name__ == "__main__":
    main()
