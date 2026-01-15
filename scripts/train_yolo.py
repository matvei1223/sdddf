#!/usr/bin/env python3
"""Train a YOLOv8 model for your custom classes.

Usage:
  python scripts/train_yolo.py --data data/rust_players.yaml --epochs 100 --img 960
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--img", type=int, default=960, help="Image size")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model to fine-tune (e.g. yolov8n.pt)",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.img)


if __name__ == "__main__":
    main()
