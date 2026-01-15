#!/usr/bin/env python3
"""Train a YOLOv8 model for your custom classes.

Usage:
  python scripts/train_yolo.py --data data/rust_players.yaml --epochs 100 --img 960
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument(
        "--dataset-dir",
        help="Dataset root dir (used to auto-generate YAML if --data does not exist)",
    )
    parser.add_argument(
        "--names",
        help="Comma-separated class names to auto-generate YAML (e.g. player,enemy)",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--img", type=int, default=960, help="Image size")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model to fine-tune (e.g. yolov8n.pt)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        if not args.dataset_dir or not args.names:
            raise SystemExit(
                "YAML not found. Provide --dataset-dir and --names to auto-generate it."
            )
        dataset_dir = Path(args.dataset_dir)
        names = [name.strip() for name in args.names.split(",") if name.strip()]
        if not names:
            raise SystemExit("No class names provided in --names.")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_lines = [
            f"path: {dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
        data_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    model = YOLO(args.model)
    model.train(data=str(data_path), epochs=args.epochs, imgsz=args.img)


if __name__ == "__main__":
    main()
