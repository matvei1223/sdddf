#!/usr/bin/env python3
"""Train a YOLOv8 model for your custom classes.

Usage:
  python scripts/train_yolo.py --data data/rust_players.yaml --epochs 100 --img 960
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def _resolve_dataset_paths(data_path: Path) -> tuple[Path | None, Path | None]:
    content = data_path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"path", "train", "val"}:
            values[key] = value.strip()

    if "train" not in values or "val" not in values:
        return None, None

    base_path = Path(values.get("path", "."))
    if not base_path.is_absolute():
        base_path = (data_path.parent / base_path).resolve()

    train_path = Path(values["train"])
    val_path = Path(values["val"])
    if not train_path.is_absolute():
        train_path = base_path / train_path
    if not val_path.is_absolute():
        val_path = base_path / val_path

    return train_path, val_path


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
        "--device",
        default="cpu",
        help="Training device (e.g. cpu, 0, 0,1). Defaults to 'cpu'.",
    )
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
        if not dataset_dir.exists():
            raise SystemExit(
                f"Dataset directory not found: {dataset_dir}. "
                "Pass the real path to your dataset folder."
            )
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

    train_dir, val_dir = _resolve_dataset_paths(data_path)
    if train_dir and val_dir:
        missing = [str(path) for path in (train_dir, val_dir) if not path.exists()]
        if missing:
            raise SystemExit(
                "Dataset paths not found: "
                + ", ".join(missing)
                + ". Ensure images/train and images/val exist and the YAML path is correct."
            )

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.img,
        device=args.device,
    )


if __name__ == "__main__":
    main()
