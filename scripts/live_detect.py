#!/usr/bin/env python3
"""Real-time screen detection with YOLOv8.

Example:
  python scripts/live_detect.py --model runs/detect/train/weights/best.pt --classes player
"""

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import mss
import numpy as np
from ultralytics import YOLO


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int


@dataclass
class DetectConfig:
    model_path: str
    conf: float
    iou: float
    classes: Optional[List[str]]
    region: Optional[Region]
    display_scale: float


def parse_region(value: Optional[str]) -> Optional[Region]:
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError("Region must be left,top,width,height")
    left, top, width, height = (int(p) for p in parts)
    return Region(left=left, top=top, width=width, height=height)


def build_config(args: argparse.Namespace) -> DetectConfig:
    classes = args.classes.split(",") if args.classes else None
    region = parse_region(args.region)
    return DetectConfig(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        classes=classes,
        region=region,
        display_scale=args.display_scale,
    )


def resolve_class_ids(model: YOLO, class_names: Optional[Iterable[str]]) -> Optional[List[int]]:
    if not class_names:
        return None
    name_map = {name.lower(): idx for idx, name in model.names.items()}
    class_ids = []
    for name in class_names:
        key = name.strip().lower()
        if key not in name_map:
            raise ValueError(f"Class '{name}' not found in model names: {model.names}")
        class_ids.append(name_map[key])
    return class_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Live screen detection with YOLOv8")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv8 model (.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument(
        "--classes",
        help="Comma-separated class names to keep (e.g. player,enemy).",
    )
    parser.add_argument(
        "--region",
        help="Optional capture region left,top,width,height (pixels)",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale factor for display window (e.g. 0.75)",
    )
    args = parser.parse_args()

    config = build_config(args)
    model = YOLO(config.model_path)
    class_ids = resolve_class_ids(model, config.classes)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        if config.region:
            monitor = {
                "left": config.region.left,
                "top": config.region.top,
                "width": config.region.width,
                "height": config.region.height,
            }

        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            results = model.predict(
                frame,
                conf=config.conf,
                iou=config.iou,
                classes=class_ids,
                verbose=False,
            )

            annotated = results[0].plot()
            if config.display_scale != 1.0:
                annotated = cv2.resize(
                    annotated,
                    None,
                    fx=config.display_scale,
                    fy=config.display_scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imshow("Live Detect", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
