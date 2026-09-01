"""Command-line entry points for local CourtVision development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from courtvision.court.calibration import CourtCalibration
from courtvision.geometry.mapper import CourtMapper
from courtvision.video.reader import VideoReader
from courtvision.visualization.court import render_calibration_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="courtvision")
    subparsers = parser.add_subparsers(dest="command", required=True)
    process = subparsers.add_parser(
        "process-video",
        help="Read a frame and render its manual court calibration.",
    )
    process.add_argument("--video", type=Path, required=True)
    process.add_argument("--calibration", type=Path, required=True)
    process.add_argument("--output", type=Path, required=True)
    process.add_argument("--start-ms", type=float, default=0.0)
    return parser


def process_video(args: argparse.Namespace) -> int:
    calibration = CourtCalibration.from_json(args.calibration)
    mapper = CourtMapper.from_calibration(calibration)

    with VideoReader(args.video, start_timestamp_ms=args.start_ms, max_frames=1) as reader:
        frame = next(iter(reader), None)
        if frame is None:
            raise RuntimeError("No frame was available in the requested video range")

        preview = render_calibration_preview(frame.image, calibration, mapper)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), preview):
            raise RuntimeError(f"Could not write preview to {args.output}")

        result = {
            "video": str(args.video),
            "output": str(args.output),
            "frame_id": frame.frame_id,
            "timestamp_ms": frame.timestamp_ms,
            "metadata": reader.metadata.model_dump(),
        }
        print(json.dumps(result, indent=2))
    return 0


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "process-video":
        raise SystemExit(process_video(args))
