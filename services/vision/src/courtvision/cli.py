"""Command-line entry points for local CourtVision development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2

from courtvision.court.calibration import CourtCalibration
from courtvision.geometry.mapper import CourtMapper
from courtvision.pipeline import PlayerTrackingPipeline
from courtvision.players.yolo import YoloPersonDetector
from courtvision.video.reader import VideoReader
from courtvision.visualization.court import render_calibration_preview
from courtvision.visualization.tracking import render_tracking_preview


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

    track = subparsers.add_parser(
        "track-players",
        help="Detect and track the active players in a video.",
    )
    track.add_argument("--video", type=Path, required=True)
    track.add_argument("--calibration", type=Path, required=True)
    track.add_argument("--output-video", type=Path, required=True)
    track.add_argument("--output-events", type=Path, required=True)
    track.add_argument("--match-id", default="local_match")
    track.add_argument("--model", default="yolo11n.pt")
    track.add_argument("--confidence", type=float, default=0.20)
    track.add_argument("--device")
    track.add_argument("--frame-skip", type=int, default=0)
    track.add_argument("--max-frames", type=int)
    track.add_argument("--start-ms", type=float, default=0.0)
    track.add_argument("--end-ms", type=float)
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


def track_players(args: argparse.Namespace) -> int:
    calibration = CourtCalibration.from_json(args.calibration)
    mapper = CourtMapper.from_calibration(calibration)
    detector = YoloPersonDetector(
        args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    pipeline = PlayerTrackingPipeline(
        match_id=args.match_id,
        detector=detector,
        mapper=mapper,
    )

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_events.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    event_count = 0
    started_at = perf_counter()

    try:
        with (
            VideoReader(
                args.video,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
                start_timestamp_ms=args.start_ms,
                end_timestamp_ms=args.end_ms,
            ) as reader,
            args.output_events.open("w", encoding="utf-8") as events_file,
        ):
            output_fps = reader.metadata.fps / (args.frame_skip + 1)
            for frame in reader:
                result = pipeline.process(frame)
                preview = render_tracking_preview(
                    frame.image,
                    result.candidates,
                    result.events,
                    timestamp_ms=frame.timestamp_ms,
                )
                if writer is None:
                    height, width = preview.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create output video: {args.output_video}")
                writer.write(preview)
                for event in result.events:
                    events_file.write(event.model_dump_json() + "\n")
                frame_count += 1
                event_count += len(result.events)
            if frame_count == 0:
                raise RuntimeError("No frames were available in the requested video range")
    finally:
        if writer is not None:
            writer.release()

    elapsed_seconds = perf_counter() - started_at
    summary = {
        "video": str(args.video),
        "output_video": str(args.output_video),
        "output_events": str(args.output_events),
        "frames_processed": frame_count,
        "events_written": event_count,
        "track_switches": pipeline.tracker.track_switch_count,
        "processing_fps": frame_count / elapsed_seconds if elapsed_seconds else 0,
    }
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "process-video":
        raise SystemExit(process_video(args))
    if args.command == "track-players":
        raise SystemExit(track_players(args))
