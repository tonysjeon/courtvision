from courtvision.cli import build_parser


def test_video_tracking_defaults() -> None:
    args = build_parser().parse_args(
        [
            "track-video",
            "--video",
            "clip.mov",
            "--calibration",
            "calibration.json",
            "--output-video",
            "tracking.mp4",
            "--output-events",
            "tracking.jsonl",
        ]
    )

    assert args.confidence == 0.2
    assert args.model == "yolo11n.pt"
    assert args.frame_skip == 0
