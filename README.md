# CourtVision

CourtVision translates tennis match footage into a top-down view of player and ball movement
during each rally.

It is an enhanced version of a UCLA Tennis Consulting project I led, expanded with
camera-aware calibration and more stable movement tracking.

## Demo

![CourtVision tracking demo](assets/courtvision-demo.gif)

The broadcast view shows the detections used by the tracker. The right panel maps that
movement onto the court, including the ball's estimated path through the air.

## Pipeline

1. Calibrate the visible court from its four baseline corners.
2. Detect and track both players while compensating for broadcast camera movement.
3. Detect the ball and estimate its path between frames, including airborne shots and bounces.
4. Export an annotated video, top-down court view, and frame-by-frame JSONL events.

## Tech stack

Python · OpenCV · NumPy · PyTorch · Ultralytics YOLO · TrackNet · Pydantic

## Development

Requires Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/).

```bash
make install
make lint
make test
```

Place MP4 or MOV footage and its calibration file in `data/sample/`. Generated videos and
events are written to `data/processed/`.

```bash
make process-video
make track-video VIDEO=data/sample/sample.mov
```

Ball tracking expects TrackNet tennis weights at `models/tracknet-tennis.pt`.
