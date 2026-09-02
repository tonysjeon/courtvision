# CourtVision

CourtVision maps points from recorded tennis footage onto a normalized court.

Right now, the project can calibrate a court, detect the two active players and tennis ball,
and map their movement onto a normalized court.

## Requirements

- Python 3.11 through 3.14
- [uv](https://docs.astral.sh/uv/)
- An MP4 or MOV tennis clip

## Setup

```bash
make install
make lint
make test
```

## Try it

1. Put a local clip at `data/sample/sample.mp4`.
2. Copy `data/sample/calibration.example.json` to `data/sample/calibration.json`.
3. Update the four pixel coordinates for the visible singles-court baseline corners.
4. Run:

```bash
make process-video
```

The preview is written to `data/processed/court_preview.png`. Videos, local calibration files,
and generated output in these directories are ignored by Git.

To track the players and ball:

```bash
make track-video VIDEO=data/sample/sample.mov
```

This writes an annotated video and JSONL tracking events to `data/processed/`.

To use different paths or a different starting time:

```bash
uv run courtvision process-video \
  --video /path/to/clip.mov \
  --calibration /path/to/calibration.json \
  --output data/processed/preview.png \
  --start-ms 5000
```

## Layout

```text
services/vision/src/courtvision/
├── court/           # calibration models and file loading
├── ball/            # tennis-ball candidate detection
├── geometry/        # image/court perspective transforms
├── players/         # person detection and active-player selection
├── tracking/        # persistent tracks and event schemas
├── video/           # timestamp-aware frame ingestion
└── visualization/   # normalized court and calibration previews
```
