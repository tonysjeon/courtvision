# CourtVision

CourtVision turns recorded singles tennis footage into normalized court positions and,
eventually, tactical analytics. This repository currently contains the first vertical slice:
video ingestion, manual court calibration, homography, and a visual calibration preview.

The complete V1 plan is documented in [docs/V1_SCOPE.md](docs/V1_SCOPE.md).

## Requirements

- Python 3.11 through 3.14
- [uv](https://docs.astral.sh/uv/)
- An MP4 or MOV tennis clip recorded from a standard broadcast-style camera

## Setup

```bash
make install
make lint
make test
```

## Generate a calibration preview

1. Put a local clip at `data/sample/sample.mp4`.
2. Copy `data/sample/calibration.example.json` to `data/sample/calibration.json`.
3. Update the four pixel coordinates for the visible singles-court baseline corners.
4. Run:

```bash
make process-video
```

The command writes `data/processed/court_preview.png`. The left side shows the source frame
and selected court polygon; the right side shows the normalized court. Local videos,
calibration files, and generated previews are intentionally ignored by Git.

To use different paths or a different starting time:

```bash
uv run courtvision process-video \
  --video /path/to/clip.mov \
  --calibration /path/to/calibration.json \
  --output data/processed/preview.png \
  --start-ms 5000
```

## Current package layout

```text
services/vision/src/courtvision/
├── court/           # calibration models and file loading
├── geometry/        # image/court perspective transforms
├── video/           # timestamp-aware frame ingestion
└── visualization/   # normalized court and calibration previews
```

The vision package runs without Kafka, Spark, Snowflake, or the frontend. Those integrations
will be introduced as later vertical slices after court mapping and player tracking are stable.
