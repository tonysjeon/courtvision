import json
from pathlib import Path

import pytest
from courtvision.court.calibration import CourtCalibration
from pydantic import ValidationError


def test_loads_versioned_calibration(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "video": "clip.mp4",
                "keypoints": {
                    "near_left_baseline": {"x": 100, "y": 700},
                    "near_right_baseline": {"x": 1100, "y": 700},
                    "far_left_baseline": {"x": 400, "y": 200},
                    "far_right_baseline": {"x": 800, "y": 200},
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = CourtCalibration.from_json(path)

    assert calibration.video == "clip.mp4"
    assert calibration.keypoints.near_left_baseline.x == 100


def test_supports_flat_spec_format() -> None:
    calibration = CourtCalibration.model_validate(
        {
            "near_left_baseline": [100, 700],
            "near_right_baseline": [1100, 700],
            "far_left_baseline": [400, 200],
            "far_right_baseline": [800, 200],
        }
    )

    assert calibration.keypoints.far_right_baseline.x == 800


def test_rejects_negative_pixel_coordinates() -> None:
    with pytest.raises(ValidationError):
        CourtCalibration.model_validate(
            {
                "near_left_baseline": [-1, 700],
                "near_right_baseline": [1100, 700],
                "far_left_baseline": [400, 200],
                "far_right_baseline": [800, 200],
            }
        )


def test_requires_both_frame_dimensions() -> None:
    with pytest.raises(ValidationError, match="provided together"):
        CourtCalibration.model_validate(
            {
                "frame_width": 1920,
                "keypoints": {
                    "near_left_baseline": [100, 700],
                    "near_right_baseline": [1100, 700],
                    "far_left_baseline": [400, 200],
                    "far_right_baseline": [800, 200],
                },
            }
        )
