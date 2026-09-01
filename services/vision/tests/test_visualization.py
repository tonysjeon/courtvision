import numpy as np
import pytest
from courtvision.court.calibration import CourtCalibration
from courtvision.geometry.mapper import CourtMapper
from courtvision.visualization.court import render_calibration_preview, render_normalized_court


def test_renders_normalized_court() -> None:
    court = render_normalized_court(width=300, height=450)
    assert court.shape == (450, 300, 3)
    assert np.any(court == 255)


def test_renders_side_by_side_calibration_preview() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    calibration = CourtCalibration.model_validate(
        {
            "near_left_baseline": [155, 692],
            "near_right_baseline": [1065, 692],
            "far_left_baseline": [415, 242],
            "far_right_baseline": [809, 242],
        }
    )
    mapper = CourtMapper.from_calibration(calibration)

    preview = render_calibration_preview(frame, calibration, mapper)

    assert preview.shape == (720, 1920, 3)
    assert np.any(preview != 0)


def test_rejects_tiny_canvas() -> None:
    with pytest.raises(ValueError, match="at least"):
        render_normalized_court(width=100, height=100)
