import numpy as np
import pytest
from courtvision.court.calibration import CourtCalibration
from courtvision.geometry.mapper import CourtMapper
from courtvision.visualization.court import (
    court_position_to_canvas,
    render_calibration_preview,
    render_normalized_court,
)


def test_renders_normalized_court() -> None:
    court = render_normalized_court(width=300, height=450)
    assert court.shape == (450, 300, 3)
    assert np.any(court == 255)


def test_draws_doubles_alleys_and_extended_net() -> None:
    court = render_normalized_court(width=600, height=900)
    background = np.array([54, 125, 76], dtype=np.uint8)

    assert np.array_equal(court[200, 96], np.array([255, 255, 255]))
    assert np.array_equal(court[200, 147], np.array([255, 255, 255]))
    assert not np.array_equal(court[449, 80], background)
    assert np.array_equal(court[430, 80], background)


def test_maps_normalized_positions_to_singles_corners() -> None:
    assert court_position_to_canvas(0, 0, 600, 900) == (147, 791)
    assert court_position_to_canvas(100, 100, 600, 900) == (452, 108)


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
