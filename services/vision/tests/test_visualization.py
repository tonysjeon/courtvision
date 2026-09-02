import numpy as np
import pytest
from courtvision.court.calibration import CourtCalibration
from courtvision.geometry.mapper import CourtMapper
from courtvision.players.detection import BoundingBox, PersonDetection
from courtvision.players.filtering import PlayerCandidate
from courtvision.tracking.events import CourtPosition, PixelPosition, PlayerTrackingEvent
from courtvision.visualization.court import (
    court_position_to_canvas,
    render_calibration_preview,
    render_normalized_court,
)
from courtvision.visualization.tracking import render_tracking_preview


def test_renders_normalized_court() -> None:
    court = render_normalized_court(width=300, height=450)
    assert court.shape == (450, 300, 3)
    assert np.any(court == 255)


def test_draws_doubles_alleys_and_extended_net() -> None:
    court = render_normalized_court(width=600, height=900)
    background = np.array([54, 125, 76], dtype=np.uint8)

    assert np.array_equal(court[200, 142], np.array([255, 255, 255]))
    assert np.array_equal(court[200, 181], np.array([255, 255, 255]))
    assert not np.array_equal(court[450, 130], background)
    assert np.array_equal(court[430, 130], background)


def test_maps_normalized_positions_to_singles_corners() -> None:
    assert court_position_to_canvas(0, 0, 600, 900) == (181, 792)
    assert court_position_to_canvas(100, 100, 600, 900) == (419, 107)


def test_uses_regulation_doubles_court_proportions() -> None:
    court = render_normalized_court(width=600, height=900)
    white = np.all(court == np.array([255, 255, 255], dtype=np.uint8), axis=2)
    doubles_width = 458 - 142
    court_length = 792 - 107

    assert court_length / doubles_width == pytest.approx(23.77 / 10.97, rel=0.01)
    assert np.any(white)


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


def test_renders_player_tracking_preview() -> None:
    frame = np.zeros((450, 600, 3), dtype=np.uint8)
    detection = PersonDetection(
        frame_id=1,
        timestamp_ms=40,
        confidence=0.9,
        bbox=BoundingBox(x1=250, y1=300, x2=300, y2=400),
    )
    candidate = PlayerCandidate(
        player_id="near_player",
        detection=detection,
        pixel_x=275,
        pixel_y=400,
        court_x=50,
        court_y=-5,
    )
    event = PlayerTrackingEvent(
        match_id="demo",
        frame_id=1,
        timestamp_ms=40,
        object_id="near_player",
        track_id=1,
        confidence=0.9,
        pixel_position=PixelPosition(x=275, y=400),
        court_position=CourtPosition(x=50, y=-5),
    )

    preview = render_tracking_preview(
        frame,
        {"near_player": candidate},
        [event],
        timestamp_ms=40,
    )

    assert preview.shape == (450, 900, 3)
    assert np.any(preview != 0)
