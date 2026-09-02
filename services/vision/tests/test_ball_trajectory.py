import numpy as np
from courtvision.geometry.mapper import CourtMapper
from courtvision.tracking.events import BallTrackingEvent, CourtPosition, PixelPosition
from courtvision.tracking.trajectory import TrajectoryFrame, smooth_ball_trajectory


def event(frame_id: int, x: float, y: float, confidence: float = 0.9) -> BallTrackingEvent:
    return BallTrackingEvent(
        match_id="match",
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        confidence=confidence,
        pixel_position=PixelPosition(x=x, y=y),
        court_position=CourtPosition(x=x, y=y),
    )


def test_estimates_missing_frame_from_both_sides() -> None:
    frames = [TrajectoryFrame(index, index * 40) for index in range(5)]
    events = [event(0, 10, 20), None, None, None, event(4, 50, 20)]

    result = smooth_ball_trajectory(frames, events, CourtMapper(np.eye(3)))

    assert all(item is not None for item in result)
    middle = result[2]
    assert middle is not None and middle.is_interpolated
    assert 25 < middle.pixel_position.x < 35


def test_does_not_bridge_long_gap_between_rallies() -> None:
    frames = [TrajectoryFrame(index, index * 100) for index in range(10)]
    events = [event(0, 10, 20)] + [None] * 8 + [event(9, 80, 20)]

    result = smooth_ball_trajectory(
        frames,
        events,
        CourtMapper(np.eye(3)),
        max_anchor_gap_ms=500,
    )

    assert result[0] is events[0]
    assert result[9] is events[9]
    assert all(item is None for item in result[1:9])


def test_smooths_noisy_observed_coordinates() -> None:
    frames = [TrajectoryFrame(index, index * 40) for index in range(5)]
    events = [
        event(0, 10, 20),
        event(1, 20, 20),
        event(2, 36, 26, confidence=0.4),
        event(3, 40, 20),
        event(4, 50, 20),
    ]

    result = smooth_ball_trajectory(frames, events, CourtMapper(np.eye(3)))

    middle = result[2]
    assert middle is not None
    assert middle.pixel_position.y < 26
