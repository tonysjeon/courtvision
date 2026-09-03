import numpy as np
import pytest
from courtvision.geometry.mapper import CourtMapper
from courtvision.tracking.events import (
    BallTrackingEvent,
    CourtPosition,
    PixelPosition,
    PlayerTrackingEvent,
)
from courtvision.tracking.trajectory import (
    TrajectoryFrame,
    _bounce_anchor,
    _directional_measurement,
    smooth_ball_trajectory,
)


def event(frame_id: int, x: float, y: float, confidence: float = 0.9) -> BallTrackingEvent:
    return BallTrackingEvent(
        match_id="match",
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        confidence=confidence,
        pixel_position=PixelPosition(x=x, y=y),
        court_position=CourtPosition(x=x, y=y),
    )


def player_event(frame_id: int, object_id: str, x: float, y: float) -> PlayerTrackingEvent:
    return PlayerTrackingEvent(
        match_id="match",
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        object_id=object_id,
        track_id=1 if object_id == "near_player" else 2,
        confidence=0.9,
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


def test_uses_each_frames_camera_corrected_mapper() -> None:
    frames = [TrajectoryFrame(index, index * 40) for index in range(3)]
    events = [event(0, 10, 20), None, event(2, 30, 20)]
    frame_mappers = [
        CourtMapper(np.array([[1, 0, -offset], [0, 1, 0], [0, 0, 1]], dtype=float))
        for offset in (0, 10, 20)
    ]

    result = smooth_ball_trajectory(
        frames,
        events,
        CourtMapper(np.eye(3)),
        frame_mappers=frame_mappers,
    )

    assert result[1] is not None
    assert result[1].court_position.x == pytest.approx(10, abs=1)


def test_ground_projection_ignores_airborne_depth_reversal() -> None:
    frames = [TrajectoryFrame(index, index * 40) for index in range(7)]
    events = [
        event(0, 50, 100),
        event(1, 50, 130),
        event(2, 50, 70),
        event(3, 50, 5),
        event(4, 50, -20),
        event(5, 50, 30),
        event(6, 50, 0),
    ]
    players = [
        [
            player_event(index, "far_player", 50, 100),
            player_event(index, "near_player", 50, 0),
        ]
        for index in range(7)
    ]

    result = smooth_ball_trajectory(
        frames,
        events,
        CourtMapper(np.eye(3)),
        player_events=players,
    )

    court_depths = [item.court_position.y for item in result if item is not None]
    assert all(left >= right for left, right in zip(court_depths, court_depths[1:], strict=False))


def test_ground_projection_passes_through_observed_bounce() -> None:
    frames = [TrajectoryFrame(index, index * 40) for index in range(7)]
    events = [
        event(0, 10, 100),
        event(1, 20, 120),
        event(2, 30, 90),
        event(3, 45, 65),
        event(4, 80, 25),
        event(5, 90, 35),
        event(6, 100, 0),
    ]
    bounce = _bounce_anchor(
        frames,
        events,
        (0, "far_player", CourtPosition(x=10, y=100)),
        (6, "near_player", CourtPosition(x=100, y=0)),
        CourtMapper(np.eye(3)),
        None,
    )

    assert bounce is not None
    index, position = bounce
    assert index == 4
    assert position.x == pytest.approx(80)
    assert position.y == pytest.approx(25)


def test_ground_projection_continues_a_clipped_final_shot() -> None:
    frames = [TrajectoryFrame(index, index * 100) for index in range(9)]
    depths = [100, 70, 30, 0, 30, 70, 100, 80, 60]
    events = [event(index, 50, depth) for index, depth in enumerate(depths)]
    players = [
        [
            player_event(index, "far_player", 40, 105),
            player_event(index, "near_player", 60, -5),
        ]
        for index in range(9)
    ]

    result = smooth_ball_trajectory(
        frames,
        events,
        CourtMapper(np.eye(3)),
        player_events=players,
    )

    final = result[-1]
    assert final is not None
    assert final.court_position.y < 50


def test_uses_recent_shot_direction_to_reduce_cross_track_noise() -> None:
    from courtvision.tracking.trajectory import _Measurement

    history = [_Measurement(index * 40, np.array([index * 10.0, 20.0]), 0.9) for index in range(4)]

    corrected, reset = _directional_measurement(
        np.array([40.0, 50.0]),
        0.4,
        160,
        history,
    )

    assert not reset
    assert corrected[1] < 40


def test_resets_direction_on_confident_reversal() -> None:
    from courtvision.tracking.trajectory import _Measurement

    history = [_Measurement(index * 40, np.array([index * 10.0, 20.0]), 0.9) for index in range(4)]
    measurement = np.array([20.0, 20.0])

    corrected, reset = _directional_measurement(measurement, 0.9, 160, history)

    assert reset
    assert np.array_equal(corrected, measurement)
