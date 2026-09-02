import numpy as np
from courtvision.ball.detection import BallCandidate
from courtvision.geometry.mapper import CourtMapper
from courtvision.tracking.ball_tracker import BallTracker


def candidate(frame_id: int, timestamp_ms: float, x: float, y: float) -> BallCandidate:
    return BallCandidate(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        confidence=0.8,
        pixel_x=x,
        pixel_y=y,
        radius=4,
    )


def test_tracks_candidate_and_bridges_short_gap() -> None:
    tracker = BallTracker(CourtMapper(np.eye(3)), max_gap_ms=250)
    tracker.update(
        match_id="match",
        frame_id=1,
        timestamp_ms=0,
        candidates=[candidate(1, 0, 10, 20)],
    )
    observed = tracker.update(
        match_id="match",
        frame_id=2,
        timestamp_ms=100,
        candidates=[candidate(2, 100, 20, 25)],
    )
    estimated = tracker.update(
        match_id="match",
        frame_id=3,
        timestamp_ms=200,
        candidates=[],
    )

    assert observed is not None and not observed.is_interpolated
    assert estimated is not None and estimated.is_interpolated
    assert estimated.pixel_position.x > observed.pixel_position.x
    assert estimated.confidence < observed.confidence


def test_preserves_null_after_gap_limit() -> None:
    tracker = BallTracker(CourtMapper(np.eye(3)), max_gap_ms=100)
    tracker.update(
        match_id="match",
        frame_id=1,
        timestamp_ms=0,
        candidates=[candidate(1, 0, 10, 20)],
    )

    event = tracker.update(match_id="match", frame_id=3, timestamp_ms=200, candidates=[])

    assert event is None


def test_reacquires_after_gap_limit() -> None:
    tracker = BallTracker(CourtMapper(np.eye(3)), max_gap_ms=100)
    tracker.update(
        match_id="match",
        frame_id=1,
        timestamp_ms=0,
        candidates=[candidate(1, 0, 10, 20)],
    )

    event = tracker.update(
        match_id="match",
        frame_id=3,
        timestamp_ms=200,
        candidates=[candidate(3, 200, 90, 80)],
    )

    assert event is not None
    assert event.pixel_position.x == 90
    assert not event.is_interpolated


def test_prefers_candidate_near_velocity_prediction() -> None:
    tracker = BallTracker(CourtMapper(np.eye(3)))
    tracker.update(
        match_id="match",
        frame_id=1,
        timestamp_ms=0,
        candidates=[candidate(1, 0, 10, 20)],
    )
    tracker.update(
        match_id="match",
        frame_id=2,
        timestamp_ms=100,
        candidates=[candidate(2, 100, 20, 20)],
    )

    event = tracker.update(
        match_id="match",
        frame_id=3,
        timestamp_ms=200,
        candidates=[candidate(3, 200, 29, 20), candidate(3, 200, 100, 100)],
    )

    assert event is not None
    assert event.pixel_position.x == 29
