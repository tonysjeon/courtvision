from courtvision.players.detection import BoundingBox, PersonDetection
from courtvision.players.filtering import PlayerCandidate, PlayerId
from courtvision.tracking.player_tracker import PlayerTracker


def candidate(player_id: PlayerId, frame_id: int, x: float, y: float) -> PlayerCandidate:
    detection = PersonDetection(
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        confidence=0.9,
        bbox=BoundingBox(x1=x - 5, y1=max(0, y - 10), x2=x + 5, y2=y),
    )
    return PlayerCandidate(
        player_id=player_id,
        detection=detection,
        pixel_x=x,
        pixel_y=y,
        court_x=x,
        court_y=y,
    )


def test_keeps_track_id_across_normal_movement_and_short_gap() -> None:
    tracker = PlayerTracker(max_missing_frames=5, max_displacement_per_frame=5)

    first = tracker.update(
        match_id="match",
        frame_id=1,
        timestamp_ms=40,
        candidates={"near_player": candidate("near_player", 1, 50, 10)},
    )
    second = tracker.update(
        match_id="match",
        frame_id=3,
        timestamp_ms=120,
        candidates={"near_player": candidate("near_player", 3, 56, 12)},
    )

    assert first[0].track_id == second[0].track_id
    assert tracker.track_switch_count == 0


def test_logs_track_switch_for_impossible_jump() -> None:
    tracker = PlayerTracker(max_displacement_per_frame=2)
    first = tracker.update(
        match_id="match",
        frame_id=1,
        timestamp_ms=40,
        candidates={"near_player": candidate("near_player", 1, 10, 10)},
    )
    second = tracker.update(
        match_id="match",
        frame_id=2,
        timestamp_ms=80,
        candidates={"near_player": candidate("near_player", 2, 50, 10)},
    )

    assert first[0].track_id != second[0].track_id
    assert tracker.track_switch_count == 1


def test_expires_track_after_long_gap() -> None:
    tracker = PlayerTracker(max_missing_frames=2)
    first = tracker.update(
        match_id="match",
        frame_id=1,
        timestamp_ms=40,
        candidates={"far_player": candidate("far_player", 1, 50, 90)},
    )
    tracker.update(match_id="match", frame_id=4, timestamp_ms=160, candidates={})
    second = tracker.update(
        match_id="match",
        frame_id=5,
        timestamp_ms=200,
        candidates={"far_player": candidate("far_player", 5, 50, 90)},
    )

    assert first[0].track_id != second[0].track_id
    assert tracker.track_switch_count == 0


def test_does_not_offer_stale_position_to_selector() -> None:
    tracker = PlayerTracker(max_missing_frames=2)
    tracker.update(
        match_id="match",
        frame_id=1,
        timestamp_ms=40,
        candidates={"near_player": candidate("near_player", 1, 50, 10)},
    )

    assert tracker.previous_positions(4) == {}


def test_smooths_far_player_position_without_changing_near_player_position() -> None:
    tracker = PlayerTracker(
        max_displacement_per_frame=20,
        far_player_smoothing=0.5,
        far_player_vertical_smoothing=0.2,
    )
    tracker.update(
        match_id="match",
        frame_id=1,
        timestamp_ms=40,
        candidates={
            "near_player": candidate("near_player", 1, 40, 10),
            "far_player": candidate("far_player", 1, 40, 90),
        },
    )

    events = tracker.update(
        match_id="match",
        frame_id=2,
        timestamp_ms=80,
        candidates={
            "near_player": candidate("near_player", 2, 50, 20),
            "far_player": candidate("far_player", 2, 50, 80),
        },
    )
    by_player = {event.object_id: event for event in events}

    assert by_player["near_player"].court_position.x == 50
    assert by_player["near_player"].court_position.y == 20
    assert by_player["far_player"].court_position.x == 45
    assert by_player["far_player"].court_position.y == 88
