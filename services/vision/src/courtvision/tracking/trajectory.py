"""Offline smoothing for a detected tennis-ball trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from courtvision.geometry.mapper import CourtMapper
from courtvision.tracking.events import (
    BallTrackingEvent,
    CourtPosition,
    PixelPosition,
    PlayerTrackingEvent,
)


@dataclass(frozen=True, slots=True)
class TrajectoryFrame:
    frame_id: int
    timestamp_ms: float


class _FilterStep(NamedTuple):
    filtered_state: np.ndarray
    filtered_covariance: np.ndarray
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    transition: np.ndarray


class _Measurement(NamedTuple):
    timestamp_ms: float
    position: np.ndarray
    confidence: float


def smooth_ball_trajectory(
    frames: list[TrajectoryFrame],
    events: list[BallTrackingEvent | None],
    mapper: CourtMapper,
    *,
    player_events: list[list[PlayerTrackingEvent]] | None = None,
    max_anchor_gap_ms: float = 600.0,
    acceleration_noise: float = 3500.0,
) -> list[BallTrackingEvent | None]:
    """Estimate a continuous 2D path using observations before and after each frame."""
    if len(frames) != len(events):
        raise ValueError("frames and events must have the same length")

    anchors = [
        index
        for index, event in enumerate(events)
        if event is not None and not event.is_interpolated
    ]
    output: list[BallTrackingEvent | None] = [None] * len(frames)
    if not anchors:
        return output

    segment_start = 0
    for anchor_offset in range(1, len(anchors) + 1):
        at_end = anchor_offset == len(anchors)
        gap_too_large = not at_end and (
            frames[anchors[anchor_offset]].timestamp_ms
            - frames[anchors[anchor_offset - 1]].timestamp_ms
            > max_anchor_gap_ms
        )
        if not at_end and not gap_too_large:
            continue
        segment_anchors = anchors[segment_start:anchor_offset]
        _smooth_segment(
            frames,
            events,
            segment_anchors,
            output,
            mapper,
            acceleration_noise=acceleration_noise,
        )
        segment_start = anchor_offset
    if player_events is not None:
        if len(player_events) != len(frames):
            raise ValueError("player_events and frames must have the same length")
        _apply_ground_projection(frames, output, player_events)
    return output


def _smooth_segment(
    frames: list[TrajectoryFrame],
    events: list[BallTrackingEvent | None],
    anchors: list[int],
    output: list[BallTrackingEvent | None],
    mapper: CourtMapper,
    *,
    acceleration_noise: float,
) -> None:
    if not anchors:
        return
    if len(anchors) == 1:
        output[anchors[0]] = events[anchors[0]]
        return

    start, end = anchors[0], anchors[-1]
    anchor_set = set(anchors)
    first = events[start]
    second = events[anchors[1]]
    assert first is not None and second is not None
    initial_seconds = max(
        0.001,
        (frames[anchors[1]].timestamp_ms - frames[start].timestamp_ms) / 1000,
    )
    state = np.array(
        [
            first.pixel_position.x,
            first.pixel_position.y,
            (second.pixel_position.x - first.pixel_position.x) / initial_seconds,
            (second.pixel_position.y - first.pixel_position.y) / initial_seconds,
        ],
        dtype=np.float64,
    )
    covariance = np.diag([100.0, 100.0, 250_000.0, 250_000.0])
    steps: list[_FilterStep] = []
    accepted_anchors: set[int] = set()
    measurement_history: list[_Measurement] = []

    for index in range(start, end + 1):
        if index == start:
            transition = np.eye(4)
            predicted_state = state.copy()
            predicted_covariance = covariance.copy()
        else:
            elapsed_seconds = max(
                0.001,
                (frames[index].timestamp_ms - frames[index - 1].timestamp_ms) / 1000,
            )
            transition = _transition(elapsed_seconds)
            process_noise = _process_noise(elapsed_seconds, acceleration_noise)
            predicted_state = transition @ state
            predicted_covariance = transition @ covariance @ transition.T + process_noise

        state = predicted_state
        covariance = predicted_covariance
        event = events[index] if index in anchor_set else None
        if event is not None:
            raw_measurement = np.array(
                [event.pixel_position.x, event.pixel_position.y],
                dtype=np.float64,
            )
            measurement, direction_reset = _directional_measurement(
                raw_measurement,
                event.confidence,
                frames[index].timestamp_ms,
                measurement_history,
            )
            if direction_reset:
                measurement_history.clear()
            residual = measurement - state[:2]
            uncertainty = 10.0 + (1.0 - event.confidence) * 25.0
            innovation_covariance = covariance[:2, :2] + np.eye(2) * uncertainty**2
            innovation_limit = max(160.0, 4.0 * np.sqrt(np.trace(innovation_covariance)))
            if index == start or np.linalg.norm(residual) <= innovation_limit:
                observation = np.array(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                )
                gain = covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
                state = state + gain @ residual
                covariance = (np.eye(4) - gain @ observation) @ covariance
                accepted_anchors.add(index)
                measurement_history.append(
                    _Measurement(
                        frames[index].timestamp_ms,
                        raw_measurement,
                        event.confidence,
                    )
                )
                measurement_history = measurement_history[-6:]
        steps.append(
            _FilterStep(
                state.copy(),
                covariance.copy(),
                predicted_state.copy(),
                predicted_covariance.copy(),
                transition.copy(),
            )
        )

    smoothed_states = [step.filtered_state.copy() for step in steps]
    smoothed_covariances = [step.filtered_covariance.copy() for step in steps]
    for offset in range(len(steps) - 2, -1, -1):
        next_step = steps[offset + 1]
        gain = (
            steps[offset].filtered_covariance
            @ next_step.transition.T
            @ np.linalg.pinv(next_step.predicted_covariance)
        )
        smoothed_states[offset] += gain @ (smoothed_states[offset + 1] - next_step.predicted_state)
        smoothed_covariances[offset] += (
            gain @ (smoothed_covariances[offset + 1] - next_step.predicted_covariance) @ gain.T
        )

    match_id = first.match_id
    anchor_confidences = {
        index: events[index].confidence for index in accepted_anchors if events[index] is not None
    }
    for offset, index in enumerate(range(start, end + 1)):
        pixel_x = float(smoothed_states[offset][0])
        pixel_y = float(smoothed_states[offset][1])
        court_x, court_y = mapper.transform(pixel_x, pixel_y)
        is_interpolated = index not in accepted_anchors
        confidence = anchor_confidences.get(index)
        if confidence is None:
            confidence = _interpolated_confidence(index, anchor_confidences)
        output[index] = BallTrackingEvent(
            match_id=match_id,
            frame_id=frames[index].frame_id,
            timestamp_ms=frames[index].timestamp_ms,
            confidence=confidence,
            is_interpolated=is_interpolated,
            pixel_position=PixelPosition(x=pixel_x, y=pixel_y),
            court_position=CourtPosition(x=court_x, y=court_y),
        )


def _transition(elapsed_seconds: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, elapsed_seconds, 0.0],
            [0.0, 1.0, 0.0, elapsed_seconds],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def _directional_measurement(
    measurement: np.ndarray,
    confidence: float,
    timestamp_ms: float,
    history: list[_Measurement],
) -> tuple[np.ndarray, bool]:
    """Blend a detection with the recent shot direction and identify real reversals."""
    recent = [item for item in history if timestamp_ms - item.timestamp_ms <= 400]
    if len(recent) < 3:
        return measurement, False

    times = np.array([item.timestamp_ms for item in recent], dtype=np.float64)
    times = (times - times[-1]) / 1000
    positions = np.array([item.position for item in recent])
    weights = np.array([item.confidence for item in recent], dtype=np.float64)
    design = np.column_stack((times, np.ones_like(times)))
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_positions = positions * np.sqrt(weights)[:, None]
    coefficients = np.linalg.lstsq(weighted_design, weighted_positions, rcond=None)[0]
    velocity = coefficients[0]
    elapsed_seconds = (timestamp_ms - recent[-1].timestamp_ms) / 1000
    prediction = coefficients[1] + velocity * elapsed_seconds

    step = measurement - recent[-1].position
    speed = np.linalg.norm(velocity)
    step_size = np.linalg.norm(step)
    reverses_direction = (
        confidence >= 0.75
        and speed > 80
        and step_size > 3
        and np.dot(step, velocity) < -0.35 * speed * step_size
    )
    if reverses_direction:
        return measurement, True

    prediction_weight = 0.15 + (1.0 - confidence) * 0.4
    if speed > 80:
        direction = velocity / speed
        residual = measurement - prediction
        cross_track = residual - np.dot(residual, direction) * direction
        if np.linalg.norm(cross_track) > 35:
            prediction_weight = max(prediction_weight, 0.5)
    prediction_weight = float(np.clip(prediction_weight, 0.15, 0.55))
    blended = measurement * (1.0 - prediction_weight) + prediction * prediction_weight
    return blended, False


def _process_noise(elapsed_seconds: float, acceleration_noise: float) -> np.ndarray:
    position = elapsed_seconds**4 / 4
    cross = elapsed_seconds**3 / 2
    velocity = elapsed_seconds**2
    variance = acceleration_noise**2
    return variance * np.array(
        [
            [position, 0.0, cross, 0.0],
            [0.0, position, 0.0, cross],
            [cross, 0.0, velocity, 0.0],
            [0.0, cross, 0.0, velocity],
        ],
    )


def _interpolated_confidence(index: int, anchors: dict[int, float]) -> float:
    neighbors = sorted(anchors, key=lambda anchor: abs(anchor - index))[:2]
    if not neighbors:
        return 0.05
    distance = min(abs(anchor - index) for anchor in neighbors)
    confidence = min(anchors[anchor] for anchor in neighbors)
    return max(0.05, confidence * (0.92**distance))


def _apply_ground_projection(
    frames: list[TrajectoryFrame],
    ball_events: list[BallTrackingEvent | None],
    player_events: list[list[PlayerTrackingEvent]],
) -> None:
    """Separate apparent ball height from its top-down ground movement."""
    segment_start = None
    for index in range(len(ball_events) + 1):
        has_ball = index < len(ball_events) and ball_events[index] is not None
        if has_ball and segment_start is None:
            segment_start = index
        if not has_ball and segment_start is not None:
            _project_segment_to_ground(
                frames,
                ball_events,
                player_events,
                segment_start,
                index - 1,
            )
            segment_start = None


def _project_segment_to_ground(
    frames: list[TrajectoryFrame],
    ball_events: list[BallTrackingEvent | None],
    player_events: list[list[PlayerTrackingEvent]],
    start: int,
    end: int,
) -> None:
    visits = _side_visits(frames, ball_events, start, end)
    anchors = [
        anchor
        for visit in visits
        if (anchor := _contact_anchor(visit, ball_events, player_events)) is not None
    ]
    if len(anchors) < 2:
        return

    anchors = _alternating_anchors(anchors)
    if len(anchors) < 2:
        return

    for left, right in zip(anchors, anchors[1:], strict=False):
        left_index, _, left_position = left
        right_index, _, right_position = right
        duration = max(
            1.0,
            frames[right_index].timestamp_ms - frames[left_index].timestamp_ms,
        )
        for index in range(left_index, right_index + 1):
            event = ball_events[index]
            if event is None:
                continue
            progress = (frames[index].timestamp_ms - frames[left_index].timestamp_ms) / duration
            court_x = left_position.x + progress * (right_position.x - left_position.x)
            court_y = left_position.y + progress * (right_position.y - left_position.y)
            ball_events[index] = event.model_copy(
                update={"court_position": CourtPosition(x=court_x, y=court_y)}
            )

    first_index, _, first_position = anchors[0]
    for index in range(start, first_index):
        event = ball_events[index]
        if event is not None:
            ball_events[index] = event.model_copy(update={"court_position": first_position})
    last_index, _, last_position = anchors[-1]
    for index in range(last_index + 1, end + 1):
        event = ball_events[index]
        if event is not None:
            ball_events[index] = event.model_copy(update={"court_position": last_position})


def _side_visits(
    frames: list[TrajectoryFrame],
    ball_events: list[BallTrackingEvent | None],
    start: int,
    end: int,
) -> list[tuple[str, list[int]]]:
    visits: list[tuple[str, list[int]]] = []
    for index in range(start, end + 1):
        event = ball_events[index]
        if event is None:
            continue
        if event.court_position.y <= 25:
            side = "near_player"
        elif event.court_position.y >= 75:
            side = "far_player"
        else:
            continue
        if (
            visits
            and visits[-1][0] == side
            and frames[index].timestamp_ms - frames[visits[-1][1][-1]].timestamp_ms <= 800
        ):
            visits[-1][1].append(index)
        else:
            visits.append((side, [index]))
    return visits


def _contact_anchor(
    visit: tuple[str, list[int]],
    ball_events: list[BallTrackingEvent | None],
    player_events: list[list[PlayerTrackingEvent]],
) -> tuple[int, str, CourtPosition] | None:
    side, indices = visit
    candidates = []
    for index in indices:
        ball = ball_events[index]
        if ball is None:
            continue
        player = next((event for event in player_events[index] if event.object_id == side), None)
        if player is None:
            continue
        distance = np.hypot(
            ball.pixel_position.x - player.pixel_position.x,
            ball.pixel_position.y - player.pixel_position.y,
        )
        candidates.append((distance, index, player))
    if not candidates:
        return None
    _, index, player = min(candidates, key=lambda candidate: candidate[0])
    # Player positions provide the best single-camera estimate of the ball's
    # ground location at racket contact. Keep a small amount of the ball's
    # apparent lateral position so wide contact points are still visible.
    ball = ball_events[index]
    assert ball is not None
    court_x = 0.75 * player.court_position.x + 0.25 * ball.court_position.x
    court_y = float(np.clip(player.court_position.y, -12.0, 112.0))
    return index, side, CourtPosition(x=court_x, y=court_y)


def _alternating_anchors(
    anchors: list[tuple[int, str, CourtPosition]],
) -> list[tuple[int, str, CourtPosition]]:
    output = []
    for anchor in sorted(anchors, key=lambda item: item[0]):
        if output and output[-1][1] == anchor[1]:
            output[-1] = anchor
        else:
            output.append(anchor)
    return output
