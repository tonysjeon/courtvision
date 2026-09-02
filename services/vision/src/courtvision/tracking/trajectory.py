"""Offline smoothing for a detected tennis-ball trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from courtvision.geometry.mapper import CourtMapper
from courtvision.tracking.events import BallTrackingEvent, CourtPosition, PixelPosition


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


def smooth_ball_trajectory(
    frames: list[TrajectoryFrame],
    events: list[BallTrackingEvent | None],
    mapper: CourtMapper,
    *,
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
            measurement = np.array(
                [event.pixel_position.x, event.pixel_position.y],
                dtype=np.float64,
            )
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
