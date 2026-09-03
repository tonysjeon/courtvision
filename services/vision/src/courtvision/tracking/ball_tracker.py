"""Velocity-aware tracking for a single tennis ball."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from courtvision.ball.detection import BallCandidate
from courtvision.geometry.mapper import CourtMapper
from courtvision.tracking.events import BallTrackingEvent, CourtPosition, PixelPosition


@dataclass(slots=True)
class _BallState:
    frame_id: int
    timestamp_ms: float
    observed_timestamp_ms: float
    pixel_x: float
    pixel_y: float
    velocity_x: float
    velocity_y: float
    confidence: float


class BallTracker:
    """Choose a plausible candidate and bridge only brief missing observations."""

    def __init__(
        self,
        mapper: CourtMapper,
        *,
        max_speed_pixels_per_second: float = 5000.0,
        max_gap_ms: float = 300.0,
        velocity_smoothing: float = 0.45,
    ) -> None:
        self.mapper = mapper
        self.max_speed_pixels_per_second = max_speed_pixels_per_second
        self.max_gap_ms = max_gap_ms
        self.velocity_smoothing = velocity_smoothing
        self._state: _BallState | None = None

    def update(
        self,
        *,
        match_id: str,
        frame_id: int,
        timestamp_ms: float,
        candidates: list[BallCandidate],
    ) -> BallTrackingEvent | None:
        if (
            self._state is not None
            and timestamp_ms - self._state.observed_timestamp_ms > self.max_gap_ms
        ):
            self._state = None
        candidate = self._select_candidate(candidates, timestamp_ms)
        if candidate is not None:
            self._observe(candidate)
            return self._event(match_id, is_interpolated=False)

        state = self._state
        if state is None or timestamp_ms - state.observed_timestamp_ms > self.max_gap_ms:
            return None

        elapsed_ms = max(0.0, timestamp_ms - state.timestamp_ms)
        state.pixel_x += state.velocity_x * elapsed_ms
        state.pixel_y += state.velocity_y * elapsed_ms
        state.frame_id = frame_id
        state.timestamp_ms = timestamp_ms
        gap_ratio = (timestamp_ms - state.observed_timestamp_ms) / self.max_gap_ms
        state.confidence = max(0.05, state.confidence * (1 - gap_ratio) * 0.5)
        return self._event(match_id, is_interpolated=True)

    def _select_candidate(
        self,
        candidates: list[BallCandidate],
        timestamp_ms: float,
    ) -> BallCandidate | None:
        if not candidates:
            return None
        state = self._state
        if state is None:
            return candidates[0]

        elapsed_ms = max(1.0, timestamp_ms - state.timestamp_ms)
        predicted_x = state.pixel_x + state.velocity_x * elapsed_ms
        predicted_y = state.pixel_y + state.velocity_y * elapsed_ms
        max_distance = 30.0 + self.max_speed_pixels_per_second * elapsed_ms / 1000
        ranked = sorted(
            candidates,
            key=lambda item: (
                hypot(item.pixel_x - predicted_x, item.pixel_y - predicted_y) - item.confidence * 80
            ),
        )
        best = ranked[0]
        distance = hypot(best.pixel_x - predicted_x, best.pixel_y - predicted_y)
        return best if distance <= max_distance else None

    def _observe(self, candidate: BallCandidate) -> None:
        state = self._state
        if state is None:
            self._state = _BallState(
                frame_id=candidate.frame_id,
                timestamp_ms=candidate.timestamp_ms,
                observed_timestamp_ms=candidate.timestamp_ms,
                pixel_x=candidate.pixel_x,
                pixel_y=candidate.pixel_y,
                velocity_x=0.0,
                velocity_y=0.0,
                confidence=candidate.confidence,
            )
            return

        elapsed_ms = max(1.0, candidate.timestamp_ms - state.timestamp_ms)
        measured_velocity_x = (candidate.pixel_x - state.pixel_x) / elapsed_ms
        measured_velocity_y = (candidate.pixel_y - state.pixel_y) / elapsed_ms
        alpha = self.velocity_smoothing
        state.velocity_x += alpha * (measured_velocity_x - state.velocity_x)
        state.velocity_y += alpha * (measured_velocity_y - state.velocity_y)
        state.frame_id = candidate.frame_id
        state.timestamp_ms = candidate.timestamp_ms
        state.observed_timestamp_ms = candidate.timestamp_ms
        state.pixel_x = candidate.pixel_x
        state.pixel_y = candidate.pixel_y
        state.confidence = candidate.confidence

    def _event(self, match_id: str, *, is_interpolated: bool) -> BallTrackingEvent:
        state = self._state
        assert state is not None
        court_x, court_y = self.mapper.transform(state.pixel_x, state.pixel_y)
        return BallTrackingEvent(
            match_id=match_id,
            frame_id=state.frame_id,
            timestamp_ms=state.timestamp_ms,
            confidence=state.confidence,
            is_interpolated=is_interpolated,
            pixel_position=PixelPosition(x=state.pixel_x, y=state.pixel_y),
            court_position=CourtPosition(x=court_x, y=court_y),
        )
