"""Persistent role-aware tracking for the two active players."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import hypot

from courtvision.players.filtering import PlayerCandidate, PlayerId
from courtvision.tracking.events import CourtPosition, PixelPosition, PlayerTrackingEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TrackState:
    track_id: int
    frame_id: int
    court_x: float
    court_y: float


class PlayerTracker:
    """Keep near/far identities stable through short misses and normal movement."""

    def __init__(
        self,
        *,
        max_missing_frames: int = 30,
        max_displacement_per_frame: float = 8.0,
    ) -> None:
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")
        if max_displacement_per_frame <= 0:
            raise ValueError("max_displacement_per_frame must be positive")
        self.max_missing_frames = max_missing_frames
        self.max_displacement_per_frame = max_displacement_per_frame
        self._states: dict[PlayerId, _TrackState] = {}
        self._next_track_id = 1
        self.track_switch_count = 0

    def previous_positions(self, frame_id: int) -> dict[PlayerId, tuple[float, float]]:
        self._expire_old_tracks(frame_id)
        return {
            player_id: (state.court_x, state.court_y) for player_id, state in self._states.items()
        }

    def update(
        self,
        *,
        match_id: str,
        frame_id: int,
        timestamp_ms: float,
        candidates: dict[PlayerId, PlayerCandidate],
    ) -> list[PlayerTrackingEvent]:
        self._expire_old_tracks(frame_id)
        events = []
        for player_id, candidate in candidates.items():
            state = self._states.get(player_id)
            if state is None or not self._continues_track(state, candidate, frame_id):
                if state is not None:
                    self.track_switch_count += 1
                    logger.warning(
                        "player track switch",
                        extra={"player_id": player_id, "frame_id": frame_id},
                    )
                state = _TrackState(
                    track_id=self._next_track_id,
                    frame_id=frame_id,
                    court_x=candidate.court_x,
                    court_y=candidate.court_y,
                )
                self._next_track_id += 1
                self._states[player_id] = state
            else:
                state.frame_id = frame_id
                state.court_x = candidate.court_x
                state.court_y = candidate.court_y

            events.append(
                PlayerTrackingEvent(
                    match_id=match_id,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    object_id=player_id,
                    track_id=state.track_id,
                    confidence=candidate.detection.confidence,
                    pixel_position=PixelPosition(x=candidate.pixel_x, y=candidate.pixel_y),
                    court_position=CourtPosition(x=candidate.court_x, y=candidate.court_y),
                )
            )
        return events

    def _expire_old_tracks(self, frame_id: int) -> None:
        expired = [
            player_id
            for player_id, state in self._states.items()
            if frame_id - state.frame_id > self.max_missing_frames
        ]
        for player_id in expired:
            del self._states[player_id]

    def _continues_track(
        self,
        state: _TrackState,
        candidate: PlayerCandidate,
        frame_id: int,
    ) -> bool:
        frame_gap = max(1, frame_id - state.frame_id)
        displacement = hypot(candidate.court_x - state.court_x, candidate.court_y - state.court_y)
        return displacement <= self.max_displacement_per_frame * frame_gap
