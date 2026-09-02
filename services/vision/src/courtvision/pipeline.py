"""End-to-end player tracking for a decoded video frame."""

from __future__ import annotations

from dataclasses import dataclass

from courtvision.geometry.mapper import CourtMapper
from courtvision.players.detection import PersonDetection, PersonDetector
from courtvision.players.filtering import ActivePlayerSelector, PlayerCandidate, PlayerId
from courtvision.tracking.events import PlayerTrackingEvent
from courtvision.tracking.player_tracker import PlayerTracker
from courtvision.video.reader import VideoFrame


@dataclass(frozen=True, slots=True)
class PlayerTrackingResult:
    detections: list[PersonDetection]
    candidates: dict[PlayerId, PlayerCandidate]
    events: list[PlayerTrackingEvent]


class PlayerTrackingPipeline:
    def __init__(
        self,
        *,
        match_id: str,
        detector: PersonDetector,
        mapper: CourtMapper,
        selector: ActivePlayerSelector | None = None,
        tracker: PlayerTracker | None = None,
    ) -> None:
        self.match_id = match_id
        self.detector = detector
        self.mapper = mapper
        self.selector = selector or ActivePlayerSelector()
        self.tracker = tracker or PlayerTracker()

    def process(self, frame: VideoFrame) -> PlayerTrackingResult:
        detections = self.detector.detect(
            frame.image,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
        )
        candidates = self.selector.select(
            detections,
            self.mapper,
            self.tracker.previous_positions(frame.frame_id),
        )
        events = self.tracker.update(
            match_id=self.match_id,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            candidates=candidates,
        )
        return PlayerTrackingResult(detections=detections, candidates=candidates, events=events)
