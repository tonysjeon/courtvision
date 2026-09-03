"""End-to-end player tracking for a decoded video frame."""

from __future__ import annotations

from dataclasses import dataclass

from courtvision.ball.detection import BallCandidate, BallDetector
from courtvision.geometry.camera_motion import CameraMotionCompensator
from courtvision.geometry.mapper import CourtMapper
from courtvision.players.detection import PersonDetection, PersonDetector
from courtvision.players.filtering import ActivePlayerSelector, PlayerCandidate, PlayerId
from courtvision.tracking.ball_tracker import BallTracker
from courtvision.tracking.events import BallTrackingEvent, PlayerTrackingEvent
from courtvision.tracking.player_tracker import PlayerTracker
from courtvision.video.reader import VideoFrame


@dataclass(frozen=True, slots=True)
class PlayerTrackingResult:
    mapper: CourtMapper
    detections: list[PersonDetection]
    candidates: dict[PlayerId, PlayerCandidate]
    events: list[PlayerTrackingEvent]
    ball_candidates: list[BallCandidate]
    ball_event: BallTrackingEvent | None


class PlayerTrackingPipeline:
    def __init__(
        self,
        *,
        match_id: str,
        detector: PersonDetector,
        mapper: CourtMapper,
        selector: ActivePlayerSelector | None = None,
        tracker: PlayerTracker | None = None,
        ball_detector: BallDetector | None = None,
        ball_tracker: BallTracker | None = None,
        camera_motion: CameraMotionCompensator | None = None,
    ) -> None:
        self.match_id = match_id
        self.detector = detector
        self.mapper = mapper
        self.selector = selector or ActivePlayerSelector()
        self.tracker = tracker or PlayerTracker()
        self.ball_detector = ball_detector
        self.ball_tracker = ball_tracker
        self.camera_motion = camera_motion
        if (ball_detector is None) != (ball_tracker is None):
            raise ValueError("ball_detector and ball_tracker must be provided together")

    def process(self, frame: VideoFrame) -> PlayerTrackingResult:
        mapper = self.mapper
        if self.camera_motion is not None:
            mapper = self.camera_motion.mapper_for_frame(frame.image)
            if self.ball_tracker is not None:
                self.ball_tracker.mapper = mapper
        detections = self.detector.detect(
            frame.image,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
        )
        candidates = self.selector.select(
            detections,
            mapper,
            self.tracker.previous_positions(frame.frame_id),
        )
        events = self.tracker.update(
            match_id=self.match_id,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            candidates=candidates,
        )
        ball_candidates = []
        ball_event = None
        if self.ball_detector is not None and self.ball_tracker is not None:
            ball_candidates = self.ball_detector.detect(
                frame.image,
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                excluded_boxes=[candidate.detection.bbox for candidate in candidates.values()],
            )
            ball_event = self.ball_tracker.update(
                match_id=self.match_id,
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                candidates=ball_candidates,
            )
            if ball_event is not None and ball_event.is_interpolated:
                ball_x = ball_event.pixel_position.x
                ball_y = ball_event.pixel_position.y
                if any(
                    candidate.detection.bbox.x1 <= ball_x <= candidate.detection.bbox.x2
                    and candidate.detection.bbox.y1 <= ball_y <= candidate.detection.bbox.y2
                    for candidate in candidates.values()
                ):
                    ball_event = None
        return PlayerTrackingResult(
            mapper=mapper,
            detections=detections,
            candidates=candidates,
            events=events,
            ball_candidates=ball_candidates,
            ball_event=ball_event,
        )
