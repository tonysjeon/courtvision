import numpy as np
from courtvision.geometry.mapper import CourtMapper
from courtvision.pipeline import PlayerTrackingPipeline
from courtvision.players.detection import BoundingBox, PersonDetection
from courtvision.video.reader import VideoFrame


class FixedDetector:
    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: float,
    ) -> list[PersonDetection]:
        return [
            PersonDetection(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                confidence=0.95,
                bbox=BoundingBox(x1=40, y1=0, x2=60, y2=10),
            ),
            PersonDetection(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                confidence=0.9,
                bbox=BoundingBox(x1=40, y1=70, x2=60, y2=90),
            ),
        ]


def test_processes_frame_into_tracking_events() -> None:
    pipeline = PlayerTrackingPipeline(
        match_id="demo",
        detector=FixedDetector(),
        mapper=CourtMapper(np.eye(3)),
    )
    frame = VideoFrame(frame_id=2, timestamp_ms=80, image=np.zeros((100, 100, 3)))

    result = pipeline.process(frame)

    assert [event.object_id for event in result.events] == ["near_player", "far_player"]
    assert all(event.match_id == "demo" for event in result.events)
