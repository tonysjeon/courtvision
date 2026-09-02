import cv2
import numpy as np
from courtvision.ball.motion import MotionBallDetector
from courtvision.geometry.mapper import CourtMapper
from courtvision.players.detection import BoundingBox


def test_detects_small_yellow_ball_near_court() -> None:
    detector = MotionBallDetector(CourtMapper(np.eye(3)))
    frame = np.zeros((140, 140, 3), dtype=np.uint8)
    cv2.circle(frame, (60, 70), 5, (0, 255, 255), -1)

    candidates = detector.detect(frame, frame_id=1, timestamp_ms=40)

    assert candidates
    assert candidates[0].pixel_x == 60
    assert candidates[0].pixel_y == 70


def test_rejects_ball_colored_candidate_inside_player_box() -> None:
    detector = MotionBallDetector(CourtMapper(np.eye(3)))
    frame = np.zeros((140, 140, 3), dtype=np.uint8)
    cv2.circle(frame, (60, 70), 5, (0, 255, 255), -1)

    candidates = detector.detect(
        frame,
        frame_id=1,
        timestamp_ms=40,
        excluded_boxes=[BoundingBox(x1=50, y1=50, x2=70, y2=90)],
    )

    assert candidates == []


def test_rejects_large_yellow_region() -> None:
    detector = MotionBallDetector(CourtMapper(np.eye(3)))
    frame = np.zeros((140, 140, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (100, 100), (0, 255, 255), -1)

    assert detector.detect(frame, frame_id=1, timestamp_ms=40) == []
