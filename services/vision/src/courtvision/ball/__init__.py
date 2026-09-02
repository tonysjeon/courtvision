"""Tennis-ball detection."""

from courtvision.ball.detection import BallCandidate, BallDetector
from courtvision.ball.motion import MotionBallDetector
from courtvision.ball.tracknet import TrackNetBallDetector

__all__ = ["BallCandidate", "BallDetector", "MotionBallDetector", "TrackNetBallDetector"]
