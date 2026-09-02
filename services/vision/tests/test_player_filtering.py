import numpy as np
import pytest
from courtvision.court.calibration import CourtCalibration
from courtvision.geometry.mapper import CourtMapper
from courtvision.players.detection import BoundingBox, PersonDetection
from courtvision.players.filtering import ActivePlayerSelector


def detection(x: float, ground_y: float, confidence: float = 0.9) -> PersonDetection:
    return PersonDetection(
        frame_id=1,
        timestamp_ms=40,
        confidence=confidence,
        bbox=BoundingBox(x1=x - 5, y1=max(0, ground_y - 10), x2=x + 5, y2=ground_y),
    )


@pytest.fixture
def selector() -> ActivePlayerSelector:
    return ActivePlayerSelector(court_margin=20)


def test_selects_near_and_far_players_and_rejects_spectator(
    selector: ActivePlayerSelector,
) -> None:
    mapper = CourtMapper.from_calibration(
        CourtCalibration.model_validate(
            {
                "near_left_baseline": [0, 0],
                "near_right_baseline": [100, 0],
                "far_left_baseline": [0, 100],
                "far_right_baseline": [100, 100],
            }
        )
    )

    selected = selector.select(
        [detection(45, 5), detection(55, 95), detection(180, 40)],
        mapper,
    )

    assert selected["near_player"].court_y == pytest.approx(5)
    assert selected["far_player"].court_y == pytest.approx(95)


def test_prefers_previous_position(selector: ActivePlayerSelector) -> None:
    mapper = CourtMapper(np.eye(3))

    selected = selector.select(
        [detection(20, 10, 0.99), detection(75, 15, 0.7)],
        mapper,
        {"near_player": (78, 14)},
    )

    assert selected["near_player"].court_x == pytest.approx(75)
