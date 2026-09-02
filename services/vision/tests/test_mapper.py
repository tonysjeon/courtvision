import numpy as np
import pytest
from courtvision.court.calibration import CourtCalibration
from courtvision.geometry.mapper import CourtMapper


@pytest.fixture
def mapper() -> CourtMapper:
    calibration = CourtCalibration.model_validate(
        {
            "near_left_baseline": [100, 700],
            "near_right_baseline": [1100, 700],
            "far_left_baseline": [400, 200],
            "far_right_baseline": [800, 200],
        }
    )
    return CourtMapper.from_calibration(calibration)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ((100, 700), (0, 0)),
        ((1100, 700), (100, 0)),
        ((400, 200), (0, 100)),
        ((800, 200), (100, 100)),
    ],
)
def test_maps_calibration_corners(
    mapper: CourtMapper,
    source: tuple[float, float],
    expected: tuple[float, float],
) -> None:
    assert mapper.transform(*source) == pytest.approx(expected, abs=1e-5)


def test_round_trips_coordinates(mapper: CourtMapper) -> None:
    source = (575.0, 430.0)
    court = mapper.transform(*source)
    assert mapper.inverse_transform(*court) == pytest.approx(source, abs=1e-5)


def test_rejects_singular_matrix() -> None:
    with pytest.raises(ValueError, match="singular"):
        CourtMapper(np.zeros((3, 3)))


def test_rejects_collinear_calibration() -> None:
    calibration = CourtCalibration.model_validate(
        {
            "near_left_baseline": [0, 0],
            "near_right_baseline": [1, 1],
            "far_left_baseline": [2, 2],
            "far_right_baseline": [3, 3],
        }
    )
    with pytest.raises(ValueError, match="usable court area"):
        CourtMapper.from_calibration(calibration)
