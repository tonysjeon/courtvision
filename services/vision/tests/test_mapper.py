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


def test_scales_calibration_to_current_frame_dimensions() -> None:
    calibration = CourtCalibration.model_validate(
        {
            "frame_width": 1000,
            "frame_height": 500,
            "keypoints": {
                "near_left_baseline": [100, 400],
                "near_right_baseline": [900, 400],
                "far_left_baseline": [300, 100],
                "far_right_baseline": [700, 100],
            },
        }
    )

    mapper = CourtMapper.from_calibration(calibration, frame_width=2000, frame_height=1000)

    assert mapper.transform(200, 800) == pytest.approx((0, 0), abs=1e-5)
    assert mapper.transform(1400, 200) == pytest.approx((100, 100), abs=1e-5)


def test_composes_current_frame_alignment_before_court_mapping() -> None:
    mapper = CourtMapper(np.eye(3)).after_image_transform(
        np.array([[1, 0, -20], [0, 1, 10], [0, 0, 1]], dtype=float)
    )

    assert mapper.transform(30, 40) == pytest.approx((10, 50))
