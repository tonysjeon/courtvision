import cv2
import numpy as np
import pytest
from courtvision.geometry.camera_motion import CameraMotionCompensator
from courtvision.geometry.mapper import CourtMapper


def test_aligns_zoomed_frame_to_reference_coordinates() -> None:
    generator = np.random.default_rng(4)
    reference = generator.integers(0, 256, (360, 480), dtype=np.uint8)
    reference = cv2.GaussianBlur(reference, (3, 3), 0)
    reference_to_current = np.array(
        [[1.08, 0, -12], [0, 1.08, 18], [0, 0, 1]],
        dtype=float,
    )
    current = cv2.warpPerspective(reference, reference_to_current, (480, 360))
    compensator = CameraMotionCompensator(
        reference,
        CourtMapper(np.eye(3)),
        minimum_inliers=20,
        smoothing=1,
    )

    mapper = compensator.mapper_for_frame(current)
    current_point = cv2.perspectiveTransform(
        np.float32([[[220, 160]]]),
        reference_to_current,
    )[0, 0]

    assert mapper.transform(*current_point) == pytest.approx((220, 160), abs=1.5)


def test_prefers_colored_court_plane_over_background_features() -> None:
    reference = np.full((360, 480, 3), (100, 145, 120), dtype=np.uint8)
    court = np.float32([[90, 320], [390, 320], [175, 80], [305, 80]])
    cv2.fillConvexPoly(reference, court.astype(np.int32), (158, 122, 109))
    reference_to_current = np.array(
        [[1.06, 0.01, -18], [0, 1.08, 14], [0, 0.00002, 1]],
        dtype=float,
    )
    current = cv2.warpPerspective(reference, reference_to_current, (480, 360))
    normalized = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]])
    reference_mapper = CourtMapper(cv2.getPerspectiveTransform(court, normalized))
    compensator = CameraMotionCompensator(reference, reference_mapper, smoothing=1)

    mapper = compensator.mapper_for_frame(current)
    current_point = cv2.perspectiveTransform(
        np.float32([[[240, 200]]]),
        reference_to_current,
    )[0, 0]

    assert mapper.transform(*current_point) == pytest.approx(
        reference_mapper.transform(240, 200),
        abs=2,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"match_ratio": 1}, "match_ratio"),
        ({"minimum_inliers": 3}, "minimum_inliers"),
        ({"smoothing": 0}, "smoothing"),
    ],
)
def test_rejects_invalid_settings(kwargs: dict[str, float], message: str) -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match=message):
        CameraMotionCompensator(image, CourtMapper(np.eye(3)), **kwargs)
