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
