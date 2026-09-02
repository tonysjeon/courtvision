import numpy as np
from courtvision.ball.tracknet import _largest_foreground_center


def test_finds_center_of_largest_foreground_region() -> None:
    heatmap = np.zeros((360, 640), dtype=np.uint8)
    heatmap[99:102, 198:203] = 240
    heatmap[20, 20] = 255

    center = _largest_foreground_center(heatmap)

    assert center is not None
    center_x, center_y, radius, peak = center
    assert center_x == 200
    assert center_y == 100
    assert radius > 1
    assert peak == 240


def test_returns_none_without_foreground() -> None:
    heatmap = np.zeros((360, 640), dtype=np.uint8)

    assert _largest_foreground_center(heatmap) is None
