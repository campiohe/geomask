import numpy as np
import pytest
from shapely import Polygon

from src.masker import Masker


@pytest.mark.parametrize(
    "polygon,resolution,expected_mask",
    [
        # Simple square polygon with resolution 0.5
        ([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], 0.5, [[True, True], [True, True]]),
        # Smaller resolution leading to larger grid
        (
            [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
            0.25,
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
            ],
        ),
        # Larger resolution with partial inclusion
        ([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], 1, [[True]]),
        # Triangle with small resolution
        (
            [(0, 0), (1, 0), (0.5, 1), (0, 0)],
            0.25,
            [
                [True, True, True, True],
                [False, True, True, True],
                [False, True, True, True],
                [False, False, True, False],
            ],
        ),
        # Complex polygon (irregular shape)
        (
            [(0, 0), (2, 0), (1, 2), (0, 0)],
            0.5,
            [
                [True, True, True, True],
                [False, True, True, True],
                [False, True, True, True],
                [False, False, True, False],
            ],
        ),
        # Concave polygon with higher resolution
        (
            [(0, 0), (2, 0), (1, 2), (0, 1), (0, 0)],
            0.25,
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, False],
                [True, True, True, True, True, True, True, False],
                [False, True, True, True, True, True, False, False],
                [False, False, True, True, True, True, False, False],
                [False, False, False, True, True, False, False, False],
            ],
        ),
    ],
)
def test_masker(polygon, resolution, expected_mask):
    ds = Masker.mask(Polygon(polygon), resolution)
    assert np.all(np.array(expected_mask, dtype=np.bool_) == ds.mask.values)
