from itertools import product

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import genesis as gs
import genesis.utils.geom as gu
from genesis.ext.isaacgym.terrain_utils import SubTerrain, random_uniform_terrain


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_native_spline(n):
    # Share the expensive Genesis fixture across the input-shape/dtype matrix.
    for channels, uneven, scalar_query, input_dtype in product(
        (1, 3), (False, True), (False, True), (np.float32, np.float64)
    ):
        x = np.arange(n, dtype=input_dtype)
        if uneven:
            x = x**1.5
        x -= 1.0
        y = np.stack([np.sin(x + j) for j in range(channels)], axis=1)
        if channels == 1:
            y = y[:, 0]
        xv = (
            np.array(0.2, dtype=input_dtype)
            if scalar_query
            else np.concatenate((x, [x[0] - 0.2, x[-1] + 0.2])).astype(input_dtype)
        )
        result = gu.cubic_spline_1d(x, y, xv)
        expected = CubicSpline(x, y)(np.atleast_1d(xv)).reshape(-1, channels)
        assert result.shape == expected.shape
        tolerance = 1e-10 if gs.np_float == np.float64 and input_dtype == np.float64 else 2e-5
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("n", [0, 1])
def test_insufficient_knots(n):
    with pytest.raises(ValueError, match="At least two knots"):
        gu.cubic_spline_1d(
            np.arange(n, dtype=gs.np_float), np.zeros((n, 2), dtype=gs.np_float), np.array([0.0], dtype=gs.np_float)
        )


@pytest.mark.parametrize("coarse_shape", [(2, 4), (3, 4), (4, 2), (4, 3), (2, 3), (4, 4)])
def test_native_terrain(coarse_shape):
    nx, ny = coarse_shape
    terrain = SubTerrain(width=nx * 2, length=ny * 2, horizontal_scale=0.5, vertical_scale=0.01)
    np.random.seed(5)
    coarse = np.random.choice(np.arange(-10, 11), (nx, ny))
    x, y = np.linspace(0, nx, nx), np.linspace(0, ny, ny)
    xv, yv = np.linspace(0, nx, nx * 2), np.linspace(0, ny, ny * 2)
    expected = np.rint(CubicSpline(x, CubicSpline(y, coarse.T)(yv).T)(xv))
    np.random.seed(5)
    random_uniform_terrain(terrain, -0.1, 0.1, step=0.01, downsampled_scale=1.0)
    assert np.isfinite(terrain.height_field_raw).all()
    assert terrain.height_field_raw.shape == (nx * 2, ny * 2)
    assert terrain.height_field_raw.dtype == gs.np_float
    np.testing.assert_allclose(terrain.height_field_raw, expected, atol=1e-10)
