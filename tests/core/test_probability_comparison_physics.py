"""Small actual-Genesis checks for replay frames, contact settings and known faces."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.probability.compare_real import Coin, Die, Rules, metrics, orientation_error_deg, run_genesis


PROFILE = {"name": "test_contact", "friction": 0.6, "timeconst_s": 0.02,
           "dampratio": 1.0, "rolling_m": 0.0001, "torsional_m": 0.005}


@pytest.mark.parametrize("shape", [Coin(0.02, 0.045, 0.008), Die(0.08, 0.08)])
def test_non_aligned_release_frames_and_ballistic_motion(shape):
    initial = np.zeros((2, 13))
    initial[:, :3] = [[0, 0, 0.4], [0.1, 0, 0.5]]
    initial[:, 3:7] = Rotation.from_euler("xyz", [[30, 25, 70], [-20, 40, -60]], degrees=True).as_quat(scalar_first=True)
    initial[:, 7:10] = [[0.1, 0.2, 0.8], [-0.2, 0.1, 0.7]]
    initial[:, 10:] = [[1, 2, 3], [-2, 5, 1]]
    t, x, metadata = run_genesis(initial, shape, PROFILE, 0.001, 2, 0.04, 0.005, 11, "cpu")
    # Test harness initializes Genesis on the requested --backend; the fallback
    # argument above is unused when that backend is already initialized.
    np.testing.assert_allclose(x[:, 0, :3], initial[:, :3], atol=1e-6)
    np.testing.assert_allclose(x[:, 0, 7:], initial[:, 7:], atol=1e-5)
    assert np.max(orientation_error_deg(x[:, 0, 3:7], initial[:, 3:7])) < 0.001
    assert np.isfinite(x).all()
    expected = initial[:, None, :3] + initial[:, None, 7:10] * t[None, :, None]
    expected[:, :, 2] -= 0.5 * 9.81 * t[None, :] ** 2
    # First-order integration error at a 0.5 ms substep is ~0.10 mm by t=40 ms.
    np.testing.assert_allclose(x[:, :, :3], expected, atol=0.0002, rtol=0)
    assert metadata["mass_kg"] == pytest.approx(shape.mass_kg, abs=1e-7)


@pytest.mark.parametrize("shape", [Coin(0.02, 0.045, 0.008), Die(0.08, 0.08)])
def test_contact_replay_retains_known_faces_and_seed_reproducibility(shape):
    angles = [[0, 0, 0], [180, 0, 0]] if shape.kind == "coin" else [
        [0, 0, 0], [0, -90, 0], [90, 0, 0], [-90, 0, 0], [0, 90, 0], [180, 0, 0],
    ]
    initial = np.zeros((len(angles), 13))
    initial[:, 2] = shape.half_height + 0.08
    initial[:, 3:7] = Rotation.from_euler("xyz", angles, degrees=True).as_quat(scalar_first=True)
    t, x, metadata = run_genesis(initial, shape, PROFILE, 0.002, 2, 0.8, 0.004, 11, "cpu")
    assert np.isfinite(x).all()
    results = [metrics(t, trial, shape, Rules()) for trial in x]
    expected = ["heads", "tails"] if shape.kind == "coin" else list("123456")
    assert [m["outcome"] for m in results] == expected
    assert all(m["settling_time_s"] is not None for m in results)
    np.testing.assert_allclose(x[:, -1, 2], shape.half_height, atol=0.001)
    for key in (f"{shape.kind}_sol_params", "floor_sol_params"):
        np.testing.assert_allclose(metadata[key], [0.02, 1, 0.9, 0.95, 0.001, 0.5, 2], rtol=1e-6)
    t2, x2, _ = run_genesis(initial, shape, PROFILE, 0.002, 2, 0.8, 0.004, 22, "cpu")
    np.testing.assert_array_equal(t, t2)
    # Seeds do not perturb the measured releases. Allow backend numeric variation.
    np.testing.assert_allclose(x, x2, atol=1e-5, rtol=1e-5)
