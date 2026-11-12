"""Independent geometric, accounting, and input checks for matched coin/die replays."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.probability.compare_real import (
    Coin, Die, Rules, align_track, bootstrap_mean, clearance, object_xml, metrics,
    orientation_error_deg, read_manifest, read_tracks, shape_from_manifest,
    support_height, validate_track, wilson, write_tracks,
)

pytestmark = pytest.mark.parametrize("backend", [None])


def stationary_track(angle=0):
    t = np.arange(0, 0.61, 0.01)
    x = np.zeros((len(t), 13))
    x[:, 3:7] = Rotation.from_euler("x", angle, degrees=True).as_quat(scalar_first=True)
    x[:, 2] = 0.045 if angle == 90 else 0.004
    return t, x


def test_rotation_distance_sign_and_known_angle():
    identity = np.array([1.0, 0, 0, 0])
    quarter_turn = np.array([np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    assert orientation_error_deg(identity, -identity) == pytest.approx(0)
    assert orientation_error_deg(identity, quarter_turn) == pytest.approx(90)
    assert orientation_error_deg(identity, 2 * quarter_turn) == pytest.approx(90)
    assert orientation_error_deg(quarter_turn.astype(np.float32), quarter_turn) < 1e-5


def test_interpolation_uses_rotation_geodesic_and_refuses_extrapolation():
    t, x = stationary_track()
    x[:, 0] = 2 * t
    x[:, 3:7] = Rotation.from_euler("z", 100 * t[:, None], degrees=True).as_quat(scalar_first=True)
    x[1::2, 3:7] *= -1
    target = np.array([0.005, 0.145, 0.595])
    result = align_track(t, x, target)
    np.testing.assert_allclose(result[:, 0], 2 * target)
    np.testing.assert_allclose(orientation_error_deg(np.array([1, 0, 0, 0]), result[:, 3:7]), target * 100)
    with pytest.raises(ValueError, match="Extrapolation"):
        align_track(t, x, np.array([0, 0.7]))


@pytest.mark.parametrize("angle,outcome", [(0, "heads"), (180, "tails"), (90, "edge")])
def test_confirmed_rest_classifies_heads_tails_and_edge(angle, outcome):
    t, x = stationary_track(angle)
    result = metrics(t, x, Coin(0.02, 0.045, 0.008), Rules())
    assert result["outcome"] == outcome
    assert result["settling_time_s"] == 0
    assert result["bounce_count"] == 0


def test_airborne_apex_and_rolling_edge_are_not_rest():
    t, x = stationary_track()
    x[:, 2] = 0.2
    assert metrics(t, x, Coin(0.02, 0.045, 0.008), Rules())["settling_time_s"] is None
    t, x = stationary_track(90)
    x[:, 10] = 2
    result = metrics(t, x, Coin(0.02, 0.045, 0.008), Rules())
    assert result["outcome"] == "unresolved"
    assert result["settling_time_s"] is None


def test_late_movement_invalidates_earlier_quiet_period():
    t, x = stationary_track()
    x[-1, 7] = 1
    assert metrics(t, x, Coin(0.02, 0.045, 0.008), Rules())["settling_time_s"] is None


def test_bounce_count_excludes_release_and_reports_incomplete_rebound():
    t, x = stationary_track()
    x[:, 7] = 1  # Keep classification separate from flight detection.
    x[:10, 2] = 0.1  # Initial flight is not a rebound.
    x[15:25, 2] = 0.034  # First completed rebound: COM apex 34 mm.
    x[30:40, 2] = 0.014  # Second completed rebound: COM apex 14 mm.
    x[50:, 2] = 0.02  # Third flight has not landed within the observation window.
    result = metrics(t, x, Coin(0.02, 0.045, 0.008), Rules())
    assert result["bounce_count"] == 2
    assert result["bounce_heights_m"] == pytest.approx([0.034, 0.014])
    assert result["bounce_clearances_m"] == pytest.approx([0.03, 0.01])
    assert result["incomplete_rebound"]


def test_tiny_contact_jitter_does_not_count_as_bounce():
    t, x = stationary_track()
    x[10:15, 2] = 0.0055  # 1.5 mm clearance, below the 2 mm bounce threshold.
    assert metrics(t, x, Coin(0.02, 0.045, 0.008), Rules())["bounce_count"] == 0


def test_cylinder_support_at_flat_edge_and_tilt():
    t, x = stationary_track()
    x[:3, 3:7] = Rotation.from_euler("x", [[0], [90], [45]], degrees=True).as_quat(scalar_first=True)
    x[:3, 2] = [0.004, 0.045, (0.045 + 0.004) / np.sqrt(2)]
    np.testing.assert_allclose(clearance(x[:3], Coin(0.02, 0.045, 0.008))[0], 0, atol=1e-9)


@pytest.mark.parametrize("problem", ["duplicate_time", "nan", "bad_quaternion", "wrong_shape"])
def test_bad_tracks_are_rejected(problem):
    t, x = stationary_track()
    if problem == "duplicate_time":
        t[2] = t[1]
    elif problem == "nan":
        x[3, 1] = np.nan
    elif problem == "bad_quaternion":
        x[3, 3:7] = 0
    else:
        x = x[:, :7]
    with pytest.raises(ValueError):
        validate_track(t, x)


def test_csv_round_trip_and_no_silent_sorting(tmp_path):
    t, x = stationary_track()
    path = tmp_path / "tracks.csv"
    write_tracks(path, {"independent_1": (t, x)})
    read_t, read_x = read_tracks(path)["independent_1"]
    np.testing.assert_allclose(read_t, t)
    np.testing.assert_allclose(read_x, x)
    t[1], t[2] = t[2], t[1]
    write_tracks(path, {"independent_1": (t, x)})
    with pytest.raises(ValueError, match="strictly increasing"):
        read_tracks(path)


def test_intervals_keep_zero_counts_and_singleton_uncertainty():
    low, high = wilson(50, 100)
    assert low == pytest.approx(0.40383153)
    assert high == pytest.approx(0.59616847)
    assert wilson(0, 100)[1] > 0
    mean, low, high = bootstrap_mean([4.0])
    assert mean == 4
    assert np.isnan(low) and np.isnan(high)
    mean, low, high = bootstrap_mean(np.array([[1, 2], [1, 2]]))
    np.testing.assert_allclose(mean, [1, 2])
    np.testing.assert_allclose(low, high)


def test_coin_inertia_uses_measured_geometry():
    coin = Coin(0.02, 0.045, 0.008)
    assert coin.inertia == pytest.approx([0.0000102316666667, 0.0000102316666667, 0.00002025])
    root = object_xml(coin)
    assert float(root.find("worldbody/body/inertial").get("mass")) == 0.02
    assert root.find("worldbody/body/geom").get("size") == "0.045 0.004"
    with pytest.raises(ValueError):
        Coin(0, 0.045, 0.008)


def test_manifest_requires_explicit_data_provenance(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"coin": {"mass_kg": 0.02, "radius_m": 0.045, "thickness_m": 0.008}}))
    with pytest.raises(ValueError, match="data_kind"):
        read_manifest(path)


@pytest.mark.parametrize("angles,face", [([0, 0, 0], "1"), ([0, -90, 0], "2"),
    ([90, 0, 0], "3"), ([-90, 0, 0], "4"), ([0, 90, 0], "5"), ([180, 0, 0], "6")])
def test_all_die_faces_at_rest(angles, face):
    t, x = stationary_track()
    x[:, 2] = 0.04
    x[:, 3:7] = Rotation.from_euler("xyz", angles, degrees=True).as_quat(scalar_first=True)
    result = metrics(t, x, Die(0.08, 0.08), Rules())
    assert result["outcome"] == face
    assert result["settling_time_s"] == 0
    assert result["bounce_count"] == 0


def test_cube_support_at_face_edge_and_corner_and_ambiguous_rest():
    angles = [[0, 0, 0], [45, 0, 0], [45, np.degrees(np.arcsin(1 / np.sqrt(3))), 0]]
    q = Rotation.from_euler("xyz", angles, degrees=True).as_quat(scalar_first=True)
    height, _ = support_height(q, Die(0.08, 0.08))
    np.testing.assert_allclose(height, 0.04 * np.array([1, np.sqrt(2), np.sqrt(3)]), atol=1e-10)
    t, x = stationary_track()
    x[:, 3:7], x[:, 2] = q[1], height[1]
    result = metrics(t, x, Die(0.08, 0.08), Rules())
    assert result["outcome"] == "unresolved"  # No face confidently points up.
    assert result["settling_time_s"] == 0
    x[:, 10] = 3
    assert metrics(t, x, Die(0.08, 0.08), Rules())["settling_time_s"] is None


def test_die_geometry_inertia_and_manifest_dispatch():
    die = shape_from_manifest({"kind": "die", "die": {"mass_kg": 0.08, "edge_m": 0.08}})
    assert isinstance(die, Die)
    assert die.inertia == pytest.approx([0.0000853333333333] * 3)
    geom = object_xml(die).find("worldbody/body/geom")
    assert geom.get("type") == "box"
    assert geom.get("size") == "0.04 0.04 0.04"
    assert len(die.outcomes) == 7
    assert isinstance(shape_from_manifest({"coin": {"mass_kg": 0.02, "radius_m": 0.045, "thickness_m": 0.008}}), Coin)
    with pytest.raises(ValueError):
        Die(0.08, -0.08)
    with pytest.raises(ValueError, match="kind"):
        shape_from_manifest({"kind": "sphere"})
