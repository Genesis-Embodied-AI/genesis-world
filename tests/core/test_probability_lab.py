"""Outcome geometry, experiment accounting, reproducibility and real contact checks."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from genesis.utils.misc import tensor_to_array

from examples.probability.model import Config, Results, State, Trial, classify, perturb_observation, sample_launch
from examples.probability.model import wilson_interval
from examples.probability.world import World, object_xml


@pytest.mark.parametrize("backend", [None])
def test_face_labels_and_edge_rejection():
    quats = Rotation.from_euler("xyz", [[0, 0, 0], [0, -90, 0], [90, 0, 0], [-90, 0, 0],
                                        [0, 90, 0], [180, 0, 0]], degrees=True).as_quat(scalar_first=True)
    np.testing.assert_array_equal(classify(quats, "die"), np.arange(6))
    np.testing.assert_array_equal(classify(quats[[0, 5, 1]], "coin"), [0, 1, -1])
    edge = Rotation.from_euler("x", 45, degrees=True).as_quat(scalar_first=True)
    assert classify(edge, "die")[0] == -1
    with pytest.raises(ValueError):
        classify(np.zeros(4), "coin")


@pytest.mark.parametrize("backend", [None])
def test_launch_seeds_are_independent_of_batching():
    config = Config()
    batch = sample_launch(config, [10, 20, 30])
    solo = sample_launch(config, [20])
    np.testing.assert_array_equal(batch.quaternion[1], solo.quaternion[0])
    np.testing.assert_array_equal(batch.velocity[1], solo.velocity[0])
    np.testing.assert_allclose(np.linalg.norm(batch.quaternion, axis=1), 1)
    np.testing.assert_allclose(np.linalg.norm(batch.velocity[:, 3:], axis=1), config.spin)
    face_up = sample_launch(replace(config, protocol="face_up"), [10, 20])
    np.testing.assert_array_equal(classify(face_up.quaternion, "coin"), [0, 0])


@pytest.mark.parametrize("backend", [None])
def test_zero_uncertainty_repeats_observation_and_does_not_mutate_it():
    config = Config(position_sigma=0, angle_sigma=0, velocity_sigma=0, spin_sigma=0)
    observed = sample_launch(config, [17])
    before = observed.quaternion.copy()
    samples = perturb_observation(config, observed, 99)
    np.testing.assert_allclose(samples.position, np.repeat(observed.position, 64, axis=0))
    np.testing.assert_allclose(samples.velocity, np.repeat(observed.velocity, 64, axis=0))
    np.testing.assert_allclose(samples.quaternion, np.repeat(observed.quaternion, 64, axis=0))
    np.testing.assert_array_equal(observed.quaternion, before)


@pytest.mark.parametrize("backend", [None])
def test_statistics_keep_unresolved_separate():
    trials = [Trial(i, outcome, 1, [], [], [], [], [], []) for i, outcome in enumerate([0, 1, -1, 0])]
    result = Results(Config(), trials)
    np.testing.assert_array_equal(result.counts, [2, 1])
    assert result.unresolved == 1
    assert result.p_value is None
    assert wilson_interval(0, 0) == (0, 1)
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038315, abs=1e-6)
    assert high == pytest.approx(0.5961685, abs=1e-6)
    assert wilson_interval(0, 100)[0] == pytest.approx(0)
    assert wilson_interval(100, 100)[1] == pytest.approx(1)


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("field,value", [("seed", -1), ("seed", 2**32), ("dt", 0),
                                       ("height", np.nan), ("spin", -1), ("forecast_samples", 1)])
def test_invalid_experiment_parameters(field, value):
    with pytest.raises(ValueError):
        replace(Config(), **{field: value})


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("kind", ["coin", "die"])
def test_markings_cannot_change_mass_or_collisions(kind):
    root = object_xml(kind)
    body = root.find("worldbody/body")
    assert body.find("inertial") is not None
    geoms = body.findall("geom")
    assert len([g for g in geoms if g.get("contype") != "0"]) == 1
    for geom in geoms[1:]:
        assert geom.get("mass") == "0"
        assert geom.get("contype") == geom.get("conaffinity") == "0"


@pytest.mark.parametrize("kind", ["coin", "die"])
def test_known_faces_drop_and_settle(kind, show_viewer):
    """Eighty steps include free fall, contact and the 0.25 s rest confirmation."""
    angles = [[0, 0, 0], [180, 0, 0]] if kind == "coin" else [
        [0, 0, 0], [0, -90, 0], [90, 0, 0], [-90, 0, 0], [0, 90, 0], [180, 0, 0],
    ]
    n = len(angles)
    state = State(np.tile([0., 0., .08], (n, 1)),
                  Rotation.from_euler("xyz", angles, degrees=True).as_quat(scalar_first=True), np.zeros((n, 6)))
    world = World(Config(kind=kind), n, render=show_viewer)
    try:
        world.reset(state)
        # Check the public setters/getters before integrating; a wrong root link would fail here.
        np.testing.assert_allclose(world.state().position, state.position, atol=1e-6)
        np.testing.assert_allclose(tensor_to_array(world.body.get_mass()), .02 if kind == "coin" else .08)
        for _ in range(80):
            resting, faces = world.step()
            if show_viewer:
                world.image()
        assert resting.all()
        assert world.elapsed == 0.8
        np.testing.assert_array_equal(faces, np.arange(n))
        final = world.state()
        assert np.isfinite(final.velocity).all()
        expected_height = .004 if kind == "coin" else .04
        np.testing.assert_allclose(final.position[:, 2], expected_height, atol=.001)
    finally:
        world.close()
