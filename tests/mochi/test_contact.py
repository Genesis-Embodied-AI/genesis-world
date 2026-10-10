import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose
from .utils import make_static_config, plane_collision_response

# (penalty_coefficient, half_distance, threshold, friction, falloff_vel, viscous_friction, normal_damping, max_align)
CONTACT_PARAMS = {
    "coulomb": (1e9, 5e-3, 1e-3, 0.5, 1e-2, 0.0, 0.0, 0.0),
    "viscous": (1e9, 5e-3, 1e-3, 0.0, 1e-2, 100.0, 0.0, 0.0),
    "damping": (1e9, 5e-3, 1e-3, 0.3, 1e-2, 0.0, 50.0, 0.0),
}


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("params_name", ["coulomb", "viscous", "damping"])
@pytest.mark.parametrize("friction_model", ["c1", "cinf"])
def test_collision_response_is_gradient_of_potential(params_name, friction_model):
    from genesis.engine.solvers.mochi.data import FRICTION_MODEL

    params = CONTACT_PARAMS[params_name]
    dt_stage = 0.01
    rng = np.random.default_rng(0)
    normal = np.array([0.3, -0.2, 0.93])
    normal /= np.linalg.norm(normal)
    colliding_normal = -normal
    # Samples over the whole penalty ramp, displaced from their stage-start position along and across the plane.
    distances = np.array([0.003, -0.001, 0.002, -0.005, -0.012, 0.0005])
    points = distances[:, None] * normal[None, :] + rng.normal(scale=0.01, size=(len(distances), 3)) * (1 - normal)
    points_stage_start = points - rng.normal(scale=0.5 * dt_stage, size=points.shape)
    for k, sliding in enumerate((1e-6, 1e-4, 3e-3)):
        points_stage_start[k] = points[k] - sliding * np.array([0.0, 1.0, 0.0])

    for use_fitted in (True, False):
        config = make_static_config(
            friction_model=FRICTION_MODEL.CINF if friction_model == "cinf" else FRICTION_MODEL.C1,
            use_fitted_friction_hessian=use_fitted,
        )
        energy, force, dforce = plane_collision_response(
            points, points_stage_start, normal, colliding_normal, params, dt_stage, config
        )
        assert np.all(energy >= 0.0)
        assert energy[distances > 0.001].max() == 0.0

        # Central finite differences of the energy and of the force with respect to the sample position. The stage
        # displacement moves with the sample, so the dissipation terms are differentiated too.
        eps = 1e-7
        force_fd = np.zeros_like(force)
        dforce_fd = np.zeros_like(dforce)
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = eps
            energy_p, force_p, _ = plane_collision_response(
                points + delta, points_stage_start, normal, colliding_normal, params, dt_stage, config
            )
            energy_m, force_m, _ = plane_collision_response(
                points - delta, points_stage_start, normal, colliding_normal, params, dt_stage, config
            )
            force_fd[:, axis] = -(energy_p - energy_m) / (2 * eps)
            dforce_fd[:, :, axis] = (force_p - force_m) / (2 * eps)
        scale = np.abs(force).max()
        assert_allclose(force, force_fd, atol=1e-5 * scale, rtol=1e-5)
        if not use_fitted:
            # The exact derivative is checked away from the stick-slip regularization width, where the C1 model has a
            # kink in the second derivative.
            hess_scale = np.abs(dforce).max()
            assert_allclose(dforce, dforce_fd, atol=1e-4 * hess_scale, rtol=1e-4)
        # Symmetric and negative semi-definite by construction.
        assert_allclose(dforce, np.swapaxes(dforce, 1, 2), tol=1e-9 * max(np.abs(dforce).max(), 1.0))
        assert np.all(np.linalg.eigvalsh(dforce) <= 1e-6 * max(np.abs(dforce).max(), 1.0))


@pytest.mark.required
@pytest.mark.precision("64")
def test_collider_distance_fields(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        show_viewer=show_viewer,
    )
    center = (0.1, 0.2, 0.3)
    box = scene.add_entity(
        gs.morphs.Box(size=(1.0, 1.0, 1.0), pos=center, fixed=True), material=gs.materials.Mochi.Rigid()
    )
    grid_box = scene.add_entity(
        gs.morphs.Box(size=(1.0, 1.0, 1.0), pos=(center[0], center[1] + 5.0, center[2]), fixed=True),
        material=gs.materials.Mochi.Rigid(collider_type="sdf", sdf_cell_size=0.02),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=1.0, pos=(center[0], center[1] - 5.0, center[2]), fixed=True),
        material=gs.materials.Mochi.Rigid(),
    )
    plane = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, 1.0)), material=gs.materials.Mochi.Rigid())
    scene.build()
    solver = scene.mochi_solver

    # Probes along x through the cube center: outside, inside near the face, inside, outside, far outside.
    x_offsets = np.array([-0.6, 0.3, -0.4, 0.9, 1.4])
    expected = np.array([0.1, -0.2, -0.1, 0.4, 0.9])
    expected_grad = np.array([[-1, 0, 0], [1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    points = np.array(box.get_pos().cpu().numpy())[None, :] + x_offsets[:, None] * np.array([[1.0, 0.0, 0.0]])
    distances, gradients, is_valid = solver.get_collider_distances(box.geoms[0].idx, points)
    assert bool(is_valid.all())
    assert_allclose(distances, expected, tol=1e-9)
    assert_allclose(gradients, expected_grad, tol=1e-9)
    # The grid only answers inside its padded extent (10% of the size beyond the surface); a far probe is invalid.
    grid_offsets = np.array([0.3, -0.4, -0.55, 0.55, 1.4])
    grid_expected = np.array([-0.2, -0.1, 0.05, 0.05, 0.9])
    grid_expected_grad = np.array([[1, 0, 0], [-1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    points = np.array(grid_box.get_pos().cpu().numpy())[None, :] + grid_offsets[:, None] * np.array([[1.0, 0.0, 0.0]])
    distances, gradients, is_valid = solver.get_collider_distances(grid_box.geoms[0].idx, points)
    assert_allclose(tensor_to_array(is_valid), [True, True, True, True, False], tol=0.0)
    assert_allclose(distances[:4], grid_expected[:4], atol=5e-4, rtol=0.0)
    assert_allclose(gradients[:4], grid_expected_grad[:4], atol=5e-2, rtol=0.0)

    points = np.array(sphere.get_pos().cpu().numpy())[None, :] + np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    distances, gradients, is_valid = solver.get_collider_distances(sphere.geoms[0].idx, points)
    assert_allclose(distances, [-1.0, 0.0, 1.0, np.sqrt(2.0) - 1.0], tol=1e-9)
    assert_allclose(np.linalg.norm(gradients[1:].cpu().numpy(), axis=-1), 1.0, tol=1e-9)

    points = np.array([[0.0, 0.0, -1.0], [3.0, -2.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    distances, gradients, is_valid = solver.get_collider_distances(plane.geoms[0].idx, points)
    assert_allclose(distances, [-2.0, -1.0, 0.0, 1.0], tol=1e-9)
    assert_allclose(gradients, np.tile([[0.0, 0.0, 1.0]], (4, 1)), tol=1e-9)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("params_name", ["coulomb", "viscous", "damping"])
@pytest.mark.parametrize("friction_model", ["c1", "cinf"])
def test_collision_response_tangent_is_symmetric(params_name, friction_model):
    # The contact force derivative is built from grad grad^T, I - n n^T and n n^T terms only, so it is symmetric by
    # construction: the solver stores it as six floats (see func_mat3_to_sym6), which is exact only as long as every
    # response path keeps the symmetry.
    from genesis.engine.solvers.mochi.data import FRICTION_MODEL

    params = CONTACT_PARAMS[params_name]
    dt_stage = 0.01
    rng = np.random.default_rng(1)
    normal = np.array([0.1, 0.4, 0.911])
    normal /= np.linalg.norm(normal)
    distances = np.array([0.003, -0.001, 0.002, -0.005, -0.012, 0.0005])
    points = distances[:, None] * normal[None, :] + rng.normal(scale=0.01, size=(len(distances), 3)) * (1 - normal)
    points_stage_start = points - rng.normal(scale=0.5 * dt_stage, size=points.shape)
    for use_fitted in (True, False):
        for implicit_normal in (True, False):
            config = make_static_config(
                friction_model=FRICTION_MODEL.CINF if friction_model == "cinf" else FRICTION_MODEL.C1,
                use_fitted_friction_hessian=use_fitted,
                implicit_normal_force_for_dissipation=implicit_normal,
            )
            _, _, dforce = plane_collision_response(
                points, points_stage_start, normal, -normal, params, dt_stage, config
            )
            assert_allclose(dforce, np.swapaxes(dforce, -1, -2), tol=1e-12)
