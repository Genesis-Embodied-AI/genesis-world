import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose
from .reference import load_reference, mochi_to_genesis


def _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -10.0), **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
        mochi_options=gs.options.MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


@pytest.mark.required
@pytest.mark.precision("64")
def test_free_fall_impact_and_rest_match_mochi(show_viewer):
    reference = load_reference("box_drop_plane")
    ref_pos = mochi_to_genesis(reference["pos"])
    ref_vel = mochi_to_genesis(reference["vel"])
    ref_force = mochi_to_genesis(reference["contact_force"])
    dt = float(reference["dt"])
    gravity = float(reference["gravity"])

    scene = _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -gravity))
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    box = scene.add_entity(
        gs.morphs.Box(size=(1.0, 1.0, 1.0), pos=(0.0, 0.0, 1.0)), material=gs.materials.Mochi.Rigid()
    )
    scene.build()

    pos, vel, force = [], [], []
    for _ in range(len(ref_pos)):
        scene.step()
        pos.append(tensor_to_array(box.get_pos()))
        vel.append(tensor_to_array(box.get_vel()))
        force.append(tensor_to_array(box.get_links_net_contact_force())[0])
    pos, vel, force = np.array(pos), np.array(vel), np.array(force)

    # Backward Euler free fall: the velocity is integrated first, so z_1 = z_0 - g dt^2.
    assert_allclose(pos[0], [0.0, 0.0, 1.0 - gravity * dt**2], tol=1e-9)
    # Resting contact: the box floats a fraction of the smoothing ramp inside the penalty threshold, its weight is
    # carried by the contact force, and it is at rest.
    mass = box.get_mass()
    assert -3e-3 < pos[-1, 2] - 0.5 < 1e-3
    assert_allclose(force[-1], [0.0, 0.0, mass * gravity], rtol=1e-2, atol=1e-6)
    assert_allclose(vel[-1], 0.0, atol=1e-6)
    contacts = box.get_contacts()
    assert len(contacts["distance"]) == 6
    assert_allclose(contacts["normal"], np.tile([[0.0, 0.0, 1.0]], (6, 1)), tol=1e-9)
    assert bool((contacts["distance"] < 0.0).all())
    assert_allclose(contacts["force_a"].sum(dim=0), force[-1], tol=1e-9)
    # Parity with the original engine through impact and rest.
    assert_allclose(pos, ref_pos, atol=1e-8, rtol=0.0)
    assert_allclose(vel, ref_vel, atol=1e-6, rtol=0.0)
    assert_allclose(force, ref_force, atol=1e-3, rtol=1e-6)


@pytest.mark.required
@pytest.mark.precision("64")
def test_stacking_matches_mochi(show_viewer):
    reference = load_reference("two_boxes_stack")
    dt = float(reference["dt"])
    scene = _mochi_scene(show_viewer, dt)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    bottom = scene.add_entity(
        gs.morphs.Box(size=(1.0, 1.0, 1.0), pos=(0.0, 0.0, 0.5)), material=gs.materials.Mochi.Rigid()
    )
    top = scene.add_entity(
        gs.morphs.Box(size=(0.5, 0.5, 0.5), pos=(0.0, 0.0, 1.3)), material=gs.materials.Mochi.Rigid()
    )
    scene.build()

    pos_bottom, pos_top = [], []
    for _ in range(len(reference["pos_top"])):
        scene.step()
        pos_bottom.append(tensor_to_array(bottom.get_pos()))
        pos_top.append(tensor_to_array(top.get_pos()))
    assert_allclose(pos_bottom, mochi_to_genesis(reference["pos_bottom"]), atol=1e-8, rtol=0.0)
    assert_allclose(pos_top, mochi_to_genesis(reference["pos_top"]), atol=1e-8, rtol=0.0)

    # The interface between the boxes sits within the penalty threshold of the nominal height and each body carries
    # the weight above it.
    assert abs(pos_top[-1][2] - 0.25 - (pos_bottom[-1][2] + 0.5)) < 3e-3
    assert_allclose(top.get_links_net_contact_force()[0], [0.0, 0.0, top.get_mass() * 10.0], rtol=1e-2, atol=1e-6)
    assert_allclose(bottom.get_links_net_contact_force()[0], [0.0, 0.0, bottom.get_mass() * 10.0], rtol=1e-2, atol=1e-6)
    contacts = top.get_contacts(with_entity=bottom)
    assert len(contacts["distance"]) >= 4
    assert_allclose(contacts["force_a"].sum(dim=0), [0.0, 0.0, top.get_mass() * 10.0], rtol=1e-2, atol=1e-6)
    assert_allclose(top.get_vel(), 0.0, atol=1e-6)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("friction_model", ["c1", "cinf"])
def test_coulomb_friction_slide_and_stick(friction_model, show_viewer):
    reference = load_reference("box_slide_friction")
    dt = float(reference["dt"])
    mu = 0.5
    scene = _mochi_scene(show_viewer, dt, n_newton_iterations=12, friction_model=friction_model)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid(friction=mu))
    box = scene.add_entity(
        gs.morphs.Box(size=(0.4, 0.4, 0.05), pos=(0.0, 0.0, 0.025)), material=gs.materials.Mochi.Rigid(friction=mu)
    )
    scene.build()
    for _ in range(20):
        scene.step()
    box.set_dofs_velocity([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pos, vel = [], []
    for _ in range(len(reference["vel"])):
        scene.step()
        pos.append(tensor_to_array(box.get_pos()))
        vel.append(tensor_to_array(box.get_vel()))
    pos, vel = np.array(pos), np.array(vel)

    # Sliding decelerates at mu g and the box stops after v0^2 / (2 mu g) = 0.4 m, then sticks.
    times = dt * np.arange(1, 30)
    assert_allclose(vel[:29, 0], 2.0 - mu * 10.0 * times, rtol=2e-3, atol=2e-3)
    assert_allclose(vel[-1], 0.0, atol=1e-6)
    assert abs(pos[-1, 0] - 0.4) < 0.02
    assert_allclose(box.get_ang(), 0.0, atol=1e-6)
    if friction_model == "c1":
        assert_allclose(pos, mochi_to_genesis(reference["pos"]), atol=1e-8, rtol=0.0)
        assert_allclose(vel, mochi_to_genesis(reference["vel"]), atol=1e-7, rtol=0.0)


@pytest.mark.required
@pytest.mark.precision("64")
def test_batched_envs_are_independent(show_viewer):
    n_envs = 4
    scene = _mochi_scene(show_viewer, 0.02)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    box = scene.add_entity(
        gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.6)), material=gs.materials.Mochi.Rigid()
    )
    scene.build(n_envs=n_envs)
    # Different drop heights per environment; the other environments must not see the change.
    heights = 0.6 + 0.1 * np.arange(n_envs)
    box.set_pos(np.stack([np.zeros(n_envs), np.zeros(n_envs), heights], axis=-1))
    for _ in range(50):
        scene.step()
    pos = tensor_to_array(box.get_pos())
    assert pos.shape == (n_envs, 3)
    # Every environment ends at rest on the plane, at the same height whatever the drop height.
    assert_allclose(pos[:, 2], pos[0, 2], tol=1e-6)
    assert -3e-3 < pos[0, 2] - 0.2 < 1e-3
    assert_allclose(box.get_vel(), 0.0, atol=1e-6)

    # A pose set in one environment only affects that environment.
    box.set_pos(np.array([[0.0, 0.0, 1.0]]), envs_idx=[1])
    scene.step()
    pos = tensor_to_array(box.get_pos())
    assert pos[1, 2] > 0.9
    assert_allclose(pos[[0, 2, 3], 2], pos[0, 2], tol=1e-6)

    contacts = box.get_contacts(is_padded=True)
    assert contacts["valid_mask"].shape[0] == n_envs
    n_contacts = contacts["valid_mask"].sum(dim=1)
    assert n_contacts[1] == 0 and bool((n_contacts[[0, 2, 3]] == 6).all())


@pytest.mark.precision("64")
@pytest.mark.parametrize("integrator", ["backward_euler", "bdf2"])
@pytest.mark.parametrize("use_newton_euler_inertia", [False, True])
def test_free_rotation_angular_momentum(integrator, use_newton_euler_inertia, show_viewer):
    dt = 0.01
    scene = _mochi_scene(
        show_viewer,
        dt,
        gravity=(0.0, 0.0, 0.0),
        integrator=integrator,
        use_newton_euler_inertia=use_newton_euler_inertia,
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.4, 0.6), pos=(0.0, 0.0, 1.0)), material=gs.materials.Mochi.Rigid(has_gravity=False)
    )
    scene.build()
    box.set_dofs_velocity([0.3, 0.0, 0.0, 0.0, 0.0, 4.0])
    for _ in range(100):
        scene.step()
    # Uniform translation is exact; the spin about a principal axis is exact for the Newton-Euler inertia while the
    # variational merit dissipates a fraction (omega dt)^2 of the spin per backward Euler step, much less with BDF2.
    assert_allclose(box.get_pos(), [0.3, 0.0, 1.0], tol=1e-9)
    ang = tensor_to_array(box.get_ang())
    assert_allclose(ang[:2], 0.0, atol=1e-9)
    if use_newton_euler_inertia:
        assert_allclose(ang[2], 4.0, rtol=1e-4, atol=0.0)
    elif integrator == "bdf2":
        assert 3.5 < ang[2] < 4.0 + 1e-9
    else:
        assert 3.3 < ang[2] < 3.6


@pytest.mark.precision("64")
def test_grid_collider_and_pcg_rest(show_viewer):
    scene = _mochi_scene(show_viewer, 0.01, linear_solver="pcg", n_newton_iterations=6)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    table = scene.add_entity(
        gs.morphs.Box(size=(2.0, 2.0, 0.2), pos=(0.0, 0.0, 0.1), fixed=True),
        material=gs.materials.Mochi.Rigid(collider_type="sdf"),
    )
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(0.2, 0.1, 0.6), euler=(20.0, 30.0, 0.0)),
        material=gs.materials.Mochi.Rigid(),
    )
    sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(-0.5, -0.3, 0.5)), material=gs.materials.Mochi.Rigid())
    scene.build()
    for _ in range(200):
        scene.step()
    # Both bodies come to rest on the table top (z = 0.2) within the contact threshold, lying flat.
    assert -3e-3 < cube.get_pos()[2].item() - 0.35 < 1e-3
    assert -3e-3 < sphere.get_pos()[2].item() - 0.3 < 1e-3
    assert_allclose(cube.get_vel(), 0.0, atol=1e-4)
    assert_allclose(sphere.get_vel(), 0.0, atol=1e-4)
    quat = tensor_to_array(cube.get_quat())
    assert_allclose(quat[1:3], 0.0, atol=1e-3)
    info = scene.mochi_solver.get_convergence_info()
    assert int(info["status"][0]) == 1


@pytest.mark.precision("64")
def test_state_reset_restores_trajectory(show_viewer):
    scene = _mochi_scene(show_viewer, 0.02, integrator="bdf2")
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    box = scene.add_entity(
        gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.6), euler=(10.0, 5.0, 0.0)),
        material=gs.materials.Mochi.Rigid(),
    )
    scene.build()
    for _ in range(10):
        scene.step()
    state = scene.get_state()
    trajectory = []
    for _ in range(30):
        scene.step()
        trajectory.append(tensor_to_array(box.get_pos()))
    scene.reset(state)
    for i_step in range(30):
        scene.step()
        assert_allclose(box.get_pos(), trajectory[i_step], tol=1e-9)


def _normal_damping_for_restitution(restitution, impact_speed):
    """Normal viscous damping coefficient producing a given coefficient of restitution at a given approach speed
    (rational fit of the Hunt-Crossley-like response of the penalty contact)."""
    if restitution >= 1.0:
        return 0.0
    return (
        (1.0 - restitution) * (1.0 + 4.5 * restitution) / (restitution * (1.0 + 8.0 / 3.0 * restitution)) / impact_speed
    )


@pytest.mark.precision("64")
@pytest.mark.parametrize("restitution, speed", [(0.3, 1.0), (0.6, 3.0), (1.0, 1.0)])
def test_coefficient_of_restitution_from_normal_damping(restitution, speed, show_viewer):
    scene = _mochi_scene(
        show_viewer,
        1e-4,
        gravity=(0.0, 0.0, 0.0),
        integrator="bdf2",
        implicit_normal_force_for_dissipation=True,
        n_newton_iterations=8,
    )
    damping = _normal_damping_for_restitution(restitution, 2.0 * speed)
    size_1, size_2 = 0.1, 0.15
    gap = 0.5 * (size_1 + size_2) + 0.003
    material = gs.materials.Mochi.Rigid(has_gravity=False, normal_viscous_damping=damping, friction=0.0)
    box_1 = scene.add_entity(gs.morphs.Box(size=(size_1,) * 3, pos=(-0.5 * gap, 0.0, 1.0)), material=material)
    box_2 = scene.add_entity(gs.morphs.Box(size=(size_2,) * 3, pos=(0.5 * gap, 0.0, 1.0)), material=material)
    scene.build()
    box_1.set_dofs_velocity([speed, 0.0, 0.0, 0.0, 0.0, 0.0])
    box_2.set_dofs_velocity([-speed, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        scene.step()

    # One-dimensional restitution law for the head-on collision.
    mass_1, mass_2 = box_1.get_mass(), box_2.get_mass()
    v_1 = ((mass_1 - restitution * mass_2) * speed - (1.0 + restitution) * mass_2 * speed) / (mass_1 + mass_2)
    v_2 = (-(mass_2 - restitution * mass_1) * speed + (1.0 + restitution) * mass_1 * speed) / (mass_1 + mass_2)
    vel_1, vel_2 = tensor_to_array(box_1.get_vel()), tensor_to_array(box_2.get_vel())
    assert_allclose(vel_1[0], v_1, atol=0.02 * 2.0 * speed, rtol=0.0)
    assert_allclose(vel_2[0], v_2, atol=0.02 * 2.0 * speed, rtol=0.0)
    assert_allclose(vel_1[1:], 0.0, atol=0.02 * speed)
    assert_allclose(vel_2[1:], 0.0, atol=0.02 * speed)
    assert box_2.get_pos()[0].item() - box_1.get_pos()[0].item() > gap
