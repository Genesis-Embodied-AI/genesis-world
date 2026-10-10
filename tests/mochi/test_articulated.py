import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose
from .reference import load_reference


@pytest.fixture(scope="session")
def pendulum_urdf_path(tmp_path_factory):
    """Planar pendulum: a massless arm on a continuous joint about x carrying a point mass 1 m below the pivot."""
    robot = ET.Element("robot", name="pendulum")
    ET.SubElement(robot, "link", name="base")
    joint = ET.SubElement(robot, "joint", name="pivot", type="continuous")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(joint, "axis", xyz="1 0 0")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="arm")
    arm = ET.SubElement(robot, "link", name="arm")
    inertial = ET.SubElement(arm, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 -1", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="1")
    ET.SubElement(inertial, "inertia", ixx="1e-6", iyy="1e-6", izz="1e-6", ixy="0", ixz="0", iyz="0")
    visual = ET.SubElement(arm, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 -0.5", rpy="0 0 0")
    ET.SubElement(ET.SubElement(visual, "geometry"), "box", size="0.02 0.02 1")
    path = tmp_path_factory.mktemp("mochi") / "pendulum.urdf"
    ET.ElementTree(robot).write(path)
    return str(path)


def _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -10.0), **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
        mochi_options=gs.options.MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("integrator", ["backward_euler", "bdf2"])
def test_pendulum_period_and_energy(pendulum_urdf_path, integrator, show_viewer):
    dt = 0.005
    gravity = 10.0
    armature = 0.1
    theta_0 = 0.1
    scene = _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -gravity), integrator=integrator, n_newton_iterations=8)
    pendulum = scene.add_entity(
        gs.morphs.URDF(file=pendulum_urdf_path, pos=(0.0, 0.0, 2.0), fixed=True, default_armature=armature),
        material=gs.materials.Mochi.Rigid(),
    )
    scene.build()
    assert scene.mochi_solver.n_dofs == 1

    pendulum.set_dofs_position([theta_0])
    # Period of a point mass on a massless arm with rotor inertia, including the first amplitude correction.
    period = 2.0 * math.pi * math.sqrt((1.0 + armature) / gravity) * (1.0 + theta_0**2 / 16.0)
    n_periods = 5
    angles = []
    for _ in range(round(n_periods * period / dt)):
        scene.step()
        angles.append(tensor_to_array(pendulum.get_dofs_position())[0])
    angles = np.array(angles)
    crossings = np.flatnonzero(np.diff(np.sign(angles)))
    periods = 2.0 * dt * np.diff(crossings)
    assert_allclose(periods.mean(), period, rtol=1e-2, atol=0.0)
    amplitude = np.abs(angles[-int(period / dt) :]).max()
    # Backward Euler damps the swing; BDF2 keeps the amplitude within a fraction of a percent over 5 periods.
    if integrator == "bdf2":
        assert 0.99 * theta_0 < amplitude <= theta_0 + 1e-9
    else:
        assert 0.7 * theta_0 < amplitude < 0.95 * theta_0
    assert int(scene.mochi_solver.get_convergence_info()["status"][0]) == 1


@pytest.mark.required
@pytest.mark.precision("64")
def test_arm_pd_control_limits_and_contact(show_viewer):
    dt = 0.01
    scene = _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -9.81), n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"), material=gs.materials.Mochi.Rigid()
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.65, 0.0, 0.05)), material=gs.materials.Mochi.Rigid()
    )
    scene.build()
    assert franka.n_dofs == 9

    # Position control: the arm settles at the target up to the gravity sag kp allows.
    kp = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0])
    kv = np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0])
    q_target = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8, 0.02, 0.02])
    franka.set_dofs_kp(kp)
    franka.set_dofs_kv(kv)
    franka.control_dofs_position(q_target)
    for _ in range(200):
        scene.step()
    q = tensor_to_array(franka.get_dofs_position())
    assert_allclose(q, q_target, atol=2e-2, rtol=0.0)
    assert_allclose(franka.get_dofs_velocity(), 0.0, atol=1e-4)
    # The control force balances gravity at rest: the joint-space residual of the arm vanishes.
    control_force = tensor_to_array(franka.get_dofs_control_force())
    assert np.abs(control_force[:7]).max() > 1.0
    assert int(scene.mochi_solver.get_convergence_info()["status"][0]) == 1
    # The box rests on the plane next to the arm, untouched.
    assert -3e-3 < tensor_to_array(box.get_pos())[2] - 0.05 < 1e-3

    # Gravity compensation through force control holds the pose.
    franka.control_dofs_force(control_force)
    for _ in range(50):
        scene.step()
    assert_allclose(franka.get_dofs_position(), q, atol=2e-3, rtol=0.0)

    # Released, the arm falls onto its joint limits, which hold up to the penalty compliance.
    franka.control_dofs_force(np.zeros(9))
    for _ in range(500):
        scene.step()
    q = tensor_to_array(franka.get_dofs_position())
    lower, upper = (tensor_to_array(limit) for limit in franka.get_dofs_limit())
    violation = np.maximum(np.maximum(q - upper, lower - q), 0.0)
    assert violation.max() < 5e-3
    assert violation.max() > 0.0
    assert_allclose(franka.get_dofs_velocity(), 0.0, atol=2e-2)


@pytest.mark.precision("64")
def test_articulated_free_base_chain_and_batching(pendulum_urdf_path, show_viewer):
    # A floating-base pendulum (free root + revolute joint) dropped in a batch: every environment must reproduce the
    # single-body free fall of the total mass and keep the revolute joint at rest under symmetric initial conditions.
    dt = 0.01
    n_envs = 3
    scene = _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -10.0), n_newton_iterations=8)
    pendulum = scene.add_entity(
        gs.morphs.URDF(file=pendulum_urdf_path, pos=(0.0, 0.0, 3.0), fixed=False, default_armature=0.0),
        material=gs.materials.Mochi.Rigid(has_gravity=True),
    )
    scene.build(n_envs=n_envs)
    assert scene.mochi_solver.n_dofs == 7
    heights = 3.0 + 0.5 * np.arange(n_envs)
    pendulum.set_pos(np.stack([np.zeros(n_envs), np.zeros(n_envs), heights], axis=-1))
    n_steps = 40
    for _ in range(n_steps):
        scene.step()
    # Backward Euler free fall of the base: z = z0 - g dt^2 n (n + 1) / 2.
    z_expected = heights - 10.0 * dt**2 * n_steps * (n_steps + 1) / 2.0
    assert_allclose(tensor_to_array(pendulum.get_pos())[:, 2], z_expected, atol=1e-6, rtol=0.0)
    assert_allclose(tensor_to_array(pendulum.get_dofs_position())[:, 6], 0.0, atol=1e-9)
    assert_allclose(tensor_to_array(pendulum.get_dofs_velocity())[:, 6], 0.0, atol=1e-9)


@pytest.fixture(scope="session")
def double_pendulum_urdf_path(tmp_path_factory):
    """Two 1 m links on revolute joints about x hanging from a fixed base, each carrying a 0.1 m cube at its lower
    end whose mass and inertia follow from the density (the mochi reference model)."""
    robot = ET.Element("robot", name="double_pendulum")
    ET.SubElement(robot, "link", name="base")
    parent = "base"
    for i_link in range(2):
        joint = ET.SubElement(robot, "joint", name=f"joint_{i_link}", type="continuous")
        ET.SubElement(joint, "origin", xyz="0 0 0" if i_link == 0 else "0 0 -1", rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz="1 0 0")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=f"link_{i_link}")
        link = ET.SubElement(robot, "link", name=f"link_{i_link}")
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz="0 0 -1", rpy="0 0 0")
        ET.SubElement(ET.SubElement(collision, "geometry"), "box", size="0.1 0.1 0.1")
        parent = f"link_{i_link}"
    path = tmp_path_factory.mktemp("mochi") / "double_pendulum.urdf"
    ET.ElementTree(robot).write(path)
    return str(path)


@pytest.mark.required
@pytest.mark.precision("64")
def test_double_pendulum_matches_mochi(double_pendulum_urdf_path, show_viewer):
    reference = load_reference("double_pendulum")
    dt = float(reference["dt"])
    scene = _mochi_scene(
        show_viewer, dt, gravity=(0.0, 0.0, -10.0), n_newton_iterations=8, newton_abs_tol=1e-10, newton_rel_tol=1e-12
    )
    pendulum = scene.add_entity(
        gs.morphs.URDF(file=double_pendulum_urdf_path, pos=(0.0, 0.0, 3.0), fixed=True, default_armature=0.0),
        material=gs.materials.Mochi.Rigid(),
    )
    scene.build()
    assert scene.mochi_solver.n_dofs == 2
    assert_allclose([link.desc.mass for link in pendulum.links[1:]], 1.0, tol=1e-9)

    pendulum.set_dofs_position([0.6, -0.3])
    angles, velocities = [], []
    for _ in range(len(reference["angles"])):
        scene.step()
        angles.append(tensor_to_array(pendulum.get_dofs_position()))
        velocities.append(tensor_to_array(pendulum.get_dofs_velocity()))
    assert_allclose(angles, reference["angles"], atol=1e-7, rtol=0.0)
    assert_allclose(velocities, reference["velocities"], atol=1e-6, rtol=0.0)
