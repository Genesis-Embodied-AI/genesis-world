import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose


@pytest.fixture(scope="session")
def joint_torque_pendulums():
    # Four independent single-DOF pendulums (a point mass at distance 1 m from a hinge), one per gearbox-loss kind:
    # none, armature inertia, Coulomb frictionloss, viscous damping. Spaced 2 m apart so they never collide.
    mjcf = ET.Element("mujoco", model="joint_torque_pendulums")
    ET.SubElement(mjcf, "compiler", angle="radian")
    worldbody = ET.SubElement(mjcf, "worldbody")
    losses = (("0", "0", "0"), ("0.5", "0", "0"), ("0", "0.4", "0"), ("0", "0", "0.3"))
    for i, (armature, frictionloss, damping) in enumerate(losses):
        arm = ET.SubElement(worldbody, "body", name=f"arm{i}", pos=f"{2 * i} 0 0")
        ET.SubElement(
            arm,
            "joint",
            name=f"j{i}",
            type="hinge",
            axis="0 1 0",
            armature=armature,
            frictionloss=frictionloss,
            damping=damping,
        )
        mass = ET.SubElement(arm, "body", pos="0 0 -1.0")
        ET.SubElement(mass, "geom", type="sphere", size="0.05", mass="1.0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def joint_torque_contact_pendulum():
    # Single lossless pendulum; the wall it presses against is added separately in the test.
    mjcf = ET.Element("mujoco", model="joint_torque_contact_pendulum")
    ET.SubElement(mjcf, "compiler", angle="radian")
    worldbody = ET.SubElement(mjcf, "worldbody")
    arm = ET.SubElement(worldbody, "body", name="arm")
    ET.SubElement(arm, "joint", name="hinge", type="hinge", axis="0 1 0")
    mass = ET.SubElement(arm, "body", pos="0 0 -1.0")
    ET.SubElement(mass, "geom", type="sphere", size="0.05", mass="1.0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_joint_torque(joint_torque_pendulums, show_viewer, tol, n_envs):
    # One MJCF packs four single-DOF pendulums, each with a different gearbox loss (none, armature, frictionloss,
    # damping); a single JointTorqueSensor reads all four output efforts. The loss parameters are read back from the
    # built model so the assertions stay in sync with the MJCF.
    MASS = 1.0  # link mass, kg
    LENGTH = 1.0  # pivot-to-mass distance, m
    RADIUS = 0.05  # mass sphere radius, m
    # Inertia of a solid sphere about the pivot: parallel-axis m * L**2 plus the sphere's own 2/5 * m * R**2.
    INERTIA = MASS * LENGTH**2 + 0.4 * MASS * RADIUS**2
    GRAVITY = 9.81  # m/s^2
    DT = 0.01
    INIT_ANGLE = np.pi / 6
    TAU = 10.0  # above gravity so every joint swings forward and the losses reduce the transmitted effort, N m

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        # Explicit Euler makes the velocity update exactly vel += dt * qacc, so the finite-differenced qacc_num equals
        # the solver's qacc and the analytical torque identity holds to float tolerance.
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
        ),
        show_viewer=show_viewer,
    )
    pendulums = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=joint_torque_pendulums,
            default_armature=0.0,
        ),
    )
    sensor = scene.add_sensor(
        gs.sensors.JointTorque(
            entity_idx=pendulums.idx,
        ),
    )
    scene.build(n_envs=n_envs)

    armature = pendulums.get_dofs_armature()
    damping = pendulums.get_dofs_damping()
    # Implicit-damping integration adds a first-order damping * dt term to the effective inertia (the same correction
    # applied to the armature in test_position_control), so the Newton identity below stays exact for the damping joint.
    effective_inertia = INERTIA + damping * DT

    pendulums.set_qpos(INIT_ANGLE)

    # Spin the joints up from rest so the lossy ones are clearly sliding forward before checking dissipation.
    for _ in range(10):
        pendulums.control_dofs_force(TAU)
        scene.step()

    vel_prev = pendulums.get_dofs_velocity()
    for _ in range(20):
        theta_before = pendulums.get_dofs_position()
        pendulums.control_dofs_force(TAU)
        scene.step()

        vel_after = pendulums.get_dofs_velocity()
        qacc_num = (vel_after - vel_prev) / DT

        tau_s = sensor.read()
        tau_ctrl = pendulums.get_dofs_control_force()

        # Newton ground truth: the sensor reads the physical effort reaching each link.
        tau_phys = effective_inertia * qacc_num + MASS * GRAVITY * LENGTH * torch.sin(theta_before)
        assert_allclose(tau_s, tau_phys, tol=tol)

        # Lossless joint (index 0): sensor equals the commanded effort exactly.
        assert_allclose(tau_s[..., 0], tau_ctrl[..., 0], tol=tol)
        assert_allclose(tau_s[..., 0], TAU, tol=tol)
        # Armature joint (index 1): the armature absorbs its share of the acceleration load.
        assert_allclose((tau_s - tau_ctrl)[..., 1], -armature[..., 1] * qacc_num[..., 1], tol=tol)
        # Damping joint (index 3): sensor is reduced by damping * vel (velocity before the step).
        assert_allclose(tau_s[..., 3], tau_ctrl[..., 3] - damping[..., 3] * vel_prev[..., 3], tol=tol)
        # Friction (index 2) and damping (index 3) dissipate part of the command, so the sensor reads less than it
        # while the joints slide forward.
        assert (tau_s[..., 2] < tau_ctrl[..., 2]).all()
        assert (tau_s[..., 3] < tau_ctrl[..., 3]).all()

        vel_prev = vel_after


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_joint_torque_with_contact(joint_torque_contact_pendulum, show_viewer, tol, n_envs):
    # A lossless pendulum pressed against a fixed wall: contact forces flow through the equations of motion and never
    # appear in the sensor formula, so the reading stays equal to the commanded effort throughout the impact.
    TAU = 10.0

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    pendulum = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=joint_torque_contact_pendulum,
            default_armature=0.0,
        ),
    )
    # Box face at x=0.75; the mass (sphere r=0.05) at theta=pi/4 sits at x=0.707, so contact is active from step 1.
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.85, 0.0, -0.70),
            size=(0.2, 0.5, 0.2),
            fixed=True,
        ),
    )
    sensor = scene.add_sensor(
        gs.sensors.JointTorque(
            entity_idx=pendulum.idx,
        ),
    )
    scene.build(n_envs=n_envs)

    # Start at 45 deg already touching the wall; positive torque keeps pressing into it.
    pendulum.set_qpos(np.pi / 4)

    for _ in range(30):
        pendulum.control_dofs_force(TAU)
        scene.step()
        assert_allclose(sensor.read(), pendulum.get_dofs_control_force(), tol=tol)
        assert_allclose(sensor.read(), TAU, tol=tol)
