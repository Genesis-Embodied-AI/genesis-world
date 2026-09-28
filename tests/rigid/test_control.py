import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.assets import get_hf_dataset


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_position_control(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=1,  # This is essential to be able to emulate native PD control
        ),
        rigid_options=gs.options.RigidOptions(
            disable_constraint=True,
            integrator=gs.integrator.approximate_implicitfast,
            batch_links_info=True,
            batch_dofs_info=True,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=2, env_spacing=(1.0, 1.0))

    MOTORS_POS_TARGET = torch.tensor(
        [0.6900, -0.1100, -0.7200, -2.7300, -0.1500, 2.6400, 0.8900, 0.0400, 0.0400],
        dtype=gs.tc_float,
        device=gs.device,
    )
    MOTORS_VEL_TARGET = torch.rand_like(MOTORS_POS_TARGET)
    MOTORS_KP = torch.tensor(
        [4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0],
        dtype=gs.tc_float,
        device=gs.device,
    )
    MOTORS_KD = torch.tensor(
        [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0],
        dtype=gs.tc_float,
        device=gs.device,
    )

    # FIXME: We do NOT raise exception anymore when setting control targets that would have no effect
    # robot.set_dofs_kp(torch.zeros_like(MOTORS_KP), envs_idx=0)
    # robot.set_dofs_kv(torch.zeros_like(MOTORS_KD), envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_position_velocity(MOTORS_POS_TARGET, MOTORS_VEL_TARGET, envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_velocity(MOTORS_VEL_TARGET, envs_idx=0)
    # robot.set_dofs_kv(MOTORS_KD, envs_idx=0)
    # robot.control_dofs_velocity(MOTORS_VEL_TARGET, envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    # robot.control_dofs_position_velocity(MOTORS_POS_TARGET, MOTORS_VEL_TARGET, envs_idx=0)

    robot.set_dofs_kp(MOTORS_KP, envs_idx=0)
    robot.set_dofs_kv(MOTORS_KD, envs_idx=0)
    robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    robot.control_dofs_position_velocity(MOTORS_POS_TARGET, MOTORS_VEL_TARGET, envs_idx=0)

    # Must update DoF armature to emulate implicit damping for force control.
    # This is equivalent to the first-order correction term involved in implicit integration scheme,
    # in the particular case where `approximate_implicitfast` integrator is used.
    # Note that the low-level internal API is used because invweights must NOT be updated, otherwise
    # the test cannot pass. This is unecessary and not recommended for practical applications.
    # robot.set_dofs_armature(robot.get_dofs_armature(envs_idx=1) + MOTORS_KD * scene.sim._substep_dt, envs_idx=1)
    dofs_armature = scene.rigid_solver.dyn_info.dofs.armature.to_numpy()
    dofs_armature[:, 1] += tensor_to_array(MOTORS_KD * scene.sim._substep_dt)
    scene.rigid_solver.dyn_info.dofs.armature.from_numpy(dofs_armature)

    force_range = qd_to_torch(scene.rigid_solver.dyn_info.dofs.force_range)
    for i in range(200):
        dofs_pos = robot.get_qpos(envs_idx=1)
        dofs_vel = robot.get_dofs_velocity(envs_idx=1)
        dofs_torque = MOTORS_KP * (MOTORS_POS_TARGET - dofs_pos) + MOTORS_KD * (MOTORS_VEL_TARGET - dofs_vel)
        dofs_torque.clamp_(force_range[:, 1, 0], force_range[:, 1, 1])
        robot.control_dofs_force(dofs_torque, envs_idx=1)
        scene.step()
        qf_applied = scene.rigid_solver.dyn_state.dofs.qf_applied.to_numpy().T
        # dofs_torque = robot.get_dofs_control_force()
        assert_allclose(qf_applied[1], dofs_torque, tol=1e-6)
        assert_allclose(qf_applied[0], qf_applied[1], tol=1e-6)

    A = 0.1
    f = 1.0
    scene.reset()
    robot.set_dofs_kp(MOTORS_KP, envs_idx=1)
    robot.set_dofs_kv(MOTORS_KD, envs_idx=1)
    force_range[:, 1, 0] = float("-inf")
    force_range[:, 1, 1] = float("+inf")
    scene.rigid_solver.dyn_info.dofs.force_range.from_numpy(tensor_to_array(force_range))
    for i in range(1000):
        # The clock of the environment being tracked, which is the one the targets are sent to.
        t = scene.get_time(envs_idx=1)
        pos_target = A * torch.sin(2 * np.pi * f * t)
        vel_target = A * 2 * np.pi * f * torch.cos(2 * np.pi * f * t)
        robot.control_dofs_position_velocity(pos_target, vel_target, envs_idx=1)
        scene.step()
        assert_allclose(pos_target, robot.get_dofs_position(envs_idx=1), tol=1e-2)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("robot_path", ["xml/franka_emika_panda/panda.xml"])
def test_reset_control(robot_path, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=robot_path,
        )
    )
    scene.build()
    qpos = np.random.rand(robot.n_dofs)
    robot.set_dofs_position(qpos)
    robot.control_dofs_position(torch.zeros((robot.n_dofs,), dtype=gs.tc_float, device=gs.device))
    old_control_force = robot.get_dofs_control_force()
    scene.reset()
    new_control_force = robot.get_dofs_control_force()
    assert old_control_force.abs().max() > gs.EPS
    assert_allclose(new_control_force, 0, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("n_envs, substeps", [(0, 5), (2, 1)])
def test_drone_propellers_force_application(n_envs, substeps, show_viewer, tol):
    BASE_RPM = 15000
    RPM_DELTA = 0.01
    SUBSTEP_DT = 0.004
    TOTAL_SUBSTEPS = 10
    GRAVITY = -9.81
    CF2X_ARMS = np.array(
        (
            (0.028, -0.028, 0.0),
            (-0.028, -0.028, 0.0),
            (-0.028, 0.028, 0.0),
            (0.028, 0.028, 0.0),
        )
    )
    n_steps = TOTAL_SUBSTEPS // substeps

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=SUBSTEP_DT * substeps,
            substeps=substeps,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        # The closed-form recurrence below is Euler-exact.
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 0.0, 1.6),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            pos=(0.0, 0.0, 1.0),
            file="urdf/drones/cf2x.urdf",
        ),
    )
    scene.build(n_envs=n_envs)

    # The rotor arms are authored as the propeller links' inertial origins, which alignment must leave in place.
    propellers_idx_local = [drone.get_link(name).idx_local for name in drone.morph.propellers_link_name]
    propellers_pos = drone.get_links_pos(links_idx_local=propellers_idx_local, ref=gs.link_ref_frame.link_COM)
    assert_allclose(propellers_pos - drone.get_pos()[..., None, :], CF2X_ARMS, tol=tol)

    # Reset clears the tracked external force and accepts each supported RPM input shape.
    drone.set_propellers_rpm(BASE_RPM)
    with np.testing.assert_raises(gs.GenesisException):
        drone.set_propellers_rpm(BASE_RPM)
    scene.reset()
    drone.set_propellers_rpm((BASE_RPM,) * 4)
    scene.reset()
    rpm_shape = (n_envs, 4) if n_envs else (4,)
    drone.set_propellers_rpm(torch.full(rpm_shape, fill_value=BASE_RPM))
    scene.reset()

    # Thrust held over several steps: the propeller forces set once per step act on every substep and are cleared
    # before the next step, so the vertical response follows the implicit-damping Euler recurrence exactly, with
    # the semi-implicit position update integrating each new velocity. The recurrence depends only on the total
    # number of substeps, which the sweep pins across different outer step counts.
    pos_z = drone.get_dofs_position(dofs_idx_local=2)[..., 0]
    for _ in range(n_steps):
        drone.set_propellers_rpm(BASE_RPM)
        scene.step()
    thrust = drone.n_propellers * drone.KF * BASE_RPM**2
    weight = drone.get_mass() * GRAVITY
    mass = drone.get_mass_mat()[..., 2, 2]
    damping = drone.get_dofs_damping(dofs_idx_local=2)[..., 0]
    vel_z = 0.0
    for _ in range(TOTAL_SUBSTEPS):
        vel_z = (mass * vel_z + SUBSTEP_DT * (thrust + weight)) / (mass + damping * SUBSTEP_DT)
        pos_z = pos_z + SUBSTEP_DT * vel_z
    assert_allclose(drone.get_dofs_position(dofs_idx_local=2)[..., 0], pos_z, tol=tol)
    assert_allclose(drone.get_dofs_velocity(dofs_idx_local=2)[..., 0], vel_z, tol=tol)
    assert_allclose(drone.get_dofs_velocity(dofs_idx_local=[3, 4, 5]), 0.0, tol=tol)
    scene.reset()

    # One differential-thrust step isolates the rotor-arm moment and cancels pitch and yaw torque.
    roll_rpm = tuple(BASE_RPM * scale for scale in (1 + RPM_DELTA, 1 + RPM_DELTA, 1 - RPM_DELTA, 1 - RPM_DELTA))
    drone.set_propellers_rpm(roll_rpm)
    scene.step()
    thrust = drone.KF * np.square(roll_rpm)
    roll_torque = np.cross(CF2X_ARMS, np.outer(thrust, (0.0, 0.0, 1.0))).sum(axis=0)[0]
    inertia = drone.get_mass_mat()[..., 3, 3]
    damping = drone.get_dofs_damping(dofs_idx_local=3)[..., 0]
    roll_rate = 0.0
    for _ in range(substeps):
        roll_rate = (inertia * roll_rate + SUBSTEP_DT * roll_torque) / (inertia + damping * SUBSTEP_DT)
    assert_allclose(drone.get_dofs_velocity(dofs_idx_local=3)[..., 0], roll_rate, tol=tol)
    assert_allclose(drone.get_dofs_velocity(dofs_idx_local=[4, 5]), 0.0, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_drone_advanced(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="drone_sus/*")
    drones = []
    for offset, merge_fixed_links in ((-0.3, False), (0.3, True)):
        drone = scene.add_entity(
            morph=gs.morphs.Drone(
                file=f"{asset_path}/drone_sus/drone_sus.urdf",
                merge_fixed_links=merge_fixed_links,
                pos=(0.0, offset, 1.5),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        drones.append(drone)
    scene.build()

    for drone in drones:
        assert drone.base_link.parent_idx == -1
        assert_equal([link.root_idx for link in drone.links], drone.base_link.idx)
        assert all(drone.link_start <= link.parent_idx < link.idx for link in drone.links[1:])

    for drone in drones:
        chain_dofs = range(6, drone.n_dofs)
        drone.set_dofs_armature(drone.get_dofs_armature(chain_dofs) + 1e-3, chain_dofs)

    # Wait for the drones to land on the ground and hold straight
    for i in range(400):
        for drone in drones:
            drone.set_propellers_rpm(50000.0)
        scene.step()
        if i > 350:
            assert scene.rigid_solver.collider.collider_state.n_contacts.to_numpy()[0] == 2
            assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0, tol=2e-3)

    # Push the drones symmetrically and wait for them to collide
    drones[0].set_dofs_velocity([0.2], [1])
    drones[1].set_dofs_velocity([-0.2], [1])
    for i in range(150):
        for drone in drones:
            drone.set_propellers_rpm(50000.0)
        scene.step()
        if scene.rigid_solver.collider.collider_state.n_contacts.to_numpy()[0] > 2:
            break
    else:
        raise AssertionError

    tol = 1e-2
    pos_1 = drones[0].get_pos()
    pos_2 = drones[1].get_pos()
    assert abs(pos_1[0] - pos_2[0]) < tol
    assert abs(pos_1[1] + pos_2[1]) < tol
    assert abs(pos_1[2] - pos_2[2]) < tol
    quat_1 = drones[0].get_quat()
    quat_2 = drones[1].get_quat()
    assert abs(quat_1[1] + quat_2[1]) < tol
    assert abs(quat_1[2] - quat_2[2]) < tol
    assert abs(quat_1[2] - quat_2[2]) < tol
