import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["two_aligned_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
# Swept over compatibility, which decides whether every step recomputes the whole derived state or only what is owed.
@pytest.mark.parametrize("mujoco_compatibility", [True, False])
def test_link_velocity(gs_sim, tol):
    # Check the velocity for a few "easy" special cases
    (gs_robot,) = gs_sim.entities

    gs_robot.set_dofs_velocity([0.0, 1.0])
    assert_allclose(gs_sim.rigid_solver.dyn_state.links.cd_vel.to_numpy(), 0, tol=tol)

    gs_robot.set_dofs_velocity([1.0, 0.0])
    cvel_0, cvel_1 = gs_sim.rigid_solver.dyn_state.links.cd_vel.to_numpy()[:, 0]
    assert_allclose(cvel_0, np.array([0.0, 0.5, 0.0]), tol=tol)
    assert_allclose(cvel_1, np.array([0.0, 0.5, 0.0]), tol=tol)

    gs_robot.set_qpos([0.0, np.pi / 2.0])
    gs_robot.set_dofs_velocity([0.0, 1.2])
    COM = gs_sim.rigid_solver.dyn_state.links.root_COM[0, 0]
    assert_allclose(COM, np.array([0.375, 0.125, 0.0]), tol=tol)
    xanchor = gs_sim.rigid_solver.dyn_state.joints.xanchor[1, 0]
    assert_allclose(xanchor, np.array([0.5, 0.0, 0.0]), tol=tol)
    cvel_0, cvel_1 = gs_sim.rigid_solver.dyn_state.links.cd_vel.to_numpy()[:, 0]
    assert_allclose(cvel_0, 0, tol=tol)
    assert_allclose(cvel_1, np.array([-1.2 * (0.125 - 0.0), 1.2 * (0.375 - 0.5), 0.0]), tol=tol)

    # Check that the velocity is valid for a random configuration
    gs_robot.set_qpos([-0.7, 0.2])
    gs_robot.set_dofs_velocity([3.0, 13.0])
    xanchor = gs_sim.rigid_solver.dyn_state.joints.xanchor[1, 0]
    theta_0, theta_1 = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(xanchor[0], 0.5 * np.cos(theta_0), tol=tol)
    assert_allclose(xanchor[1], 0.5 * np.sin(theta_0), tol=tol)
    COM = gs_sim.rigid_solver.dyn_state.links.root_COM[0, 0]
    COM_0 = np.array([0.25 * np.cos(theta_0), 0.25 * np.sin(theta_0), 0.0])
    COM_1 = np.array(
        [
            0.5 * np.cos(theta_0) + 0.25 * np.cos(theta_0 + theta_1),
            0.5 * np.sin(theta_0) + 0.25 * np.sin(theta_0 + theta_1),
            0.0,
        ]
    )
    link_COM0, link_COM1 = gs_sim.rigid_solver.get_links_pos(ref=gs.link_ref_frame.link_COM)

    assert_allclose(link_COM0, COM_0, tol=tol)
    assert_allclose(link_COM1, COM_1, tol=tol)
    assert_allclose(COM, 0.5 * (COM_0 + COM_1), tol=tol)

    cvel_0, cvel_1 = gs_sim.rigid_solver.dyn_state.links.cd_vel.to_numpy()[:, 0]
    omega_0, omega_1 = gs_sim.rigid_solver.dyn_state.links.cd_ang.to_numpy()[:, 0, 2]
    assert_allclose(omega_0, 3.0, tol=tol)
    assert_allclose(omega_1 - omega_0, 13.0, tol=tol)
    cvel_0_ = omega_0 * np.array([-COM[1], COM[0], 0.0])
    assert_allclose(cvel_0, cvel_0_, tol=tol)
    cvel_1_ = cvel_0 + (omega_1 - omega_0) * np.array([xanchor[1] - COM[1], COM[0] - xanchor[0], 0.0])
    assert_allclose(cvel_1, cvel_1_, tol=tol)

    xpos_0, xpos_1 = gs_sim.rigid_solver.dyn_state.links.pos.to_numpy()[:, 0]
    assert_allclose(xpos_0, 0.0, tol=tol)
    assert_allclose(xpos_1, xanchor, tol=tol)
    xvel_0, xvel_1 = gs_sim.rigid_solver.get_links_vel()
    assert_allclose(xvel_0, 0.0, tol=tol)
    xvel_1_ = omega_0 * np.array([-xpos_1[1], xpos_1[0], 0.0])
    assert_allclose(xvel_1, xvel_1_, tol=tol)
    civel_0, civel_1 = gs_sim.rigid_solver.get_links_vel(ref=gs.link_ref_frame.link_COM)
    civel_0_ = omega_0 * np.array([-COM_0[1], COM_0[0], 0.0])
    assert_allclose(civel_0, civel_0_, tol=tol)
    civel_1_ = omega_0 * np.array([-COM_1[1], COM_1[0], 0.0]) + (omega_1 - omega_0) * np.array(
        [xanchor[1] - COM_1[1], COM_1[0] - xanchor[0], 0.0]
    )
    assert_allclose(civel_1, civel_1_, tol=tol)

    # A mass or a center of mass written moves the center the velocities are quoted about, and the write alone has to
    # leave them right: the same identity, against the masses and the placements the solver now carries.
    solver = gs_sim.rigid_solver
    for writing in ("mass", "COM"):
        if writing == "mass":
            solver.set_links_mass([1.0, 3.0])
        else:
            solver.set_links_COM(solver.get_links_COM() + torch.tensor([0.05, -0.02, 0.0]))

        masses = tensor_to_array(solver.get_links_mass())
        centers = tensor_to_array(solver.get_links_pos(ref=gs.link_ref_frame.link_COM))
        anchor = tensor_to_array(gs_robot.joints[1].get_anchor_pos())
        turn = tensor_to_array(solver.get_links_ang())[:, 2]
        weighed = (masses[:, None] * centers).sum(axis=0) / masses.sum()
        assert_allclose(tensor_to_array(solver.get_links_pos(ref=gs.link_ref_frame.root_COM)), weighed, tol=tol)

        # The velocity of a point of a turning link, taken at each link's own center of mass.
        for i_l, (center, spin) in enumerate(zip(centers, turn)):
            expected = turn[0] * np.array([-center[1], center[0], 0.0])
            if i_l:
                expected += (spin - turn[0]) * np.array([anchor[1] - center[1], center[0] - anchor[0], 0.0])
            reported = tensor_to_array(solver.get_links_vel(i_l, ref=gs.link_ref_frame.link_COM))
            assert_allclose(reported, expected, tol=tol)


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["pendulum"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_pendulum_links_acc(gs_sim, tol):
    pendulum = gs_sim.entities[0]
    g = gs_sim.rigid_solver._gravity[0][2]

    # Make sure that the linear and angular acceleration matches expectation
    theta = np.random.rand()
    theta_dot = np.random.rand()
    pendulum.set_qpos([theta])
    pendulum.set_dofs_velocity([theta_dot])
    for _ in range(100):
        # Backup state before integration
        theta = gs_sim.rigid_solver.qpos[0, 0]
        theta_dot = gs_sim.rigid_solver.dyn_state.dofs.vel[0, 0]

        # Run one simulation step
        gs_sim.scene.step()

        # Angular acceleration:
        # * acc_ang_x = - sin(theta) * g
        acc_ang = gs_sim.rigid_solver.get_links_acc_ang()
        assert_allclose(acc_ang[0], 0, tol=tol)
        assert_allclose(acc_ang[2], np.array([-np.sin(theta) * g, 0.0, 0.0]), tol=tol)
        # Linear spatial acceleration:
        # * acc_spatial_lin_y = sin(theta) * g
        acc_spatial_lin_world = gs_sim.rigid_solver.dyn_state.links.cacc_lin.to_numpy()
        assert_allclose(acc_spatial_lin_world[0], 0, tol=tol)
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), np.sin(theta)],
                [0.0, -np.sin(theta), np.cos(theta)],
            ]
        )
        acc_spatial_lin_local = R @ acc_spatial_lin_world[2, 0]
        assert_allclose(acc_spatial_lin_local, np.array([0.0, np.sin(theta) * g, 0.0]), tol=tol)
        # Linear true acceleration:
        # * acc_classical_lin_y = sin(theta) * g (tangential angular acceleration effect)
        # * acc_classical_lin_z = - theta_dot ** 2  (radial centripedal effect)
        acc_classical_lin_world = tensor_to_array(gs_sim.rigid_solver.get_links_acc())
        assert_allclose(acc_classical_lin_world[0], 0, tol=tol)
        acc_classical_lin_local = R @ acc_classical_lin_world[2]
        assert_allclose(acc_classical_lin_local, np.array([0.0, np.sin(theta) * g, -(theta_dot**2)]), tol=tol)

    # Hold the pendulum straight using PD controller and check again
    pendulum.set_dofs_kp([4000.0])
    pendulum.set_dofs_kv([100.0])
    pendulum.control_dofs_position([0.5 * np.pi])
    for _ in range(400):
        gs_sim.scene.step()
    acc_classical_lin_world = gs_sim.rigid_solver.get_links_acc()
    assert_allclose(acc_classical_lin_world, 0, tol=tol)


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["double_pendulum"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_double_pendulum_links_acc(gs_sim, tol):
    robot = gs_sim.entities[0]

    # Make sure that the linear and angular acceleration matches expectation
    qpos = np.random.rand(2)
    qvel = np.random.rand(2)
    robot.set_qpos(qpos)
    robot.set_dofs_velocity(qvel)
    for _ in range(100):
        # Backup state before integration
        theta = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        theta_dot = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]

        # Run one simulation step
        gs_sim.scene.step()

        # Backup acceleration before integration
        theta_ddot = gs_sim.rigid_solver.dyn_state.dofs.acc.to_numpy()[:, 0]

        # Angular acceleration
        acc_ang = tensor_to_array(gs_sim.rigid_solver.get_links_acc_ang())
        assert_allclose(acc_ang[0], 0, tol=tol)
        assert_allclose(acc_ang[1], [theta_ddot[0], 0.0, 0.0], tol=tol)
        assert_allclose(acc_ang[-1], [theta_ddot[0] + theta_ddot[1], 0.0, 0.0], tol=tol)

        # Linear spatial acceleration
        cacc_spatial_lin_world = gs_sim.rigid_solver.dyn_state.links.cacc_lin.to_numpy()[[0, 2, 4], 0]
        com = gs_sim.rigid_solver.dyn_state.links.root_COM.to_numpy()[-1, 0]
        pos = gs_sim.rigid_solver.dyn_state.links.pos.to_numpy()[[0, 2, 4], 0]
        assert_allclose(cacc_spatial_lin_world[1], np.cross(acc_ang[2], com), tol=tol)
        acc_spatial_lin_world = cacc_spatial_lin_world + np.cross(acc_ang[[0, 2, 4]], pos - com)
        assert_allclose(acc_spatial_lin_world[0], 0, tol=tol)
        theta_world = theta.cumsum()
        R = np.array(
            [
                [np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)],
                [np.zeros_like(theta), np.cos(theta_world), np.sin(theta_world)],
                [np.zeros_like(theta), -np.sin(theta_world), np.cos(theta_world)],
            ]
        )
        acc_spatial_lin_local = np.matmul(np.moveaxis(R, 2, 0), acc_spatial_lin_world[1:, :, None])[..., 0]
        assert_allclose(acc_spatial_lin_local[0], np.array([0.0, -theta_ddot[0], 0.0]), tol=tol)
        assert_allclose(
            acc_spatial_lin_local[1],
            R[..., 1] @ (R[..., 0].T @ np.array([0.0, -theta_ddot[0], theta_dot[0] * theta_dot[1]]))
            + np.array([0.0, -theta_ddot.sum(), 0.0]),
            tol=tol,
        )

        # Linear true acceleration
        acc_classical_lin_world = tensor_to_array(gs_sim.rigid_solver.get_links_acc()[[0, 2, 4]])
        assert_allclose(acc_classical_lin_world[0], 0, tol=tol)
        acc_classical_lin_local = np.matmul(np.moveaxis(R, 2, 0), acc_classical_lin_world[1:, :, None])[..., 0]
        assert_allclose(acc_classical_lin_local[0], np.array([0.0, -theta_ddot[0], -(theta_dot[0] ** 2)]), tol=tol)
        assert_allclose(
            acc_classical_lin_local[1],
            R[..., 1] @ acc_classical_lin_world[1] + np.array([0.0, -theta_ddot.sum(), -(theta_dot.sum() ** 2)]),
            tol=tol,
        )

    # Hold the double pendulum straight using PD controller and check again
    robot.set_dofs_kp([6000.0, 4000.0])
    robot.set_dofs_kv([200.0, 150.0])
    robot.control_dofs_position([0.5 * np.pi, 0.0])
    for _ in range(900):
        gs_sim.scene.step()
    acc_classical_lin_world = gs_sim.rigid_solver.get_links_acc()
    assert_allclose(acc_classical_lin_world, 0, tol=tol)


@pytest.mark.required
@pytest.mark.merge_fixed_links(False)
@pytest.mark.parametrize("model_name", ["pendulum"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_jacobian(gs_sim, tol):
    (pendulum,) = gs_sim.entities

    angle = 0.7
    pendulum.set_qpos(np.array([angle], dtype=gs.np_float))
    gs_sim.scene.step()

    link = pendulum.get_link("PendulumArm_0")

    p_local = np.array([0.05, -0.02, 0.12], dtype=gs.np_float)
    J_o = tensor_to_array(pendulum.get_jacobian(link))
    J_p = tensor_to_array(pendulum.get_jacobian(link, p_local))

    c, s = np.cos(angle), np.sin(angle)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=gs.np_float,
    )
    r_world = Rx @ p_local
    r_cross = np.array(
        [
            [0, -r_world[2], r_world[1]],
            [r_world[2], 0, -r_world[0]],
            [-r_world[1], r_world[0], 0],
        ],
        dtype=gs.np_float,
    )

    lin_o, ang_o = J_o[:3, 0], J_o[3:, 0]
    lin_expected = lin_o - r_cross @ ang_o

    assert_allclose(J_p[3:, 0], ang_o, tol=tol)
    assert_allclose(J_p[:3, 0], lin_expected, tol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_joint_get_anchor_pos_and_axis(n_envs):
    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=n_envs)
    batch_shape = (n_envs,) if n_envs > 0 else ()

    joint = robot.joints[1]
    anchor_pos = joint.get_anchor_pos()
    assert anchor_pos.shape == (*batch_shape, 3)
    expected_pos = scene.rigid_solver.dyn_state.joints.xanchor.to_numpy()
    assert_allclose(anchor_pos, expected_pos[joint.idx], tol=gs.EPS)

    anchor_axis = joint.get_anchor_axis()
    assert anchor_axis.shape == (*batch_shape, 3)
    expected_axis = scene.rigid_solver.dyn_state.joints.xaxis.to_numpy()
    assert_allclose(anchor_axis, expected_axis[joint.idx], tol=gs.EPS)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_inverse_kinematics_multilink(show_viewer, tol):
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(1,),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    # Add one extra entity, just to make sure there is no idx offset issues
    scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.2, 0.05),
        ),
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
        ),
    )
    scene.build(n_envs=2)
    scene.reset()

    index_finger_distal = robot.get_link("index_finger_distal")
    middle_finger_distal = robot.get_link("middle_finger_distal")
    wrist = robot.get_link("wrist")
    index_finger_pos = np.array([[0.6, 0.5, 0.2]])
    middle_finger_pos = np.array([[0.63, 0.5, 0.2]])
    wrist_pos = index_finger_pos - np.array([[0.0, 0.0, 0.2]])

    qpos, err = robot.inverse_kinematics_multilink(
        links=(index_finger_distal, middle_finger_distal, wrist),
        poss=(index_finger_pos, middle_finger_pos, wrist_pos),
        envs_idx=(1,),
        pos_tol=tol,
        rot_tol=tol,
        max_solver_iters=100,
        return_error=True,
    )
    assert qpos.shape == (1, robot.n_qs)
    assert err.shape == (1, 3, 6)
    assert_allclose(err, 0.0, atol=tol)

    robot.set_qpos(qpos, envs_idx=(1,))
    if show_viewer:
        scene.visualizer.update(force=True)
    assert_allclose(index_finger_distal.get_pos(envs_idx=(1,)), index_finger_pos, tol=tol)
    assert_allclose(middle_finger_distal.get_pos(envs_idx=(1,)), middle_finger_pos, tol=tol)
    assert_allclose(wrist.get_pos(envs_idx=(1,)), wrist_pos, tol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_inverse_kinematics_local_point(n_envs, show_viewer, tol):
    # local_point positions an offset point of the link at the target. The floating-base go2 also exercises the
    # FREE root joint (base translation and orientation degrees of freedom) at a non-zero entity offset.
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    floating = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 1.5, 0.42),
        ),
    )
    scene.build(n_envs=n_envs)

    end_effector = robot.get_link("hand")

    # Define a local offset point in the end-effector frame (e.g., 10cm along Z-axis)
    local_offset = torch.tensor([0.0, 0.0, 0.1], dtype=gs.tc_float, device=gs.device)

    # Create different target positions and quaternions for each environment
    num_envs = max(n_envs, 1)
    target_pos_base = torch.tensor(
        [[0.5, 0.2, 0.4], [0.45, 0.15, 0.35], [0.55, 0.25, 0.45]], dtype=gs.tc_float, device=gs.device
    )[:num_envs]
    target_quat_base = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.9239, 0.3827, 0.0], [0.0, 0.9239, -0.3827, 0.0]],
        dtype=gs.tc_float,
        device=gs.device,
    )[:num_envs]

    # Handle different shapes based on n_envs
    if n_envs > 0:
        target_pos = target_pos_base
        target_quat = target_quat_base
    else:
        target_pos = target_pos_base[0]
        target_quat = target_quat_base[0]

    targets = ((target_pos, target_quat),)
    if n_envs > 0:
        targets += ((target_pos[0], target_quat[0]),)

    # Solve IK with local_point (local_offset stays 1D - it gets broadcast internally)
    qpos_solutions = []
    for query_pos, query_quat in targets:
        qpos_solution, err = robot.inverse_kinematics(
            link=end_effector,
            pos=query_pos,
            quat=query_quat,
            local_point=local_offset,
            # Reuse the first environment's solution so identical targets compare the same IK branch.
            init_qpos=qpos_solutions[0][0] if qpos_solutions else None,
            pos_tol=tol,
            rot_tol=tol,
            max_solver_iters=100,
            return_error=True,
        )
        assert_allclose(err, 0.0, atol=tol)
        qpos_solutions.append(qpos_solution)

    qpos = qpos_solutions[0]
    if n_envs > 0:
        assert_equal(qpos_solutions[1], qpos_solutions[1][0])
        assert_allclose(qpos_solutions[1][0], qpos[0], tol=tol)

    # Apply the solution
    robot.set_qpos(qpos)

    # Verify the offset point is at the target position
    link_pos = end_effector.get_pos()
    link_quat = end_effector.get_quat()

    # Transform local offset to world frame
    world_offset = gu.transform_by_quat(local_offset, link_quat)
    actual_point_pos = link_pos + world_offset

    # Check that the offset point reached the target
    assert_allclose(actual_point_pos, target_pos, tol=tol)

    # Floating-base IK: place an offset point of the go2 trunk (its FREE root link) at a target pose, reached purely
    # through the FREE joint's translation and orientation degrees of freedom.
    base = floating.base_link
    base_offset = torch.tensor([0.1, 0.05, 0.0], dtype=gs.tc_float, device=gs.device)
    base_pos_all = [[0.1, 1.5, 0.6], [0.05, 1.45, 0.55]][:num_envs]
    base_quat_all = torch.tensor(
        [[0.9239, 0.0, 0.0, 0.3827], [1.0, 0.0, 0.0, 0.0]], dtype=gs.tc_float, device=gs.device
    )[:num_envs]

    base_qpos_live = floating.get_qpos().clone()
    base_links_pos_live = floating.get_links_pos(relative=False).clone()
    base_vgeoms_idx = [vgeom.idx for link in floating.links for vgeom in link.vgeoms]
    base_vgeoms_pos_live = scene.rigid_solver.get_vgeoms_pos(base_vgeoms_idx).clone()

    base_qpos, base_err = floating.inverse_kinematics(
        link=base,
        pos=base_pos_all,
        quat=base_quat_all,
        local_point=base_offset,
        max_solver_iters=100,
        pos_tol=tol,
        rot_tol=tol,
        return_error=True,
    )
    assert_allclose(base_err, 0.0, atol=tol)

    # The solve reports a configuration without moving the entity to it, so the live state is what it was.
    assert_allclose(floating.get_qpos(), base_qpos_live, tol=gs.EPS)
    assert_allclose(floating.get_links_pos(relative=False), base_links_pos_live, tol=gs.EPS)
    assert_allclose(scene.rigid_solver.get_vgeoms_pos(base_vgeoms_idx), base_vgeoms_pos_live, tol=gs.EPS)

    floating.set_qpos(base_qpos)
    assert_allclose(base.get_pos() + gu.transform_by_quat(base_offset, base.get_quat()), base_pos_all, tol=tol)
    base_rpy = gu.quat_to_xyz(gu.transform_quat_by_quat(gu.inv_quat(base_quat_all), base.get_quat()), rpy=True)
    assert_allclose(base_rpy, 0.0, atol=tol)

    # Also verify via forward kinematics
    links_pos, links_quat = scene.rigid_solver.forward_kinematics_query(robot, qpos)

    # Handle indexing based on n_envs
    if n_envs > 0:
        fk_link_pos = links_pos[:, end_effector.idx_local]
        fk_link_quat = links_quat[:, end_effector.idx_local]
    else:
        fk_link_pos = links_pos[end_effector.idx_local]
        fk_link_quat = links_quat[end_effector.idx_local]

    fk_world_offset = gu.transform_by_quat(local_offset, fk_link_quat)
    fk_actual_point_pos = fk_link_pos + fk_world_offset
    assert_allclose(fk_actual_point_pos, target_pos, tol=tol)

    if show_viewer:
        scene.visualizer.update()


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_inverse_kinematics_multilink_local_points(show_viewer, tol):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
        ),
    )
    scene.build()

    index_finger = robot.get_link("index_finger_distal")
    middle_finger = robot.get_link("middle_finger_distal")

    # Different local offsets for each finger (e.g., fingertip positions)
    index_local_offset = torch.tensor([0.0, 0.0, 0.02], dtype=gs.tc_float, device=gs.device)
    middle_local_offset = torch.tensor([0.0, 0.0, 0.02], dtype=gs.tc_float, device=gs.device)

    # Target positions for the fingertips
    index_target = torch.tensor([0.6, 0.5, 0.2], dtype=gs.tc_float, device=gs.device)
    middle_target = torch.tensor([0.63, 0.5, 0.2], dtype=gs.tc_float, device=gs.device)

    # Solve multi-link IK with local_points
    qpos, err = robot.inverse_kinematics_multilink(
        links=[index_finger, middle_finger],
        poss=[index_target, middle_target],
        local_points=[index_local_offset, middle_local_offset],
        pos_tol=tol,
        rot_tol=tol,
        max_solver_iters=100,
        return_error=True,
    )
    assert_allclose(err, 0.0, atol=tol)

    # Apply solution
    robot.set_qpos(qpos)
    if show_viewer:
        scene.visualizer.update(force=True)

    # Verify each offset point is at its target
    for link, local_offset, target in [
        (index_finger, index_local_offset, index_target),
        (middle_finger, middle_local_offset, middle_target),
    ]:
        link_pos = link.get_pos()
        link_quat = link.get_quat()
        world_offset = gu.transform_by_quat(local_offset, link_quat)
        actual_point_pos = link_pos + world_offset
        assert_allclose(actual_point_pos, target, tol=tol)


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_multi_robot_inverse_kinematics(show_viewer, tol):
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())

    robot_positions = [
        (0.0, -0.5, 0.005),
        (0.0, 0.0, 0.005),
        (0.0, 0.5, 0.005),
    ]
    robots: list[RigidEntity] = []
    for pos in robot_positions:
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda_non_overlap.xml",
                pos=pos,
                convexify=True,
            ),
        )
        robots.append(robot)

    scene.build()

    for robot, pos in zip(robots, robot_positions):
        target_pos = np.array(pos) + [0.4, 0.0, 0.4]
        qpos, err = robot.inverse_kinematics(
            link=robot.get_link("hand"),
            pos=target_pos,
            quat=[0, 1, 0, 0],
            pos_tol=tol,
            rot_tol=tol,
            max_solver_iters=100,
            return_error=True,
        )
        assert_allclose(err, 0.0, atol=tol)
        robot.set_qpos(qpos)
        ee_pos = robot.get_link("hand").get_pos()
        assert_allclose(target_pos, ee_pos, atol=tol)


@pytest.mark.slow("gpu")  # gpu ~300s
@pytest.mark.required
@pytest.mark.parametrize("n_envs, batch_dofs_info", [(0, True), (2, False), (2, True)])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_path_planning_avoidance(backend, n_envs, batch_dofs_info, show_viewer, tol):
    CUBE_SIZE = 0.07
    DOF_LIMIT = 0.5

    # FIXME: Implement a more robust plan planning algorithm
    if sys.platform == "darwin" and backend == gs.gpu:
        pytest.skip(reason="This algorithm is very fragile and fail to converge on MacOS.")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=batch_dofs_info,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cubes = []
    for pos_x in (-0.15, 0.15):
        for y_i in range(-3, 3):
            cube = scene.add_entity(
                gs.morphs.Box(
                    size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                    pos=(pos_x, CUBE_SIZE * y_i, 0.75),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(
                    color=(*np.random.rand(3), 0.7),
                ),
            )
            cubes.append(cube)
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=n_envs)
    collider_state = scene.rigid_solver.collider.collider_state

    hand = franka.get_link("hand")
    hand_pos_ref = torch.tensor([0.3, 0.1, 0.1], dtype=gs.tc_float, device=gs.device)
    hand_quat_ref = torch.tensor([0.3073, 0.5303, 0.7245, -0.2819], dtype=gs.tc_float, device=gs.device)
    if n_envs > 0:
        hand_pos_ref = hand_pos_ref.repeat((n_envs, 1))
        hand_quat_ref = hand_quat_ref.repeat((n_envs, 1))
    qpos_goal = franka.inverse_kinematics(hand, pos=hand_pos_ref, quat=hand_quat_ref)
    qpos_goal[..., -2:] = 0.04
    franka.set_qpos(qpos_goal)
    scene.visualizer.update()
    scene.rigid_solver.collider.detection()
    assert not collider_state.n_contacts.to_numpy().any()
    franka.set_qpos(torch.zeros_like(qpos_goal))

    num_waypoints = 300
    if n_envs == 0:
        free_path, return_valid_mask = franka.plan_path(
            qpos_goal=qpos_goal,
            num_waypoints=num_waypoints,
            resolution=0.05,
            ignore_collision=True,
            return_valid_mask=True,
        )
    else:
        return_valid_mask = torch.zeros((n_envs,), dtype=torch.bool, device=gs.device)
        free_path = torch.empty((num_waypoints, n_envs, franka.n_dofs), dtype=gs.tc_float, device=gs.device)
        for i in range(n_envs):
            free_path[:, i : i + 1], return_valid_mask[i : i + 1] = franka.plan_path(
                qpos_goal=qpos_goal[i : i + 1],
                envs_idx=[i],
                num_waypoints=num_waypoints,
                resolution=0.05,
                ignore_collision=True,
                return_valid_mask=True,
            )
    assert return_valid_mask.all()
    assert_allclose(free_path[0], 0.0, tol=tol)
    assert_allclose(free_path[-1], qpos_goal, tol=tol)

    avoidance_path, return_valid_mask = franka.plan_path(
        qpos_goal=qpos_goal,
        num_waypoints=300,
        ignore_collision=False,
        return_valid_mask=True,
        resolution=0.05,
        max_nodes=4000,
        max_retry=40,
    )
    assert return_valid_mask.all()
    assert_allclose(avoidance_path[0], 0.0, tol=tol)
    assert_allclose(avoidance_path[-1], qpos_goal, tol=tol)

    for path, avoid_collision in ((free_path, False), (avoidance_path, True)):
        max_penetration = float("-inf")
        for waypoint in path:
            franka.set_qpos(waypoint)
            scene.visualizer.update()

            # Check if the cube is colliding with the robot
            scene.rigid_solver.collider.detection()
            n_contacts = collider_state.n_contacts.to_numpy()
            for i_b in range(max(scene.n_envs, 1)):
                for i_c in range(n_contacts[i_b]):
                    contact_link_a = collider_state.contact_data.link_a[i_c, i_b]
                    contact_link_b = collider_state.contact_data.link_b[i_c, i_b]
                    penetration = collider_state.contact_data.penetration[i_c, i_b]
                    if any(i_g < len(cubes) for i_g in (contact_link_a, contact_link_b)):
                        max_penetration = max(max_penetration, penetration)

        args = (max_penetration, 5e-3)
        np.testing.assert_array_less(*(args if avoid_collision else args[::-1]))

        assert_allclose(hand_pos_ref, hand.get_pos(), tol=5e-4)
        hand_quat_diff = gu.transform_quat_by_quat(gu.inv_quat(hand_quat_ref), hand.get_quat())
        theta = 2 * torch.arctan2(torch.linalg.norm(hand_quat_diff[..., 1:]), torch.abs(hand_quat_diff[..., 0]))
        assert_allclose(theta, 0.0, tol=5e-3)

    # A limit tightened once the scene is built is the one the planner samples within: a goal it excludes cannot be
    # reached, while one it allows still can, and no waypoint leaves what the solver would let the robot hold.
    franka.set_dofs_limit(-DOF_LIMIT, DOF_LIMIT, dofs_idx_local=slice(0, 7))
    assert_allclose(franka.get_dofs_limit()[1][..., :7], DOF_LIMIT, tol=gs.EPS)
    franka.set_qpos(torch.zeros_like(qpos_goal))

    qpos_goal_within = torch.zeros_like(qpos_goal)
    qpos_goal_within[..., :7] = 0.5 * DOF_LIMIT
    within_path, within_valid_mask = franka.plan_path(
        qpos_goal=qpos_goal_within,
        num_waypoints=300,
        ignore_collision=True,
        resolution=0.05,
        return_valid_mask=True,
    )
    assert within_valid_mask.all()
    np.testing.assert_array_less(tensor_to_array(within_path[..., :7].abs()), DOF_LIMIT + tol)

    # Reachable for the limits the model was parsed with, which is what a snapshot taken at build would hold.
    qpos_goal_beyond = torch.zeros_like(qpos_goal)
    qpos_goal_beyond[..., 0] = 2.0 * DOF_LIMIT
    _, beyond_valid_mask = franka.plan_path(
        qpos_goal=qpos_goal_beyond,
        num_waypoints=300,
        ignore_collision=True,
        resolution=0.05,
        return_valid_mask=True,
    )
    assert not beyond_valid_mask.any()


@pytest.mark.required
def test_setters(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )

    ghost_box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.4, 0.2, 0.1),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 0.7),
        ),
    )
    ghost_robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_prismatic.urdf",
            fixed=False,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 0.7),
        ),
    )

    scene.build(n_envs=2)

    assert_allclose(ghost_box.get_vAABB(), ((-0.20, -0.10, -0.05), (0.20, 0.10, 0.05)), tol=tol)
    assert_allclose(ghost_robot.get_vAABB(), ((-0.05, -0.05, -0.05), (0.15, 0.05, 0.05)), tol=tol)

    ghost_box.set_pos([1.0, 2.0, 3.0])
    assert_allclose(ghost_box.get_vAABB(), ((0.80, 1.90, 2.95), (1.20, 2.10, 3.05)), tol=tol)
    ghost_box.set_pos([0.0, 0.0, 0.0])
    ghost_box.set_quat([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    assert_allclose(ghost_box.get_vAABB()[0], ((-0.20, -0.05, -0.10), (0.20, 0.05, 0.1)), tol=tol)
    assert_allclose(ghost_box.get_vAABB()[1], ((-0.05, -0.10, -0.2), (0.05, 0.10, 0.2)), tol=tol)

    n_dofs = ghost_robot.n_dofs
    dofs_idx_cases = (
        (-1, (n_dofs - 1,)),
        ([-n_dofs, -1], (0, n_dofs - 1)),
        ((-2, -1), (n_dofs - 2, n_dofs - 1)),
        (np.array((-2, -1), dtype=np.int32), (n_dofs - 2, n_dofs - 1)),
        (np.array((-2.0, -1.0), dtype=np.float64), (n_dofs - 2, n_dofs - 1)),
        (torch.tensor((-2, -1), dtype=torch.int32), (n_dofs - 2, n_dofs - 1)),
        (range(-n_dofs, 0), tuple(range(n_dofs))),
        (range(-2, 1), (n_dofs - 2, n_dofs - 1, 0)),
        (slice(-2, None), (n_dofs - 2, n_dofs - 1)),
        (slice(-1, None, -1), tuple(range(n_dofs - 1, -1, -1))),
        (np.array((False, True, True, True, True, True, False)), (1, 2, 3, 4, 5)),
        (torch.tensor((False, True, True, True, True, True, False)), (1, 2, 3, 4, 5)),
    )
    box_baseline = torch.arange(ghost_box.n_dofs, device=gs.device) * 0.01 + 0.1
    robot_baseline = torch.arange(n_dofs, device=gs.device) * 0.01 + 0.2
    ghost_box.set_dofs_position(box_baseline)
    ghost_robot.set_dofs_position(robot_baseline)

    for dofs_idx_local in (
        [-n_dofs - 1, -1],
        (-n_dofs - 1, -1),
        range(-n_dofs - 1, 0, n_dofs),
        np.array((-n_dofs - 1, -1), dtype=np.int32),
        torch.tensor((-n_dofs - 1, -1), dtype=torch.int32, device=gs.device),
    ):
        assert_equal(
            ghost_robot._get_global_idx(dofs_idx_local, n_dofs, ghost_robot.dof_start, unsafe=True),
            (ghost_robot.dof_start - n_dofs - 1, ghost_robot.dof_end - 1),
        )

    for dofs_idx_local, expected_dofs_idx_local in dofs_idx_cases:
        assert_equal(
            ghost_robot.get_dofs_position(dofs_idx_local),
            ghost_robot.get_dofs_position(expected_dofs_idx_local),
        )

        values = torch.arange(1, len(expected_dofs_idx_local) + 1, device=gs.device) * 0.01
        ghost_box.set_dofs_position(box_baseline)
        ghost_robot.set_dofs_position(robot_baseline)
        ghost_robot.set_dofs_position(values, dofs_idx_local=dofs_idx_local)
        box_result = ghost_box.get_dofs_position()
        robot_result = ghost_robot.get_dofs_position()

        ghost_box.set_dofs_position(box_baseline)
        ghost_robot.set_dofs_position(robot_baseline)
        ghost_robot.set_dofs_position(values, dofs_idx_local=expected_dofs_idx_local)
        assert_equal(box_result, ghost_box.get_dofs_position())
        assert_equal(robot_result, ghost_robot.get_dofs_position())

    for dofs_idx_local in (
        {0, 1},
        [[0, 1]],
        ["a"],
        np.array((True,), dtype=bool),
        torch.tensor((True,), dtype=torch.bool),
    ):
        with pytest.raises(gs.GenesisException):
            ghost_robot.get_dofs_position(dofs_idx_local)

    ghost_box.set_dofs_position(torch.zeros(ghost_box.n_dofs, device=gs.device))
    ghost_robot.set_dofs_position(torch.zeros(n_dofs, device=gs.device))
    ghost_robot.set_dofs_position([0.1, -0.1], dofs_idx_local=-1)
    assert_allclose(ghost_robot.get_vAABB()[0], ((-0.05, -0.05, -0.05), (0.25, 0.05, 0.05)), tol=tol)
    assert_allclose(ghost_robot.get_vAABB()[1], ((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)), tol=tol)

    ghost_robot.set_qpos([1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    assert_allclose(ghost_robot.get_vAABB(), ((0.95, 1.95, 2.95), (2.15, 2.05, 3.05)), tol=tol)

    frozen_vaabb = [tensor_to_array(entity.get_vAABB()) for entity in scene.entities]
    for _ in range(5):
        scene.step()
    assert_allclose([tensor_to_array(entity.get_vAABB()) for entity in scene.entities], frozen_vaabb, tol=gs.EPS)


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_track_rigid(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0.5, 0.42),
        ),
    )
    ghost = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0.5, 0.42),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.0, 1.0, 0.7),
        ),
    )
    scene.build(n_envs=2, env_spacing=(0.5, 0.5))

    for _ in range(20):
        scene.step()
        ghost.set_qpos(robot.get_qpos())
        assert_allclose(ghost.get_vAABB(), robot.get_vAABB(), tol=tol)
    assert_allclose(ghost.get_links_pos(), robot.get_links_pos(), tol=tol)

    ghost.set_dofs_velocity(robot.get_dofs_velocity())
    assert_allclose(ghost.get_links_vel(), robot.get_links_vel(), tol=tol)

    frozen_ghost_vaabb = ghost.get_vAABB()
    frozen_robot_vaabb = robot.get_vAABB()
    for _ in range(20):
        scene.step()

    assert_allclose(ghost.get_vAABB(), frozen_ghost_vaabb, tol=gs.EPS)
    with pytest.raises(AssertionError):
        assert_allclose(robot.get_vAABB(), frozen_robot_vaabb, atol=0.1)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinematics_queries_span_entities_and_solvers(n_envs, show_viewer, tol):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 1.0, 1.2),
            camera_lookat=(0.2, 0.6, 0.3),
        ),
        show_viewer=show_viewer,
    )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    go2 = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 1.5, 0.42),
        ),
    )
    fixed_box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.0, 0.0, 0.05),
            fixed=True,
        ),
    )
    ghost = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Kinematic(),
    )
    scene.build(n_envs=n_envs)

    # Queried at the configuration it already holds, every entity must report its live link poses: whatever its offset
    # in the solver configuration, with or without a degree of freedom, and whichever solver backs it.
    for entity in (franka, go2, fixed_box, ghost):
        links_pos, links_quat = entity.solver.forward_kinematics_query(entity, entity.get_qpos())
        assert_allclose(links_pos, entity.get_links_pos(relative=False), tol=gs.EPS)
        assert_allclose(links_quat, entity.get_links_quat(relative=False), tol=gs.EPS)

    # Callers hand the query whatever array-like configuration they hold, a NumPy trajectory included.
    links_pos, links_quat = franka.solver.forward_kinematics_query(franka, tensor_to_array(franka.get_qpos()))
    assert_allclose(links_pos, franka.get_links_pos(relative=False), tol=gs.EPS)
    assert_allclose(links_quat, franka.get_links_quat(relative=False), tol=gs.EPS)

    # Inverse kinematics reaches its target for an entity of either solver.
    target_pos = [[0.45, 0.1, 0.5], [0.4, 0.0, 0.55]][: max(n_envs, 1)]
    if n_envs == 0:
        target_pos = target_pos[0]
    for entity in (franka, ghost):
        end_effector = entity.get_link("hand")
        qpos_live = entity.get_qpos().clone()
        links_pos_live = entity.get_links_pos(relative=False).clone()
        vgeoms_idx = [vgeom.idx for link in entity.links for vgeom in link.vgeoms]
        vgeoms_pos_live = entity.solver.get_vgeoms_pos(vgeoms_idx).clone()

        qpos, err_pose = entity.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            max_solver_iters=100,
            pos_tol=tol,
            rot_tol=tol,
            return_error=True,
        )
        assert_allclose(err_pose, 0.0, atol=tol)

        # The solve reports a configuration without moving the entity to it, so the live state is what it was.
        assert_allclose(entity.get_qpos(), qpos_live, tol=gs.EPS)
        assert_allclose(entity.get_links_pos(relative=False), links_pos_live, tol=gs.EPS)
        assert_allclose(entity.solver.get_vgeoms_pos(vgeoms_idx), vgeoms_pos_live, tol=gs.EPS)
        with pytest.raises(AssertionError):
            assert_allclose(end_effector.get_pos(), target_pos, tol=tol)

        entity.set_qpos(qpos)
        assert_allclose(end_effector.get_pos(), target_pos, tol=tol)

    # Carrying more targets than the error buffer is sized for is rejected, rather than written past.
    n_tgts = franka.solver._options.IK_max_targets
    with pytest.raises(gs.GenesisException):
        franka.inverse_kinematics_multilink(
            links=franka.links[: n_tgts + 1],
            poss=[target_pos] * (n_tgts + 1),
        )
