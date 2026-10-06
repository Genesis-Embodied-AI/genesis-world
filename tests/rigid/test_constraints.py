import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs, batched", [(0, False), (2, True)])
def test_equality_joint_scaling(show_viewer, scaled_mjcf_joint_equalities, n_envs, batched, tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_joints_info=batched,
            batch_dofs_info=batched,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.15, -0.75, 4.0),
            camera_lookat=(0.15, -0.75, 0.0),
            camera_up=(0.0, 1.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    SCALE = 2.0
    entity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=scaled_mjcf_joint_equalities,
            scale=SCALE,
        ),
    )
    scene.build(n_envs=n_envs)

    COEFFICIENTS = (0.2, 0.4, -0.3, 0.2, -0.1)
    DRIVER_POSITION = 0.5
    FOLLOWER_POSITION = (
        COEFFICIENTS[0]
        + COEFFICIENTS[1] * DRIVER_POSITION
        + COEFFICIENTS[2] * DRIVER_POSITION**2
        + COEFFICIENTS[3] * DRIVER_POSITION**3
        + COEFFICIENTS[4] * DRIVER_POSITION**4
    )
    JOINT_PAIRS = (
        ("hinge_hinge", "hinge", "hinge"),
        ("slide_slide", "slide", "slide"),
        ("slide_hinge", "slide", "hinge"),
        ("hinge_slide", "hinge", "slide"),
    )
    TARGET_POSITION = 0.25
    UNRELATED_POSITION = 1.0
    qpos = entity.get_qpos()
    for name, driver_type, follower_type in JOINT_PAIRS:
        (i_driver_q,) = entity.get_joint(f"{name}_driver").qs_idx_local
        (i_follower_q,) = entity.get_joint(f"{name}_follower").qs_idx_local
        qpos[..., i_driver_q] = DRIVER_POSITION * (SCALE if driver_type == "slide" else 1.0)
        qpos[..., i_follower_q] = FOLLOWER_POSITION * (SCALE if follower_type == "slide" else 1.0)
    (i_target_q,) = entity.get_joint("target").qs_idx_local
    (i_unrelated_q,) = entity.get_joint("unrelated").qs_idx_local
    qpos[..., i_target_q] = TARGET_POSITION * SCALE
    qpos[..., i_unrelated_q] = UNRELATED_POSITION * SCALE
    entity.set_qpos(qpos)
    scene.step()

    qpos = entity.get_qpos()
    for name, driver_type, follower_type in JOINT_PAIRS:
        (i_driver_q,) = entity.get_joint(f"{name}_driver").qs_idx_local
        (i_follower_q,) = entity.get_joint(f"{name}_follower").qs_idx_local
        driver_scale = SCALE if driver_type == "slide" else 1.0
        follower_scale = SCALE if follower_type == "slide" else 1.0
        driver_position = qpos[..., i_driver_q] / driver_scale
        expected_follower_position = (
            COEFFICIENTS[0]
            + COEFFICIENTS[1] * driver_position
            + COEFFICIENTS[2] * driver_position**2
            + COEFFICIENTS[3] * driver_position**3
            + COEFFICIENTS[4] * driver_position**4
        )
        assert_allclose(qpos[..., i_follower_q] / follower_scale, expected_follower_position, tol=tol)

    assert_allclose(qpos[..., i_target_q] / SCALE, TARGET_POSITION, tol=tol)
    assert_allclose(qpos[..., i_unrelated_q] / SCALE, UNRELATED_POSITION, tol=tol)


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_dynamic_weld(show_viewer, tol):
    CUBE_POS = (0.65, 0.0, 0.02)
    HANGING_BOX_POS = (0.0, 1.0, 0.4)
    GRAVITY = 9.81
    # A constant impedance (dmin = dmax) makes the sag of the hanging box under its weight closed-form
    WELD_SOL_PARAMS = (0.05, 1.0, 0.95, 0.95, 0.001, 0.5, 2.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 0.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=CUBE_POS,
        ),
        surface=gs.surfaces.Default(
            color=(1, 0, 0),
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/universal_robots_ur5e/ur5e.xml",
        ),
    )
    fixed_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 1.0, 0.5),
            fixed=True,
        ),
    )
    hanging_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 1.0, 0.02),
        ),
    )
    scene.build(n_envs=4, env_spacing=(3.0, 3.0))

    end_effector = robot.get_link("ee_virtual_link")

    # Compute up and down robot configurations
    ee_pos_up = np.array((0.65, 0.0, 0.5), dtype=gs.np_float)
    ee_pos_down = np.array((0.65, 0.0, 0.15), dtype=gs.np_float)
    qpos_up = robot.inverse_kinematics(
        link=end_effector,
        pos=np.tile(ee_pos_up, (4, 1)),
        quat=np.tile(np.array((0.0, 1.0, 0.0, 0.0), dtype=gs.np_float), (4, 1)),
    )
    qpos_down = robot.inverse_kinematics(
        link=end_effector,
        pos=np.tile(ee_pos_down, (4, 1)),
        quat=np.tile(np.array((0.0, 1.0, 0.0, 0.0), dtype=gs.np_float), (4, 1)),
    )

    # move to pre-grasp pose
    robot.control_dofs_position(qpos_up)
    for i in range(120):
        scene.step()

    # reach
    robot.control_dofs_position(qpos_down)
    for i in range(70):
        scene.step()

    # add weld constraint and move back up. The hanging box is welded in the air afterwards, so that deleting the
    # cube weld goes through the swap-remove path and must preserve the full record of the hanging box weld.
    scene.sim.rigid_solver.add_weld_constraint(cube.base_link.idx, end_effector.idx, envs_idx=(0, 1, 2))
    hanging_box.set_pos(HANGING_BOX_POS)
    scene.sim.rigid_solver.add_weld_constraint(
        hanging_box.base_link.idx, fixed_box.base_link.idx, sol_params=WELD_SOL_PARAMS
    )
    robot.control_dofs_position(qpos_up)
    for _ in range(60):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), tensor_to_array(cube.get_quat())
    assert_allclose(gu.quat_to_rotvec(cubes_quat), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 2]], dim=0), 0.0, tol=tol)
    assert_allclose(cubes_pos[3], CUBE_POS, tol=1e-3)
    assert_allclose(cubes_pos[-1] - cubes_pos[0], ee_pos_down - ee_pos_up, tol=1e-2)
    # At rest, the reference acceleration of the weld, -k * d * sag, balances its regularization, (1 - d) / d * g,
    # with the stiffness k = 1 / (d * timeconst * dampratio)^2: the weld sags by (1 - d) * g * (timeconst * dampratio)^2.
    timeconst, dampratio, _dmin, dmax, *_ = WELD_SOL_PARAMS
    hanging_box_sag = (1.0 - dmax) * GRAVITY * (timeconst * dampratio) ** 2
    hanging_box_pos = (HANGING_BOX_POS[0], HANGING_BOX_POS[1], HANGING_BOX_POS[2] - hanging_box_sag)
    assert_allclose(hanging_box.get_pos(), hanging_box_pos, tol=1e-5)

    # drop
    scene.sim.rigid_solver.delete_weld_constraint(cube.base_link.idx, end_effector.idx, envs_idx=(0, 1))
    weld_const_info = scene.sim.rigid_solver.get_weld_constraints(as_tensor=True, to_torch=True)
    links_ab = torch.stack((weld_const_info["link_a"], weld_const_info["link_b"]), dim=-1)
    box_weld = (hanging_box.base_link.idx, fixed_box.base_link.idx)
    cube_weld = (cube.base_link.idx, end_effector.idx)
    no_weld = (-1, -1)
    assert_equal(links_ab, [[box_weld, no_weld], [box_weld, no_weld], [cube_weld, box_weld], [box_weld, no_weld]])
    for _ in range(110):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), tensor_to_array(cube.get_quat())
    assert_allclose(gu.quat_to_rotvec(cubes_quat), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 3]], dim=0), 0.0, tol=1e-2)
    assert_allclose(cubes_pos[2] - cubes_pos[0], ee_pos_up - ee_pos_down, tol=1e-3)
    assert_allclose(hanging_box.get_pos(), hanging_box_pos, tol=1e-5)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_dynamic_screw(n_envs, show_viewer, tol):
    DT = 0.01
    PITCH = 1.5e-3
    TORQUE = 1.5e-4
    FORCE = 2.0e-3
    UPPER = 4.0e-3
    # A soft constraint yields under load in proportion to its regularization (1 - impedance): the stiffest impedance
    # keeps the loaded screws on their analytical motion, while the screw turning in place keeps the default parameters.
    SOL_PARAMS = (0.02, 1.0, 0.9999, 0.9999, 0.001, 0.5, 2.0)
    N_DRIVEN, N_RELEASED = 50, 30

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -1.0, 0.6),
            camera_lookat=(0.6, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    # One screw per pair of boxes, tilted so that its axis is none of the world axes: a finite pitch, an infinite one
    # (sliding), a zero one (turning in place), a finite one with friction, and a finite one with an upper limit.
    sockets, parts = [], []
    for i in range(5):
        sockets.append(
            scene.add_entity(
                gs.morphs.Box(
                    size=(0.06, 0.06, 0.06),
                    pos=(0.3 * i, 0.0, 0.0),
                    euler=(30.0, 45.0, 0.0),
                    fixed=True,
                ),
            )
        )
        parts.append(
            scene.add_entity(
                gs.morphs.Box(
                    size=(0.02, 0.02, 0.04),
                    pos=(0.3 * i, 0.0, 0.2),
                ),
            )
        )
    scene.build(n_envs=n_envs)

    # Each part starts aligned with its socket, half sunk into it through the center of its top face: a contact would
    # throw it out, so the motions below also check that the two links of a screw stop colliding.
    socket_quat = sockets[0].get_quat()
    axis = gu.transform_by_quat(torch.tensor((0.0, 0.0, 1.0), dtype=gs.tc_float, device=gs.device), socket_quat)
    for socket, part in zip(sockets, parts):
        part.set_pos(socket.get_pos() + 0.03 * axis)
        part.set_quat(socket_quat)
    axis = np.atleast_2d(tensor_to_array(axis))
    parts_pos_0 = np.stack([np.atleast_2d(tensor_to_array(part.get_pos())) for part in parts], axis=-2)

    solver = scene.sim.rigid_solver
    links_idx = [(socket.base_link.idx, part.base_link.idx) for socket, part in zip(sockets, parts)]
    parts_link_idx = [link_idx for _, link_idx in links_idx]
    solver.add_screw_constraint(*links_idx[0], axis=(0.0, 0.0, 1.0), pitch=PITCH, sol_params=SOL_PARAMS)
    solver.add_screw_constraint(*links_idx[1], axis=(0.0, 0.0, 2.0), pitch=float("inf"), sol_params=SOL_PARAMS)
    solver.add_screw_constraint(*links_idx[2], axis=(0.0, 0.0, 1.0), pitch=0.0)
    solver.add_screw_constraint(
        *links_idx[3],
        axis=(0.0, 0.0, 1.0),
        pitch=PITCH,
        pos=(0.0, 0.0, 0.01),
        frictionloss=0.5 * TORQUE,
        sol_params=SOL_PARAMS,
    )
    solver.add_screw_constraint(
        *links_idx[4], axis=(0.0, 0.0, 1.0), pitch=PITCH, limit=(-float("inf"), UPPER), sol_params=SOL_PARAMS
    )
    with pytest.raises(gs.GenesisException):
        solver.add_screw_constraint(*links_idx[0], axis=(0.0, 0.0, 1.0), pitch=PITCH)
    with pytest.raises(gs.GenesisException):
        solver.add_screw_constraint(*links_idx[0], axis=(0.0, 0.0, 0.0), pitch=PITCH)
    with pytest.raises(gs.GenesisException):
        solver.add_screw_constraint(*links_idx[2], axis=(0.0, 0.0, 1.0), pitch=0.0, limit=(0.0, UPPER))

    # A torque about the screw axis on every part and a force along it on the sliding and turning ones, in the frames of
    # the parts (which only turn about the axis): the sliding screw resists the torque, the turning one the force.
    force = ((0.0, 0.0, 0.0), (0.0, 0.0, FORCE), (0.0, 0.0, FORCE), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for _ in range(N_DRIVEN):
        solver.apply_links_external_wrench(force=force, torque=(0.0, 0.0, TORQUE), links_idx=parts_link_idx, local=True)
        scene.step()

    # Constant accelerations integrated by the semi-implicit Euler scheme from rest give x_N = a * dt^2 * N * (N + 1) / 2.
    # A screw turns its part along with its travel, so the turn sees the inertia of both, inertia + mass * pitch^2.
    # Friction takes half of the torque, and the limited screw rests at its limit.
    mass = tensor_to_array(parts[0].get_links_mass())[..., 0]
    inertia = tensor_to_array(parts[0].get_links_inertia())[..., 0, 2, 2]
    inertia_screw = inertia + mass * PITCH**2
    driven_factor = DT**2 * N_DRIVEN * (N_DRIVEN + 1) / 2
    turn_expected = np.array(
        [
            TORQUE / inertia_screw * driven_factor,
            0.0,
            TORQUE / inertia * driven_factor,
            0.5 * TORQUE / inertia_screw * driven_factor,
            UPPER / PITCH,
        ]
    )
    travel_expected = np.array(
        [PITCH * turn_expected[0], FORCE / mass * driven_factor, 0.0, PITCH * turn_expected[3], UPPER]
    )
    parts_pos = np.stack([np.atleast_2d(tensor_to_array(part.get_pos())) for part in parts], axis=-2)
    parts_quat = np.stack([np.atleast_2d(tensor_to_array(part.get_quat())) for part in parts], axis=-2)
    assert_allclose(parts_pos, parts_pos_0 + travel_expected[:, None] * axis[:, None], tol=1e-5)
    quat_turn = gu.axis_angle_to_quat(turn_expected, np.array((0.0, 0.0, 1.0), dtype=gs.np_float))
    quat_expected = gu.transform_quat_by_quat(
        np.tile(quat_turn, (len(parts_quat), 1, 1)),
        np.repeat(np.atleast_2d(tensor_to_array(socket_quat))[:, None], len(parts), axis=1),
    )
    quat_error = gu.transform_quat_by_quat(parts_quat, gu.inv_quat(quat_expected))
    assert_allclose(gu.quat_to_rotvec(quat_error), 0.0, tol=1e-3)
    screw_const_info = solver.get_screw_constraints(as_tensor=True, to_torch=False)
    assert_allclose(screw_const_info["travel"], np.broadcast_to(travel_expected, parts_pos.shape[:-1]), tol=1e-5)
    assert_equal(screw_const_info["link_b"], np.broadcast_to(parts_link_idx, parts_pos.shape[:-1]))

    # Released in the first env alone, the first part travels freely under an axial force and keeps spinning as it was.
    # Still screwed in the others, the force turns it instead, through the inertia it sees along the axis.
    vel_0 = np.atleast_2d(tensor_to_array(parts[0].get_vel()))
    ang_0 = np.atleast_2d(tensor_to_array(parts[0].get_ang()))
    pos_0 = np.atleast_2d(tensor_to_array(parts[0].get_pos()))
    solver.delete_screw_constraint(*links_idx[0], envs_idx=[0] if n_envs > 0 else None)
    # Deleting a constraint moves the last one of its env into its slot, so each env is compared as a set.
    screw_const_info = solver.get_screw_constraints(as_tensor=False, to_torch=False)
    links_b_expected = [parts_link_idx[1:], *([parts_link_idx] * (n_envs - 1))]
    for links_b, links_b_expected_i in zip(screw_const_info["link_b"], links_b_expected, strict=True):
        assert_equal(np.sort(links_b), links_b_expected_i)
    for _ in range(N_RELEASED):
        solver.apply_links_external_wrench(force=(0.0, 0.0, FORCE), links_idx=parts_link_idx[:1], local=True)
        scene.step()
    released_factor = DT**2 * N_RELEASED * (N_RELEASED + 1) / 2
    travel_free = N_RELEASED * DT * (vel_0 * axis).sum(axis=-1) + FORCE / mass * released_factor
    travel_screwed = N_RELEASED * DT * (vel_0 * axis).sum(axis=-1) + PITCH**2 * FORCE / inertia_screw * released_factor
    pos = np.atleast_2d(tensor_to_array(parts[0].get_pos()))
    ang = np.atleast_2d(tensor_to_array(parts[0].get_ang()))
    assert_allclose(pos[0], pos_0[0] + N_RELEASED * DT * vel_0[0] + FORCE / mass * released_factor * axis[0], tol=1e-5)
    # The free part spins about its symmetry axis, a rate its Euler equations conserve (the rest of its angular velocity
    # nutates with whatever sideways spin the screw released it with).
    assert_allclose((ang[0] * axis[0]).sum(), (ang_0[0] * axis[0]).sum(), tol=1e-5)
    assert_allclose(((pos - pos_0) * axis).sum(axis=-1)[1:], travel_screwed[1:], tol=1e-5)
    assert_allclose(((pos - pos_0) * axis).sum(axis=-1)[0], travel_free[0], tol=1e-5)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_dynamic_weld_scene_reset():
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            max_dynamic_constraints=10,
        ),
        show_viewer=False,
    )
    box1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0, 0, 0.5),
        )
    )
    box2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.2, 0, 0.5),
        )
    )
    scene.build(n_envs=2)

    solver = scene.rigid_solver
    n_eq_base = solver.rigid_info.n_equalities[None]

    solver.add_weld_constraint(box1.base_link_idx, box2.base_link_idx)
    assert solver.constraint_solver.constraint_state.qd_n_equalities[0] == n_eq_base + 1
    assert solver.constraint_solver.constraint_state.qd_n_equalities[1] == n_eq_base + 1

    scene.reset(state=scene.get_state(), envs_idx=[0])
    assert solver.constraint_solver.constraint_state.qd_n_equalities[0] == n_eq_base
    assert solver.constraint_solver.constraint_state.qd_n_equalities[1] == n_eq_base + 1


@pytest.mark.required
def test_urdf_mimic(show_viewer, tol, scaled_urdf_mimic):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    hand = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/hand.urdf",
            fixed=True,
        ),
    )
    mimic = scene.add_entity(
        gs.morphs.URDF(
            file=scaled_urdf_mimic,
            scale=2.0,
            fixed=True,
        ),
    )
    scene.build()
    assert scene.rigid_solver.n_equalities == 5

    JOINT_NAMES = (
        "revolute_revolute_driver_joint",
        "revolute_revolute_follower_joint",
        "prismatic_prismatic_driver_joint",
        "prismatic_prismatic_follower_joint",
        "prismatic_revolute_driver_joint",
        "prismatic_revolute_follower_joint",
        "revolute_prismatic_driver_joint",
        "revolute_prismatic_follower_joint",
    )
    qs_idx_local = [idx for name in JOINT_NAMES for idx in mimic.get_joint(name).qs_idx_local]
    hand.set_dofs_velocity((0.0, 1.0))
    for _ in range(80):
        scene.step()

    qpos = mimic.get_qpos(qs_idx_local=qs_idx_local)
    assert_allclose(qpos[..., 1] - 2.0 * qpos[..., 0], 0.25, tol=tol)
    assert_allclose(qpos[..., 3] - 2.0 * qpos[..., 2], 0.5, tol=tol)
    assert_allclose(qpos[..., 5] - qpos[..., 4], 0.25, tol=tol)
    assert_allclose(qpos[..., 7] - 4.0 * qpos[..., 6], 0.5, tol=tol)

    hand_qpos = hand.get_qpos()
    assert_allclose(hand_qpos[..., -1], hand_qpos[..., -2], tol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_get_constraints_api(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.2, 0.0, 0.05),
        )
    )
    scene.build(n_envs=2)

    link_a, link_b = robot.base_link.idx, cube.base_link.idx
    scene.sim.rigid_solver.add_weld_constraint(link_a, link_b, envs_idx=[1])
    with np.testing.assert_raises(AssertionError):
        scene.sim.rigid_solver.add_weld_constraint(link_a, link_b, envs_idx=[1])

    for as_tensor, to_torch in ((True, True), (True, False), (False, True), (False, False)):
        weld_const_info = scene.sim.rigid_solver.get_weld_constraints(as_tensor, to_torch)
        link_a_, link_b_ = weld_const_info["link_a"], weld_const_info["link_b"]
        if as_tensor:
            assert_allclose((link_a_[0], link_b_[0]), ((-1,), (-1,)), tol=0)
        else:
            assert_allclose((link_a_[0], link_b_[0]), ((), ()), tol=0)
        assert_allclose((link_a_[1], link_b_[1]), ((link_a,), (link_b,)), tol=0)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs, batched", [(0, False), (3, True)])
def test_set_sol_params(n_envs, batched, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_joints_info=batched,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.4, 0.1),
            euler=(0, 0, 90),
        ),
    )
    scene.build(n_envs=2)
    assert scene.sim._substep_dt == 0.01

    for objs, batched in ((robot.joints, batched), (robot.geoms, False), (robot.equalities, True)):
        for obj in objs:
            sol_params = obj.get_sol_params() + 1.0
            obj.set_sol_params(sol_params)
            with pytest.raises(AssertionError):
                assert_allclose(obj.get_sol_params(), sol_params, tol=tol)
            obj.set_sol_params([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
            assert_allclose(obj.get_sol_params(), [2.0e-02, 0.5, 1e-4, 1e-4, 0.0, 1e-4, 1.0], tol=tol)
