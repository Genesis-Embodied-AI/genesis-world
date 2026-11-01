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


@pytest.mark.required
def test_weld_and_connect(connected_boxes, show_viewer, tol):
    CUBE_POS = (0.65, 0.0, 0.02)
    # The hanging box is welded overlapping the fixed box, which only holds because the weld stops their collision
    HANGING_BOX_POS = (0.0, 1.0, 0.48)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, 0.0, 0.6),
            camera_lookat=(0.2, 0.0, 0.3),
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
    # The hung box swings on a connect pivot at the top of the gap between the two boxes into the fixed box, which stops
    # it since connected links keep colliding, where it would otherwise swing on to hang under the pivot.
    boxes = scene.add_entity(
        gs.morphs.MJCF(
            file=connected_boxes,
            pos=(0.0, -1.0, 0.0),
        ),
    )
    scene.build(n_envs=4, env_spacing=(3.0, 3.0))

    # At rest, the hung box leans on the fixed one by the inner edge of its bottom face, rotated about the pivot by the
    # angle phi such that gap / 2 * cos(phi) - 2 * h * sin(phi) = -gap / 2, h being the half size of the boxes.
    BOX_HALF_SIZE = 0.02
    box_hung = boxes.get_link("box_hung")
    (connect,) = boxes.equalities
    gap = 2.0 * (connect.eq_data[0] - BOX_HALF_SIZE)
    rest_angle = np.arctan2(-2.0 * BOX_HALF_SIZE, 0.5 * gap) + np.arccos(
        -0.5 * gap / np.hypot(0.5 * gap, 2.0 * BOX_HALF_SIZE)
    )

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
    scene.sim.rigid_solver.add_weld_constraint(hanging_box.base_link.idx, fixed_box.base_link.idx)
    robot.control_dofs_position(qpos_up)
    for _ in range(60):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), tensor_to_array(cube.get_quat())
    assert_allclose(gu.quat_to_rotvec(cubes_quat), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 2]], dim=0), 0.0, tol=tol)
    assert_allclose(cubes_pos[3], CUBE_POS, tol=1e-3)
    assert_allclose(cubes_pos[-1] - cubes_pos[0], ee_pos_down - ee_pos_up, tol=1e-2)
    assert_allclose(hanging_box.get_pos(), HANGING_BOX_POS, tol=1e-3)

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
    assert_allclose(hanging_box.get_pos(), HANGING_BOX_POS, tol=1e-3)
    assert_allclose(gu.quat_to_rotvec(tensor_to_array(box_hung.get_quat())), (0.0, rest_angle, 0.0), atol=1e-2)


@pytest.mark.required
def test_dynamic_screw(swinging_spindle, show_viewer, tol):
    DT = 0.01
    SUBSTEPS = 2
    GRAVITY = 9.81
    PITCH = 0.06
    LIMIT_LOWER = -0.01
    POST_HEIGHT = 0.5
    NUT_Z = 0.4
    N_STEPS_SCREWED = 30
    HINGE_SPIN_RATE = 2.0
    SWING_RATE = 3.0
    N_STEPS_RELEASED = 10

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=SUBSTEPS,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
            max_dynamic_constraints=7,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -5.0, 1.85),
            camera_lookat=(3.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    # Every nut is screwed to a post going through it, the two staying overlapped because the screw stops their collision
    post_helix, post_free, post_held, post_stopped = (
        scene.add_entity(
            gs.morphs.Box(
                size=(0.02, 0.02, POST_HEIGHT),
                pos=(x, 0.0, 0.5 * POST_HEIGHT),
                fixed=True,
            ),
            vis_mode="collision",
        )
        for x in (0.0, 1.0, 2.0, 3.0)
    )
    nut_helix, nut_free, nut_held, nut_stopped = (
        scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.02),
                pos=(x, 0.0, NUT_Z),
            ),
            vis_mode="collision",
        )
        for x in (0.0, 1.0, 2.0, 3.0)
    )
    # The resting nut starts above the tip of its post, onto which it falls and rests until a screw takes it down
    post_resting = scene.add_entity(
        gs.morphs.Box(
            size=(0.02, 0.02, POST_HEIGHT),
            pos=(4.0, 0.0, 0.5 * POST_HEIGHT),
            fixed=True,
        ),
        vis_mode="collision",
    )
    nut_resting = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.02),
            pos=(4.0, 0.0, POST_HEIGHT + 0.015),
        ),
        vis_mode="collision",
    )
    # The spindle swings, carrying a free nut and a nut sharing its kinematic tree, which slides and turns on it
    # through two joints.
    spindle = scene.add_entity(
        gs.morphs.MJCF(
            file=swinging_spindle,
            pos=(6.0, 0.0, 0.0),
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=2, env_spacing=(0.0, 1.5))

    # Every nut but the helix one is screwed differently in each environment.
    rigid = scene.sim.rigid_solver
    rigid.add_screw_constraint(post_helix.base_link.idx, nut_helix.base_link.idx, axis=(0.0, 0.0, 1.0), pitch=PITCH)

    # The free nut slides in the first environment and spins on a hinge in the second.
    rigid.add_screw_constraint(
        post_free.base_link.idx, nut_free.base_link.idx, axis=(0.0, 0.0, 1.0), pitch=np.inf, envs_idx=[0]
    )
    rigid.add_screw_constraint(
        post_free.base_link.idx, nut_free.base_link.idx, axis=(0.0, 0.0, 1.0), pitch=0.0, envs_idx=[1]
    )
    nut_free.set_dofs_velocity(HINGE_SPIN_RATE, dofs_idx_local=5, envs_idx=[1])

    # The held nut is held by friction against its turn in the first environment, and along its travel in the second.
    rigid.add_screw_constraint(
        post_held.base_link.idx,
        nut_held.base_link.idx,
        axis=(0.0, 0.0, 1.0),
        pitch=PITCH,
        frictionloss=1.0,
        envs_idx=[0],
    )
    rigid.add_screw_constraint(
        post_held.base_link.idx,
        nut_held.base_link.idx,
        axis=(0.0, 0.0, 1.0),
        pitch=np.inf,
        frictionloss=10.0,
        envs_idx=[1],
    )

    # The stopped nut is seated by a lower limit on an axis going through the center of its post in the first
    # environment, and stopped by an upper limit on an axis pointing down in the second.
    rigid.add_screw_constraint(
        post_stopped.base_link.idx,
        nut_stopped.base_link.idx,
        axis=(0.0, 0.0, 1.0),
        pitch=PITCH,
        pos=(0.0, 0.0, 0.0),
        limit=(LIMIT_LOWER, np.inf),
        envs_idx=[0],
    )
    rigid.add_screw_constraint(
        post_stopped.base_link.idx,
        nut_stopped.base_link.idx,
        axis=(0.0, 0.0, -1.0),
        pitch=PITCH,
        limit=(-np.inf, -LIMIT_LOWER),
        envs_idx=[1],
    )

    spindle_link = spindle.get_link("spindle")
    free_nut_link = spindle.get_link("free_nut")
    for nut_link in (spindle.get_link("nut"), free_nut_link):
        rigid.add_screw_constraint(spindle_link.idx, nut_link.idx, axis=(0.0, 0.0, 1.0), pitch=PITCH)
    spindle.set_dofs_velocity(SWING_RATE, dofs_idx_local=0)

    with pytest.raises(gs.GenesisException, match="already coupled"):
        rigid.add_screw_constraint(nut_helix.base_link.idx, post_helix.base_link.idx, axis=(0.0, 0.0, 1.0), pitch=PITCH)
    with pytest.raises(gs.GenesisException, match="Invalid screw parameters"):
        rigid.add_screw_constraint(
            post_helix.base_link.idx, nut_helix.base_link.idx, axis=(0.0, 0.0, np.inf), pitch=PITCH
        )

    # The nut descends with the acceleration g_eff of a mass m sliding along the axis while spinning with an axial
    # inertia I, the two being coupled by the travel per radian p: g_eff = g * m * p^2 / (m * p^2 + I). The
    # compliance of the coupling row lets the turn lag the travel slightly, hence the relative tolerances.
    mass = tensor_to_array(nut_helix.get_links_mass())[0]
    inertia = tensor_to_array(nut_helix.get_links_inertia())[0, 2, 2]
    pitch_rad = PITCH / (2.0 * np.pi)
    g_eff = GRAVITY * mass * pitch_rad**2 / (mass * pitch_rad**2 + inertia)
    # Semi-implicit Euler moves a body under a constant acceleration a by -a * dt^2 * n * (n + 1) / 2 after n substeps.
    n_substeps = N_STEPS_SCREWED * SUBSTEPS
    fall_factor = (DT / SUBSTEPS) ** 2 * n_substeps * (n_substeps + 1) / 2
    for _ in range(N_STEPS_SCREWED):
        scene.step()
    # The post is fixed and the screw axis is the world z axis, so the travel of a nut is its height change.
    travel_helix = nut_helix.get_pos()[..., 2] - NUT_Z
    assert_allclose(travel_helix, -g_eff * fall_factor, rtol=0.01)
    assert_allclose(
        gu.quat_to_rotvec(tensor_to_array(nut_helix.get_quat()))[..., 2], travel_helix / pitch_rad, tol=0.02
    )

    free_travel = nut_free.get_pos()[..., 2] - NUT_Z
    free_turn = gu.quat_to_rotvec(tensor_to_array(nut_free.get_quat()))[..., 2]
    assert_allclose(free_travel[0], -GRAVITY * fall_factor, tol=tol)
    assert_allclose(free_turn[0], 0.0, atol=1e-3)
    assert_allclose(free_travel[1], 0.0, atol=2e-4)
    assert_allclose(free_turn[1], HINGE_SPIN_RATE * N_STEPS_SCREWED * DT, tol=1e-2)

    held_travel = nut_held.get_pos()[..., 2] - NUT_Z
    assert_allclose(held_travel[0], 0.0, atol=5e-4)
    assert_allclose(held_travel[1], 0.0, atol=2e-3)

    # Both nuts follow the axis of the swinging spindle, and the travel of the jointed one follows its turn.
    swing, travel_spindle, turn_spindle = spindle.get_qpos()[..., :3].unbind(-1)
    assert (swing.abs() > 0.1).all()
    assert_allclose(travel_spindle, turn_spindle * pitch_rad, atol=1e-4)
    free_nut_pos = gu.inv_transform_by_trans_quat(
        free_nut_link.get_pos(), spindle_link.get_pos(), spindle_link.get_quat()
    )
    assert_allclose(free_nut_pos[..., :2], 0.0, atol=2e-5)

    rest_z = POST_HEIGHT + 0.01
    assert_allclose(nut_resting.get_pos()[..., 2], rest_z, atol=1e-3)

    # In the second environment, the helix nut is released, colliding with its post again while it keeps screwing down
    # in the first, and the resting nut is screwed, descending through its post while it keeps resting on it in the
    # first. Each environment then has its own valid collision pairs.
    rigid.delete_screw_constraint(nut_helix.base_link.idx, post_helix.base_link.idx, envs_idx=[1])
    rigid.add_screw_constraint(
        post_resting.base_link.idx, nut_resting.base_link.idx, axis=(0.0, 0.0, 1.0), pitch=PITCH, envs_idx=[1]
    )
    vel_z = nut_helix.get_vel()[..., 2]
    vel_z_resting = nut_resting.get_vel()[..., 2]
    scene.step()
    assert_equal(nut_helix.get_contacts(with_entity=post_helix)["valid_mask"].any(dim=-1), (False, True))
    assert_equal(nut_resting.get_contacts(with_entity=post_resting)["valid_mask"].any(dim=-1), (True, False))
    for _ in range(N_STEPS_RELEASED - 1):
        scene.step()
    assert_allclose(nut_helix.get_vel()[0, 2] - vel_z[0], -g_eff * N_STEPS_RELEASED * DT, rtol=0.01)
    assert_allclose(nut_resting.get_pos()[0, 2], rest_z, atol=1e-3)
    assert_allclose(nut_resting.get_vel()[1, 2] - vel_z_resting[1], -g_eff * N_STEPS_RELEASED * DT, rtol=0.01)

    # Both limits stop the nut at the same height, the travel being counted down along the axis in the second
    # environment.
    assert_allclose(nut_stopped.get_pos()[..., 2] - NUT_Z, LIMIT_LOWER, tol=5e-4)

    # An environment selected twice gets the screw once.
    rigid.add_screw_constraint(
        post_helix.base_link.idx, nut_helix.base_link.idx, axis=(0.0, 0.0, 1.0), pitch=PITCH, envs_idx=[1, 1]
    )
    n_screws = (rigid.get_equality_constraints()["type"] == gs.EQUALITY_TYPE.SCREW).sum(dim=-1)
    assert_equal(n_screws, (6, 7))
    with pytest.raises(gs.GenesisException, match="max_dynamic_constraints"):
        rigid.add_weld_constraint(post_helix.base_link.idx, nut_helix.base_link.idx, envs_idx=[1])


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
    with pytest.raises(gs.GenesisException, match="already coupled"):
        scene.sim.rigid_solver.add_weld_constraint(link_a, link_b, envs_idx=[1])

    for as_tensor, to_torch in ((True, True), (True, False), (False, True), (False, False)):
        weld_const_info = scene.sim.rigid_solver.get_weld_constraints(as_tensor, to_torch)
        link_a_, link_b_ = weld_const_info["link_a"], weld_const_info["link_b"]
        if as_tensor:
            assert_allclose((link_a_[0], link_b_[0]), ((-1,), (-1,)), tol=0)
        else:
            assert_allclose((link_a_[0], link_b_[0]), ((), ()), tol=0)
        assert_allclose((link_a_[1], link_b_[1]), ((link_a,), (link_b,)), tol=0)


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
