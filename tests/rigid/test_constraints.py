import mujoco
import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.mujoco_parity import simulate_and_check_mujoco_consistency


@pytest.mark.parametrize("model_name", ["mimic_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_joint(gs_sim, mj_sim, gs_solver, tol):
    # there is an equality constraint
    assert gs_sim.rigid_solver.n_equalities == 1

    qpos = np.array((0.0, -1.0))
    qvel = np.array((1.0, -0.3))
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=300, tol=tol)

    # check if the two joints are equal
    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(gs_qpos[0], gs_qpos[1], tol=tol)


@pytest.mark.required
def test_equality_joint_scaling(show_viewer, scaled_mjcf_joint_equalities, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_joint_limit=False,
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
    scene.build()

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
    qpos = entity.get_qpos()
    for name, driver_type, follower_type in JOINT_PAIRS:
        (driver_idx,) = entity.get_joint(f"{name}_driver").qs_idx_local
        (follower_idx,) = entity.get_joint(f"{name}_follower").qs_idx_local
        qpos[..., driver_idx] = DRIVER_POSITION * (SCALE if driver_type == "slide" else 1.0)
        qpos[..., follower_idx] = FOLLOWER_POSITION * (SCALE if follower_type == "slide" else 1.0)
    entity.set_qpos(qpos)
    scene.step()

    qpos = entity.get_qpos()
    for name, driver_type, follower_type in JOINT_PAIRS:
        (driver_idx,) = entity.get_joint(f"{name}_driver").qs_idx_local
        (follower_idx,) = entity.get_joint(f"{name}_follower").qs_idx_local
        driver_scale = SCALE if driver_type == "slide" else 1.0
        follower_scale = SCALE if follower_type == "slide" else 1.0
        driver_position = qpos[..., driver_idx] / driver_scale
        expected_follower_position = (
            COEFFICIENTS[0]
            + COEFFICIENTS[1] * driver_position
            + COEFFICIENTS[2] * driver_position**2
            + COEFFICIENTS[3] * driver_position**3
            + COEFFICIENTS[4] * driver_position**4
        )
        assert_allclose(qpos[..., follower_idx] / follower_scale, expected_follower_position, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs, batch_info", [(0, False), (2, True)])
def test_equality_joint_without_joint2(show_viewer, single_joint_equality, n_envs, batch_info, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_joint_limit=False,
            batch_joints_info=batch_info,
            batch_dofs_info=batch_info,
        ),
        show_viewer=show_viewer,
    )
    SCALE = 2.0
    entity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=single_joint_equality,
            scale=SCALE,
        ),
    )
    scene.build(n_envs=n_envs)

    TARGET_POSITION = 0.25
    UNRELATED_POSITION = 1.0
    (target_idx,) = entity.get_joint("target").qs_idx_local
    (unrelated_idx,) = entity.get_joint("unrelated").qs_idx_local
    qpos = entity.get_qpos()
    qpos[..., unrelated_idx] = UNRELATED_POSITION * SCALE
    entity.set_qpos(qpos)
    for _ in range(80):
        scene.step()

    qpos = entity.get_qpos() / SCALE
    assert_allclose(qpos[..., target_idx], TARGET_POSITION, atol=2e-3)
    assert_allclose(qpos[..., unrelated_idx], UNRELATED_POSITION, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/four_bar_linkage_weld.xml", "weld.xml", "connect.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_link(gs_sim, mj_sim, gs_solver, xml_path):
    # Must disable self-collision caused by closing the kinematic chain (adjacent link filtering is not enough)
    gs_sim.rigid_solver._enable_collision = False
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Must the time constant of the constraints to improve numerical stability
    TIME_CONSTANT = 0.02
    for entity in gs_sim.entities:
        for equality in entity.equalities:
            equality.set_sol_params((TIME_CONSTANT, *tensor_to_array(equality.desc.sol_params)[1:]))
    mj_sim.model.eq_solref[:, 0] = TIME_CONSTANT

    # Randomize the initial condition for force convergence of the constraints
    np.random.seed(0)
    qpos = np.random.rand(gs_sim.rigid_solver.n_qs) * 0.1

    # Note that the world frame in which weld constraint is computed is different between Mujoco and Genesis for sites.
    # Mujoco is using site 1, whereas Genesis is using parent link frame of site 1 since it has no notion of site.
    ignore_constraints = np.any(
        (mj_sim.model.eq_objtype == mujoco.mjtObj.mjOBJ_SITE) & (mj_sim.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    )
    simulate_and_check_mujoco_consistency(
        gs_sim, mj_sim, qpos, num_steps=300, tol=1e-7, ignore_constraints=ignore_constraints
    )


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_dynamic_weld(show_viewer, tol):
    CUBE_POS = (0.65, 0.0, 0.02)
    HANGING_BOX_POS = (0.0, 1.0, 0.4)

    scene = gs.Scene(
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
