import mujoco
import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.mujoco_parity import (
    check_mujoco_data_consistency,
    check_mujoco_model_consistency,
    init_paired_simulators,
    set_paired_inertial_properties,
    simulate_and_check_mujoco_consistency,
)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["scaled_mjcf_joint_equalities"])
@pytest.mark.parametrize("gs_solver, gs_integrator", [(gs.constraint_solver.Newton, gs.integrator.implicitfast)])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_joint(gs_sim, mj_sim, tol):
    (entity,) = gs_sim.entities
    qpos = entity.get_qpos()
    (i_unrelated_q,) = entity.get_joint("unrelated").qs_idx_local
    qpos[i_unrelated_q] = 1.0
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos=qpos, num_steps=50, tol=tol)


@pytest.mark.parametrize("model_name", ["mimic_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_joint_mimic(gs_sim, mj_sim, tol):
    assert gs_sim.rigid_solver.n_equalities == 1

    qpos = np.array((0.0, -1.0))
    qvel = np.array((1.0, -0.3))
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=300, tol=tol)

    (entity,) = gs_sim.entities
    qpos = entity.get_qpos()
    assert_allclose(qpos[0], qpos[1], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/four_bar_linkage_weld.xml", "weld.xml", "connect.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_link(gs_sim, mj_sim):
    # Must disable self-collision caused by closing the kinematic chain (adjacent link filtering is not enough)
    gs_sim.rigid_solver._enable_collision = False
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Set the time constant of the constraints on both engines to improve numerical stability
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


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["box_plan"])
@pytest.mark.parametrize(
    "gs_solver, gs_integrator",
    [
        (gs.constraint_solver.CG, gs.integrator.implicitfast),
        (gs.constraint_solver.CG, gs.integrator.Euler),
        (gs.constraint_solver.Newton, gs.integrator.implicitfast),
        (gs.constraint_solver.Newton, gs.integrator.Euler),
        # Elliptic (second-order) friction cone must match MuJoCo's elliptic cone. The box lands and slides with an
        # initial tangential + angular velocity so the tangential cone rows are exercised in both the sliding (cone
        # boundary) and sticking (bottom) regimes.
        pytest.param(
            gs.constraint_solver.CG,
            gs.integrator.implicitfast,
            marks=pytest.mark.friction_cone(gs.friction_cone.elliptic),
            id="CG-implicitfast-elliptic",
        ),
        pytest.param(
            gs.constraint_solver.Newton,
            gs.integrator.implicitfast,
            marks=pytest.mark.friction_cone(gs.friction_cone.elliptic),
            id="Newton-implicitfast-elliptic",
        ),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_box_plane_dynamics(gs_sim, mj_sim, tol):
    cube_pos = np.array([0.0, 0.0, 0.6])
    cube_quat = np.random.rand(4)
    cube_quat /= np.linalg.norm(cube_quat)
    qpos = np.concatenate((cube_pos, cube_quat))
    qvel = np.random.rand(6) * 0.2
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=150, tol=tol)


@pytest.mark.required
@pytest.mark.split_entities
@pytest.mark.parametrize("model_name", ["free_boxes_and_slider"])
@pytest.mark.parametrize("gs_solver, gs_integrator", [(gs.constraint_solver.Newton, gs.integrator.implicitfast)])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_scene_aggregates_hold_across_entities(gs_sim, mj_sim, tol):
    # Mujoco holds the three boxes in one model while Genesis holds one entity per box. The mean inertia the constraint
    # solver scales its tolerances by is a scene aggregate, so it must come out the same however the same bodies are
    # grouped into entities, which one entity per box is what tells apart. The sliding box carries an armature on a
    # body MuJoCo weighs by a rule of its own, so its constraint weights hold the general rule both engines settle on.
    # Each box resting on the plane is an island of its own on both sides, a thousandfold apart in mass, so each one
    # converges on its own inertia scale and the light box settles to the same rest as MuJoCo's.
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=10, tol=tol)

    # The runtime inertial setters derive the constraint weights and the mean inertia anew, which the model consistency
    # check then holds against the constants MuJoCo recomputes for the same change.
    set_paired_inertial_properties(
        gs_sim, mj_sim, armature_ratio=3.0, mass_ratio=0.5, inertia_ratio=2.0, com_offset=(0.01, -0.02, 0.03)
    )
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=10, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["hinge_slide"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_frictionloss(gs_sim, mj_sim, tol):
    qvel = np.array([0.7, -0.9])
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qvel=qvel, num_steps=2000, tol=tol)

    (entity,) = gs_sim.entities
    assert_allclose(entity.get_dofs_velocity(), 0.0, tol=1e-2)


@pytest.mark.required
@pytest.mark.friction_torsional(True)
@pytest.mark.parametrize(
    "model_name",
    [
        "sphere_plane_spin",
        pytest.param("sphere_plane_roll", marks=pytest.mark.friction_rolling(True)),
    ],
)
@pytest.mark.parametrize(
    "gs_solver, gs_integrator",
    [
        (gs.constraint_solver.Newton, gs.integrator.Euler),
        pytest.param(
            gs.constraint_solver.Newton,
            gs.integrator.Euler,
            marks=pytest.mark.friction_cone(gs.friction_cone.elliptic),
            id="Newton-Euler-elliptic",
        ),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_torsional_and_rolling_friction(gs_sim, mj_sim, tol):
    # Sliding while spinning and rolling couples every friction axis through slip, stick, and rest. The slight
    # initial penetration makes the contact exist from the first step.
    qpos = np.array([0.0, 0.0, 0.0999, 1.0, 0.0, 0.0, 0.0])
    qvel = np.array([0.5, 0.0, 0.0, 0.0, 4.0, 3.0])
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos=qpos, qvel=qvel, num_steps=60, tol=tol)


@pytest.mark.required
@pytest.mark.adjacent_collision(True)
@pytest.mark.parametrize("model_name", ["chain_capsule_hinge_mesh"])  # FIXME: , "chain_capsule_hinge_capsule"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_simple_kinematic_chain(gs_sim, mj_sim, tol):
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=200, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/walker.xml"])
@pytest.mark.parametrize(
    "gs_solver",
    [
        gs.constraint_solver.CG,
        # gs.constraint_solver.Newton,  # FIXME: This test is not passing because collision detection is too sensitive
    ],
)
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_walker(gs_sim, mj_sim, gjk_collision, tol):
    # Force numpy seed because this test is very sensitive to the initial condition
    np.random.seed(0)
    (gs_robot,) = gs_sim.entities
    qpos = np.zeros((gs_robot.n_qs,))
    qpos[2] += 0.5
    qvel = np.random.rand(gs_robot.n_dofs) * 0.2

    # Make sure it is possible to set the configuration vector without failure
    qpos = gs_robot.get_dofs_position()
    gs_robot.set_dofs_position(qpos)
    assert_allclose(gs_robot.get_dofs_position(), qpos, tol=gs.EPS)
    qpos = torch.rand(gs_robot.n_dofs).clip(*gs_robot.get_dofs_limit())
    gs_robot.set_dofs_position(qpos)
    assert_allclose(gs_robot.get_dofs_position(), qpos, tol=gs.EPS)

    # Cannot simulate any longer because collision detection is very sensitive
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=90, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/one_ball_joint.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_one_ball_joint(gs_sim, mj_sim, tol):
    # FIXME: Mujoco is detecting collision for some reason...
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=600, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/rope_ball.xml", "xml/rope_hinge.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_rope_ball(gs_sim, mj_sim, gs_solver, tol):
    # Make sure it is possible to set the configuration vector without failure
    qpos = gs_sim.rigid_solver.get_dofs_position()
    gs_sim.rigid_solver.set_dofs_position(qpos)
    assert_allclose(gs_sim.rigid_solver.get_dofs_position(), qpos, tol=gs.EPS)
    qpos = torch.rand(gs_sim.rigid_solver.n_dofs).clip(*gs_sim.rigid_solver.get_dofs_limit())
    gs_sim.rigid_solver.set_dofs_position(qpos)
    assert_allclose(gs_sim.rigid_solver.get_dofs_position(), qpos, tol=gs.EPS)

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=5e-9)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["linear_deformable.urdf"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_urdf_rope(gs_sim, mj_sim, gs_solver, xml_path):
    # Must increase sol params to improve numerical stability
    sol_params = gu.default_solver_params()
    sol_params[0] = 0.02
    gs_sim.rigid_solver.set_global_sol_params(sol_params)
    mj_sim.model.jnt_solref[:, 0] = sol_params[0]
    mj_sim.model.geom_solref[:, 0] = sol_params[0]
    mj_sim.model.eq_solref[:, 0] = sol_params[0]

    # The smooth acceleration divides chain-accumulated rounding by link masses of a tenth of a gram, putting its
    # agreement floor four decades above the working precision.
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=5e-5)


@pytest.mark.required
@pytest.mark.parametrize(
    "model_name, gjk_collision",
    [
        ("xml/tet_tet.xml", True),
        ("xml/tet_ball.xml", True),
        ("xml/tet_capsule.xml", True),
        # Multi-vertex contact patches between discrete meshes, recovered by clipping the touching faces; the rows
        # above settle on single-point vertex-face contacts.
        ("tet_meshball", True),
        # The same patches through the MPR pipeline, whose manifold comes from exhaustive mesh supports and perturbed
        # re-detections.
        ("tet_meshball", False),
    ],
)
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("multi_contact", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tet_primitive_shapes(gs_sim, mj_sim, gs_integrator, gs_solver, multi_contact, tol):
    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    # FIXME: Because of very small numerical error, error could be this large even if there is no logical error.
    # Multi-contact perturbation introduces slightly larger errors due to GJK implementation differences.
    # Both implementations agree to machine precision on most steps, but the capsule scene holds a grazing contact
    # whose occasional hard solves amplify rounding-order differences into distinct CG iterate paths.
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=700, tol=5e-6)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["humanoid_ball_floor"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_stickman(gs_sim, mj_sim, tol):
    # Make sure that the simulation is deterministic
    (gs_robot,) = gs_sim.entities
    gs_sim.scene.reset()
    gs_sim.scene.step()
    dofs_vel = gs_robot.get_dofs_velocity()
    for _ in range(50):
        gs_sim.scene.reset()
        gs_sim.scene.step()
        assert_equal(gs_robot.get_dofs_velocity(), dofs_vel)

    # A falling humanoid puts every capsule of the model on the ground in turn, so the contact set it exercises is far
    # richer than the other models here. Consistency is asserted step by step against MuJoCo rather than through the
    # pose it eventually settles in, which depends on a chaotic tumble and says nothing about compatibility.
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=500, tol=5e-9 if gs.np_float == np.float64 else tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["general_actuator"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_general_actuator(gs_sim, mj_sim, tol):
    (entity,) = gs_sim.entities

    # The force range of a dof composes every bound the file states for it in the order MuJoCo clamps: for the motor
    # its control range through the gear, then its own force range, which lies past the joint-level 'actuatorfrcrange'
    # and so collapses the force onto that joint bound. The PD gains come from the actuator.
    lower, upper = entity.get_dofs_force_range()
    assert_allclose(lower, [-20.0, -np.inf, 4.0], tol=tol)
    assert_allclose(upper, [20.0, np.inf, 4.0], tol=tol)
    assert_allclose(entity.get_dofs_kp(dofs_idx_local=[0]), 100.0, tol=tol)
    assert_allclose(entity.get_dofs_kv(dofs_idx_local=[0]), 2.0, tol=tol)

    # get_dofs_kp raises for all DOFs (joint 1 is non-PD-reducible from parser)
    with pytest.raises(gs.GenesisException):
        entity.get_dofs_kp()

    # but succeeds for the PD joint (joint 0)
    entity.get_dofs_kp(dofs_idx_local=[0])

    # Set different control modes per DOF via public API
    entity.control_dofs_force(0.0, dofs_idx_local=[0])
    entity.control_dofs_velocity(0.0, dofs_idx_local=[1])
    entity.control_dofs_position(0.0, dofs_idx_local=[2])
    ctrl_mode = gs_sim.rigid_solver.dyn_state.dofs.ctrl_mode.to_numpy()[:, 0]
    assert ctrl_mode[entity.dof_start + 0] == gs.CTRL_MODE.FORCE
    assert ctrl_mode[entity.dof_start + 1] == gs.CTRL_MODE.VELOCITY
    assert ctrl_mode[entity.dof_start + 2] == gs.CTRL_MODE.POSITION

    # control_dofs_position overrides all to POSITION
    entity.control_dofs_position([0.0, 0.0, 0.0])
    ctrl_mode = gs_sim.rigid_solver.dyn_state.dofs.ctrl_mode.to_numpy()[:, 0]
    assert (ctrl_mode[entity.dof_start : entity.dof_start + 3] == gs.CTRL_MODE.POSITION).all()

    # Disable constraints, keep actuation enabled
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    gs_sim.rigid_solver._enable_collision = False
    gs_sim.rigid_solver._enable_joint_limit = False
    gs_sim.rigid_solver._disable_constraint = True
    gs_sim.rigid_solver.collider.clear()
    gs_sim.rigid_solver.constraint_solver.clear()

    # Compare all dynamic quantities against MuJoCo with both PD and general actuators active.
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    init_paired_simulators(gs_sim, mj_sim, qpos=[0.2, 0.1, 0.0], qvel=[0.1, -0.1, 0.0])

    # Both the PD joint (kp(100) * 0.3 rad = 30 N.m against its 20 N.m joint bound) and the motor (gear(5) * ctrl(1) =
    # 5 N.m raised to its 6 N.m force floor, past its 4 N.m joint bound) saturate, so the comparison exercises the clamp.
    mj_sim.data.ctrl[:] = [0.5, 0.3, 1.0]
    entity.control_dofs_position([0.5, 0.3, 0.0])
    entity.control_dofs_force(5.0, dofs_idx_local=[2])

    # Pre-step so that Genesis computes qf_applied (needed for data consistency checks)
    mj_sim.data.qpos[:] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_sim.data.qvel[:] = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]
    mujoco.mj_step(mj_sim.model, mj_sim.data)
    gs_sim.scene.step()

    for _ in range(99):
        check_mujoco_data_consistency(gs_sim, mj_sim, tol=tol, ignore_constraints=True)

        mj_sim.data.qpos[:] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[:] = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]
        mujoco.mj_step(mj_sim.model, mj_sim.data)
        gs_sim.scene.step()

    # Validate setter/getter round-trips for actuator parameters
    entity.set_dofs_act_gain([200.0], dofs_idx_local=[1])
    assert_allclose(entity.get_dofs_act_gain()[1], 200.0, tol=1e-6)
    entity.set_dofs_act_bias([0.5], [-100.0], [-5.0], dofs_idx_local=[1])
    b0, b1, b2 = entity.get_dofs_act_bias()
    assert_allclose(b0[1], 0.5, tol=1e-6)
    assert_allclose(b1[1], -100.0, tol=1e-6)
    assert_allclose(b2[1], -5.0, tol=1e-6)

    # set_dofs_kp restores PD on joint 1: act_gain=kp, act_bias[0]=0, act_bias[1]=-kp
    entity.set_dofs_kp([50.0], dofs_idx_local=[1])
    assert_allclose(entity.get_dofs_kp(dofs_idx_local=[0, 1]), [100.0, 50.0], tol=1e-6)
    b0, b1, _ = entity.get_dofs_act_bias()
    assert_allclose(b0[1], 0.0, tol=1e-6)
    assert_allclose(b1[1], -50.0, tol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/franka_emika_panda/panda.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_robot_kinematics(gs_sim, mj_sim, tol):
    # Disable all constraints and actuation
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    gs_sim.rigid_solver.dyn_state.dofs.ctrl_mode.fill(int(gs.CTRL_MODE.FORCE))
    gs_sim.rigid_solver._enable_collision = False
    gs_sim.rigid_solver._enable_joint_limit = False
    gs_sim.rigid_solver._disable_constraint = True
    gs_sim.rigid_solver.collider.clear()
    gs_sim.rigid_solver.constraint_solver.clear()

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)

    (gs_robot,) = gs_sim.entities
    dof_bounds = gs_sim.rigid_solver.dyn_info.dofs.limit.to_numpy()
    for _ in range(100):
        qpos = dof_bounds[:, 0] + (dof_bounds[:, 1] - dof_bounds[:, 0]) * np.random.rand(gs_robot.n_qs)
        init_paired_simulators(gs_sim, mj_sim, qpos)
        check_mujoco_data_consistency(gs_sim, mj_sim, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["compound_joint"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
def test_jacobian_compound_joints(gs_sim, mj_sim, tol):
    (gs_robot,) = gs_sim.entities
    end_link = gs_robot.get_link("seg2")
    end_body_id = mujoco.mj_name2id(mj_sim.model, mujoco.mjtObj.mjOBJ_BODY, "seg2")
    jacp = np.empty((3, mj_sim.model.nv), dtype=np.float64)
    jacr = np.empty((3, mj_sim.model.nv), dtype=np.float64)

    for qpos in ((0.0, 0.0, 0.0), (0.3, -0.5, 0.7)):
        init_paired_simulators(gs_sim, mj_sim, qpos)
        mujoco.mj_jacBody(mj_sim.model, mj_sim.data, jacp, jacr, end_body_id)

        assert_allclose(gs_robot.get_jacobian(end_link), np.concatenate([jacp, jacr]), tol=tol)
