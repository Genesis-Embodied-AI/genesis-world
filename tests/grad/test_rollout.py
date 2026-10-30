import gc
import weakref

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from .utils import make_diff_scene_pair


@pytest.mark.parametrize(
    "model_name", ["grad_free", "grad_revolute", pytest.param("grad_free_with_revolute", marks=pytest.mark.required)]
)
def test_horizon_truncation_matches_independent_scenes(model_name, request, tol, show_viewer):
    mjcf = request.getfixturevalue(model_name)
    horizon = 5
    B = 2

    def build(show_viewer=False):
        pair = make_diff_scene_pair(mjcf, n_envs=B, substeps=4, gravity=(0.0, 0.0, 0.0), show_viewer=show_viewer)
        return pair.scene_ana, pair.entity_ana

    def run_segment(scene, entity, velocity):
        entity.set_dofs_velocity(velocity)
        for _ in range(horizon):
            scene.step()
        return (entity.get_qpos() ** 2).sum()

    # Scene A: one scene, snapshot + reset between two horizons.
    scene_a, robot_a = build(show_viewer=show_viewer)
    v1 = np.random.standard_normal((B, robot_a.n_dofs))
    v2 = np.random.standard_normal((B, robot_a.n_dofs))
    scene_a.reset()
    v1_a = torch.tensor(v1, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    loss_h1_a = run_segment(scene_a, robot_a, v1_a)
    qpos_mid_a = tensor_to_array(robot_a.get_qpos())
    time_mid_a = scene_a.get_time().clone()
    scene_a.backward(loss_h1_a)
    # backward restores the physics state it snapshotted, so the clocks describing it come back too
    assert_allclose(scene_a.get_time(), time_mid_a, tol=gs.EPS)
    grad1_a = tensor_to_array(v1_a.grad).copy()

    v2_a = torch.tensor(v2, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    loss_h2_a = run_segment(scene_a, robot_a, v2_a)
    qpos_end_a = tensor_to_array(robot_a.get_qpos())
    scene_a.backward(loss_h2_a)
    grad2_a = tensor_to_array(v2_a.grad).copy()

    # Scene B: horizon 1 only, whose backward pass returns the terminal snapshot Scene C resumes from.
    scene_b, robot_b = build()
    scene_b.reset()
    v1_b = torch.tensor(v1, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    loss_h1_b = run_segment(scene_b, robot_b, v1_b)
    qpos_mid_b = tensor_to_array(robot_b.get_qpos())
    snapshot_b = scene_b.backward(loss_h1_b)
    grad1_b = tensor_to_array(v1_b.grad).copy()

    assert_equal(qpos_mid_a, qpos_mid_b)
    assert_equal(loss_h1_a, loss_h1_b)
    assert_allclose(grad1_a, grad1_b, tol=tol)

    # Scene C: fresh scene resumed from B's mid-trajectory snapshot.
    scene_c, robot_c = build()
    scene_c.reset(snapshot_b)
    v2_c = torch.tensor(v2, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    loss_h2_c = run_segment(scene_c, robot_c, v2_c)
    qpos_end_c = tensor_to_array(robot_c.get_qpos())
    scene_c.backward(loss_h2_c)
    grad2_c = tensor_to_array(v2_c.grad).copy()

    assert_equal(qpos_end_a, qpos_end_c)
    assert_equal(loss_h2_a, loss_h2_c)
    assert_allclose(grad2_a, grad2_c, tol=tol)


@pytest.mark.required
def test_sim_vs_solver_state_grad_parity(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        ),
    )
    scene.build()

    ctrl = torch.tensor(np.random.randn(robot.n_dofs), dtype=gs.tc_float, device=gs.device, requires_grad=True)

    grads = []
    for is_sim_state_source in (False, True):
        scene.reset()
        robot.set_dofs_velocity(ctrl)
        scene.step()
        if is_sim_state_source:
            sim_state = scene.get_state()
            chassis_pos = sim_state.solvers_state[scene.solvers.index(scene.rigid_solver)].links_pos[:, 0].squeeze()
        else:
            chassis_pos = robot.get_state().pos.squeeze()
        loss = torch.linalg.norm(chassis_pos)
        loss.backward()
        # The backward pass of a step replays it from the state it started from, so the clock follows the physics the
        # backward pass rewinds instead of counting that replay as newly simulated time.
        assert_allclose(scene.get_time(), 0.0, tol=gs.EPS)
        grads.append(ctrl.grad.detach().clone())
        ctrl.grad.zero_()
        assert (grads[-1][..., :3].abs() > gs.EPS).all()
        assert_allclose(grads[-1][..., 3:], 0.0, atol=gs.EPS)

    assert_allclose(*grads, atol=gs.EPS)

    # A state queried after a setter reflects it, even within the step of an earlier query
    scene.reset()
    robot.get_state()
    robot.set_pos((0.1, -0.2, 0.3))
    assert_allclose(robot.get_state().pos[0], (0.1, -0.2, 0.3), tol=gs.EPS)


@pytest.mark.required
def test_closed_loop_rollout_follows_torch_semantics(grad_revolute, precision, show_viewer):
    # The control forces of a torch policy read the state queried before each step, are given to both arms and are
    # penalized in the loss, so their graph is shared by the loss and the rollout.
    HORIZON, DT = 50, 0.01

    scenes, arms = [], []
    for requires_grad in (True, False):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=DT,
                gravity=(0.0, 0.0, 0.0),
                requires_grad=requires_grad,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,
                disable_constraint=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.2, -1.2, 0.8),
                camera_lookat=(0.0, 0.0, 0.2),
            ),
            show_viewer=show_viewer and requires_grad,
        )
        for y in (-0.3, 0.3):
            arms.append(
                scene.add_entity(
                    gs.morphs.MJCF(
                        file=grad_revolute,
                        pos=(0.0, y, 0.0),
                    ),
                )
            )
        scene.build(n_envs=2)
        scenes.append(scene)
    scene_ana, scene_fd = scenes
    arms_ana, arms_fd = arms[:2], arms[2:]

    target = torch.tensor([[0.4], [-0.3]], dtype=gs.tc_float, device=gs.device)
    params = np.array([0.6, 3.0, 8.0, 0.5])

    def rollout(scene, arms, params):
        amplitude, frequency, kp, kd = torch.as_tensor(params, dtype=gs.tc_float, device=gs.device)
        scene.reset()
        effort = 0.0
        for i_step in range(HORIZON):
            # A query without gradient recording, as for logging, leaves the next query of the step differentiable
            with torch.no_grad():
                arms[0].get_qpos()
            force = amplitude * torch.sin(frequency * i_step * DT) + kp * (target - arms[0].get_qpos())
            force = force - kd * arms[0].get_dofs_velocity()
            for arm in arms:
                arm.control_dofs_force(force)
            effort = effort + (force**2).sum()
            scene.step()
        return sum([((arm.get_qpos() - target) ** 2).sum() for arm in arms]) + 1e-4 * effort

    params_ana = torch.tensor(params, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    rollout(scene_ana, arms_ana, params_ana).backward()
    # The graph of a rollout runs backward again while it is retained, and the state behaves as a tensor the simulator
    # mutates in place: a mutation torch does not record makes a later backward through the rollout raise.
    loss = rollout(scene_ana, arms_ana, params_ana)
    (grad_functional,) = torch.autograd.grad(loss, params_ana, retain_graph=True)
    assert_allclose(grad_functional, params_ana.grad, tol=gs.EPS)
    (grad_retained,) = torch.autograd.grad(loss, params_ana, retain_graph=True)
    assert_allclose(grad_retained, grad_functional, tol=gs.EPS)
    with torch.no_grad():
        arms_ana[0].control_dofs_force(0.0)
    with pytest.raises(gs.GenesisException, match="changed since this operation"):
        torch.autograd.grad(loss, params_ana)
    # So does a change of the model, which the backward pass of every step would replay with
    loss = rollout(scene_ana, arms_ana, params_ana)
    arms_ana[0].set_dofs_kv(arms_ana[0].get_dofs_kv())
    with pytest.raises(gs.GenesisException, match="changed since this operation"):
        torch.autograd.grad(loss, params_ana)
    # A gradient that is not finite raises during the backward pass, naming the step it reaches first, and so does an
    # error the last steps of the rollout flagged
    loss = rollout(scene_ana, arms_ana, params_ana)
    with pytest.raises(gs.GenesisException, match=f"Backward pass of step {HORIZON - 1}: .*'inf' gradient"):
        torch.autograd.grad(loss * float("nan"), params_ana)
    # So does one that a later write of the state cuts from the start of the rollout, where the gradient stays finite
    loss = rollout(scene_ana, arms_ana, params_ana)
    for arm in arms_ana:
        arm.set_dofs_position(0.0)
        arm.set_dofs_velocity(0.0)
        arm.control_dofs_force(0.0)
    scene_ana.step()
    loss = loss + float("nan") * arms_ana[0].get_dofs_position().sum()
    with pytest.raises(gs.GenesisException, match=f"Backward pass of step {HORIZON}: .*'inf' gradient"):
        torch.autograd.grad(loss, params_ana)
    loss = rollout(scene_ana, arms_ana, params_ana)
    arms_ana[0].set_dofs_velocity(float("nan"))
    scene_ana.step()
    with pytest.raises(gs.GenesisException, match="^Invalid accelerations"):
        torch.autograd.grad(loss, params_ana)
    # A 'nan' control force, unbounded by the force range of the dofs, drives the state to infinity
    loss = rollout(scene_ana, arms_ana, params_ana)
    arms_ana[0].control_dofs_force(float("nan"))
    scene_ana.step()
    with pytest.raises(gs.GenesisException, match="^Invalid accelerations"):
        torch.autograd.grad(loss, params_ana)

    # A write given an index several times leaves the value of its last occurrence, which alone gets a gradient, whether
    # the write is recorded or not
    for force_repeated, dofs_idx_local, envs_idx, force_last in (
        ([[0.3], [-0.5]], None, [1, 1], [[0.0], [-0.5]]),
        ([[0.3, -0.5], [0.2, 0.4]], [0, 0], None, [[-0.5], [0.4]]),
    ):
        force_ana = torch.tensor(force_repeated, dtype=gs.tc_float, device=gs.device, requires_grad=True)
        velocities = []
        for scene, arm, force, force_dofs_idx_local, force_envs_idx in (
            (scene_fd, arms_fd[0], force_last, None, None),
            (scene_fd, arms_fd[0], force_repeated, dofs_idx_local, envs_idx),
            (scene_ana, arms_ana[0], force_ana, dofs_idx_local, envs_idx),
        ):
            scene.reset()
            with torch.no_grad():
                arm.control_dofs_force(0.0)
            arm.control_dofs_force(force, dofs_idx_local=force_dofs_idx_local, envs_idx=force_envs_idx)
            scene.step()
            velocities.append(arm.get_dofs_velocity())
        assert_allclose(velocities[1], velocities[0], tol=gs.EPS)
        assert_allclose(velocities[2], velocities[0], tol=gs.EPS)
        (force_grad,) = torch.autograd.grad(velocities[2].sum(), force_ana)
        assert_allclose(force_grad[0] if envs_idx else force_grad[:, 0], 0.0, tol=gs.EPS)
        assert ((force_grad[1] if envs_idx else force_grad[:, 1]).abs() > gs.EPS).all()

    # fp32 needs a large step to clear the state-noise floor. fp64 needs a small step to bound truncation error.
    eps = 3e-5 if precision == "64" else 3e-2
    fd_grad = np.zeros_like(params)
    for i in range(len(params)):
        losses = []
        for sign in (+1, -1):
            perturbed = params.copy()
            perturbed[i] += sign * eps
            losses.append(float(rollout(scene_fd, arms_fd, perturbed)))
        fd_grad[i] = (losses[0] - losses[1]) / (2.0 * eps)
    assert (np.abs(fd_grad) > 1e-3).all()
    tol = 2e-10 if precision == "64" else 1e-4
    assert_allclose(params_ana.grad, fd_grad, rtol=tol, atol=tol)

    # A differentiable scene is freed once the user drops it, whatever rollout it recorded
    scene_ref = weakref.ref(scene_ana)
    del scenes, arms, arm, scene, scene_ana, arms_ana, velocities
    gc.collect()
    assert scene_ref() is None


@pytest.mark.required
def test_state_getters_grad_matches_fd(grad_free_with_hinge_ball_slide, precision, show_viewer):
    # A proportional-derivative (PD) controller written with the getters feeds the configuration and velocity it reads
    # back into the control forces, and the loss reads link poses and dof positions: the gradient of its gains follows
    # both. The dof positions are derived from the configuration as forward kinematics does, for every joint type.
    HORIZON = 30
    pair = make_diff_scene_pair(
        grad_free_with_hinge_ball_slide, n_envs=2, gravity=(0.0, 0.0, -1.0), show_viewer=show_viewer
    )
    velocity = np.linspace(-0.8, 0.9, 11) * np.array([[1.0], [-0.7]])
    params = np.array([3.0, 0.4, 0.5, 0.3])
    joints_dofs, base_rotation_dofs = slice(6, 11), slice(3, 6)

    # In a differentiable scene, the getters track gradients and give the values the solver holds
    reads = []
    for scene, entity in ((pair.scene_ana, pair.entity_ana), (pair.scene_fd, pair.entity_fd)):
        scene.reset()
        entity.set_dofs_velocity(velocity)
        for _ in range(7):
            scene.step()
        reads.append(
            {
                "qpos": entity.get_qpos(),
                "dofs_position": entity.get_dofs_position(),
                "dofs_position_selected": entity.get_dofs_position(dofs_idx_local=[8, 3, 8], envs_idx=[1]),
                "dofs_velocity": entity.get_dofs_velocity(),
                "links_pos": entity.get_links_pos(),
                "links_pos_solver": entity.get_links_pos(relative=False),
                "links_pos_com": entity.get_links_pos(ref=gs.link_ref_frame.link_COM),
                "links_quat": entity.get_links_quat(),
                "pos": entity.get_pos(),
                "quat": entity.get_quat(),
            }
        )
    for name, value in reads[0].items():
        assert value.requires_grad, name
        assert not reads[1][name].requires_grad, name
        assert_allclose(value, reads[1][name], tol=gs.EPS, err_msg=name)

    def rollout(scene, entity, params):
        kp, kd, amplitude, rotation = torch.as_tensor(params, dtype=gs.tc_float, device=gs.device)
        scene.reset()
        entity.set_dofs_velocity(velocity)
        for i_step in range(HORIZON):
            # One rotation dof of the free joint and of the ball joint, mid-rollout, whose other rotation dofs keep the
            # positions the controller drove them to
            if i_step == HORIZON // 2:
                entity.set_dofs_position(
                    rotation * torch.tensor([1.0, -0.5], dtype=gs.tc_float, device=gs.device), dofs_idx_local=[4, 8]
                )
            position = entity.get_dofs_position(joints_dofs)
            force = kp * (amplitude * np.sin(0.2 * i_step) - position) - kd * entity.get_dofs_velocity(joints_dofs)
            entity.control_dofs_force(force, joints_dofs)
            scene.step()
        return (entity.get_links_pos()[:, 1:] ** 2).sum() + (entity.get_dofs_position(base_rotation_dofs) ** 2).sum()

    params_ana = torch.tensor(params, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    rollout(pair.scene_ana, pair.entity_ana, params_ana).backward()

    eps = 1e-5 if precision == "64" else 1e-2
    fd_grad = np.zeros_like(params)
    for i in range(len(params)):
        losses = []
        for sign in (+1, -1):
            perturbed = params.copy()
            perturbed[i] += sign * eps
            losses.append(float(rollout(pair.scene_fd, pair.entity_fd, perturbed)))
        fd_grad[i] = (losses[0] - losses[1]) / (2.0 * eps)
    assert (np.abs(fd_grad) > 1e-4).all()
    assert_allclose(params_ana.grad, fd_grad, tol=2e-10 if precision == "64" else 1e-4)


@pytest.mark.required
# The SAP coupler, one of the couplings refused, supports 64 bits precision alone
@pytest.mark.precision("64")
def test_reject_non_differentiable_solvers():
    box = gs.morphs.Box(pos=(0.0, 0.0, 0.5), size=(0.1, 0.1, 0.1))
    for materials, options, component_name in (
        ((gs.materials.PBD.Liquid(),), {}, "PBDSolver"),
        ((gs.materials.SPH.Liquid(),), {}, "SPHSolver"),
        ((gs.materials.FEM.Elastic(),), {"fem_options": gs.options.FEMOptions(use_implicit_solver=True)}, "FEMSolver"),
        ((gs.materials.FEM.Elastic(), gs.materials.Rigid()), {}, "LegacyCoupler"),
        ((gs.materials.Rigid(),), {"coupler_options": gs.options.SAPCouplerOptions()}, "SAPCoupler"),
    ):
        scene = gs.Scene(sim_options=gs.options.SimOptions(requires_grad=True), **options, show_viewer=False)
        for material in materials:
            scene.add_entity(morph=box, material=material)
        with pytest.raises(gs.GenesisException, match=component_name):
            scene.build()
