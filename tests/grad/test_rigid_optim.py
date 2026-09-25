import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose
from .utils import make_diff_scene_pair


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("control_target", ["init_vel", "control_force"])
@pytest.mark.debug(False)
def test_reference_trajectory_recovery_converges(control_target, grad_cartpole, show_viewer):
    # Reproduce a reference cartpole trajectory by optimizing either the initial velocity or the per-step control
    # forces; every env must drive its per-env loss below both a relative-reduction and an absolute threshold. The
    # reference is exactly reproducible, so a correct gradient lets Adam crush the loss to its optimizer plateau
    # (fp32 and fp64 reach the same floor). init_vel (4 params) converges far below control_force (128 per-step
    # forces), hence the per-target thresholds - each pinned just above the measured converged loss.
    N_STEPS, N_ITER, LR, B = 32, 150, 1e-2, 2
    REL_REDUCTION, ABS_THRESHOLD = {
        "init_vel": (2e-6, 2e-7),
        "control_force": (1e-3, 5e-7),
    }[control_target]
    pair = make_diff_scene_pair(grad_cartpole, n_envs=B, show_viewer=show_viewer)
    N_DOFS = pair.entity_ana.n_dofs
    scene_ref, robot_ref = pair.scene_fd, pair.entity_fd
    scene_opt, robot_opt = pair.scene_ana, pair.entity_ana
    rng = np.random.default_rng(seed=11 if control_target == "init_vel" else 23)

    scene_ref.reset()
    if control_target == "init_vel":
        target_ctrl = rng.normal(size=(B, N_DOFS)) * 0.5
        robot_ref.set_dofs_velocity(target_ctrl)
        for _ in range(N_STEPS):
            scene_ref.step()
    else:
        target_ctrl = rng.normal(size=(N_STEPS, B, N_DOFS)) * 0.2
        for t in range(N_STEPS):
            robot_ref.control_dofs_force(target_ctrl[t])
            scene_ref.step()
    ref_state = scene_ref.rigid_solver.get_state()
    target_qpos = ref_state.qpos.detach().clone()
    target_vel = ref_state.dofs_vel.detach().clone()

    if control_target == "init_vel":
        init_vel = gs.tensor(target_ctrl + rng.normal(size=(B, N_DOFS)) * 0.3, dtype=gs.tc_float, requires_grad=True)
        params = [init_vel]
    else:
        forces = [
            gs.tensor(target_ctrl[t] + rng.normal(size=(B, N_DOFS)) * 0.1, dtype=gs.tc_float, requires_grad=True)
            for t in range(N_STEPS)
        ]
        params = forces
    optimizer = torch.optim.Adam(params, lr=LR)

    loss_history = []
    for _ in range(N_ITER):
        optimizer.zero_grad(set_to_none=False)
        scene_opt.reset()
        if control_target == "init_vel":
            robot_opt.set_dofs_velocity(init_vel)
            for _ in range(N_STEPS):
                scene_opt.step()
        else:
            for t in range(N_STEPS):
                robot_opt.control_dofs_force(forces[t])
                scene_opt.step()
        state = scene_opt.rigid_solver.get_state()
        diff_pos = (state.qpos - target_qpos).reshape(B, -1)
        diff_vel = (state.dofs_vel - target_vel).reshape(B, -1)
        loss_per_env = (diff_pos**2).sum(dim=-1) + (diff_vel**2).sum(dim=-1)
        loss_history.append(tensor_to_array(loss_per_env).copy())
        loss_per_env.sum().backward()
        optimizer.step()

    history = np.array(loss_history)
    initial, final = history[0], history[-1]
    rel_ratios = final / initial
    assert_allclose(rel_ratios, 0.0, atol=REL_REDUCTION, err_msg=f"loss reduction insufficient (initial={initial})")
    assert_allclose(final, 0.0, atol=ABS_THRESHOLD, err_msg="final loss above absolute threshold")


@pytest.mark.slow
@pytest.mark.required
@pytest.mark.debug(False)
def test_goal_pose_optimization_converges(show_viewer):
    goal_pos = gs.tensor([0.7, 1.0, 0.05])
    goal_quat = gs.tensor([0.3, 0.2, 0.1, 0.9])
    goal_quat = goal_quat / torch.norm(goal_quat, dim=-1, keepdim=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0, 0, -1),
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            disable_constraint=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -0.15, 2.42),
            camera_lookat=(0.5, 0.5, 0.1),
        ),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0),
            size=(0.1, 0.1, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.0, 0.0, 1.0),
        ),
    )
    scene.build()

    init_pos = gs.tensor([0.3, 0.1, 0.28], requires_grad=True)
    init_quat = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    optimizer = torch.optim.Adam([init_pos, init_quat], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-3)

    for _ in range(200):
        scene.reset()
        box.set_pos(init_pos)
        box.set_quat(init_quat)
        for _ in range(100):
            scene.step()
        box_state = box.get_state()
        loss = torch.abs(box_state.pos - goal_pos).sum() + torch.abs(box_state.quat - goal_quat).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            init_quat.data = init_quat / torch.norm(init_quat, dim=-1, keepdim=True)

    assert_allclose(loss, 0.0, atol=1e-2)
