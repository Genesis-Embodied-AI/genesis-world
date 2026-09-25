import math

import numpy as np
import pytest

import genesis as gs

from .utils import assert_grad_matches_fd, make_diff_scene_pair


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "model_name",
    [
        "grad_free",
        "grad_revolute",
        "grad_prismatic",
        "grad_spherical",
        "grad_free_with_revolute",
        "grad_chain3",
        "grad_cartpole",
        "grad_hopper",
    ],
)
def test_fk_grad_matches_fd(model_name, request, precision, show_viewer):
    is_tall = model_name in ("grad_cartpole", "grad_hopper")
    B = 2
    pair = make_diff_scene_pair(
        request.getfixturevalue(model_name),
        n_envs=B,
        substeps=4,
        show_viewer=show_viewer,
        camera_pos=(2.5, -2.5, 1.8) if is_tall else (1.2, -1.2, 0.8),
        camera_lookat=(0.0, 0.0, 0.9) if is_tall else (0.0, 0.0, 0.2),
    )
    n_dofs = pair.entity_ana.n_dofs
    n_links = pair.entity_ana.n_links

    # Single-link joints read the entity pose; multi-link topologies read the rigid-solver per-link pose.
    is_single_link = model_name in ("grad_free", "grad_revolute", "grad_prismatic", "grad_spherical")
    # (setter, output, input_seed) per joint, plus the position and quaternion target seeds. Single-step
    # force->position is only checked on cartpole, where its sensitivity clears the finite-difference floor.
    checks_by_joint = {
        "grad_free": ((("pos", "pos", 10), ("quat", "quat", 11), ("vel", "pos", 12), ("vel", "quat", 13)), 1, 2),
        "grad_revolute": ((("vel", "pos", 30), ("vel", "quat", 31), ("force", "quat", 32)), 21, 22),
        "grad_prismatic": ((("vel", "pos", 50),), 41, 0),
        "grad_spherical": ((("vel", "pos", 70), ("vel", "quat", 71), ("force", "quat", 72)), 61, 62),
        "grad_free_with_revolute": (
            (("pos", "pos", 70), ("quat", "quat", 71), ("vel", "pos", 72), ("vel", "quat", 73)),
            61,
            62,
        ),
        "grad_chain3": ((("vel", "pos", 90), ("vel", "quat", 91)), 81, 82),
        "grad_cartpole": ((("vel", "pos", 190), ("vel", "quat", 191), ("force", "pos", 192)), 181, 182),
        "grad_hopper": ((("vel", "pos", 210), ("vel", "quat", 211)), 201, 202),
    }
    checks, pos_seed, quat_seed = checks_by_joint[model_name]

    # Per-topology finite-difference floors (fp64 tolerance, fp32 tolerance, fp32 step): free-joint topologies
    # are the noisiest at both precisions and need a smaller fp32 step.
    fp64_tol, fp32_tol, fp32_eps = {
        "grad_free": (1e-9, 2e-4, 1e-2),
        "grad_revolute": (1e-10, 5e-5, 3e-2),
        "grad_prismatic": (1e-10, 5e-6, 3e-2),
        "grad_spherical": (1e-10, 1e-4, 3e-2),
        "grad_free_with_revolute": (1e-9, 2e-4, 1e-2),
        "grad_chain3": (2e-10, 1e-4, 3e-2),
        "grad_cartpole": (1e-10, 5e-5, 3e-2),
        "grad_hopper": (5e-10, 2e-4, 3e-2),
    }[model_name]

    pos_shape = (B, 3) if is_single_link else (B, n_links, 3)
    quat_shape = (B, 4) if is_single_link else (B, n_links, 4)
    tgt_pos = gs.tensor(np.random.RandomState(pos_seed).standard_normal(pos_shape), dtype=gs.tc_float).reshape(-1)
    tgt_quat = gs.tensor(np.random.RandomState(quat_seed).standard_normal(quat_shape), dtype=gs.tc_float).reshape(-1)
    for setter, output, input_seed in checks:
        rng = np.random.default_rng(input_seed)
        if setter == "pos":
            step_input = rng.standard_normal((B, 3))
        elif setter == "quat":
            step_input = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), (B, 4)).copy()
            step_input = step_input + 0.05 * rng.standard_normal((B, 4))
            step_input = step_input / np.linalg.norm(step_input, axis=-1, keepdims=True)
        else:
            step_input = rng.standard_normal((B, n_dofs))

        def apply_vel_per_env(e, x):
            # Same-step per-environment commands: each call must keep its own tape slot and gradient path.
            e.set_dofs_velocity(x[:1], envs_idx=[0])
            e.set_dofs_velocity(x[1:], envs_idx=[1])

        apply_fn = {
            "pos": lambda e, x: e.set_pos(x),
            "quat": lambda e, x: e.set_quat(x),
            "vel": apply_vel_per_env,
            "force": lambda e, x: e.control_dofs_force(x),
        }[setter]

        target = tgt_pos if output == "pos" else tgt_quat

        def loss_fn(scene, entity, tgt=target, out=output, sl=is_single_link):
            if sl:
                pose = entity.get_state().pos if out == "pos" else entity.get_state().quat
            else:
                state = scene.rigid_solver.get_state()
                pose = state.links_pos if out == "pos" else state.links_quat
            return ((pose.reshape(-1) - tgt) ** 2).sum()

        assert_grad_matches_fd(
            pair,
            [step_input],
            apply_fn,
            loss_fn,
            rtol=fp64_tol if precision == "64" else fp32_tol,
            atol=fp64_tol if precision == "64" else fp32_tol,
            eps=3e-5 if precision == "64" else fp32_eps,
        )


@pytest.mark.required
@pytest.mark.parametrize(
    "model_name",
    [
        "grad_free",
        "grad_revolute",
        "grad_prismatic",
        "grad_free_with_revolute",
        "grad_chain3",
        "grad_spherical",
        "grad_cartpole",
        "grad_hopper",
    ],
)
@pytest.mark.debug(False)
def test_fk_multistep_force_grad_matches_fd(model_name, request, precision, show_viewer):
    # Ten distinct per-step control forces, each of which must receive an independent adjoint across the unroll.
    # (output kind: entity state pos / quat or rigid-solver links, per-link output shape, target seed, fp32
    # tolerance). Anchored single-joint topologies (revolute, spherical) read the quaternion: their base link
    # position is pinned at the joint anchor, so a position loss would be constant and the check vacuous.
    output, output_shape, seed, fp32_tol = {
        "grad_free": ("state_pos", (3,), 161, 5e-6),
        "grad_revolute": ("state_quat", (4,), 162, 5e-6),
        "grad_prismatic": ("state_pos", (3,), 163, 2e-6),
        "grad_free_with_revolute": ("links", (2, 3), 164, 5e-5),
        "grad_chain3": ("links", (3, 3), 165, 2e-5),
        "grad_spherical": ("state_quat", (4,), 166, 1e-4),
        "grad_cartpole": ("links", (2, 3), 167, 2e-5),
        "grad_hopper": ("links", (5, 3), 168, 1e-4),
    }[model_name]
    is_tall = model_name in ("grad_cartpole", "grad_hopper")
    pair = make_diff_scene_pair(
        request.getfixturevalue(model_name),
        n_envs=0,
        substeps=4,
        show_viewer=show_viewer,
        camera_pos=(2.5, -2.5, 1.8) if is_tall else (1.2, -1.2, 0.8),
        camera_lookat=(0.0, 0.0, 0.9) if is_tall else (0.0, 0.0, 0.2),
    )
    n_dofs = pair.entity_ana.n_dofs
    target = gs.tensor(np.random.RandomState(seed).standard_normal((1, *output_shape)), dtype=gs.tc_float).reshape(-1)
    inputs = [np.random.default_rng(seed * 100 + t).standard_normal((n_dofs,)) for t in range(10)]

    def loss_fn(scene, entity):
        if output == "state_pos":
            pose = entity.get_state().pos
        elif output == "state_quat":
            pose = entity.get_state().quat
        else:
            pose = scene.rigid_solver.get_state().links_pos
        return ((pose.reshape(-1) - target) ** 2).sum()

    def apply_force(entity, force):
        # Split the same-step control across two dof subsets (the standard arm + gripper pattern): each call must
        # keep its own tape slot and gradient path. The second subset is passed as a slice, a valid index form the
        # tape key must accept on every backend.
        if n_dofs == 1:
            entity.control_dofs_force(force)
        else:
            entity.control_dofs_force(force[..., :1], dofs_idx_local=[0])
            entity.control_dofs_force(force[..., 1:], dofs_idx_local=slice(1, n_dofs))

    # fp32 needs a large step to clear the state-noise floor; fp64 needs a small step to bound truncation error.
    fp64_tol = 5e-10 if model_name == "grad_hopper" else 1e-10
    assert_grad_matches_fd(
        pair,
        inputs,
        apply_force,
        loss_fn,
        rtol=fp64_tol if precision == "64" else fp32_tol,
        atol=fp64_tol if precision == "64" else fp32_tol,
        eps=3e-5 if precision == "64" else 3e-2,
    )


@pytest.mark.required
@pytest.mark.parametrize("control_mode", ["position", "velocity"])
def test_per_step_pd_target_grad_matches_fd(control_mode, grad_revolute, precision, show_viewer):
    # Per-step-varying PD targets are scenario commands, replayed by the backward unroll; the gradient of a tracked
    # initial velocity through the controlled rollout must match finite differences.
    pair = make_diff_scene_pair(
        grad_revolute,
        substeps=4,
        show_viewer=show_viewer,
    )
    for entity in (pair.entity_ana, pair.entity_fd):
        entity.set_dofs_kp(4.0)
        entity.set_dofs_kv(0.8)
    targets = [0.3 * math.sin(0.7 * t) for t in range(10)]

    def step_fn(entity, i_step):
        if control_mode == "position":
            entity.control_dofs_position(targets[i_step])
        else:
            entity.control_dofs_velocity(targets[i_step])

    assert_grad_matches_fd(
        pair,
        [np.array([1.5])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: scene.rigid_solver.get_state().qpos[0, 0] ** 2,
        n_steps=10,
        step_fn=step_fn,
        rtol=1e-10 if precision == "64" else 5e-5,
        atol=1e-10 if precision == "64" else 5e-5,
        eps=3e-5 if precision == "64" else 3e-2,
    )
