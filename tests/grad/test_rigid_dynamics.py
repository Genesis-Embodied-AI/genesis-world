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
def test_fk_grad_matches_fd(model_name, request, precision, show_viewer, tol):
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
    # (setter, output, input_seed) per joint, then the position and quaternion target seeds. Anchored single-joint
    # topologies (revolute, spherical) read the quaternion only: their base link position is pinned at the joint
    # anchor, so a position check would compare two zero gradients. A single step of force moves the pose by a
    # squared timestep, a gradient the fp32 tolerance cannot tell from zero, so the force checks read the velocity.
    checks_by_joint = {
        "grad_free": (("pos", "pos", 10), ("quat", "quat", 11), ("vel", "pos", 12), ("vel", "quat", 13)),
        "grad_revolute": (("vel", "quat", 31), ("force", "vel", 32)),
        "grad_prismatic": (("vel", "pos", 50),),
        "grad_spherical": (("vel", "quat", 71), ("force", "vel", 72)),
        "grad_free_with_revolute": (("pos", "pos", 70), ("quat", "quat", 71), ("vel", "pos", 72), ("vel", "quat", 73)),
        "grad_chain3": (("vel", "pos", 90), ("vel", "quat", 91)),
        "grad_cartpole": (("vel", "pos", 190), ("vel", "quat", 191), ("force", "vel", 192)),
        "grad_hopper": (("vel", "pos", 210), ("vel", "quat", 211)),
    }
    target_seeds_by_joint = {
        "grad_free": (1, 2),
        "grad_revolute": (21, 22),
        "grad_prismatic": (41, 0),
        "grad_spherical": (61, 62),
        "grad_free_with_revolute": (61, 62),
        "grad_chain3": (81, 82),
        "grad_cartpole": (181, 182),
        "grad_hopper": (201, 202),
    }
    checks = checks_by_joint[model_name]
    pos_seed, quat_seed = target_seeds_by_joint[model_name]

    pos_shape = (B, 3) if is_single_link else (B, n_links, 3)
    quat_shape = (B, 4) if is_single_link else (B, n_links, 4)
    tgt_pos = gs.tensor(np.random.RandomState(pos_seed).standard_normal(pos_shape), dtype=gs.tc_float).reshape(-1)
    tgt_quat = gs.tensor(np.random.RandomState(quat_seed).standard_normal(quat_shape), dtype=gs.tc_float).reshape(-1)
    tgt_vel = gs.tensor(np.random.RandomState(pos_seed + quat_seed).standard_normal((B, n_dofs)), dtype=gs.tc_float)
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

        target = {"pos": tgt_pos, "quat": tgt_quat, "vel": tgt_vel.reshape(-1)}[output]

        def loss_fn(scene, entity, tgt=target, out=output, sl=is_single_link):
            if out == "vel":
                pose = scene.rigid_solver.get_state().dofs_vel
            elif sl:
                pose = entity.get_state().pos if out == "pos" else entity.get_state().quat
            else:
                state = scene.rigid_solver.get_state()
                pose = state.links_pos if out == "pos" else state.links_quat
            return ((pose.reshape(-1) - tgt) ** 2).sum()

        # The quaternion input is normalized on entry, so its response bends within a fraction of a unit and the
        # finite-difference step must stay far below the one the linear position, velocity and force inputs afford.
        if setter == "quat":
            eps = 1e-3 if precision == "64" else 1e-1
        else:
            eps = 3e-1 if precision == "64" else 3e0
        assert_grad_matches_fd(pair, [step_input], apply_fn, loss_fn, rtol=tol, atol=tol, eps=eps)


@pytest.mark.required
@pytest.mark.debug(False)
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
@pytest.mark.parametrize("loss_kind", ["velocity", pytest.param("pose", marks=pytest.mark.precision("64"))])
def test_fk_multistep_force_grad_matches_fd(model_name, loss_kind, request, precision, show_viewer, tol):
    # Ten distinct per-step control forces, each of which must receive an independent adjoint across the unroll. A
    # force moves the final joint velocity by one timestep over the inertia it drives whatever the step it acts at,
    # where its imprint on the final pose fades with the steps left and drops under the fp32 tolerance for the late
    # ones, so the pose loss runs in double precision only. (pose output kind: entity state pos / quat or rigid-solver
    # links, per-link output shape, target seed). Anchored single-joint topologies (revolute, spherical) read the
    # quaternion: their base link position is pinned at the joint anchor, so a position loss would be constant.
    output, output_shape, seed = {
        "grad_free": ("state_pos", (3,), 161),
        "grad_revolute": ("state_quat", (4,), 162),
        "grad_prismatic": ("state_pos", (3,), 163),
        "grad_free_with_revolute": ("links", (2, 3), 164),
        "grad_chain3": ("links", (3, 3), 165),
        "grad_spherical": ("state_quat", (4,), 166),
        "grad_cartpole": ("links", (2, 3), 167),
        "grad_hopper": ("links", (5, 3), 168),
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
    target_shape = (n_dofs,) if loss_kind == "velocity" else (1, *output_shape)
    target = gs.tensor(np.random.RandomState(seed).standard_normal(target_shape), dtype=gs.tc_float).reshape(-1)
    inputs = [np.random.default_rng(seed * 100 + t).standard_normal((n_dofs,)) for t in range(10)]

    def loss_fn(scene, entity):
        if loss_kind == "velocity":
            out = scene.rigid_solver.get_state().dofs_vel
        elif output == "state_pos":
            out = entity.get_state().pos
        elif output == "state_quat":
            out = entity.get_state().quat
        else:
            out = scene.rigid_solver.get_state().links_pos
        return ((out.reshape(-1) - target) ** 2).sum()

    def apply_force(entity, force):
        # Split the same-step control across two dof subsets (the standard arm + gripper pattern): each call must
        # keep its own tape slot and gradient path. The second subset is passed as a slice, a valid index form the
        # tape key must accept on every backend.
        if n_dofs == 1:
            entity.control_dofs_force(force)
        else:
            entity.control_dofs_force(force[..., :1], dofs_idx_local=[0])
            entity.control_dofs_force(force[..., 1:], dofs_idx_local=slice(1, n_dofs))

    assert_grad_matches_fd(
        pair, inputs, apply_force, loss_fn, rtol=tol, atol=tol, eps=1e-3 if precision == "64" else 1e0
    )


@pytest.mark.required
@pytest.mark.parametrize("control_mode", ["position", "velocity"])
def test_per_step_pd_target_grad_matches_fd(control_mode, grad_revolute, precision, show_viewer, tol):
    # Per-step-varying PD targets are scenario commands, replayed by the backward unroll; the gradient of a tracked
    # initial velocity through the controlled rollout must match finite differences.
    pair = make_diff_scene_pair(grad_revolute, substeps=4, show_viewer=show_viewer)
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
        rtol=tol,
        atol=tol,
        eps=1e-3 if precision == "64" else 3e0,
    )
