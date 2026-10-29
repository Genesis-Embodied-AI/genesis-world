import math

import numpy as np
import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose
from .utils import assert_grad_matches_fd, make_diff_scene_pair


@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "model_name",
    [
        "grad_free",
        "grad_revolute",
        "grad_prismatic",
        pytest.param("grad_spherical", marks=pytest.mark.required),
        pytest.param("grad_free_with_revolute", marks=pytest.mark.required),
        "grad_chain3",
        "grad_cartpole",
        pytest.param("grad_hopper", marks=pytest.mark.required),
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
        "grad_revolute": ((("vel", "quat", 31), ("force", "quat", 32)), 21, 22),
        "grad_prismatic": ((("vel", "pos", 50),), 41, 0),
        "grad_spherical": ((("vel", "quat", 71), ("force", "quat", 72)), 61, 62),
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

    # Per-topology tolerances (fp64, fp32), pinned to the measured finite-difference floors
    fp64_tol, fp32_tol = {
        "grad_free": (2e-10, 5e-5),
        "grad_revolute": (1e-10, 5e-7),
        "grad_prismatic": (1e-10, 5e-6),
        "grad_spherical": (1e-10, 1e-6),
        "grad_free_with_revolute": (2e-10, 5e-5),
        "grad_chain3": (1e-10, 1e-4),
        "grad_cartpole": (1e-10, 5e-5),
        "grad_hopper": (1e-10, 1e-4),
    }[model_name]

    pos_shape = (B, 3) if is_single_link else (B, n_links, 3)
    quat_shape = (B, 4) if is_single_link else (B, n_links, 4)
    tgt_pos = torch.tensor(
        np.random.RandomState(pos_seed).standard_normal(pos_shape), dtype=gs.tc_float, device=gs.device
    ).reshape(-1)
    # The orientations stay close to the identity over the rollout, and so does their target, which keeps the loss small
    # enough for its rounding to stay below the finite-difference floor.
    tgt_quat = torch.tensor(
        np.array([1.0, 0.0, 0.0, 0.0]) + 0.1 * np.random.RandomState(quat_seed).standard_normal(quat_shape),
        dtype=gs.tc_float,
        device=gs.device,
    ).reshape(-1)
    for setter, output, input_seed in checks:
        # The setter is given before the first step and again before the second, mid-rollout
        rng = np.random.default_rng(input_seed)
        step_inputs = []
        for _ in range(2):
            if setter == "pos":
                step_input = rng.standard_normal((B, 3))
            elif setter == "quat":
                step_input = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), (B, 4)).copy()
                step_input = step_input + 0.05 * rng.standard_normal((B, 4))
                step_input = step_input / np.linalg.norm(step_input, axis=-1, keepdims=True)
            else:
                step_input = rng.standard_normal((B, n_dofs))
            step_inputs.append(step_input)

        def apply_vel_per_env(e, x):
            # Same-step per-environment commands, each call keeping its own gradient path
            e.set_dofs_velocity(x[:1], envs_idx=[0])
            e.set_dofs_velocity(x[1:], envs_idx=[1])

        apply_fn = {
            "pos": lambda e, x: e.set_pos(x),
            "quat": lambda e, x: e.set_quat(x),
            "vel": apply_vel_per_env,
            "force": lambda e, x: e.control_dofs_force(x),
        }[setter]

        target = tgt_pos if output == "pos" else tgt_quat

        # A wrench is applied at a world point between each setter and the step it applies to, whose moment arm follows
        # the pose the setter left. The pose is read there too.
        poses = []

        def read_pose(entity, out=output, sl=is_single_link):
            if sl:
                return entity.get_state().pos if out == "pos" else entity.get_state().quat
            state = entity.solver.get_state()
            return state.links_pos if out == "pos" else state.links_quat

        def clear_poses(scene, entity):
            poses.clear()

        def push_and_append_pose(entity, i_step):
            entity.apply_links_external_wrench(
                force=(0.3, -0.2, 0.1), links_idx_local=[n_links - 1], pos=(0.1, -0.1, 0.3)
            )
            poses.append(read_pose(entity))

        def loss_fn(scene, entity, tgt=target):
            return sum([((pose.reshape(-1) - tgt) ** 2).sum() for pose in (*poses, read_pose(entity))])

        # The normalization of a quaternion makes its setter the most curved input, which bounds its step by the
        # truncation error. Every other step is bounded by the rounding error alone.
        fp64_eps, fp32_eps = (1.5e-5, 5e-3) if setter == "quat" else (3.6e-4, 1.2e-1)
        assert_grad_matches_fd(
            pair,
            step_inputs,
            apply_fn,
            loss_fn,
            rtol=fp64_tol if precision == "64" else fp32_tol,
            atol=fp64_tol if precision == "64" else fp32_tol,
            eps=fp64_eps if precision == "64" else fp32_eps,
            setup_fn=clear_poses,
            step_fn=push_and_append_pose,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "grad_free",
        "grad_revolute",
        "grad_prismatic",
        pytest.param("grad_free_with_revolute", marks=pytest.mark.required),
        "grad_chain3",
        pytest.param("grad_spherical", marks=pytest.mark.required),
        "grad_cartpole",
        pytest.param("grad_hopper", marks=pytest.mark.required),
    ],
)
@pytest.mark.debug(False)
def test_fk_multistep_force_grad_matches_fd(model_name, request, precision, show_viewer):
    # Distinct control forces over the unroll, each of which must receive an independent adjoint.
    # (output kind: entity state pos / quat or rigid-solver links, per-link output shape, target seed, fp32
    # tolerance). Anchored single-joint topologies (revolute, spherical) read the quaternion: their base link
    # position is pinned at the joint anchor, so a position loss would be constant and the check vacuous.
    output, output_shape, seed, fp32_tol = {
        "grad_free": ("state_pos", (3,), 161, 1e-6),
        "grad_revolute": ("state_quat", (4,), 162, 1e-6),
        "grad_prismatic": ("state_pos", (3,), 163, 5e-6),
        "grad_free_with_revolute": ("links", (2, 3), 164, 1e-5),
        "grad_chain3": ("links", (3, 3), 165, 5e-6),
        "grad_spherical": ("state_quat", (4,), 166, 2e-5),
        "grad_cartpole": ("links", (2, 3), 167, 5e-6),
        "grad_hopper": ("links", (5, 3), 168, 5e-5),
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
    target = torch.tensor(
        np.random.RandomState(seed).standard_normal((1, *output_shape)), dtype=gs.tc_float, device=gs.device
    ).reshape(-1)
    # A force is given every third step and holds over the steps in between, the next one overwriting it. An external
    # wrench, whose moment arm follows the pose the earlier forces drove the link to, is applied to one step in between.
    inputs = [
        np.random.default_rng(seed * 100 + t).standard_normal((n_dofs,)) if t % 3 == 0 else None for t in range(10)
    ]
    inputs[4] = np.random.default_rng(seed * 100 + 4).standard_normal((1, 3))

    def loss_fn(scene, entity):
        if output == "state_pos":
            pose = entity.get_state().pos
        elif output == "state_quat":
            pose = entity.get_state().quat
        else:
            pose = entity.get_links_pos()
        return ((pose.reshape(-1) - target) ** 2).sum()

    def apply_force(entity, force):
        # The wrench is the one input given per link, hence two-dimensional
        if force.ndim == 2:
            entity.apply_links_external_wrench(
                force=force, links_idx_local=[entity.n_links - 1], pos=(0.01, 0.02, -0.03), local=True
            )
        # Split the same-step control across two dof subsets (the standard arm + gripper pattern), each call keeping its
        # own gradient path. The second subset is passed as a slice, and given one scalar broadcast to all its dofs,
        # which receives the gradient summed over them.
        elif n_dofs == 1:
            entity.control_dofs_force(force)
        else:
            entity.control_dofs_force(force[..., :1], dofs_idx_local=[0])
            entity.control_dofs_force(force[..., 1], dofs_idx_local=slice(1, n_dofs))

    # At both precisions, the finite-difference step is bounded by the rounding error alone, which falls as it grows
    assert_grad_matches_fd(
        pair,
        inputs,
        apply_force,
        loss_fn,
        rtol=1e-10 if precision == "64" else fp32_tol,
        atol=1e-10 if precision == "64" else fp32_tol,
        eps=1.2e-4 if precision == "64" else 1.2e-1,
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
        lambda scene, entity: entity.get_qpos()[0] ** 2,
        n_steps=10,
        step_fn=step_fn,
        rtol=1e-10 if precision == "64" else 5e-5,
        atol=1e-10 if precision == "64" else 5e-5,
        eps=3e-5 if precision == "64" else 3e-2,
    )


@pytest.mark.required
def test_requires_grad_parity(grad_free, grad_hopper, grad_chain3, tol, show_viewer):
    scenes = []
    for requires_grad in (True, False):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                requires_grad=requires_grad,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, -3.5, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
            ),
            show_viewer=show_viewer and requires_grad,
        )
        for mjcf, pos in (
            (grad_free, (0.0, 0.0, 1.0)),
            (grad_hopper, (1.5, 0.0, 0.0)),
            (grad_chain3, (-1.5, 0.0, 1.0)),
        ):
            scene.add_entity(
                gs.morphs.MJCF(
                    file=mjcf,
                    pos=pos,
                ),
            )
        scene.build(n_envs=2)
        scenes.append(scene)

    for entities in zip(*[scene.entities for scene in scenes]):
        dofs_vel = np.random.uniform(-2.0, 2.0, (2, entities[0].n_dofs))
        for entity in entities:
            entity.set_dofs_velocity(dofs_vel)
    for _ in range(30):
        for scene in scenes:
            scene.step()
        for entity_grad, entity in zip(*[scene.entities for scene in scenes]):
            assert_allclose(entity_grad.get_qpos(), entity.get_qpos(), tol=tol)
            assert_allclose(entity_grad.get_dofs_velocity(), entity.get_dofs_velocity(), tol=tol)
