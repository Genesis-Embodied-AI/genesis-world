from typing import NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose


class DiffScenePair(NamedTuple):
    scene_ana: "gs.Scene"
    entity_ana: RigidEntity
    scene_fd: "gs.Scene"
    entity_fd: RigidEntity


def make_diff_scene_pair(
    mjcf,
    *,
    n_envs=0,
    substeps=1,
    dt=0.01,
    gravity=(0.0, 0.0, -9.81),
    enable_joint_limit=False,
    disable_constraint=True,
    modes=(True, False),
    show_viewer=False,
    camera_pos=(1.2, -1.2, 0.8),
    camera_lookat=(0.0, 0.0, 0.2),
):
    """Build a diff-mode scene (scene_ana, the only one backward() runs on) and a production-mode reference
    (scene_fd, the one finite differences perturb) from the same MJCF with identical config.

    The two kernels produce bit-identical forward states, so FD on scene_fd is a valid reference for scene_ana's
    analytical gradient. `modes` selects which of the pair to build (True = diff-mode scene_ana, False = production
    scene_fd); tests consuming only one side skip the other's kernel compilation."""
    scenes = {}
    entities = {}
    for requires_grad in modes:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=dt,
                substeps=substeps,
                gravity=gravity,
                requires_grad=requires_grad,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,
                enable_joint_limit=enable_joint_limit,
                disable_constraint=disable_constraint,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
            ),
            show_viewer=show_viewer and requires_grad,
        )
        entity = scene.add_entity(
            gs.morphs.MJCF(
                file=mjcf,
            ),
        )
        scene.build(n_envs=n_envs)
        scenes[requires_grad] = scene
        entities[requires_grad] = entity
    return DiffScenePair(scenes.get(True), entities.get(True), scenes.get(False), entities.get(False))


def assert_grad_matches_fd(
    pair, inputs, apply_fn, loss_fn, *, rtol, atol, eps, n_steps=None, setup_fn=None, step_fn=None
):
    """Central finite-difference check of a tracked setter's reverse-mode gradient.

    Input i of `inputs` is applied via `apply_fn(entity, x)` before step i and receives its own adjoint. The scene
    runs `n_steps` steps (default len(inputs)); extra steps run without re-applying. `setup_fn(scene, entity)` runs
    once after reset for untracked initialization; `step_fn(entity, i_step)` runs before every step on both scenes
    for undifferentiated per-step scenario commands (e.g. PD targets). The FD reference perturbs each input entry in
    turn, so the cost is O(n_steps * total input size). rtol / atol / eps are per-scenario, pinned to the measured
    finite-difference floor."""
    base = [np.array(inp, dtype=np.float64) for inp in inputs]
    total_steps = len(base) if n_steps is None else n_steps

    # Analytical pass on the diff-mode scene: apply each tracked input before its step, then backprop the loss.
    pair.scene_ana.reset()
    if setup_fn is not None:
        setup_fn(pair.scene_ana, pair.entity_ana)
    x_anas = []
    for i_step in range(total_steps):
        if i_step < len(base):
            x = gs.tensor(base[i_step], dtype=gs.tc_float, requires_grad=True)
            x_anas.append(x)
            apply_fn(pair.entity_ana, x)
        if step_fn is not None:
            step_fn(pair.entity_ana, i_step)
        pair.scene_ana.step()
    loss = loss_fn(pair.scene_ana, pair.entity_ana)
    assert loss.requires_grad, "loss does not require grad - output is not grad-aware"
    loss.backward()
    ana_grads = []
    for i_step, x in enumerate(x_anas):
        assert x.grad is not None, f"input {i_step}: x.grad is None after backward"
        ana_grads.append(tensor_to_array(x.grad))

    # Finite-difference reference on the production scene: perturb each entry of each input by +/- eps, re-run the
    # full trajectory for both signs, and central-difference the loss.
    for i_input in range(len(base)):
        fd_grad = np.zeros_like(base[i_input])
        for i_entry in range(base[i_input].size):
            perturbed = []
            for sign in (+1, -1):
                pair.scene_fd.reset()
                if setup_fn is not None:
                    setup_fn(pair.scene_fd, pair.entity_fd)
                for i_step in range(total_steps):
                    if i_step < len(base):
                        inp = base[i_step].copy()
                        if i_step == i_input:
                            inp.reshape(-1)[i_entry] += sign * eps
                        apply_fn(pair.entity_fd, inp)
                    if step_fn is not None:
                        step_fn(pair.entity_fd, i_step)
                    pair.scene_fd.step()
                perturbed.append(float(loss_fn(pair.scene_fd, pair.entity_fd)))
            fd_grad.reshape(-1)[i_entry] = (perturbed[0] - perturbed[1]) / (2.0 * eps)
        assert_allclose(
            ana_grads[i_input],
            fd_grad,
            rtol=rtol,
            atol=atol,
            err_msg=f"input {i_input}: FD vs analytical mismatch",
        )
