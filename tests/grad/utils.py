import dataclasses
import weakref
from typing import NamedTuple

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.grad.bridge import LaunchRecorder
from genesis.utils.misc import qd_to_torch, tensor_to_array

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
    """Check the reverse-mode gradient of the inputs given to a rollout against central finite differences (FD).

    Input i of `inputs` is applied via `apply_fn(entity, x)` before step i and receives its own adjoint, None for a step
    given no input. The scene runs `n_steps` steps (default len(inputs)), the extra ones without input.
    `setup_fn(scene, entity)` runs once after reset, for initialization without gradient. `step_fn(entity, i_step)` runs
    before every step on both scenes, for per-step scenario commands without gradient (e.g. position targets). The FD
    reference perturbs each input entry in turn, so the cost is O(n_steps * total input size). rtol / atol / eps are
    per-scenario, pinned to the measured finite-difference floor."""
    base = [None if inp is None else np.array(inp, dtype=np.float64) for inp in inputs]
    total_steps = len(base) if n_steps is None else n_steps

    # Analytical pass on the diff-mode scene: apply each tracked input before its step, then backprop the loss.
    pair.scene_ana.reset()
    if setup_fn is not None:
        setup_fn(pair.scene_ana, pair.entity_ana)
    x_anas = {}
    for i_step in range(total_steps):
        if i_step < len(base) and base[i_step] is not None:
            x = torch.tensor(base[i_step], dtype=gs.tc_float, device=gs.device, requires_grad=True)
            x_anas[i_step] = x
            apply_fn(pair.entity_ana, x)
        if step_fn is not None:
            step_fn(pair.entity_ana, i_step)
        pair.scene_ana.step()
    loss = loss_fn(pair.scene_ana, pair.entity_ana)
    assert loss.requires_grad, "loss does not require grad - output is not grad-aware"
    loss.backward()
    ana_grads = {}
    for i_step, x in x_anas.items():
        assert x.grad is not None, f"input {i_step}: x.grad is None after backward"
        ana_grads[i_step] = tensor_to_array(x.grad)

    # Finite-difference reference on the production scene: perturb each entry of each input by +/- eps, re-run the
    # full trajectory for both signs, and central-difference the loss.
    for i_input in x_anas:
        fd_grad = np.zeros_like(base[i_input])
        for i_entry in range(base[i_input].size):
            perturbed = []
            for sign in (+1, -1):
                pair.scene_fd.reset()
                if setup_fn is not None:
                    setup_fn(pair.scene_fd, pair.entity_fd)
                for i_step in range(total_steps):
                    if i_step < len(base) and base[i_step] is not None:
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


class StepOverwriteChecker:
    """
    Check that the backward pass of a recorded rollout saves every overwrite it needs (see LaunchRecorder).

    Before each entry of a recorder, it copies every array its launches take through struct arguments, and once the entry
    is over it reports each of them the entry changed without the recorder saving it, when an earlier entry of the same
    recorder used it. The arrays a kernel reaches through the solver it is a method of, such as the frames of the
    particle solvers, are outside of its reach.
    """

    def __init__(self, monkeypatch):
        self.overwrites: set[str] = set()
        checker = self
        insert, entry, reverse, exit_ = (
            LaunchRecorder.insert,
            LaunchRecorder.entry,
            LaunchRecorder.reverse,
            LaunchRecorder.__exit__,
        )

        def checked_insert(recorder, kernel, args):
            # The copies of the checker are kernel launches of their own without zero-copy views
            if checker._is_copying:
                return
            if not recorder._is_grouping:
                checker.close_entry(recorder)
            checker.track_launch(recorder, kernel, args)
            insert(recorder, kernel, args)

        def checked_entry(recorder, *args, **kwargs):
            checker.close_entry(recorder)
            return entry(recorder, *args, **kwargs)

        def checked_reverse(recorder, *args, **kwargs):
            checker.close_entry(recorder)
            return reverse(recorder, *args, **kwargs)

        # A recorder of the calls between two steps is exited after each call, before the step runs unrecorded
        def checked_exit(recorder, *args):
            checker.close_entry(recorder)
            return exit_(recorder, *args)

        monkeypatch.setattr(LaunchRecorder, "insert", checked_insert)
        monkeypatch.setattr(LaunchRecorder, "__exit__", checked_exit)
        monkeypatch.setattr(LaunchRecorder, "entry", checked_entry)
        monkeypatch.setattr(LaunchRecorder, "reverse", checked_reverse)
        self._is_copying = False
        # The arrays each recorder's closed entries used, the values before the open entry, and its launches, released
        # with the recorder, whose identity a later one may take
        self._arrays_used: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._values_before: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._launches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def track_launch(self, recorder, kernel, args) -> None:
        # Only an array an earlier entry used can be an overwrite the backward pass misses
        arrays_used = self._arrays_used.get(recorder, set())
        values_before = self._values_before.setdefault(recorder, {})
        self._is_copying = True
        for path, array in self._iter_arrays_used(kernel, args):
            if id(array) in arrays_used and id(array) not in values_before:
                values_before[id(array)] = (".".join(path), array, qd_to_torch(array, copy=True))
        self._is_copying = False
        self._launches.setdefault(recorder, []).append((kernel, args))

    def close_entry(self, recorder) -> None:
        values_before = self._values_before.pop(recorder, {})
        launches = self._launches.pop(recorder, [])
        if not launches:
            return
        arrays_used = self._arrays_used.setdefault(recorder, set())
        arrays_saved = {id(array) for array, _ in recorder.entries[-1].arrays_saved}
        self._is_copying = True
        for key, (path, array, value) in values_before.items():
            if key not in arrays_saved:
                value_after = qd_to_torch(array, copy=True)
                # Entries the solver leaves at NaN compare equal to themselves
                if value.is_floating_point():
                    is_unchanged = torch.allclose(value_after, value, rtol=0.0, atol=0.0, equal_nan=True)
                else:
                    is_unchanged = torch.equal(value_after, value)
                if not is_unchanged:
                    self.overwrites.add(f"{launches[0][0].func.__name__} overwrites {path}")
        self._is_copying = False
        for kernel, args in launches:
            for _, array in self._iter_arrays_used(kernel, args):
                arrays_used.add(id(array))

    @staticmethod
    def _iter_arrays_used(kernel, args):
        """Yield the arrays of its struct arguments that a launch uses, with their paths of parameter and fields."""
        key = kernel.ensure_compiled(*args)
        flat_names_used = kernel.used_py_dataclass_parameters_by_key_enforcing.get(key)
        for arg_meta, arg in zip(kernel.arg_metas, args):
            if dataclasses.is_dataclass(arg):
                stack = [((arg_meta.name,), arg)]
                while stack:
                    path, struct = stack.pop()
                    for field in dataclasses.fields(struct):
                        value = getattr(struct, field.name)
                        if isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray)):
                            array_path = (*path, field.name)
                            if flat_names_used is None or "__qd_" + "__qd_".join(array_path) in flat_names_used:
                                yield array_path, value
                        elif dataclasses.is_dataclass(value):
                            stack.append(((*path, field.name), value))
