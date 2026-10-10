import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

# The three ways a step can run must produce the same trajectory and the same Newton iteration counts: the pipeline
# (host-driven loops), the monolith (one thread per environment) and the graph kernel (device-side loops, batched).
STEP_KERNELS = ("pipeline", "monolith", "graph")


def _scene(step_kernel, show_viewer, **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(step_kernel=step_kernel, **mochi_kwargs),
        show_viewer=show_viewer,
    )


def _step_and_record(scene, n_steps):
    """Step, recording the Newton iteration counts and statuses of every environment after every step."""
    n_iter, status = [], []
    for _ in range(n_steps):
        scene.step()
        info = scene.mochi_solver.get_convergence_info()
        n_iter.append(tensor_to_array(info["n_iter"]))
        status.append(tensor_to_array(info["status"]))
    return np.array(n_iter), np.array(status)


def _run_rigid(step_kernel, show_viewer, n_steps):
    scene = _scene(step_kernel, show_viewer, n_newton_iterations=4)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    lower = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0.0, 0.0, 0.3)), material=gs.materials.Mochi.Rigid()
    )
    upper = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.05, 0.0, 0.6)), material=gs.materials.Mochi.Rigid()
    )
    scene.build(n_envs=4)
    # different heights per environment so that the Newton solves of the batch are heterogeneous
    upper.set_pos(np.array([[0.05, 0.0, 0.6 + 0.1 * i_b] for i_b in range(4)]))
    n_iter, status = _step_and_record(scene, n_steps)
    return (
        tensor_to_array(lower.get_pos()),
        tensor_to_array(upper.get_pos()),
        tensor_to_array(upper.get_quat()),
        n_iter,
        status,
    )


def _run_soft(step_kernel, show_viewer, n_steps):
    scene = _scene(step_kernel, show_viewer, n_newton_iterations=4)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    slab = scene.add_entity(
        gs.morphs.Box(size=(0.4, 0.4, 0.1), pos=(0.0, 0.0, 0.05), maxvolume=0.001, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=2e5, nu=0.4, rho=1000.0),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.12, 0.12, 0.1), pos=(0.0, 0.0, 0.2)), material=gs.materials.Mochi.Rigid()
    )
    scene.build(n_envs=3)
    box.set_pos(np.array([[0.0, 0.0, 0.2], [0.05, 0.0, 0.25], [-0.05, 0.03, 0.3]]))
    n_iter, status = _step_and_record(scene, n_steps)
    return tensor_to_array(slab.get_vertices_position()), tensor_to_array(box.get_pos()), n_iter, status


@pytest.mark.precision("64")
def test_step_kernels_agree_rigid(show_viewer):
    results = {kernel: _run_rigid(kernel, show_viewer, 40) for kernel in STEP_KERNELS}
    reference = results["pipeline"]
    assert reference[3].max() >= 1
    for kernel in STEP_KERNELS[1:]:
        for a, b in zip(reference[:3], results[kernel][:3]):
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)
        np.testing.assert_array_equal(reference[3], results[kernel][3])
        np.testing.assert_array_equal(reference[4], results[kernel][4])


@pytest.mark.precision("64")
def test_step_kernels_agree_soft(show_viewer):
    results = {kernel: _run_soft(kernel, show_viewer, 30) for kernel in STEP_KERNELS}
    reference = results["pipeline"]
    assert reference[2].max() >= 1
    for kernel in STEP_KERNELS[1:]:
        # The batched assembly and the per-environment assembly sum the same terms in different orders; the
        # conjugate gradient then differs at rounding level.
        for a, b in zip(reference[:2], results[kernel][:2]):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(reference[2], results[kernel][2])
        np.testing.assert_array_equal(reference[3], results[kernel][3])
