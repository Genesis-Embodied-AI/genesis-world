import math

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_torch, tensor_to_array

from ..utils.assertions import assert_allclose
from .utils import assert_grad_matches_fd, make_diff_scene_pair


@pytest.mark.required
@pytest.mark.debug(False)
def test_joint_limit_grad_matches_fd(grad_slider_limit, precision, show_viewer):
    # Forward: the slider limit must actually bound the cart (it drifts freely when the constraint is off).
    off = make_diff_scene_pair(
        grad_slider_limit,
        n_envs=2,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=True,
        modes=(False,),
    )
    off.scene_fd.reset()
    off.entity_fd.set_dofs_velocity(100.0)
    for _ in range(60):
        off.scene_fd.step()
    assert (off.scene_fd.rigid_solver.get_state().qpos[:, 0].abs() > 50.0).all()

    on = make_diff_scene_pair(
        grad_slider_limit,
        n_envs=2,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=True,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    on.scene_fd.reset()
    on.entity_fd.set_dofs_velocity(100.0)
    for _ in range(60):
        on.scene_fd.step()
    assert (on.scene_fd.rigid_solver.get_state().qpos[:, 0].abs() <= 4.5).all()

    # Backward, with mixed per-environment activity: env 0 drives into the active |x|=4 limit while env 1 stays well
    # inside it, so the adjoint solve faces different constraint counts in the same batch. Sanity-check the split.
    on.scene_fd.reset()
    on.entity_fd.set_dofs_velocity([[100.0], [2.0]])
    for _ in range(5):
        on.scene_fd.step()
    qpos_end = on.scene_fd.rigid_solver.get_state().qpos
    assert abs(qpos_end[0, 0]) > 3.5
    assert abs(qpos_end[1, 0]) < 1.0

    assert_grad_matches_fd(
        on,
        [np.array([[100.0], [2.0]])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: (scene.rigid_solver.get_state().qpos[:, 0] ** 2).sum(),
        n_steps=5,
        rtol=1e-10 if precision == "64" else 5e-4,
        atol=1e-10 if precision == "64" else 5e-4,
        eps=3e-4 if precision == "64" else 3e-2,
    )

    # Inactive-path parity: with the limit enabled but never hit, the adjoint must equal the no-limit baseline - the
    # inactive constraint branch must inject no spurious gradient.
    off_solver = make_diff_scene_pair(
        grad_slider_limit,
        n_envs=2,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
        modes=(True,),
    )
    grads = {}
    for pair, key in ((off_solver, "off"), (on, "on")):
        pair.scene_ana.reset()
        v = gs.tensor([[0.5], [0.3]], dtype=gs.tc_float, requires_grad=True)
        pair.entity_ana.set_dofs_velocity(v)
        pair.scene_ana.step()
        loss = (pair.scene_ana.rigid_solver.get_state().qpos[:, 0] ** 2).sum()
        loss.backward()
        grads[key] = tensor_to_array(v.grad)
    assert_allclose(
        grads["on"],
        grads["off"],
        rtol=1e-6 if precision == "64" else 1e-4,
        atol=1e-9 if precision == "64" else 1e-6,
    )


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["grad_slider_limit", "grad_cartpole", "grad_hopper"])
@pytest.mark.debug(False)
def test_per_step_force_into_limit_grad_matches_fd(model_name, request, precision, show_viewer):
    # Per-step control-force adjoint driving a joint into its limit, across three topologies. A constant force over
    # the horizon pushes the tracked dof into the active band; the setup-sanity assert guards against a vacuous run.
    # (gravity, n_steps, per-step force, loss reads links_pos, sanity dof, sanity threshold, initial dof pose,
    # fp32 tolerance).
    gravity, n_steps, per_step_force, is_links_loss, sanity_dof, sanity_thresh, init_pos, fp32_tol = {
        "grad_slider_limit": ((0.0, 0.0, 0.0), 10, [500.0], False, 0, 3.5, None, 1e-4),
        "grad_cartpole": ((0.0, 0.0, -9.81), 15, [2000.0, 0.0], False, 0, 3.5, [0.0, -math.pi], 2e-4),
        "grad_hopper": ((0.0, 0.0, 0.0), 10, [0.0, 0.0, 0.0, 0.0, 0.0, 200.0], True, 5, 0.7, None, 5e-5),
    }[model_name]

    pair = make_diff_scene_pair(
        request.getfixturevalue(model_name),
        substeps=4,
        dt=1.0 / 60.0,
        gravity=gravity,
        enable_joint_limit=True,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    forces = [np.array(per_step_force) for _ in range(n_steps)]

    def setup_fn(scene, entity):
        if init_pos is not None:
            entity.set_dofs_position(init_pos)

    def loss_fn(scene, entity):
        state = scene.rigid_solver.get_state()
        return (state.links_pos.reshape(-1) ** 2).sum() if is_links_loss else state.qpos[0, 0] ** 2

    pair.scene_fd.reset()
    setup_fn(pair.scene_fd, pair.entity_fd)
    for force in forces:
        pair.entity_fd.control_dofs_force(force)
        pair.scene_fd.step()
    reached = abs(pair.scene_fd.rigid_solver.get_state().qpos[0, sanity_dof])
    assert reached > sanity_thresh, f"setup error: {model_name} did not reach its limit band (q={reached})"

    assert_grad_matches_fd(
        pair,
        forces,
        lambda e, x: e.control_dofs_force(x),
        loss_fn,
        setup_fn=setup_fn,
        rtol=1e-10 if precision == "64" else fp32_tol,
        atol=1e-10 if precision == "64" else fp32_tol,
        eps=3e-2,
    )


@pytest.mark.required
def test_frictionloss_grad_matches_fd(grad_revolute_frictionloss, precision, show_viewer):
    pair = make_diff_scene_pair(
        grad_revolute_frictionloss,
        n_envs=2,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    pair.scene_fd.reset()
    pair.scene_fd.step()
    cs = pair.scene_fd.rigid_solver.constraint_solver.constraint_state
    assert (qd_to_torch(cs.n_constraints_frictionloss) == 1).all()

    assert_grad_matches_fd(
        pair,
        [np.array([[2.0], [1.2]])],
        lambda e, x: e.set_dofs_velocity(x),
        lambda scene, entity: (scene.rigid_solver.get_state().qpos[:, 0] ** 2).sum(),
        n_steps=10,
        rtol=1e-10 if precision == "64" else 5e-5,
        atol=1e-10 if precision == "64" else 5e-5,
        eps=3e-6 if precision == "64" else 1e-2,
    )


@pytest.mark.required
@pytest.mark.parametrize(
    "model_name, n_rows",
    [
        ("grad_hinge_pair_joint_eq_linear", 1),
        ("grad_hinge_pair_joint_eq_quadratic", 1),
        ("grad_hinge_joint_eq_constant", 1),
        ("grad_connect_loop", 3),
        ("grad_weld_pair", 6),
    ],
)
def test_equality_grad_matches_fd(model_name, n_rows, request, precision, show_viewer):
    pair = make_diff_scene_pair(
        request.getfixturevalue(model_name),
        n_envs=2,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    pair.scene_fd.reset()
    pair.scene_fd.step()
    cs = pair.scene_fd.rigid_solver.constraint_solver.constraint_state
    assert (qd_to_torch(cs.n_constraints_equality) == n_rows).all()

    # Large initial velocities: the anchor velocity-product bias entering aref is quadratic in velocity, so its
    # adjoint contribution only clears the fp64 tolerance band when the joints spin fast.
    def loss_fn(scene, entity):
        qpos = scene.rigid_solver.get_state().qpos
        return (qpos[:, 0] ** 2 + 0.7 * qpos[:, 1] ** 2).sum()

    assert_grad_matches_fd(
        pair,
        [np.array([[4.0, -2.5], [-3.0, 2.0]])],
        lambda e, x: e.set_dofs_velocity(x),
        loss_fn,
        n_steps=10,
        rtol=1e-10 if precision == "64" else 5e-5,
        atol=1e-10 if precision == "64" else 5e-5,
        eps=1e-3 if precision == "64" else 3e-2,
    )


@pytest.mark.required
def test_all_constraint_groups_grad_matches_fd(grad_all_eq_fric, precision, show_viewer):
    # Integration scene: frictionloss + equality joint + connect + weld on disjoint link pairs. Guards row-offset
    # bookkeeping across every differentiated constraint group at once; per-group formulas are pinned elsewhere.
    pair = make_diff_scene_pair(
        grad_all_eq_fric,
        substeps=4,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, 0.0),
        enable_joint_limit=False,
        disable_constraint=False,
        show_viewer=show_viewer,
    )
    pair.scene_fd.reset()
    pair.scene_fd.step()
    cs = pair.scene_fd.rigid_solver.constraint_solver.constraint_state
    assert qd_to_torch(cs.n_constraints_equality)[0] == 10
    assert qd_to_torch(cs.n_constraints_frictionloss)[0] == 1

    weights = np.array([1.0, 0.7, 1.3, 0.5, 0.9, 1.1, 0.6])

    def loss_fn(scene, entity):
        qpos = scene.rigid_solver.get_state().qpos[0]
        return sum(weights[d] * qpos[d] ** 2 for d in range(7))

    assert_grad_matches_fd(
        pair,
        [np.array([0.8, -0.3, 0.5, -0.2, 0.2, -0.3, 0.4])],
        lambda e, x: e.set_dofs_velocity(x),
        loss_fn,
        n_steps=10,
        rtol=1e-10 if precision == "64" else 5e-5,
        atol=1e-10 if precision == "64" else 5e-5,
        eps=3e-4 if precision == "64" else 3e-3,
    )
