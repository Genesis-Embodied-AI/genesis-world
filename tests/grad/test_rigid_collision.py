import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils import set_random_seed
from genesis.utils.geom import R_to_quat
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

from ..utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("shape", ["ground_box", "box", "sphere", "capsule"])
def test_contact_per_step_force_grad_matches_fd(shape, grad_capsule, precision, show_viewer):
    def _build_contact_scene(shape, mjcf_capsule, *, requires_grad, show_viewer=False):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                substeps=2,
                requires_grad=requires_grad,
            ),
            rigid_options=gs.options.RigidOptions(
                # A non-neutral impratio makes the regularized cone coefficient observable to finite differences
                impratio=2.0,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.2, -1.2, 0.8),
                camera_lookat=(0.0, 0.0, 0.2),
            ),
            show_viewer=show_viewer,
        )
        if shape == "ground_box":
            scene.add_entity(
                gs.morphs.Box(
                    size=(2.0, 2.0, 0.2),
                    pos=(0.0, 0.0, 0.1),
                    fixed=True,
                ),
                vis_mode="collision",
            )
            obj = scene.add_entity(
                gs.morphs.Box(
                    size=(0.4, 0.4, 0.4),
                    pos=(0.0, 0.0, 0.4),
                ),
                vis_mode="collision",
            )
        else:
            scene.add_entity(gs.morphs.Plane())
            if shape == "box":
                obj = scene.add_entity(
                    gs.morphs.Box(
                        size=(0.4, 0.4, 0.4),
                        pos=(0.0, 0.0, 0.3),
                    ),
                    vis_mode="collision",
                )
            elif shape == "sphere":
                obj = scene.add_entity(
                    gs.morphs.Sphere(
                        radius=0.2,
                        pos=(0.0, 0.0, 0.3),
                    ),
                    vis_mode="collision",
                )
            elif shape == "capsule":
                obj = scene.add_entity(
                    gs.morphs.MJCF(
                        file=mjcf_capsule,
                        align=False,
                    ),
                    vis_mode="collision",
                )
            else:
                raise ValueError(shape)
        scene.build(n_envs=0)
        return scene, obj

    def _n_contacts(scene):
        return qd_to_numpy(scene.rigid_solver.collider.collider_state.n_contacts)[0]

    # Rest z puts the body's lowest point on its support: box / sphere half extent 0.2, upright capsule
    # radius 0.1 + half length 0.2 = 0.3, box-on-ground centered at 0.40.
    rest_z = {"ground_box": 0.40, "box": 0.20, "sphere": 0.20, "capsule": 0.30}[shape]
    rest_dofs = [0.0, 0.0, rest_z, 0.0, 0.0, 0.0]
    n_settle = 12
    n_steps = 2
    eps = 1e-2
    # Contact force gradients are tiny (stiff contact barely moves): at fp32 the FD reference mostly rounds to
    # zero, so the absolute tolerance carries the check; at fp64 the relative tolerance does and the absolute one
    # only guards the degenerate-FD case.
    fd_rtol = 1e-6 if precision == "64" else 2e-3
    fd_atol = 1e-14 if precision == "64" else 1e-6
    # The small tangential preload keeps the contact manifold away from a gain / loss transition, which the
    # tangential finite-difference perturbation would otherwise cross (the in-test contact-count guard trips).
    base_force = np.array([0.5, 0.0, -8.0, 0.0, 0.0, 0.0])
    init_force = np.broadcast_to(base_force, (n_steps, 6)).copy()

    def settle(scene, obj):
        # Settling under the same base force as the grad window keeps the contact manifold in its loaded
        # configuration throughout, so the in-window contact-count guards hold.
        obj.set_dofs_position(gs.tensor(rest_dofs, dtype=gs.tc_float).sceneless())
        for _ in range(n_settle):
            obj.control_dofs_force(base_force)
            scene.step()

    scene_ana, obj_ana = _build_contact_scene(shape, grad_capsule, requires_grad=True, show_viewer=show_viewer)
    scene_ana.reset()
    settle(scene_ana, obj_ana)
    nc = _n_contacts(scene_ana)
    assert nc > 0, f"setup error: {shape} not in contact after settle (n_contacts={nc})"
    forces = [gs.tensor(init_force[t], dtype=gs.tc_float, requires_grad=True) for t in range(n_steps)]
    for t in range(n_steps):
        obj_ana.control_dofs_force(forces[t])
        scene_ana.step()
        assert _n_contacts(scene_ana) == nc, "contact set changed during grad window - FD invalid"
    loss = (scene_ana.rigid_solver.get_state().qpos[0, :3] ** 2).sum()
    scene_ana.backward(loss)
    ana = np.stack([tensor_to_array(f.grad) for f in forces])

    # The FD reference must also run in diff mode: differentiable scenes route contacts through diff_gjk, whose
    # contact set and forces differ slightly from the production narrowphase, so a production-mode reference would
    # measure that forward gap instead of the gradient.
    scene_fd, obj_fd = _build_contact_scene(shape, grad_capsule, requires_grad=True)

    def loss_at(perturbed):
        scene_fd.reset()
        settle(scene_fd, obj_fd)
        for t in range(n_steps):
            obj_fd.control_dofs_force(perturbed[t])
            scene_fd.step()
            assert _n_contacts(scene_fd) == nc, "contact set changed under FD perturbation"
        return float((scene_fd.rigid_solver.get_state().qpos[0, :3] ** 2).sum().detach())

    # Both the normal (z) and a tangential (x) force component: the tangential adjoint flows through the friction
    # rows and their contact-frame chains, which the normal component alone leaves untested.
    for t in range(n_steps):
        for i_d in (2, 0):
            plus = init_force.copy()
            plus[t, i_d] += eps
            minus = init_force.copy()
            minus[t, i_d] -= eps
            fd = (loss_at(plus) - loss_at(minus)) / (2 * eps)
            assert_allclose(
                ana[t, i_d], fd, rtol=fd_rtol, atol=fd_atol, err_msg=f"contact force.grad mismatch at t={t} dof={i_d}"
            )


@pytest.mark.required
def test_contact_no_tunneling_forward(show_viewer):
    # Differentiable contact detection must route convex-convex pairs through the monolithic diff_gjk path so
    # stacked boxes keep colliding under requires_grad. Observable: each top box stays on its support and comes to
    # rest instead of tunneling.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 1.2),
            camera_lookat=(0.0, 0.0, 0.4),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    tops = []
    for x in (0.8, -0.8):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.6, 0.6, 0.4),
                pos=(x, 0.0, 0.2),
                fixed=True,
            ),
            vis_mode="collision",
        )
        tops.append(
            scene.add_entity(
                gs.morphs.Box(
                    size=(0.4, 0.4, 0.4),
                    pos=(x, 0.0, 0.6),
                ),
                vis_mode="collision",
            )
        )
    scene.build()

    for _ in range(20):
        scene.step()

    for top, x in zip(tops, (0.8, -0.8)):
        assert_allclose(top.get_pos(), (x, 0.0, 0.6), atol=2e-4)
        assert_allclose(top.get_dofs_velocity(), 0.0, atol=0.05)


@pytest.mark.required
def test_diff_unsupported_configuration_raises():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.2,
            pos=(0.0, 0.0, 0.2),
            fixed=True,
        ),
    )
    ellipsoid_mjcf = ET.Element("mujoco", model="ellipsoid")
    ellipsoid_body = ET.SubElement(ET.SubElement(ellipsoid_mjcf, "worldbody"), "body", name="ellipsoid", pos="0 0 0.5")
    ET.SubElement(ellipsoid_body, "geom", type="ellipsoid", size="0.2 0.15 0.1")
    ET.SubElement(ellipsoid_body, "joint", name="ellipsoid_joint", type="free")
    scene.add_entity(
        gs.morphs.MJCF(file=ET.tostring(ellipsoid_mjcf, encoding="unicode")),
    )

    # A sphere/ellipsoid pair has an everywhere-curved Minkowski boundary on which diff_gjk's EPA never converges,
    # so it would silently tunnel; the build must reject it instead. A sphere/sphere pair is exempt, since it is
    # reconstructed in closed form.
    with pytest.raises(gs.GenesisException):
        scene.build()

    # Torsional / rolling friction rows have no manual reverse; the build must reject them
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_torsional_friction=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
    )
    with pytest.raises(gs.GenesisException):
        scene.build()

    # Attach merges kinematic trees across entities, which the per-entity backward kernels cannot reverse
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        show_viewer=False,
    )
    base = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
        ),
    )
    child = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.8),
        ),
    )
    with pytest.raises(gs.GenesisException):
        child.attach(base)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.debug(False)
def test_contact_detection_jacobian_matches_fd():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        show_viewer=False,
    )
    box_size = 0.25
    vec_one = np.array([1.0, 1.0, 1.0])
    box_pos_offset = (0.0, 0.0, 0.0) + 0.5 * box_size * vec_one
    box0 = scene.add_entity(
        gs.morphs.Box(
            size=box_size * vec_one,
            pos=box_pos_offset,
        ),
    )
    box1 = scene.add_entity(
        gs.morphs.Box(
            size=box_size * vec_one,
            pos=box_pos_offset + 0.8 * box_size * np.array([0, 0, 1]),
        ),
    )
    # A sphere pair overlapping off-axis, placed away from the boxes: it exercises the closed-form sphere-sphere
    # reconstruction (func_differentiable_sphere_contact) through the same loss and finite-difference sweep.
    sphere0 = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.2,
            pos=(2.0, 0.0, 0.2),
        ),
    )
    sphere1 = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.15,
            pos=(2.02, 0.01, 0.46),
        ),
    )
    scene.build()
    collider = scene.sim.rigid_solver.collider

    box1.set_quat(R_to_quat(gs.euler_to_R([np.deg2rad(3.0), np.deg2rad(3.0), np.deg2rad(3.0)])))
    box0_init_pos = box0.get_pos().clone()
    box1_init_pos = box1.get_pos().clone()
    box0_init_quat = box0.get_quat().clone()
    box1_init_quat = box1.get_quat().clone()
    sphere0_init_pos = sphere0.get_pos().clone()
    sphere1_init_pos = sphere1.get_pos().clone()

    collider.detection()
    contacts = collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)
    normal = contacts["normal"].requires_grad_()
    position = contacts["position"].requires_grad_()
    penetration = contacts["penetration"].requires_grad_()
    loss = ((normal * position).sum(dim=-1) * penetration).sum()
    dL_dnormal = torch.autograd.grad(loss, normal, retain_graph=True)[0]
    dL_dposition = torch.autograd.grad(loss, position, retain_graph=True)[0]
    dL_dpenetration = torch.autograd.grad(loss, penetration)[0]

    collider.backward(dL_dposition, dL_dnormal, dL_dpenetration)
    dL_dpos = qd_to_torch(scene.sim.rigid_solver.dyn_state.geoms.pos.grad)
    dL_dquat = qd_to_torch(scene.sim.rigid_solver.dyn_state.geoms.quat.grad)

    fd_eps = 1e-5
    trials = 100

    def directional_error(dL_dx, x_type):
        error_rel = 0.0
        for _ in range(trials):
            rand_dx = torch.nn.functional.normalize(torch.randn_like(dL_dx), dim=-1)
            dL = (rand_dx * dL_dx).sum()
            losses = []
            for sign in (1, -1):
                if x_type == "pos":
                    box0.set_pos(box0_init_pos + sign * rand_dx[0, 0] * fd_eps)
                    box1.set_pos(box1_init_pos + sign * rand_dx[1, 0] * fd_eps)
                    sphere0.set_pos(sphere0_init_pos + sign * rand_dx[2, 0] * fd_eps)
                    sphere1.set_pos(sphere1_init_pos + sign * rand_dx[3, 0] * fd_eps)
                    box0.set_quat(box0_init_quat)
                    box1.set_quat(box1_init_quat)
                else:
                    box0.set_pos(box0_init_pos)
                    box1.set_pos(box1_init_pos)
                    sphere0.set_pos(sphere0_init_pos)
                    sphere1.set_pos(sphere1_init_pos)
                    box0.set_quat(box0_init_quat + sign * rand_dx[0, 0] * fd_eps)
                    box1.set_quat(box1_init_quat + sign * rand_dx[1, 0] * fd_eps)
                collider.collider_state.n_contacts.fill(0)
                collider.detection()
                c = collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)
                losses.append(((c["normal"] * c["position"]).sum(dim=-1) * c["penetration"]).sum())
            dL_fd = (losses[0] - losses[1]) / (2 * fd_eps)
            error_rel += (dL - dL_fd).abs() / max(dL.abs(), dL_fd.abs(), gs.EPS)
        return error_rel / trials

    assert_allclose(directional_error(dL_dpos, "pos"), 0.0, atol=1e-4)
    assert_allclose(directional_error(dL_dquat, "quat"), 0.0, atol=1e-4)

    # Coincident sphere centers: the forward falls back to an arbitrary contact direction, and the backward must
    # reuse that fallback instead of dividing by the zero distance, which would emit NaN gradients.
    sphere0.set_pos(sphere0_init_pos)
    sphere1.set_pos(sphere0_init_pos)
    box0.set_pos(box0_init_pos)
    box1.set_pos(box1_init_pos)
    box0.set_quat(box0_init_quat)
    box1.set_quat(box1_init_quat)
    collider.collider_state.n_contacts.fill(0)
    collider.detection()
    contacts = collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)
    assert torch.isfinite(contacts["normal"]).all()
    assert torch.isfinite(contacts["position"]).all()
    normal = contacts["normal"].requires_grad_()
    position = contacts["position"].requires_grad_()
    penetration = contacts["penetration"].requires_grad_()
    loss = ((normal * position).sum(dim=-1) * penetration).sum()
    dL_dnormal = torch.autograd.grad(loss, normal, retain_graph=True)[0]
    dL_dposition = torch.autograd.grad(loss, position, retain_graph=True)[0]
    dL_dpenetration = torch.autograd.grad(loss, penetration)[0]
    collider.backward(dL_dposition, dL_dnormal, dL_dpenetration)
    assert torch.isfinite(qd_to_torch(scene.sim.rigid_solver.dyn_state.geoms.pos.grad)).all()


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.debug(False)
def test_constraint_solver_backward_matches_fd(monkeypatch):
    # Engine modules resolve gs dtypes at import time, which requires gs.init: import after initialization
    from genesis.engine.solvers.rigid.constraint.solver import func_solve_body, func_solve_init
    from genesis.engine.solvers.rigid.rigid_solver import kernel_step_1

    # fp64 is required: the parameter-side inputs (aref, efc_D, jac, mass) move the loss by far less than its own
    # fp32 resolution at any usable step size, so their finite-difference deltas drown at fp32

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(
            size=(1, 1, 1),
            pos=(10, 10, 0.49),
        ),
    )
    # A settled stack of three boxes forms one island whose dof list the solver reorders, and a pair stacked at first
    # then set apart leaves the factor of their former coupling in place.
    for i_box in range(3):
        scene.add_entity(
            gs.morphs.Box(
                size=(1, 1, 1),
                pos=(-10, 10, 0.5 + 1.0 * i_box),
            ),
        )
    scene.add_entity(
        gs.morphs.Box(
            size=(1, 1, 1),
            pos=(10, -10, 0.5),
        ),
    )
    top_box = scene.add_entity(
        gs.morphs.Box(
            size=(1, 1, 1),
            pos=(10, -10, 1.5),
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build()
    rigid_solver = scene._sim.rigid_solver
    constraint_solver = rigid_solver.constraint_solver

    for _ in range(25):
        scene.step()
    top_box.set_pos((10, -14, 0.49))
    franka.set_qpos([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])

    def constraint_solver_resolve():
        func_solve_init(
            rigid_solver.dyn_state,
            constraint_solver.constraint_state,
            rigid_solver.dyn_info,
            rigid_solver.rigid_info,
            rigid_solver.rigid_config,
            is_decomposed=False,
        )
        func_solve_body(
            rigid_solver.dyn_state,
            constraint_solver.constraint_state,
            rigid_solver.dyn_info,
            rigid_solver.rigid_info,
            rigid_solver.rigid_config,
            constraint_solver._n_iterations,
        )

    monkeypatch.setattr(constraint_solver, "resolve", constraint_solver_resolve)

    kernel_step_1(
        rigid_solver.dyn_state,
        constraint_solver.constraint_state,
        rigid_solver.dyn_info,
        rigid_solver.rigid_info,
        rigid_solver.rigid_config,
        is_forward_pos_updated=True,
        is_forward_vel_updated=True,
        is_backward=False,
    )
    constraint_solver.add_equality_constraints()
    rigid_solver.collider.detection()
    constraint_solver.add_inequality_constraints()
    constraint_solver.resolve()

    def compute_loss(input_mass, input_jac, input_aref, input_efc_D, input_force):
        rigid_solver.rigid_info.mass_mat.from_numpy(input_mass)
        constraint_solver.constraint_state.jac.from_numpy(input_jac)
        constraint_solver.constraint_state.aref.from_numpy(input_aref)
        constraint_solver.constraint_state.efc_D.from_numpy(input_efc_D)
        rigid_solver.dyn_state.dofs.force.from_numpy(input_force)
        updated_acc_smooth = np.linalg.solve(input_mass[..., 0], input_force[..., 0])
        rigid_solver.dyn_state.dofs.acc_smooth.from_numpy(updated_acc_smooth[..., None])
        constraint_solver.resolve()
        # Collapse to a Python scalar right away: the reduction reads a zero-copy view of qacc, and a deferred
        # evaluation (MPS is lazy) would otherwise run only after the next call overwrites the buffer in place.
        return float(((qd_to_torch(constraint_solver.qacc) - target_qacc) ** 2).mean())

    init_input_mass = qd_to_numpy(rigid_solver.rigid_info.mass_mat, copy=True)
    init_input_jac = qd_to_numpy(constraint_solver.constraint_state.jac, copy=True)
    init_input_aref = qd_to_numpy(constraint_solver.constraint_state.aref, copy=True)
    init_input_efc_D = qd_to_numpy(constraint_solver.constraint_state.efc_D, copy=True)
    init_input_force = qd_to_numpy(rigid_solver.dyn_state.dofs.force, copy=True)

    set_random_seed(0)
    init_output_qacc = qd_to_torch(constraint_solver.qacc)
    target_qacc = torch.from_numpy(np.random.randn(*init_output_qacc.shape)).to(device=gs.device)
    target_qacc = target_qacc * init_output_qacc.abs().mean()

    output_qacc = qd_to_torch(constraint_solver.qacc, copy=True).requires_grad_(True)
    loss = ((output_qacc - target_qacc) ** 2).mean()
    dL_dqacc = tensor_to_array(torch.autograd.grad(loss, output_qacc)[0])
    constraint_solver.constraint_state.dL_dqacc.from_numpy(dL_dqacc)
    constraint_solver.backward()

    dL_dM = qd_to_numpy(constraint_solver.constraint_state.dL_dM)
    dL_djac = qd_to_numpy(constraint_solver.constraint_state.dL_djac)
    dL_daref = qd_to_numpy(constraint_solver.constraint_state.dL_daref)
    dL_defc_D = qd_to_numpy(constraint_solver.constraint_state.dL_defc_D)
    dL_dforce = qd_to_numpy(constraint_solver.constraint_state.dL_dforce)

    fd_eps = 1e-3
    trials = 200
    for dL_dx, x_type in (
        (dL_dforce, "force"),
        (dL_daref, "aref"),
        (dL_defc_D, "efc_D"),
        (dL_djac, "jac"),
        (dL_dM, "mass"),
    ):
        error = 0.0
        for _ in range(trials):
            rand_dx = np.random.randn(*dL_dx.shape)
            rand_dx = rand_dx / max(
                np.linalg.norm(rand_dx, axis=0 if x_type in ("force", "aref", "efc_D") else (0, 1)), gs.EPS
            )
            if x_type == "mass":
                rand_dx = (rand_dx + np.moveaxis(rand_dx, 0, 1)) * 0.5
            dL = (rand_dx * dL_dx).sum()

            inputs = dict(
                input_mass=init_input_mass,
                input_jac=init_input_jac,
                input_aref=init_input_aref,
                input_efc_D=init_input_efc_D,
                input_force=init_input_force,
            )
            key = {
                "force": "input_force",
                "aref": "input_aref",
                "efc_D": "input_efc_D",
                "jac": "input_jac",
                "mass": "input_mass",
            }[x_type]
            init_x = inputs[key]
            loss_p = compute_loss(**{**inputs, key: init_x + rand_dx * fd_eps})
            loss_m = compute_loss(**{**inputs, key: init_x - rand_dx * fd_eps})
            dL_fd = (loss_p - loss_m) / (2 * fd_eps)
            error += abs(dL - dL_fd) / max(abs(dL), abs(dL_fd), gs.EPS)
        assert_allclose(error / trials, 0.0, atol=1e-4)
