import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose


@pytest.mark.slow  # ~75s
@pytest.mark.required
@pytest.mark.debug(False)
def test_mpm_rigid_push_grad(show_viewer):
    HORIZON = 8

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 0.55),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -0.15, 2.42),
            camera_lookat=(0.5, 0.5, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
            fixed=True,
        )
    )
    stick = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            scale=0.6,
            pos=(0.5, 0.5, 0.05),
            euler=(90.0, 0.0, 0.0),
        ),
        # Heavy, so that the object it pushes barely changes its motion
        material=gs.materials.Rigid(
            rho=1e6,
            coup_friction=8.0,
            gravity_compensation=1.0,
        ),
    )
    obj = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(0.2, 0.1, 0.05),
            upper=(0.4, 0.3, 0.15),
        ),
        material=gs.materials.MPM.Elastic(
            rho=500,
            sampler="regular",
        ),
    )
    # A second stick, far from the object, whose position the loss reads: it moves by its velocity times the step
    stick_far = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            scale=0.6,
            pos=(0.8, -0.8, 0.4),
            euler=(90.0, 0.0, 0.0),
        ),
    )
    scene.build(n_envs=2)

    # The stick stands in the way of the object in the first env, and above it in the second
    stick.set_pos(
        torch.tensor([[0.3, 0.19, 0.1], [0.3, 0.19, 0.5]], dtype=gs.tc_float, device=gs.device, requires_grad=True)
    )
    obj.set_position(torch.tensor([0.3, 0.3, 0.1], dtype=gs.tc_float, device=gs.device, requires_grad=True))
    obj.set_velocity(torch.tensor([0.0, -1.0, 0.0], dtype=gs.tc_float, device=gs.device, requires_grad=True))
    goal = torch.tensor([0.5, 0.8, 0.05], dtype=gs.tc_float, device=gs.device)

    loss = 0.0
    v_list, v_far_list = [], []
    for i in range(HORIZON):
        v_i = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=gs.tc_float, device=gs.device, requires_grad=True)
        stick.set_dofs_velocity(v_i, dofs_idx_local=slice(0, 3))
        v_list.append(v_i)
        v_far_i = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=gs.tc_float, device=gs.device, requires_grad=True
        )
        stick_far.set_dofs_velocity(v_far_i, dofs_idx_local=slice(0, 3))
        v_far_list.append(v_far_i)
        scene.step()
        if i == HORIZON // 2:
            mpm_particles = scene.get_state().solvers_state[scene.solvers.index(scene.mpm_solver)]
            loss += torch.pow(mpm_particles.pos[mpm_particles.active == 1] - goal, 2).sum()
        if i == HORIZON - 2:
            loss += torch.pow(obj.get_state().pos - goal, 2).sum()
            loss += stick_far.get_state().pos.sum()
    loss.backward()

    # Every step but the last must move the object (non-zero velocity gradient). The last step comes after every state
    # the loss reads, so the loss does not depend on its input, which gets no gradient, as in torch.
    for v_i in v_list[:-1]:
        assert (v_i.grad.abs() > gs.EPS).any()
    assert v_list[-1].grad is None
    for v_far_i in v_far_list[:-1]:
        assert_allclose(v_far_i.grad, scene.dt, tol=gs.EPS)
    assert v_far_list[-1].grad is None


@pytest.mark.required
def test_mpm_grad_descends(optimize, show_viewer):
    # Far apart, every body drives a loss of its own through inputs of its own, one backward pass covering them all.
    # The gradients go through the velocity derivative of the force fields, the actuation of the muscle, and the
    # return maps of the blocks yielding in their head-on collisions.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=32,
            lower_bound=(-1.5, -1.25, -0.5),
            upper_bound=(1.5, 1.25, 0.5),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.8, 1.15),
            camera_lookat=(-0.15, -0.1, 0.05),
        ),
        show_viewer=show_viewer,
    )
    muscle = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-0.85, -0.1, 0.1),
            upper=(-0.65, 0.1, 0.2),
        ),
        material=gs.materials.MPM.Muscle(
            E=5e4,
            nu=0.3,
            rho=1000.0,
            sampler="regular",
            n_groups=2,
        ),
    )
    # Every block stands off the faces of the grid cells, where the rounding of a position would pick the cell, and
    # every shift of a body is a multiple of 0.25, whole cells
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.503, 0.002, 0.001),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(
            E=5e4,
            sampler="regular",
        ),
    )
    materials = (
        gs.materials.MPM.ElastoPlastic(
            E=5e4,
            sampler="regular",
            von_mises_yield_stress=500.0,
        ),
        gs.materials.MPM.ElastoPlastic(
            E=5e4,
            sampler="regular",
            use_von_mises=False,
        ),
        gs.materials.MPM.Snow(
            E=5e4,
            sampler="regular",
        ),
        gs.materials.MPM.Sand(
            E=5e4,
            sampler="regular",
        ),
        gs.materials.MPM.Liquid(
            E=5e4,
            sampler="regular",
            viscous=True,
        ),
    )
    blocks, targets = [], []
    for i_material, material in enumerate(materials):
        blocks.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(-0.052, 0.002 + 0.5 * (i_material - 2), 0.001),
                    size=(0.1, 0.1, 0.1),
                ),
                material=material,
            )
        )
        targets.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(0.058, 0.012 + 0.5 * (i_material - 2), 0.001),
                    size=(0.1, 0.1, 0.1),
                ),
                material=material,
            )
        )
    for force_field in (
        gs.force_fields.Constant(direction=(0.0, 0.0, 1.0), strength=2.0),
        gs.force_fields.Drag(linear=3.0, quadratic=5.0),
        gs.force_fields.Point(strength=1.0, position=(0.3, 0.2, 0.1), flow=2.0),
    ):
        scene.add_force_field(force_field).activate()
    scene.build(n_envs=2)
    # One muscle group on each side of the box
    muscle_group = (muscle.init_particles[:, 0] > -0.75).astype(gs.np_int)
    muscle.set_muscle(muscle_group=muscle_group)

    # A particle given several times takes the value of its last occurrence, which alone gets a gradient, whether the
    # particles are given once for every environment or row by row, and whether they count from the start or the end
    for particles_idx_local in ([0, 0], [[0, 0], [1, 1]], [muscle.n_particles - 1, -1]):
        scene.reset()
        vels_repeated = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]] * 2, dtype=gs.tc_float, device=gs.device, requires_grad=True
        )
        muscle.set_particles_vel(vels_repeated, particles_idx_local=particles_idx_local)
        vel = muscle.get_state().vel[[0, 1], np.array(particles_idx_local).reshape(-1, 2)[:, 0]]
        assert_allclose(vel, vels_repeated[:, 1], tol=gs.EPS)
        (vels_grad,) = torch.autograd.grad(vel.sum(), vels_repeated)
        assert_allclose(vels_grad[:, 0], 0.0, tol=gs.EPS)
        assert_allclose(vels_grad[:, 1], 1.0, tol=gs.EPS)

    goal_muscle = torch.tensor((-0.65, -0.05, 0.0), dtype=gs.tc_float, device=gs.device)
    goal_block = torch.tensor((0.6, -0.05, 0.02), dtype=gs.tc_float, device=gs.device)
    goals_blocks = [
        torch.tensor((0.1, -0.05 + 0.5 * (i_material - 2), 0.02), dtype=gs.tc_float, device=gs.device)
        for i_material in range(len(materials))
    ]
    bodies = (muscle, block, *blocks)
    goals = (goal_muscle, goal_block, *goals_blocks)

    def rollout(params):
        actus, vel_muscle, vel_block, *vels_blocks = params
        scene.reset()
        muscle.set_velocity(vel_muscle)
        block.set_velocity(vel_block)
        for body, vel in zip(blocks, vels_blocks):
            body.set_velocity(vel)
        for body in targets:
            body.set_velocity(torch.tensor(((-1.5, 0.0, 0.0),) * 2, dtype=gs.tc_float, device=gs.device))
        for actu in (actus[0], actus[1], None, actus[2]):
            if actu is not None:
                muscle.set_actuation(actu)
            scene.step()
        return torch.stack([((body.get_state().pos - goal) ** 2).sum() for body, goal in zip(bodies, goals)])

    actus = np.array([[0.5, 0.2], [0.1, 0.4], [0.3, 0.6]])
    vels_block = np.array([[0.3, -0.2, 0.4], [-0.5, 0.1, 0.2]])
    vels_blocks = np.array([[1.5, -0.2, 0.3], [1.2, 0.1, -0.2]])
    params, losses = optimize(
        rollout, [actus, np.array([0.1, -0.2, 0.3]), vels_block, *(vels_blocks,) * len(materials)], n_iters=20, lr=0.1
    )
    # Every body comes closer to its goal at every step of the optimization of its own inputs
    assert (losses[1:] < losses[:-1]).all()
    assert (losses[-1] < 0.75 * losses[0]).all()
    # The particles hold the actuation set last
    with torch.no_grad():
        rollout(params)
    assert_allclose(muscle.get_particles_actu(), tensor_to_array(params[0])[2][muscle_group], tol=gs.EPS)


@pytest.mark.required
@pytest.mark.precision("64")
def test_fem_grad_matches_fd(precision, show_viewer):
    # Far apart, the muscle drives its loss by its actuations and the FEM block its own through the rigid box, its near
    # end held at the box and its far end pulled toward it by springs.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            # The explicit FEM integration needs small substeps
            dt=4e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        # The collisions of the FEM solver carry no gradient (see 'LegacyCoupler.is_differentiable')
        coupler_options=gs.options.LegacyCouplerOptions(
            rigid_fem=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.65, -0.45, 0.45),
            camera_lookat=(0.17, 0.0, 0.15),
        ),
        show_viewer=show_viewer,
    )
    muscle = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.3, 0.0, 0.3),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.FEM.Muscle(
            E=5e4,
            nu=0.3,
            n_groups=2,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.01, 0.0),
            size=(0.08, 0.08, 0.08),
        ),
        vis_mode="collision",
    )
    fem_block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.15, 0.0, 0.0),
            size=(0.1, 0.05, 0.05),
        ),
        material=gs.materials.FEM.Elastic(
            E=5e4,
        ),
    )
    scene.build(n_envs=2)
    # One muscle group on each side of the box
    elems_centroid = tensor_to_array(muscle.init_positions)[muscle.elems].mean(axis=1)
    muscle.set_muscle(muscle_group=(elems_centroid[:, 0] > 0.3).astype(gs.np_int))
    fem_verts_x = tensor_to_array(fem_block.init_positions)[:, 0]
    fem_block.set_vertex_constraints(np.nonzero(fem_verts_x < 0.1 + gs.EPS)[0], link=box.links[0])
    fem_block.set_vertex_constraints(
        np.nonzero(fem_verts_x > 0.2 - gs.EPS)[0], link=box.links[0], is_soft_constraint=True, stiffness=100.0
    )

    goal = torch.tensor((0.4, -0.05, 0.0), dtype=gs.tc_float, device=gs.device)

    def rollout(params):
        actus, vel_muscle, vel_box = params
        scene.reset()
        muscle.set_velocity(vel_muscle)
        box.set_dofs_velocity(vel_box)
        for i_step in range(20):
            if i_step < 2:
                muscle.set_actuation(actus[i_step])
            elif i_step == 3:
                muscle.set_actuation(actus[2])
            scene.step()
            if i_step == 3:
                loss_muscle = ((muscle.get_state().pos - goal) ** 2).sum()
        return torch.stack([loss_muscle, (fem_block.get_state().pos ** 2).sum()])

    EPS_MUSCLE, EPS_BOX = 1e-5, 1e-6
    params = [
        np.array([[0.4, 0.1], [0.2, 0.5], [0.6, 0.3]]),
        np.array([-0.2, 0.1, 0.2]),
        np.tile([0.1, 0.2, -0.1, 0.5, -0.3, 0.4], (2, 1)),
    ]
    params_ana = [torch.tensor(param, dtype=gs.tc_float, device=gs.device, requires_grad=True) for param in params]
    rollout(params_ana).sum().backward()
    # A pair of rollouts perturbs an input of the muscle and one of the box at once, each driving its own loss. The box
    # takes the finite differences of the first environment, the second one running the same inputs apart.
    entries_muscle = [(i_param, idx) for i_param in (0, 1) for idx in np.ndindex(params[i_param].shape)]
    entries_box = [(2, (0, j)) for j in range(params[2].shape[1])]
    fd_grads = [np.zeros_like(param) for param in params]
    with torch.no_grad():
        for i_entry, entry_muscle in enumerate(entries_muscle):
            entries = [entry_muscle, *entries_box[i_entry : i_entry + 1]]
            losses = []
            for sign in (+1, -1):
                params_shifted = [param.copy() for param in params]
                for i_param, idx in entries:
                    params_shifted[i_param][idx] += sign * (EPS_BOX if i_param == 2 else EPS_MUSCLE)
                losses.append(rollout(params_shifted))
            for i_param, idx in entries:
                i_loss, eps = (1, EPS_BOX) if i_param == 2 else (0, EPS_MUSCLE)
                fd_grads[i_param][idx] = float(losses[0][i_loss] - losses[1][i_loss]) / (2.0 * eps)
    for param_ana, fd_grad in zip(params_ana[:2], fd_grads[:2]):
        assert (np.abs(fd_grad) > 1e-3).all()
        assert_allclose(param_ana.grad, fd_grad, tol=2e-10)
    # The environments run the same inputs apart, so they get the same gradient
    assert_allclose(params_ana[2].grad[1], params_ana[2].grad[0], tol=1e-10)
    assert_allclose(params_ana[2].grad[0], fd_grads[2][0], tol=5e-10)


@pytest.mark.required
def test_hybrid_grad_descends(optimize, show_viewer):
    # The links of the arm drive its MPM muscle, so the gradient goes from the soft particles to the dofs of the arm
    # through that coupling and the implicit MPM solve
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(0.0, 0.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.65, -0.15, 0.5),
            camera_lookat=(0.65, 0.45, 0.2),
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=(0.45, 0.45, 0.2),
            file="urdf/simple/two_link_arm.urdf",
            scale=0.2,
            fixed=True,
        ),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
            material_soft=gs.materials.MPM.Muscle(
                E=1e4,
                nu=0.45,
                sampler="regular",
            ),
            # The particles take the whole velocity of the links, rather than the default hundredth of it, so that the
            # arm moves the soft part
            soft_dv_coef=1.0,
        ),
    )
    scene.build()

    def centroid_after(vel):
        scene.reset()
        robot.set_dofs_velocity(vel)
        for _ in range(3):
            scene.step()
        return robot.part_soft.get_state().pos.mean(dim=-2)

    # The goal is where the arm takes the soft part at another velocity, so that it is reachable
    with torch.no_grad():
        goal = centroid_after(torch.tensor((0.5, 0.5), dtype=gs.tc_float, device=gs.device)).clone()
    _, losses = optimize(
        lambda params: ((centroid_after(params[0]) - goal) ** 2).sum(), [np.array([1.5, -0.8])], n_iters=4, lr=0.3
    )
    assert (losses[1:] < losses[:-1]).all()
    assert (losses[-1] < 0.25 * losses[0]).all()


@pytest.mark.required
def test_coupling_wrench_grad_descends(optimize, show_viewer):
    # The block alone moves the box, which starts at rest, by its contact and the springs of its trailing end, so that
    # the gradient of the pose of the box with respect to the velocity of the block goes through the coupling alone
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.5),
            upper_bound=(0.5, 0.5, 0.5),
            grid_density=16,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, -0.3, 0.2),
            camera_lookat=(-0.06, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-0.18, -0.05, -0.05),
            upper=(-0.03, 0.05, 0.05),
        ),
        material=gs.materials.MPM.Elastic(
            E=5e4,
            # Its front particles, half a spacing behind its face, start 1 cm from the box and hit it within the rollout
            sampler="regular",
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.01, 0.0),
            size=(0.08, 0.08, 0.08),
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=2)
    block.set_particle_constraints(
        block.get_particles_in_bbox((-0.18, -0.05, -0.05), (-0.13, 0.05, 0.05)), box.links[0].idx, stiffness=10.0
    )
    goal = torch.tensor((0.01, 0.01, 0.0), dtype=gs.tc_float, device=gs.device)

    def rollout(params):
        (vel,) = params
        scene.reset()
        block.set_velocity(vel)
        for _ in range(8):
            scene.step()
        return ((box.get_state().pos - goal) ** 2).sum(dim=-1) + (box.get_state().quat[..., 1:] ** 2).sum(dim=-1)

    vel_start = np.tile([0.5, 0.0, 0.0], (2, 1))
    vel = torch.tensor(vel_start, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    (grad,) = torch.autograd.grad(rollout([vel]).sum(), vel)
    # The environments run the same inputs apart, so they get the same gradient, and a rollout after a reset starts free
    # of the coupling wrench the previous one left
    (grad_rerun,) = torch.autograd.grad(rollout([vel]).sum(), vel)
    assert_allclose(grad[1], grad[0], tol=1e-6)
    assert_allclose(grad_rerun, grad, tol=1e-6)
    _, losses = optimize(rollout, [vel_start], n_iters=3, lr=0.05)
    assert (losses[1:] < losses[:-1]).all()
    assert (losses[-1] < 0.25 * losses[0]).all()
