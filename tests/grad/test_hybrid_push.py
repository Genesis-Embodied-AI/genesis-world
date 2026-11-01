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
            dt=2e-3,
            substeps=10,
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
        # Heavy enough for the explicit coupling impulse of a high friction to stay small next to its momentum
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
def test_deformable_inputs_grad_matches_fd(precision, show_viewer):
    # The backward pass replays each step from the state it started from, the actuation holding over the steps that give
    # none. An MPM muscle and an explicit FEM muscle, far apart and uncoupled, each get their own inputs.
    scenes, boxes_by_scene = [], []
    for requires_grad in (True, False):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=4e-3,
                substeps=2,
                requires_grad=requires_grad,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.5, -0.5, -0.1),
                upper_bound=(0.5, 0.5, 0.5),
                grid_density=32,
            ),
            # The coupling of FEM and MPM carries no gradient (see 'LegacyCoupler.is_differentiable')
            coupler_options=gs.options.LegacyCouplerOptions(
                fem_mpm=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.6, -0.9, 0.5),
                camera_lookat=(0.0, 0.0, 0.15),
            ),
            show_viewer=show_viewer and requires_grad,
        )
        mpm_box = scene.add_entity(
            morph=gs.morphs.Box(
                lower=(-0.35, -0.1, 0.1),
                upper=(-0.15, 0.1, 0.2),
            ),
            material=gs.materials.MPM.Muscle(
                E=5e4,
                nu=0.3,
                rho=1000.0,
                sampler="regular",
                n_groups=2,
            ),
        )
        fem_box = scene.add_entity(
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
        scene.build(n_envs=2)
        # One muscle group on each side of each box
        mpm_muscle_group = (mpm_box.init_particles[:, 0] > -0.25).astype(gs.np_int)
        mpm_box.set_muscle(muscle_group=mpm_muscle_group)
        elems_centroid = tensor_to_array(fem_box.init_positions)[fem_box.elems].mean(axis=1)
        fem_box.set_muscle(muscle_group=(elems_centroid[:, 0] > 0.3).astype(gs.np_int))
        scenes.append(scene)
        boxes_by_scene.append((mpm_box, fem_box))

    def rollout(scene, boxes, actus, vels):
        scene.reset()
        for box, vel, actu in zip(boxes, vels, actus):
            box.set_velocity(vel)
            box.set_actuation(actu[0])
        scene.step()
        for box, actu in zip(boxes, actus):
            box.set_actuation(actu[1])
        scene.step()
        scene.step()
        for box, actu in zip(boxes, actus):
            box.set_actuation(actu[2])
        scene.step()
        # The loss of each box, which alone depends on its inputs since the boxes are uncoupled
        goals = ((-0.15, -0.05, 0.0), (0.4, -0.05, 0.0))
        return tuple(
            ((box.get_state().pos - torch.tensor(goal, dtype=gs.tc_float, device=gs.device)) ** 2).sum()
            for box, goal in zip(boxes, goals)
        )

    # Actuations of each box (MPM, FEM) at steps 0, 1 and 3, then their initial velocities
    actus = np.array([[[0.5, 0.2], [0.1, 0.4], [0.3, 0.6]], [[0.4, 0.1], [0.2, 0.5], [0.6, 0.3]]])
    vels = np.array([[0.1, -0.2, 0.3], [-0.2, 0.1, 0.2]])
    # Every substep of a step reads the actuation set last
    for scene, boxes in zip(scenes, boxes_by_scene):
        rollout(scene, boxes, actus, vels)
        assert_allclose(boxes[0].get_particles_actu(), actus[0, 2][mpm_muscle_group], tol=gs.EPS)

    # A particle given several times takes the value of its last occurrence, which alone gets a gradient, whether the
    # particles are given once for every environment or row by row, and whether they count from the start or the end
    n_particles = boxes_by_scene[0][0].n_particles
    for particles_idx_local in ([0, 0], [[0, 0], [1, 1]], [n_particles - 1, -1]):
        scenes[0].reset()
        vels_repeated = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]] * 2, dtype=gs.tc_float, device=gs.device, requires_grad=True
        )
        boxes_by_scene[0][0].set_particles_vel(vels_repeated, particles_idx_local=particles_idx_local)
        vel = boxes_by_scene[0][0].get_state().vel[[0, 1], np.array(particles_idx_local).reshape(-1, 2)[:, 0]]
        assert_allclose(vel, vels_repeated[:, 1], tol=gs.EPS)
        (vels_grad,) = torch.autograd.grad(vel.sum(), vels_repeated)
        assert_allclose(vels_grad[:, 0], 0.0, tol=gs.EPS)
        assert_allclose(vels_grad[:, 1], 1.0, tol=gs.EPS)

    actus_ana = torch.tensor(actus, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    vels_ana = torch.tensor(vels, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    sum(rollout(scenes[0], boxes_by_scene[0], actus_ana, vels_ana)).backward()

    # fp32 needs a large step to clear the state-noise floor. fp64 needs a small step to bound truncation error. The
    # finite difference of the inputs of a box reads the loss of that box alone, clear of the rounding of the other.
    eps = 1e-5 if precision == "64" else 1e-2
    inputs = (actus, vels)
    inputs_fd_grad = (np.zeros_like(actus), np.zeros_like(vels))
    for i_input, (value, fd_grad) in enumerate(zip(inputs, inputs_fd_grad)):
        for idx in np.ndindex(value.shape):
            losses = []
            for sign in (+1, -1):
                perturbed_inputs = list(inputs)
                perturbed_inputs[i_input] = value.copy()
                perturbed_inputs[i_input][idx] += sign * eps
                losses.append(float(rollout(scenes[1], boxes_by_scene[1], *perturbed_inputs)[idx[0]]))
            fd_grad[idx] = (losses[0] - losses[1]) / (2.0 * eps)
        assert (np.abs(fd_grad) > 1e-3).all()
    tol = 2e-9 if precision == "64" else 2e-4
    assert_allclose(actus_ana.grad, inputs_fd_grad[0], rtol=tol, atol=tol)
    assert_allclose(vels_ana.grad, inputs_fd_grad[1], rtol=tol, atol=tol)


@pytest.mark.required
def test_coupling_wrench_grad_matches_fd(precision, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.5),
            upper_bound=(0.5, 0.5, 0.5),
            grid_density=16,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.6, 0.4),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-0.2, -0.05, -0.05),
            upper=(-0.05, 0.05, 0.05),
        ),
        material=gs.materials.MPM.Elastic(
            E=5e4,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.01, 0.0),
            size=(0.08, 0.08, 0.08),
        ),
    )
    scene.build(n_envs=2)
    # The trailing end of the block follows the box through springs, so the gradient also flows from the box pose to it
    block.set_particle_constraints(
        block.get_particles_in_bbox((-0.2, -0.05, -0.05), (-0.15, 0.05, 0.05)), box.links[0].idx, stiffness=10.0
    )

    def rollout(vel):
        scene.reset()
        block.set_velocity((0.5, 0.0, 0.0))
        box.set_dofs_velocity(vel)
        for _ in range(20):
            scene.step()
        return (box.get_state().pos ** 2).sum() + (box.get_state().quat[..., 1:] ** 2).sum()

    tol = 2e-10 if precision == "64" else 1e-4
    vel = np.tile([0.1, 0.2, -0.1, 0.5, -0.3, 0.4], (2, 1))
    vel_ana = torch.tensor(vel, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    loss = rollout(vel_ana)
    # The block pushes the box, which would otherwise keep its velocity
    assert (box.get_dofs_velocity()[..., 0] > vel_ana[..., 0] + 5e-3).all()
    loss.backward()
    # The finite differences run on the same scene without gradients, which simulates the same physics. A rollout after
    # a reset starts free of the coupling wrench the previous one left.
    eps = 1e-6 if precision == "64" else 1e-2
    fd_grad = np.zeros_like(vel)
    with torch.no_grad():
        for i, j in np.ndindex(vel.shape):
            losses = []
            for sign in (+1, -1):
                perturbed = vel.copy()
                perturbed[i, j] += sign * eps
                losses.append(float(rollout(perturbed)))
            fd_grad[i, j] = (losses[0] - losses[1]) / (2.0 * eps)
    (grad_rerun,) = torch.autograd.grad(rollout(vel_ana), vel_ana)
    assert_allclose(grad_rerun, vel_ana.grad, tol=tol)
    assert_allclose(vel_ana.grad, fd_grad, rtol=tol, atol=tol)
