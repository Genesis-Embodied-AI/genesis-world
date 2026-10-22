import numpy as np
import pytest
import torch

import genesis as gs

from ..utils.assertions import assert_allclose


@pytest.mark.slow  # ~350s
@pytest.mark.required
@pytest.mark.debug(False)
def test_mpm_tool_push_grad(show_viewer):
    HORIZON = 10

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
        options=gs.options.ToolEntityOptions(
            morph=gs.morphs.Mesh(
                file="meshes/stirrer.obj",
                scale=0.6,
                pos=(0.5, 0.5, 0.05),
                euler=(90.0, 0.0, 0.0),
            ),
            material=gs.materials.Tool(
                friction=8.0,
            ),
        ),
    )
    obj = scene.add_entity(
        options=gs.options.MPMEntityOptions(
            morph=gs.morphs.Box(
                lower=(0.2, 0.1, 0.05),
                upper=(0.4, 0.3, 0.15),
            ),
            material=gs.materials.MPM.Elastic(
                rho=500,
            ),
        ),
    )
    scene.build(n_envs=2)

    stick.set_position(
        torch.tensor([[0.3, 0.1, 0.28], [0.3, 0.1, 0.5]], dtype=gs.tc_float, device=gs.device, requires_grad=True)
    )
    obj.set_position(torch.tensor([0.3, 0.3, 0.1], dtype=gs.tc_float, device=gs.device, requires_grad=True))
    obj.set_velocity(torch.tensor([0.0, -1.0, 0.0], dtype=gs.tc_float, device=gs.device, requires_grad=True))
    goal = torch.tensor([0.5, 0.8, 0.05], dtype=gs.tc_float, device=gs.device)

    loss = 0.0
    v_list = []
    for i in range(HORIZON):
        v_i = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=gs.tc_float, device=gs.device, requires_grad=True)
        stick.set_velocity(vel=v_i)
        v_list.append(v_i)
        scene.step()
        if i == HORIZON // 2:
            mpm_particles = scene.get_state().solvers_state[scene.solvers.index(scene.mpm_solver)]
            loss += torch.pow(mpm_particles.pos[mpm_particles.active == 1] - goal, 2).sum()
        if i == HORIZON - 2:
            loss += torch.pow(obj.get_state().pos - goal, 2).sum()
    loss.backward()

    # Every step but the last must move the object (non-zero velocity gradient). The last step comes after every state
    # the loss reads, so the loss does not depend on its input, which gets no gradient, as in torch.
    for v_i in v_list[:-1]:
        assert (v_i.grad.abs() > gs.EPS).any()
    assert v_list[-1].grad is None


@pytest.mark.required
def test_muscle_actuation_grad_matches_fd(precision, show_viewer):
    # With checkpoint windows of two steps, the backward pass replays the second and third actuations within a window,
    # and restores the second from a checkpoint at the start of the next window, over which it holds.
    scenes, boxes = [], []
    for requires_grad in (True, False):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=4e-3,
                substeps=2,
                substeps_local=4,
                requires_grad=requires_grad,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.5, -0.5, -0.1),
                upper_bound=(0.5, 0.5, 0.5),
                grid_density=32,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.6, -0.6, 0.4),
                camera_lookat=(0.0, 0.0, 0.15),
            ),
            show_viewer=show_viewer and requires_grad,
        )
        box = scene.add_entity(
            options=gs.options.MPMEntityOptions(
                morph=gs.morphs.Box(
                    lower=(-0.1, -0.1, 0.1),
                    upper=(0.1, 0.1, 0.2),
                ),
                material=gs.materials.MPM.Muscle(
                    E=5e4,
                    nu=0.3,
                    rho=1000.0,
                    sampler="regular",
                    n_groups=2,
                ),
            ),
        )
        scene.build(n_envs=2)
        # One muscle group on each side of the box
        muscle_group = (box.init_particles[:, 0] > 0.0).astype(np.int32)
        box.set_muscle(muscle_group=muscle_group)
        scenes.append(scene)
        boxes.append(box)

    def rollout(scene, box, actus):
        scene.reset()
        box.set_actuation(actus[0])
        scene.step()
        box.set_actuation(actus[1])
        scene.step()
        scene.step()
        box.set_actuation(actus[2])
        scene.step()
        return (box.get_state().pos[..., 2] ** 2).sum()

    actus = np.array([[0.5, 0.2], [0.1, 0.4], [0.3, 0.6]])
    # Every substep of a step reads the actuation set last
    for scene, box in zip(scenes, boxes):
        rollout(scene, box, actus)
        assert_allclose(box.get_particles_actu(), actus[2][muscle_group], tol=gs.EPS)

    actus_ana = [torch.tensor(actu, dtype=gs.tc_float, device=gs.device, requires_grad=True) for actu in actus]
    rollout(scenes[0], boxes[0], actus_ana).backward()

    # fp32 needs a large step to clear the state-noise floor. fp64 needs a small step to bound truncation error.
    eps = 1e-5 if precision == "64" else 1e-2
    fd_grad = np.zeros_like(actus)
    for i, j in np.ndindex(actus.shape):
        losses = []
        for sign in (+1, -1):
            perturbed = actus.copy()
            perturbed[i, j] += sign * eps
            losses.append(float(rollout(scenes[1], boxes[1], perturbed)))
        fd_grad[i, j] = (losses[0] - losses[1]) / (2.0 * eps)
    assert (np.abs(fd_grad) > 1e-3).all()
    tol = 2e-9 if precision == "64" else 2e-4
    assert_allclose(torch.stack([actu.grad for actu in actus_ana]), fd_grad, rtol=tol, atol=tol)


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
        options=gs.options.MPMEntityOptions(
            morph=gs.morphs.Box(
                lower=(-0.2, -0.05, -0.05),
                upper=(-0.05, 0.05, 0.05),
            ),
            material=gs.materials.MPM.Elastic(
                E=5e4,
            ),
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.01, 0.0),
            size=(0.08, 0.08, 0.08),
        ),
    )
    scene.build(n_envs=2)

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
    eps = 1e-5 if precision == "64" else 1e-2
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
