from typing import NamedTuple

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose

# The motion of the implicit MPM is smooth in its inputs only while no particle crosses a face of its grid cell: the
# trilinear transfer between the particles and the nodes has a weight gradient that jumps across a face, and with it
# the deformation gradient and the energy of the particle. A finite difference across a crossing measures that jump
# rather than the derivative. The finite differences of the MPM thus take a perturbation small enough that no particle
# of the rollout changes cell, which every perturbed rollout checks against the unperturbed one (see '_grads_and_fd').
MPM_FD_EPS = 1e-6


def _step_mpm(scene, cells=None):
    if cells is not None:
        solver = scene.sim.mpm_solver
        pos = torch.cat([entity.get_particles_pos() for entity in solver.entities], dim=-2)
        cells.append(torch.floor(pos * solver.inv_dx))
    scene.step()


def _assert_same_cells(cells, cells_ref):
    for step, (cell, cell_ref) in enumerate(zip(cells, cells_ref)):
        n_moved = int((cell != cell_ref).any(dim=-1).sum())
        assert n_moved == 0, f"The perturbation moves {n_moved} MPM particles to another cell at step {step}"


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


class _FdCase(NamedTuple):
    params: np.ndarray
    i_loss: int
    eps: float
    # The entries whose finite difference is taken, every entry when None
    idx: tuple[tuple[int, ...], ...] | None = None


def _grads_and_fd(rollout, cases, is_mpm=True):
    # The backward pass runs on the sum of the losses, which each depend on the cases of their own body alone. A
    # rollout of the finite differences perturbs one entry of every loss at once, the bodies standing far apart, on the
    # same scene without gradients, which simulates the same physics, with no MPM particle crossing a cell face (see
    # MPM_FD_EPS).
    params_ana = [torch.tensor(case.params, dtype=gs.tc_float, device=gs.device, requires_grad=True) for case in cases]
    torch.atleast_1d(rollout(params_ana)).sum().backward()
    fd_grads = [np.zeros_like(case.params) for case in cases]
    entries_by_loss = [[] for _ in range(max(case.i_loss for case in cases) + 1)]
    for i_case, case in enumerate(cases):
        idx_all = np.ndindex(case.params.shape) if case.idx is None else case.idx
        entries_by_loss[case.i_loss].extend((i_case, idx) for idx in idx_all)
    with torch.no_grad():
        cells_ref = [] if is_mpm else None
        rollout([case.params for case in cases], cells_ref)
        for i_entry in range(max(len(entries) for entries in entries_by_loss)):
            entries = [entries[i_entry] for entries in entries_by_loss if i_entry < len(entries)]
            losses = []
            for sign in (+1, -1):
                params = [case.params.copy() for case in cases]
                for i_case, idx in entries:
                    params[i_case][idx] += sign * cases[i_case].eps
                cells = [] if is_mpm else None
                losses.append(torch.atleast_1d(rollout(params, cells)))
                if is_mpm:
                    _assert_same_cells(cells, cells_ref)
            for i_case, idx in entries:
                i_loss = cases[i_case].i_loss
                fd_grads[i_case][idx] = float(losses[0][i_loss] - losses[1][i_loss]) / (2.0 * cases[i_case].eps)
    return params_ana, fd_grads


@pytest.mark.required
# fp32 cannot resolve the Newton tolerance the finite differences need
@pytest.mark.precision("64")
def test_mpm_grad_matches_fd(precision, show_viewer):
    # The bodies stand far apart in one scene, each driving a loss of its own, so that one backward pass and one
    # finite difference per input cover them all. The force fields act on the whole domain, hence on every body: the
    # gradient of the initial velocity goes through their derivative with respect to the velocity (zero for the
    # constant field, the drag's |v| v and the flow of the point attractor otherwise). The muscle drives its particles
    # by its actuations. Each pair of blocks collides head-on and yields, so that the gradient of the initial velocity
    # of the first goes through the reverse of the return map of every substep after the impact, and through the
    # plastic volume ratio Jp for the snow and the sand.
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
        newton_options=gs.options.NewtonOptions(
            # The adjoint differentiates the converged motion, which the finite differences of the motion where the
            # Newton solve stops approach only once it has converged far below its default tolerance, the more so as their
            # small perturbation divides the error of the solve (see MPM_FD_EPS)
            velocity_tol=1e-12,
            # The default positive definite projection of the Hessian departs from the exact derivative wherever it
            # clamps a mode of the material, so the finite differences match the exact Hessian alone
            adjoint_hessian="exact",
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -3.5, 2.0),
            camera_lookat=(0.0, 0.0, 0.0),
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
    # Every block stands off the faces of the grid cells, where the rounding of a position would pick the cell (see
    # MPM_FD_EPS), and every shift of a body is a multiple of 0.25, whole cells
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

    def rollout(params, cells=None):
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
            _step_mpm(scene, cells)
        return torch.stack([((body.get_state().pos - goal) ** 2).sum() for body, goal in zip(bodies, goals)])

    actus = np.array([[0.5, 0.2], [0.1, 0.4], [0.3, 0.6]])
    vels_block = np.array([[0.3, -0.2, 0.4], [-0.5, 0.1, 0.2]])
    vels_blocks = np.array([[1.5, -0.2, 0.3], [1.2, 0.1, -0.2]])
    cases = [
        _FdCase(params=actus, i_loss=0, eps=MPM_FD_EPS),
        _FdCase(params=np.array([0.1, -0.2, 0.3]), i_loss=0, eps=MPM_FD_EPS),
        _FdCase(params=vels_block, i_loss=1, eps=MPM_FD_EPS),
        *(_FdCase(params=vels_blocks, i_loss=2 + i_material, eps=MPM_FD_EPS) for i_material in range(len(materials))),
    ]
    params_ana, fd_grads = _grads_and_fd(rollout, cases)
    for param_ana, fd_grad, tol in zip(params_ana, fd_grads, (5e-8, 5e-8, 2e-9, *(5e-9,) * len(materials))):
        assert (np.abs(fd_grad) > 1e-3).all()
        assert_allclose(param_ana.grad, fd_grad, tol=tol)
    # The particles hold the actuation set last
    rollout([case.params for case in cases])
    assert_allclose(muscle.get_particles_actu(), actus[2][muscle_group], tol=gs.EPS)


@pytest.mark.required
@pytest.mark.precision("64")
def test_fem_grad_matches_fd(precision, show_viewer):
    # The muscle drives its elements by its actuations, and the FEM block follows the rigid box, its near end held at
    # the box and its far end pulled toward it by springs, so the gradient also flows from the box pose to the block.
    # The muscle stands far from the block, each driving a loss of its own.
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
            camera_pos=(0.9, -0.7, 0.6),
            camera_lookat=(0.2, 0.0, 0.15),
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

    def rollout(params, cells=None):
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

    cases = [
        _FdCase(params=np.array([[0.4, 0.1], [0.2, 0.5], [0.6, 0.3]]), i_loss=0, eps=1e-5),
        _FdCase(params=np.array([-0.2, 0.1, 0.2]), i_loss=0, eps=1e-5),
        # The finite differences of the first environment, the second one running the same inputs apart
        _FdCase(
            params=np.tile([0.1, 0.2, -0.1, 0.5, -0.3, 0.4], (2, 1)),
            i_loss=1,
            eps=1e-6,
            idx=tuple((0, j) for j in range(6)),
        ),
    ]
    params_ana, fd_grads = _grads_and_fd(rollout, cases, is_mpm=False)
    for param_ana, fd_grad in zip(params_ana[:2], fd_grads[:2]):
        assert (np.abs(fd_grad) > 1e-3).all()
        assert_allclose(param_ana.grad, fd_grad, tol=2e-10)
    # The environments run the same inputs apart, so they get the same gradient
    assert_allclose(params_ana[2].grad[1], params_ana[2].grad[0], tol=1e-10)
    assert_allclose(params_ana[2].grad[0], fd_grads[2][0], tol=5e-10)


@pytest.mark.required
# fp32 cannot resolve the Newton tolerance the finite differences need
@pytest.mark.precision("64")
def test_hybrid_grad_matches_fd(precision, show_viewer):
    # The links of a rigid arm drive the MPM muscle around them after the grid-to-particle (G2P) transfer, so the
    # gradient goes from the soft particles to the dofs of the arm through that coupling and the implicit MPM solve
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
        newton_options=gs.options.NewtonOptions(
            # The adjoint differentiates the converged motion, which the finite differences of the motion where the
            # Newton solve stops approach only once it has converged far below its default tolerance, the more so as their
            # small perturbation divides the error of the solve (see MPM_FD_EPS)
            velocity_tol=1e-12,
            # The default positive definite projection of the Hessian departs from the exact derivative wherever it
            # clamps a mode of the material, so the finite differences match the exact Hessian alone
            adjoint_hessian="exact",
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.65, -0.35, 0.6),
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
            # The coupling scales the velocity the links impose on the particles by exp(-dt damping)
            damping=300.0,
        ),
    )
    scene.build()

    goal = torch.tensor((0.5, 0.5, 0.25), dtype=gs.tc_float, device=gs.device)

    def rollout(params, cells=None):
        (vel,) = params
        scene.reset()
        robot.set_dofs_velocity(vel)
        for _ in range(3):
            _step_mpm(scene, cells)
        return ((robot.part_soft.get_state().pos - goal) ** 2).sum()

    # The loss sums over every particle a gradient small beside it, so its rounding, which the perturbation divides,
    # takes an eps ten times as large as MPM_FD_EPS.
    cases = [_FdCase(params=np.array([1.5, -0.8]), i_loss=0, eps=10.0 * MPM_FD_EPS)]
    (vel_ana,), (fd_grad,) = _grads_and_fd(rollout, cases)
    assert (np.abs(fd_grad) > 1e-4).all()
    assert_allclose(vel_ana.grad, fd_grad, rtol=5e-6, atol=5e-9)


@pytest.mark.required
# fp32 cannot resolve the Newton tolerance the finite differences need
@pytest.mark.precision("64")
def test_coupling_wrench_grad_matches_fd(precision, show_viewer):
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
        newton_options=gs.options.NewtonOptions(
            # The adjoint differentiates the converged motion, which the finite differences of the motion where the
            # joint Newton solve of MPM and Rigid stops approach only once it has converged far below its default
            # tolerance, the more so as their small perturbation divides the error of the solve (see MPM_FD_EPS)
            velocity_tol=1e-12,
            # The default positive definite projection of the Hessian departs from the exact derivative wherever it
            # clamps a mode of the material, so the finite differences match the exact Hessian alone
            adjoint_hessian="exact",
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.6, 0.4),
            camera_lookat=(0.0, 0.0, 0.0),
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
    )
    scene.build(n_envs=2)
    # The trailing end of the block follows the box through springs, so the gradient also flows from the box pose to it
    block.set_particle_constraints(
        block.get_particles_in_bbox((-0.18, -0.05, -0.05), (-0.13, 0.05, 0.05)), box.links[0].idx, stiffness=10.0
    )

    def rollout(params, cells=None):
        (vel,) = params
        scene.reset()
        block.set_velocity((0.5, 0.0, 0.0))
        box.set_dofs_velocity(vel)
        for _ in range(8):
            _step_mpm(scene, cells)
        return (box.get_state().pos ** 2).sum() + (box.get_state().quat[..., 1:] ** 2).sum()

    # The finite differences of the first environment, the second one running the same inputs apart. A rollout after a
    # reset starts free of the coupling wrench the previous one left.
    cases = [
        _FdCase(
            params=np.tile([0.1, 0.2, -0.1, 0.5, -0.3, 0.4], (2, 1)),
            i_loss=0,
            eps=MPM_FD_EPS,
            idx=tuple((0, j) for j in range(6)),
        )
    ]
    (vel_ana,), (fd_grad,) = _grads_and_fd(rollout, cases)
    # The environments run the same inputs apart, so they get the same gradient
    assert_allclose(vel_ana.grad[1], vel_ana.grad[0], tol=1e-10)
    loss = rollout([vel_ana])
    # The block pushes the box, which would otherwise keep its velocity
    assert (box.get_dofs_velocity()[..., 0] > vel_ana[..., 0] + 5e-3).all()
    (grad_rerun,) = torch.autograd.grad(loss, vel_ana)
    assert_allclose(grad_rerun, vel_ana.grad, tol=1e-10)
    assert_allclose(vel_ana.grad[0], fd_grad[0], tol=5e-10)
