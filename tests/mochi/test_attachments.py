import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mochi.linear_solver import kernel_condense_dense
from genesis.engine.solvers.mochi.mochi_solver import _kernel_activate_all_envs
from genesis.engine.solvers.mochi.soft import kernel_soft_condense_dense
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose


def _hang_scene(show_viewer, maxvolume, **mochi_kwargs):
    """A rigid box hanging in mid-air from the bottom vertices of a soft cube whose top vertices are held: the box is
    supported only through the attachments, so its equilibrium checks both coupling directions."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8, **mochi_kwargs),
        show_viewer=show_viewer,
    )
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5), maxvolume=maxvolume, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.4, rho=1000.0),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.35)),
        material=gs.materials.Mochi.Rigid(rho=1000.0),
    )
    verts = cube.init_positions
    top = np.where(verts[:, 2] > 0.55 - 1e-4)[0]
    bottom = np.where(verts[:, 2] < 0.45 + 1e-4)[0]
    cube.attach_to_link(box.base_link, bottom, stiffness=1e6, damping=1.0)
    return scene, cube, box, top


def _assert_hangs(scene, cube, box, top):
    cube.set_vertices_fixed(top)
    for _ in range(240):
        scene.step()
    # The box weight flows through the attachments and the soft cube into the held vertices: at rest the box sags by
    # the elastic stretch of the cube (~0.3 mm) plus the attachment stretch (~1 um), nothing more.
    assert_allclose(tensor_to_array(box.get_pos()), (0.0, 0.0, 0.35), atol=2e-3)
    assert_allclose(tensor_to_array(box.get_vel()), 0.0, atol=1e-3)
    assert (qd_to_numpy(scene.mochi_solver.island_state.n_islands) == 1).all()


@pytest.mark.precision("64")
def test_attachment_hang_dense_batched(show_viewer):
    scene, cube, box, top = _hang_scene(show_viewer, 2e-4)
    scene.build(n_envs=2)
    assert scene.mochi_solver.n_attachments == 4
    _assert_hangs(scene, cube, box, top)
    assert qd_to_numpy(scene.mochi_solver.island_state.uses_dense).all()


@pytest.mark.precision("64")
def test_attachment_hang_pcg(show_viewer):
    scene, cube, box, top = _hang_scene(show_viewer, 1.5e-5)
    scene.build()
    _assert_hangs(scene, cube, box, top)
    assert not qd_to_numpy(scene.mochi_solver.island_state.uses_dense).any()


@pytest.mark.precision("64")
def test_attachment_to_static_link(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        show_viewer=show_viewer,
    )
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5), maxvolume=2e-4, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.4, rho=1000.0),
    )
    plate = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.02), pos=(0.0, 0.0, 0.6), fixed=True),
        material=gs.materials.Mochi.Rigid(),
    )
    verts = cube.init_positions
    top = np.where(verts[:, 2] > 0.55 - 1e-4)[0]
    cube.attach_to_link(plate.base_link, top, stiffness=1e6)
    scene.build()
    scene.mochi_solver.enable_entity_contact(cube, plate, False)
    for _ in range(240):
        scene.step()
    # The cube hangs from the static plate through its attached top vertices instead of falling.
    pos = tensor_to_array(cube.get_vertices_position())
    assert_allclose(pos[top, 2], 0.55, atol=2e-3)
    assert pos[:, 2].min() > 0.44
    assert_allclose(tensor_to_array(cube.get_vertices_velocity()), 0.0, atol=1e-3)


@pytest.mark.precision("64")
def test_attachment_step_kernels_agree(show_viewer):
    positions = []
    for step_kernel in ("pipeline", "monolith"):
        scene, cube, box, top = _hang_scene(show_viewer, 2e-4, step_kernel=step_kernel)
        scene.build()
        cube.set_vertices_fixed(top)
        for _ in range(30):
            scene.step()
        positions.append(
            np.concatenate(
                [tensor_to_array(box.get_pos()).reshape(-1), tensor_to_array(cube.get_vertices_position()).reshape(-1)]
            )
        )
    assert_allclose(positions[0], positions[1], atol=1e-9)


@qd.kernel
def _kernel_shift_vertex(i_v: int, k: int, delta: float, soft_state: qd.template()):
    soft_state.verts_pos[i_v, 0][k] += delta


@pytest.mark.precision("64")
def test_attachment_hessian_finite_difference(show_viewer):
    # Finite differences over the vertex degrees of freedom check the attachment's vertex block and both coupling
    # blocks (the latter also through the symmetry assertion); the link block shares its form with the equalities. A
    # uniform stretch about a point above the cube keeps every tetrahedron in tension, where the projected elastic
    # tangent is exact.
    scene, cube, _box, top = _hang_scene(show_viewer, 2e-4, linear_solver="ldlt")
    scene.build()
    cube.set_vertices_fixed(top)
    solver = scene.mochi_solver
    scene.step()

    rng = np.random.default_rng(0)
    pos = tensor_to_array(cube.get_vertices_position())
    center = pos.mean(axis=0) + (0.0, 0.0, 0.05)
    target = center + 1.01 * (pos - center) + 1e-6 * rng.standard_normal(pos.shape)
    for i_v in range(cube.n_vertices):
        for k in range(3):
            _kernel_shift_vertex(i_v, k, float(target[i_v, k] - pos[i_v, k]), solver.soft_state)

    def assemble():
        _kernel_activate_all_envs(solver.mochi_state, solver.rigid_config)
        solver._assemble(assem_res=True, assem_dres=True, skip_ls_done=False)
        return qd_to_numpy(solver.mochi_state.res)[:, 0].copy()

    assemble()
    kernel_condense_dense(
        solver.dyn_state,
        solver.dyn_info,
        solver.mochi_info,
        solver.mochi_state,
        solver.contact_state,
        solver.island_state,
        solver.eq_info,
        solver.eq_state,
        solver.rigid_config,
        solver.mochi_config.has_equalities,
    )
    kernel_soft_condense_dense(
        solver.dyn_state,
        solver.dyn_info,
        solver.mochi_state,
        solver.soft_info,
        solver.soft_state,
        solver.island_state,
        solver.rigid_config,
    )
    H = qd_to_numpy(solver.mochi_state.H_dense)[0].copy()
    n_rigid_dofs = solver.n_dofs
    eps = 1e-6
    H_fd = np.zeros((H.shape[0], 3 * cube.n_vertices))
    for i_v in range(cube.n_vertices):
        for k in range(3):
            _kernel_shift_vertex(i_v, k, eps, solver.soft_state)
            res_plus = assemble()
            _kernel_shift_vertex(i_v, k, -2.0 * eps, solver.soft_state)
            res_minus = assemble()
            _kernel_shift_vertex(i_v, k, eps, solver.soft_state)
            H_fd[:, 3 * i_v + k] = (res_plus - res_minus) / (2.0 * eps)
    # Fixed vertices carry Dirichlet rows and columns in the dense system but a raw residual: compare free rows and
    # free vertex columns only.
    free = np.ones(H.shape[0], dtype=bool)
    free[(n_rigid_dofs + 3 * np.asarray(top)[:, None] + np.arange(3)[None, :]).reshape(-1)] = False
    free_vert_dofs = np.where(free[n_rigid_dofs:])[0]
    scale = np.abs(H).max()
    assert_allclose(H, H.T, atol=1e-9 * scale, rtol=0.0)
    assert_allclose(
        H[np.ix_(free, n_rigid_dofs + free_vert_dofs)], H_fd[np.ix_(free, free_vert_dofs)], atol=1e-6 * scale, rtol=0.0
    )
    assert np.linalg.eigvalsh(0.5 * (H + H.T)[np.ix_(free, free)]).min() > 0.0
