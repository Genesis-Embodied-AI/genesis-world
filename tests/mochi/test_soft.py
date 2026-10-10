import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mochi.linear_solver import kernel_condense_dense
from genesis.engine.solvers.mochi.mochi_solver import _kernel_activate_all_envs
from genesis.engine.solvers.mochi.soft import kernel_soft_condense_dense
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose
from .reference import load_reference, mochi_to_genesis


def _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -9.8), **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
        mochi_options=gs.options.MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


def _write_tet_files(path_stem, verts, tets):
    node_path = f"{path_stem}.node"
    with open(node_path, "w") as f:
        f.write(f"{len(verts)} 3 0 0\n")
        f.writelines(
            f"{i_v} {float(vert[0])!r} {float(vert[1])!r} {float(vert[2])!r}\n" for i_v, vert in enumerate(verts)
        )
    with open(f"{path_stem}.ele", "w") as f:
        f.write(f"{len(tets)} 4 0\n")
        f.writelines(f"{i_t} {int(tet[0])} {int(tet[1])} {int(tet[2])} {int(tet[3])}\n" for i_t, tet in enumerate(tets))
    return node_path


@pytest.mark.required
@pytest.mark.precision("64")
def test_soft_cube_drop_matches_mochi(tmp_path, show_viewer):
    reference = load_reference("soft_cube_drop")
    dt = float(reference["dt"])
    verts = mochi_to_genesis(reference["rest_positions"])
    node_path = _write_tet_files(tmp_path / "soft_cube", verts, reference["tets"])
    scene = _mochi_scene(show_viewer, dt, n_newton_iterations=8, newton_abs_tol=1e-10, newton_rel_tol=1e-12)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid(friction=0.5))
    cube = scene.add_entity(
        gs.morphs.Mesh(file=node_path),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0, friction=0.5),
    )
    scene.build()
    assert cube.n_vertices == 27
    assert cube.n_elements == 48
    assert cube.n_surfaces == 48
    assert_allclose(cube.get_vertices_position(), verts, tol=1e-12)
    assert_allclose(cube.mass, 8.0, tol=1e-9)

    positions = []
    for _ in range(len(reference["positions"])):
        scene.step()
        positions.append(tensor_to_array(cube.get_vertices_position()))
    # The impact step stops the two Newton solves at slightly different iterates (a few 1e-6 m).
    assert_allclose(positions, mochi_to_genesis(reference["positions"]), atol=1e-5, rtol=0.0)
    # The finite-difference velocities are consistent with the positions.
    velocity = tensor_to_array(cube.get_vertices_velocity())
    assert_allclose(velocity, (positions[-1] - positions[-2]) / dt, atol=1e-9, rtol=0.0)


@pytest.mark.required
@pytest.mark.precision("64")
def test_soft_free_fall(show_viewer):
    dt, g = 0.01, 9.8
    scene = _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -g))
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 1.0), maxvolume=0.002),
        material=gs.materials.Mochi.Elastic(),
    )
    scene.build()
    rest = tensor_to_array(cube.get_vertices_position())
    n_steps = 30
    for _ in range(n_steps):
        scene.step()
    pos = tensor_to_array(cube.get_vertices_position())
    # Backward Euler free fall: x_n = x_0 - g h^2 n (n + 1) / 2, v_n = -g h n, and the shape is preserved.
    assert_allclose(pos[:, 2].mean() - rest[:, 2].mean(), -g * dt * dt * n_steps * (n_steps + 1) / 2, tol=1e-9)
    assert_allclose(tensor_to_array(cube.get_vertices_velocity())[:, 2], -g * dt * n_steps, tol=1e-9)
    assert_allclose(pos - pos.mean(axis=0), rest - rest.mean(axis=0), tol=1e-10)


@pytest.mark.precision("64")
def test_soft_cube_rests_on_plane(show_viewer):
    E, rho, size = 1e5, 1000.0, 0.2
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    cube = scene.add_entity(
        gs.morphs.Box(size=(size, size, size), pos=(0.0, 0.0, 0.5 * size + 0.02), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=E, nu=0.45, rho=rho),
    )
    scene.build()
    for _ in range(120):
        scene.step()
    pos = tensor_to_array(cube.get_vertices_position())
    assert_allclose(cube.get_vertices_velocity(), 0.0, atol=1e-6)
    assert int(scene.mochi_solver.get_convergence_info()["status"][0]) == 1
    # Net contact force balances the weight; penetration stays at the penalty compliance.
    assert_allclose(
        tensor_to_array(cube.get_vertices_contact_force()).sum(axis=0), (0.0, 0.0, cube.mass * 9.8), tol=1e-3
    )
    assert -2e-3 < pos[:, 2].min() < 0.0
    # Compression of a column under its own weight: rho g L / (2 E).
    height = pos[:, 2].max() - pos[:, 2].min()
    assert_allclose(size - height, rho * 9.8 * size * size / (2.0 * E), rtol=0.3)


@pytest.mark.precision("64")
def test_soft_fixed_vertices_hang(show_viewer):
    E, rho, size = 1e5, 1000.0, 0.2
    scene = _mochi_scene(show_viewer, 0.02, n_newton_iterations=8, linear_solver="pcg")
    bar = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, size), pos=(0.0, 0.0, 1.0), maxvolume=0.0005),
        material=gs.materials.Mochi.Elastic(E=E, nu=0.3, rho=rho, mass_damping=5.0),
    )
    scene.build()
    rest = tensor_to_array(bar.get_vertices_position())
    top = np.flatnonzero(rest[:, 2] > rest[:, 2].max() - 1e-6)
    bar.set_vertices_fixed(top)
    for _ in range(300):
        scene.step()
    pos = tensor_to_array(bar.get_vertices_position())
    assert_allclose(pos[top], rest[top], tol=1e-12)
    assert_allclose(bar.get_vertices_velocity(), 0.0, atol=1e-5)
    # Elongation of a bar hanging under its own weight: rho g L^2 / (2 E).
    elongation = rest[:, 2].min() - pos[:, 2].min()
    assert_allclose(elongation, rho * 9.8 * size * size / (2.0 * E), rtol=0.25)
    # Released, the bar falls freely.
    bar.set_vertices_fixed(top, is_fixed=False)
    for _ in range(10):
        scene.step()
    assert tensor_to_array(bar.get_vertices_velocity())[:, 2].max() < -0.5


@pytest.mark.precision("64")
def test_soft_rigid_stack_batched(show_viewer):
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    # Contact acts through the samples of the deformable surface, so the slab must be meshed finely enough for the box
    # footprint to cover several boundary triangles.
    slab = scene.add_entity(
        gs.morphs.Box(size=(0.6, 0.6, 0.1), pos=(0.0, 0.0, 0.05), maxvolume=0.0002, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=2e5, nu=0.4, rho=1000.0),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.16, 0.16, 0.1), pos=(0.0, 0.0, 0.2)),
        material=gs.materials.Mochi.Rigid(rho=2000.0),
    )
    scene.build(n_envs=2)
    assert not scene.mochi_solver.mochi_config.has_dense
    box.set_pos(np.array([[0.0, 0.0, 0.2], [0.1, 0.0, 0.25]]))
    for _ in range(150):
        scene.step()
    box_pos = tensor_to_array(box.get_pos())
    # The two boxes rest at different spots of the (non-uniform) mesh, hence the loose match.
    assert_allclose(box_pos[:, 2], box_pos[0, 2], tol=1e-3)
    assert_allclose(box_pos[1, 0], 0.1, tol=1e-3)
    # The box rests on the slab, which sags under it.
    assert 0.13 < box_pos[0, 2] < 0.15
    slab_force = tensor_to_array(slab.get_vertices_contact_force()).sum(axis=1)
    box_force = tensor_to_array(box.get_links_net_contact_force())[:, 0]
    box_mass = float(tensor_to_array(box.get_mass()))
    for i_b in range(2):
        assert_allclose(box_force[i_b], (0.0, 0.0, box_mass * 9.8), atol=1e-2, rtol=1e-5)
        assert_allclose(slab_force[i_b], (0.0, 0.0, slab.mass * 9.8), atol=1e-2, rtol=1e-5)
    assert_allclose(tensor_to_array(slab.get_vertices_velocity()), 0.0, atol=1e-4)


@qd.kernel
def _kernel_shift_vertex(i_v: int, k: int, delta: float, soft_state: qd.template()):
    soft_state.verts_pos[i_v, 0][k] += delta


@pytest.mark.precision("64")
def test_soft_hessian_finite_difference(show_viewer):
    scene = _mochi_scene(
        show_viewer,
        0.01,
        gravity=(0.0, 0.0, 0.0),
        n_newton_iterations=8,
        use_fitted_friction_hessian=False,
        linear_solver="ldlt",
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid(friction=0.6))
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.0995), maxvolume=0.002),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    scene.build()
    solver = scene.mochi_solver
    scene.step()
    # A uniform stretch about a point above the cube keeps every tetrahedron in tension, where the projected tangent is
    # the exact one, presses the bottom face into the plane and slides the samples (friction is active); a little
    # noise breaks the symmetry of the configuration.
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
    dof_start = solver.n_dofs
    H = qd_to_numpy(solver.mochi_state.H_dense)[0][dof_start:, dof_start:].copy()
    assert int(qd_to_numpy(solver.soft_state.n_soft_hits)[0]) > 0
    eps = 1e-6
    H_fd = np.zeros_like(H)
    for i_v in range(cube.n_vertices):
        for k in range(3):
            _kernel_shift_vertex(i_v, k, eps, solver.soft_state)
            res_plus = assemble()
            _kernel_shift_vertex(i_v, k, -2.0 * eps, solver.soft_state)
            res_minus = assemble()
            _kernel_shift_vertex(i_v, k, eps, solver.soft_state)
            H_fd[:, 3 * i_v + k] = (res_plus - res_minus)[dof_start:] / (2.0 * eps)
    assert_allclose(H, H.T, atol=1e-9 * np.abs(H).max(), rtol=0.0)
    assert_allclose(H, H_fd, atol=1e-6 * np.abs(H).max(), rtol=0.0)
    assert np.linalg.eigvalsh(0.5 * (H + H.T)).min() > 0.0


@pytest.mark.precision("64")
def test_soft_soft_stack(show_viewer):
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    bottom = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0.0, 0.0, 0.05), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=2e5, nu=0.4, rho=1000.0),
    )
    top = scene.add_entity(
        gs.morphs.Box(size=(0.12, 0.12, 0.12), pos=(0.0, 0.0, 0.2), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    scene.build()
    for _ in range(150):
        scene.step()
    bottom_pos = tensor_to_array(bottom.get_vertices_position())
    top_pos = tensor_to_array(top.get_vertices_position())
    # The top cube rests on the slab (the soft collider has no contact skin, so it sinks a few millimeters in).
    assert bottom_pos[:, 2].max() - 4e-3 < top_pos[:, 2].min() < bottom_pos[:, 2].max()
    assert_allclose(top.get_vertices_velocity(), 0.0, atol=1e-5)
    assert_allclose(
        tensor_to_array(top.get_vertices_contact_force()).sum(axis=0), (0.0, 0.0, top.mass * 9.8), atol=1e-2
    )
    assert_allclose(
        tensor_to_array(bottom.get_vertices_contact_force()).sum(axis=0), (0.0, 0.0, bottom.mass * 9.8), atol=1e-2
    )


@pytest.mark.precision("64")
def test_rigid_ball_on_coarse_soft_slab(show_viewer):
    radius = 0.03
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    # The slab is far coarser than the ball: contact is carried by the ball's samples against the slab's distance field.
    slab = scene.add_entity(
        gs.morphs.Box(size=(0.6, 0.6, 0.1), pos=(0.0, 0.0, 0.05), maxvolume=0.004),
        material=gs.materials.Mochi.Elastic(E=2e5, nu=0.4, rho=1000.0),
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(radius=radius, pos=(0.05, 0.02, 0.25)), material=gs.materials.Mochi.Rigid(rho=2000.0)
    )
    scene.build()
    assert slab.n_vertices < 20
    for _ in range(150):
        scene.step()
    slab_top = tensor_to_array(slab.get_vertices_position())[:, 2].max()
    ball_z = float(tensor_to_array(ball.get_pos())[2])
    assert slab_top + radius - 5e-3 < ball_z < slab_top + radius
    ball_mass = float(tensor_to_array(ball.get_mass()))
    assert_allclose(tensor_to_array(ball.get_links_net_contact_force())[0], (0.0, 0.0, ball_mass * 9.8), atol=1e-3)
    assert_allclose(
        tensor_to_array(slab.get_vertices_contact_force()).sum(axis=0), (0.0, 0.0, slab.mass * 9.8), atol=1e-2
    )
    assert_allclose(ball.get_dofs_velocity(), 0.0, atol=1e-6)


@pytest.mark.precision("64")
def test_soft_soft_hessian_finite_difference(show_viewer):
    scene = _mochi_scene(
        show_viewer,
        0.01,
        gravity=(0.0, 0.0, 0.0),
        n_newton_iterations=8,
        use_fitted_friction_hessian=False,
        linear_solver="ldlt",
    )
    bottom = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0.0, 0.0, 0.05), maxvolume=0.002),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    top = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.02, 0.01, 0.148), maxvolume=0.002),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    scene.build()
    solver = scene.mochi_solver
    scene.step()
    # Stretch both bodies about a common point so that every tetrahedron is in tension (exact projected tangent), the
    # top cube is pressed into the slab and the samples slide (friction is active).
    rng = np.random.default_rng(1)
    for entity in (bottom, top):
        pos = tensor_to_array(entity.get_vertices_position())
        center = np.array([0.0, 0.0, 0.1 if entity is bottom else 0.2])
        target = center + 1.01 * (pos - center) + 1e-6 * rng.standard_normal(pos.shape)
        for i_v in range(entity.n_vertices):
            for k in range(3):
                _kernel_shift_vertex(entity.v_start + i_v, k, float(target[i_v, k] - pos[i_v, k]), solver.soft_state)

    def assemble():
        _kernel_activate_all_envs(solver.mochi_state, solver.rigid_config)
        solver._assemble(assem_res=True, assem_dres=True, skip_ls_done=False)
        return qd_to_numpy(solver.mochi_state.res)[:, 0].copy()

    assemble()
    assert int(qd_to_numpy(solver.soft_state.n_sc_hits)[0]) > 0
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
    n_verts = solver.n_soft_verts
    H = qd_to_numpy(solver.mochi_state.H_dense)[0].copy()
    eps = 1e-6
    H_fd = np.zeros_like(H)
    for i_v in range(n_verts):
        for k in range(3):
            _kernel_shift_vertex(i_v, k, eps, solver.soft_state)
            res_plus = assemble()
            _kernel_shift_vertex(i_v, k, -2.0 * eps, solver.soft_state)
            res_minus = assemble()
            _kernel_shift_vertex(i_v, k, eps, solver.soft_state)
            H_fd[:, 3 * i_v + k] = (res_plus - res_minus) / (2.0 * eps)
    assert_allclose(H, H.T, atol=1e-9 * np.abs(H).max(), rtol=0.0)
    # The deformable-collider tangent drops the derivatives of the barycentric weights, of the pulled-back gradient
    # direction and the curvature of the distance field (as mochi does), so it only approximates the exact Hessian; a
    # sign error in any coupling block would show up as a mismatch of the order of the contact stiffness.
    assert_allclose(H, H_fd, atol=2e-2 * np.abs(H).max(), rtol=0.0)


@pytest.mark.precision("64")
@pytest.mark.parametrize("integrator", ["backward_euler", "bdf2"])
def test_soft_moving_dirichlet(show_viewer, integrator):
    dt, v_drive = 0.01, np.array([0.2, 0.0, 0.1])
    scene = _mochi_scene(
        show_viewer,
        dt,
        gravity=(0.0, 0.0, 0.0),
        n_newton_iterations=8,
        integrator=integrator,
        newton_abs_tol=1e-9,
        newton_rel_tol=1e-12,
    )
    bar = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.3), pos=(0.0, 0.0, 1.0), maxvolume=0.001),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.3, rho=1000.0, mass_damping=2.0),
    )
    scene.build()
    rest = tensor_to_array(bar.get_vertices_position())
    top = np.flatnonzero(rest[:, 2] > rest[:, 2].max() - 1e-6)
    n_steps = 60
    for i_step in range(n_steps):
        target = rest[top] + (i_step + 1) * dt * v_drive
        bar.set_vertices_target(top, target)
        scene.step()
        pos = tensor_to_array(bar.get_vertices_position())
        vel = tensor_to_array(bar.get_vertices_velocity())
        # The driven vertices reach their prescribed positions exactly, and their finite-difference velocity is the
        # drive velocity under both integrators (the BDF2 extrapolation is consistent with a constant velocity).
        assert_allclose(pos[top], target, tol=1e-12)
        assert_allclose(vel[top], v_drive, tol=1e-9)
    # The free part of the bar is dragged along with the driven face.
    assert np.all(np.linalg.norm(vel - v_drive, axis=1) < 0.5 * np.linalg.norm(v_drive))
    assert_allclose(pos.mean(axis=0) - rest.mean(axis=0), n_steps * dt * v_drive, rtol=0.2, atol=1e-3)
    # Released, the vertices are free again and keep moving with the body.
    bar.set_vertices_fixed(top, is_fixed=False)
    scene.step()
    pos_next = tensor_to_array(bar.get_vertices_position())
    assert np.linalg.norm(pos_next[top] - pos[top], axis=1).max() > 0.5 * dt * np.linalg.norm(v_drive)
