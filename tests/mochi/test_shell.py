import numpy as np
import pytest
import quadrants as qd
import trimesh

import genesis as gs
from genesis.engine.solvers.mochi.shell import func_shell_elastic, func_shell_rest_data
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose
from .reference import load_reference, mochi_to_genesis


def _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -9.8), **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
        mochi_options=gs.options.MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


def _sheet_mesh(n_cells, size):
    """Square sheet in the x-y plane, consistently wound (normals along +z)."""
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            faces.append([a, a + 1, a + n_cells + 2])
            faces.append([a, a + n_cells + 2, a + n_cells + 1])
    return verts, np.array(faces)


def _write_obj(path, verts, faces):
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)
    return str(path)


@qd.kernel
def _kernel_shell_stencil(
    x_in: qd.types.ndarray(),
    X_in: qd.types.ndarray(),
    missing: qd.types.ndarray(),
    lam: float,
    mu: float,
    alpha: float,
    beta: float,
    energy: qd.types.ndarray(),
    res: qd.types.ndarray(),
    K: qd.types.ndarray(),
    project: qd.template(),
):
    x = qd.Matrix.zero(gs.qd_float, 6, 3)
    X = qd.Matrix.zero(gs.qd_float, 6, 3)
    for a in qd.static(range(6)):
        for i in qd.static(range(3)):
            x[a, i] = x_in[a, i]
            X[a, i] = X_in[a, i]
    is_missing = qd.Vector([missing[0] != 0, missing[1] != 0, missing[2] != 0])
    area, A_inv, B = func_shell_rest_data(X, is_missing, 1e-30)
    zero = qd.Matrix.zero(gs.qd_float, 2, 2)
    e, r, Km, _eps_m, _s = func_shell_elastic(
        x, X, is_missing, area, A_inv, B, lam, mu, alpha, beta, 0.0, zero, zero, 1e-30, project, True
    )
    energy[0] = e
    for p in qd.static(range(18)):
        res[p] = r[p]
        for q in qd.static(range(18)):
            K[p, q] = Km[p, q]


def _shell_stencil(x, X, missing, params, project):
    energy = np.zeros(1)
    res = np.zeros(18)
    K = np.zeros((18, 18))
    _kernel_shell_stencil(
        np.ascontiguousarray(x),
        np.ascontiguousarray(X),
        np.asarray(missing, dtype=np.int32),
        *params,
        energy,
        res,
        K,
        project,
    )
    return energy[0], res, K


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("missing", [(0, 0, 0), (1, 0, 0), (0, 1, 1)])
def test_shell_stencil_finite_difference(missing):
    rng = np.random.default_rng(3)
    params = (300.0, 400.0, 2e-3, 7e-3)
    X = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0.03, 0.09, 0], [0.13, 0.09, 0.02], [-0.06, 0.05, -0.01], [0.05, -0.08, 0.015]],
        dtype=float,
    )
    X[:3] += 0.005 * rng.standard_normal((3, 3))
    # Rest state: no energy, no residual, positive-semidefinite tangent.
    energy, res, K = _shell_stencil(X, X, missing, params, True)
    assert_allclose(energy, 0.0, atol=1e-14)
    assert_allclose(res, 0.0, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-9 * np.abs(K).max()
    # Generic deformation: the residual is the gradient of the energy, the tangent its derivative up to the dropped
    # geometric bending term.
    x = X + 0.01 * rng.standard_normal(X.shape)
    energy, res, K = _shell_stencil(x, X, missing, params, False)
    h = 1e-6
    res_fd = np.zeros(18)
    K_fd = np.zeros((18, 18))
    for p in range(18):
        dx = np.zeros(18)
        dx[p] = h
        e_plus, r_plus, _ = _shell_stencil(x + dx.reshape(6, 3), X, missing, params, False)
        e_minus, r_minus, _ = _shell_stencil(x - dx.reshape(6, 3), X, missing, params, False)
        res_fd[p] = (e_plus - e_minus) / (2.0 * h)
        K_fd[:, p] = (r_plus - r_minus) / (2.0 * h)
    assert_allclose(res, res_fd, atol=1e-8 * np.abs(res).max(), rtol=0.0)
    assert_allclose(K, K.T, atol=1e-12 * np.abs(K).max(), rtol=0.0)
    assert_allclose(K, K_fd, atol=5e-3 * np.abs(K).max(), rtol=0.0)
    for v in range(3):
        if missing[v]:
            assert_allclose(res[3 * (3 + v) : 3 * (4 + v)], 0.0, atol=0.0)
    # Compression: the projected membrane tangent is positive-semidefinite.
    x = X.copy()
    x[:3] = X[:3].mean(axis=0) + 0.9 * (X[:3] - X[:3].mean(axis=0))
    _, _, K_proj = _shell_stencil(x, X, missing, params, True)
    assert np.linalg.eigvalsh(K_proj).min() > -1e-9 * np.abs(K_proj).max()


@pytest.mark.required
@pytest.mark.precision("64")
def test_cloth_drop_matches_mochi(tmp_path, show_viewer):
    reference = load_reference("cloth_drop")
    dt = float(reference["dt"])
    verts = mochi_to_genesis(reference["rest_positions"])
    obj_path = _write_obj(tmp_path / "cloth.obj", verts, reference["faces"])
    scene = _mochi_scene(show_viewer, dt, n_newton_iterations=8, newton_abs_tol=1e-10, newton_rel_tol=1e-12)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid(friction=0.5))
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path),
        material=gs.materials.Mochi.Shell(E=1e4, nu=0.3, rho=200.0, thickness=1e-3, friction=0.5, collider_type="none"),
    )
    scene.build()
    assert cloth.is_shell
    assert cloth.n_vertices == 25
    assert cloth.n_elements == 32
    assert_allclose(cloth.get_vertices_position(), verts, tol=1e-12)
    assert_allclose(cloth.mass, 200.0 * 1e-3 * 0.16, tol=1e-12)

    positions = []
    for _ in range(len(reference["positions"])):
        scene.step()
        positions.append(tensor_to_array(cloth.get_vertices_position()))
    assert_allclose(positions, mochi_to_genesis(reference["positions"]), atol=1e-5, rtol=0.0)


@pytest.mark.precision("64")
def test_shell_free_fall(tmp_path, show_viewer):
    dt, g = 0.01, 9.8
    verts, faces = _sheet_mesh(3, 0.3)
    obj_path = _write_obj(tmp_path / "sheet.obj", verts, faces)
    scene = _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -g))
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 1.0), euler=(30.0, 0.0, 0.0)),
        material=gs.materials.Mochi.Shell(),
    )
    scene.build()
    rest = tensor_to_array(cloth.get_vertices_position())
    n_steps = 30
    for _ in range(n_steps):
        scene.step()
    pos = tensor_to_array(cloth.get_vertices_position())
    assert_allclose(pos[:, 2].mean() - rest[:, 2].mean(), -g * dt * dt * n_steps * (n_steps + 1) / 2, tol=1e-9)
    assert_allclose(tensor_to_array(cloth.get_vertices_velocity())[:, 2], -g * dt * n_steps, tol=1e-9)
    assert_allclose(pos - pos.mean(axis=0), rest - rest.mean(axis=0), tol=1e-10)


@pytest.mark.precision("64")
def test_shell_hanging_sheet(tmp_path, show_viewer):
    E, rho, t, size = 1e5, 1000.0, 2e-3, 0.5
    verts, faces = _sheet_mesh(10, size)
    obj_path = _write_obj(tmp_path / "sheet.obj", verts, faces)
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 1.0), euler=(90.0, 0.0, 0.0)),
        material=gs.materials.Mochi.Shell(E=E, nu=0.3, rho=rho, thickness=t, mass_damping=2.0),
    )
    scene.build()
    rest = tensor_to_array(cloth.get_vertices_position())
    top = np.flatnonzero(rest[:, 2] > rest[:, 2].max() - 1e-6)
    cloth.set_vertices_fixed(top)
    for _ in range(200):
        scene.step()
    pos = tensor_to_array(cloth.get_vertices_position())
    assert_allclose(pos[top], rest[top], tol=1e-12)
    assert_allclose(cloth.get_vertices_velocity(), 0.0, atol=1e-6)
    # Elongation of a sheet hanging under its own weight: rho_2d g L^2 / (2 E t) (membrane stiffness E t).
    elongation = rest[:, 2].min() - pos[:, 2].min()
    assert_allclose(elongation, 0.5 * rho * t * 9.8 * size * size / (E * t), rtol=0.1)
    # Released, it falls freely.
    cloth.set_vertices_fixed(top, is_fixed=False)
    for _ in range(10):
        scene.step()
    assert tensor_to_array(cloth.get_vertices_velocity())[:, 2].max() < -0.5


@pytest.mark.precision("64")
def test_rigid_ball_on_cloth(tmp_path, show_viewer):
    radius, collider_radius = 0.05, 0.02
    verts, faces = _sheet_mesh(10, 0.5)
    obj_path = _write_obj(tmp_path / "sheet.obj", verts, faces)
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 0.5)),
        material=gs.materials.Mochi.Shell(E=1e5, nu=0.3, rho=1000.0, thickness=2e-3, collider_radius=collider_radius),
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(radius=radius, pos=(0.0, 0.0, 0.7)), material=gs.materials.Mochi.Rigid(rho=500.0)
    )
    scene.build()
    # A single cloth without self-contact leaves the point-cloud collider with nothing to collide with: the spheres
    # only act against other point-cloud entities (mochi's rule), never against rigid bodies.
    assert not scene.mochi_solver._has_pc_colliders
    for _ in range(240):
        scene.step()
    # The ball rests on the cloth lying on the ground: its bottom sits within the contact band of the shell samples
    # colliding against its sphere collider.
    cloth_top = tensor_to_array(cloth.get_vertices_position())[:, 2].max()
    ball_z = float(tensor_to_array(ball.get_pos())[2])
    assert cloth_top + radius - 5e-3 < ball_z < cloth_top + radius + 2e-3
    # A residual slow roll on the (not perfectly symmetric) cloth remains.
    ball_vel = tensor_to_array(ball.get_dofs_velocity())
    assert_allclose(ball_vel[:3], 0.0, atol=1e-4)
    assert_allclose(ball_vel[3:], 0.0, atol=1e-2)
    ball_mass = float(tensor_to_array(ball.get_mass()))
    assert_allclose(tensor_to_array(ball.get_links_net_contact_force())[0], (0.0, 0.0, ball_mass * 9.8), atol=1e-3)
    assert_allclose(
        tensor_to_array(cloth.get_vertices_contact_force()).sum(axis=0), (0.0, 0.0, cloth.mass * 9.8), atol=1e-3
    )


@pytest.mark.precision("64")
def test_cloth_drapes_over_box(tmp_path, show_viewer):
    # A sheet draped over a fixed box settles on its faces and edges: the box's samples must not act on the cloth's
    # collider spheres (point-cloud colliders never collide with rigid bodies) or the two contact models fight,
    # the drape oscillates on the box and is ejected once the residual blows up.
    box_pos, box_half = np.array([0.0, 0.0, 0.15]), np.array([0.15, 0.15, 0.15])
    verts, faces = _sheet_mesh(12, 0.8)
    obj_path = _write_obj(tmp_path / "sheet.obj", verts, faces)
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=tuple(box_pos), fixed=True), material=gs.materials.Mochi.Rigid()
    )
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 0.6)),
        material=gs.materials.Mochi.Shell(
            E=2e4, nu=0.3, rho=200.0, thickness=2e-3, friction=0.6, collider_radius=0.02, penalty_coefficient=1e7
        ),
    )
    scene.build()
    for _ in range(300):
        scene.step()
    pos = tensor_to_array(cloth.get_vertices_position())
    # The drape rests: no oscillation of the cloth on the box.
    assert np.abs(tensor_to_array(cloth.get_vertices_velocity())).max() < 0.05
    # The lowest cloth vertex hangs down the sides without tunnelling under the box, and the center lies on the top
    # face: no vertex sits deeper than the contact ramp of the penalty.
    q = np.abs(pos - box_pos) - box_half
    sdf = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)
    assert sdf.min() > -2.5e-3
    # The center of the sheet ends on the top face: within the contact ramp below it, small wrinkles above it.
    center = np.linalg.norm(pos[:, :2] - box_pos[:2], axis=-1) < 0.1
    box_top = box_pos[2] + box_half[2]
    assert np.all(pos[center, 2] > box_top - 2.5e-3)
    assert np.all(pos[center, 2] < box_top + 1e-2)
