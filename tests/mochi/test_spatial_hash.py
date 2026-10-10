import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import qd_to_numpy


def _mochi_scene(show_viewer, dt, **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


def _sheet_obj(path, n_cells, size):
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            faces.append([a, a + 1, a + n_cells + 2])
            faces.append([a, a + n_cells + 2, a + n_cells + 1])
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return path


def _point_cloud_hits(solver):
    """Per environment, the set of (colliding kind, sample, collider vertex) of the point-cloud contacts."""
    soft_state = solver.soft_state
    solver._is_contacts_recorded = False
    solver._record_contacts()
    n_hits = qd_to_numpy(soft_state.n_pc_hits)
    kind = qd_to_numpy(soft_state.pc_hit_kind_a)
    sample = qd_to_numpy(soft_state.pc_hit_sample_a)
    vert = qd_to_numpy(soft_state.pc_hit_vert_b)
    return [
        {(int(kind[i, i_b]), int(sample[i, i_b]), int(vert[i, i_b])) for i in range(int(n_hits[i_b]))}
        for i_b in range(len(n_hits))
    ]


def _tetrahedron_hits(solver):
    """Per environment, the set of (colliding kind, sample, collider tetrahedron) of the tetrahedral contacts."""
    soft_state = solver.soft_state
    solver._is_contacts_recorded = False
    solver._record_contacts()
    n_hits = qd_to_numpy(soft_state.n_sc_hits)
    kind = qd_to_numpy(soft_state.sc_hit_kind_a)
    sample = qd_to_numpy(soft_state.sc_hit_sample_a)
    elem = qd_to_numpy(soft_state.sc_hit_elem_b)
    return [
        {(int(kind[i, i_b]), int(sample[i, i_b]), int(elem[i, i_b])) for i in range(int(n_hits[i_b]))}
        for i_b in range(len(n_hits))
    ]


@pytest.mark.precision("64")
def test_point_cloud_hash_matches_brute_force(tmp_path, show_viewer):
    # Two sheets stacked within their contact range under a ball: sheet-sheet and self contacts (the ball
    # presses the sheets through their samples against its sphere collider; rigid bodies never hit the collider
    # spheres).
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=4)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    material = gs.materials.Mochi.Shell(
        E=2e4, nu=0.3, rho=200.0, thickness=1e-3, collider_radius=0.01, penalty_coefficient=1e7, self_contact=True
    )
    path = _sheet_obj(str(tmp_path / "sheet.obj"), 12, 0.4)
    lower = scene.add_entity(gs.morphs.Mesh(file=path, pos=(0.0, 0.0, 0.012)), material=material)
    upper = scene.add_entity(gs.morphs.Mesh(file=path, pos=(0.03, 0.02, 0.04)), material=material)
    ball = scene.add_entity(gs.morphs.Sphere(radius=0.05, pos=(0.0, 0.0, 0.15)), material=gs.materials.Mochi.Rigid())
    scene.build(n_envs=2)
    ball.set_pos(np.array([[0.0, 0.0, 0.15], [0.08, -0.05, 0.12]]))
    for _ in range(25):
        scene.step()
    solver = scene.mochi_solver
    assert solver._has_pc_colliders

    hits_hash = _point_cloud_hits(solver)
    # A cell larger than the scene puts every collider vertex in the 27 cells of every sample: brute force with the
    # same evaluation.
    cell = float(qd_to_numpy(solver.soft_info.pc_hash_cell))
    solver.soft_info.pc_hash_cell.fill(1e3)
    hits_brute = _point_cloud_hits(solver)
    solver.soft_info.pc_hash_cell.fill(cell)

    for i_b in range(2):
        assert len(hits_brute[i_b]) > 100
        assert hits_hash[i_b] == hits_brute[i_b]
    assert hits_hash[0] != hits_hash[1]
    # Every point-cloud contact has a deformable sample on the colliding side.
    assert {kind for kind, _, _ in hits_hash[0]} == {1}
    del lower, upper


@pytest.mark.precision("64")
def test_tetrahedron_tree_matches_brute_force(show_viewer):
    # A soft cube resting on a soft slab under a rigid ball: sample points of one body inside the tetrahedra of the
    # other and rigid samples inside both.
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=4)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0.0, 0.0, 0.05), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=2e5, nu=0.4, rho=1000.0),
    )
    scene.add_entity(
        gs.morphs.Box(size=(0.12, 0.12, 0.12), pos=(0.0, 0.0, 0.16), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    ball = scene.add_entity(gs.morphs.Sphere(radius=0.03, pos=(0.0, 0.0, 0.25)), material=gs.materials.Mochi.Rigid())
    scene.build(n_envs=2)
    ball.set_pos(np.array([[0.0, 0.0, 0.25], [0.1, 0.0, 0.14]]))
    for _ in range(40):
        scene.step()
    solver = scene.mochi_solver
    assert solver._has_soft_colliders

    hits_tree = _tetrahedron_hits(solver)
    solver.soft_info.tet_tree_brute_force.fill(1)
    hits_brute = _tetrahedron_hits(solver)
    solver.soft_info.tet_tree_brute_force.fill(0)

    for i_b in range(2):
        assert len(hits_brute[i_b]) > 20
        assert hits_tree[i_b] == hits_brute[i_b]
    assert hits_tree[0] != hits_tree[1]
