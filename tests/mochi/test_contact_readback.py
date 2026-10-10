import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose


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
    return str(path)


def _as_arrays(contacts):
    return {key: tensor_to_array(value) for key, value in contacts.items()}


def _scatter(contacts, side, n_vertices, entity_idx):
    """Force on the vertices of a deformable entity from the records where it is on the given side ('a' or 'b')."""
    is_side = contacts[f"entity_{side}"] == entity_idx
    verts = contacts[f"verts_{side}"][is_side]
    bary = contacts[f"bary_{side}"][is_side]
    force = contacts[f"force_{side}"][is_side]
    out = np.zeros((n_vertices, 3))
    for k in range(verts.shape[1]):
        valid = verts[:, k] >= 0
        np.add.at(out, verts[valid, k], bary[valid, k, None] * force[valid])
    return out


@pytest.mark.precision("64")
def test_soft_on_rigid_contacts(show_viewer):
    size, threshold = 0.2, 1e-3
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    plane = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid(penalty_threshold=threshold))
    cube = scene.add_entity(
        gs.morphs.Box(size=(size, size, size), pos=(0.0, 0.0, 0.5 * size + 0.01), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0, penalty_threshold=threshold),
    )
    scene.build()
    for _ in range(90):
        scene.step()
    contacts = _as_arrays(cube.get_contacts())
    n_contacts = len(contacts["entity_a"])
    assert n_contacts > 0
    assert np.all(contacts["entity_a"] == cube.idx)
    assert np.all(contacts["entity_b"] == plane.idx)
    assert np.all(contacts["link_a"] == -1)
    assert np.all(contacts["geom_a"] == -1)
    assert np.all(contacts["link_b"] == plane.base_link.idx)
    assert np.all(contacts["geom_b"] == plane.geoms[0].idx)
    assert np.all((contacts["verts_a"] >= 0) & (contacts["verts_a"] < cube.n_vertices))
    assert_allclose(contacts["bary_a"].sum(axis=1), 1.0, tol=1e-12)
    assert np.all(contacts["verts_b"] == -1)
    assert_allclose(contacts["normal"], np.tile((0.0, 0.0, 1.0), (n_contacts, 1)), tol=1e-12)
    assert np.all(contacts["distance"] < threshold)
    assert np.all(contacts["distance"] > threshold - 5e-3)
    assert_allclose(contacts["position"][:, 2], contacts["distance"], tol=1e-12)
    assert np.all(contacts["weight"] > 0.0)
    # The records account for the whole contact force on the vertices of the body, which balances its weight.
    force_verts = tensor_to_array(cube.get_vertices_contact_force())
    assert_allclose(_scatter(contacts, "a", cube.n_vertices, cube.idx), force_verts, atol=1e-9)
    assert_allclose(contacts["force_a"].sum(axis=0), (0.0, 0.0, cube.mass * 9.8), atol=2e-3)
    # The same records are seen from the collider's side, and with the pair filter.
    for entity, other in ((plane, cube), (cube, plane)):
        view = _as_arrays(entity.get_contacts(with_entity=other))
        assert len(view["entity_a"]) == n_contacts
        assert_allclose(view["force_a"], contacts["force_a"], tol=0.0)
    assert len(plane.get_contacts(with_entity=plane)["entity_a"]) == 0


@pytest.mark.precision("64")
def test_soft_on_soft_contacts(show_viewer):
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    plane = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    bottom = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0.0, 0.0, 0.05), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    top = scene.add_entity(
        gs.morphs.Box(size=(0.12, 0.12, 0.12), pos=(0.0, 0.0, 0.2), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    scene.build()
    assert scene.mochi_solver._has_soft_colliders
    for _ in range(150):
        scene.step()
    contacts = _as_arrays(top.get_contacts(with_entity=bottom))
    n_contacts = len(contacts["entity_a"])
    assert n_contacts > 0
    is_top_a = contacts["entity_a"] == top.idx
    assert np.all(contacts["entity_b"][is_top_a] == bottom.idx)
    assert np.all(contacts["entity_a"][~is_top_a] == bottom.idx)
    assert np.all(contacts["entity_b"][~is_top_a] == top.idx)
    for key in ("link_a", "link_b", "geom_a", "geom_b"):
        assert np.all(contacts[key] == -1)
    for side in ("a", "b"):
        n_vertices = np.where(contacts[f"entity_{side}"] == top.idx, top.n_vertices, bottom.n_vertices)
        assert np.all((contacts[f"verts_{side}"] >= 0) & (contacts[f"verts_{side}"] < n_vertices[:, None]))
        assert_allclose(contacts[f"bary_{side}"].sum(axis=1), 1.0, tol=1e-12)
    # A sample point is located inside the collider body (non-negative barycentric weights) and recorded within the
    # smoothing band of the penalty (the rest-shape distance field is slightly positive near the surface).
    assert np.all(contacts["bary_b"] > -1e-12)
    assert np.all(contacts["distance"] < 1e-2)
    assert_allclose(contacts["position"][:, 2], 0.1, atol=1e-2)
    # Records where the upper body is the sample side and where it is the collider side together account for its
    # vertex contact forces, which balance its weight.
    force_top = _scatter(contacts, "a", top.n_vertices, top.idx) + _scatter(contacts, "b", top.n_vertices, top.idx)
    assert_allclose(force_top, tensor_to_array(top.get_vertices_contact_force()), atol=1e-9)
    # Slow frictional creep of the upper body leaves a small lateral component.
    assert_allclose(force_top.sum(axis=0), (0.0, 0.0, top.mass * 9.8), atol=1e-2)
    assert len(bottom.get_contacts(with_entity=top)["entity_a"]) == n_contacts
    # The lower body also touches the ground.
    all_bottom = _as_arrays(bottom.get_contacts())
    assert set(np.unique(all_bottom["entity_b"])) == {plane.idx, bottom.idx} or set(
        np.unique(np.concatenate([all_bottom["entity_a"], all_bottom["entity_b"]]))
    ) == {plane.idx, bottom.idx, top.idx}


@pytest.mark.precision("64")
def test_point_cloud_contacts(tmp_path, show_viewer):
    # A rigid ball and a small cloth patch rest on a corner-held net: the ball is carried by the net's samples
    # colliding against its sphere collider (point-cloud colliders never collide with rigid bodies, as mochi), the
    # patch by the point-cloud contact between the two cloths.
    radius, collider_radius, n_cells, size = 0.05, 0.02, 10, 0.5
    obj_path = _sheet_obj(tmp_path / "sheet.obj", n_cells, size)
    patch_path = _sheet_obj(tmp_path / "patch.obj", 4, 0.15)
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 0.5)),
        material=gs.materials.Mochi.Shell(
            E=1e6, nu=0.3, rho=1000.0, thickness=2e-3, collider_radius=collider_radius, mass_damping=2.0
        ),
    )
    patch = scene.add_entity(
        gs.morphs.Mesh(file=patch_path, pos=(0.15, 0.15, 0.55)),
        material=gs.materials.Mochi.Shell(
            E=1e6, nu=0.3, rho=1000.0, thickness=2e-3, collider_radius=collider_radius, mass_damping=2.0
        ),
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(radius=radius, pos=(0.0, 0.0, 0.5 + radius + 0.01)),
        material=gs.materials.Mochi.Rigid(rho=500.0),
    )
    scene.build()
    assert scene.mochi_solver._has_pc_colliders
    verts = tensor_to_array(cloth.get_vertices_position())
    corners = np.flatnonzero((np.abs(verts[:, 0]) > 0.5 * size - 1e-6) & (np.abs(verts[:, 1]) > 0.5 * size - 1e-6))
    assert len(corners) == 4
    cloth.set_vertices_fixed(corners)
    for _ in range(240):
        scene.step()
    # The ball's contacts are shell samples of the net against its sphere collider.
    contacts = _as_arrays(ball.get_contacts())
    n_contacts = len(contacts["entity_a"])
    assert n_contacts > 0
    assert np.all(contacts["entity_a"] == cloth.idx)
    assert np.all(contacts["entity_b"] == ball.idx)
    assert np.all(contacts["link_a"] == -1)
    assert np.all(contacts["geom_a"] == -1)
    assert np.all(contacts["link_b"] == ball.base_link.idx)
    assert np.all(contacts["geom_b"] == ball.geoms[0].idx)
    assert np.all((contacts["verts_a"] >= 0) & (contacts["verts_a"] < cloth.n_vertices))
    assert_allclose(contacts["bary_a"].sum(axis=1), 1.0, tol=1e-12)
    assert np.all(contacts["verts_b"] == -1)
    # Records cover the contact range of the penalty (the threshold).
    assert np.all(contacts["distance"] < 1.1e-3)
    # The collider gradient points away from the ball at every contact.
    ball_pos = tensor_to_array(ball.get_pos())
    assert np.all(np.einsum("ij,ij->i", contacts["normal"], contacts["position"] - ball_pos) > 0.0)
    # Records are consistent with the per-body readbacks: net force on the ball, force on the cloth vertices.
    ball_force = tensor_to_array(ball.get_links_net_contact_force())[0]
    assert_allclose(-contacts["force_a"].sum(axis=0), ball_force, atol=1e-9)
    # The ball still rolls slowly in the sag of the net: only the vertical balance is tight.
    assert_allclose(ball_force[2], float(tensor_to_array(ball.get_mass())) * 9.8, atol=1e-2)
    assert np.linalg.norm(ball_force[:2]) < 0.05
    assert len(cloth.get_contacts(with_entity=ball)["entity_a"]) == n_contacts
    # The patch rests on the net through point-cloud contacts: single-vertex collider records between the two cloths.
    pc_contacts = _as_arrays(patch.get_contacts(with_entity=cloth))
    n_pc = len(pc_contacts["entity_a"])
    assert n_pc > 0
    assert set(np.unique(pc_contacts["entity_a"])) | set(np.unique(pc_contacts["entity_b"])) == {cloth.idx, patch.idx}
    assert np.all(pc_contacts["link_a"] == -1)
    assert np.all(pc_contacts["link_b"] == -1)
    assert np.all(pc_contacts["verts_b"][:, 0] >= 0)
    assert np.all(pc_contacts["verts_b"][:, 1:] == -1)
    assert_allclose(pc_contacts["bary_b"], np.tile((1.0, 0.0, 0.0, 0.0), (n_pc, 1)), tol=0.0)
    # All records involving the cloth scattered onto its vertices match the vertex force readback.
    cloth_contacts = _as_arrays(cloth.get_contacts())
    assert_allclose(
        _scatter(cloth_contacts, "a", cloth.n_vertices, cloth.idx)
        + _scatter(cloth_contacts, "b", cloth.n_vertices, cloth.idx),
        tensor_to_array(cloth.get_vertices_contact_force()),
        atol=1e-9,
    )
