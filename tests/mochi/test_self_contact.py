import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array


def _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -9.8), **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
        mochi_options=gs.options.MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


def _strip_obj(path, n_x, n_y, length, width):
    xs = np.linspace(-0.5 * length, 0.5 * length, n_x + 1)
    ys = np.linspace(-0.5 * width, 0.5 * width, n_y + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_x):
        for j in range(n_y):
            a = i * (n_y + 1) + j
            faces.append([a, a + 1, a + n_y + 2])
            faces.append([a, a + n_y + 2, a + n_y + 1])
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return str(path)


def _hairpin_points(leg_length, gap, n_leg, n_arc, z_offset=0.0):
    """Two parallel legs joined by a half circle; the second leg is raised by z_offset."""
    x = np.linspace(0.0, leg_length, n_leg + 1)
    leg_a = np.stack([x, np.full_like(x, -0.5 * gap), np.zeros_like(x)], axis=-1)
    angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_arc + 1)[1:-1]
    arc = np.stack(
        [
            leg_length + 0.5 * gap * np.cos(angles),
            0.5 * gap * np.sin(angles),
            z_offset * (angles + 0.5 * np.pi) / np.pi,
        ],
        axis=-1,
    )
    leg_b = np.stack([x[::-1], np.full_like(x, 0.5 * gap), np.full_like(x, z_offset)], axis=-1)
    return np.concatenate([leg_a, arc, leg_b])


@pytest.mark.precision("64")
@pytest.mark.parametrize("self_contact", [True, False])
def test_shell_self_contact(tmp_path, show_viewer, self_contact):
    length, width, n_x, n_y, radius = 0.6, 0.15, 12, 3, 0.02
    obj_path = _strip_obj(tmp_path / "strip.obj", n_x, n_y, length, width)
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid(penalty_coefficient=1e7))
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 1e-3)),
        material=gs.materials.Mochi.Shell(
            E=1e5,
            nu=0.3,
            rho=200.0,
            thickness=1e-3,
            collider_radius=radius,
            penalty_coefficient=1e7,
            mass_damping=5.0,
            self_contact=self_contact,
        ),
    )
    scene.build()
    rest = tensor_to_array(cloth.get_vertices_position())
    pinned = np.flatnonzero(rest[:, 0] < -0.5 * length + 1e-6)
    driven = np.flatnonzero(rest[:, 0] > 0.5 * length - 1e-6)
    cloth.set_vertices_fixed(pinned)
    # Fold the strip over onto itself: the free edge travels along a half circle about the middle of the strip and
    # lands above the pinned edge.
    n_fold = 90
    for i_step in range(n_fold):
        angle = np.pi * (i_step + 1) / n_fold
        target = rest[driven].copy()
        target[:, 0] = 0.5 * length * np.cos(angle)
        target[:, 2] = 0.5 * length * np.sin(angle) + 1.5 * radius * (i_step + 1) / n_fold + 1e-3
        cloth.set_vertices_target(driven, target)
        scene.step()
    for _ in range(150):
        scene.step()
    pos = tensor_to_array(cloth.get_vertices_position())
    # Interior vertices of the folded half (excluding the driven edge and the fold).
    upper = np.flatnonzero((rest[:, 0] > 0.1 * length) & (rest[:, 0] < 0.5 * length - 1e-6))
    lower = np.flatnonzero(rest[:, 0] < -0.1 * length)
    height = np.median(pos[upper, 2]) - np.median(pos[lower, 2])
    if self_contact:
        # The upper layer rests on the collider spheres of the lower layer.
        assert 0.6 * radius < height < 1.3 * radius
    else:
        # Without self-contact the upper layer falls through onto the ground.
        assert height < 0.3 * radius


def _segments_min_distance(pa, pb):
    """Smallest distance between the segments of two polylines (pa, pb: (n, 3) node arrays)."""
    best = np.inf
    for i in range(len(pa) - 1):
        for j in range(len(pb) - 1):
            p0, u = pa[i], pa[i + 1] - pa[i]
            q0, v = pb[j], pb[j + 1] - pb[j]
            w = p0 - q0
            a, b, c, d, e = u @ u, u @ v, v @ v, u @ w, v @ w
            den = a * c - b * b
            s_ = np.clip((b * e - c * d) / den, 0.0, 1.0) if den > 1e-14 else 0.0
            t_ = (b * s_ + e) / c
            if t_ < 0.0:
                t_, s_ = 0.0, np.clip(-d / a, 0.0, 1.0)
            elif t_ > 1.0:
                t_, s_ = 1.0, np.clip((b - d) / a, 0.0, 1.0)
            best = min(best, np.linalg.norm(w + s_ * u - t_ * v))
    return best


@pytest.mark.precision("64")
@pytest.mark.parametrize("self_contact", [True, False])
def test_rod_self_contact(show_viewer, self_contact):
    # Segments as long as the radius so that the chain of collider spheres has no gaps a rod could slip through; the
    # second leg is slightly raised so that the legs can pass over each other instead of meeting head-on in a plane.
    leg_length, gap, radius, n_leg, n_arc = 0.4, 0.1, 0.01, 40, 16
    points = _hairpin_points(leg_length, gap, n_leg, n_arc, z_offset=0.3 * radius)
    scene = _mochi_scene(show_viewer, 0.01, gravity=(0.0, 0.0, 0.0), n_newton_iterations=8)
    rod = scene.add_entity(
        gs.morphs.Rod(points=points, radius=radius),
        material=gs.materials.Mochi.Rod(
            E=1e7,
            nu=0.3,
            rho=1000.0,
            mass_damping=1.0,
            self_contact=self_contact,
            self_contact_exclusion_ratio=3.0,
        ),
    )
    scene.build()
    rest = tensor_to_array(rod.get_vertices_position())
    n_nodes = len(points)
    # Swap the sides of the two clamped ends (two nodes each) so that the legs have to pass each other.
    ends = np.array([0, 1, n_nodes - 2, n_nodes - 1])
    # The prescribed end segments cross by construction: measure the free parts of the legs.
    leg_a = np.arange(3, n_leg + 1)
    leg_b = n_nodes - 1 - leg_a
    n_press = 100
    distance = np.inf
    for i_step in range(n_press + 150):
        if i_step < n_press:
            s = (i_step + 1) / n_press
            target = rest[ends].copy()
            target[:, 1] = rest[ends, 1] * (1.0 - 2.0 * s)
            rod.set_vertices_target(ends, target)
        scene.step()
        pos = tensor_to_array(rod.get_vertices_position())
        assert np.all(np.isfinite(pos))
        # Closest approach of the two legs over the whole motion (they end up on swapped sides either way).
        distance = min(distance, _segments_min_distance(pos[leg_a], pos[leg_b]))
    if self_contact:
        # The legs pass over each other, kept apart by the collider spheres of their nodes.
        assert distance > 0.6 * radius
    else:
        # Without self-contact the legs pass through each other's spheres at their initial offset.
        assert distance < 0.45 * radius
