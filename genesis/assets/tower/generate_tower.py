#!/usr/bin/env python3
"""Generate stacking tower toy assets for physics simulation.

Produces for each piece: a visual GLB, a collision GLB, and a URDF.
- Visual meshes have angle-based smooth normals.
- Collision meshes are pre-decomposed into convex sub-meshes (no COACD needed).
- URDF files reference both visual and collision meshes.

All meshes are centered at origin, Z-up.
"""

import logging
import os
import textwrap

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

########################## dimensions (meters) ##########################

BASE_DIAMETER = 0.105
BASE_HEIGHT = 0.02

POLE_DIAMETER = 0.015
POLE_HEIGHT = 0.145  # from top of base to top of pole

RING_HEIGHT = 0.02
RING_HOLE_DIAMETER = 0.016  # 1 mm clearance per side around pole
RING_DIAMETERS = (0.095, 0.085, 0.075, 0.065, 0.0575, 0.050)

BALL_DIAMETER = 0.045
BALL_HOLE_DEPTH = 0.025

FILLET_RADIUS = 0.003  # fillet on ring edges
FILLET_ARC_POINTS = 8

CYLINDER_SECTIONS = 64
COLLISION_WEDGES = 16  # number of convex wedges for ring collision meshes
COLLISION_THICKNESS_MARGIN = 0.0  # Extra thickness to compensate for penetration at rest

RING_COLORS = [
    ("white", (0.95, 0.95, 0.95, 1.0)),
    ("teal", (0.60, 0.80, 0.70, 1.0)),
    ("mint", (0.78, 0.88, 0.80, 1.0)),
    ("pink", (0.90, 0.55, 0.60, 1.0)),
    ("yellow", (0.85, 0.72, 0.35, 1.0)),
    ("white", (0.95, 0.95, 0.95, 1.0)),
]

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

# Angle threshold (degrees) for smooth vs sharp edges.
SMOOTH_ANGLE_DEG = 50.0


########################## smooth normals ##########################


def _auto_smooth(mesh, angle_deg=SMOOTH_ANGLE_DEG):
    """Return a copy with angle-based smooth normals.

    1. Unmerge all vertices (each face gets its own 3 vertices -> flat shading).
    2. Group co-located vertices by position.
    3. Within each group, greedily merge vertices whose face normals are within
       the angle threshold -> smooth shading on curved surfaces, sharp rim edges.
    """
    cos_threshold = np.cos(np.radians(angle_deg))

    verts = mesh.vertices[mesh.faces.ravel()]
    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
    fn = mesh.face_normals

    vert_face = np.repeat(np.arange(len(fn)), 3)

    keys = np.round(verts, decimals=7)
    order = np.lexsort(keys.T)
    sorted_keys = keys[order]
    breaks = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    group_ids = np.empty(len(order), dtype=np.int64)
    group_ids[0] = 0
    group_ids[1:] = np.cumsum(breaks)

    merge_target = np.arange(len(verts), dtype=np.int64)

    n_groups = group_ids[-1] + 1
    group_starts = np.searchsorted(group_ids, np.arange(n_groups), side="left")
    group_ends = np.searchsorted(group_ids, np.arange(n_groups), side="right")

    for g in range(n_groups):
        members = order[group_starts[g] : group_ends[g]]
        if len(members) <= 1:
            continue

        normals = fn[vert_face[members]]

        clusters = [[0]]
        for i in range(1, len(members)):
            placed = False
            for cluster in clusters:
                if np.dot(normals[i], normals[cluster[0]]) > cos_threshold:
                    cluster.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        for cluster in clusters:
            target = members[cluster[0]]
            for idx in cluster[1:]:
                merge_target[members[idx]] = target

    new_faces = merge_target[faces]
    unique, inverse = np.unique(new_faces, return_inverse=True)
    return trimesh.Trimesh(
        vertices=verts[unique],
        faces=inverse.reshape(-1, 3),
        process=False,
    )


def _assign_cylindrical_uvs(mesh):
    """Assign cylindrical UV coordinates based on vertex positions."""
    verts = mesh.vertices
    u = (np.arctan2(verts[:, 1], verts[:, 0]) / (2 * np.pi)) % 1.0
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    v = (verts[:, 2] - z_min) / max(z_max - z_min, 1e-10)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([u, v]))


########################## filleted ring via revolve ##########################


def _create_filleted_ring(
    outer_r, inner_r, height, fillet_r=FILLET_RADIUS, sections=CYLINDER_SECTIONS, arc_n=FILLET_ARC_POINTS
):
    """Create a hollow ring with filleted edges by revolving a 2D profile."""
    h2 = height / 2
    f = min(fillet_r, h2 * 0.9, (outer_r - inner_r) / 2 * 0.9)

    pts = []

    def arc(cx, cz, start_angle):
        """Append a quarter-circle arc (excluding first point)."""
        for i in range(1, arc_n + 1):
            a = start_angle + (np.pi / 2) * i / arc_n
            pts.append([cx + f * np.cos(a), cz + f * np.sin(a)])

    # Trace cross-section starting at inner wall bottom
    pts.append([inner_r, -h2 + f])

    # Bottom-inner fillet → bottom flat
    arc(inner_r + f, -h2 + f, np.pi)

    # Bottom flat
    pts.append([outer_r - f, -h2])

    # Bottom-outer fillet → outer wall
    arc(outer_r - f, -h2 + f, -np.pi / 2)

    # Outer wall
    pts.append([outer_r, h2 - f])

    # Top-outer fillet → top flat
    arc(outer_r - f, h2 - f, 0)

    # Top flat
    pts.append([inner_r + f, h2])

    # Top-inner fillet → inner wall
    arc(inner_r + f, h2 - f, np.pi / 2)

    # Profile closes back to start automatically via revolve
    return trimesh.creation.revolve(np.array(pts), sections=sections)


########################## collision mesh helpers ##########################


def _ring_collision_mesh(outer_radius, inner_radius, height, fillet_r=FILLET_RADIUS, n_wedges=COLLISION_WEDGES):
    """Pre-decompose a ring into convex wedge sub-meshes with filleted inner hole.

    A chamfer on the inner top/bottom edges helps rings slide onto the pole.
    Each wedge remains convex by adding intermediate points along the fillet.
    """
    h2 = height / 2
    f = min(fillet_r * 2 / 3, h2 * 0.9, (outer_radius - inner_radius) / 2 * 0.9)

    # Inner edge profile: chamfer at top and bottom of the hole
    # Points go from bottom-inner-wall up to top-inner-wall, with fillets
    inner_rz = [
        (inner_radius, -h2 + f),  # bottom of inner wall (above fillet)
        (inner_radius + f, -h2),  # bottom flat (fillet end)
        (inner_radius + f, h2),  # top flat (fillet start)
        (inner_radius, h2 - f),  # top of inner wall (below fillet)
    ]

    scene = trimesh.Scene()
    for i in range(n_wedges):
        theta0 = 2 * np.pi * i / n_wedges
        theta1 = 2 * np.pi * (i + 1) / n_wedges
        thetas = np.linspace(theta0, theta1, 4)
        points = []
        for theta in thetas:
            c, s = np.cos(theta), np.sin(theta)
            # Outer wall points (no fillet, just top/bottom)
            for z in (-h2, h2):
                points.append([outer_radius * c, outer_radius * s, z])
            # Inner wall points (with fillet profile)
            for r, z in inner_rz:
                points.append([r * c, r * s, z])
        wedge = trimesh.convex.convex_hull(np.array(points))
        scene.add_geometry(wedge, node_name=f"wedge_{i:02d}")
    return scene


########################## URDF generation ##########################

_URDF_TEMPLATE = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="utf-8"?>
    <robot name="{name}">
      <link name="world"/>
      <joint name="world_to_{name}" type="floating">
        <parent link="world"/>
        <child link="{name}"/>
      </joint>
      <link name="{name}">
        <inertial>
          <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
        <visual>
          <geometry>
            <mesh filename="{vis_file}" scale="1 1 1"/>
          </geometry>
        </visual>
        <collision>
          <geometry>
            <mesh filename="{coll_file}" scale="1 1 1"/>
          </geometry>
        </collision>
      </link>
    </robot>
"""
)


def _write_urdf(name, vis_file, coll_file):
    """Write a URDF file referencing visual and collision meshes."""
    path = os.path.join(ASSETS_DIR, f"{name}.urdf")
    with open(path, "w") as f:
        f.write(_URDF_TEMPLATE.format(name=name, vis_file=vis_file, coll_file=coll_file))
    return path


########################## piece generators ##########################


def generate_base_pole():
    """Generate base disc + pole capsule."""
    base = trimesh.creation.cylinder(
        radius=BASE_DIAMETER / 2,
        height=BASE_HEIGHT,
        sections=CYLINDER_SECTIONS,
    )

    pole_radius = POLE_DIAMETER / 2
    pole_total = POLE_HEIGHT + BASE_HEIGHT
    pole_cylinder_height = pole_total - 2 * pole_radius
    pole = trimesh.creation.capsule(
        height=pole_cylinder_height,
        radius=pole_radius,
        count=[CYLINDER_SECTIONS, 8],
    )
    pole_z = -BASE_HEIGHT / 2 + pole_total / 2
    pole.apply_translation([0, 0, pole_z])

    # Assign cylindrical UVs for wood texture
    _assign_cylindrical_uvs(base)
    _assign_cylindrical_uvs(pole)

    # Visual and collision are the same (already 2 convex sub-meshes)
    scene = trimesh.Scene()
    scene.add_geometry(base, node_name="base")
    scene.add_geometry(pole, node_name="pole")

    vis_file = "base_pole.glb"
    scene.export(os.path.join(ASSETS_DIR, vis_file))

    _write_urdf("base_pole", vis_file, vis_file)

    logger.info(
        f"  base_pole  | base {BASE_DIAMETER * 100:.2f}cm"
        f" + pole {POLE_DIAMETER * 100:.2f}cm x {POLE_HEIGHT * 100:.2f}cm"
    )


def generate_ring(index, outer_diameter):
    """Generate a hollow ring with separate visual and collision meshes."""
    name = f"ring_{index + 1:02d}"
    outer_r = outer_diameter / 2
    inner_r = RING_HOLE_DIAMETER / 2

    # Visual: filleted ring via revolved profile + smooth normals
    ring = _create_filleted_ring(outer_r, inner_r, RING_HEIGHT)

    # Export with angle-based smooth normals
    vis_file = f"{name}.glb"
    _auto_smooth(ring).export(os.path.join(ASSETS_DIR, vis_file))

    # Collision: pre-decomposed convex wedges (thicker to compensate for penetration at rest)
    coll_file = f"{name}_coll.glb"
    coll_scene = _ring_collision_mesh(outer_r, inner_r, RING_HEIGHT + COLLISION_THICKNESS_MARGIN)
    coll_scene.export(os.path.join(ASSETS_DIR, coll_file))

    _write_urdf(name, vis_file, coll_file)

    color_name = RING_COLORS[index][0]
    logger.info(f"  {name}     | diam {outer_diameter * 100:.2f}cm | {color_name}")


def generate_ball():
    """Generate a sphere with a blind hole, separate visual and collision meshes."""
    # Visual: sphere with hole + smooth normals
    ball = trimesh.creation.icosphere(subdivisions=3, radius=BALL_DIAMETER / 2)
    hole = trimesh.creation.cylinder(
        radius=RING_HOLE_DIAMETER / 2,
        height=BALL_HOLE_DEPTH * 2,
        sections=CYLINDER_SECTIONS,
    )
    hole_z = -BALL_DIAMETER / 2 + BALL_HOLE_DEPTH / 2
    hole.apply_translation([0, 0, hole_z])
    ball_vis = trimesh.boolean.difference([ball, hole], engine="manifold")

    # Export with smooth normals + spherical UV coordinates
    vis_file = "ball.glb"
    smoothed = _auto_smooth(ball_vis)
    verts = smoothed.vertices
    r = np.linalg.norm(verts, axis=1).clip(min=1e-10)
    u = (np.arctan2(verts[:, 1], verts[:, 0]) / (2 * np.pi)) % 1.0
    v = np.arccos(np.clip(verts[:, 2] / r, -1, 1)) / np.pi
    smoothed.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([u, v]))
    smoothed.export(os.path.join(ASSETS_DIR, vis_file))

    # Collision: sphere with hole, decomposed into convex pieces.
    # Split into a convex top cap (no hole) + angular × vertical wedges for the
    # ring region. Each wedge samples points on the sphere surface (outer) and
    # hole wall (inner). Convex hulls of sphere-surface points never exceed the
    # sphere, so the outer shape is preserved exactly.
    coll_file = "ball_coll.glb"
    ball_r = BALL_DIAMETER / 2
    hole_r = RING_HOLE_DIAMETER / 2

    # Z where sphere radius equals hole radius (bottom of material)
    z_bottom = -np.sqrt(ball_r**2 - hole_r**2)
    # Z where hole cylinder ends inside the sphere
    z_hole_top = hole_z + BALL_HOLE_DEPTH  # = -ball_r + 1.5 * BALL_HOLE_DEPTH

    coll_scene = trimesh.Scene()

    # 1) Top cap: sphere above z_hole_top (convex, one piece)
    cap = trimesh.creation.icosphere(subdivisions=2, radius=ball_r)
    clip_box = trimesh.creation.box(extents=[ball_r * 4, ball_r * 4, ball_r * 4])
    clip_box.apply_translation([0, 0, z_hole_top - ball_r * 2])
    cap = trimesh.boolean.difference([cap, clip_box], engine="manifold")
    coll_scene.add_geometry(cap, node_name="top_cap")

    # 2) Ring region wedges: angular × vertical layers
    n_layers = 4
    z_layers = np.linspace(z_bottom, z_hole_top, n_layers + 1)
    n_wedges = COLLISION_WEDGES
    n_lon = 4  # longitude samples per wedge

    for li in range(n_layers):
        z_lo, z_hi = z_layers[li], z_layers[li + 1]
        z_mid = (z_lo + z_hi) / 2
        z_samples = [z_lo, z_mid, z_hi]

        for wi in range(n_wedges):
            theta0 = 2 * np.pi * wi / n_wedges
            theta1 = 2 * np.pi * (wi + 1) / n_wedges
            thetas = np.linspace(theta0, theta1, n_lon)

            points = []
            for z in z_samples:
                r_sphere = np.sqrt(max(ball_r**2 - z**2, 0))
                for theta in thetas:
                    c, s = np.cos(theta), np.sin(theta)
                    # Outer: sphere surface
                    points.append([r_sphere * c, r_sphere * s, z])
                    # Inner: hole wall
                    points.append([hole_r * c, hole_r * s, z])

            wedge = trimesh.convex.convex_hull(np.array(points))
            coll_scene.add_geometry(wedge, node_name=f"layer{li}_wedge{wi:02d}")

    coll_scene.export(os.path.join(ASSETS_DIR, coll_file))

    _write_urdf("ball", vis_file, coll_file)

    logger.info(f"  ball       | diam {BALL_DIAMETER * 100:.2f}cm | hole {BALL_HOLE_DEPTH * 100:.2f}cm deep")


########################## main ##########################


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Generating stacking tower assets...")

    generate_base_pole()

    for i, diameter in enumerate(RING_DIAMETERS):
        generate_ring(i, diameter)

    generate_ball()

    logger.info(f"Done — assets in {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
