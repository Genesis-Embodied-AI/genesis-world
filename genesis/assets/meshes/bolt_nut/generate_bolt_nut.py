"""Procedurally generate a mating bolt + hex nut with ISO-metric-style threads.

The thread is built as a radial-displacement helical grid: at every (theta, z) the radius is a single-valued raised-
cosine function of the helical phase, so the resulting surface is watertight and star-shaped in every z-slice (caps are
simple fans). The nut's internal thread is cut by subtracting a slightly enlarged copy of the same thread solid from a
hex blank, so pitch, lead and handedness match the bolt by construction. Units are meters (Genesis is SI).
"""

import os

import numpy as np
import trimesh

# --- ISO-metric-ish parameters (M24 x 3 coarse), in meters ---
PITCH = 3.0e-3
R_MAJOR = 12.0e-3
THREAD_DEPTH = 1.6e-3
R_MINOR = R_MAJOR - THREAD_DEPTH
CLEARANCE = 0.30e-3  # radial gap of the nut thread w.r.t. the bolt thread

N_THETA = 40  # angular segments per turn
N_Z_PER_PITCH = 12  # axial resolution (samples per pitch)


def threaded_rod(length, r_minor, r_major, pitch, handed=+1.0, tip_chamfer=0.0):
    """Watertight solid rod whose side carries a single helical thread.

    radius(theta, z) = r_minor + depth * prof(phase), phase = z/pitch - handed*theta/2pi. prof is a 0->1->0
    raised-cosine over one pitch; unlike a sharp triangle it rounds the crest and root, giving well-conditioned SDF
    normals and stable mesh contact. A positive tip_chamfer clamps the top end to a 45-degree cone of that radial
    depth, shaving the crests down to a narrow nose so the bolt starts squarely into the nut.
    """
    depth = r_major - r_minor
    n_turns = length / pitch
    n_z = max(2, int(round(n_turns * N_Z_PER_PITCH)) + 1)

    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    z = np.linspace(0.0, length, n_z)
    tt, zz = np.meshgrid(theta, z, indexing="ij")  # [N_THETA, n_z]

    phase = (zz / pitch - handed * tt / (2.0 * np.pi)) % 1.0
    prof = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)  # 0 at root, 1 at crest, smooth
    r = r_minor + depth * prof

    if tip_chamfer > 0.0:
        # 45-degree cone capping the top: radius is clamped to r_tip at the very end and rises one-for-one with the
        # axial distance below it, so the crests within tip_chamfer of the tip are cut back to a clean lead-in nose.
        r_tip = r_minor - tip_chamfer
        r = np.minimum(r, r_tip + (length - zz))

    x = r * np.cos(tt)
    y = r * np.sin(tt)
    verts = np.stack([x, y, zz], axis=-1).reshape(-1, 3)

    def vid(i, k):
        return (i % N_THETA) * n_z + k

    faces = []
    for i in range(N_THETA):
        for k in range(n_z - 1):
            a, b = vid(i, k), vid(i, k + 1)
            c, d = vid(i + 1, k), vid(i + 1, k + 1)
            faces.append([a, b, d])
            faces.append([a, d, c])

    # End caps: fan from a center vertex on the axis (cross-section is star-shaped).
    bot_c = len(verts)
    top_c = bot_c + 1
    verts = np.vstack([verts, [0.0, 0.0, 0.0], [0.0, 0.0, length]])
    for i in range(N_THETA):
        faces.append([bot_c, vid(i + 1, 0), vid(i, 0)])
        faces.append([top_c, vid(i, n_z - 1), vid(i + 1, n_z - 1)])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def hex_prism(across_flats, height, z0=0.0, top_bevel=0.0, bottom_bevel=0.0):
    """Hexagonal prism centered on z-axis, spanning [z0, z0 + height].

    A positive top_bevel / bottom_bevel chamfers the corresponding outer edge at 45 degrees by intersecting the prism
    with an axisymmetric envelope that tapers inward there, matching the chamfered corners of a real hex head / nut.
    """
    circum = across_flats / np.sqrt(3.0)
    prism = trimesh.creation.cylinder(radius=circum, height=height, sections=6)
    prism.apply_translation([0.0, 0.0, z0 + height / 2.0])
    if top_bevel <= 0.0 and bottom_bevel <= 0.0:
        return prism

    # Envelope: a barrel a hair wider than the hex corners (so the flats pass through untouched) whose top / bottom
    # rims step inward by the bevel depth over an equal axial distance, i.e. a 45-degree cut. Its convex hull is a
    # clean watertight solid, and the intersection rounds off only the sharp top / bottom corners of the prism.
    z1 = z0 + height
    r_out = circum + 1.0e-3
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)

    def ring(radius, z_level):
        return np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z_level)], axis=-1)

    rings = [ring(r_out, z0 + bottom_bevel), ring(r_out, z1 - top_bevel)]
    if bottom_bevel > 0.0:
        rings.append(ring(r_out - bottom_bevel, z0))
    if top_bevel > 0.0:
        rings.append(ring(r_out - top_bevel, z1))
    envelope = trimesh.Trimesh(vertices=np.vstack(rings)).convex_hull
    beveled = trimesh.boolean.intersection([prism, envelope])
    beveled.fix_normals()
    return beveled


def make_bolt():
    shaft_len = 32.0e-3
    head_af = 36.0e-3
    head_h = 11.0e-3
    shaft = threaded_rod(shaft_len, R_MINOR, R_MAJOR, PITCH, tip_chamfer=0.4e-3)
    # Head sits below the shaft (shaft grows in +z from z=0); its outer end (the bottom, away from the shaft) is
    # chamfered like a real hex head, while the shaft-side shoulder stays a sharp flat bearing face.
    head = hex_prism(head_af, head_h, z0=-head_h, bottom_bevel=4.0e-3)
    # A short smooth collar avoids a degenerate union seam at the thread root.
    collar = trimesh.creation.cylinder(radius=R_MINOR, height=2.0e-3)
    collar.apply_translation([0.0, 0.0, 0.0])
    bolt = trimesh.boolean.union([shaft, head, collar])
    bolt.fix_normals()
    return bolt


def make_nut():
    nut_h = 18.0e-3
    nut_af = 36.0e-3
    # Both outer edges of the nut are chamfered, as on a standard hex nut.
    blank = hex_prism(nut_af, nut_h, z0=0.0, top_bevel=3.0e-3, bottom_bevel=3.0e-3)
    # Tap = same thread, enlarged radially by CLEARANCE, longer than the nut.
    overshoot = 4.0e-3
    tap = threaded_rod(
        nut_h + 2.0 * overshoot,
        R_MINOR + CLEARANCE,
        R_MAJOR + CLEARANCE,
        PITCH,
    )
    tap.apply_translation([0.0, 0.0, -overshoot])
    nut = trimesh.boolean.difference([blank, tap])
    nut.fix_normals()
    return nut


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    bolt = make_bolt()
    nut = make_nut()
    for name, mesh in (("bolt", bolt), ("nut", nut)):
        # Export to STL (compact binary; Genesis welds the per-triangle vertex soup on load for non-convex meshes).
        path = os.path.join(here, f"{name}.stl")
        mesh.export(path)
        print(
            f"{name}: watertight={mesh.is_watertight} verts={len(mesh.vertices)} "
            f"faces={len(mesh.faces)} volume={mesh.volume:.3e} -> {path}"
        )


if __name__ == "__main__":
    main()
