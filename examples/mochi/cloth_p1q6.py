"""Effect of the contact quadrature on how a sharp edge shows through a cloth: the drape half of `cloth.py` with six
boundary samples per triangle (`boundary_element_type="P1Q6"`) instead of the default three (`"P1Q3"`).

Contact is sampled: the box is only felt at the quadrature points of the cloth triangles, so its edges poke through
the flat cloth between samples when the mesh is coarse relative to the box (the original mochi shows the same
artifact at the same magnitude). More samples hold the cloth closer to the surface: on this scene the deepest point
of the cloth surface below the box edge improves from about -7.6 mm (P1Q3) to about -6.1 mm, and the deepest cloth
vertex from about -1.6 mm to about -0.3 mm. Run this file twice to compare, watching the printed metric or the
viewer framed on the box edge:

    python cloth_p1q6.py -v
    python cloth_p1q6.py -v -q P1Q3

A finer sheet (shorter chords across the edge) or a larger `penalty_threshold` on the box material (contact then
engages above the surface) reduce the artifact further; see the README.
"""

import argparse

import numpy as np
import trimesh

import genesis as gs

BOX_POS = np.array([-0.8, 0.0, 0.15])
BOX_HALF = np.array([0.15, 0.15, 0.15])


def sheet_mesh(path, n_cells, size):
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
    return path, np.array(faces)


def box_sdf(points):
    """Signed distance of points to the fixed box (negative inside)."""
    q = np.abs(points - BOX_POS) - BOX_HALF
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def surface_points(verts, faces, n_sub=8):
    """Dense barycentric samples of the cloth surface (the mid-triangle points are where an edge pokes through)."""
    ij = [(i, j) for i in range(n_sub + 1) for j in range(n_sub + 1 - i)]
    bary = np.array([[1.0 - (i + j) / n_sub, i / n_sub, j / n_sub] for i, j in ij])
    return np.einsum("qk,fkd->fqd", bary, verts[faces]).reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument(
        "-q", "--quadrature", choices=("P1Q3", "P1Q6"), default="P1Q6", help="Boundary samples per triangle (3 or 6)"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8, boundary_element_type=args.quadrature),
        viewer_options=gs.options.ViewerOptions(camera_pos=(-0.35, -0.45, 0.55), camera_lookat=(-0.72, -0.05, 0.3)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    sheet_path, faces = sheet_mesh("/tmp/mochi_cloth_p1q6_sheet.obj", 12, 0.8)
    scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=tuple(BOX_POS), fixed=True),
        material=gs.materials.Mochi.Rigid(),
    )
    drape = scene.add_entity(
        gs.morphs.Mesh(file=sheet_path, pos=(-0.8, 0.0, 0.6)),
        material=gs.materials.Mochi.Shell(
            E=2e4, nu=0.3, rho=200.0, thickness=2e-3, friction=0.6, collider_radius=0.02, penalty_coefficient=1e7
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.4, 0.3)),
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(-0.35, -0.45, 0.55), lookat=(-0.72, -0.05, 0.3))
    scene.build()

    if args.record:
        cam.start_recording(save_to_filename=f"cloth_{args.quadrature.lower()}.mp4", fps=30)
    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 30 == 0:
            info = scene.mochi_solver.get_convergence_info()
            nodes = np.asarray(drape.get_vertices_position(), dtype=np.float64)
            surface = float(box_sdf(surface_points(nodes, faces)).min())
            node = float(box_sdf(nodes).min())
            print(
                f"step {i_step:4d} [{args.quadrature}]: box edge sdf surface={surface * 1e3:7.2f} mm "
                f"node={node * 1e3:7.2f} mm newton iterations={int(info['n_iter'][0])}"
            )
        if args.record:
            cam.render()

    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
