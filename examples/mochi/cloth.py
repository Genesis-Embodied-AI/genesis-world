"""Thin shells simulated by the MochiSolver: a cloth sheet draped over a rigid box, and a rigid ball caught by a sheet
whose corners are held and then swayed sideways (moving Dirichlet targets).

Light cloth carries little inertia, so the default contact stiffness of 1e9 Pa/m makes the residual jump by orders of
magnitude when it lands on an edge and trips the divergence check; 1e7 Pa/m keeps the contact stiff enough for cloth.

The shell membrane and bending terms, the rigid bodies and the penalty contact enter one implicit Newton solve. Shell
samples collide against the rigid colliders from both sides (this is also how the net catches the ball); the spheres
placed at the cloth vertices (point-cloud collider) only act against other cloths and rods.
"""

import argparse

import numpy as np
import trimesh

import genesis as gs


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
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.5, -2.5, 1.8), camera_lookat=(0.0, 0.0, 0.5)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    sheet_path = sheet_mesh("/tmp/mochi_cloth_sheet.obj", 12, 0.8)
    box = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(-0.8, 0.0, 0.15), fixed=True),
        material=gs.materials.Mochi.Rigid(),
    )
    drape = scene.add_entity(
        gs.morphs.Mesh(file=sheet_path, pos=(-0.8, 0.0, 0.6)),
        material=gs.materials.Mochi.Shell(
            E=2e4, nu=0.3, rho=200.0, thickness=2e-3, friction=0.6, collider_radius=0.02, penalty_coefficient=1e7
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.4, 0.3)),
    )
    net = scene.add_entity(
        gs.morphs.Mesh(file=sheet_path, pos=(0.8, 0.0, 0.8)),
        material=gs.materials.Mochi.Shell(
            E=5e4, nu=0.3, rho=300.0, thickness=2e-3, collider_radius=0.02, penalty_coefficient=1e7
        ),
        surface=gs.surfaces.Default(color=(0.3, 0.6, 0.9)),
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(radius=0.08, pos=(0.8, 0.0, 1.2)),
        material=gs.materials.Mochi.Rigid(rho=300.0),
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(2.5, -2.5, 1.8), lookat=(0.0, 0.0, 0.5))
    scene.build()

    corners = np.flatnonzero(np.abs(net.init_positions[:, :2]).max(axis=1) > 0.4 - 1e-6)
    corners = corners[
        (np.abs(net.init_positions[corners, 0]) > 0.4 - 1e-6) & (np.abs(net.init_positions[corners, 1]) > 0.4 - 1e-6)
    ]
    net.set_vertices_fixed(corners)
    corners_rest = net.init_positions[corners]

    if args.record:
        cam.start_recording(save_to_filename="cloth.mp4", fps=30)
    for i_step in range(args.n_steps):
        # After the ball has settled, drive the corners along a sinusoidal sway; they reach the prescribed positions
        # exactly at the end of each step.
        if i_step >= 120:
            t = (i_step + 1 - 120) * scene.options.sim.dt
            net.set_vertices_target(corners, corners_rest + np.array([0.15 * np.sin(np.pi * t), 0.0, 0.0]))
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            drape_z = float(drape.get_vertices_position()[:, 2].min())
            net_z = float(net.get_vertices_position()[:, 2].min())
            print(
                f"step {i_step:4d}: drape zmin={drape_z:.4f} net zmin={net_z:.4f} ball x={float(ball.get_pos()[0]):.4f} "
                f"z={float(ball.get_pos()[2]):.4f} "
                f"box contact={float(box.get_links_net_contact_force()[0, 2]):.2f} N newton iterations={int(info['n_iter'][0])}"
            )
        if args.record:
            cam.render()

    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
