"""A t-shirt shell (3593 nodes, 7076 triangles) with self-contact falling onto the ground plane.

The mesh is the t-shirt asset of the original mochi engine (`tests/mochi/benchmark/assets/`); this is the
`cloth_tshirt` benchmark scene run as a standalone demo, with mochi's own settings for this garment: a Newton cap of 2
per step and the implicit normal force driving the friction dissipation. The shell collides with itself through the
spheres placed at its vertices; samples near a vertex in the rest configuration are excluded, so the sleeves and the
body of the shirt stack without passing through each other.
"""

import argparse
import os

import genesis as gs

ASSETS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "mochi", "benchmark", "assets")
)


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
        mochi_options=gs.options.MochiOptions(n_newton_iterations=2, implicit_normal_force_for_dissipation=True),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.7, -1.2, 0.8), camera_lookat=(-0.5, 0.0, 0.1)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    shirt = scene.add_entity(
        gs.morphs.Mesh(file=os.path.join(ASSETS, "tshirt_3593.obj"), pos=(-0.5, 0.0, 0.1)),
        material=gs.materials.Mochi.Shell(
            E=1e5,
            nu=0.25,
            rho=1000.0,
            thickness=2e-3,
            collider_radius=1.5e-2,
            self_contact=True,
        ),
        surface=gs.surfaces.Default(color=(0.85, 0.3, 0.35)),
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(0.7, -1.2, 0.8), lookat=(-0.5, 0.0, 0.1))
    scene.build()

    if args.record:
        cam.start_recording(save_to_filename="cloth_tshirt.mp4", fps=30)

    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            shirt_z = float(shirt.get_vertices_position()[:, 2].max())
            print(f"step {i_step:4d}: shirt zmax={shirt_z:.4f} newton iterations={int(info['n_iter'][0])}")
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
