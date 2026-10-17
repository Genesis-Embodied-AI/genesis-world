"""A tetrahedral duck (1899 nodes, 8608 stable neo-Hookean elements) dropped onto the ground plane.

The mesh is the duck asset of the original mochi engine, converted to a tetgen `.node`/`.ele` pair in
`tests/mochi/benchmark/assets/`; this is the `soft_duck` benchmark scene run as a standalone demo. The finite elements
and the penalty contact against the plane enter one implicit Newton solve, so the 1/60 s step stays stable through the
impact.
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
        mochi_options=gs.options.MochiOptions(n_newton_iterations=20),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.2, -0.7, 1.4), camera_lookat=(-0.5, 1.0, 0.3)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    duck = scene.add_entity(
        gs.morphs.Mesh(file=os.path.join(ASSETS, "duck_1899.node"), pos=(-0.5, 1.0, 1.0)),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
        surface=gs.surfaces.Default(color=(0.95, 0.8, 0.2)),
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(1.2, -0.7, 1.4), lookat=(-0.5, 1.0, 0.3))
    scene.build()

    if args.record:
        cam.start_recording(save_to_filename="soft_duck.mp4", fps=30)

    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            duck_z = float(duck.get_vertices_position()[:, 2].min())
            print(f"step {i_step:4d}: duck zmin={duck_z:.4f} newton iterations={int(info['n_iter'][0])}")
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
