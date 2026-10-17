"""Rigid bodies simulated by the MochiSolver: a sphere and a cube dropped onto a table standing on the ground.

The MochiSolver is a fully-implicit solver with smooth penalty contact, so a 16.7 ms step is stable. Contact between the
bodies uses the analytic sphere/box distance fields of the primitives and the signed-distance grid of the table mesh.
"""

import argparse

import genesis as gs


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
        viewer_options=gs.options.ViewerOptions(camera_pos=(3.0, -3.0, 2.0), camera_lookat=(0.0, 0.0, 0.5)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    scene.add_entity(
        gs.morphs.Box(size=(1.6, 1.0, 0.1), pos=(0.0, 0.0, 0.75), fixed=True),
        material=gs.materials.Mochi.Rigid(collider_type="sdf"),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=0.2, pos=(-0.5, 0.0, 1.5)),
        material=gs.materials.Mochi.Rigid(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.5, 0.1, 1.6), euler=(20.0, 30.0, 0.0)),
        material=gs.materials.Mochi.Rigid(),
    )
    if args.record:
        cam = scene.add_camera(
            res=(640, 360),
            pos=(3.0, -3.0, 2.0),
            lookat=(0.0, 0.0, 0.5),
        )
    scene.build()

    if args.record:
        cam.start_recording(save_to_filename="rigid_bodies.mp4", fps=30)
    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            print(
                f"step {i_step:4d}: sphere z={sphere.get_pos()[2]:.4f} cube z={cube.get_pos()[2]:.4f} "
                f"newton iterations={int(info['n_iter'][0])}"
            )
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
