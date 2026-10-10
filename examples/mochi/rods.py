"""Rods simulated by the MochiSolver: a rope clamped at both ends sagging under gravity, a stiff cantilever, and a
loose rope dropped onto the ground.

Rods are discrete elastic rods (stretching, bending and twisting) whose nodes and segment twists are unknowns of the
same implicit Newton solve as the other Mochi bodies. The tube drawn around the centerline follows the material frame
of the segments. Rods collide through samples on their centerline, so the dropped rope comes to rest with its
centerline on the ground.
"""

import argparse

import numpy as np

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
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.5, -2.5, 1.5), camera_lookat=(0.0, 0.0, 0.8)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    n_segments = 40
    points = np.stack(
        [np.linspace(-0.75, 0.75, n_segments + 1), np.zeros(n_segments + 1), np.zeros(n_segments + 1)], -1
    )
    rope = scene.add_entity(
        gs.morphs.Rod(points=points, radius=0.01, pos=(0.0, 0.0, 1.2)),
        material=gs.materials.Mochi.Rod(E=5e6, nu=0.3, rho=1200.0, mass_damping=1.0),
        surface=gs.surfaces.Default(color=(0.8, 0.6, 0.2)),
    )
    beam = scene.add_entity(
        gs.morphs.Rod(points=points[: n_segments // 2 + 1] + (0.75, 0.0, 0.0), radius=0.015, pos=(0.0, 0.5, 0.8)),
        material=gs.materials.Mochi.Rod(E=2e9, nu=0.3, rho=2000.0, mass_damping=2.0),
        surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9)),
    )
    wave = np.stack(
        [
            np.linspace(-0.5, 0.5, n_segments + 1),
            0.15 * np.sin(np.linspace(0.0, 4.0 * np.pi, n_segments + 1)),
            np.zeros(n_segments + 1),
        ],
        -1,
    )
    loose = scene.add_entity(
        gs.morphs.Rod(points=wave, radius=0.01, pos=(0.0, -0.6, 0.4), euler=(15.0, 0.0, 0.0)),
        material=gs.materials.Mochi.Rod(E=5e6, nu=0.3, rho=1200.0, mass_damping=1.0),
        surface=gs.surfaces.Default(color=(0.4, 0.8, 0.4)),
    )
    if args.record:
        cam = scene.add_camera(
            res=(640, 360),
            pos=(2.5, -2.5, 1.5),
            lookat=(0.0, 0.0, 0.8),
        )
    scene.build()

    rope.set_vertices_fixed([0, n_segments])
    beam.set_vertices_fixed([0, 1])

    if args.record:
        cam.start_recording(save_to_filename="rods.mp4", fps=30)

    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            rope_z = float(rope.get_vertices_position()[:, 2].min())
            beam_z = float(beam.get_vertices_position()[-1, 2])
            loose_z = float(loose.get_vertices_position()[:, 2].mean())
            print(
                f"step {i_step:4d}: rope lowest z={rope_z:.4f} beam tip z={beam_z:.4f} "
                f"dropped rope mean z={loose_z:.4f} newton iterations={int(info['n_iter'][0])}"
            )
        if args.record:
            cam.render()

    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
