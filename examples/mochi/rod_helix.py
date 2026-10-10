"""A helical spring (129 nodes) hanging from its first node, stretching and oscillating under gravity.

The centerline is the helix asset of the original mochi engine (`tests/mochi/benchmark/assets/`); this is the
`rod_helix` benchmark scene run as a standalone demo. The discrete elastic rod carries stretching, bending and
twisting (one twist angle per segment as an extra unknown); the stiffnesses follow from the radius and the elastic
moduli of steel-like material (E = G = 1 GPa here). Contact is disabled: the spring never touches anything, it only
rings on its own elasticity.
"""

import argparse
import os

import numpy as np

import genesis as gs

ASSETS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "mochi", "benchmark", "assets")
)
ROD_RADIUS = 1e-2
ROD_E = 1e9
ROD_G = 1e9
ROD_RHO = 1000.0


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
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.0, -1.0, 1.0), camera_lookat=(0.0, 0.0, 0.7)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    area = np.pi * ROD_RADIUS**2
    second_moment = 0.25 * np.pi * ROD_RADIUS**4
    polar_moment = 0.5 * np.pi * ROD_RADIUS**4
    rod = scene.add_entity(
        gs.morphs.Rod(points=np.load(os.path.join(ASSETS, "helix_129.npy")), radius=ROD_RADIUS, pos=(0.0, 0.0, 1.0)),
        material=gs.materials.Mochi.Rod(
            collider_type="none",
            axial_stiffness=ROD_E * area,
            flexural_stiffness=ROD_E * second_moment,
            torsional_stiffness=ROD_G * polar_moment,
            linear_density=ROD_RHO * area,
            linear_rotational_inertia=ROD_RHO * polar_moment,
        ),
        surface=gs.surfaces.Default(color=(0.7, 0.7, 0.75)),
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(1.0, -1.0, 1.0), lookat=(0.0, 0.0, 0.7))
    scene.build()
    rod.set_vertices_fixed([0])

    if args.record:
        cam.start_recording(save_to_filename="rod_helix.mp4", fps=30)

    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            tip_z = float(rod.get_vertices_position()[-1, 2])
            print(f"step {i_step:4d}: tip z={tip_z:.4f} newton iterations={int(info['n_iter'][0])}")
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
