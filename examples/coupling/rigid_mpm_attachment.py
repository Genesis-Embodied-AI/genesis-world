"""
MPM to Rigid Link Attachment

Demonstrates attaching MPM particles to rigid links using soft constraints.
"""

import argparse
import os

import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=20,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=64,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.5),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.4),
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    rigid_box = scene.add_entity(
        gs.morphs.Box(pos=(0.0, 0.0, 0.55), size=(0.12, 0.12, 0.05), fixed=False),
    )

    mpm_cube = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=5e4,
            nu=0.3,
            rho=1000,
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.35),
            size=(0.15, 0.15, 0.15),
        ),
    )

    scene.build()

    # Attach top particles of MPM cube to the rigid box
    mask = mpm_cube.get_particles_in_bbox((-0.08, -0.08, 0.41), (0.08, 0.08, 0.44))
    mpm_cube.set_particle_constraints(mask, rigid_box.links[0].idx, stiffness=1e5)

    n_steps = 500 if "PYTEST_VERSION" not in os.environ else 1
    initial_z = 0.55

    for i in range(n_steps):
        z_offset = 0.15 * (1 - abs((i % 200) - 100) / 100.0)
        target_qpos = torch.tensor([0.0, 0.0, initial_z + z_offset, 1.0, 0.0, 0.0, 0.0], device=gs.device)
        rigid_box.set_qpos(target_qpos)
        scene.step()


if __name__ == "__main__":
    main()
