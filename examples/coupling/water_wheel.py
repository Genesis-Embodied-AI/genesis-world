import argparse
import os

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, default="sph", choices=("sph", "mpm"), help="Fluid solver to simulate the water with"
    )
    parser.add_argument("--recon", action="store_true", help="Reconstruct a surface mesh from the particles")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.5, 0.0),
            upper_bound=(1.0, 1.5, 4.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.02,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 6.5, 3.2),
            camera_lookat=(0.5, 1.5, 1.5),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(gs.morphs.Plane())
    wheel = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/wheel/fancy_wheel.urdf",
            pos=(0.5, 0.25, 1.6),
            euler=(0, 0, 0),
            fixed=True,
            convexify=False,
        ),
    )

    emitter = scene.add_emitter(
        material=getattr(gs.materials, args.solver.upper()).Liquid(
            sampler="regular",
        ),
        max_particles=100000,
        surface=gs.surfaces.Glass(
            color=(0.7, 0.85, 1.0, 0.7),
            vis_mode="recon" if args.recon else "particle",
        ),
    )
    scene.build(n_envs=0)

    horizon = 500 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        emitter.emit(
            pos=np.array([0.5, 1.0, 3.5]),
            direction=np.array([0.0, 0, -1.0]),
            speed=5.0,
            droplet_shape="circle",
            droplet_size=0.22,
        )
        scene.step()


if __name__ == "__main__":
    main()
