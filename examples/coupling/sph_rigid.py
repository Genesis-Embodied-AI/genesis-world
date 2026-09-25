import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(0.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 2.4),
        ),
        vis_options=gs.options.VisOptions(
            visualize_sph_boundary=True,
            rendered_envs_idx=[0],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -3.15, 2.42),
            camera_lookat=(0.5, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    water = scene.add_entity(
        material=gs.materials.SPH.Liquid(
            sampler="regular",
            mu=0.01,
        ),
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.6),
            size=(0.9, 1.6, 1.2),
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.7, 0.9, 1.0),
        ),
    )

    cube = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 2.4),
            size=(0.2, 0.2, 0.2),
            euler=(30, 40, 0),
            fixed=False,
        ),
    )

    scene.build()

    horizon = 500 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
