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
            dt=2e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, 0.0, -0.1),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.03, -0.03, -0.08),
            upper_bound=(1.03, 1.03, 1.0),
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
            visualize_sph_boundary=True,
            rendered_envs_idx=[0],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -3, 1.42),
            camera_lookat=(0.5, 0.5, 0.4),
            camera_fov=30,
        ),
        show_viewer=args.vis,
    )

    water = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.4, 0.5, 0.25),
            size=(0.7, 0.9, 0.5),
        ),
        material=gs.materials.SPH.Liquid(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),
            vis_mode="particle",
        ),
    )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            pos=(0.5, 0.5, 0.7),
            scale=0.07,
            euler=(90.0, 0.0, 90.0),
        ),
        material=gs.materials.MPM.Elastic(
            rho=200,
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.8, 0.2, 1.0),
            vis_mode="particle",
        ),
    )

    scene.build(n_envs=0)

    horizon = 800 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
