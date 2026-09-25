import os
import argparse
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=64,
            enable_CPIC=True,
            lower_bound=(-1.0, -1.0, -0.01),
            upper_bound=(1.0, 1.0, 2.0),
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, 0.9, 3.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    cutter = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cross_cutter.obj",
            scale=0.8,
            pos=(0.0, 0.0, 0.3),
            euler=(90, 0, 0),
            fixed=True,
            convexify=False,
        ),
        surface=gs.surfaces.Iron(),
    )
    dragon = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/dragon/dragon.obj",
            scale=0.007,
            euler=(0, 0, 90),
            pos=(0.3, -0.0, 1.3),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="pbs-64",
        ),
        surface=gs.surfaces.Rough(
            color=(0.6, 1.0, 0.8, 1.0),
            vis_mode="particle",
        ),
    )
    scene.build(n_envs=0)

    horizon = 400 if "PYTEST_VERSION" not in os.environ else 5
    for _ in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
