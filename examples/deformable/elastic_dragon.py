import argparse
import os


import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=10,
            gravity=(0, 0, -9.8),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2, 2, 1.5),
            camera_lookat=(0, 0, 0.5),
            camera_up=(0, 0, 1),
        ),
        show_viewer=args.vis,
    )

    mat_elastic = gs.materials.PBD.Elastic()

    bunny = scene.add_entity(
        material=mat_elastic,
        morph=gs.morphs.Mesh(
            file="meshes/dragon/dragon.obj",
            scale=0.003,
            pos=(0, 0, 0.8),
        ),
        surface=gs.surfaces.Default(),
    )
    scene.build()

    horizon = 1000 if "PYTEST_VERSION" not in os.environ else 5
    # forward pass
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
