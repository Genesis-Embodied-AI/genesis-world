import argparse
import os
from time import time

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        vis_options=gs.options.VisOptions(
            plane_reflection=False,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0, 0, 0),
        ),
    )
    cam_0 = scene.add_camera(
        res=(640, 480),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
    )
    scene.build()

    horizon = 2000 if "PYTEST_VERSION" not in os.environ else 5
    t = time()
    for i in range(horizon):
        cam_0.render(rgb=True, depth=True)
    print(horizon / (time() - t), "FPS")


if __name__ == "__main__":
    main()
