import argparse
import os

import numpy as np

import genesis as gs
from genesis.options.sensors import RasterizerCameraOptions
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the images of the wrist camera")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.5, 1.5),
            camera_lookat=(0.4, 0.0, 0.3),
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    for i, color in enumerate(((0.9, 0.2, 0.2), (0.2, 0.8, 0.3), (0.2, 0.4, 0.9))):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.55, 0.25 * (i - 1), 0.04),
                size=(0.08, 0.08, 0.08),
            ),
            surface=gs.surfaces.Rough(
                color=color,
            ),
        )

    # The calibration of a fisheye wrist camera, in the Kannala-Brandt model of OpenCV, at half its resolution. The
    # camera looks along the fingers, the -z axis of the camera along the z axis of the hand.
    scale = 0.5
    wrist_camera = scene.add_sensor(
        RasterizerCameraOptions(
            res=(int(1600 * scale), int(1200 * scale)),
            lens=gs.options.FisheyeLens(
                intrinsics=(
                    (532.684 * scale, 0.0, 821.626 * scale),
                    (0.0, 532.656 * scale, 615.761 * scale),
                    (0.0, 0.0, 1.0),
                ),
                distortion=(0.033033, 0.0017692, 0.0023736, -0.00081624),
            ),
            entity_idx=franka.idx,
            link_idx_local=franka.get_link("hand").idx_local,
            offset_T=np.array(
                (
                    (1.0, 0.0, 0.0, 0.09),
                    (0.0, -1.0, 0.0, 0.0),
                    (0.0, 0.0, -1.0, 0.04),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
            near=0.02,
            far=10.0,
        )
    )
    scene.build()

    # The arm bends over the boxes, the hand pointing down
    qpos_home = np.array((0.0, -0.3, 0.0, -2.4, 0.0, 2.1, 0.785, 0.04, 0.04))
    franka.set_dofs_position(qpos_home)

    if args.vis:
        # An optional dependency, which showing the images alone requires
        import matplotlib.pyplot as plt

        plt.ion()
        image = plt.imshow(tensor_to_array(wrist_camera.read().rgb))
        plt.axis("off")

    # The arm sweeps over the boxes, the wrist camera seeing them and the floor far around them
    horizon = 400 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        qpos = qpos_home.copy()
        qpos[0] = 0.6 * np.sin(2.0 * np.pi * i / 200)
        franka.control_dofs_position(qpos)
        scene.step()
        rgb = tensor_to_array(wrist_camera.read().rgb)
        if args.vis:
            image.set_data(rgb)
            plt.pause(0.001)


if __name__ == "__main__":
    main()
