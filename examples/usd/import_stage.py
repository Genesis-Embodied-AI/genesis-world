import argparse
import os

import numpy as np
from huggingface_hub import snapshot_download

import genesis as gs
from genesis.utils.misc import tensor_to_array
import genesis.utils.geom as gu


class JointAnimator:
    """
    A simple JointAnimator to animate the joints' positions of the scene.

    It uses the sin function to interpolate between the lower and upper limits of the joints.
    """

    def __init__(self, scene: gs.Scene):
        self.rigid_solver = scene.sim.rigid_solver
        self.joint_lower, self.joint_upper = map(tensor_to_array, self.rigid_solver.get_dofs_limit())

        init_positions = tensor_to_array(self.rigid_solver.get_dofs_position())
        normalized_init_pos = np.where(
            (self.joint_upper - self.joint_lower) > gs.EPS,
            2.0 * (init_positions - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0,
            0.0,
        )
        self.init_phase = np.arcsin(normalized_init_pos)

        self.rigid_solver.set_dofs_kp(gu.default_dofs_kp(self.rigid_solver.n_dofs))

    def animate(self, scene: gs.Scene):
        t = tensor_to_array(scene.get_time())
        theta = np.pi * t + self.init_phase
        target = (self.joint_upper + self.joint_lower + (self.joint_upper - self.joint_lower) * np.sin(theta)) / 2
        self.rigid_solver.control_dofs_position(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=5000 if "PYTEST_VERSION" not in os.environ else 1,
        help="Number of simulation steps",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, 2.0, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        show_FPS=False,
    )

    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="c50bfe3e354e105b221ef4eb9a79504650709dd2",
        allow_patterns="usd/Refrigerator055/*",
        max_workers=1,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    entities = scene.add_stage(
        morph=gs.morphs.USD(
            file=f"{asset_path}/usd/Refrigerator055/Refrigerator055.usd",
            pos=(0, 0, 0.9),
            euler=(0, 0, 180),
        ),
    )

    scene.build()

    joint_animator = JointAnimator(scene)

    for _ in range(args.steps):
        joint_animator.animate(scene)
        scene.step()


if __name__ == "__main__":
    main()
