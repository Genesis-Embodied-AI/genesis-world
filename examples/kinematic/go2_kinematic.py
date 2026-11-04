"""
Example: KinematicEntity as a ghost reference motion.

Creates a Go2 quadruped as a physics-simulated entity alongside a second Go2 loaded as a KinematicEntity
(visualization-only ghost). The kinematic entity follows a simple sinusoidal joint trajectory while the physics robot
is free to fall and interact with the ground. This demonstrates how KinematicEntity can display a reference motion
without affecting simulation speed or physics.
"""

import argparse
import math
import os

import genesis as gs


AMPLITUDE = 0.3
FREQ = 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.42),
        ),
    )

    # The ghost is a kinematic entity: it visualizes the commanded pose without taking part in the dynamics.
    ghost = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.42),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.4, 0.7, 1.0),
            opacity=0.5,
        ),
    )

    scene.build()

    joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    #                    FR                FL                RR                RL
    joint_angles = [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5]

    dofs_idx = [robot.get_joint(name).dofs_idx_local[0] for name in joint_names]

    robot.set_dofs_position(joint_angles, dofs_idx)
    ghost.set_dofs_position(joint_angles, dofs_idx)

    for step in range(500 if "PYTEST_VERSION" not in os.environ else 5):
        t = step * scene.options.sim.dt

        # Sinusoidal reference trajectory for the ghost
        offset = AMPLITUDE * math.sin(2.0 * math.pi * FREQ * t)

        ref_angles = joint_angles.copy()

        # Oscillate thigh joints (indices 1, 4, 7, 10)
        ref_angles[1] += offset
        ref_angles[4] += offset
        ref_angles[7] -= offset
        ref_angles[10] -= offset

        ghost.set_dofs_position(ref_angles, dofs_idx)

        scene.step()


if __name__ == "__main__":
    main()
