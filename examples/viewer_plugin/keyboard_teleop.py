"""
Keyboard Controls:
Up	- Move Forward (North)
Down	- Move Backward (South)
Left	- Move Left (West)
Right	- Move Right (East)
n	- Move Up
m	- Move Down
j	- Rotate Counterclockwise
k	- Rotate Clockwise
u	- Reset Scene
space	- Press to close gripper, release to open gripper
esc	- Quit

Plus all default viewer controls (press 'i' to see them)
"""

import os
import random

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

if __name__ == "__main__":
    gs.init(precision="32", logging_level="info", backend=gs.cpu)
    np.set_printoptions(precision=7, suppress=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=4,
            gravity=(0, 0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            constraint_timeconst=0.01,
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
            camera_fov=50,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    robot = scene.add_entity(
        material=gs.materials.Rigid(
            gravity_compensation=1,
        ),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            euler=(0, 0, 0),
        ),
    )

    cube = scene.add_entity(
        material=gs.materials.Rigid(
            rho=300,
        ),
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.07),
            size=(0.04, 0.04, 0.04),
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 1, 0.5),
        ),
    )

    target = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(
            color=(1, 0.5, 0.5, 1),
        ),
    )

    scene.build()

    # Initialize robot control state
    robot_init_pos = np.array([0.5, 0, 0.55])
    robot_init_quat = gu.xyz_to_quat(np.array([0, np.pi, 0]))  # Rotation around Y axis

    # Get DOF indices
    n_dofs = robot.n_dofs
    motors_dof = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    ee_link = robot.get_link("hand")

    # Initialize target pose
    target_pos = robot_init_pos.copy()
    target_quat = robot_init_quat.copy()

    # Control parameters
    dpos = 0.002
    drot = 0.01

    # Helper function to reset robot
    def reset_robot():
        """Reset robot and cube to initial positions."""
        target_pos[:] = robot_init_pos.copy()
        target_quat[:] = robot_init_quat.copy()
        target.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q[:-2], motors_dof)

        # Randomize cube position
        cube.set_pos((random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), 0.05))
        random_angle = random.uniform(0, np.pi * 2)
        cube.set_quat(gu.xyz_to_quat(np.array([0, 0, random_angle])))

    # Initialize robot pose
    reset_robot()

    # Robot teleoperation callback functions
    def move(dpos: tuple[float, float, float]):
        target_pos[:] += np.array(dpos, dtype=gs.np_float)

    def rotate(drot: float):
        drot_quat = gu.xyz_to_quat(np.array([0, 0, drot]))
        target_quat[:] = gu.transform_quat_by_quat(target_quat, drot_quat)

    def toggle_gripper(close: bool = True):
        pos = -1.0 if close else 1.0
        robot.control_dofs_force(np.array([pos, pos]), fingers_dof)

    is_running = True

    def stop():
        global is_running
        is_running = False

    # Register robot teleoperation keybindings
    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
        Keybind("rotate_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(drot,)),
        Keybind("rotate_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(-drot,)),
        Keybind("reset_scene", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_robot),
        Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=toggle_gripper, args=(True,)),
        Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=toggle_gripper, args=(False,)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )

    try:
        while is_running:
            # Update target entity visualization
            target.set_qpos(np.concatenate([target_pos, target_quat]))

            # Control arm with inverse kinematics
            q, err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
            robot.control_dofs_position(q[:-2], motors_dof)

            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
