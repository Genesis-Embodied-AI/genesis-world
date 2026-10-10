"""Keyboard teleoperation of a Franka arm over two stacked cloths draped on a grid of cubes, all simulated by the
MochiSolver: the arm tracks an interactively moved target frame by inverse kinematics, and the gripper grabs the
cloth. The articulated arm, the shells (with self-contact and cloth-on-cloth contact through their point-cloud
colliders) and the penalty contact enter one implicit Newton solve, so no coupler is involved. The mochi counterpart
of the IPC robot-cloth teleoperation example.

To grab the cloth, plunge the open fingers into the slack fabric between the cubes so a fold tents up between the
pads, then close: pinching a taut single ply from above slips out of the smooth penalty contact, a gathered fold
holds.

Keyboard Controls:
Up	- Move Forward (North)
Down	- Move Backward (South)
Left	- Move Left (West)
Right	- Move Right (East)
k	- Move Up
j	- Move Down
n/m	- Yaw Left/Right (Rotate around Z axis)
u/o	- Pitch Up/Down (Rotate around Y axis)
l/;	- Roll Left/Right (Rotate around X axis)
\\	- Reset the target frame
space	- Press to close gripper, release to open gripper
esc	- Quit
"""

import argparse
import os

import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

DELTA_POS = 0.003
DELTA_ROT = 0.02


def sheet_mesh(path, n_cells, size):
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            faces.append([a, a + 1, a + n_cells + 2])
            faces.append([a, a + n_cells + 2, a + n_cells + 1])
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Quadrants CPU threads; more than 1 parallelizes the solver loops of large deformable scenes",
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64", cpu_threads=args.cpu_threads)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=4),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.0, 1.5),
            camera_lookat=(0.5, 0.0, 0.2),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, 0.005)),
        # Rubber-pad friction (the FinRay pads of the Schunk model use the same value).
        material=gs.materials.Mochi.Rigid(friction=3.0, viscous_friction=1.0),
    )

    # Two stacked cloth sheets over the cubes; light cloth takes a softer contact stiffness (see the examples README).
    cloth_kwargs = {
        "nu": 0.3,
        "rho": 200.0,
        "thickness": 1e-3,
        "friction": 1.0,
        "penalty_coefficient": 1e7,
        "self_contact": True,
    }
    scene.add_entity(
        gs.morphs.Mesh(file=sheet_mesh("/tmp/mochi_teleop_sheet20.obj", 20, 0.5), pos=(0.5, 0.0, 0.1)),
        material=gs.materials.Mochi.Shell(E=6e4, collider_radius=1.2e-2, **cloth_kwargs),
        surface=gs.surfaces.Plastic(color=(0.3, 0.1, 0.8, 1.0)),
    )
    scene.add_entity(
        gs.morphs.Mesh(file=sheet_mesh("/tmp/mochi_teleop_sheet12.obj", 12, 0.3), pos=(0.5, 0.0, 0.14)),
        material=gs.materials.Mochi.Shell(E=6e4, collider_radius=1.2e-2, **cloth_kwargs),
        surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0)),
    )

    # 16 fixed rigid cubes uniformly distributed under the cloth (4x4 grid).
    cube_size = 0.05
    grid_spacing = 0.15
    for i in range(4):
        for j in range(4):
            scene.add_entity(
                gs.morphs.Box(
                    pos=((i + 1.7) * grid_spacing, (j - 1.5) * grid_spacing, 0.5 * cube_size),
                    size=(cube_size, cube_size, cube_size),
                    fixed=True,
                ),
                material=gs.materials.Mochi.Rigid(friction=0.5),
                surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 0.8)),
            )

    motor_dofs_idx = slice(0, 7)
    finger_dofs_idx = slice(7, 9)
    ee_link = franka.get_link("hand")

    target_init_pos = np.array([0.5, 0.0, 0.6], dtype=gs.np_float)
    target_init_quat = gu.xyz_to_quat(np.array([0.0, 180.0, 0.0], dtype=gs.np_float), degrees=True)
    target_pos, target_quat = target_init_pos.copy(), target_init_quat.copy()

    scene.build()

    franka.set_dofs_kp(500.0, dofs_idx_local=finger_dofs_idx)
    franka.set_dofs_kv(50.0, dofs_idx_local=finger_dofs_idx)

    qpos = (2.2116, -1.5328, -0.7347, -1.7235, -1.3377, 0.7519, -1.4410, 0.04, 0.04)
    franka.set_qpos(qpos)
    franka.control_dofs_position(qpos)

    target_ik = scene.draw_debug_frame(
        T=gu.trans_quat_to_T(target_pos, target_quat),
        axis_length=0.15,
        origin_size=0.01,
        axis_radius=0.007,
    )
    scene.viewer.update(force=True)

    if scene.viewer is None:
        gs.logger.warning("Viewer is not active. Keyboard input requires the Genesis viewer.")
        return

    gripper_close = np.array(False, dtype=gs.np_bool)
    is_running = True

    def move(dpos_xyz: tuple[float, float, float]):
        target_pos[:] += dpos_xyz

    def rotate(axis_idx: int, delta: float):
        delta_xyz = np.zeros(3, dtype=gs.np_float)
        delta_xyz[axis_idx] = delta
        delta_quat = gu.xyz_to_quat(delta_xyz)
        target_quat[:] = gu.transform_quat_by_quat(target_quat, delta_quat)

    def reset_scene():
        target_pos[:], target_quat[:] = target_init_pos, target_init_quat
        pose = gu.trans_quat_to_T(target_pos, target_quat)
        scene.update_debug_objects((target_ik,), (pose,))
        qpos = franka.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motor_dofs_idx)
        franka.control_dofs_position(qpos[motor_dofs_idx], dofs_idx_local=motor_dofs_idx)

    def set_gripper(close: bool):
        gripper_close[()] = close

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-DELTA_POS, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((DELTA_POS, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -DELTA_POS, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, DELTA_POS, 0),)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, DELTA_POS),)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -DELTA_POS),)),
        Keybind("yaw_left", Key.N, KeyAction.HOLD, callback=rotate, args=(2, DELTA_ROT)),
        Keybind("yaw_right", Key.M, KeyAction.HOLD, callback=rotate, args=(2, -DELTA_ROT)),
        Keybind("pitch_up", Key.U, KeyAction.HOLD, callback=rotate, args=(1, DELTA_ROT)),
        Keybind("pitch_down", Key.O, KeyAction.HOLD, callback=rotate, args=(1, -DELTA_ROT)),
        Keybind("roll_left", Key.L, KeyAction.HOLD, callback=rotate, args=(0, DELTA_ROT)),
        Keybind("roll_right", Key.SEMICOLON, KeyAction.HOLD, callback=rotate, args=(0, -DELTA_ROT)),
        Keybind("reset_scene", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_scene),
        Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=set_gripper, args=(True,)),
        Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=set_gripper, args=(False,)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )

    try:
        while is_running and scene.viewer.is_alive():
            pose = gu.trans_quat_to_T(target_pos, target_quat)
            scene.update_debug_objects((target_ik,), (pose,))

            qpos = franka.inverse_kinematics(
                link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motor_dofs_idx
            )
            franka.control_dofs_position(qpos[motor_dofs_idx], motor_dofs_idx)

            if gripper_close[()]:
                # Stop the pads a few millimetres apart: squeezing further ejects the pinched fold.
                franka.control_dofs_position(0.002, dofs_idx_local=finger_dofs_idx)
            else:
                franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs_idx)

            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
