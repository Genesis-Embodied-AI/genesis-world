"""
Keyboard Controls:
Up	- Move Forward (North)
Down	- Move Backward (South)
Left	- Move Left (West)
Right	- Move Right (East)
n	- Move Up
m	- Move Down
j/k	- Yaw Left/Right (Rotate around Z axis)
i/o	- Pitch Up/Down (Rotate around Y axis)
l/;	- Roll Left/Right (Rotate around X axis)
u	- Reset Scene
space	- Press to close gripper, release to open gripper
esc	- Quit
"""

import argparse
import os

import numpy as np
from huggingface_hub import snapshot_download

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

DELTA_POS = 0.003
DELTA_ROT = 0.02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coup-type",
        type=str,
        default="two_way_soft_constraint",
        choices=["two_way_soft_constraint", "external_articulation"],
        help="How the articulated robot is coupled to the IPC solver",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            newton_tolerance=1e-1,
            newton_translation_tolerance=1,
            newton_semi_implicit_enable=False,  # Must be false to avoid time stealing artifact
            n_linesearch_iterations=8,
            linesearch_report_energy=False,
            linear_system_tolerance=1e-3,
            contact_enable=True,
            contact_d_hat=0.001,
            contact_resistance=1e7,
            constraint_strength_translation=100.0,
            constraint_strength_rotation=100.0,
            enable_rigid_rigid_contact=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.0, 1.5),
            camera_lookat=(0.5, 0.0, 0.2),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    # Add flat floor
    scene.add_entity(gs.morphs.Plane())

    # Add Franka robot
    franka_material_kwargs = dict(
        coup_type=args.coup_type,
    )
    if args.coup_type == "two_way_soft_constraint":
        franka_material_kwargs["coup_links"] = ("left_finger", "right_finger")
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
            pos=(0.0, 0.0, 0.005),
        ),
        material=gs.materials.Rigid(**franka_material_kwargs),
    )

    # Add cloths
    cloth_asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="8aa8fcd60500b9f3a36c356080224bdb1be9ee59",
        allow_patterns="/IPC/grid20x20.obj",
        max_workers=1,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{cloth_asset_path}/IPC/grid20x20.obj",
            scale=0.5,
            pos=(0.5, 0.0, 0.1),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=6e4,
            nu=0.49,
            rho=200,
            friction_mu=0.5,
            thickness=0.001,
            bending_stiffness=10.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.1, 0.8, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{cloth_asset_path}/IPC/grid20x20.obj",
            scale=0.3,
            pos=(0.5, 0.0, 0.14),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=6e4,
            nu=0.49,
            rho=200,
            friction_mu=0.5,
            thickness=0.001,
            bending_stiffness=40.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    # Add 16 rigid cubes uniformly distributed under the cloth (4x4 grid)
    cube_size = 0.05
    cube_height = 0.02501
    grid_spacing = 0.15
    for i in range(4):
        for j in range(4):
            x = (i + 1.7) * grid_spacing
            y = (j - 1.5) * grid_spacing
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(x, y, cube_height),
                    size=(cube_size, cube_size, cube_size),
                    fixed=True,
                ),
                material=gs.materials.Rigid(
                    rho=500,
                    coup_friction=0.5,
                    coup_type="ipc_only",
                ),
                surface=gs.surfaces.Plastic(
                    color=(0.8, 0.3, 0.2, 0.8),
                ),
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

    # Setting initial configuration is not supported by coupling mode "external_articulation"
    if args.coup_type != "external_articulation":
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
                # FIXME: The grip closes on position because force control on the fingers behaves erratically here.
                franka.control_dofs_position(-0.03, dofs_idx_local=finger_dofs_idx)
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
