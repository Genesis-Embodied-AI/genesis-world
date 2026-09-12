"""
Interactive demo of tactile sensors attached to Franka Panda grippers and matplotlib visualization.
Sensor types: ContactDepthProbe, ContactProbe, ElastomerTaxel, KinematicTaxel, ProximityTaxel.

Note that the sensor readings here have not been calibrated to any units, and is purely for visualization purposes.
"""

import argparse
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.vis.keybindings import Key, KeyAction, Keybind

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.sensors.base_sensor import Sensor

OBJ_DENSITY = 300
OBJ_SIZE = 0.04
CUBE_INIT_XY = (0.5, 0.1)
SPHERE_INIT_XY = (0.5, -0.1)
ROBOT_INIT_HEIGHT = 0.18

DPOS = 0.004
DROT = 0.02


def _add_tactile_sensor(
    scene: gs.Scene,
    entity: "RigidEntity",
    link_idx_local: int,
    sensor_type: str,
    probe_local_pos: np.ndarray,
    probe_normal: tuple[float, float, float],
    track_link_idx: tuple[int, ...],
) -> "Sensor":
    common = dict(
        entity_idx=entity.idx,
        link_idx_local=link_idx_local,
        draw_debug=True,
    )
    if sensor_type == "elastomer":
        return scene.add_sensor(
            gs.sensors.ElastomerTaxel(
                probe_local_pos=probe_local_pos,
                probe_local_normal=probe_normal,
                probe_radius=0.002,
                track_link_idx=track_link_idx,
                lambda_d=5000.0,
                lambda_s=4000.0,
                dilate_scale=1.0,
                shear_scale=1.0,
                n_sample_points=1000,
                normal_exponent=1.0,
                compressibility=0.8,
                debug_point_cloud_radius=0.001,
                **common,
            )
        )
    probe_local_pos = probe_local_pos.reshape(-1, 3)  # flatten
    if sensor_type == "depth":
        return scene.add_sensor(
            gs.sensors.ContactDepthProbe(
                probe_local_pos=probe_local_pos,
                probe_radius=0.002,
                **common,
            )
        )
    if sensor_type == "kinematic":
        return scene.add_sensor(
            gs.sensors.KinematicTaxel(
                probe_local_pos=probe_local_pos,
                probe_radius=0.002,
                normal_stiffness=2000.0,
                normal_damping=1.0,
                normal_exponent=1.5,
                shear_scalar=1.0,
                twist_scalar=1.0,
                **common,
            )
        )
    if sensor_type == "proximity":
        return scene.add_sensor(
            gs.sensors.ProximityTaxel(
                probe_local_pos=probe_local_pos,
                track_link_idx=track_link_idx,
                probe_radius=0.01,
                n_sample_points=1000,
                stiffness=150.0,
                shear_coupling=1.0,
                twist_scalar=1.0,
                probe_local_normal=probe_normal,
                **common,
            )
        )

    raise ValueError(sensor_type)


def _force_torque_field(sensors: "tuple[Sensor, ...]") -> tuple[torch.Tensor, torch.Tensor]:
    readings = [sensor.read() for sensor in sensors]
    return torch.stack([r.force for r in readings]), torch.stack([r.torque for r in readings])


def _plot_tactile_sensor(
    scene: gs.Scene,
    sensor_type: str,
    labels: tuple[str, ...],
    sensors: "tuple[Sensor, ...]",
    plot_normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> None:
    if not IS_MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available; skipping plot setup.")
        return

    positions = sensors[0].probe_local_pos.reshape(-1, 3)
    if sensor_type == "elastomer":
        # ElastomerTaxel read() is grid-shaped (ny, nx, 3); flatten each finger to (N, 3) before stacking.
        scene.add_recorder(
            lambda: torch.stack([sensor.read().reshape(-1, 3) for sensor in sensors]),
            gs.recorders.MPLVectorFieldPlot(
                title="ElastomerTaxel marker displacements",
                positions=positions,
                normal=plot_normal,
                scale_factor=0.1,
                max_magnitude=0.08,
                subplot_titles=labels,
            ),
        )
    elif sensor_type == "kinematic":
        scene.add_recorder(
            lambda: _force_torque_field(sensors),
            gs.recorders.MPLVectorFieldPlot(
                title="KinematicTaxel force + twist",
                positions=positions,
                normal=plot_normal,
                scale_factor=0.01,
                max_magnitude=1.0,
                subplot_titles=labels,
                twist_scale_factor=0.002,
                twist_max_magnitude=2.0,
            ),
        )
    elif sensor_type == "proximity":
        scene.add_recorder(
            lambda: _force_torque_field(sensors),
            gs.recorders.MPLVectorFieldPlot(
                title="ProximityTaxel force + twist",
                positions=positions,
                normal=plot_normal,
                scale_factor=10.0,
                max_magnitude=1.0,
                subplot_titles=labels,
                twist_scale_factor=0.001,
                twist_max_magnitude=1.0,
            ),
        )
    elif sensor_type == "depth":
        scene.add_recorder(
            lambda: tuple(sensor.read().max() for sensor in sensors),
            gs.recorders.MPLLinePlot(
                labels=labels,
                x_label="step",
                y_label="depth",
                history_length=200,
                title="ContactDepthProbe max depth",
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Franka fingertip tactile with selectable sensor type")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument(
        "--sensor",
        choices=("elastomer", "depth", "kinematic", "proximity"),
        default="elastomer",
        help="Tactile sensor implementation",
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-t", "--seconds", type=float, default=4.0, help="Seconds to simulate (headless mode)")
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_timeconst=0.01,
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=(OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
            pos=(*CUBE_INIT_XY, OBJ_SIZE / 2),
        ),
        material=gs.materials.Rigid(
            rho=OBJ_DENSITY,
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 1.0, 0.5),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=OBJ_SIZE / 2,
            pos=(*SPHERE_INIT_XY, OBJ_SIZE / 2),
        ),
        material=gs.materials.Rigid(
            rho=OBJ_DENSITY,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.5, 0.5),
        ),
    )

    probe_normal = (0.0, -1.0, 0.0)
    probe_local_pos = gu.generate_grid_points_on_plane(
        lo=(-0.006, 0.0, 0.04),
        hi=(0.008, 0.0, 0.05),
        normal=probe_normal,
        nx=8,
        ny=8,
    )
    track_idx = (int(cube.base_link_idx), int(sphere.base_link_idx))

    left = _add_tactile_sensor(
        scene, franka, franka.get_link("left_finger").idx_local, args.sensor, probe_local_pos, probe_normal, track_idx
    )
    right = _add_tactile_sensor(
        scene, franka, franka.get_link("right_finger").idx_local, args.sensor, probe_local_pos, probe_normal, track_idx
    )
    if args.vis and "PYTEST_VERSION" not in os.environ:
        _plot_tactile_sensor(scene, args.sensor, ("left", "right"), (left, right), plot_normal=probe_normal)

    scene.build()

    n_dofs = franka.n_dofs
    motor_dofs_idx = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    ee_link = franka.get_link("hand")

    # Headless starts the arm right above the cube it will grasp; the GUI starts it over the sphere so both objects
    # are within easy teleoperation reach.
    init_xy = CUBE_INIT_XY if not args.vis else SPHERE_INIT_XY
    target_init_pos = np.array((*init_xy, ROBOT_INIT_HEIGHT), dtype=gs.np_float)
    target_init_quat = gu.euler_to_quat((0.0, 180.0, 0.0))
    target_pos, target_quat = target_init_pos.copy(), target_init_quat.copy()

    target_ik = None
    if args.vis:
        target_ik = scene.draw_debug_frame(
            T=gu.trans_quat_to_T(target_pos, target_quat),
            axis_length=0.15,
            origin_size=0.01,
            axis_radius=0.007,
        )
        scene.viewer.update(force=True)

    def toggle_gripper(close: bool):
        pos = -0.03 if close else 0.04
        franka.control_dofs_position(pos, dofs_idx_local=fingers_dof)

    def reset_robot():
        target_pos[:], target_quat[:] = target_init_pos, target_init_quat
        if target_ik is not None:
            scene.update_debug_objects((target_ik,), (gu.trans_quat_to_T(target_pos, target_quat),))

        qpos = franka.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motor_dofs_idx)
        franka.set_qpos(qpos[motor_dofs_idx], motor_dofs_idx)
        toggle_gripper(False)

        cube.set_pos(cube.base_link.desc.pos)
        sphere.set_pos(sphere.base_link.desc.pos)

    reset_robot()

    is_running = True

    if args.vis:

        def move(dpos_xyz: tuple[float, float, float]):
            target_pos[:] += dpos_xyz

        def rotate(drot: float):
            drot_quat = gu.xyz_to_quat(np.array([0, 0, drot]))
            target_quat[:] = gu.transform_quat_by_quat(target_quat, drot_quat)

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-DPOS, 0, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((DPOS, 0, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -DPOS, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, DPOS, 0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, DPOS),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -DPOS),)),
            Keybind("rotate_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(DROT,)),
            Keybind("rotate_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(-DROT,)),
            Keybind("reset_scene", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_robot),
            Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=toggle_gripper, args=(True,)),
            Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=toggle_gripper, args=(False,)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )

    grasp_pos = np.array((*CUBE_INIT_XY, 0.125), dtype=gs.np_float)
    lift_pos = np.array((*CUBE_INIT_XY, 0.30), dtype=gs.np_float)

    i_step = 0
    try:
        while is_running:
            if not args.vis:
                # Automatic pickup: hover over the cube, descend around it, squeeze, then lift.
                if i_step < 40:
                    target_pos[:] = target_init_pos
                    toggle_gripper(close=False)
                elif i_step < 100:
                    target_pos[:] = grasp_pos
                    toggle_gripper(close=False)
                elif i_step < 150:
                    target_pos[:] = grasp_pos
                    toggle_gripper(close=True)
                else:
                    target_pos[:] = lift_pos
                    toggle_gripper(close=True)

            if target_ik is not None:
                scene.update_debug_objects((target_ik,), (gu.trans_quat_to_T(target_pos, target_quat),))

            qpos = franka.inverse_kinematics(
                link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motor_dofs_idx
            )
            franka.control_dofs_position(qpos[motor_dofs_idx], motor_dofs_idx)

            scene.step()
            i_step += 1

            t = scene.get_time()
            for label, sensor in zip(("left", "right"), (left, right)):
                data = sensor.read()
                if args.sensor == "elastomer":
                    magnitude = data.norm(dim=-1)
                    if magnitude.max() > gs.EPS:
                        print(f"({label}) t={t:.2f}s  max|displacement|={magnitude.max():.5f}")
                elif args.sensor == "depth":
                    max_depth = data.max()
                    if max_depth > gs.EPS:
                        print(f"({label}) t={t:.2f}s  max depth={max_depth:.4f}")
                elif args.sensor in ("kinematic", "proximity"):
                    force_mag = data.force.norm(dim=-1).max()
                    # |twist| is the plotted torque about the view normal; print it to calibrate twist_scale_factor.
                    twist_mag = (data.torque @ data.torque.new_tensor(probe_normal)).abs().max()
                    if force_mag > gs.EPS:
                        print(f"({label}) t={t:.2f}s  max|F|={force_mag:.4f}  max|twist|={twist_mag:.5f}")

            if "PYTEST_VERSION" in os.environ:
                break
            if not args.vis and t >= args.seconds:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
