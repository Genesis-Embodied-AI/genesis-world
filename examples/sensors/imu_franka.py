import argparse
import os

import numpy as np
from tqdm import tqdm

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-t", "--seconds", type=float, default=3, help="Number of seconds to simulate")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    end_effector = franka.get_link("hand")
    motors_dof = (0, 1, 2, 3, 4, 5, 6)

    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=franka.idx,
            link_idx_local=end_effector.idx_local,
            pos_offset=(0.0, 0.0, 0.15),
            # noise parameters
            acc_cross_axis_coupling=(0.0, 0.01, 0.02),
            gyro_cross_axis_coupling=(0.03, 0.04, 0.05),
            acc_noise=(0.01, 0.01, 0.01),
            gyro_noise=(0.01, 0.01, 0.01),
            acc_random_walk=(0.001, 0.001, 0.001),
            gyro_random_walk=(0.001, 0.001, 0.001),
            delay=0.01,
            jitter=0.01,
            draw_debug=True,
        )
    )
    if args.vis:
        xyz = ("x", "y", "z")
        labels = {"lin_acc": xyz, "true_lin_acc": xyz, "ang_vel": xyz, "true_ang_vel": xyz}

        def data_func():
            data = imu.read()
            true_data = imu.read_ground_truth()
            return {
                "lin_acc": data.lin_acc,
                "true_lin_acc": true_data.lin_acc,
                "ang_vel": data.ang_vel,
                "true_ang_vel": true_data.ang_vel,
            }

        if IS_PYQTGRAPH_AVAILABLE:
            scene.add_recorder(
                data_func,
                gs.recorders.PyQtLinePlot(labels=labels, title="IMU Ground Truth Data"),
            )
        elif IS_MATPLOTLIB_AVAILABLE:
            gs.logger.info("pyqtgraph not found, falling back to matplotlib.")
            scene.add_recorder(
                data_func,
                gs.recorders.MPLLinePlot(labels=labels, title="IMU Ground Truth Data"),
            )
        else:
            print("matplotlib or pyqtgraph not found, skipping real-time plotting.")

    scene.add_recorder(
        data_func=lambda: imu.read()._asdict(),
        rec_options=gs.recorders.NPZFile(filename="out/imu_data.npz"),
    )

    scene.build()

    franka.set_dofs_kp(
        np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0]),
    )
    franka.set_dofs_kv(
        np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0]),
    )
    franka.set_dofs_force_range(
        np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -100.0, -100.0]),
        np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0]),
    )

    circle_center = np.array([0.4, 0.0, 0.5])
    circle_radius = 0.15
    rate = np.deg2rad(2.0)

    try:
        steps = int(args.seconds / args.dt) if "PYTEST_VERSION" not in os.environ else 5
        for i in tqdm(range(steps)):
            scene.step()

            # control franka to move in a circle and draw the target positions
            pos = circle_center + np.array([np.cos(i * rate), np.sin(i * rate), 0]) * circle_radius
            qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=pos,
                quat=np.array([0.0, 1.0, 0.0, 0.0]),
            )
            franka.control_dofs_position(qpos[:-2], motors_dof)
            scene.draw_debug_sphere(pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5))
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        scene.stop_recording()

        print("Ground truth data:")
        print(imu.read_ground_truth())
        print("Measured data:")
        print(imu.read())


if __name__ == "__main__":
    main()
