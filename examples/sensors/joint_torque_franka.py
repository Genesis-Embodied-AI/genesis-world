"""
Joint torque sensing on a Franka Emika Panda using the JointTorqueSensor.

The sensor measures the generalized effort at each actuator output shaft:

    actuator_force = tau_control - armature * qacc + tau_frictionloss + tau_damping

In free space the reading is approximately the gravity + Coriolis load; when the arm presses against the wall box it
also captures the contact reaction.

Three live plots show, for joints J0, J1 and J2, the comparison between the control torque, the sensor reading and
their difference.
"""

import argparse
import os

import torch
from tqdm import tqdm

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-t", "--seconds", type=float, default=5.0, help="Number of seconds to simulate")
    args = parser.parse_args()

    steps = int(args.seconds / args.dt) if "PYTEST_VERSION" not in os.environ else 5

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
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )

    # Fixed wall: the robot presses its end-effector into it in the second half.
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.60, 0.0, 0.50),
            size=(0.04, 0.30, 0.20),
            fixed=True,
        ),
    )

    end_effector = franka.get_link("hand")
    motors_dof = (0, 1, 2, 3, 4, 5, 6)

    joint_torque = scene.add_sensor(
        gs.sensors.JointTorque(
            entity_idx=franka.idx,
            dofs_idx_local=motors_dof,
        )
    )

    # One subplot per representative joint (J0, J1, J2), each showing control torque / sensor torque / difference.
    plotted_joints = range(3)
    channel_labels = ("tau_ctrl", "tau_sensor", "difference")
    labels = {f"J{i}": channel_labels for i in plotted_joints}

    def data_func():
        tau_sensor = joint_torque.read()
        tau_ctrl = franka.get_dofs_control_force(motors_dof)
        return {f"J{i}": torch.stack([tau_ctrl[i], tau_sensor[i], tau_sensor[i] - tau_ctrl[i]]) for i in plotted_joints}

    if args.vis:
        if IS_PYQTGRAPH_AVAILABLE:
            scene.add_recorder(
                data_func,
                gs.recorders.PyQtLinePlot(labels=labels, title="JointTorqueSensor - Franka"),
            )
        elif IS_MATPLOTLIB_AVAILABLE:
            gs.logger.info("pyqtgraph not found, falling back to matplotlib.")
            scene.add_recorder(
                data_func,
                gs.recorders.MPLLinePlot(labels=labels, history_length=steps, title="JointTorqueSensor - Franka"),
            )
        else:
            print("matplotlib or pyqtgraph not found, skipping real-time plotting.")

    scene.build()

    franka.set_dofs_armature(1.0)
    franka.set_dofs_frictionloss(10.0)

    franka.set_dofs_kp([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0])
    franka.set_dofs_kv([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0])
    franka.set_dofs_force_range(
        [-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -100.0, -100.0],
        [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0],
    )

    qpos_hold = franka.inverse_kinematics(link=end_effector, pos=[0.45, 0.0, 0.50], quat=[0.0, 1.0, 0.0, 0.0])
    franka.set_qpos(qpos_hold)

    qpos_push = franka.inverse_kinematics(link=end_effector, pos=[0.70, 0.0, 0.50], quat=[0.0, 1.0, 0.0, 0.0])

    try:
        contact_step = steps // 2
        for i in tqdm(range(steps)):
            target = qpos_hold if i < contact_step else qpos_push
            franka.control_dofs_position(target[:-2], motors_dof)
            scene.step()
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
        scene.stop_recording()


if __name__ == "__main__":
    main()
