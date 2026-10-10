"""An articulated arm simulated by the MochiSolver: a Franka Panda on the ground tracks joint targets with PD control.

Joint drives, joint limits, armature and damping enter the same implicit Newton solve as the rigid-body inertia and
the contact penalty, so stiff position gains remain stable at a 10 ms step.
"""

import argparse

import numpy as np
import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81), substeps=2),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.0, -2.0, 1.5), camera_lookat=(0.0, 0.0, 0.5)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Mochi.Rigid(friction=1.0, viscous_friction=1.0),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        # material=gs.materials.Mochi.Rigid(friction=1.0, viscous_friction=1.0),
        material=gs.materials.Mochi.Elastic(
            E=2e4, nu=0.4, rho=800.0, stiffness_damping=1e-3, friction=1.0, viscous_friction=1.0
        ),
        surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9)),
    )
    if args.record:
        cam = scene.add_camera(
            res=(1280, 720),
            pos=(2.0, -2.0, 1.5),
            lookat=(0.0, 0.0, 0.5),
        )
    scene.build()

    if args.record:
        cam.start_recording(save_to_filename="articulated_arm.mp4", fps=30)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    # end_effector = franka.get_link("hand")

    # init
    franka.set_qpos((-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04))

    # hold
    # TODO: AttributeError: 'MochiOptions' object has no attribute 'IK_max_targets'
    # qpos = franka.inverse_kinematics(link=end_effector, pos=(0.65, 0.0, 0.13), quat=(0, 1, 0, 0))
    qpos = torch.tensor([-1.0095, 1.5617, 1.3595, -1.6830, -1.5855, 1.7817, 1.4595, 0.0400, 0.0400])
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(50):
        scene.step()
        if args.record:
            cam.render()

    # grasp
    for i in range(50):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()
        if args.record:
            cam.render()

    # lift
    # TODO: AttributeError: 'MochiOptions' object has no attribute 'IK_max_targets'
    # qpos = franka.inverse_kinematics(link=end_effector, pos=(0.65, 0.0, 0.3), quat=(0, 1, 0, 0))
    qpos = torch.tensor([-1.0418, 1.3805, 1.5885, -1.7021, -1.3798, 1.5783, 1.4467, 0.0188, 0.0188])
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(args.n_steps):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()
        if args.record:
            cam.render()

    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
