"""A Franka arm pressing the middle of a 64-node rope lying on the ground.

This is the `rope_arm` benchmark scene (the reinforcement-learning shaped acceptance scene of
`tests/mochi/benchmark`) run as a standalone demo: the hand descends onto the rope under PD joint control while the
fingertips press with 1 N each, pinning the rope against the ground. The discrete elastic rod (stretching, bending,
twisting), the articulated arm and the penalty contact enter one implicit Newton solve (mochi's default budget of 4
iterations per step); the rope collides with the ground and the fingertips through its centerline samples.
"""

import argparse

import numpy as np

import genesis as gs

FRANKA_START_QPOS = (-1.0418, 1.3805, 1.5885, -1.7021, -1.3798, 1.5783, 1.4467, 0.04, 0.04)
FRANKA_TARGET_QPOS = (-1.0095, 1.5617, 1.3595, -1.6830, -1.5855, 1.7817, 1.4595, 0.04, 0.04)
WORK_XY = (0.65, 0.0)
ROPE_NODES, ROPE_LENGTH, ROPE_RADIUS = 64, 0.5, 5e-3
ROPE_E, ROPE_G, ROPE_RHO = 1e7, 1e7, 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=4),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.5, -0.8, 0.7), camera_lookat=(0.65, 0.0, 0.15)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Mochi.Rigid(friction=1.0, viscous_friction=1.0),
    )
    # The rope lies on the ground along x through the work point; the descending hand presses its middle.
    xs = WORK_XY[0] + np.linspace(-0.5 * ROPE_LENGTH, 0.5 * ROPE_LENGTH, ROPE_NODES)
    points = np.stack([xs, np.full(ROPE_NODES, WORK_XY[1]), np.full(ROPE_NODES, ROPE_RADIUS + 1e-3)], axis=-1)
    area = np.pi * ROPE_RADIUS**2
    second_moment = 0.25 * np.pi * ROPE_RADIUS**4
    polar_moment = 0.5 * np.pi * ROPE_RADIUS**4
    rope = scene.add_entity(
        gs.morphs.Rod(points=points, radius=ROPE_RADIUS),
        material=gs.materials.Mochi.Rod(
            axial_stiffness=ROPE_E * area,
            flexural_stiffness=ROPE_E * second_moment,
            torsional_stiffness=ROPE_G * polar_moment,
            linear_density=ROPE_RHO * area,
            linear_rotational_inertia=ROPE_RHO * polar_moment,
        ),
        surface=gs.surfaces.Default(color=(0.8, 0.6, 0.3)),
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(1.5, -0.8, 0.7), lookat=(0.65, 0.0, 0.15))
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_qpos(np.array(FRANKA_START_QPOS))
    franka.control_dofs_position(np.array(FRANKA_TARGET_QPOS)[motors_dof], motors_dof)
    franka.control_dofs_force(np.full(2, -1.0), fingers_dof)

    if args.record:
        cam.start_recording(save_to_filename="rope_arm.mp4", fps=30)

    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            mid_z = float(rope.get_vertices_position()[ROPE_NODES // 2, 2])
            hand_z = float(franka.get_link("hand").get_pos()[2])
            print(
                f"step {i_step:4d}: rope mid z={mid_z:.4f} hand z={hand_z:.4f} "
                f"newton iterations={int(info['n_iter'][0])}"
            )
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
