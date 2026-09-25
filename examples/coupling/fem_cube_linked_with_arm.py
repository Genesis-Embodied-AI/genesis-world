import os
import argparse

import numpy as np
from tqdm import tqdm

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-3, help="Simulation time step")
    parser.add_argument("--substeps", type=int, default=1, help="Number of solver substeps per step")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level=None)

    steps = int(1.0 / args.dt if "PYTEST_VERSION" not in os.environ else 5)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
            gravity=(0, 0, -9.81),
        ),
        # The linear corotated model the cube uses only supplies the energy derivatives the implicit solver needs,
        # so the explicit solver cannot integrate this scene.
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            enable_vertex_constraints=True,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    # Setup scene entities
    scene.add_entity(gs.morphs.Plane())

    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.05),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e4,  # stiffness
            nu=0.45,  # compressibility (0 to 0.5)
            rho=1000.0,  # density
            model="linear_corotated",
        ),
    )
    arm = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, 0),
        ),
    )

    # Recording every simulation step would be far denser than any player needs, and the step size here is tiny.
    video_fps = min(100.0, 1.0 / args.dt)
    cam = scene.add_camera(
        res=(640, 480),
        pos=(-2.0, 3.0, 2.0),
        lookat=(0.5, 0.5, 0.5),
        fov=30,
    )

    scene.build()

    video_filename = f"out/cube_link_arm_dt={args.dt}_substeps={args.substeps}.mp4"
    cam.start_recording(save_to_filename=video_filename, fps=video_fps)

    try:
        joint_names = [j.name for j in arm.joints]
        dofs_idx_local = []
        for j in arm.joints:
            dofs_idx_local += j.dofs_idx_local
        end_joint = arm.get_joint(joint_names[-1])

        arm.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        arm.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        arm.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        for _ in range(100):
            arm.set_dofs_position(
                np.array([0.9643, -0.3213, -0.6685, -2.3139, -0.2890, 2.0335, -1.6014, 0.0306, 0.0306]), dofs_idx_local
            )
            scene.step()

        print("cube init pos", cube.init_positions)
        pin_idx = [1, 5]
        cube.set_vertex_constraints(verts_idx_local=pin_idx, link=end_joint.link)
        print("Cube initial positions:", cube.init_positions[pin_idx])
        scene.draw_debug_spheres(poss=cube.init_positions[pin_idx], radius=0.02, color=(1.0, 0.0, 1.0, 0.8))

        arm_target_pos = (0.3, 0.2, 0.8)
        scene.draw_debug_spheres(poss=[arm_target_pos], radius=0.02, color=(0.0, 1.0, 0.0, 0.8))
        qpos = arm.inverse_kinematics(
            link=end_joint.link,
            pos=np.array(arm_target_pos, gs.np_float),
            quat=np.array((0.0, 1.0, 0.0, 0.0), gs.np_float),
        )
        arm_path_waypoints = arm.plan_path(qpos_goal=qpos, num_waypoints=steps)

        for waypoint in tqdm(arm_path_waypoints, total=len(arm_path_waypoints)):
            arm.control_dofs_position(waypoint)
            scene.step()

        print("Now dropping the cube")
        cube.remove_vertex_constraints()
        for _ in tqdm(range(steps), total=steps):
            arm.control_dofs_position(arm_path_waypoints[-1])
            scene.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        cam.stop_recording()
        gs.logger.info(f"Saved video to {video_filename}")


if __name__ == "__main__":
    main()
