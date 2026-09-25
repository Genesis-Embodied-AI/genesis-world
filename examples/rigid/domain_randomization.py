import argparse

import numpy as np
import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            # Giving each environment its own masses and centers of mass is what this exists for: without it the
            # link info is shared, and a per-environment value has nowhere to live.
            batch_links_info=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0, 0.4),
        ),
    )
    n_envs = 8
    scene.build(n_envs=n_envs)

    robot.set_friction_ratio(
        friction_ratio=0.5 + torch.rand(scene.n_envs, robot.n_links),
        links_idx_local=np.arange(0, robot.n_links),
    )

    # set mass of a single link
    link = robot.get_link("RR_thigh")
    ori_mass = robot.get_links_mass()
    print("original mass", link.get_mass(), ori_mass)
    link.set_mass(1)
    print("diff mass", robot.get_links_mass() - ori_mass)

    robot.set_links_mass(
        robot.get_links_mass() * (0.75 + 0.5 * torch.rand(scene.n_envs, robot.n_links, device=gs.device))
    )
    # The center of mass is where it is, so a randomized model is the authored one shifted rather than one whose every
    # link carries its mass at its own origin.
    robot.set_links_COM(
        com=robot.get_links_COM() + (-0.05 + 0.1 * torch.rand(scene.n_envs, robot.n_links, 3, device=gs.device))
    )
    robot.set_links_inertia(
        robot.get_links_inertia() * (0.75 + 0.5 * torch.rand(scene.n_envs, robot.n_links, 1, 1, device=gs.device))
    )
    aabb = robot.get_AABB()

    joints_name = (
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
    )
    motors_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]

    robot.set_dofs_kp(np.full(12, 20), motors_dof_idx)
    robot.set_dofs_kv(np.full(12, 1), motors_dof_idx)
    default_dof_pos = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5])
    # padding to n_env x n_dofs
    default_dof_pos = np.tile(default_dof_pos, (n_envs, 1))
    robot.control_dofs_position(default_dof_pos, motors_dof_idx)

    scene.step()


if __name__ == "__main__":
    main()
