"""
Heterogeneous Simulation Example
================================

This example demonstrates heterogeneous simulation, where different parallel
environments can have different geometry variants for the same entity.

Variant Assignment Rules:
    When passing a list of morphs to scene.add_entity(), variants are distributed
    across environments using the following rules:

    1. When n_envs >= n_variants:
       Balanced block assignment. Environments are divided into blocks, with each
       block assigned to one variant. For example, with 4 variants and 8 environments:
       - Environments 0-1 -> Variant 0
       - Environments 2-3 -> Variant 1
       - Environments 4-5 -> Variant 2
       - Environments 6-7 -> Variant 3

    2. When n_envs < n_variants:
       Each environment i gets variant i (0-indexed). Variants beyond n_envs are
       unused. For example, with 4 variants and 2 environments:
       - Environment 0 -> Variant 0 (first morph in list)
       - Environment 1 -> Variant 1 (second morph in list)
       - Variants 2 and 3 are unused

Usage:
    python heterogeneous_simulation.py -v -b 4  # 4 environments (matches 4 variants)
    python heterogeneous_simulation.py -v -b 8  # 8 environments (2 per variant)
    python heterogeneous_simulation.py -v -b 2  # 2 environments (only first 2 variants used)
"""

import argparse

import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-b", "--num-envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32")
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    # Define 4 geometry variants - see module docstring for variant assignment rules
    morphs_heterogeneous = [
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),  # Variant 0
        gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.65, 0.0, 0.02)),  # Variant 1
        gs.morphs.Sphere(radius=0.015, pos=(0.65, 0.0, 0.02)),  # Variant 2
        gs.morphs.Sphere(radius=0.025, pos=(0.65, 0.0, 0.02)),  # Variant 3
    ]
    grasping_object = scene.add_entity(
        morph=morphs_heterogeneous,
    )
    scene.build(n_envs=args.num_envs, env_spacing=(1, 1))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    l_qpos = [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]
    if args.num_envs == 0:
        franka.set_qpos(np.array(l_qpos))
    else:
        franka.set_qpos(np.array([l_qpos] * args.num_envs))
    scene.step()

    AABB = grasping_object.get_AABB()
    mass = grasping_object.get_mass()
    print("heterogeneous AABB", AABB)
    print("heterogeneous mass", mass)

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.135]] * args.num_envs),
        quat=np.array([[0, 1, 0, 0]] * args.num_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)

    # hold
    for i in range(100):
        print("hold", i)
        scene.step()

    # grasp
    finder_pos = 0.0
    for i in range(100):
        print("grasp", i)
        franka.control_dofs_position(qpos[..., :-2], motors_dof)
        franka.control_dofs_position(np.array([[finder_pos, finder_pos]] * args.num_envs), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.3]] * args.num_envs),
        quat=np.array([[0, 1, 0, 0]] * args.num_envs),
    )
    for i in range(200):
        print("lift", i)
        franka.control_dofs_position(qpos[..., :-2], motors_dof)
        franka.control_dofs_position(np.array([[finder_pos, finder_pos]] * args.num_envs), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
