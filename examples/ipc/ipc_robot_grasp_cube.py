import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ipc", action="store_true", help="Disable the IPC coupler")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument(
        "--coup-type",
        type=str,
        default="two_way_soft_constraint",
        choices=["two_way_soft_constraint", "external_articulation"],
        help="How the articulated robot is coupled to the IPC solver",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="info")

    coupler_options = None
    if not args.no_ipc:
        coupler_options = gs.options.IPCCouplerOptions(
            newton_translation_tolerance=10.0,
            constraint_strength_translation=10.0,
            constraint_strength_rotation=10.0,
            enable_rigid_ground_contact=False,
            enable_rigid_rigid_contact=False,
        )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        coupler_options=coupler_options,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 1.0, 1.0),
            camera_lookat=(0.3, 0.0, 0.5),
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    franka_material_kwargs = dict(
        coup_friction=0.8,
        coup_type=args.coup_type,
    )
    if args.coup_type == "two_way_soft_constraint":
        franka_material_kwargs["coup_links"] = ("left_finger", "right_finger")
    franka_material = gs.materials.Rigid(**franka_material_kwargs) if not args.no_ipc else None
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
        ),
        material=franka_material,
    )

    if not args.no_ipc:
        cube_material = gs.materials.FEM.Elastic(
            E=5.0e4,
            nu=0.45,
            rho=1000.0,
            friction_mu=0.5,
            model="stable_neohookean",
        )
    else:
        cube_material = gs.materials.Rigid()
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.65, 0.0, 0.03),
            size=(0.05, 0.05, 0.05),
        ),
        material=cube_material,
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.2, 0.5),
        ),
    )

    scene.build()

    motors_dof, fingers_dof = slice(0, 7), slice(7, 9)
    end_effector = franka.get_link("hand")

    franka.set_dofs_kp([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 500.0, 500.0])
    franka.set_dofs_kv([100.0, 100.0], fingers_dof)

    qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.4], quat=[0.0, 1.0, 0.0, 0.0])
    if not args.no_ipc or args.coup_type == "external_articulation":
        franka.control_dofs_position(qpos[motors_dof], dofs_idx_local=motors_dof)
        franka.control_dofs_position(0.04, dofs_idx_local=fingers_dof)
        for _ in range(200 if "PYTEST_VERSION" not in os.environ else 1):
            scene.step()
    else:
        franka.set_dofs_position(qpos)

    # Lower the grapper half way to grasping position
    qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.25], quat=[0.0, 1.0, 0.0, 0.0])
    franka.control_dofs_position(qpos[motors_dof], dofs_idx_local=motors_dof)
    for _ in range(100 if "PYTEST_VERSION" not in os.environ else 1):
        scene.step()

    # Reach grasping position
    qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.135], quat=[0.0, 1.0, 0.0, 0.0])
    franka.control_dofs_position(qpos[motors_dof], dofs_idx_local=motors_dof)
    for _ in range(50 if "PYTEST_VERSION" not in os.environ else 1):
        scene.step()

    # Grasp the cube
    franka.control_dofs_position(qpos[motors_dof], dofs_idx_local=motors_dof)
    franka.control_dofs_position(0.0, dofs_idx_local=fingers_dof)
    for _ in range(10 if "PYTEST_VERSION" not in os.environ else 1):
        scene.step()

    # Lift the cube
    qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.3], quat=[0.0, 1.0, 0.0, 0.0])
    franka.control_dofs_position(qpos[motors_dof], dofs_idx_local=motors_dof)
    for _ in range(50 if "PYTEST_VERSION" not in os.environ else 1):
        scene.step()


if __name__ == "__main__":
    main()
