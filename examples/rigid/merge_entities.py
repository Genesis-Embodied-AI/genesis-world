import argparse
import numpy as np
import genesis as gs


COMB = {
    "urdf2urdf",
    "urdf2mjcf",
    "mjcf2urdf",
    "mjcf2mjcf",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("--comb", type=str, default="urdf2urdf", choices=COMB, help="Entities to merge")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )

    if args.comb == "urdf2urdf" or args.comb == "urdf2mjcf":
        gs.logger.info("loading URDF panda arm")
        franka = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/panda_bullet/panda_nohand.urdf",
                merge_fixed_links=False,
                fixed=True,
            ),
        )
    else:
        gs.logger.info("loading MJCF panda arm")
        franka = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda_nohand.xml",
            ),
        )

    if args.comb == "urdf2urdf" or args.comb == "mjcf2urdf":
        gs.logger.info("loading URDF panda hand")
        hand = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/panda_bullet/hand.urdf",
                merge_fixed_links=False,
            ),
        )
    else:
        gs.logger.info("loading MJCF panda hand")
        hand = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/hand.xml",
            ),
        )

    print([link.name for link in franka.links])
    print([link.name for link in hand.links])
    hand.attach(franka, "attachment")

    scene.build()

    arm_joints_name = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    )
    arm_dofs_idx = [franka.get_joint(name).dofs_idx_local[0] for name in arm_joints_name]

    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
        arm_dofs_idx,
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200]),
        arm_dofs_idx,
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12]),
        np.array([87, 87, 87, 87, 12, 12, 12]),
        arm_dofs_idx,
    )

    gripper_joints_name = (
        "finger_joint1",
        "finger_joint2",
    )
    gripper_dofs_idx = [hand.get_joint(name).dofs_idx_local[0] for name in gripper_joints_name]

    hand.set_dofs_kp(
        np.array([100, 100]),
        gripper_dofs_idx,
    )
    hand.set_dofs_kv(
        np.array([10, 10]),
        gripper_dofs_idx,
    )
    hand.set_dofs_force_range(
        np.array([-100, -100]),
        np.array([100, 100]),
        gripper_dofs_idx,
    )

    # PD control
    for i in range(750):
        if i == 0:
            franka.control_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0]),
                arm_dofs_idx,
            )
            hand.control_dofs_position(
                np.array([0.04, 0.04]),
                gripper_dofs_idx,
            )
        elif i == 250:
            franka.control_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5]),
                arm_dofs_idx,
            )
            hand.control_dofs_position(
                np.array([0.0, 0.0]),
                gripper_dofs_idx,
            )
        elif i == 500:
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0]),
                arm_dofs_idx,
            )
            hand.control_dofs_position(
                np.array([0.04, 0.04]),
                gripper_dofs_idx,
            )

        scene.step()


if __name__ == "__main__":
    main()
