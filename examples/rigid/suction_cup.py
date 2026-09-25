import genesis as gs
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        show_viewer=args.vis,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        surface=gs.surfaces.Plastic(
            color=(1, 0, 0),
        ),
    )
    cube_2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.4, 0.2, 0.02),
        ),
        surface=gs.surfaces.Plastic(
            color=(0, 1, 0),
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        vis_mode="collision",
    )
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.25]),
        quat=np.array([0, 1, 0, 0]),
    )
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=100,  # 1s duration
    )
    # execute the planned path
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        franka.control_dofs_force(np.array([0.5, 0.5]), fingers_dof)
        scene.step()

    # allow robot to reach the last waypoint
    for i in range(100):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.130]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    # add suction / weld constraint
    rigid = scene.sim.rigid_solver
    link_cube = cube.get_link("box_baselink").idx
    link_franka = franka.get_link("hand").idx
    rigid.add_weld_constraint(link_cube, link_franka)

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.28]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.4, 0.2, 0.18]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(100):
        scene.step()

    # release
    rigid.delete_weld_constraint(link_cube, link_franka)
    for i in range(400):
        scene.step()


if __name__ == "__main__":
    main()
