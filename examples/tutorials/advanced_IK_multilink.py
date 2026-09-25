import os
import numpy as np
import genesis as gs
import genesis.utils.geom as gu

gs.init(backend=gs.cpu, precision="32", logging_level="info")

scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(
        enable_collision=False,
        enable_joint_limit=False,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, -2, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    show_viewer=True,
)


scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

scene.build()

# debug frames for IK target visualization
target_quat = np.array([0, 1, 0, 0])
T_init = gu.trans_quat_to_T(np.zeros(3), target_quat)
target_left_frame = scene.draw_debug_frame(
    T_init,
    axis_length=0.1,
    origin_size=0.01,
    axis_radius=0.005,
    color=(1, 0.5, 0.5, 1),
)
target_right_frame = scene.draw_debug_frame(
    T_init,
    axis_length=0.1,
    origin_size=0.01,
    axis_radius=0.005,
    color=(0.5, 1.0, 0.5, 1),
)

center = np.array([0.4, -0.2, 0.25])
r = 0.1

left_finger = robot.get_link("left_finger")
right_finger = robot.get_link("right_finger")

horizon = 2000 if "PYTEST_VERSION" not in os.environ else 5
for i in range(horizon):
    target_pos_left = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r
    target_pos_right = target_pos_left + np.array([0.0, 0.03, 0])

    T_left = gu.trans_quat_to_T(target_pos_left, target_quat)
    T_right = gu.trans_quat_to_T(target_pos_right, target_quat)
    scene.update_debug_objects((target_left_frame, target_right_frame), (T_left, T_right))

    q = robot.inverse_kinematics_multilink(
        links=[left_finger, right_finger],
        poss=[target_pos_left, target_pos_right],
        quats=[target_quat, target_quat],
        rot_mask=[False, False, True],  # only restrict direction of z-axis
    )

    # Note that this IK is for visualization purposes, so here we do not call scene.step(), but only update the state and the visualizer
    # In actual control applications, you should instead use robot.control_dofs_position() and scene.step()
    robot.set_dofs_position(q)
    scene.visualizer.update()
