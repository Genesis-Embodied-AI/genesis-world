"""A Franka Panda whose hand is replaced by the Schunk WSG-50 FinRay gripper, picking a box off the ground.

The arm is loaded without its hand and the gripper is merged into the same kinematic tree with `RigidEntity.attach`,
so the seven arm joints, the gripper prismatic drive and its mimic coupling form one articulated body. Each FinRay
finger is a tetrahedral deformable body whose base vertices are rigidly attached to the prismatic finger link
(`MochiSoftEntity.attach_to_link`), rendered through the model's detailed `.gltf` visual mesh skinned by the
simulation tetrahedra. The joint drives, the finite elements, the attachments and the contact all enter one implicit
Newton solve, so the compliant fingers wrap around the box and carry it as the arm lifts.

An attachment anchors each vertex in the frame of the link it coincides with when the scene is built, so the
fingers are placed at the pose the parsed model gives their link. The arm then moves to the grasp configuration and
the finger vertices follow rigidly (`MochiSoftEntity.set_position`), which leaves every attachment at zero
violation.

Requires the LBM eval models (https://huggingface.co/datasets/toyota-research-institute/lbm_eval_models); point
`--asset-dir` at its `robots/schunk_grippers` directory.
"""

import argparse
import os
import tempfile

import numpy as np
from finray_gripper import FINGER_MESH, FINGER_VISUAL_MESH, trim_urdf

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.mochi_entity.mochi_soft_entity import load_vtk_tet_files
from genesis.utils.misc import tensor_to_array

BOX_SIZE = 0.03
BOX_XY = (0.45, 0.0)
# Standoff of the gripper mount frame from the flange along the tool axis. The URDF carries the mount frame at the
# center of the Schunk body, so bolting the body onto the flange face takes half its depth of clearance.
MOUNT_OFFSET = 0.03625
# Roll of the gripper about the tool axis, turning the finger separation axis a quarter turn off the flange x axis.
MOUNT_EULER = (0.0, 0.0, 90.0)
# Height of the gripper mount frame above the ground at the grasp: it puts the box between the finger tips, where
# the angled FinRay pads close tightest, and leaves the tips a few millimeters clear of the ground.
GRASP_HEIGHT = 0.16
LIFT_HEIGHT = 0.40
# Prismatic target closing the fingers onto the box: the angled pads meet closer than the box is wide, so the box
# holds them open and they press back on it.
CLOSE_STROKE = 0.03
# Elbow-up seed of the inverse-kinematics solves, the neutral configuration of the parsed model being a singular
# straight-up posture outside the elbow joint limit.
SEED_QPOS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
# Tool orientation of both waypoints: the tool axis points down and the fingers separate along the world y axis.
TOOL_QUAT = (0.0, 1.0, 0.0, 0.0)


def link_neutral_pose(link):
    """World pose a link takes from the parse-time link frames, which is where the build starts it for a model whose
    joint coordinates all start at zero.

    The deformable fingers must be placed where the link they attach to stands when the scene is built, and the
    solver only reports link poses once the scene is built.
    """
    pos, quat = gu.zero_pos(), gu.identity_quat()
    while link is not None:
        pos, quat = gu.transform_pos_quat_by_trans_quat(pos, quat, link.desc.pos, link.desc.quat)
        link = link.solver.links[link.parent_idx] if link.parent_idx >= 0 else None
    return pos, quat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=400, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument(
        "--asset-dir",
        default="/home/kashu/datasets/lbm_eval_models/robots/schunk_grippers",
        help="Directory holding schunk_wsg_50_finray_fr3_mount.urdf and its assets/",
    )
    args = parser.parse_args()

    urdf_path = os.path.join(args.asset_dir, "schunk_wsg_50_finray_fr3_mount.urdf")
    if not os.path.exists(urdf_path):
        print(f"Skipped: '{urdf_path}' not found (see --asset-dir).")
        return

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.1, -1.1, 0.8), camera_lookat=(0.35, 0.0, 0.3)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda_nohand.xml"),
        material=gs.materials.Mochi.Rigid(friction=1.0, viscous_friction=1.0),
    )
    gripper = scene.add_entity(
        gs.morphs.URDF(
            file=trim_urdf(urdf_path, os.path.join(tempfile.mkdtemp(prefix="mochi_finray_"), "gripper.urdf")),
        ),
        material=gs.materials.Mochi.Rigid(friction=1.5, viscous_friction=1.0),
    )
    # The gripper mount frame is welded onto the flange of the arm, standing the body clear of the wrist and rolled
    # about the tool axis. The mount rotation that turns the Schunk body along the tool axis is baked into the URDF.
    gripper.attach(
        franka,
        "attachment",
        pos=(0.0, 0.0, MOUNT_OFFSET),
        quat=gu.xyz_to_quat(np.array(MOUNT_EULER), rpy=True, degrees=True),
    )
    fingers, fingers_link, fingers_verts_local = [], [], []
    for side in ("L", "R"):
        link = gripper.get_link(f"finray_finger_{side}")
        link_pos, link_quat = link_neutral_pose(link)
        mesh_path = os.path.join(args.asset_dir, "assets", FINGER_MESH.format(side=side))
        # The morph rotation acts about the vertex centroid: offset the position so the mesh origin lands on the
        # link origin once rotated into the link frame.
        com = load_vtk_tet_files(mesh_path)[0].mean(axis=0)
        pos = link_pos - com + gu.transform_by_quat(com, link_quat)
        finger = scene.add_entity(
            gs.morphs.Mesh(
                file=mesh_path,
                pos=tuple(pos),
                quat=tuple(link_quat),
            ),
            material=gs.materials.Mochi.Elastic(E=3e7, nu=0.4, rho=1200.0, friction=1.5, viscous_friction=1.0),
            # No color: the visual mesh keeps the base-color texture of the model's gltf.
            surface=gs.surfaces.Default(),
        )
        # The detailed visual mesh of the model, skinned by the simulation tetrahedra; the URDF's visual tag
        # rotates it by 90 degrees about x into the frame of the collision (simulation) mesh.
        finger.set_visual_mesh(
            file=os.path.join(args.asset_dir, "assets", FINGER_VISUAL_MESH.format(side=side)),
            euler=(90.0, 0.0, 0.0),
        )
        # The rest geometry in the frame of the finger link, which both selects the base plane the finger mounts on
        # (link-frame z = 0) and carries the vertices along when the arm moves to the grasp configuration.
        verts_local = gu.inv_transform_by_trans_quat(finger.init_positions, link_pos, link_quat)
        finger.attach_to_link(link, np.where(verts_local[:, 2] < 0.004)[0], stiffness=1e6, damping=1.0)
        fingers.append(finger)
        fingers_link.append(link)
        fingers_verts_local.append(verts_local)
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE,) * 3,
            pos=(*BOX_XY, 0.5 * BOX_SIZE),
        ),
        material=gs.materials.Mochi.Rigid(rho=300.0, friction=1.5, viscous_friction=1.0),
        surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9)),
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(1.1, -1.1, 0.8), lookat=(0.35, 0.0, 0.3))
    scene.build()

    # The gripper body abuts the flange it is welded to, and the fingers overlap the gripper body near their mounts
    # and each other when closed: contact acts only against the box and the ground.
    scene.mochi_solver.enable_entity_contact(gripper, franka, False)
    scene.mochi_solver.enable_entity_contact(fingers[0], fingers[1], False)
    for finger in fingers:
        scene.mochi_solver.enable_entity_contact(finger, gripper, False)

    # Both waypoints place the gripper mount frame, which the standoff carries ahead of the flange the solve steers.
    flange = franka.get_link("attachment")
    tool = gripper.get_link("mount_base")
    qpos_grasp = franka.inverse_kinematics(
        link=flange,
        pos=(*BOX_XY, GRASP_HEIGHT),
        quat=TOOL_QUAT,
        local_point=(0.0, 0.0, MOUNT_OFFSET),
        init_qpos=SEED_QPOS,
    )
    qpos_lift = franka.inverse_kinematics(
        link=flange,
        pos=(*BOX_XY, LIFT_HEIGHT),
        quat=TOOL_QUAT,
        local_point=(0.0, 0.0, MOUNT_OFFSET),
        init_qpos=qpos_grasp,
    )
    franka.set_qpos(qpos_grasp)
    for finger, link, verts_local in zip(fingers, fingers_link, fingers_verts_local):
        link_pos, link_quat = tensor_to_array(link.get_pos()), tensor_to_array(link.get_quat())
        finger.set_position(gu.transform_by_trans_quat(verts_local, link_pos, link_quat))
    franka.control_dofs_position(qpos_grasp)
    gripper.set_dofs_kp(np.array([2e4]), np.array([0]))
    gripper.set_dofs_kv(np.array([5.0]), np.array([0]))
    gripper.control_dofs_position(np.array([0.0]), np.array([0]))

    if args.record:
        cam.start_recording(save_to_filename="articulated_finray_grasp.mp4", fps=30)

    for i_step in range(args.n_steps):
        if 60 <= i_step < 200:
            # Close onto the box over slightly more than two seconds, the fingers deforming around it.
            gripper.control_dofs_position(np.array([CLOSE_STROKE * (i_step - 59) / 140.0]), np.array([0]))
        elif 200 <= i_step < 320:
            # Interpolate the joint targets over two seconds: a step target to the raised pose would jerk the box
            # out of the pads at more than a meter per second.
            franka.control_dofs_position(qpos_grasp + (qpos_lift - qpos_grasp) * ((i_step - 199) / 120.0))
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            grip_force = np.linalg.norm(tensor_to_array(fingers[0].get_vertices_contact_force()).sum(axis=0))
            print(
                f"step {i_step:4d}: box z={box.get_pos()[2]:.4f} tool z={tool.get_pos()[2]:.4f} "
                f"grip force={grip_force:6.3f} N newton iterations={info['n_iter'][0]}"
            )
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
