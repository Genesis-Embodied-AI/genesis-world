"""A Schunk WSG-50 gripper whose FinRay fingers are simulated as deformable bodies attached to the finger links.

The gripper URDF is loaded as an articulated rigid body with the finger geometry stripped; each finger is a
tetrahedral mesh (the `.vtk` deformable meshes shipped with the model) whose base vertices are rigidly attached to
the corresponding prismatic finger link (`MochiSoftEntity.attach_to_link`), and is rendered through the model's
detailed `.gltf` visual mesh skinned by the simulation tetrahedra (`MochiSoftEntity.set_visual_mesh`). The prismatic drive, the mimic joint
coupling, the finite elements, the attachments and the contact all enter one implicit Newton solve, so closing the
compliant fingers on the box makes them wrap around it.

Requires the LBM eval models (https://huggingface.co/datasets/toyota-research-institute/lbm_eval_models); point
`--asset-dir` at its `robots/schunk_grippers` directory.
"""

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
from genesis.engine.entities.mochi_entity.mochi_soft_entity import load_vtk_tet_files

FINGER_LINKS = ("finray_finger_L", "finray_finger_R")
FINGER_MESH = "finray_finger_MA-049-PT-0005_MA-049-PT-0006_2024_07_23_{side}_low.vtk"
FINGER_VISUAL_MESH = "finray_finger_MA-049-PT-0005_MA-049-PT-0006_2024_07_23_{side}.gltf"
# Recentered prismatic coordinates: q = 0 is the open gripper (+-0.045 m from the nominal finger pose), positive
# q closes. The mimic coupling q2 = -q1 keeps the fingers symmetric in the shifted coordinates as well.
OPEN_SHIFT = 0.045
BASE_POS = (0.0, 0.0, 0.16)
FINGER_OFFSET_Y = 0.03625  # finger joint origin along the Schunk base y axis (the tool axis after the mount rotation)
BOX_SIZE = 0.03


def trim_urdf(urdf_path, out_path):
    """Strip the finger geometry (simulated as deformable bodies instead), shrink the finger link inertia, recenter
    the finger joints so that q = 0 is the open gripper, and absolutize the remaining mesh paths."""
    ET.register_namespace("drake", "http://drake.mit.edu")
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.iter("link"):
        if link.get("name") not in FINGER_LINKS:
            continue
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
        inertial = link.find("inertial")
        inertial.find("mass").set("value", "0.001")
        inertia = inertial.find("inertia")
        for axis in ("ixx", "iyy", "izz"):
            inertia.set(axis, "1e-7")
        for axis in ("ixy", "ixz", "iyz"):
            inertia.set(axis, "0")
    for joint, shift, lower, upper in (
        ("finger_joint_1", -OPEN_SHIFT, -0.01, OPEN_SHIFT),
        ("finger_joint_2", OPEN_SHIFT, -OPEN_SHIFT, 0.01),
    ):
        element = next(j for j in root.iter("joint") if j.get("name") == joint)
        origin = element.find("origin")
        xyz = [float(v) for v in origin.get("xyz").split()]
        xyz[0] += shift
        origin.set("xyz", " ".join(repr(v) for v in xyz))
        limit = element.find("limit")
        limit.set("lower", repr(lower))
        limit.set("upper", repr(upper))
    for mesh in root.iter("mesh"):
        mesh.set("filename", os.path.join(os.path.dirname(urdf_path), mesh.get("filename")))
    tree.write(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Quadrants CPU threads; more than 1 parallelizes the solver loops of large deformable scenes",
    )
    parser.add_argument("-n", "--n_steps", type=int, default=800, help="Number of simulation steps")
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

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64", cpu_threads=args.cpu_threads)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.45, -0.45, 0.32), camera_lookat=(0.0, 0.0, 0.1)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    gripper = scene.add_entity(
        gs.morphs.URDF(
            file=trim_urdf(urdf_path, os.path.join(tempfile.mkdtemp(prefix="mochi_finray_"), "gripper.urdf")),
            pos=BASE_POS,
            euler=(180.0, 0.0, 0.0),
            fixed=True,
        ),
        material=gs.materials.Mochi.Rigid(friction=1.5, viscous_friction=1.0),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(BOX_SIZE,) * 3, pos=(0.0, 0.0, 0.5 * BOX_SIZE)),
        material=gs.materials.Mochi.Rigid(rho=300.0, friction=1.5, viscous_friction=1.0),
        surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9)),
    )
    # World pose of the finger links at q = 0 under the fixed base: the mount rotation baked into the URDF leaves the
    # finger frames axis-aligned with the root, offset by the joint origin (the base is rotated by 180 deg about x).
    fingers = []
    for side, sign in (("L", -1.0), ("R", 1.0)):
        link_pos = np.array(BASE_POS) + np.array([sign * OPEN_SHIFT, -sign * 5e-5, -FINGER_OFFSET_Y])
        mesh_path = os.path.join(args.asset_dir, "assets", FINGER_MESH.format(side=side))
        # The morph rotation acts about the vertex centroid: offset the position so the mesh origin (the mount
        # plane, link-frame z = 0) lands on the link origin under the 180 deg flip.
        com = load_vtk_tet_files(mesh_path)[0].mean(axis=0)
        pos = link_pos - 2.0 * np.array([0.0, com[1], com[2]])
        finger = scene.add_entity(
            gs.morphs.Mesh(
                file=mesh_path,
                pos=tuple(pos),
                euler=(180.0, 0.0, 0.0),
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
        # The finger mounts at its base plane (link-frame z = 0, the highest vertices under the downward tool axis).
        base_verts = np.where(finger.init_positions[:, 2] > link_pos[2] - 0.004)[0]
        finger.attach_to_link(gripper.get_link(f"finray_finger_{side}"), base_verts, stiffness=1e6, damping=1.0)
        fingers.append(finger)

    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(0.45, -0.45, 0.32), lookat=(0.0, 0.0, 0.1))
    scene.build()

    # The fingers overlap the gripper body near their mounts and each other when fully closed: contact acts only
    # against the box and the ground.
    scene.mochi_solver.enable_entity_contact(fingers[0], fingers[1], False)
    for finger in fingers:
        scene.mochi_solver.enable_entity_contact(finger, gripper, False)
    gripper.set_dofs_kp(np.array([2e4]), np.array([0]))
    gripper.set_dofs_kv(np.array([5.0]), np.array([0]))
    gripper.control_dofs_position(np.array([0.0]), np.array([0]))

    if args.record:
        cam.start_recording(save_to_filename="finray_gripper.mp4", fps=30)

    for i_step in range(args.n_steps):
        if 60 <= i_step < 200:
            # Close onto the box over one second: the commanded opening (2.4 cm) is narrower than the box, so the
            # compliant fingers deform around it.
            gripper.control_dofs_position(np.array([0.025 * (i_step - 59) / 60.0]), np.array([0]))
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            box_z = float(box.get_pos()[2])
            grip_force = -float(np.asarray(fingers[0].get_vertices_contact_force().cpu())[:, 0].sum())
            tip_gap = float(
                np.asarray(fingers[1].get_vertices_position()[:, 0].min().cpu())
                - np.asarray(fingers[0].get_vertices_position()[:, 0].max().cpu())
            )
            print(
                f"step {i_step:4d}: box z={box_z:.4f} finger gap={tip_gap:+.4f} grip force={grip_force:+.3f} N "
                f"newton iterations={int(info['n_iter'][0])}"
            )
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
