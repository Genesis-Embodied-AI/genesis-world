"""Scripted Franka grasp of cloth bridging two fixed supports.

The commanded pose is the center between the fingertip pads, not the Panda
``hand`` link origin. The script settles the cloth, approaches with an open
gripper, descends into the support gap, closes, lifts, transports, and releases.
"""

from __future__ import annotations

import argparse

import numpy as np

import genesis as gs

from cloth_grid_asset import cloth_grid_asset
from franka_gripper import gripper_center_from_hand_pose, hand_pose_from_gripper_center

DT = 0.01
CLOTH_RESOLUTION = 25
CLOTH_SIZE = 0.32
SUPPORT_SIZE = 0.05
SUPPORT_GAP = 0.12
SUPPORT_TOP = 0.44
SUPPORT_OFFSET = 0.5 * (SUPPORT_GAP + SUPPORT_SIZE)
BRIDGE_CENTER = np.array((0.55, 0.0, SUPPORT_TOP), dtype=np.float64)
PALM_DOWN = np.array((0.0, 1.0, 0.0, 0.0), dtype=np.float64)
HOME_QPOS = np.array(
    (0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854, 0.04, 0.04),
    dtype=np.float64,
)


def _cloth_patch_center(cloth) -> np.ndarray:
    positions = cloth.get_state().pos.cpu().numpy().reshape(CLOTH_RESOLUTION, CLOTH_RESOLUTION, 3)
    middle = CLOTH_RESOLUTION // 2
    return positions[middle - 2 : middle + 3, middle - 2 : middle + 3].mean(axis=(0, 1))


def _gripper_center(hand) -> np.ndarray:
    center, _ = gripper_center_from_hand_pose(
        hand.get_pos().cpu().numpy().reshape(3),
        hand.get_quat().cpu().numpy().reshape(4),
    )
    return center


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("--no-check", action="store_true", help="Skip end-of-script grasp assertions")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="info", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        engine_options=gs.options.NewtonEngineOptions(
            contact_d_hat=1e-3,
            contact_friction_mu=2.0,
            contact_resistance=1e4,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1100, 720),
            camera_pos=(1.45, -1.25, 1.05),
            camera_lookat=(0.52, 0.0, 0.48),
            camera_fov=38,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        vis_mode="collision",
    )
    for x in (BRIDGE_CENTER[0] - SUPPORT_OFFSET, BRIDGE_CENTER[0] + SUPPORT_OFFSET):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(x, 0.0, SUPPORT_TOP - 0.5 * SUPPORT_SIZE),
                size=(SUPPORT_SIZE, SUPPORT_SIZE, SUPPORT_SIZE),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.72, 0.67, 0.55, 1.0)),
        )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(cloth_grid_asset(resolution=CLOTH_RESOLUTION, size=CLOTH_SIZE)),
            pos=tuple(BRIDGE_CENTER + np.array((0.0, 0.0, 0.03))),
        ),
        material=gs.materials.FEM.QCloth(
            E=2e4,
            shear_modulus=2e3,
            rho=200.0,
            thickness=1e-3,
            bending_youngs_modulus=3e3,
        ),
        surface=gs.surfaces.Default(color=(0.25, 0.45, 0.90, 1.0)),
    )
    scene.build()

    arm_dofs = np.arange(7, dtype=np.int32)
    finger_dofs = np.arange(7, 9, dtype=np.int32)
    franka.set_qpos(HOME_QPOS)
    franka.set_dofs_kp(
        np.array((4500, 4500, 3500, 3500, 2000, 2000, 2000), dtype=np.float64),
        dofs_idx_local=arm_dofs,
    )
    franka.set_dofs_kv(
        np.array((450, 450, 350, 350, 200, 200, 200), dtype=np.float64),
        dofs_idx_local=arm_dofs,
    )
    franka.set_dofs_force_range(-400.0, 400.0, dofs_idx_local=arm_dofs)
    franka.set_dofs_kp(500.0, dofs_idx_local=finger_dofs)
    franka.set_dofs_kv(20.0, dofs_idx_local=finger_dofs)
    franka.set_dofs_force_range(-100.0, 100.0, dofs_idx_local=finger_dofs)
    hand = franka.get_link("hand")
    left_finger = franka.get_link("left_finger")
    right_finger = franka.get_link("right_finger")

    def report_phase(name: str) -> np.ndarray:
        patch = _cloth_patch_center(cloth)
        center = _gripper_center(hand)
        fingers = franka.get_dofs_position(finger_dofs).cpu().numpy()
        left = left_finger.get_pos().cpu().numpy().reshape(3)
        right = right_finger.get_pos().cpu().numpy().reshape(3)
        print(
            f"{name}: patch={patch.tolist()}, center={center.tolist()}, "
            f"finger_qpos={fingers.tolist()}, finger_roots={(left.tolist(), right.tolist())}"
        )
        return patch

    def command(center_position: np.ndarray, finger_position: float) -> None:
        hand_position, hand_quaternion = hand_pose_from_gripper_center(center_position, PALM_DOWN)
        target_qpos = franka.inverse_kinematics(
            link=hand,
            pos=hand_position,
            quat=hand_quaternion,
            init_qpos=franka.get_qpos(),
            max_samples=1,
            max_solver_iters=12,
            damping=0.05,
            max_step_size=0.1,
            dofs_idx_local=arm_dofs,
        )
        franka.control_dofs_position(target_qpos[arm_dofs], dofs_idx_local=arm_dofs)
        franka.control_dofs_position(finger_position, dofs_idx_local=finger_dofs)

    def hold(frames: int, center_position: np.ndarray | None, finger_position: float) -> None:
        for _ in range(frames):
            if center_position is None:
                franka.control_dofs_position(HOME_QPOS[:7], dofs_idx_local=arm_dofs)
                franka.control_dofs_position(finger_position, dofs_idx_local=finger_dofs)
            else:
                command(center_position, finger_position)
            scene.step()

    def move(frames: int, destination: np.ndarray, finger_position: float) -> None:
        start = _gripper_center(hand)
        for frame in range(frames):
            alpha = (frame + 1) / frames
            command((1.0 - alpha) * start + alpha * destination, finger_position)
            scene.step()

    def move_fingers(frames: int, center_position: np.ndarray, destination: float) -> None:
        start = franka.get_dofs_position(finger_dofs).mean().item()
        for frame in range(frames):
            alpha = (frame + 1) / frames
            command(center_position, (1.0 - alpha) * start + alpha * destination)
            scene.step()

    hold(100, None, 0.04)
    settled_patch = report_phase("settled")

    hover = BRIDGE_CENTER + np.array((0.0, 0.0, 0.20))
    grasp = BRIDGE_CENTER + np.array((0.0, 0.0, -0.018))
    lifted = grasp + np.array((0.0, 0.0, 0.16))
    transported = lifted + np.array((-0.14, 0.10, 0.0))

    move(70, hover, 0.04)
    move(70, grasp, 0.04)
    hold(100, grasp, 0.0)
    report_phase("closed")
    move(90, lifted, 0.0)
    lifted_patch = report_phase("lifted")
    move(120, transported, 0.0)
    transported_patch = report_phase("transported")
    move_fingers(60, transported, 0.04)
    released_patch = report_phase("released")

    print(
        "cloth patch centers:",
        f"settled={settled_patch.tolist()}",
        f"lifted={lifted_patch.tolist()}",
        f"transported={transported_patch.tolist()}",
        f"released={released_patch.tolist()}",
    )
    if not args.no_check:
        if lifted_patch[2] - settled_patch[2] <= 0.04:
            raise RuntimeError("Franka closed its gripper but did not lift the cloth patch")
        if np.linalg.norm(transported_patch[:2] - lifted_patch[:2]) <= 0.05:
            raise RuntimeError("Franka lifted the cloth but did not transport it")
        if released_patch[2] >= transported_patch[2] - 0.005:
            raise RuntimeError("Cloth patch did not separate from the gripper after release")


if __name__ == "__main__":
    main()
