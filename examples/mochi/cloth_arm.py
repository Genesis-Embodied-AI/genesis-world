"""A Franka arm pressing a 31x31 cloth sheet with self-contact lying on the ground.

This is the `cloth_arm` benchmark scene (the reinforcement-learning shaped acceptance scene of
`tests/mochi/benchmark`) run as a standalone demo: the hand starts about 0.30 m above the cloth and is driven down to
about 0.13 m by PD joint control while the fingertips press with 1 N each, wrinkling the sheet against the ground.
The articulated arm, the shell elements, the self-contact of the cloth and the penalty contact all enter one implicit
Newton solve (mochi's default budget of 4 iterations per step).
"""

import argparse

import numpy as np
import trimesh

import genesis as gs

FRANKA_START_QPOS = (-1.0418, 1.3805, 1.5885, -1.7021, -1.3798, 1.5783, 1.4467, 0.04, 0.04)
FRANKA_TARGET_QPOS = (-1.0095, 1.5617, 1.3595, -1.6830, -1.5855, 1.7817, 1.4595, 0.04, 0.04)
WORK_XY = (0.65, 0.0)
CLOTH_CELLS, CLOTH_SIZE, CLOTH_HEIGHT = 30, 0.3, 0.01


def sheet_mesh(path, n_cells, size):
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            faces.append([a, a + 1, a + n_cells + 2])
            faces.append([a, a + n_cells + 2, a + n_cells + 1])
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=4),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.8, -1.0, 0.9), camera_lookat=(0.65, 0.0, 0.2)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Mochi.Rigid(friction=1.0, viscous_friction=1.0),
    )
    cloth = scene.add_entity(
        gs.morphs.Mesh(
            file=sheet_mesh("/tmp/mochi_cloth_arm_sheet.obj", CLOTH_CELLS, CLOTH_SIZE),
            pos=(*WORK_XY, CLOTH_HEIGHT),
        ),
        material=gs.materials.Mochi.Shell(
            E=2e4,
            nu=0.3,
            rho=200.0,
            thickness=1e-3,
            friction=0.6,
            collider_radius=5e-3,
            penalty_coefficient=1e7,
            self_contact=True,
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.4, 0.3)),
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(1.8, -1.0, 0.9), lookat=(0.65, 0.0, 0.2))
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_qpos(np.array(FRANKA_START_QPOS))
    franka.control_dofs_position(np.array(FRANKA_TARGET_QPOS)[motors_dof], motors_dof)
    franka.control_dofs_force(np.full(2, -1.0), fingers_dof)

    if args.record:
        cam.start_recording(save_to_filename="cloth_arm.mp4", fps=30)

    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            cloth_z = float(cloth.get_vertices_position()[:, 2].max())
            hand_z = float(franka.get_link("hand").get_pos()[2])
            print(
                f"step {i_step:4d}: cloth zmax={cloth_z:.4f} hand z={hand_z:.4f} "
                f"newton iterations={int(info['n_iter'][0])}"
            )
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
