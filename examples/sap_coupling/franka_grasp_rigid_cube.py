import argparse
import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="Run on GPU instead of CPU",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            pcg_threshold=1e-10,
            linesearch_ftol=1e-10,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.3, 0.0, 0.1),
            camera_lookat=(0.65, 0.0, 0.1),
        ),
        show_viewer=args.vis,
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            friction=1.0,
            coup_friction=1.0,
        ),
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        material=gs.materials.Rigid(
            friction=1.0,
            coup_friction=1.0,
        ),
    )

    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # init
    franka.set_qpos((-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04))

    # hold
    qpos = franka.inverse_kinematics(link=end_effector, pos=(0.65, 0.0, 0.13), quat=(0, 1, 0, 0))
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(15):
        scene.step()

    # grasp
    for i in range(10):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(link=end_effector, pos=(0.65, 0.0, 0.3), quat=(0, 1, 0, 0))
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(40):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
