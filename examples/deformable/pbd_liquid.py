import argparse
import os


import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
        ),
        pbd_options=gs.options.PBDOptions(
            max_density_solver_iterations=10,
            max_viscosity_solver_iterations=1,
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 1.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            rho=1.0,
            sampler="regular",
            density_relaxation=1.0,
            viscosity_relaxation=0.0,
        ),
        morph=gs.morphs.Box(
            lower=(0.2, 0.1, 0.1),
            upper=(0.4, 0.3, 0.5),
        ),
    )
    scene.build(n_envs=0)

    horizon = 4000 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
