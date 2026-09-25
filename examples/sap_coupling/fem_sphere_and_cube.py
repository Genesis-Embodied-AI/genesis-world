import argparse
import genesis as gs
import os
from huggingface_hub import snapshot_download


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

    n_steps = 200 if "PYTEST_VERSION" not in os.environ else 2

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0, 0, 0),
        ),
        show_viewer=args.vis,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.1),
            radius=0.1,
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.4,
            model="linear_corotated",
        ),
    )
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="4d96c3512df4421d4dd3d626055d0d1ebdfdd7cc",
        allow_patterns="cube8.obj",
        max_workers=1,
    )
    cube = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/cube8.obj",
            pos=(0.0, 0.0, 0.4),
            scale=0.1,
        ),
        material=gs.materials.FEM.Elastic(
            model="linear_corotated",
        ),
    )
    scene.build()

    for _ in range(n_steps):
        scene.step()


if __name__ == "__main__":
    main()
