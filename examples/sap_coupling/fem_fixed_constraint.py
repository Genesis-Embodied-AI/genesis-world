import argparse
import math
import os

import torch
import genesis as gs
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

    n_steps = 150 if "PYTEST_VERSION" not in os.environ else 2

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    fem_material_linear_corotated = gs.materials.FEM.Elastic(
        model="linear_corotated",
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            enable_vertex_constraints=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(-0.6, 0.8, 0),
        ),
        show_viewer=args.vis,
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
            pos=(0.0, 0.0, 0.7),
            scale=0.1,
        ),
        material=fem_material_linear_corotated,
    )
    scene.build()

    verts_idx = [0]

    # Run simulation
    for i in range(n_steps):
        target_poss = cube.init_positions[verts_idx] + torch.tensor(
            (0.15 * (math.cos(0.04 * i) - 1.0), 0.15 * math.sin(0.04 * i), 0.0)
        )
        cube.set_vertex_constraints(verts_idx, target_poss)
        scene.step(update_visualizer=False)
        if args.vis:
            scene.visualizer.context.draw_debug_sphere(
                pos=target_poss.squeeze(), radius=0.01, color=(1, 0, 1, 0.8), persistent=False
            )
            scene.visualizer.update(force=False, auto=True)


if __name__ == "__main__":
    main()
