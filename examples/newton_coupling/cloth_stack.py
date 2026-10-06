"""Drop a small stack of QCloth sheets with Scene.step-driven IPC contact."""

from __future__ import annotations

import argparse

import numpy as np

import genesis as gs

from cloth_grid_asset import cloth_grid_asset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-n", "--steps", type=int, default=300)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="info", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005),
        coupler_options=gs.options.NewtonCouplerOptions(
            contact_d_hat=1e-3,
            contact_friction_mu=0.0,
            contact_resistance=1e5,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 640),
            camera_pos=(0.75, -0.85, 0.65),
            camera_lookat=(0.0, 0.0, 0.16),
            camera_fov=35,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )

    material = gs.materials.FEM.QCloth(
        E=6e4,
        shear_modulus=3e4,
        rho=200.0,
        thickness=1e-3,
        bending_youngs_modulus=1e6,
    )
    cloths = []
    for layer, (resolution, size, z, offset_x, offset_y, yaw) in enumerate(
        (
            (25, 0.30, 0.10, 0.000, 0.000, 0.0),
            (21, 0.25, 0.18, 0.008, 0.004, 13.0),
        )
    ):
        cloths.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=str(cloth_grid_asset(resolution=resolution, size=size)),
                    pos=(offset_x, offset_y, z),
                    euler=(0.0, 0.0, yaw),
                ),
                material=material,
                surface=gs.surfaces.Default(color=(0.20 + 0.15 * layer, 0.40, 0.85 - 0.12 * layer, 1.0)),
            )
        )

    scene.build()
    for _ in range(args.steps):
        scene.step()

    positions = np.concatenate([cloth.get_state().pos.cpu().numpy().reshape(-1, 3) for cloth in cloths], axis=0)
    min_z = float(positions[:, 2].min())
    max_z = float(positions[:, 2].max())
    print(
        "cloth stack:",
        f"layers={len(cloths)}",
        f"steps={args.steps}",
        f"min_z={min_z:.6g}",
        f"max_z={max_z:.6g}",
    )
    if not np.isfinite(positions).all():
        raise RuntimeError("Cloth stack produced non-finite vertex positions")
    if min_z < -1e-4:
        raise RuntimeError(f"Cloth stack penetrated the analytical floor (min_z={min_z:.6g})")


if __name__ == "__main__":
    main()
