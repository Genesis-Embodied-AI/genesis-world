"""
IPC Cloth Simulation Example

This example demonstrates cloth simulation using IPC (Incremental Potential Contact)
with Genesis. The cloth is simulated using NeoHookeanShell constitution.
"""

import argparse
import os

from huggingface_hub import snapshot_download

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,  # Contact barrier distance (10mm) - must be appropriate for mesh resolution
            two_way_coupling=True,  # Enable two-way coupling (forces from IPC to Genesis rigid bodies)
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 2.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=args.vis,
    )

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Cloth using Cloth material
    # Note: Using coarse grid mesh to avoid IPC thickness violations
    # The built-in cloth.obj is too dense for IPC's contact detection
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="8aa8fcd60500b9f3a36c356080224bdb1be9ee59",
        allow_patterns="IPC/grid20x20.obj",
        max_workers=1,
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.5,
            pos=(0.0, 0.0, 1.0),
            euler=(120, -30, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e5,  # Young's modulus (Pa) - soft cloth (100 kPa)
            nu=0.499,  # Poisson's ratio - nearly incompressible
            rho=200,  # Density (kg/m^3)
            thickness=0.001,  # Shell thickness (m) - 1mm
            bending_stiffness=50.0,  # Bending resistance
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    cube_size = 0.2
    cube_height = 0.3  # Height below cloth
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.25, 0, cube_height),
            size=(cube_size, cube_size, cube_size),
        ),
        material=gs.materials.Rigid(
            rho=500,
            coup_type="ipc_only",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )

    # Optional: Add another FEM volume object
    soft_ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.25, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e3,
            nu=0.3,
            rho=1000.0,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.3, 0.8),
        ),
    )

    scene.build(n_envs=1)

    # Simulation loop
    horizon = 100 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
