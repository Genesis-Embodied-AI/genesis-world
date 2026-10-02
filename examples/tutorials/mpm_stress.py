"""Record elastic Cauchy stress and derive von Mises stress during a box drop.

Run ``python examples/tutorials/mpm_stress.py --gpu --output stress.npz`` to save particle observations.
Stress tensors are in world coordinates and pascals; positions and times are in meters and seconds. This example
uses MPM.Elastic. The readout is an observation without gradients and does not isolate contact forces.
"""

import argparse
import os

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--output", help="Save particle observations to a NumPy .npz file")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=32,
            lower_bound=(-0.3, -0.3, 0.0),
            upper_bound=(0.3, 0.3, 0.6),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -0.8, 0.6),
            camera_lookat=(0.0, 0.0, 0.15),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())
    elastic = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.25),
            size=(0.12, 0.12, 0.12),
        ),
        material=gs.materials.MPM.Elastic(
            E=1e4,
            nu=0.3,
            sampler="regular",
        ),
        surface=gs.surfaces.Default(
            vis_mode="particle",
        ),
    )
    scene.build()

    positions, stresses, von_mises_stresses = [], [], []
    identity = torch.eye(3, dtype=gs.tc_float, device=gs.device)
    horizon = 100 if "PYTEST_VERSION" not in os.environ else 5
    for _ in range(horizon):
        scene.step()
        stress = elastic.get_particles_stress()
        mean_stress = stress.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        deviatoric_stress = stress - mean_stress[..., None, None] * identity
        von_mises = (1.5 * deviatoric_stress.square().sum(dim=(-2, -1))).sqrt()
        if args.output:
            positions.append(tensor_to_array(elastic.get_particles_pos()))
            stresses.append(tensor_to_array(stress))
            von_mises_stresses.append(tensor_to_array(von_mises))

    if args.output:
        np.savez(
            args.output,
            time_s=np.arange(1, horizon + 1) * scene.dt,
            positions_m=np.stack(positions),
            stress_pa=np.stack(stresses),
            von_mises_pa=np.stack(von_mises_stresses),
        )


if __name__ == "__main__":
    main()
