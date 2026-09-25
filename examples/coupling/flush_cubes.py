import os
import argparse
import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=20,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=64,
            lower_bound=(-0.45, -0.65, -0.01),
            upper_bound=(0.45, 0.65, 1.0),
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
            rendered_envs_idx=[0],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.5, 1.0, 1.42),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=22,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    cube0 = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            rho=400,
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.25, 0.4),
            size=(0.12, 0.12, 0.12),
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 0.5, 0.5, 1.0),
            vis_mode="particle",
        ),
    )

    cube0 = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            rho=400,
        ),
        morph=gs.morphs.Sphere(
            pos=(0.15, 0.45, 0.5),
            radius=0.06,
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 1.0, 0.5, 1.0),
            vis_mode="particle",
        ),
    )

    cube0 = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            rho=400,
        ),
        morph=gs.morphs.Cylinder(
            pos=(-0.15, 0.45, 0.6),
            radius=0.05,
            height=0.14,
        ),
        surface=gs.surfaces.Rough(
            color=(0.5, 1.0, 1.0, 1.0),
            vis_mode="particle",
        ),
    )
    emitter1 = scene.add_emitter(
        material=gs.materials.MPM.Liquid(
            sampler="random",
        ),
        max_particles=800000,
        surface=gs.surfaces.Rough(
            color=(0.0, 0.9, 0.4, 1.0),
        ),
    )
    emitter2 = scene.add_emitter(
        material=gs.materials.MPM.Liquid(
            sampler="random",
        ),
        max_particles=800000,
        surface=gs.surfaces.Rough(
            color=(0.0, 0.4, 0.9, 1.0),
        ),
    )
    scene.build(n_envs=0)

    horizon = 100 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        emitter1.emit(
            pos=np.array([0.16, -0.4, 0.5]),
            direction=np.array([0.0, 0.0, -1.0]),
            speed=1.5,
            droplet_shape="circle",
            droplet_size=0.16,
        )
        emitter2.emit(
            pos=np.array([-0.16, -0.4, 0.5]),
            direction=np.array([0.0, 0.0, -1.0]),
            speed=1.5,
            droplet_shape="circle",
            droplet_size=0.16,
        )
        scene.step()


if __name__ == "__main__":
    main()
