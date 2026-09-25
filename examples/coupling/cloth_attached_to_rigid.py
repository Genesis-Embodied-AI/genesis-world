import argparse
import math
import os


import genesis as gs
import genesis.utils.geom as gu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=100 if "PYTEST_VERSION" not in os.environ else 25,
        help="Number of simulation steps",
    )
    parser.add_argument("--num-teleports", type=int, default=5, help="Number of times the attachment is teleported")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="info")

    dt: float = 2e-2
    particle_size: float = 1e-2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=particle_size,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    rigid_material = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

    # create ground plane
    scene.add_entity(gs.morphs.Plane(), rigid_material)

    box_morph = gs.morphs.Box(pos=[0.25, 0.25, 0.25], size=[0.25, 0.25, 0.25])
    box = scene.add_entity(box_morph, rigid_material)

    cloth_pos = (0.25, 0.25, 0.25 + 0.125 + particle_size)
    cloth_scale = 0.5
    cloth_morph = gs.morphs.Mesh(pos=cloth_pos, scale=cloth_scale, file="meshes/cloth.obj")
    cloth_material = gs.materials.PBD.Cloth()
    cloth_surface = gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0))  # , vis_mode="particle")
    cloth = scene.add_entity(cloth_morph, cloth_material, cloth_surface)

    scene.build(n_envs=0)

    particles_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    box_link_idx = box.link_start

    cloth.fix_particles_to_link(box_link_idx, particles_idx_local=particles_idx)

    box.set_dofs_velocity([-0.0, 1.0, 0.0], dofs_idx_local=[0, 1, 2])

    for i in range(args.steps):
        scene.step()

    for j in range(args.num_teleports):
        if j == 2:
            cloth.release_particle(particles_idx)

        new_pos = (0.2 * math.sin(j), 0.2 * math.cos(j), 0.5)
        new_rot = gu.euler_to_quat((-30 * j, -40 * j, 70 * j))
        box.set_pos(new_pos)
        box.set_quat(new_rot)

        for _ in range(args.steps):
            scene.step()


if __name__ == "__main__":
    main()
