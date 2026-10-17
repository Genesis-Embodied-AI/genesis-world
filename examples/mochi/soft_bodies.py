"""Deformable bodies simulated by the MochiSolver: a soft sphere and a soft cube dropped onto the ground, and a rigid
box landing on a soft slab.

Rigid and deformable bodies share one implicit Newton solve: the tetrahedral finite elements (stable neo-Hookean
material) and the penalty contact against the rigid colliders enter the same residual and Hessian, so no coupler is
involved. Contact acts through samples of the deformable surface: mesh the soft bodies finely enough for the rigid
bodies that land on them (`nobisect=False` lets tetgen refine the surface).
"""

import argparse

import genesis as gs


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
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.5, -2.5, 1.5), camera_lookat=(0.0, 0.0, 0.3)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=0.15, pos=(-0.6, 0.0, 0.6), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=5e4, nu=0.45, rho=1000.0),
        surface=gs.surfaces.Default(color=(0.9, 0.5, 0.2)),
    )
    sphere2 = scene.add_entity(
        gs.morphs.Sphere(radius=0.1, pos=(-0.6, 0.0, 1.0), maxvolume=0.0005, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=5e4, nu=0.45, rho=1000.0),
        surface=gs.surfaces.Default(color=(0.9, 0.5, 0.2)),
    )
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(0.0, 0.0, 0.7), euler=(20.0, 30.0, 0.0), maxvolume=0.001),
        material=gs.materials.Mochi.Elastic(E=2e4, nu=0.4, rho=800.0, stiffness_damping=1e-3),
        surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9)),
    )
    slab = scene.add_entity(
        gs.morphs.Box(size=(0.6, 0.6, 0.1), pos=(0.8, 0.0, 0.05), maxvolume=0.0002, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.45, rho=1000.0),
        surface=gs.surfaces.Default(color=(0.5, 0.9, 0.5)),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.15, 0.15, 0.15), pos=(0.8, 0.0, 0.5)),
        material=gs.materials.Mochi.Rigid(rho=1500.0),
    )
    if args.record:
        cam = scene.add_camera(
            res=(640, 360),
            pos=(2.5, -2.5, 1.5),
            lookat=(0.0, 0.0, 0.3),
        )
    scene.build()

    if args.record:
        cam.start_recording(save_to_filename="soft_bodies.mp4", fps=30)

    for i_step in range(args.n_steps):
        scene.step()
        if i_step % 60 == 0:
            info = scene.mochi_solver.get_convergence_info()
            sphere_z = float(sphere.get_vertices_position()[:, 2].min())
            cube_z = float(cube.get_vertices_position()[:, 2].min())
            slab_top = float(slab.get_vertices_position()[:, 2].max())
            print(
                f"step {i_step:4d}: sphere zmin={sphere_z:.4f} cube zmin={cube_z:.4f} slab zmax={slab_top:.4f} "
                f"box z={float(box.get_pos()[2]):.4f} newton iterations={int(info['n_iter'][0])}"
            )
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
