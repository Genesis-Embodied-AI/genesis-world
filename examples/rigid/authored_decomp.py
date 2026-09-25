import argparse
import os

import genesis as gs
from genesis.utils.misc import qd_to_torch

BASE_HEIGHT = 0.020
RING_HEIGHT = 0.020
BALL_HEIGHT = 0.0215
POLE_HEIGHT = 0.145
RINGS_ORDER = (0, 1, 2, 3, 5, 4)
RING_COLORS = [
    (0.95, 0.95, 0.95, 1.0),
    (0.60, 0.80, 0.70, 1.0),
    (0.78, 0.88, 0.80, 1.0),
    (0.90, 0.55, 0.60, 1.0),
    (0.85, 0.72, 0.35, 1.0),
    (0.95, 0.95, 0.95, 1.0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-d", "--debug", action="store_true", help="Draw per-geom centers of mass")
    parser.add_argument("-s", "--steps", type=int, default=2000, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step")
    parser.add_argument("--gjk", action="store_true", help="Use the GJK collision detector")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=args.gjk,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, 0.0, 0.3),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Base/pole rests directly on the plane.
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/base_pole.urdf",
            pos=(0.0, 0.0, BASE_HEIGHT / 2),
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=600.0,
        ),
        vis_mode="collision" if args.debug else "visual",
        visualize_contact=False,
    )

    # Stack the rings in the demo order. A slight overlap (-1e-4) ensures they rest in contact at init.
    height = BASE_HEIGHT
    for ring_idx in RINGS_ORDER:
        scene.add_entity(
            morph=gs.morphs.URDF(
                file=f"tower/ring_{ring_idx + 1:02d}.urdf",
                pos=(0.0, 0.0, height + (RING_HEIGHT - 1e-4) / 2),
                file_meshes_are_zup=True,
            ),
            surface=gs.surfaces.Default(
                color=RING_COLORS[ring_idx],
            ),
            material=gs.materials.Rigid(
                rho=600.0,
            ),
            vis_mode="collision" if args.debug else "visual",
            visualize_contact=args.debug,
        )
        height += RING_HEIGHT - 1e-4

    # Ball caps the pole.
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/ball.urdf",
            pos=(0.0, 0.0, height + BALL_HEIGHT),
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=600.0,
        ),
        vis_mode="collision" if args.debug else "visual",
        visualize_contact=False,
    )

    scene.build()

    if args.debug:
        # geoms.center holds the center of mass in the geom frame, so it needs the geom pose to reach world space.
        geoms_com_local = qd_to_torch(scene.rigid_solver.dyn_info.geoms.center)
        geoms_pos = scene.rigid_solver.get_geoms_pos()
        geoms_quat = scene.rigid_solver.get_geoms_quat()
        obj_com_world = gs.utils.geom.transform_by_trans_quat(geoms_com_local, geoms_pos, geoms_quat)
        scene.draw_debug_spheres(poss=obj_com_world, radius=0.002, color=(1, 1, 1, 1))
        scene.visualizer.update(force=True)

    horizon = args.steps if "PYTEST_VERSION" not in os.environ else 5
    for _ in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
