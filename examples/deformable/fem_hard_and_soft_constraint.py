import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

import genesis as gs

SCENE_POS = np.array([0.5, 0.5, 1.0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-3, help="Simulation time step")
    parser.add_argument("--substeps", type=int, default=1, help="Number of solver substeps per step")
    parser.add_argument("-t", "--seconds", type=float, default=5, help="Seconds to simulate")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()
    args.seconds = 0.01 if "PYTEST_VERSION" in os.environ else args.seconds

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
            gravity=(0, 0, -9.81),
        ),
        # The linear corotated model these entities use only supplies the energy derivatives the implicit solver
        # needs, so the explicit solver cannot integrate this scene.
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    blob_pos = SCENE_POS + np.array([-0.3, -0.3, 0])
    blob_fixed_bounds = np.array([[-0.1, -0.1, 0.085], [0.1, 0.1, 0.1]])
    cylinder_pos = SCENE_POS + np.array([-0.3, 0.3, 0])
    cylinder_fixed_pos = np.array([0.05, 0.0, 0.1])
    blob_options = gs.entities.FEM(
        morph=gs.morphs.Sphere(
            pos=blob_pos,
            radius=0.1,
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e4,
            nu=0.45,
            rho=1000.0,
            model="linear_corotated",
        ),
    )
    cube_options = gs.entities.FEM(
        morph=gs.morphs.Box(
            pos=SCENE_POS + np.array([0.3, 0.3, 0]),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e6,
            nu=0.45,
            rho=1000.0,
            model="linear_corotated",
        ),
    )
    cylinder_options = gs.entities.FEM(
        morph=gs.morphs.Cylinder(
            pos=cylinder_pos,
            height=0.2,
            radius=0.05,
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e5,
            nu=0.45,
            rho=1000.0,
            model="linear_corotated",
        ),
    )

    blob_options.select(gs.indices.Vertex.in_box(*blob_fixed_bounds)).set_is_fixed(True)
    cylinder_options.select(gs.indices.Vertex.closest_to(cylinder_fixed_pos, max_distance=0.01)).set_is_fixed(True)

    blob = scene.add_entity(options=blob_options)
    cube = scene.add_entity(options=cube_options)
    cylinder = scene.add_entity(options=cylinder_options)

    # Recording every simulation step would be far denser than any player needs, and the step size here is tiny.
    video_fps = min(100.0, 1.0 / args.dt)

    cam = scene.add_camera(
        res=(640, 480),
        pos=(-2.0, 3.0, 2.0),
        lookat=SCENE_POS + np.array([0.0, 0.0, -0.8]),
        fov=30,
        debug=True,
    )

    scene.build()

    video_filename = f"out/fem_hard_soft_dt={args.dt}_substeps={args.substeps}.mp4"
    cam.start_recording(save_to_filename=video_filename, fps=video_fps)

    pinned_idx = [0]
    circle_radius = 0.3

    circle_period = 10.0
    angle_step = 2 * np.pi * args.dt / circle_period
    current_angle = 0.0

    initial_vertex_pos = cube.init_positions[pinned_idx]
    circle_center = initial_vertex_pos - torch.tensor(
        [-circle_radius * np.cos(current_angle), -circle_radius * np.sin(current_angle), 0.0],
        device=cube.init_positions.device,
        dtype=cube.init_positions.dtype,
    )

    def get_next_circle_position():
        """Get next position on circular path with incremental step."""
        nonlocal current_angle
        offset = torch.tensor(
            [-circle_radius * np.cos(current_angle), -circle_radius * np.sin(current_angle), 0.0],
            device=cube.init_positions.device,
            dtype=cube.init_positions.dtype,
        )
        current_angle += angle_step
        return circle_center + offset

    debug_circle = None
    total_steps = int(args.seconds / args.dt)

    scene.draw_debug_box(bounds=blob_pos + blob_fixed_bounds, color=(1, 0, 0, 0.3), wireframe=False)
    scene.draw_debug_spheres(poss=[cylinder_pos + cylinder_fixed_pos], radius=0.01, color=(1, 0, 0, 1))

    try:
        target_positions = blob.init_positions[pinned_idx]
        blob.set_vertex_constraints(pinned_idx, target_positions, is_soft_constraint=True, stiffness=1e4)

        target_positions = get_next_circle_position()
        debug_circle = scene.draw_debug_spheres(poss=target_positions, radius=0.01, color=(1, 0, 0, 1))
        cube.set_vertex_constraints(pinned_idx, target_positions)

        for _ in tqdm(range(total_steps), total=total_steps):
            if debug_circle is not None:
                scene.clear_debug_object(debug_circle)

            new_pos = get_next_circle_position()
            debug_circle = scene.draw_debug_spheres(poss=new_pos, radius=0.01, color=(1, 0, 0, 1))
            cube.update_constraint_targets(pinned_idx, new_pos)

            scene.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        cam.stop_recording()
        gs.logger.info(f"Saved video to {video_filename}")


if __name__ == "__main__":
    main()
