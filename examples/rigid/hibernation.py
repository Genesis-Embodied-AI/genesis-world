"""Islands + hibernation showcase: a body comes to rest, hibernates, and stops costing anything to simulate.

Two scenes exercise the same pipeline:

- ``ducks``: a grid of ducks is dropped onto the ground and scatters. Each duck is its own entity and island, so it
  hibernates the moment it settles; the step rate climbs sharply as the pile comes to rest.
- ``dominos``: a spiral of toppling dominoes loaded from a single MJCF entity. Every domino is a free body of that one
  entity, yet each is its own island, so a domino hibernates whether it is still standing far ahead of the collision
  wave or already fallen behind it. Only the handful of dominoes in the travelling front stay awake.

While a body is awake the full constraint solve runs for its island; once it hibernates that island is skipped by
forward kinematics, forward dynamics, integration, and the constraint solve. The step rate is streamed to a live plot
and, with --record, both the scene and the plot are saved to video.
"""

import argparse
import os
import time

import genesis as gs
from genesis.utils.misc import qd_to_numpy
from genesis.utils.tools import FPSTracker


RECORDING_FPS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("ducks", "dominos"), default="ducks", help="Which showcase to run")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record the scene and step-rate plot to video")
    args = parser.parse_args()

    # Hibernation runs on field storage (performance_mode) and shows its benefit on CPU, where skipping sleeping
    # islands turns directly into a higher serial step rate.
    gs.init(backend=gs.cpu, performance_mode=True)

    # The step is the largest at which the bodies still settle to true rest below the hibernation threshold.
    dt = 6e-3

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=3000,
            use_hibernation=True,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    if args.scene == "ducks":
        n_side = 12
        n_layers = 3
        # A wide footprint only a few ducks deep: dropped from a modest height the ducks scatter and settle mostly
        # separated, so each forms its own island and hibernates independently. A deep pile would couple them into
        # large islands that stay awake, and a high drop would interpenetrate the hulls.
        spacing = 0.3
        layer_gap = 0.15
        z_floor = 0.35
        offset = 0.5 * (n_side - 1) * spacing
        scene.add_entity(gs.morphs.Plane())
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_layers):
                    scene.add_entity(
                        gs.morphs.Mesh(
                            file="meshes/duck.obj",
                            scale=0.02,
                            pos=(i * spacing - offset, j * spacing - offset, z_floor + k * layer_gap),
                            euler=(90.0, 0.0, (i * 37 + j * 53 + k * 71) % 360),
                        ),
                        vis_mode="collision",
                    )
        camera_pos = (0.8 * n_side * spacing, 0.8 * n_side * spacing, 0.5 * n_side * spacing)
        camera_lookat = (0.0, 0.0, -0.1 * n_side * spacing)
    else:
        # One MJCF entity whose worldbody holds the ground plane and a chain of free-body dominoes; the leading domino
        # is tilted so gravity alone kicks off the cascade.
        scene.add_entity(
            gs.morphs.MJCF(
                file="xml/dominos.xml",
            ),
            vis_mode="collision",
        )
        camera_pos = (1.2, -2.2, 1.6)
        camera_lookat = (1.2, 0.0, 0.0)

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=45,
            GUI=False,
        )

    # Two correlated signals streamed to a live plot: the step rate climbs as bodies fall asleep, and the number of
    # awake bodies tracks the moving collision front. The second subplot makes the FPS <-> hibernation link explicit.
    fps_tracker = FPSTracker(n_envs=0, alpha=0.0)
    step_rate = [0.0]
    n_awake = [0]

    def plot_data():
        return {"step_rate": [step_rate[0]], "awake_bodies": [n_awake[0]]}

    scene.add_recorder(
        plot_data,
        gs.recorders.MPLLinePlot(
            labels={"step_rate": ["steps/s"], "awake_bodies": ["awake bodies"]},
            history_length=10000,
            hz=RECORDING_FPS,
            title="Islands + hibernation",
            save_to_filename=f"out/hibernation_{args.scene}_fps.mp4" if args.record else None,
        ),
    )

    scene.build(n_envs=1)

    n_bodies = sum(1 for link in scene.rigid_solver.links if link.n_dofs > 0)
    n_awake[0] = n_bodies

    # Recording renders from within 'scene.step', so the reported step rate includes the cost of capturing a frame
    # every 1 / RECORDING_FPS of simulated time. Compare rates between scenes recorded the same way, or drop
    # '--record' to read the physics-only rate.
    if args.record:
        camera.start_recording(save_to_filename=f"out/hibernation_{args.scene}.mp4", fps=RECORDING_FPS)
    sim_clock = 0.0
    # Long enough to show the full settle (ducks) or the whole travelling cascade and its re-sleep (dominos).
    sim_seconds = 7.0 if args.scene == "ducks" else 9.0
    n_steps = int(sim_seconds / dt) if "PYTEST_VERSION" not in os.environ else 5
    for _ in range(n_steps):
        tic = time.perf_counter()
        scene.step()
        sim_clock += time.perf_counter() - tic
        measured = fps_tracker.step(sim_clock)
        if measured is not None:
            step_rate[0] = measured
        # Read the awake count outside the timed region so it does not enter the reported step rate.
        n_awake[0] = n_bodies - int(qd_to_numpy(scene.rigid_solver.dyn_state.links.is_hibernated, transpose=True).sum())
    if args.record:
        camera.stop_recording()

    gs.logger.info(
        f"{n_bodies - n_awake[0]}/{n_bodies} bodies hibernated; final step rate {step_rate[0]:,.0f} steps/s."
    )


if __name__ == "__main__":
    main()
