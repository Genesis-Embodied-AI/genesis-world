"""A wrecking ball on a real chain swings into a pile of stacked cubes.

The chain is an MJCF articulation built inline: interlocking stadium-shaped links, each a free body, alternate their
plane by a quarter turn and hold together by contact alone. The top link hangs through a fixed hook and the last one
through an eye welded to a dense steel sphere. Every link is a non-convex mesh, so its collision geometry is
convex-decomposed at build time. The whole chain starts pulled back as a straight line inclined from the vertical, so
gravity alone swings the ball into the pile, whose cubes are free boxes stacked with a hairline gap.
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

import genesis as gs
from genesis.utils.misc import qd_to_numpy


DT = 1e-2
RECORDING_FPS = 30

# Chain link: tube radius, radius of the semicircular ends of the centreline, length of its straight sides, and the
# number of links between the hook and the eye. The release angle is measured from the vertical.
RING_TUBE_RADIUS = 0.02
RING_END_RADIUS = 0.055
RING_SIDE_LENGTH = 0.2
N_RINGS = 9
RING_DENSITY = 7800.0
BALL_RADIUS = 0.35
BALL_DENSITY = 2000.0
RELEASE_ANGLE_DEG = 70.0

# Pile of cubes: edge length and the gap that keeps neighbours from starting in contact.
CUBE_SIZE = 0.25
CUBE_GAP = 2e-3


def ring_mesh():
    """Stadium-shaped chain link lying in the x-z plane with its straight sides along z: a tube swept along a closed
    centreline of two straight sides joined by semicircular ends."""
    n_side = 6
    n_end = 12
    n_around = 10
    # Centreline points and outward normals in the x-z plane, walking the right side up, the top end, the left side
    # down and the bottom end.
    z_side = np.linspace(-0.5 * RING_SIDE_LENGTH, 0.5 * RING_SIDE_LENGTH, n_side, endpoint=False)
    angle_end = np.linspace(0.0, np.pi, n_end, endpoint=False)
    normals_end = np.stack((np.cos(angle_end), np.sin(angle_end)), axis=-1)
    normals = np.concatenate(
        (
            np.tile((1.0, 0.0), (n_side, 1)),
            normals_end,
            np.tile((-1.0, 0.0), (n_side, 1)),
            -normals_end,
        )
    )
    centers = RING_END_RADIUS * normals
    centers[:, 1] += np.concatenate(
        (z_side, np.full(n_end, 0.5 * RING_SIDE_LENGTH), -z_side, np.full(n_end, -0.5 * RING_SIDE_LENGTH))
    )
    # Sweep the tube section: one circle per centreline point, spanned by the outward normal and the y axis.
    angle_tube = np.linspace(0.0, 2.0 * np.pi, n_around, endpoint=False)
    verts = np.empty((len(centers), n_around, 3))
    verts[..., 0] = centers[:, None, 0] + RING_TUBE_RADIUS * np.cos(angle_tube) * normals[:, None, 0]
    verts[..., 1] = RING_TUBE_RADIUS * np.sin(angle_tube)
    verts[..., 2] = centers[:, None, 1] + RING_TUBE_RADIUS * np.cos(angle_tube) * normals[:, None, 1]
    # Two triangles per quad of the periodic grid, wound so the normals point out of the tube.
    i_along, i_around = np.meshgrid(np.arange(len(centers)), np.arange(n_around), indexing="ij")
    corner_00 = i_along * n_around + i_around
    corner_01 = i_along * n_around + (i_around + 1) % n_around
    corner_10 = (i_along + 1) % len(centers) * n_around + i_around
    corner_11 = (i_along + 1) % len(centers) * n_around + (i_around + 1) % n_around
    faces = np.concatenate(
        (
            np.stack((corner_00, corner_11, corner_10), axis=-1).reshape(-1, 3),
            np.stack((corner_00, corner_01, corner_11), axis=-1).reshape(-1, 3),
        )
    )
    return trimesh.Trimesh(verts.reshape(-1, 3), faces, process=False)


def wrecking_ball_mjcf():
    """MJCF model of the wrecking ball, hung so that the sphere centre passes through the origin at the bottom of the
    swing, with the chain taut."""
    ring = ring_mesh()
    half_length = 0.5 * RING_SIDE_LENGTH + RING_END_RADIUS
    # Successive link centres along the chain when taut, the tube of one link against the inner end of the next. The
    # links are laid out with one tube radius of slack per link so none starts in contact.
    pitch_taut = 2.0 * (half_length - RING_TUBE_RADIUS)
    pitch = pitch_taut - RING_TUBE_RADIUS
    eye_to_sphere = half_length + BALL_RADIUS - RING_TUBE_RADIUS

    mjcf = ET.Element("mujoco", model="wrecking_ball")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(
        asset,
        "mesh",
        name="ring",
        vertex=" ".join(f"{coord:.6g}" for coord in ring.vertices.ravel()),
        face=" ".join(str(idx) for idx in ring.faces.ravel()),
    )
    ring_class = ET.SubElement(ET.SubElement(mjcf, "default"), "default", {"class": "ring"})
    ET.SubElement(ring_class, "geom", type="mesh", mesh="ring", density=f"{RING_DENSITY}", rgba="0.38 0.38 0.4 1")
    worldbody = ET.SubElement(mjcf, "worldbody")
    # The chain hangs along -z from a hook one taut chain above the origin. Rotating the frame about +y by the release
    # angle tilts the whole chain towards -x, so the ball swings towards +x, where the pile stands.
    chain_length = (N_RINGS + 1) * pitch_taut + eye_to_sphere
    frame = ET.SubElement(worldbody, "frame", pos=f"0 0 {chain_length}", euler=f"0 {RELEASE_ANGLE_DEG} 0")
    ET.SubElement(frame, "geom", {"class": "ring"})
    for i_ring in range(N_RINGS):
        body = ET.SubElement(frame, "body", pos=f"0 0 {-(i_ring + 1) * pitch}", euler=f"0 0 {90 * ((i_ring + 1) % 2)}")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", {"class": "ring"})
    eye = ET.SubElement(frame, "body", pos=f"0 0 {-(N_RINGS + 1) * pitch}", euler=f"0 0 {90 * ((N_RINGS + 1) % 2)}")
    ET.SubElement(eye, "freejoint")
    ET.SubElement(eye, "geom", {"class": "ring"})
    # The sphere sits in a jointless child body of the eye, so its collision geometry is resolved on its own: fused
    # with the ring it would be swallowed by a single hull and the eye would lose its hole.
    sphere = ET.SubElement(eye, "body", name="ball", pos=f"0 0 {-eye_to_sphere}")
    ET.SubElement(
        sphere, "geom", type="sphere", size=f"{BALL_RADIUS}", density=f"{BALL_DENSITY}", rgba="0.3 0.3 0.32 1"
    )
    return mjcf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pile-width", type=int, default=6, help="Cubes across the pile, facing the ball")
    parser.add_argument("--pile-depth", type=int, default=7, help="Cubes through the pile, along the swing")
    parser.add_argument("--pile-height", type=int, default=5, help="Cubes up the pile")
    parser.add_argument("-s", "--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--hibernation", action="store_true", help="Put the bodies at rest to sleep")
    parser.add_argument(
        "-r",
        "--record",
        action="store_true",
        help="Record the scene and the step-rate plot to 'out/wrecking_ball*.mp4'",
    )
    args = parser.parse_args()
    # The step rate the plot shows is averaged over a fifth of a simulated second
    timings_window = round(0.2 / DT)
    if args.steps <= timings_window:
        parser.error(f"--steps must exceed {timings_window}, the number of steps the step rate is averaged over.")
    horizon = timings_window + 1 if "PYTEST_VERSION" in os.environ else args.steps

    # The step rate is the point of the script, so the solver runs on field storage, its fastest layout on CPU.
    gs.init(backend=gs.gpu if args.gpu else gs.cpu, performance_mode=True)

    camera_pos = (-4.6, -6.3, 2.4)
    camera_lookat = (0.8, 0.0, 0.6)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        rigid_options=gs.options.RigidOptions(
            # Once the pile is compressed every cube can touch its six neighbours and the ground.
            max_collision_pairs=20 * args.pile_width * args.pile_depth * args.pile_height + 500,
            use_hibernation=args.hibernation,
        ),
        vis_options=gs.options.VisOptions(
            # The camera looks along +x and +y, so the light shines the same way to lift the faces it sees.
            lights=[
                {"type": "directional", "dir": (1.0, 1.0, -1.5), "color": (1.0, 1.0, 1.0), "intensity": 5.0},
            ],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
            timings_window=timings_window,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    # The ball passes through the model origin at the bottom of the swing, placed at mid-pile height.
    scene.add_entity(
        gs.morphs.MJCF(
            pos=(0.0, 0.0, 0.5 * args.pile_height * CUBE_SIZE),
            file=wrecking_ball_mjcf(),
        ),
    )

    # The pile stands just past the bottom of the swing, where the ball is at its fastest. Its front face sits closer
    # than the ball radius so the impact lands before the ball starts climbing again.
    cube_pitch = CUBE_SIZE + CUBE_GAP
    x_front = 0.5 * BALL_RADIUS
    for i_x in range(args.pile_depth):
        for i_y in range(args.pile_width):
            for i_z in range(args.pile_height):
                scene.add_entity(
                    gs.morphs.Box(
                        pos=(
                            x_front + (i_x + 0.5) * cube_pitch,
                            (i_y - 0.5 * (args.pile_width - 1)) * cube_pitch,
                            i_z * cube_pitch + 0.5 * CUBE_SIZE,
                        ),
                        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                    ),
                    surface=gs.surfaces.Rough(
                        color=(0.62, 0.48, 0.42),
                    ),
                )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
        )

    # The step rate of the physics alone, read from the scene's timings, and the number of awake bodies, streamed to a
    # live plot.
    plot_values = {"step_rate": [math.nan], "awake_bodies": [0]}

    def plot_data():
        return plot_values

    scene.add_recorder(
        plot_data,
        gs.recorders.MPLLinePlot(
            labels={"step_rate": ["steps/s"], "awake_bodies": ["awake bodies"]},
            history_length=10000,
            hz=RECORDING_FPS,
            title="Wrecking ball",
            y_log_scale=("step_rate",),
            save_to_filename="out/wrecking_ball_fps.mp4" if args.record else None,
        ),
    )

    scene.build()

    n_bodies = sum(1 for link in scene.rigid_solver.links if link.n_dofs > 0)
    plot_values["awake_bodies"][0] = n_bodies
    if camera is not None:
        camera.start_recording(save_to_filename="out/wrecking_ball.mp4", fps=RECORDING_FPS)
    # The step rate of the physics alone, read once the first step, which carries the kernel compilation, has left
    # the averaging window
    step_times = np.full(horizon, np.nan)
    for i_step in range(horizon):
        scene.step()
        if i_step >= timings_window:
            step_times[i_step] = scene.timings["physics"]
            plot_values["step_rate"][0] = 1.0 / step_times[i_step]
        plot_values["awake_bodies"][0] = n_bodies - qd_to_numpy(scene.rigid_solver.dyn_state.links.is_hibernated).sum()
    if camera is not None:
        camera.stop_recording()

    step_times = step_times[timings_window:]
    n_cubes = args.pile_width * args.pile_depth * args.pile_height
    gs.logger.info(
        f"{n_cubes} cubes: mean step {1e3 * step_times.mean():.2f} ms, slowest window {1e3 * step_times.max():.2f} ms, "
        f"real-time factor {DT / step_times.mean():.2f} (mean), {DT / step_times.max():.2f} (slowest window)."
    )


if __name__ == "__main__":
    main()
