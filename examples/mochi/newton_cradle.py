"""Newton's five-ball cradle simulated by the MochiSolver: an elastic impact chain under implicit integration.

Five 1 kg, 5 cm-radius balls hang from 1 m strings with a 2 mm surface gap between neighbors; the first ball is
released from 45 degrees and its momentum should travel through the resting pack and eject only the last ball. The
smooth penalty contact is fully elastic by default (`normal_viscous_damping=0`), so no restitution parameter exists
or is needed - what decides the fidelity of the impact chain is the time discretization. The impact of a 1e9 Pa/m
penalty lasts well under a millisecond, and an implicit integrator damps whatever it does not resolve: backward Euler
at plain 60 Hz steps transfers only ~20% of the swing to the last ball (and loses most of a free pendulum swing to
numerical damping by itself), while BDF2 with 0.33 ms substeps - the defaults here - transfers ~97% with the middle
balls barely moving, ahead of the rigid solver's contact at the same substep count.

Each pendulum is loaded as its own single-body MJCF entity: contact between the links of one entity is disabled (as
between the parts of one mochi actor), so a cradle loaded as a single articulated asset would swing straight through
itself.
"""

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np

import genesis as gs

NUM_BALLS = 5
BALL_X = (-0.204, -0.102, 0.0, 0.102, 0.204)
INITIAL_ANGLE = math.radians(45.0)


def ball_xml(i, x):
    return f"""<mujoco model="cradle_ball_{i}">
  <compiler angle="radian" coordinate="local" inertiafromgeom="false"/>
  <worldbody>
    <body name="ball_{i}" pos="{x} 0 0">
      <joint name="string_{i}" type="hinge" axis="0 1 0" damping="0" armature="0" frictionloss="0" limited="false"/>
      <inertial pos="0 0 -1" mass="1" diaginertia="0.001 0.001 0.001"/>
      <geom name="ball_{i}" type="sphere" pos="0 0 -1" size="0.05" rgba="{0.05 + 0.025 * i} {0.2 + 0.15 * i} {0.9 - 0.1375 * i} 1"/>
      <geom name="string_visual_{i}" type="capsule" fromto="0 0 -0.05 0 0 -0.95" size="0.004"
            contype="0" conaffinity="0" density="0" rgba="0.55 0.45 0.35 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=600, help="Number of 60 Hz frames")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument("--substeps", type=int, default=50, help="Implicit substeps per 60 Hz frame")
    parser.add_argument(
        "--integrator", choices=("bdf2", "backward_euler"), default="bdf2", help="Time integration scheme"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    frame_dt = 1.0 / 60.0
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=frame_dt, gravity=(0.0, 0.0, -9.81)),
        mochi_options=gs.options.MochiOptions(dt=frame_dt / args.substeps, integrator=args.integrator),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.0, -2.35, 0.25), camera_lookat=(0.0, 0.0, -0.5)),
        show_viewer=args.vis,
    )
    scene.add_entity(
        gs.morphs.Box(size=(0.648, 0.03, 0.03), pos=(0.0, 0.0, 0.0), fixed=True, collision=False),
        material=gs.materials.Mochi.Rigid(),
        surface=gs.surfaces.Default(color=(0.3, 0.3, 0.3)),
    )
    directory = tempfile.mkdtemp(prefix="mochi_cradle_")
    material = gs.materials.Mochi.Rigid(friction=0.01)
    balls = []
    for i, x in enumerate(BALL_X):
        path = Path(directory) / f"ball_{i}.xml"
        path.write_text(ball_xml(i, x))
        balls.append(scene.add_entity(gs.morphs.MJCF(file=str(path)), material=material))
    if args.record:
        cam = scene.add_camera(res=(1280, 720), pos=(0.0, -2.35, 0.25), lookat=(0.0, 0.0, -0.5))
    scene.build()

    balls[0].set_qpos(np.array([INITIAL_ANGLE]))
    if args.record:
        cam.start_recording(save_to_filename="newton_cradle.mp4", fps=60)

    peak = np.array([abs(float(ball.get_qpos().reshape(-1)[0])) for ball in balls])
    for _ in range(args.n_steps):
        scene.step()
        angles = np.array([float(ball.get_qpos().reshape(-1)[0]) for ball in balls])
        peak = np.maximum(peak, np.abs(angles))
        if args.record:
            cam.render()

    if args.record:
        cam.stop_recording()
    peak_deg = np.degrees(peak)
    print(f"Peak swing angles [deg]: {np.array2string(peak_deg, precision=2)}")
    print(f"Last-ball transfer ratio: {peak_deg[-1] / math.degrees(INITIAL_ANGLE):.3f}")


if __name__ == "__main__":
    main()
