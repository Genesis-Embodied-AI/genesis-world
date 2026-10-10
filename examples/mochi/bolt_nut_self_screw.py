"""Nut screwing itself down a fixed bolt via non-convex thread contact, simulated by the MochiSolver.

The mochi counterpart of `examples/rigid/bolt_nut_self_screw.py`: the same procedurally generated ISO-metric-style
bolt and nut (M24 x 3, see genesis/assets/meshes/bolt_nut/generate_bolt_nut.py), the bolt fixed shaft up, a steady
torque turning the free nut whose threads convert the rotation into axial travel. Where the rigid solver needs 1 ms
substeps and a stiffened constraint to hold the nut to the pitch, the implicit solver takes plain 60 Hz steps with no
substeps: the smooth penalty contact between the nut's boundary samples and the bolt's signed distance field enters
the Newton solve of every step, the nut follows the 3 mm pitch exactly (descent rate = wz / 2 pi * pitch) and stays
coaxial to well under a millimeter, and after the torque is released the self-locking thread holds it in place.

Thread contact wants parameters at the scale of the thread, far below the defaults: the radial clearance between the
threads is 0.3 mm, so the penalty threshold and smoothing half-distance (1 mm and 5 mm by default) are shrunk to
0.1 mm and 0.2 mm, and the bolt's distance field is sampled at 0.2 mm cells to resolve the 1.6 mm flanks. Two details
keep every step convergent: the torque ramps up only after the nut has settled into the threads (full torque before
engagement spins the nut through more than a radian per 16 ms step, and the engagement solve turns into a coin toss),
and it is released at the end of the travel (the seat going down, the last turn of thread going up) so the nut coasts
onto the head instead of being ground against it.
"""

import argparse
import os

import numpy as np

import genesis as gs

SETTLE_STEPS = 12
RAMP_STEPS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=400, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument("--torque", type=float, default=-1.0, help="Driving torque about z [N*m]")
    args = parser.parse_args()

    # Verified stable over the full travel for magnitudes up to 4 N*m in either direction.
    if not (0.0 < abs(args.torque) <= 4.0):
        raise ValueError(f"--torque magnitude must be in (0, 4] N*m, got {args.torque}")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2, 0.1, 0.1),
            camera_lookat=(0.0, 0.0, 0.03),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())

    # Contact at the scale of the thread: the ramp of the penalty must live inside the 0.3 mm thread clearance, and
    # the bolt's distance field must resolve the flanks.
    contact = {
        "friction": 0.2,
        "penalty_coefficient": 1e9,
        "penalty_threshold": 1e-4,
        "penalty_smoothing_half_distance": 2e-4,
    }
    steel_bolt = gs.materials.Mochi.Rigid(rho=7850.0, sdf_cell_size=2e-4, sdf_max_res=256, **contact)
    # The bolt is fixed, so it provides no contact samples: nothing ever queries the nut's own collider, and the nut
    # collides with the plane through its samples as well.
    steel_nut = gs.materials.Mochi.Rigid(rho=7850.0, collider_type="none", **contact)

    # Bolt: fixed, hex head down resting on the ground plane, threaded shaft pointing up (head top at z = 11 mm,
    # shaft tip at z = 43 mm); nut pre-engaged near the top of the shaft. The helix must survive loading untouched.
    scene.add_entity(
        gs.morphs.Mesh(
            pos=(0.0, 0.0, 0.011),
            file="meshes/bolt_nut/bolt.stl",
            decimate=False,
            convexify=False,
            fixed=True,
        ),
        material=steel_bolt,
    )
    nut = scene.add_entity(
        gs.morphs.Mesh(
            pos=(0.0, 0.0, 0.024),
            file="meshes/bolt_nut/nut.stl",
            decimate=False,
            convexify=False,
        ),
        material=steel_nut,
    )
    if args.record:
        cam = scene.add_camera(res=(640, 360), pos=(0.2, 0.1, 0.1), lookat=(0.0, 0.0, 0.03), fov=35)
    scene.build()

    if args.record:
        cam.start_recording(save_to_filename="bolt_nut_self_screw.mp4", fps=30)
    horizon = args.n_steps if "PYTEST_VERSION" not in os.environ else 5
    z0 = float(nut.get_pos()[2])
    drive_on = True
    for i_step in range(horizon):
        nut_z = float(nut.get_pos()[2])
        # Release the torque at the end of the travel and latch it off: screwing down, when the nut base reaches the
        # bolt head; unscrewing, when only the last turn of thread is still gripping. The self-locking thread holds
        # the nut where it stops.
        if (args.torque < 0.0 and nut_z < 0.0114) or (args.torque > 0.0 and nut_z > 0.038):
            drive_on = False
        ramp = 0.0 if i_step < SETTLE_STEPS else min((i_step - SETTLE_STEPS) / RAMP_STEPS, 1.0)
        nut.control_dofs_force(np.array([args.torque * ramp * drive_on]), dofs_idx_local=np.array([5]))
        scene.step()
        if i_step % 40 == 0:
            vel = np.asarray(nut.get_dofs_velocity())
            print(
                f"step {i_step:4d}: nut z = {nut_z * 1e3:6.2f} mm  travelled = {abs(nut_z - z0) * 1e3:5.2f} mm  "
                f"wz = {vel[5]:6.2f} rad/s  drive {'on' if drive_on else 'off'}"
            )
        if args.record:
            cam.render()

    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
