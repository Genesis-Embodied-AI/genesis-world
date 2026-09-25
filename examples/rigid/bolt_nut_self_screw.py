"""Nut screwing itself down a fixed bolt via non-convex thread contact.

The bolt and nut are procedurally generated with matching ISO-metric-style threads (see
genesis/assets/meshes/bolt_nut/generate_bolt_nut.py) and loaded with convexify=False so
the full helix drives collision. The bolt is fixed, shaft up; a steady torque (a wrench)
turns the nut, whose threads convert the rotation into descent until it seats on the head.

Three details matter: decimate=False (the default decimation would erase the helix),
a stiffened constraint_timeconst (the default contact is too soft and the nut slips through
the flanks), and releasing the torque at the seat (driving a seated nut strips and bounces
it; releasing lets the self-locking thread hold it).
"""

import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--torque", type=float, default=-4.0, help="Driving torque about z [N*m]")
    args = parser.parse_args()

    # At this magnitude the steel nut spins faster than the thread contact can resolve and the solve diverges (NaN)
    # even while fully engaged, so reject it outright (contact-loss cases below it are handled by the drive loop).
    if not (-5.3 - 1e-6 < args.torque < 5.3 + 1e-6):
        raise ValueError(f"--torque magnitude must be <= 5.3 N*m to keep the contact stable, got {args.torque}")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
            substeps=20,
        ),
        rigid_options=gs.options.RigidOptions(
            # Fine-thread contact needs a stiff constraint (the default 0.01 is too soft): a softer one lets the nut
            # sink through the flanks and wobble, so it advances faster than the pitch instead of being held to it.
            constraint_timeconst=4e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2, 0.1, 0.1),
            camera_lookat=(0.0, 0.0, 0.03),
            camera_fov=35,
        ),
        show_viewer=args.vis,
        show_FPS=True,
    )

    scene.add_entity(gs.morphs.Plane())

    # Realistic steel density (the default is far too light); the bolt is fixed so its inertia is moot, but the nut's
    # weight then matches a real fastener.
    steel = gs.materials.Rigid(rho=7850.0)

    # Bolt: fixed, hex head down resting on the ground plane, threaded shaft pointing up. The mesh spans z in
    # [-11, 32] mm, so pos_z = 11 mm puts the head bottom at z = 0 (on the plane), the head top at z = 11 mm and the
    # shaft tip at z = 43 mm.
    scene.add_entity(
        gs.morphs.Mesh(
            pos=(0.0, 0.0, 0.011),
            file="meshes/bolt_nut/bolt.stl",
            decimate=False,
            convexify=False,
            fixed=True,
        ),
        material=steel,
        vis_mode="collision",
    )

    # Nut: pre-engaged at the top of the shaft (base at z = 24 mm, top ~42 mm just below the tip), so it screws the
    # whole way down. The threads mesh with clearance at this pose, so contact catches the nut rather than free-falling.
    nut = scene.add_entity(
        gs.morphs.Mesh(
            pos=(0.0, 0.0, 0.024),
            file="meshes/bolt_nut/nut.stl",
            decimate=False,
            convexify=False,
        ),
        material=steel,
        vis_mode="collision",
    )

    scene.build()

    # Drive a steady torque about z (a wrench), leaving the other five DOFs free; the thread contact converts the
    # rotation into axial travel and holds the nut coaxial and upright. The sign sets the direction: negative
    # (clockwise from above) screws the nut down a right-handed thread, positive unscrews it up. The torque is released
    # (like releasing a driver's trigger) and latched off once the nut reaches the end of its travel, since keeping it
    # on against a seated or disengaged nut would strip/bounce it or spin it up into a NaN; the self-locking thread
    # holds the nut once released. Screwing down, the end is the seat: the nut center reaches z ~ 20 mm when its base
    # meets the head top (z = 11 mm), so release just above that to coast onto the head. Unscrewing up, the end is the
    # tip: the center reaches z ~ 48 mm when only a turn of thread is left gripping, so release there to spin off.
    horizon = 800 if "PYTEST_VERSION" not in os.environ else 5
    drive_ratio = 1.0
    z0 = float(nut.get_pos()[2])
    for i in range(horizon):
        nut_z = float(nut.get_pos()[2])
        # Drive until the nut reaches the end of its travel (the seat going down, the tip going up), then latch off.
        if args.torque < 0.0:
            if nut_z < 0.0202:
                drive_ratio = 0.0
        else:
            drive_ratio = min(drive_ratio, 1.0 - min(max((nut_z - 0.04) / (0.048 - 0.04), 0.0), 1.0))
        nut.control_dofs_force(args.torque * drive_ratio, dofs_idx_local=(5,))
        scene.step()
        if i % 50 == 0:
            print(f"step {i:4d}  nut_z = {nut_z * 1e3:6.2f} mm  descended = {(z0 - nut_z) * 1e3:5.2f} mm")


if __name__ == "__main__":
    main()
