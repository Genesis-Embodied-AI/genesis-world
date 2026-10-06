"""Nut driven along a fixed bolt by a screw constraint: seated on the head by its limit, then unscrewed off the tip.

The bolt and nut are the meshes of `bolt_nut_self_screw.py`, which screws the nut through the contact of their threads.
Here a screw constraint couples the travel of the nut along the bolt to its turn about it instead (3 mm a turn,
right-handed), with a lower limit where the nut seats on the head and a dry friction that holds it wherever it is left.
The bolt and the nut stop colliding while the constraint holds, so their convexified collision geometry may overlap. The
bolt lies horizontal, so the constraint also carries the weight of the nut. Once the nut has been unscrewed past the tip
of the bolt, the constraint is deleted and the nut drops to the ground.
"""

import argparse
import math
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the scene to out/ as a video")
    parser.add_argument("--torque", type=float, default=0.012, help="Driving torque about the bolt [N*m]")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.12, -0.2, 0.12),
            camera_lookat=(0.02, 0.0, 0.04),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Realistic steel density (the default is far too light), so the weight of the nut matches a real fastener.
    steel = gs.materials.Rigid(rho=7850.0)

    # Bolt: fixed, its axis along x, 5 cm above the ground. The mesh runs along its own z from the bottom of the head
    # (-11 mm) to the tip of the shaft (+32 mm).
    bolt = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bolt_nut/bolt.stl",
            pos=(0.0, 0.0, 0.05),
            euler=(0.0, 90.0, 0.0),
            fixed=True,
        ),
        material=steel,
    )
    # Nut: 18 mm long along the same axis, its origin on it and its base 13 mm along the shaft. Brass-colored, to stand out
    # against the bolt.
    nut = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bolt_nut/nut.stl",
            pos=(0.013, 0.0, 0.05),
            euler=(0.0, 90.0, 0.0),
        ),
        material=steel,
        surface=gs.surfaces.Default(
            color=(0.85, 0.65, 0.25),
        ),
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=(0.12, -0.2, 0.12),
            lookat=(0.02, 0.0, 0.04),
            fov=35,
            GUI=False,
        )

    scene.build()

    # The travel of the nut counts from where it starts: its base reaches the head at -13 mm (the seat) and clears the tip
    # at +19 mm, so the constraint is released a little further, at +21 mm. The driving torque presses the seated nut
    # onto the head with tens of newtons through the thread, so the constraint takes the stiffest impedance (the default
    # one would let the nut sink a millimeter or so into the head).
    seat, release = -0.013, 0.021
    solver = scene.sim.rigid_solver
    solver.add_screw_constraint(
        bolt.base_link.idx,
        nut.base_link.idx,
        axis=(0.0, 0.0, 1.0),
        pitch=3.0e-3 / (2.0 * math.pi),
        limit=(seat, float("inf")),
        frictionloss=0.01,
        sol_params=(0.0, 1.0, 0.9999, 0.9999, 0.001, 0.5, 2.0),
    )

    # Screw the nut onto the head until it seats, hold it there for a moment, then unscrew it until it leaves the thread.
    # The torque is applied about the axis of the nut, its local z: a positive turn moves a right-handed nut towards the
    # tip.
    horizon = 400 if "PYTEST_VERSION" not in os.environ else 5
    if args.record:
        camera.start_recording(save_to_filename="out/bolt_nut_screw_constraint.mp4", fps=30)
    torque = -args.torque
    is_screwed, n_seated = True, 0
    for i in range(horizon):
        if is_screwed:
            travel = float(solver.get_screw_constraints()["travel"][0, 0])
            if travel < seat + 1e-4:
                n_seated += 1
                if n_seated == 50:
                    torque = args.torque
            if travel > release:
                solver.delete_screw_constraint(bolt.base_link.idx, nut.base_link.idx)
                is_screwed, torque = False, 0.0
                print(f"step {i:4d}  released past the tip, the nut is free")
            if i % 25 == 0:
                print(f"step {i:4d}  travel = {travel * 1e3:6.2f} mm")
        elif i % 25 == 0:
            print(f"step {i:4d}  free nut at z = {float(nut.get_pos()[2]) * 1e3:6.2f} mm")
        solver.apply_links_external_wrench(torque=(0.0, 0.0, torque), links_idx=(nut.base_link.idx,), local=True)
        scene.step()
    if args.record:
        camera.stop_recording()


if __name__ == "__main__":
    main()
