"""Nut screwed onto a fixed bolt by a screw constraint, seated on the bolt head, then unscrewed off the tip.

Without the viewer, this runs a scripted sequence. With it, the keyboard drives the nut:
Up      - Turn the nut clockwise, seen from the tip of the bolt
Down    - Turn the nut counterclockwise, seen from the tip of the bolt
\\       - Reset the nut onto the bolt
Esc     - Quit
"""

import argparse
import math
import os

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

NUT_START_POS = (0.013, 0.0, 0.05)
NUT_SEAT_X = 0.0
NUT_RELEASE_X = 0.034


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--torque", type=float, default=0.003, help="Maximum driving torque about the nut axis [N*m]")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_screw_constraints=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.12, -0.2, 0.12),
            camera_lookat=(0.02, 0.0, 0.04),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    steel = gs.materials.Rigid(rho=7850.0)  # Density of steel

    bolt = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bolt_nut/bolt.stl",
            pos=(0.0, 0.0, 0.05),
            euler=(0.0, 90.0, 0.0),
            fixed=True,
        ),
        material=steel,
    )
    nut = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bolt_nut/nut.stl",
            pos=NUT_START_POS,
            euler=(0.0, 90.0, 0.0),
        ),
        material=steel,
        surface=gs.surfaces.Default(
            color=(0.85, 0.65, 0.25),
        ),
    )

    scene.build()

    rigid = scene.sim.rigid_solver
    bolt_idx = bolt.base_link.idx
    nut_idx = nut.base_link.idx
    nut_qpos = nut.get_qpos()

    def add_screw():
        rigid.add_screw_constraint(
            bolt_idx,
            nut_idx,
            axis=(0.0, 0.0, 1.0),
            pitch=3.0e-3,
            limit=(NUT_SEAT_X - NUT_START_POS[0], math.inf),
            frictionloss=0.002,
        )

    add_screw()
    is_screwed = True
    is_running = True
    is_reset_requested = False
    spin_direction = 0

    # Keybind callbacks run on the viewer thread, so they only set the request that the main loop carries out between
    # steps. A positive turn about the local z axis of the nut moves it towards the tip.
    def drive(direction):
        nonlocal spin_direction
        spin_direction = direction

    def request_reset():
        nonlocal is_reset_requested
        is_reset_requested = True

    def stop():
        nonlocal is_running
        is_running = False

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("turn_clockwise", Key.UP, KeyAction.PRESS, callback=drive, args=(-1,)),
            Keybind("turn_clockwise_stop", Key.UP, KeyAction.RELEASE, callback=drive, args=(0,)),
            Keybind("turn_counterclockwise", Key.DOWN, KeyAction.PRESS, callback=drive, args=(1,)),
            Keybind("turn_counterclockwise_stop", Key.DOWN, KeyAction.RELEASE, callback=drive, args=(0,)),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=request_reset),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        horizon = math.inf
    else:
        # Screw the nut down onto the head, hold it seated for half a second, then unscrew it off the thread.
        horizon = 1000 if "PYTEST_VERSION" not in os.environ else 5
        drive(-1)

    n_seated = 0
    i_step = 0
    while is_running and i_step < horizon and (not args.vis or scene.viewer.is_alive()):
        if is_reset_requested:
            if is_screwed:
                rigid.delete_screw_constraint(bolt_idx, nut_idx)
            nut.set_qpos(nut_qpos)
            add_screw()
            is_screwed = True
            is_reset_requested = False

        nut_x = nut.get_pos()[0]
        if not args.vis and is_screwed and nut_x < NUT_SEAT_X + 1e-4:
            n_seated += 1
            if n_seated == 50:
                drive(1)
        if is_screwed and nut_x > NUT_RELEASE_X:
            rigid.delete_screw_constraint(bolt_idx, nut_idx)
            is_screwed = False
            gs.logger.info(f"step {i_step:4d}  released past the tip, the nut is free")
        elif i_step % 25 == 0:
            travel = nut_x - NUT_START_POS[0]
            gs.logger.info(f"step {i_step:4d}  travel = {travel * 1e3:6.2f} mm  z = {nut.get_pos()[2] * 1e3:6.2f} mm")

        torque = 0.0
        if spin_direction != 0:
            spin_rate = gu.inv_transform_by_quat(nut.get_ang(), nut.get_quat())[2]
            torque = min(max(2.0e-3 * (spin_direction * 4.0 * math.pi - spin_rate), -args.torque), args.torque)
        rigid.apply_links_external_wrench(torque=(0.0, 0.0, torque), links_idx=(nut_idx,), local=True)
        scene.step()
        i_step += 1


if __name__ == "__main__":
    main()
