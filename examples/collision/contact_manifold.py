"""Multi-contact manifold of a box-on-box interface, recovered from the contact patch or from perturbed re-detections.

Both schemes answer the same question - which points of the touching faces the constraint solver is handed - and they
answer it differently. The contact patch clips the two touching faces against each other inside a single GJK detection
pass and reports the resulting polygon. The perturbation scheme re-runs detection on slightly rotated copies of the
pair and keeps one witness per perturbation axis, so it reports four points whatever shape the overlap has, at the
cost of the extra passes. See the 'enable_contact_patch' option of 'gs.options.RigidOptions'.

The top box is driven kinematically through a scripted sweep - edge tilt, yaw, vertical interpenetration, lateral
slide - so both schemes are exercised on identical poses, and the manifold size is reported whenever it changes.
Under '-v' the manifold is drawn as gold spheres joined into the polygon they span.
"""

import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array


HALF = 0.1  # half-extent of both boxes
TOP_FACE = HALF  # top face of the fixed box, whose center sits one half-extent above the ground plane
CORNERS = HALF * np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
# Interface overlap held by every phase but the vertical sweep, which drives it from there down to PENETRATION_DEEP.
PENETRATION = 1e-3
PENETRATION_DEEP = 0.03
MAX_TILT = 10.0
MAX_YAW = 45.0
MAX_SLIDE = 0.10
MARKER_COLOR = (1.0, 0.78, 0.05, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector",
        type=str,
        default="both",
        choices=("both", "patch", "perturbation"),
        help="Which multi-contact detection scheme to sweep",
    )
    parser.add_argument("-s", "--steps", type=int, default=1200, help="Number of simulation steps per detector")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    # The sweep spans the whole step budget whatever it is, so even a handful of steps visit every phase.
    n_steps = 8 if "PYTEST_VERSION" in os.environ else args.steps
    detectors = ("perturbation", "patch") if args.detector == "both" else (args.detector,)
    for detector in detectors:
        print(f"{detector}:")
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                # The specialized box-box detector bypasses GJK, where both manifold schemes live.
                box_box_detection=False,
                use_gjk_collision=True,
                enable_contact_patch=(detector == "patch"),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.78, -0.70, 0.50),
                camera_lookat=(0.03, 0.0, 0.13),
                camera_fov=32,
            ),
            show_viewer=args.vis,
        )
        scene.add_entity(
            gs.morphs.Plane(
                pos=(0.0, 0.0, -HALF),
            ),
        )
        base = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, 0.0),
                size=(2 * HALF, 2 * HALF, 2 * HALF),
                fixed=True,
            ),
        )
        top = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, TOP_FACE + HALF - PENETRATION),
                size=(2 * HALF, 2 * HALF, 2 * HALF),
            ),
            surface=gs.surfaces.Default(
                color=(0.85, 0.88, 0.92),
                opacity=0.35,
            ),
        )
        scene.build()

        n_contacts = 0
        for i in range(n_steps):
            # One continuous loop: edge tilt, yaw, vertical interpenetration, lateral slide, then unwind to the start.
            t = i / n_steps
            yaw, tilt, slide, penetration = MAX_YAW, 0.0, 0.0, PENETRATION
            if t < 0.20:
                # The tilt is taken at yaw 0, where the rocking edge stays over the fixed box's top face. At yaw 45
                # the deepest corner overhangs that face, so the pair would rest on an edge of the fixed box instead.
                yaw, tilt = 0.0, MAX_TILT * np.sin(np.pi * t / 0.20)
            elif t < 0.45:
                yaw = MAX_YAW * (t - 0.20) / 0.25
            elif t < 0.70:
                penetration = PENETRATION + (PENETRATION_DEEP - PENETRATION) * np.sin(np.pi * (t - 0.45) / 0.25)
            elif t < 0.90:
                slide = MAX_SLIDE * np.sin(np.pi * (t - 0.70) / 0.20)
            else:
                yaw = MAX_YAW * (1.0 - (t - 0.90) / 0.10)

            quat = gu.xyz_to_quat(np.array((tilt, 0.0, yaw)), degrees=True)
            # Hold the deepest corner exactly `penetration` below the fixed box's top face, whatever the orientation.
            z = TOP_FACE - penetration - (CORNERS @ gu.quat_to_R(quat).T)[:, 2].min()
            top.set_qpos((slide, 0.0, z, *quat))
            scene.step()

            positions = tensor_to_array(top.get_contacts(with_entity=base)["position"])
            if len(positions) != n_contacts:
                n_contacts = len(positions)
                print(
                    f"  step {i:5d}  yaw {yaw:5.1f} deg  tilt {tilt:4.1f} deg  "
                    f"penetration {1e3 * penetration:5.2f} mm  manifold {n_contacts} points"
                )
            if args.vis:
                scene.clear_debug_objects()
                scene.draw_debug_spheres(positions, radius=0.006, color=MARKER_COLOR)
                # Every contact lies on the fixed box's horizontal top face, so ordering the manifold into a simple
                # polygon is an angular sort about its centroid in the world xy-plane.
                offsets = positions - positions.mean(axis=0)
                polygon = positions[np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))]
                for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
                    scene.draw_debug_line(start, end, radius=0.0015, color=MARKER_COLOR)


if __name__ == "__main__":
    main()
