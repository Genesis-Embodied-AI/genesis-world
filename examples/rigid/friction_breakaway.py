"""Coulomb stiction under both friction cones, documenting the pyramidal cone's limitations.

A box and two spheres rest on a plane, so each contact carries the exact normal load m * g. A constant generalized
force (box, tangential) and torque (spinning sphere, about the vertical axis; rolling sphere, about a horizontal
axis) is applied at a sweep of fractions of the Coulomb limits mu * m * g, mu_torsional * m * g, and
mu_rolling * m * g, and each load is classified as held or slipped from the drift accumulated over the horizon, the
criterion of the static-friction unit tests. The pyramidal cone at its default 'impratio' of 1 mixes the normal
direction into every friction row, so it creeps far past the drift tolerance well below the Coulomb limit,
especially on the torsional and rolling axes; the elliptic cone (auto-resolved 'impratio' of 100) holds near the
limit and slips past it. See the 'friction_cone' and 'impratio' options of 'gs.options.RigidOptions'.
"""

import argparse
import xml.etree.ElementTree as ET

import genesis as gs
from genesis.utils.misc import get_assets_dir


GRAVITY = 9.81
DT = 1e-2
RADIUS = 0.1
# Half-extent of the wooden sphere asset, whose origin sits at the bottom of its y-up frame.
WOODEN_SPHERE_MESH_RADIUS = 3.486346
# Load fractions straddling the Coulomb limit, so both the sub-limit creep and the past-limit slip show up.
LOAD_RATIOS = (0.25, 0.5, 0.75, 0.95, 1.05)
# A load holds when the drift accumulated over the horizon stays below this bound, the criterion of the
# static-friction unit tests: a displacement (meters) for the box, a swept angle (radians) for the sphere.
DRIFT_TOLERANCE = 5e-3
N_STEPS = 300
# Steps simulated past the point where every entity has drifted beyond the tolerance: long enough to make the slip
# visible, short enough that nothing accelerates out of frame.
N_STEPS_SLIPPED = 80


def checkered_ball_mjcf():
    """MJCF model of a free ball: a sphere collision geom with a checker-textured ball mesh as its visual, whose
    colorful pattern makes the rotation visible."""
    mjcf = ET.Element("mujoco", model="checkered_ball")
    asset = ET.SubElement(mjcf, "asset")
    scale = RADIUS / WOODEN_SPHERE_MESH_RADIUS
    ET.SubElement(
        asset,
        "mesh",
        name="ball_visual",
        file=f"{get_assets_dir()}/meshes/wooden_sphere_OBJ/wooden_sphere.obj",
        scale=f"{scale} {scale} {scale}",
    )
    tex_kwargs = {"builtin": "checker", "rgb1": "0.9 0.2 0.2", "rgb2": "1. 0.9 0.3", "width": "128", "height": "128"}
    ET.SubElement(asset, "texture", name="checker", type="cube", **tex_kwargs)
    ET.SubElement(asset, "material", name="checker", texture="checker", texrepeat="4 4")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ball_body = ET.SubElement(worldbody, "body", name="ball", pos="0. 0. 0.")
    ET.SubElement(ball_body, "joint", name="root", type="free")
    ET.SubElement(ball_body, "geom", type="sphere", size=f"{RADIUS}")
    visual_kwargs = {"contype": "0", "conaffinity": "0", "mass": "0.", "euler": "90 0 0", "material": "checker"}
    ET.SubElement(ball_body, "geom", type="mesh", mesh="ball_visual", pos=f"0. 0. -{RADIUS}", **visual_kwargs)
    return ET.tostring(mjcf, encoding="unicode")


def measure_stiction(friction_cone, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=friction_cone,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        ),
    )
    spinner = scene.add_entity(
        gs.morphs.MJCF(
            file=checkered_ball_mjcf(),
            pos=(0.6, 0.0, RADIUS),
        ),
        material=gs.materials.Rigid(
            friction_torsional=0.05,
        ),
    )
    roller = scene.add_entity(
        gs.morphs.MJCF(
            file=checkered_ball_mjcf(),
            pos=(1.2, 0.0, RADIUS),
        ),
        material=gs.materials.Rigid(
            friction_rolling=0.05,
        ),
    )
    scene.build()

    box_load_coulomb = box.geoms[0].get_friction() * box.get_mass() * GRAVITY
    spinner_load_coulomb = spinner.geoms[0].get_friction_torsional() * spinner.get_mass() * GRAVITY
    roller_load_coulomb = roller.geoms[0].get_friction_rolling() * roller.get_mass() * GRAVITY

    is_load_held = []
    for load_ratio in LOAD_RATIOS:
        scene.reset()
        box.control_dofs_force([load_ratio * box_load_coulomb, 0.0, 0.0, 0.0, 0.0, 0.0])
        spinner.control_dofs_force([0.0, 0.0, 0.0, 0.0, 0.0, load_ratio * spinner_load_coulomb])
        roller.control_dofs_force([0.0, 0.0, 0.0, 0.0, load_ratio * roller_load_coulomb, 0.0])
        # Contact warmup under load, then drift accumulated over the horizon (the unit tests' protocol). The sphere
        # drifts integrate the spin speed so multiple slipped turns cannot alias back to a small orientation change.
        for _ in range(50):
            scene.step()
        box_pos_start = float(box.get_pos()[..., 0])
        spinner_swept_angle = 0.0
        roller_swept_angle = 0.0
        n_steps_slipped = 0
        for _ in range(N_STEPS):
            scene.step()
            spinner_swept_angle += abs(float(spinner.get_dofs_velocity()[..., 5])) * DT
            roller_swept_angle += abs(float(roller.get_dofs_velocity()[..., 4])) * DT
            box_drift = abs(float(box.get_pos()[..., 0]) - box_pos_start)
            # Once every entity has drifted past the tolerance the verdicts are settled; the bounded tail then shows
            # the slip without letting anything accelerate over the rest of the horizon.
            if min(box_drift, spinner_swept_angle, roller_swept_angle) > DRIFT_TOLERANCE:
                n_steps_slipped += 1
                if n_steps_slipped > N_STEPS_SLIPPED:
                    break
        is_load_held.append(
            (
                box_drift < DRIFT_TOLERANCE,
                spinner_swept_angle < DRIFT_TOLERANCE,
                roller_swept_angle < DRIFT_TOLERANCE,
            )
        )
    return is_load_held


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    for friction_cone in (gs.friction_cone.pyramidal, gs.friction_cone.elliptic):
        is_load_held = measure_stiction(friction_cone, args.vis)
        print(f"{friction_cone.name} cone (load as a fraction of the Coulomb limit):")
        for i_e, name in enumerate(("tangential (box)", "torsional (spinning sphere)", "rolling (rolling sphere)")):
            verdicts = ", ".join(
                f"{load_ratio:.2f}: {'holds' if is_load_held[i_load][i_e] else 'slips'}"
                for i_load, load_ratio in enumerate(LOAD_RATIOS)
            )
            print(f"  {name}: {verdicts}")


if __name__ == "__main__":
    main()
