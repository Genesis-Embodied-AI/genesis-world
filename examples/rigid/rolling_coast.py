"""A rolling ball coasting to rest under rolling friction, the everyday behavior a point contact cannot produce.

Two checkered balls are launched rolling side by side at the same speed. Sliding friction does no work on a rolling
contact, so the zero-coefficient ball coasts forever; the other carries a rolling friction coefficient and slows to
a stop within a couple of meters, like a real ball on a floor. The elliptic cone brakes it at the exact Coulomb
bound, a constant deceleration of (5/7) * friction_rolling * g / r. The checker pattern shows the rotation in the viewer.
"""

import argparse
import xml.etree.ElementTree as ET

import genesis as gs
from genesis.utils.misc import get_assets_dir


RADIUS = 0.1
# Half-extent of the wooden sphere asset, whose origin sits at the bottom of its y-up frame.
WOODEN_SPHERE_MESH_RADIUS = 3.486346


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -4.5, 2.0),
            camera_lookat=(2.0, 0.3, 0.1),
        ),
        show_viewer=args.vis,
    )

    # The contact coefficient is the maximum over the pair, so the zero-coefficient plane leaves each ball coasting
    # at its own rate.
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction_torsional=0.0,
            friction_rolling=0.0,
        ),
    )
    balls = [
        scene.add_entity(
            gs.morphs.MJCF(
                file=checkered_ball_mjcf(),
                pos=(0.0, 0.6 * i_ball, RADIUS),
            ),
            material=gs.materials.Rigid(
                friction_rolling=friction_rolling,
            ),
        )
        for i_ball, friction_rolling in enumerate((0.0, 0.005))
    ]

    scene.build()

    # Launch both balls rolling without slipping: v = w * r.
    for ball in balls:
        ball.set_dofs_velocity([1.0, 0.0, 0.0, 0.0, 10.0, 0.0])
    for i_step in range(500):
        scene.step()
        if i_step % 100 == 99:
            speeds = [float(ball.get_dofs_velocity()[..., 0]) for ball in balls]
            distances = [float(ball.get_pos()[..., 0]) for ball in balls]
            print(
                f"step {i_step + 1}: free ball {speeds[0]:.2f} m/s at {distances[0]:.2f} m, "
                f"braked ball {speeds[1]:.2f} m/s at {distances[1]:.2f} m"
            )


if __name__ == "__main__":
    main()
