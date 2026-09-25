"""In-hand pivoting under the elliptic cone with torsional friction, the maximum-realism friction configuration.

A checkered ball is pinched between two fixed plates and spun about the pinch axis, the canonical grasp scenario a
point contact cannot resist with sliding friction alone: without torsional friction the grasped ball pivots forever.
The elliptic cone with its auto-resolved high 'impratio' holds the ball against gravity without the tangential creep
of the pyramidal cone, and its exact Coulomb bound brakes the spin at the rate set by 'friction_torsional' times the
grip normal force. The checker pattern shows the rotation in the viewer.
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
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.9, -0.9, 0.6),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=args.vis,
    )

    # Two fixed plates pinch the ball with a slight interpenetration that supplies the grip normal force.
    for i_side in range(2):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.3, 0.02, 0.3),
                pos=(0.0, (2 * i_side - 1) * 0.1095, 0.3),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                friction_torsional=0.002,
            ),
        )
    ball = scene.add_entity(
        gs.morphs.MJCF(
            file=checkered_ball_mjcf(),
            pos=(0.0, 0.0, 0.3),
        ),
        material=gs.materials.Rigid(
            friction_torsional=0.002,
        ),
    )

    scene.build()

    # Let the pinch settle, then spin the ball about the pinch axis.
    for _ in range(50):
        scene.step()
    z_settled = float(ball.get_dofs_position()[..., 2])
    ball.set_dofs_velocity([0.0, 0.0, 0.0, 0.0, 8.0, 0.0])
    for i_step in range(300):
        scene.step()
        if (i_step < 100 and i_step % 20 == 19) or i_step % 100 == 99:
            spin = float(ball.get_dofs_velocity()[..., 4])
            sag = z_settled - float(ball.get_dofs_position()[..., 2])
            print(f"step {i_step + 1}: spin {spin:.2f} rad/s, sag since settling {1e3 * sag:.2f} mm")


if __name__ == "__main__":
    main()
