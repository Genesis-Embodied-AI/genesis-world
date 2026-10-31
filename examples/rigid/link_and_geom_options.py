"""Options for some links and geoms of a humanoid, selected by subtree and by geom name.

The whole right leg, the thigh and every link below it, is made denser through 'gs.LinkIndex.subtree', so it outweighs
the left one. The geoms of both feet, which the asset names 'foot1_*' and 'foot2_*', take a grippier material through
'gs.GeomIndex.named', one glob selecting them in every link at once.
"""

import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.8),
        ),
        show_viewer=args.vis,
    )

    humanoid = scene.add_entity(
        options=gs.options.RigidEntityOptions(
            morph=gs.morphs.MJCF(
                file="xml/humanoid.xml",
                recompute_inertia=True,
            ),
            links={
                gs.LinkIndex.subtree("thigh_right"): gs.options.LinkOptions(
                    material=gs.materials.Material(
                        density=3000.0,
                    ),
                ),
            },
            geoms={
                gs.GeomIndex.named("foot*"): gs.options.GeomOptions(
                    material=gs.materials.Material(
                        friction=2.0,
                    ),
                ),
            },
        ),
    )

    scene.build()

    for side in ("right", "left"):
        leg_mass = sum(humanoid.get_link(f"{part}_{side}").get_mass() for part in ("thigh", "shin", "foot"))
        feet_friction = ", ".join(f"{geom.get_friction():.1f}" for geom in humanoid.get_link(f"foot_{side}").geoms)
        print(f"{side} leg: {leg_mass:.2f} kg, foot geoms friction {feet_friction}")

    for _ in range(200):
        scene.step()


if __name__ == "__main__":
    main()
