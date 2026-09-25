"""
Entity Naming Tutorial
======================

Demonstrates the Genesis entity naming system:
- Auto-generated names based on morph type + UID
- User-specified custom names
- Scene-level entity lookup APIs
"""

import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser(description="Entity Naming Tutorial")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=args.vis)

    # Auto-generated names: {morph_type}_{uid_prefix}
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0, 0, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, 0, 0.5),
            radius=0.1,
        ),
    )
    print(f"Auto-generated: box='{box.name}', sphere='{sphere.name}'")

    # User-specified names
    ground = scene.add_entity(
        morph=gs.morphs.Plane(),
        name="ground",
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0.5, 0.5),
        ),
    )
    print(f"User name: '{ground.name}', URDF auto-name: '{robot.name}'")

    # Lookup by name
    retrieved = scene.get_entity(name="ground")
    print(f"Lookup by name: scene.get_entity(name='ground') -> {retrieved.name}")

    # Lookup by short UID (7-character prefix shown in terminal)
    short_uid = box.uid.short()
    retrieved = scene.get_entity(uid=short_uid)
    print(f"Lookup by UID: scene.get_entity(uid='{short_uid}') -> {retrieved.name}")

    # List all entity names
    print(f"All entity names: {scene.entity_names}")


if __name__ == "__main__":
    main()
