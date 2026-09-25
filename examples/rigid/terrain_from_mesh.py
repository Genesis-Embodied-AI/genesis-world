import argparse
import os

import genesis as gs
import numpy as np
from genesis.utils.terrain import mesh_to_heightfield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -50, 0),
            camera_lookat=(0, 0, 0),
        ),
        show_viewer=args.vis,
    )

    horizontal_scale = 2.0
    gs_root = os.path.dirname(os.path.abspath(gs.__file__))
    path_terrain = os.path.join(gs_root, "assets", "meshes", "terrain_45.obj")
    hf_terrain, xs, ys = mesh_to_heightfield(path_terrain, spacing=horizontal_scale, oversample=1)
    print("hf_terrain", path_terrain, hf_terrain.shape, np.max(hf_terrain))

    # default heightfield starts at 0, 0, 0
    # translate to the center of the mesh
    translation = np.array([np.nanmin(xs), np.nanmin(ys), 0])

    terrain_heightfield = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=1.0,
            height_field=hf_terrain,
            pos=translation,
        ),
        vis_mode="collision",
    )

    ball = scene.add_entity(
        gs.morphs.Sphere(
            pos=(10, 15, 10),
            radius=1,
        ),
        vis_mode="collision",
    )

    scene.build()

    for i in range(2000):
        scene.step()


if __name__ == "__main__":
    main()
