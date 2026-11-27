import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import rgb_array_to_png_bytes
from .conftest import RENDERER_TYPE


def _batch_render_scene(
    show_viewer,
    renderer,
    png_snapshot,
    use_batch_texture=False,
    use_fisheye_camera=False,
    use_directional_light=False,
    n_envs=2,
):
    CAM_RES = (128, 128)

    scene = gs.Scene(
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )

    # entities
    surface = (
        gs.surfaces.Default(diffuse_texture=gs.textures.BatchTexture.from_images(image_folder="textures"))
        if use_batch_texture
        else None
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=surface,
    )
    scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )

    # cameras
    cam = scene.add_camera(
        res=CAM_RES,
        pos=(1.5, -0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        model="fisheye" if use_fisheye_camera else "pinhole",
        GUI=show_viewer,
    )

    # lights
    if use_directional_light:
        scene.add_light(
            pos=(0.0, 0.0, 1.5),
            dir=(1.0, 1.0, -2.0),
            color=(1.0, 1.0, 0.0),
            directional=True,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
    scene.add_light(
        pos=(4.0, -4.0, 4.0),
        dir=(-1.0, 1.0, -1.0),
        directional=False,
        castshadow=True,
        cutoff=45.0,
        intensity=0.5,
    )
    scene.build(n_envs=n_envs)

    rgb_arrs, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
    assert rgb_arrs is not None

    for i in range(scene.n_envs):
        rgb_arr = rgb_arrs[i]
        assert rgb_arr.shape == (*CAM_RES, 3)
        assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot


@pytest.mark.slow  # ~300s
@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER])
def test_lights(show_viewer, renderer, png_snapshot):
    _batch_render_scene(show_viewer, renderer, png_snapshot, use_directional_light=True)


@pytest.mark.slow  # ~300s
@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER])
def test_texture(show_viewer, renderer, png_snapshot):
    _batch_render_scene(show_viewer, renderer, png_snapshot, use_batch_texture=True, n_envs=3)


@pytest.mark.slow  # ~300s
@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER])
def test_fisheye_camera(show_viewer, renderer, png_snapshot):
    _batch_render_scene(show_viewer, renderer, png_snapshot, use_fisheye_camera=True)


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER])
def test_dynamic_texture(show_viewer, renderer):
    CAM_RES = (64, 64)
    TEX_RES = (16, 16)

    scene = gs.Scene(
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    # A board filling the view of a camera looking straight down at it, without anything behind
    board = scene.add_entity(
        spec=gs.entities.Rigid(
            morph=gs.morphs.Plane(
                plane_size=(1.0, 1.0),
                tile_size=(1.0, 1.0),
            ),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.DynamicTexture(
                    resolution=TEX_RES,
                    variants={
                        "red": gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)),
                        "green": gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)),
                    },
                    is_batched=True,
                ),
            ),
        ),
    )
    cam = scene.add_camera(
        res=CAM_RES,
        pos=(0.0, 0.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=30,
        GUI=show_viewer,
    )
    scene.add_light(
        pos=(0.0, 0.0, 2.0),
        dir=(0.0, 0.0, -1.0),
        directional=True,
        castshadow=False,
        cutoff=45.0,
        intensity=1.0,
    )
    scene.build(n_envs=3)

    def get_colors():
        rgb_arrs, *_ = cam.render(rgb=True)
        rgb_arrs = tensor_to_array(rgb_arrs).astype(np.int32)
        h, w = CAM_RES
        # The dominant channel at the center of the left and right halves of each environment, -1 for black
        colors = []
        for rgb_arr in rgb_arrs:
            env_colors = []
            for pixel in (rgb_arr[h // 2, w // 4], rgb_arr[h // 2, 3 * w // 4]):
                env_colors.append(int(np.argmax(pixel)) if pixel.max() > 40 else -1)
            colors.append(tuple(env_colors))
        return colors

    RED, GREEN, BLUE, BLACK = 0, 1, 2, -1

    # Every environment shows the first variant at build
    assert get_colors() == [(RED, RED)] * 3

    # Each environment shows its own variant, set between two renders of the same step
    board.set_texture(variant=["red", "green", "red"])
    assert get_colors() == [(RED, RED), (GREEN, GREEN), (RED, RED)]

    # Writing the batched image of an environment shows it there alone, its texels of zero alpha showing what lies
    # behind them, which is nothing
    image = np.zeros((*TEX_RES[::-1], 4), dtype=np.uint8)
    image[..., 2] = 255
    image[:, TEX_RES[0] // 2 :, 3] = 255
    board.set_texture(image, envs_idx=[2])
    scene.step()
    colors = get_colors()
    assert colors[:2] == [(RED, RED), (GREEN, GREEN)]
    assert sorted(colors[2]) == [BLACK, BLUE]

    # Switching back to a variant and to the batched image again keeps the image written
    board.set_texture(variant="green", envs_idx=[2])
    assert get_colors()[2] == (GREEN, GREEN)
    board.set_texture(image, envs_idx=[2])
    assert sorted(get_colors()[2]) == [BLACK, BLUE]

    # A reset shows the first variant again
    scene.reset()
    assert get_colors() == [(RED, RED)] * 3
