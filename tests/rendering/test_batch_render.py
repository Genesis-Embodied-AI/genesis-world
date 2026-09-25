import pytest

import genesis as gs

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
