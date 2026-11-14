"""A duck switching between the variants of a dynamic texture, and a canvas painted on with the mouse, at runtime.

In parallel environments, the duck of environment i starts on variant i. Without the viewer, the textures cycle on their
own. With it, the keyboard sets them in every environment, and the mouse paints in the environment it points at, into
the batched variant of the texture, which holds one image per environment:
Left mouse  - Paint on the canvas or the duck
Right       - Next texture
Left        - Previous texture
Space       - Draw a random color per environment
C           - Clear the paint
Esc         - Quit
"""

import argparse
import math
import os
from typing import TYPE_CHECKING, NamedTuple, override

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.state_change import Subscriber
from genesis.utils.misc import tensor_to_array, with_lock
from genesis.vis.keybindings import Key, KeyAction, Keybind, MouseButton
from genesis.vis.viewer_plugins import EVENT_HANDLE_STATE, EVENT_HANDLED, RaycasterViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.texture import Texture
    from genesis.ext.pyrender.node import Node

TEXTURE_RESOLUTION = (256, 256)


class Canvas(NamedTuple):
    """A dynamic texture taking paint, with the image of its batched variant in each environment as last seen."""

    texture: "Texture"
    # The visual mesh of the geom drawing the texture, in the frame of the geom
    mesh: trimesh.Trimesh
    # The entries of each variant, as RGBA uint8 arrays of shape (n_entries, height, width, 4)
    variants: tuple[np.ndarray, ...]
    # The position of the batched variant taking the paint, and its image in each environment
    i_paint_variant: int
    images: np.ndarray
    # The center of each pixel, as (column, row) of shape (height, width, 2)
    pixels_pos: np.ndarray
    # The subscriber passing the painter the pixels drawn by others, watching no change of state
    subscriber: Subscriber


class TexturePainterPlugin(RaycasterViewerPlugin):
    """
    Paint with the left mouse button on the visual geoms drawing a dynamic texture in their diffuse map.

    The paint goes to the entities whose material sets use_visual_raycasting=True and whose dynamic texture holds one
    batched variant, in the environment under the cursor. It covers the image this environment shows, then shows the
    batched variant, so showing another variant hides it, and painting over that one starts afresh from it.

    Parameters
    ----------
    color : tuple of int, optional
        sRGB color of the paint, between 0 and 255. Default is (20, 20, 20).
    brush_radius : float, optional
        Radius of the brush, in pixels of the texture. Default is 3.0.
    """

    def __init__(self, color: tuple[int, int, int] = (20, 20, 20), brush_radius: float = 3.0) -> None:
        super().__init__(use_visual_geom=True)
        self.color = (*color, 255)
        self.brush_radius = brush_radius

        self._canvases: dict[int, Canvas] = {}
        # Stroke positions the viewer thread collects and the stepping thread paints, None ending a stroke
        self._strokes_pos: list[tuple[int, int] | None] = []
        self._is_painting = False
        # The canvas, environment and pixel painted last, None at the start of a stroke
        self._stroke_end: tuple[Canvas, int, np.ndarray] | None = None

    @property
    def _lock(self):
        # Painting draws on the textures, which takes the render lock to upload the pixels, so the plugin shares that
        # lock rather than holding one of its own while waiting for it
        return self.viewer.render_lock

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        for texture in scene.dynamic_textures:
            variants_is_batched = texture.options.variants_is_batched
            if variants_is_batched.count(True) != 1 or texture.texture_map_type != gs.TEXTURE_MAP_TYPE.DIFFUSE:
                continue
            vgeom = texture.vgeom
            variants = texture.options.get_variants_image(texture.texture_map_type)
            i_paint_variant = variants_is_batched.index(True)
            paint_entries = variants[i_paint_variant]
            images = paint_entries[np.arange(max(scene.n_envs, 1)) % len(paint_entries)]
            width, height = texture.options.resolution
            pixels_pos = np.stack(np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5), axis=-1)

            # Genesis keeps no pixels, so the painter follows those others draw, from any thread, to paint over them
            def copy_drawing(
                i_variant: int, envs_idx: np.ndarray, image: torch.Tensor, images: np.ndarray = images
            ) -> None:
                with self._lock:
                    images[envs_idx] = tensor_to_array(image)

            subscriber = Subscriber(to=frozenset())
            texture.subscribe(subscriber, copy_drawing)
            mesh = trimesh.Trimesh(vgeom.init_vverts, vgeom.init_vfaces, process=False)
            self._canvases[vgeom.uid] = Canvas(texture, mesh, variants, i_paint_variant, images, pixels_pos, subscriber)

    @with_lock
    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button != MouseButton.LEFT:
            return
        ray_hit = self._raycaster.cast(*self._screen_position_to_ray(x, y))
        if ray_hit is None or ray_hit.geom is None or ray_hit.geom.uid not in self._canvases:
            return
        self._is_painting = True
        self._strokes_pos.append((x, y))
        return EVENT_HANDLED

    @with_lock
    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self._is_painting:
            self._strokes_pos.append((x, y))
            return EVENT_HANDLED

    @with_lock
    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == MouseButton.LEFT and self._is_painting:
            self._is_painting = False
            self._strokes_pos.append(None)
            return EVENT_HANDLED

    @with_lock
    @override
    def update_on_sim_step(self) -> None:
        super().update_on_sim_step()

        strokes_pos, self._strokes_pos = self._strokes_pos, []
        canvases_envs_idx: dict[int, set[int]] = {}
        for stroke_pos in strokes_pos:
            ray_hit = None
            if stroke_pos is not None:
                ray_hit = self._raycaster.cast(*self._screen_position_to_ray(*stroke_pos))
            canvas = None
            if ray_hit is not None and ray_hit.geom is not None:
                canvas = self._canvases.get(ray_hit.geom.uid)
            if canvas is None:
                self._stroke_end = None
                continue

            # The texture coordinates of the point hit, interpolated over the face of the visual mesh holding it
            vgeom = canvas.texture.vgeom
            env_idx = ray_hit.env_idx
            vgeom_pos = tensor_to_array(vgeom.get_pos(env_idx, relative=False).squeeze(0))
            vgeom_quat = tensor_to_array(vgeom.get_quat(env_idx, relative=False).squeeze(0))
            hit_pos = ray_hit.position if env_idx is None else ray_hit.position - self.scene.envs_offset[env_idx]
            hit_pos_local = gu.inv_transform_by_trans_quat(hit_pos, vgeom_pos, vgeom_quat)
            hit_pos_mesh, _, faces_idx = canvas.mesh.nearest.on_surface(hit_pos_local[None])
            barycentric = trimesh.triangles.points_to_barycentric(canvas.mesh.triangles[faces_idx], hit_pos_mesh)
            uv = barycentric[0] @ vgeom.uvs[canvas.mesh.faces[faces_idx[0]]]
            height, width = canvas.images.shape[1:3]
            pixel_pos = uv * (width, height)

            # The first paint of a step over another variant starts from that variant, which the paint then covers
            i_env = 0 if env_idx is None else env_idx
            envs_idx = canvases_envs_idx.setdefault(vgeom.uid, set())
            if i_env not in envs_idx:
                envs_idx.add(i_env)
                i_variant = canvas.texture.active_variants_idx[i_env]
                if i_variant != canvas.i_paint_variant:
                    variant_entries = canvas.variants[i_variant]
                    canvas.images[i_env] = variant_entries[i_env % len(variant_entries)]

            # A stroke joins its successive points, unless it crosses to another canvas or environment, or across a
            # seam of the texture coordinates, which moves it to a distant part of the image
            stroke_start = pixel_pos
            if self._stroke_end is not None:
                end_canvas, end_env_idx, end_pixel_pos = self._stroke_end
                is_same_area = end_canvas is canvas and end_env_idx == i_env
                if is_same_area and np.linalg.norm(pixel_pos - end_pixel_pos) < 0.25 * max(width, height):
                    stroke_start = end_pixel_pos
            segment = pixel_pos - stroke_start
            ratio = np.clip((canvas.pixels_pos - stroke_start) @ segment / max(segment @ segment, gs.EPS), 0.0, 1.0)
            pixels_dist = np.linalg.norm(canvas.pixels_pos - stroke_start - ratio[..., None] * segment, axis=-1)
            canvas.images[i_env][pixels_dist <= self.brush_radius] = self.color
            self._stroke_end = (canvas, i_env, pixel_pos)

        for vgeom_uid, envs_idx in canvases_envs_idx.items():
            canvas = self._canvases[vgeom_uid]
            envs_idx = sorted(envs_idx)
            canvas.texture.set_texture(canvas.images[envs_idx], envs_idx if self.scene.n_envs > 0 else None)

    @override
    def on_close(self) -> None:
        super().on_close()

        for canvas in self._canvases.values():
            canvas.texture.unsubscribe(canvas.subscriber)
        self._canvases.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-b", "--num-envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, 1.5, 0.8),
            camera_lookat=(0.25, 0.25, 0.1),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    width, height = TEXTURE_RESOLUTION
    is_stripe = np.broadcast_to(np.arange(width) // 32 % 2 == 0, (height, width))
    variants = {
        "duck": gs.textures.AssetTexture(),
        "purple": gs.textures.ColorTexture(
            color=(0.71, 0.62, 0.86),
        ),
        "stripes": gs.textures.ImageTexture(
            image_array=np.where(is_stripe[..., None], [135, 206, 235], [189, 224, 254]).astype(np.uint8),
        ),
        "duckurcqa": gs.textures.ImageTexture(
            image_path="meshes/duck/duckurcqa.png",
        ),
        "paint": gs.textures.AssetTexture(
            is_batched=True,
        ),
    }
    duck = scene.add_entity(
        options=gs.entities.Rigid(
            morph=gs.morphs.Mesh(
                file="meshes/duck/duck.obj",
                scale=0.0005,
                pos=(0.0, 0.0, 0.04),
                euler=(90.0, 0.0, 90.0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.DynamicTexture(
                    resolution=TEXTURE_RESOLUTION,
                    variants=variants,
                ),
            ),
            material=gs.materials.Rigid(
                use_visual_raycasting=True,
            ),
        ),
    )
    # A white board standing behind the duck, its texture coordinates spanning the image once
    board = scene.add_entity(
        options=gs.entities.Rigid(
            morph=gs.morphs.Plane(
                pos=(0.0, -0.15, 0.13),
                normal=(0.0, 1.0, 0.0),
                plane_size=(0.3, 0.2),
                tile_size=(0.3, 0.2),
            ),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.DynamicTexture(
                    resolution=(300, 200),
                    variants={
                        "white": gs.textures.ColorTexture(color=(1.0, 1.0, 1.0)),
                        "paint": gs.textures.ColorTexture(color=(1.0, 1.0, 1.0), is_batched=True),
                    },
                ),
                double_sided=True,
            ),
            material=gs.materials.Rigid(
                use_visual_raycasting=True,
            ),
        ),
    )
    if args.vis:
        scene.viewer.add_plugin(TexturePainterPlugin())
    scene.build(n_envs=args.num_envs, env_spacing=(0.5, 0.5))

    # The variants cycled through, the paint being shown by painting alone
    variants_name = tuple(name for name, texture in variants.items() if not texture.is_batched)

    # The variant each environment shows is offset by the index of the environment
    envs_idx = np.arange(max(args.num_envs, 1))
    i_variant = 0
    is_running = True

    def show_variant(offset=0):
        nonlocal i_variant
        i_variant = (i_variant + offset) % len(variants_name)
        duck.set_texture((i_variant + envs_idx) % len(variants_name))

    def draw():
        duck.set_texture(np.random.rand(len(envs_idx), 1, 1, 3))

    def clear():
        board.set_texture("white")
        show_variant()

    def stop():
        nonlocal is_running
        is_running = False

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("next_texture", Key.RIGHT, KeyAction.PRESS, callback=show_variant, args=(1,)),
            Keybind("previous_texture", Key.LEFT, KeyAction.PRESS, callback=show_variant, args=(-1,)),
            Keybind("draw_texture", Key.SPACE, KeyAction.PRESS, callback=draw),
            Keybind("clear_texture", Key.C, KeyAction.PRESS, callback=clear),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        horizon = math.inf
    else:
        horizon = 400 if "PYTEST_VERSION" not in os.environ else 5

    show_variant()
    step = 0
    while is_running and step < horizon:
        if not args.vis and step % 100 == 99:
            show_variant(1)
        scene.step()
        step += 1


if __name__ == "__main__":
    main()
