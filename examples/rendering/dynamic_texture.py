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

import trimesh
from huggingface_hub import snapshot_download

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array, with_lock
from genesis.vis.keybindings import Key, KeyAction, Keybind, MouseButton
from genesis.vis.viewer_plugins import EVENT_HANDLE_STATE, EVENT_HANDLED, RaycasterViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidVisGeom
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node

TEXTURE_RESOLUTION = (256, 256)


class Canvas(NamedTuple):
    """A visual geom whose dynamic texture takes paint in its batched variant.

    Attributes
    ----------
    vgeom : RigidVisGeom
        The visual geom drawing the texture in its diffuse map.
    mesh : trimesh.Trimesh
        The visual mesh of the geom, in the frame of the geom.
    pixels_pos : np.ndarray
        The center of each pixel, as (column, row) of shape (height, width, 2).
    """

    vgeom: "RigidVisGeom"
    mesh: trimesh.Trimesh
    pixels_pos: np.ndarray


class TexturePainterPlugin(RaycasterViewerPlugin):
    """
    Paint with the left mouse button on the visual geoms drawing a dynamic texture with a batched variant in their
    diffuse map.

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
        self._strokes_pos: list[tuple[int, int] | None] = []
        self._is_painting = False
        self._stroke_end: tuple[Canvas, int, np.ndarray] | None = None

    @property
    def _lock(self):
        # Painting writes pixels the renderers read, so the plugin shares the render lock rather than holding its own
        return self.viewer.render_lock

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        for vgeom in scene.rigid_solver.vgeoms:
            texture = vgeom.surface.texture
            if isinstance(texture, gs.textures.DynamicTexture) and texture.is_batched:
                width, height = texture.resolution
                pixels_pos = np.stack(np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5), axis=-1)
                mesh = trimesh.Trimesh(vgeom.init_vverts, vgeom.init_vfaces, process=False)
                self._canvases[vgeom.uid] = Canvas(vgeom, mesh, pixels_pos)

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
        # The image painted on in each environment of each canvas
        canvases_images: dict[int, dict[int, np.ndarray]] = {}
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
            vgeom = canvas.vgeom
            env_idx = ray_hit.env_idx
            vgeom_pos = tensor_to_array(vgeom.get_pos(env_idx, relative=False).squeeze(0))
            vgeom_quat = tensor_to_array(vgeom.get_quat(env_idx, relative=False).squeeze(0))
            hit_pos = ray_hit.position if env_idx is None else ray_hit.position - self.scene.envs_offset[env_idx]
            hit_pos_local = gu.inv_transform_by_trans_quat(hit_pos, vgeom_pos, vgeom_quat)
            hit_pos_mesh, _, faces_idx = canvas.mesh.nearest.on_surface(hit_pos_local[None])
            barycentric = trimesh.triangles.points_to_barycentric(canvas.mesh.triangles[faces_idx], hit_pos_mesh)
            uv = barycentric[0] @ vgeom.uvs[canvas.mesh.faces[faces_idx[0]]]
            height, width = canvas.pixels_pos.shape[:2]
            pixel_pos = uv * (width, height)

            # The first paint of a step in an environment starts from the image it shows, which the paint then covers
            i_env = 0 if env_idx is None else env_idx
            images = canvases_images.setdefault(vgeom.uid, {})
            if i_env not in images:
                image = vgeom.get_texture(None if env_idx is None else [env_idx])
                images[i_env] = tensor_to_array(image if env_idx is None else image[0])

            # A stroke joins its successive points, unless it crosses to another canvas or environment, or across a
            # seam of the texture coordinates, which moves it to a distant part of the image.
            stroke_start = pixel_pos
            if self._stroke_end is not None:
                end_canvas, end_env_idx, end_pixel_pos = self._stroke_end
                is_same_area = end_canvas is canvas and end_env_idx == i_env
                if is_same_area and np.linalg.norm(pixel_pos - end_pixel_pos) < 0.25 * max(width, height):
                    stroke_start = end_pixel_pos
            segment = pixel_pos - stroke_start
            ratio = np.clip((canvas.pixels_pos - stroke_start) @ segment / max(segment @ segment, gs.EPS), 0.0, 1.0)
            pixels_dist = np.linalg.norm(canvas.pixels_pos - stroke_start - ratio[..., None] * segment, axis=-1)
            images[i_env][pixels_dist <= self.brush_radius] = self.color
            self._stroke_end = (canvas, i_env, pixel_pos)

        for vgeom_uid, images in canvases_images.items():
            envs_idx = list(images)
            self._canvases[vgeom_uid].vgeom.set_texture(
                np.stack(list(images.values())), envs_idx if self.scene.n_envs > 0 else None
            )

    @override
    def on_close(self) -> None:
        super().on_close()
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
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="d571c25abb363a5facc603b7f6be62e6e0ad46da",
        allow_patterns="textures/duckurcqa.png",
        max_workers=1,
    )
    width, height = TEXTURE_RESOLUTION
    is_stripe = np.broadcast_to(np.arange(width) // 32 % 2 == 0, (height, width))
    variants = {
        "duck": None,
        "purple": gs.textures.ColorTexture(
            color=(0.71, 0.62, 0.86),
        ),
        "stripes": gs.textures.ImageTexture(
            image_array=np.where(is_stripe[..., None], [135, 206, 235], [189, 224, 254]).astype(np.uint8),
        ),
        "duckurcqa": gs.textures.ImageTexture(
            image_path=f"{asset_path}/textures/duckurcqa.png",
        ),
    }
    duck = scene.add_entity(
        spec=gs.entities.Rigid(
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
                    is_batched=True,
                ),
            ),
            material=gs.materials.Rigid(
                use_visual_raycasting=True,
            ),
        ),
    )
    # A white board standing behind the duck, its texture coordinates spanning the image once
    board = scene.add_entity(
        spec=gs.entities.Rigid(
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
                        "white": gs.textures.ColorTexture(
                            color=(1.0, 1.0, 1.0),
                        ),
                    },
                    is_batched=True,
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

    # The variant each environment shows is offset by the index of the environment
    envs_idx = np.arange(max(args.num_envs, 1))
    i_variant = 0
    is_running = True

    def show_variant(offset=0):
        nonlocal i_variant
        i_variant = (i_variant + offset) % len(variants)
        duck.set_texture(variant=(i_variant + envs_idx) % len(variants))

    def draw():
        duck.set_texture(np.random.rand(len(envs_idx), 1, 1, 3))

    def clear():
        board.set_texture(variant="white")
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
