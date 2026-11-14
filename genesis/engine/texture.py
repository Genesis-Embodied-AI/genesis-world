"""Runtime state of the dynamic textures of a scene (see ``gs.textures.DynamicTexture``)."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
from genesis.constants import TEXTURE_MAP_TYPE
from genesis.engine.state_change import StateChange, Subscriber
from genesis.options.textures import ColorTexture, DynamicTexture, ImageTexture, texture_to_rgba
from genesis.typing import IndexType
from genesis.utils.misc import broadcast_tensor, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidVisGeom
    from genesis.engine.scene import Scene


# The variants to show, by position or by name, or the pixels to draw (see 'Texture.set_texture')
TextureValueType = int | str | Sequence[int | str] | np.ndarray | torch.Tensor | ColorTexture | ImageTexture


class Texture:
    """Runtime state of the dynamic texture of one map of a visual geom: the variant each environment shows.

    Genesis holds only the active variant indices, the position of the variant each environment shows. The renderers
    hold the pixels, each under the same contract:

    - at build, it allocates the entries of each variant ('options.get_variants_image(texture_map_type)'), a batched
      variant taking one image per environment it renders, environment i holding entry i modulo the number of entries.
      It binds them to the map of type 'texture_map_type' of 'vgeom', and subscribes a ``Subscriber`` of
      ``StateChange.VISUAL`` with a drawing callback;
    - on every drawing, the drawing callback receives the batched variant drawn in, the environments drawn and their
      pixels, as an RGBA tensor on the device of shape (len(envs_idx), height, width, 4);
    - at sync, when its subscriber holds ``StateChange.VISUAL`` pending, it clears it, then reads 'active_variants_idx'.

    The images are uint8 sRGB values in the diffuse map, and float16 linear values in the emissive map.
    """

    def __init__(
        self, scene: "Scene", options: DynamicTexture, texture_map_type: TEXTURE_MAP_TYPE, vgeom: "RigidVisGeom"
    ):
        self._scene = scene
        self._options = options
        self._texture_map_type = texture_map_type
        self._vgeom = vgeom
        (self._initial_variant_idx,) = options.get_variants_idx(options.initial_variant)
        self._active_variants_idx = np.full(scene._B, self._initial_variant_idx, dtype=gs.np_int)
        # Whether pixels were drawn in each variant of each environment since its last reset
        self._is_drawn = np.zeros((len(options.variants), scene._B), dtype=np.bool_)
        # The entries of each batched variant on the device, which a reset draws again, built on the first reset
        self._variants_seed: dict[int, torch.Tensor] = {}
        # The subscribers, each with the callback receiving the pixels of every drawing, if any
        self._subscribers: dict[Subscriber, Callable[[int, np.ndarray, torch.Tensor], None] | None] = {}

    @property
    def options(self) -> DynamicTexture:
        """The declaration of the texture, as the surface of its visual geom holds it."""
        return self._options

    @property
    def texture_map_type(self) -> TEXTURE_MAP_TYPE:
        """The map of the surface that the texture fills, which sets the format of its images."""
        return self._texture_map_type

    @property
    def vgeom(self) -> "RigidVisGeom":
        """The visual geom drawing the texture."""
        return self._vgeom

    @property
    def active_variants_idx(self) -> np.ndarray:
        """The position of the variant each environment shows, as a read-only array."""
        active_variants_idx = self._active_variants_idx.view()
        active_variants_idx.flags.writeable = False
        return active_variants_idx

    def subscribe(
        self,
        subscriber: Subscriber,
        drawing_callback: Callable[[int, np.ndarray, torch.Tensor], None] | None = None,
    ) -> None:
        """Notify a subscriber of every change of the active variants, and pass it the pixels of every drawing.

        'subscriber' is notified as ``StateChange.VISUAL`` (see 'Texture'). 'drawing_callback', if given, is called with
        the variant drawn in, the environments drawn and their pixels. Genesis keeps no pixels, so a renderer giving one
        subscribes before any pixels are drawn.
        """
        if drawing_callback is not None and self._is_drawn.any():
            gs.raise_exception("A renderer cannot be added once pixels have been drawn on a dynamic texture.")
        self._subscribers[subscriber] = drawing_callback

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Stop notifying a subscriber given to 'subscribe', and calling its drawing callback."""
        self._subscribers.pop(subscriber, None)

    def get_texture(self, envs_idx: IndexType | None = None) -> torch.Tensor:
        """Return the position of the variant each selected environment shows."""
        active_variants_idx = self._active_variants_idx[tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))]
        active_variants_idx = torch.as_tensor(active_variants_idx, dtype=gs.tc_int, device=gs.device)
        return active_variants_idx[0] if self._scene.n_envs == 0 else active_variants_idx

    def set_texture(self, texture: TextureValueType, envs_idx: IndexType | None = None) -> None:
        """Show a variant, or draw pixels, in the selected environments.

        'texture' is either variants or pixels. A variant is given by position or by name, one for all the selected
        environments, or a sequence of them, one per selected environment. Pixels are a ``ColorTexture``, an
        ``ImageTexture``, or an RGB or RGBA array of shape (height, width, channels) for all the selected environments
        or (len(envs_idx), height, width, channels) for one each. Pixels go into the batched variant the environment
        shows, an environment showing another variant switching to the only batched variant of the texture.

        An array holds uint8 values, or floats where 1.0 stands for 255: sRGB values clamped to [0, 1] in the diffuse
        map, and linear values clamped only below at 0 in the emissive map, so that an emission may exceed 1.
        """
        envs_idx = tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))
        if isinstance(texture, (ColorTexture, ImageTexture)):
            texture = texture_to_rgba(texture, self._options.resolution, self._texture_map_type)
        texture_ndim = texture.ndim if torch.is_tensor(texture) else np.ndim(texture)
        if texture_ndim <= 1:
            if torch.is_tensor(texture):
                texture = tensor_to_array(texture)
            variants_idx = self._options.get_variants_idx(texture)
            if len(variants_idx) not in (1, len(envs_idx)):
                gs.raise_exception(
                    f"A dynamic texture shows one variant, or one per selected environment, got {len(variants_idx)} "
                    f"variants for {len(envs_idx)} environments."
                )
            self._active_variants_idx[envs_idx] = variants_idx
        else:
            if texture_ndim not in (3, 4):
                gs.raise_exception(
                    "A dynamic texture is set to variants, or to pixels of shape (height, width, channels), got an "
                    f"array of {texture_ndim} dims."
                )

            # The batched variant each selected environment is drawn in
            variants_is_batched = np.array(self._options.variants_is_batched)
            batched_variants_idx = np.flatnonzero(variants_is_batched)
            if len(batched_variants_idx) == 0:
                gs.raise_exception("Pixels are drawn in a batched variant of a dynamic texture, which this one lacks.")
            active_variants_idx = self._active_variants_idx[envs_idx]
            is_active_batched = variants_is_batched[active_variants_idx]
            if len(batched_variants_idx) > 1 and not is_active_batched.all():
                gs.raise_exception(
                    "Pixels are drawn on a dynamic texture with several batched variants only in environments that "
                    "show one of them."
                )
            variants_idx = np.where(is_active_batched, active_variants_idx, batched_variants_idx[0])

            # torch.as_tensor warns on a read-only array, such as an image read from a file
            if isinstance(texture, np.ndarray) and not texture.flags.writeable:
                texture = texture.copy()
            image = torch.as_tensor(texture, device=gs.device)
            n_channels = image.shape[-1]
            if n_channels not in (3, 4):
                gs.raise_exception(f"Pixels are drawn as RGB or RGBA, got {n_channels} channels.")
            if self._texture_map_type == TEXTURE_MAP_TYPE.EMISSIVE:
                dtype, alpha = torch.float16, 1.0
                if image.is_floating_point():
                    image = image.clamp(0.0, torch.finfo(dtype).max)
                else:
                    image = image.clamp(0, 255) / 255.0
            else:
                dtype, alpha = torch.uint8, 255
                if image.is_floating_point():
                    image = image.mul(255.0).clamp_(0.0, 255.0).round_()
                else:
                    image = image.clamp(0, 255)
            width, height = self._options.resolution
            n_images = len(envs_idx)
            image = broadcast_tensor(image, dtype, (n_images, height, width, n_channels))
            if n_channels == 3:
                image = torch.cat((image, image.new_full((n_images, height, width, 1), alpha)), dim=-1)
            self._active_variants_idx[envs_idx] = variants_idx
            self._is_drawn[variants_idx, envs_idx] = True
            self._draw(variants_idx, envs_idx, image)
        for subscriber in self._subscribers:
            if StateChange.VISUAL in subscriber.to:
                subscriber.notify(StateChange.VISUAL, envs_idx)

    def reset(self, envs_idx: IndexType | None = None) -> None:
        """Show the initial variant again in the selected environments, and erase the pixels drawn in them."""
        self.set_texture(self._initial_variant_idx, envs_idx)
        envs_idx = tensor_to_array(self._scene._sanitize_envs_idx(envs_idx))
        for i_variant in np.flatnonzero(self._is_drawn[:, envs_idx].any(axis=1)):
            drawn_envs_idx = envs_idx[self._is_drawn[i_variant, envs_idx]]
            if i_variant not in self._variants_seed:
                variant_image = texture_to_rgba(
                    self._options.variants_texture[i_variant], self._options.resolution, self._texture_map_type
                )
                self._variants_seed[i_variant] = torch.as_tensor(variant_image, device=gs.device)
            variant_seed = self._variants_seed[i_variant]
            seed_idx = torch.as_tensor(drawn_envs_idx % len(variant_seed), device=gs.device)
            self._draw(np.full(len(drawn_envs_idx), i_variant), drawn_envs_idx, variant_seed[seed_idx])
            self._is_drawn[i_variant, drawn_envs_idx] = False

    def _draw(self, variants_idx: np.ndarray, envs_idx: np.ndarray, image: torch.Tensor) -> None:
        """Pass the pixels drawn in each environment to the drawing callbacks, grouped by the variant drawn in."""
        for i_variant in np.unique(variants_idx):
            is_drawn = variants_idx == i_variant
            variant_envs_idx, variant_image = envs_idx, image
            if not is_drawn.all():
                variant_envs_idx, variant_image = envs_idx[is_drawn], image[torch.as_tensor(is_drawn, device=gs.device)]
            for drawing_callback in self._subscribers.values():
                if drawing_callback is not None:
                    drawing_callback(int(i_variant), variant_envs_idx, variant_image)


def get_dynamic_texture(textures: list[Texture], texture_map_type: TEXTURE_MAP_TYPE | None) -> Texture:
    """Return the dynamic texture filling the map of type 'texture_map_type', or the only one of 'textures' if None.

    'textures' are the dynamic textures of one visual geom, one per map at most.
    """
    if texture_map_type is not None:
        textures = [texture for texture in textures if texture.texture_map_type == texture_map_type]
    if len(textures) == 1:
        return textures[0]
    if not textures:
        target = "" if texture_map_type is None else f" in the {texture_map_type.name.lower()} map"
        gs.raise_exception(f"This visual geom draws no dynamic texture{target}.")
    gs.raise_exception("This visual geom draws a dynamic texture in both maps. Pick one by 'texture_map_type'.")
