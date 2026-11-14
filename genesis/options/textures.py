import os
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Annotated, Literal, cast

import numpy as np

from PIL import Image
from pydantic import BeforeValidator, Field, PrivateAttr, StrictBool, computed_field, model_validator
from typing_extensions import Self

import genesis as gs
import genesis.utils.mesh as mu
from genesis.constants import TEXTURE_MAP_TYPE
from genesis.typing import (
    FGridType,
    LaxFArrayType,
    LaxUnitIntervalArrayType,
    NDArrayType,
    PositiveVec2IType,
    StrArrayType,
    StrictInt,
    UnitIntervalArrayType,
)
from genesis.utils.serialization import Exporting, Loading, SerializationMixin

from .options import Options


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".hdr", ".exr")
HDR_EXTENSIONS = (".hdr", ".exr")


class Texture(Options):
    """
    Base class for Genesis's texture objects.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    # Pydantic disables hashing for models with a field-based __eq__; its only opt-in (model_config frozen=True) also
    # forbids the in-place edits done during setup (apply_cutoff, check_dim reassign image_array/color). Restore
    # identity hashing instead: textures are shared by reference and stable once a surface is built, so their identity
    # is a valid key for caches such as the trimesh materials cache (see 'surface_uvs_to_trimesh_visual').
    __hash__ = object.__hash__

    def __init__(self, **data):
        super().__init__(**data)

    def check_dim(self, dim: int) -> "Texture | None":
        raise NotImplementedError

    def check_simplify(self) -> "Texture":
        raise NotImplementedError

    def apply_cutoff(self, cutoff: float) -> None:
        raise NotImplementedError

    @cached_property
    def is_black(self) -> bool:
        raise NotImplementedError

    @cached_property
    def requires_uv(self) -> bool:
        raise NotImplementedError


class ColorTexture(Texture):
    """
    A texture that consists of a single color, or of one color per environment.

    Parameters
    ----------
    color : list of float, or list of list of float
        A list of color values, stored as tuple, supporting any number of channels within the range [0.0, 1.0]. A
        batched texture takes one such list per entry, or a single one for all of them. Default is (1.0, 1.0, 1.0).
    is_batched : bool, optional
        Whether each environment shows its own entry, environment i showing entry i modulo the number of entries. Each
        renderer stores every entry. As a variant of a ``DynamicTexture``, a batched texture instead stores one image
        per rendered environment, which ``set_texture`` draws pixels into. Default is False.
    """

    color: LaxFArrayType | FGridType = (1.0, 1.0, 1.0)
    is_batched: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _validate_batched(cls, data: dict) -> dict:
        if not isinstance(data, dict) or "color" not in data:
            return data
        colors = np.asarray(data["color"], dtype=np.float64)
        if colors.ndim > 1 and not data.get("is_batched", False):
            gs.raise_exception("A color texture takes one color per entry with 'is_batched=True' alone.")
        if data.get("is_batched", False):
            data["color"] = np.atleast_2d(colors)
        return data

    def check_dim(self, dim: int) -> Texture | None:
        if self.is_batched:
            if len(self.color[0]) > dim:
                self.color, res = tuple(color[:dim] for color in self.color), tuple(color[dim:] for color in self.color)
                return ColorTexture(color=res, is_batched=True)
        elif len(self.color) > dim:
            self.color, res = self.color[:dim], self.color[dim]
            return ColorTexture(color=res)
        return None

    def check_simplify(self) -> "ColorTexture":
        return self

    def apply_cutoff(self, cutoff: float) -> None:
        if cutoff is None:
            return
        if self.is_batched:
            self.color = tuple(tuple(1.0 if c >= cutoff else 0.0 for c in color) for color in self.color)
        else:
            self.color = tuple(1.0 if c >= cutoff else 0.0 for c in self.color)

    @computed_field
    @cached_property
    def is_black(self) -> bool:
        assert gs.EPS is not None
        return bool(np.all(np.asarray(self.color) < gs.EPS))

    @computed_field
    @cached_property
    def requires_uv(self) -> bool:
        return self.is_batched


class ImageTexture(Texture, SerializationMixin):
    """
    A texture with a texture map (image), or with one image per environment.

    Parameters
    ----------
    image_path : str or list of str, optional
        Path to the image file, or one path per entry of a batched texture.
    image_array : np.ndarray, optional
        Image array, of shape (height, width) or (height, width, channels). A batched texture also takes one array per
        entry, or an array of shape (n_entries, height, width, channels).
    image_color : float or list of float, optional
        The factor that will be multiplied with the base color, stored as tuple. Default is None.
    encoding : str, optional
        The encoding way of the image. Possible values are ['srgb', 'linear']. Default is 'srgb'.

        - 'srgb': Encoding of some RGB images.
        - 'linear': All generic images, such as opacity, roughness and normal, should be encoded with 'linear'.
    is_batched : bool, optional
        Whether each environment shows its own entry, environment i showing entry i modulo the number of entries. The
        entries are resized to the resolution of the largest one, so that each costs the memory of the largest image.
        Each renderer stores every entry. As a variant of a ``DynamicTexture``, a batched texture instead stores one
        image per rendered environment, which ``set_texture`` draws pixels into. Default is False.
    """

    image_path: str | StrArrayType | None = None
    image_array: NDArrayType | None
    image_color: UnitIntervalArrayType
    encoding: Annotated[Literal["srgb", "linear"], BeforeValidator(lambda e: str(e).lower())] = "srgb"
    is_batched: StrictBool = False

    def __init__(
        self,
        *,
        image_path: str | Sequence[str] | None = None,
        image_array: np.ndarray | Sequence[np.ndarray] | None = None,
        image_color: LaxUnitIntervalArrayType | float | None = None,
        encoding: str = "srgb",
        is_batched: bool = False,
        **data,
    ) -> None:
        super().__init__(
            image_path=image_path,
            image_array=image_array,
            image_color=image_color,
            encoding=encoding,
            is_batched=is_batched,
            **data,
        )

    @model_validator(mode="before")
    @classmethod
    def _validate_and_load(cls, data: dict) -> dict:
        # A texture instance validated against this class, as a union of texture types does, is not raw input
        if not isinstance(data, dict):
            return data
        image_path, image_array = data.get("image_path"), data.get("image_array")
        is_batched = data.get("is_batched", False)

        if not ((image_path is not None) ^ (image_array is not None)):
            gs.raise_exception("Please set either `image_path` or `image_array`.")

        images_array = []
        if image_path is not None:
            images_path = [image_path] if isinstance(image_path, str) else list(image_path)
            if not images_path or (len(images_path) > 1 and not is_batched):
                gs.raise_exception("An image texture takes one image path, or one per entry with 'is_batched=True'.")
            for i, path in enumerate(images_path):
                # Look for absolute image path
                if not os.path.exists(path):
                    candidate_image_path = os.path.join(gs.utils.get_assets_dir(), path)
                    if not os.path.exists(candidate_image_path):
                        gs.raise_exception(f"File not found in either current directory or assets directory: '{path}'.")
                    images_path[i] = path = candidate_image_path

                # Load image_path as actual image_array, unless for special texture images (e.g. `.hdr` and `.exr`) that
                # are only supported by Raytracer.
                if path.endswith(HDR_EXTENSIONS):
                    if is_batched:
                        gs.raise_exception(f"A batched image texture takes no HDR image, got '{path}'.")
                    data.setdefault("encoding", "linear")
                    if data["encoding"] != "linear":
                        gs.raise_exception("HDR images requires linear encoding.")
                    if path.endswith((".exr")):
                        images_path[i] = mu.check_exr_compression(path)
                else:
                    images_array.append(mu.PIL_to_array(Image.open(path)))
            image_path = tuple(images_path) if is_batched else images_path[0]
        else:
            if isinstance(image_array, (list, tuple)) or (
                isinstance(image_array, np.ndarray) and image_array.ndim == 4
            ):
                if not is_batched:
                    gs.raise_exception("An image texture takes one image per entry with 'is_batched=True' alone.")
                images_array = list(image_array)
                if not images_array:
                    gs.raise_exception("A batched image texture takes at least one image.")
            else:
                images_array = [image_array]
            # Normalize image arrays
            for i, image in enumerate(images_array):
                if not isinstance(image, np.ndarray):
                    gs.raise_exception("`image_array` needs to be a numpy array.")
                if image.ndim not in (2, 3):
                    gs.raise_exception(f"An image is an array of 2 or 3 dimensions, got {image.ndim}.")
                if image.dtype != np.uint8:
                    if np.issubdtype(image.dtype, np.floating):
                        if image.max() <= 1.0:
                            image = (image * 255.0).round()
                        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
                    elif np.issubdtype(image.dtype, np.integer):
                        image = np.clip(image, 0, 255).astype(np.uint8)
                    elif image.dtype == np.bool_:
                        image = image.astype(np.uint8) * 255
                    else:
                        gs.raise_exception(
                            f"Unsupported image dtype {image.dtype}. Only uint8, integer, floating-point, or bool "
                            "types are supported."
                        )
                images_array[i] = image

        # The entries of a batched texture stack into one array of shape (n_entries, height, width, channels), each
        # converted to the resolution of the one with the most pixels and to the largest number of channels
        if not images_array:
            image_array = None
        elif is_batched:
            height, width = max((image.shape[:2] for image in images_array), key=lambda shape: shape[0] * shape[1])
            n_channels = max(image.shape[-1] if image.ndim == 3 else 1 for image in images_array)
            if n_channels > 4:
                gs.raise_exception(f"A batched image texture takes images of 1 to 4 channels, got {n_channels}.")
            mode = ("L", "LA", "RGB", "RGBA")[n_channels - 1]
            for i, image in enumerate(images_array):
                if image.ndim == 3 and image.shape[-1] == 1:
                    image = image[..., 0]
                if image.shape[:2] != (height, width) or (image.shape[-1] if image.ndim == 3 else 1) != n_channels:
                    image = Image.fromarray(image).convert(mode).resize((width, height), Image.Resampling.BILINEAR)
                    image = np.asarray(image)
                images_array[i] = image if image.ndim == 3 else image[..., None]
            image_array = np.stack(images_array)
        else:
            (image_array,) = images_array
        data["image_path"], data["image_array"] = image_path, image_array

        # Resolve image color
        image_color = data.get("image_color")
        if image_array is None:
            channel = 3
        elif image_array.ndim >= 3:
            channel = image_array.shape[-1]
        else:
            channel = 1
        if image_color is None:
            image_color = 1.0
        if isinstance(image_color, Iterable):
            image_color = tuple(image_color)[:channel]
        else:
            image_color = (image_color,) * channel
        data["image_color"] = image_color

        return data

    @computed_field
    @cached_property
    def is_black(self) -> bool:
        assert gs.EPS is not None
        assert self.image_color is not None
        assert self.image_array is not None
        # Black when every channel's effective value (texel x factor) is zero. Testing the factor and the array
        # independently would miss a per-channel factor that masks the only nonzero channels (e.g. a green texture
        # scaled by a red-only factor).
        channel_peaks = self.image_array.reshape(-1, self.channel).max(axis=0)
        return all(factor < gs.EPS or peak == 0 for factor, peak in zip(self.image_color, channel_peaks))

    @computed_field
    @cached_property
    def requires_uv(self) -> bool:
        return True

    @computed_field
    @property
    def channel(self) -> int:
        if self.image_array is None:
            return 3
        return self.image_array.shape[-1] if self.image_array.ndim >= 3 else 1

    @computed_field
    @cached_property
    def mean_color(self) -> NDArrayType:
        if self.image_array is None:
            return np.ones(3, dtype=np.float16)
        axes = (0, 1, 2) if self.is_batched else (0, 1)
        return cast(np.ndarray, (np.mean(self.image_array, axis=axes, dtype=np.float32) / 255).astype(np.float16))

    def check_dim(self, dim: int) -> Texture | None:
        if self.image_array is not None:
            if self.channel > dim:
                if self.is_batched:
                    self.image_array, res_array = self.image_array[..., :dim], self.image_array[..., dim : dim + 1]
                else:
                    self.image_array, res_array = self.image_array[:, :, :dim], self.image_array[:, :, dim]
                self.image_color, res_color = self.image_color[:dim], self.image_color[dim:]
                return ImageTexture(
                    image_array=res_array, image_color=res_color, encoding="linear", is_batched=self.is_batched
                ).check_simplify()
        return None

    def check_simplify(self) -> "ImageTexture | ColorTexture":
        if self.image_array is None:
            return self
        axes = (1, 2) if self.is_batched else (0, 1)
        max_color = np.max(self.image_array, axis=axes)
        min_color = np.min(self.image_array, axis=axes)
        if np.all(min_color == max_color):
            if self.is_batched:
                return ColorTexture(color=max_color / 255.0 * self.image_color, is_batched=True)
            return ColorTexture(color=max_color.reshape(-1) / 255.0 * self.image_color)
        return self

    def apply_cutoff(self, cutoff):
        if cutoff is None or self.image_array is None:  # Cutoff does not apply on image file.
            return
        # The cutoff applies to the texel alpha scaled by the material's alpha factor, which image_color holds
        self.image_array = np.where(self.image_array * self.image_color >= 255.0 * cutoff, 255, 0).astype(np.uint8)
        self.image_color = np.ones_like(self.image_color)

    def export(self, exporting: Exporting) -> dict | None:
        """Return the pixels of the image and how they are read, or None where a file can hold none of it.

        A texture read from an HDR or EXR file keeps its path rather than its pixels, so nothing of it travels.
        """
        if self.image_array is None:
            return None
        return {
            "image_path": self.image_path,
            "image_array": exporting.array(self.image_array),
            "image_color": exporting.value(self.image_color, UnitIntervalArrayType),
            "encoding": self.encoding,
            "is_batched": self.is_batched,
        }

    @classmethod
    def load(cls, raw: dict, loading: Loading) -> "ImageTexture":
        """Recreate the texture from the pixels a file holds, which the constructor would read from a file instead."""
        image_path = raw["image_path"]
        return cls.model_construct(
            image_path=image_path if isinstance(image_path, str | None) else tuple(image_path),
            image_array=loading.array(raw["image_array"]),
            image_color=loading.value(raw["image_color"], UnitIntervalArrayType),
            encoding=raw["encoding"],
            is_batched=raw.get("is_batched", False),
        )


def stack_textures(textures: Sequence["ColorTexture | ImageTexture"]) -> "ColorTexture | ImageTexture":
    """Return the batched texture whose entries are the given textures that are not batched.

    Colors stack into a batched color, and images of one factor and one encoding into a batched image of that factor.
    Otherwise, each entry turns into an image of the resolution of the first image, its factor applied to its pixels,
    since a batched image holds one factor for all its entries.
    """
    if all(isinstance(texture, ColorTexture) for texture in textures):
        return ColorTexture(color=[texture.color for texture in textures], is_batched=True)
    first = next(texture for texture in textures if isinstance(texture, ImageTexture))
    if all(
        isinstance(texture, ImageTexture)
        and texture.encoding == first.encoding
        and len(texture.image_color) == len(first.image_color)
        and np.allclose(texture.image_color, first.image_color, atol=gs.EPS)
        for texture in textures
    ):
        return ImageTexture(
            image_array=[texture.image_array for texture in textures],
            image_color=first.image_color,
            encoding=first.encoding,
            is_batched=True,
        )
    images = []
    for texture in textures:
        if isinstance(texture, ImageTexture):
            image = texture.image_array if texture.image_array.ndim == 3 else texture.image_array[..., None]
            images.append(np.round(image * np.array(texture.image_color)).astype(np.uint8))
        else:
            images.append(
                np.full((*first.image_array.shape[:2], len(texture.color)), mu.color_f32_to_u8(texture.color), np.uint8)
            )
    return ImageTexture(image_array=images, is_batched=True)


class BatchTexture(Texture):
    """
    A batch of textures for batch rendering.

    Parameters
    ----------
    textures : list[Texture | None]
        List of textures.
    """

    textures: Annotated[list[Texture | None], BeforeValidator(list)] = Field(default_factory=list)

    @staticmethod
    def from_images(
        image_paths: Sequence[str] | None = None,
        image_folder: str | None = None,
        image_arrays: Sequence[np.ndarray] | None = None,
        image_colors: Sequence[float] | Sequence[Sequence[float] | None] | None = None,
        encoding: Literal["srgb", "linear"] = "srgb",
    ) -> "BatchTexture":
        """
        Create a batch texture from images.

        Parameters
        ----------
        image_paths : list[str], optional
            List of paths to the image files.
        image_folder : str, optional
            Path to the image folder.
        image_arrays : list[np.ndarray], optional
            List of image arrays.
        image_colors : list[float | list[float]], optional
            List of color factors that will be multiplied with the base color, stored as tuple. Default is None.
        encoding : str, optional
            The encoding way of the image. Possible values are ['srgb', 'linear']. Default is 'srgb'.

            - 'srgb': Encoding of some RGB images.
            - 'linear': All generic images, such as opacity, roughness and normal, should be encoded with 'linear'.
        """
        image_sources = (image_paths, image_folder, image_arrays)
        if sum(x is not None for x in image_sources) != 1:
            gs.raise_exception("Please set exactly one of `image_paths`, `image_folder`, `image_arrays`.")

        image_textures = []
        if image_folder is not None:
            input_image_folder = image_folder
            if not os.path.exists(image_folder):
                image_folder = os.path.join(gs.utils.get_assets_dir(), image_folder)
            if not os.path.exists(image_folder):
                gs.raise_exception(
                    f"Directory not found in either current directory or assets directory: '{input_image_folder}'."
                )
            image_paths = [
                os.path.join(image_folder, image_path)
                for image_path in sorted(os.listdir(image_folder))
                if image_path.lower().endswith(IMAGE_EXTENSIONS)
            ]

        if image_paths is not None:
            num_images = len(image_paths)
        else:
            assert image_arrays is not None
            num_images = len(image_arrays)
        if num_images == 0:
            gs.raise_exception("No images found.")

        if image_colors is not None:
            if isinstance(image_colors[0], float):  # list[float]
                image_colors = [image_colors for _ in range(num_images)]
            else:  # list[list[float]]
                if len(image_colors) != num_images:
                    gs.raise_exception("The number of image colors must be the same as the number of images.")
        else:
            image_colors = [None] * num_images
        assert image_colors is not None

        if image_paths is not None:
            for image_path, image_color in zip(image_paths, image_colors):
                image_textures.append(ImageTexture(image_path=image_path, image_color=image_color, encoding=encoding))
        else:
            assert image_arrays is not None
            for image_array, image_color in zip(image_arrays, image_colors):
                image_textures.append(ImageTexture(image_array=image_array, image_color=image_color, encoding=encoding))

        return BatchTexture(textures=image_textures)

    @staticmethod
    def from_colors(
        colors: Sequence[Sequence[float]],
    ) -> "BatchTexture":
        """
        Create a batch texture from colors.

        Parameters
        ----------
        colors : list[list[float]]
            List of colors.
        """
        return BatchTexture(textures=[ColorTexture(color=color) for color in colors])

    @computed_field
    @cached_property
    def is_black(self) -> bool:
        return all(texture is None or texture.is_black for texture in self.textures)

    @computed_field
    @cached_property
    def requires_uv(self) -> bool:
        return any(texture is not None and texture.requires_uv for texture in self.textures)

    def check_dim(self, dim: int) -> "BatchTexture":
        return BatchTexture(
            textures=[texture.check_dim(dim) if texture is not None else None for texture in self.textures]
        ).check_simplify()

    def check_simplify(self) -> "BatchTexture":
        self.textures = [texture.check_simplify() if texture is not None else None for texture in self.textures]
        return self

    def apply_cutoff(self, cutoff: float) -> None:
        for texture in self.textures:
            if texture is not None:
                texture.apply_cutoff(cutoff)

    def merge(self, other: "BatchTexture") -> None:
        self.textures.extend(other.textures)


class _ImagelessTexture(Texture):
    """A texture holding no image at declaration, which the checks and simplifications of static textures keep as is."""

    def check_dim(self, dim: int) -> Texture | None:
        return None

    def check_simplify(self) -> Self:
        return self

    def apply_cutoff(self, cutoff: float) -> None:
        pass

    @computed_field
    @cached_property
    def is_black(self) -> bool:
        return False

    @computed_field
    @cached_property
    def requires_uv(self) -> bool:
        return True


class AssetTexture(_ImagelessTexture):
    """
    The texture the asset gives the map of a surface holding a dynamic texture, as a variant of that dynamic texture.

    It is valid only in the 'variants' of a ``DynamicTexture``, and resolves when the entity is added: to the image or
    the color the asset gives the map, or to no texture where it gives none. Each visual geom drawing the dynamic
    texture resolves it to the texture its own asset material gives, so one declaration serves several materials.

    Parameters
    ----------
    is_batched : bool, optional
        Whether each environment holds its own copy of the texture of the asset, which ``set_texture`` draws pixels
        into (see ``ImageTexture``). Default is False.
    """

    is_batched: StrictBool = False


def texture_to_rgba(
    texture: "ColorTexture | ImageTexture | AssetTexture | None",
    resolution: tuple[int, int] | None = None,
    texture_map_type: TEXTURE_MAP_TYPE = TEXTURE_MAP_TYPE.DIFFUSE,
) -> np.ndarray:
    """Return the entries of a texture as an RGBA array, in the format of a dynamic texture in a map.

    The array has shape (n_entries, height, width, 4), with one entry for a texture that is not batched. The diffuse map
    holds sRGB uint8 values and the emissive map linear float16 values (see ``DynamicTexture``). An image is resized to
    'resolution', given as (width, height), and converted from its own encoding. A color is sRGB in the diffuse map and
    linear in the emissive map, as for a static texture. A gray or gray-alpha color spans the three color channels, as
    an image of one or two channels does. Without 'resolution', an image keeps its own and a color spans one pixel. No
    texture, an unresolved asset texture included, gives a black and magenta checkerboard of 8 by 8 cells.
    """
    width, height = resolution if resolution is not None else (1, 1)
    if not isinstance(texture, (ColorTexture, ImageTexture)):
        is_magenta = ((np.arange(height)[:, None] * 8 // height) + (np.arange(width) * 8 // width)) % 2 == 1
        rgba = np.where(is_magenta[None, ..., None], [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0])
    else:
        if isinstance(texture, ImageTexture):
            image_array = texture.image_array
            if image_array is None:
                gs.raise_exception(f"A dynamic or batched texture takes no HDR image, got '{texture.image_path}'.")
            rgba = []
            for image in image_array if texture.is_batched else (image_array,):
                if image.ndim == 3 and image.shape[-1] == 1:
                    image = image[..., 0]
                image = Image.fromarray(image).convert("RGBA")
                if resolution is not None:
                    image = image.resize(resolution, Image.Resampling.BILINEAR)
                rgba.append(np.asarray(image) / 255.0)
            rgba, colors, is_srgb = np.stack(rgba), np.array([texture.image_color]), texture.encoding == "srgb"
        else:
            colors = np.atleast_2d(texture.color)
            rgba, is_srgb = np.ones((len(colors), height, width, 4)), texture_map_type == TEXTURE_MAP_TYPE.DIFFUSE
        # sRGB transfer function of IEC 61966-2-1, between the encoded values and the linear ones
        if texture_map_type == TEXTURE_MAP_TYPE.EMISSIVE and is_srgb:
            color = rgba[..., :3]
            rgba[..., :3] = np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)
        elif texture_map_type == TEXTURE_MAP_TYPE.DIFFUSE and not is_srgb:
            color = rgba[..., :3]
            rgba[..., :3] = np.where(color <= 0.0031308, color * 12.92, 1.055 * color ** (1.0 / 2.4) - 0.055)
        factor = np.ones((len(colors), 4))
        factor[:, :3] = colors[:, :3] if colors.shape[-1] >= 3 else colors[:, :1]
        if colors.shape[-1] in (2, 4):
            factor[:, 3] = colors[:, -1]
        rgba = rgba * factor[:, None, None]
    if texture_map_type == TEXTURE_MAP_TYPE.EMISSIVE:
        return rgba.astype(np.float16)
    return (rgba * 255.0).round().astype(np.uint8)


class DynamicTexture(_ImagelessTexture):
    """
    A texture whose image changes while the simulation runs, such as a screen or an indicator light.

    It fills the diffuse or the emissive map of a surface, and requires a mesh with texture coordinates. Each parallel
    environment shows one of a fixed set of variants declared here. A variant whose texture is batched holds one image
    per environment, which ``set_texture`` draws pixels into at runtime. Once the scene is built, ``vgeom.set_texture`` shows a
    variant or draws pixels in each environment, and ``vgeom.get_texture`` returns the variant each environment shows.
    Each visual geom showing the texture switches and draws on its own, while ``entity.set_texture`` sets the visual
    geoms it selects at once and ``entity.get_vgeoms`` finds them by the selection that declared the texture. Each
    renderer holds its own copy of the pixels, and only the rasterizer renders a dynamic texture yet.

    The variants and drawn pixels are RGBA images, whose format the map sets. The diffuse map holds 8-bit sRGB values,
    the alpha channel standing for the opacity in place of the opacity of the surface. The emissive map holds 16-bit
    linear values, which let an emission exceed 1 at twice the memory per pixel.

    Parameters
    ----------
    resolution : tuple[int, int]
        The resolution of every variant and drawn image, as (width, height) in pixels. Rendering uploads the pixels each
        time they are drawn, so a larger image costs more per drawing, while showing a variant costs nothing.
    variants : tuple[ColorTexture | ImageTexture | AssetTexture | None, ...] | dict[str, ...], optional
        The images an environment switches between, each a color or an image resized to 'resolution', an
        ``AssetTexture`` for the texture the asset gives the map, or None for no texture, which shows a black and
        magenta checkerboard. A mapping names them, so that they are shown by name as well as by position. The variants
        are fixed once the scene is built, except the batched ones, which take pixels drawn per environment, each
        environment in its own copy. A variant costs the memory of one image per visual geom showing the texture and per
        renderer, or of
        one image per environment for a batched one. Default is one variant with no texture.
    initial_variant : int | str, optional
        The variant every environment shows at build and after a reset, by position or by name. Default is 0.
    """

    resolution: PositiveVec2IType
    variants: (
        Annotated[tuple[ColorTexture | ImageTexture | AssetTexture | None, ...], Field(min_length=1, strict=False)]
        | Annotated[dict[str, ColorTexture | ImageTexture | AssetTexture | None], Field(min_length=1)]
    ) = (None,)
    initial_variant: StrictInt | str = 0

    # The texture an asset gives the map it fills, which its asset texture variants show (see 'resolve_asset_texture')
    _asset_texture: "ColorTexture | ImageTexture | None" = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_variants(self) -> Self:
        self.get_variants_idx(self.initial_variant)
        return self

    def resolve_asset_texture(self, asset_texture: Texture | None) -> "DynamicTexture":
        """Return a copy whose asset texture variants show the texture an asset gives the map.

        If 'asset_texture' is itself a dynamic texture, the asset texture that one resolved to is used instead. A batched
        asset texture variant resolves to a batched copy of that texture. The texture itself is returned when it holds
        no asset texture variant, or when the asset gives no color and no image.
        """
        if isinstance(asset_texture, DynamicTexture):
            asset_texture = asset_texture._asset_texture
        if not isinstance(asset_texture, (ColorTexture, ImageTexture)) or not any(
            isinstance(variant, AssetTexture) for variant in self.variants_texture
        ):
            return self
        variants = []
        for variant in self.variants_texture:
            if isinstance(variant, AssetTexture):
                if not variant.is_batched or asset_texture.is_batched:
                    variant = asset_texture
                else:
                    variant = stack_textures((asset_texture,))
            variants.append(variant)
        variants = dict(zip(self.variants_name, variants)) if self.variants_name else tuple(variants)
        texture = self.model_copy(update={"variants": variants})
        texture._asset_texture = asset_texture
        return texture

    @property
    def variants_name(self) -> tuple[str, ...]:
        """The name of each variant, empty when the variants are not named."""
        return tuple(self.variants) if isinstance(self.variants, dict) else ()

    @property
    def variants_texture(self) -> tuple["ColorTexture | ImageTexture | AssetTexture | None", ...]:
        """The texture of each variant, in the order of their positions."""
        return tuple(self.variants.values()) if isinstance(self.variants, dict) else self.variants

    @property
    def variants_is_batched(self) -> tuple[bool, ...]:
        """Whether each variant holds one image per environment, which pixels can be drawn into."""
        return tuple(texture is not None and texture.is_batched for texture in self.variants_texture)

    def get_variants_idx(self, variants: "int | str | Sequence[int | str] | np.ndarray") -> np.ndarray:
        """Return the positions of variants given by position or by name, raising on one the set does not hold."""
        variants_name = self.variants_name
        variants_idx = []
        for variant in np.atleast_1d(np.asarray(variants, dtype=object)).tolist():
            if isinstance(variant, str):
                if variant not in variants_name:
                    gs.raise_exception(f"No variant is named '{variant}'. Existing names: {list(variants_name)}.")
                variant = variants_name.index(variant)
            elif not isinstance(variant, (int, np.integer)) or isinstance(variant, bool):
                gs.raise_exception(
                    f"A variant is given by an integer position or a name, got {variant!r}. Give a color as a "
                    "'gs.textures.ColorTexture'."
                )
            if not 0 <= variant < len(self.variants):
                gs.raise_exception(f"Variant {variant} is out of range for {len(self.variants)} variants.")
            variants_idx.append(variant)
        return np.array(variants_idx, dtype=gs.np_int)

    def get_variants_image(self, texture_map_type: TEXTURE_MAP_TYPE) -> tuple[np.ndarray, ...]:
        """Return the entries of each variant as an RGBA array, in the format of a map (see 'texture_to_rgba')."""
        return tuple(texture_to_rgba(texture, self.resolution, texture_map_type) for texture in self.variants_texture)

    def get_initial_image(self) -> np.ndarray:
        """Return the image the initial variant shows first, as an RGBA uint8 array of shape (height, width, 4)."""
        (i_variant,) = self.get_variants_idx(self.initial_variant)
        return texture_to_rgba(self.variants_texture[i_variant], self.resolution)[0]
