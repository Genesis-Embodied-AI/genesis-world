import os
from functools import cached_property
from typing import Annotated, Sequence, Literal, Iterable, cast

import numpy as np
from PIL import Image
from pydantic import model_validator, computed_field, BeforeValidator, Field

import genesis as gs
import genesis.utils.mesh as mu
from genesis.typing import LaxUnitIntervalArrayType, LaxFArrayType, UnitIntervalArrayType, NDArrayType

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
    # is a valid key for caches such as the get_rgba cache.
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
    A texture that consists of a single color.

    Parameters
    ----------
    color : list of float
        A list of color values, stored as tuple, supporting any number of channels within the range [0.0, 1.0].
        Default is (1.0, 1.0, 1.0).
    """

    color: LaxFArrayType = (1.0, 1.0, 1.0)

    def check_dim(self, dim: int) -> Texture | None:
        if len(self.color) > dim:
            self.color, res = self.color[:dim], self.color[dim]
            return ColorTexture(color=res)
        return None

    def check_simplify(self) -> "ColorTexture":
        return self

    def apply_cutoff(self, cutoff: float) -> None:
        if cutoff is None:
            return
        self.color = tuple(1.0 if c >= cutoff else 0.0 for c in self.color)

    @computed_field
    @cached_property
    def is_black(self) -> bool:
        assert gs.EPS is not None
        return all(c < gs.EPS for c in self.color)

    @computed_field
    @cached_property
    def requires_uv(self) -> bool:
        return False


class ImageTexture(Texture, SerializationMixin):
    """
    A texture with a texture map (image).

    Parameters
    ----------
    image_path : str, optional
        Path to the image file.
    image_array : np.ndarray, optional
        Image array.
    image_color : float or list of float, optional
        The factor that will be multiplied with the base color, stored as tuple. Default is None.
    encoding : str, optional
        The encoding way of the image. Possible values are ['srgb', 'linear']. Default is 'srgb'.

        - 'srgb': Encoding of some RGB images.
        - 'linear': All generic images, such as opacity, roughness and normal, should be encoded with 'linear'.
    """

    image_path: str | None = None
    image_array: NDArrayType | None
    image_color: UnitIntervalArrayType
    encoding: Annotated[Literal["srgb", "linear"], BeforeValidator(lambda e: str(e).lower())] = "srgb"

    def __init__(
        self,
        *,
        image_path: str | None = None,
        image_array: np.ndarray | None = None,
        image_color: LaxUnitIntervalArrayType | float | None = None,
        encoding: str = "srgb",
        **data,
    ) -> None:
        super().__init__(
            image_path=image_path,
            image_array=image_array,
            image_color=image_color,
            encoding=encoding,
            **data,
        )

    @model_validator(mode="before")
    @classmethod
    def _validate_and_load(cls, data: dict) -> dict:
        image_path, image_array = data.get("image_path"), data.get("image_array")

        if not ((image_path is not None) ^ (image_array is not None)):
            gs.raise_exception("Please set either `image_path` or `image_array`.")

        if image_path is not None:
            # Look for absolute image path
            if not os.path.exists(image_path):
                candidate_image_path = os.path.join(gs.utils.get_assets_dir(), image_path)
                if not os.path.exists(candidate_image_path):
                    gs.raise_exception(
                        f"File not found in either current directory or assets directory: '{image_path}'."
                    )
                image_path = candidate_image_path

            # Load image_path as actual image_array, unless for special texture images (e.g. `.hdr` and `.exr`) that
            # are only supported by Raytracer.
            if image_path.endswith(HDR_EXTENSIONS):
                data.setdefault("encoding", "linear")
                if data["encoding"] != "linear":
                    gs.raise_exception("HDR images requires linear encoding.")
                if image_path.endswith((".exr")):
                    image_path = mu.check_exr_compression(image_path)
            else:
                image_array = mu.PIL_to_array(Image.open(image_path))
        else:
            # Normalize image array
            if not isinstance(image_array, np.ndarray):
                gs.raise_exception("`image_array` needs to be a numpy array.")
            if image_array.dtype != np.uint8:
                if np.issubdtype(image_array.dtype, np.floating):
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255.0).round()
                    image_array = np.clip(image_array, 0.0, 255.0).astype(np.uint8)
                elif np.issubdtype(image_array.dtype, np.integer):
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                elif image_array.dtype == np.bool_:
                    image_array = image_array.astype(np.uint8) * 255
                else:
                    gs.raise_exception(
                        f"Unsupported image dtype {image_array.dtype}. Only uint8, integer, floating-point, or bool "
                        "types are supported."
                    )
        data["image_path"], data["image_array"] = image_path, image_array

        # Resolve image color
        image_color = data.get("image_color")
        if image_array is None:
            channel = 3
        elif image_array.ndim == 3:
            channel = image_array.shape[2]
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
        channels = self.image_array.shape[-1] if self.image_array.ndim == 3 else 1
        channel_peaks = self.image_array.reshape(-1, channels).max(axis=0)
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
        return self.image_array.shape[2] if self.image_array.ndim == 3 else 1

    @computed_field
    @cached_property
    def mean_color(self) -> NDArrayType:
        if self.image_array is None:
            return np.ones(3, dtype=np.float16)
        return cast(np.ndarray, (np.mean(self.image_array, axis=(0, 1), dtype=np.float32) / 255).astype(np.float16))

    def check_dim(self, dim: int) -> Texture | None:
        if self.image_array is not None:
            if self.channel > dim:
                self.image_array, res_array = self.image_array[:, :, :dim], self.image_array[:, :, dim]
                self.image_color, res_color = self.image_color[:dim], self.image_color[dim:]
                return ImageTexture(image_array=res_array, image_color=res_color, encoding="linear").check_simplify()
        return None

    def check_simplify(self) -> "ImageTexture | ColorTexture":
        if self.image_array is None:
            return self
        max_color = np.max(self.image_array, axis=(0, 1))
        min_color = np.min(self.image_array, axis=(0, 1))
        if np.all(min_color == max_color):
            return ColorTexture(color=max_color.reshape(-1) / 255.0 * self.image_color)
        return self

    def apply_cutoff(self, cutoff):
        if cutoff is None or self.image_array is None:  # Cutoff does not apply on image file.
            return
        self.image_array = np.where(self.image_array >= 255.0 * cutoff, 255, 0).astype(np.uint8)

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
        }

    @classmethod
    def load(cls, raw: dict, loading: Loading) -> "ImageTexture":
        """Recreate the texture from the pixels a file holds, which the constructor would read from a file instead."""
        return cls.model_construct(
            image_path=raw["image_path"],
            image_array=loading.array(raw["image_array"]),
            image_color=loading.value(raw["image_color"], UnitIntervalArrayType),
            encoding=raw["encoding"],
        )


class BatchTexture(Texture):
    """
    A batch of textures for batch rendering.

    Parameters
    ----------
    textures : List[Optional[Texture]]
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
        image_paths : List[str], optional
            List of paths to the image files.
        image_folder : str, optional
            Path to the image folder.
        image_arrays : List[np.ndarray], optional
            List of image arrays.
        image_colors : List[Union[float, List[float]]], optional
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
            if isinstance(image_colors[0], float):  # List[float]
                image_colors = [image_colors for _ in range(num_images)]
            else:  # List[List[float]]
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
        colors : List[List[float]]
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
