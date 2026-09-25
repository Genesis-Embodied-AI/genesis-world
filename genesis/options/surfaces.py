import math
from typing import Any, ClassVar, Literal
from typing_extensions import Self

import numpy as np
from pydantic import Field, StrictBool, model_validator

import genesis as gs
from genesis.typing import FArrayType, UnitInterval, ValidFloat
from genesis.utils import mesh as mu
from genesis.utils.misc import SizeCappedCache

from .misc import FoamOptions
from .options import Options
from .textures import Texture, ColorTexture, ImageTexture, BatchTexture

MetalType = Literal["aluminium", "gold", "copper", "brass", "iron", "titanium", "vanadium", "lithium"]


############################ Base ############################
class Surface(Options):
    """
    Base class for all surfaces types in Genesis.

    A ``Surface`` object encapsulates all visual information used for rendering an entity or its sub-components (links,
    geoms, ...). The surface contains different types of textures depending on the surface type (e.g. diffuse, specular,
    roughness, metallic, normal, emissive). Each one of them is a `gs.textures.Texture` object.

    Tip
    ---
    If any of the textures only has single value (instead of a map), you can use the shortcut parameter (e.g., `color`,
    `roughness`, `metallic`, `emissive`) instead of creating a texture object.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    color : tuple | None, optional
        Color of the surface. Shortcut for the primary texture with a single color.
    opacity : float | None, optional
        Opacity of the surface. Shortcut for `opacity_texture` with a single value.
    roughness : float | None, optional
        Roughness of the surface. Shortcut for `roughness_texture` with a single value.
    metallic : float | None, optional
        Metalness of the surface. Shortcut for `metallic_texture` with a single value.
    emissive : tuple | None, optional
        Emissive color of the surface. Shortcut for `emissive_texture` with a single color.
    ior : float, optional
        Index of Refraction.
    default_roughness : float, optional
        Default roughness value when `roughness` is not set and the asset does not have a roughness texture. Defaults
        to 1.0.
    vis_mode : str | None, optional
        How the entity should be visualized, e.g.
        - 'visual': Render the entity's visual geometry.
        - 'collision': Render the entity's collision geometry.
        - 'particle': Render the entity's particle representation (if applicable).
        - 'sdf': Render the reconstructed surface mesh of the entity's sdf.
        - 'recon': Render the reconstructed surface mesh of the entity's particle representation.
    smooth : bool, optional
        Whether to smooth face normals by interpolating vertex normals.
    double_sided : bool | None, optional
        Whether to render both sides of the surface. Useful for non-watertight 2D objects. Defaults to True for Cloth
        material and False for others.
    cutoff : float
        The cutoff angle of emission. Defaults to 180.0.
    normal_diff_clamp : float, optional
        Controls the threshold for computing surface normals by interpolating vertex normals.
    recon_backend : str, optional
        Backend for surface reconstruction. Possible values are ['splashsurf', 'openvdb']. Defaults to 'splashsurf'.
    generate_foam : bool, optional
        Whether to generate foam particles for visual effects for particle-based entities.
    foam_options : gs.options.FoamOptions, optional
        Options for foam generation.
    """

    _color_target: ClassVar[str] = "diffuse_texture"

    # Shortcut fields — resolved to texture fields by _resolve_shortcuts, excluded from serialization.
    color: FArrayType | None = Field(default=None, exclude=True, repr=False)
    opacity: UnitInterval | None = Field(default=None, exclude=True, repr=False)
    roughness: UnitInterval | None = Field(default=None, exclude=True, repr=False)
    metallic: UnitInterval | None = Field(default=None, exclude=True, repr=False)
    emissive: FArrayType | None = Field(default=None, exclude=True, repr=False)

    ior: float | None = None
    default_roughness: UnitInterval = 1.0
    vis_mode: Literal["visual", "collision", "particle", "sdf", "recon"] | None = None
    smooth: StrictBool = True
    double_sided: StrictBool | None = None
    cutoff: float = 180.0
    normal_diff_clamp: float = 180.0
    recon_backend: Literal["splashsurf", "openvdb"] = "splashsurf"
    generate_foam: StrictBool = False
    foam_options: FoamOptions = Field(default_factory=FoamOptions)

    @model_validator(mode="before")
    @classmethod
    def _resolve_shortcuts(cls, data: Any) -> Any:
        # Route each shortcut into its texture counterpart. Subclasses that don't expose a given texture (e.g. Glass has
        # no opacity_texture) are skipped via the model_fields guard, and class defaults like `Rough.roughness = 1.0`
        # are honored.
        for shortcut, texture_field in (
            ("color", cls._color_target),
            ("opacity", "opacity_texture"),
            ("roughness", "roughness_texture"),
            ("metallic", "metallic_texture"),
            ("thickness", "thickness_texture"),
            ("emissive", "emissive_texture"),
        ):
            if texture_field not in cls.model_fields:
                continue
            field = cls.model_fields.get(shortcut)
            value = data.get(shortcut, field.default if field is not None else None)
            if value is None:
                continue
            if data.get(texture_field) is not None:
                gs.raise_exception(f"'{shortcut}' and '{texture_field}' cannot both be set.")
            data[texture_field] = ColorTexture(color=value)

        # Mirror the roughness shortcut into default_roughness unless the user passed an explicit value.
        if "default_roughness" not in data and "roughness" in cls.model_fields:
            roughness = data.get("roughness", cls.model_fields["roughness"].default)
            if roughness is not None:
                data["default_roughness"] = float(roughness)

        return data

    @property
    def texture(self) -> Texture | None:
        raise NotImplementedError

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        raise NotImplementedError

    @property
    def emission(self) -> Texture | None:
        return None

    @property
    def requires_uv(self) -> bool:
        return False

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        return _make_rgba(_resolve_albedo(self.texture, self.emission), None, batch)

    def update_texture(
        self,
        *,
        color_texture: Texture | None = None,
        ior: float | None = None,
        double_sided: bool | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        """
        Update the surface textures using given attributes.

        If the surface already contains corresponding textures, the existing ones have higher priority and won't be
        overridden. Force overriding can be enabled by setting force=True.
        """
        # update primary texture
        if self.texture is None or force:
            if color_texture is not None:
                self.texture = color_texture
            elif not force:
                self.texture = ColorTexture()

        # update ior
        if self.ior is None or force:
            if ior is not None:
                self.ior = ior
            elif not force:
                self.ior = 1.5

        # update double sided
        if self.double_sided is None or force:
            if double_sided is not None:
                self.double_sided = double_sided

    @staticmethod
    def _update_field(
        current: Texture | None, new: Texture | None, default: Texture | None, force: bool
    ) -> Texture | None:
        if current is None or force:
            if new is not None:
                return new
            elif not force and default is not None:
                return default
        return current

    @staticmethod
    def _extract_opacity_from(
        texture: Texture | None, emissive: Texture | None, opacity: Texture | None
    ) -> Texture | None:
        if texture is not None:
            tex = texture.check_dim(3)
            if opacity is None and tex is not None:
                opacity = tex
        if emissive is not None:
            tex = emissive.check_dim(3)
            if opacity is None and tex is not None:
                opacity = tex
        return opacity


# 512 MiB of merged RGBA textures, bounded by the byte size of the image arrays they hold (full-resolution textures
# dominate the footprint; a count-based bound would not reflect it). The entry count is also capped so that flat-color
# surfaces (no image array, e.g. per-geom randomized collision colors) cannot accumulate unbounded. Dropped on genesis
# teardown like the other caches.
_RGBA_CACHE = SizeCappedCache(max_bytes=512 * 1024 * 1024, max_entries=8192)


def _resolve_albedo(base_texture: Texture | None, emissive_texture: Texture | None) -> Texture | None:
    # Packed RGBA feeds renderers that read it as base color, so prefer the base-color texture. Fall back to emissive
    # only when the base is absent or fully black, which keeps zero-base-factor assets (visible imagery authored into
    # emissive) renderable without letting an emissive map override an ordinary authored base color. Batched textures
    # decide the fallback per entry: a mixed batch resolves each environment on its own base color rather than on
    # whether any environment is black.
    is_base_batched = isinstance(base_texture, BatchTexture)
    is_emissive_batched = isinstance(emissive_texture, BatchTexture)
    if is_base_batched or is_emissive_batched:
        base_batch = base_texture.textures if is_base_batched else [base_texture]
        emissive_batch = emissive_texture.textures if is_emissive_batched else [emissive_texture]
        # Least common multiple keeps every base/emissive pairing, matching how _make_rgba combines batch dimensions.
        count = math.lcm(len(base_batch), len(emissive_batch))
        resolved = [
            _resolve_albedo(base_batch[i % len(base_batch)], emissive_batch[i % len(emissive_batch)])
            for i in range(count)
        ]
        # Return the original batch object when no entry fell back, so its identity stays stable and _make_rgba's
        # per-instance cache keeps hitting for shared batched textures instead of rebuilding a wrapper each call.
        if is_base_batched and len(resolved) == len(base_batch) and all(r is b for r, b in zip(resolved, base_batch)):
            return base_texture
        return BatchTexture(textures=resolved)
    if base_texture is not None and not base_texture.is_black:
        return base_texture
    return emissive_texture if emissive_texture is not None else base_texture


def _make_rgba(color_texture: Texture | None, opacity_texture: Texture | None, batch: bool) -> "BatchTexture | Texture":
    # Resolve a surface's color and opacity textures into a single RGBA texture. The result is memoized on the input
    # texture instances: surfaces sharing the same textures (e.g. all textured submeshes of a GLB) then reuse a single
    # merged array instead of allocating one full-resolution copy each. A texture update swaps in a new instance, which
    # is a natural cache miss, so no explicit invalidation is needed.
    cache_key = (color_texture, opacity_texture, batch)
    cached = _RGBA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    all_textures = []
    for texture in (color_texture, opacity_texture):
        textures = texture.textures if isinstance(texture, BatchTexture) else [texture]
        all_textures.append(textures if batch else textures[:1])
    color_textures, opacity_textures = all_textures

    rgba_textures = []
    num_colors = len(color_textures)
    num_opacities = len(opacity_textures)
    num_rgba = num_colors * num_opacities // math.gcd(num_colors, num_opacities)

    for i in range(num_rgba):
        color_texture = color_textures[i % num_colors]
        opacity_texture = opacity_textures[i % num_opacities]

        if isinstance(color_texture, ColorTexture):
            if isinstance(opacity_texture, ColorTexture):
                rgba_texture = ColorTexture(color=(*color_texture.color, *opacity_texture.color))
            elif isinstance(opacity_texture, ImageTexture) and opacity_texture.image_array is not None:
                rgb_color = mu.color_f32_to_u8(color_texture.color)
                rgb_array = np.full((*opacity_texture.image_array.shape[:2], 3), rgb_color, dtype=np.uint8)
                rgba_array = np.dstack((rgb_array, opacity_texture.image_array))
                rgba_scale = (1.0, 1.0, 1.0, *opacity_texture.image_color)
                rgba_texture = ImageTexture(image_array=rgba_array, image_color=rgba_scale)
            else:
                rgba_texture = ColorTexture(color=(*color_texture.color, 1.0))

        elif isinstance(color_texture, ImageTexture) and color_texture.image_array is not None:
            if isinstance(opacity_texture, ColorTexture):
                a_color = mu.color_f32_to_u8(opacity_texture.color)
                a_array = np.full((*color_texture.image_array.shape[:2],), a_color, dtype=np.uint8)
                rgba_array = np.dstack((color_texture.image_array, a_array))
                rgba_scale = (*color_texture.image_color, 1.0)
            elif (
                isinstance(opacity_texture, ImageTexture)
                and opacity_texture.image_array is not None
                and opacity_texture.image_array.shape[:2] == color_texture.image_array.shape[:2]
            ):
                rgba_array = np.dstack((color_texture.image_array, opacity_texture.image_array))
                rgba_scale = (*color_texture.image_color, *opacity_texture.image_color)
            else:
                if isinstance(opacity_texture, ImageTexture) and opacity_texture.image_array is not None:
                    gs.logger.warning("Color and opacity image shapes do not match. Fall back to fully opaque texture.")
                a_array = np.full(color_texture.image_array.shape[:2], 255, dtype=np.uint8)
                rgba_array = np.dstack((color_texture.image_array, a_array))
                rgba_scale = (*color_texture.image_color, 1.0)
            rgba_texture = ImageTexture(image_array=rgba_array, image_color=rgba_scale)

        else:
            rgba_texture = ColorTexture(color=(1.0, 1.0, 1.0, 1.0))

        rgba_textures.append(rgba_texture)

    result = BatchTexture(textures=rgba_textures) if batch else rgba_textures[0]
    n_bytes = sum(
        t.image_array.nbytes
        for t in (result.textures if isinstance(result, BatchTexture) else [result])
        if isinstance(t, ImageTexture) and t.image_array is not None
    )
    _RGBA_CACHE.put(cache_key, result, n_bytes)
    return result


############################ Surface types ############################
class Glass(Surface):
    """
    Glass surface with specular reflection and transmission.

    Parameters
    ----------
    color : tuple | None, optional
        Specular color of the surface. Shortcut for `specular_texture` with a single color.
    roughness : float, optional
        Roughness of the surface. Defaults to 0.0.
    ior : float, optional
        Index of Refraction. Defaults to 1.5.
    subsurface : bool
        Whether to apply a simple BSSRDF subsurface to the glass material.
    thickness : float | None, optional
        The thickness of the top surface when 'subsurface' is set to True. Shortcut for `thickness_texture`.
    specular_texture : gs.textures.Texture | None, optional
        Specular texture of the surface.
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse texture of the surface.
    transmission_texture : gs.textures.Texture | None, optional
        Transmission texture of the surface.
    thickness_texture : gs.textures.Texture | None, optional
        The thickness texture of the top surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    _color_target: ClassVar[str] = "specular_texture"

    roughness: UnitInterval | None = Field(default=0.0, exclude=True, repr=False)
    ior: float | None = 1.5
    thickness: ValidFloat | None = Field(default=None, exclude=True, repr=False)

    subsurface: StrictBool = False
    specular_texture: Texture | None = None
    diffuse_texture: Texture | None = None
    transmission_texture: Texture | None = None
    thickness_texture: Texture | None = None
    roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        # Truncate specular/emissive textures to 3 channels (discard alpha for Glass which has no opacity_texture)
        if self.specular_texture is not None:
            self.specular_texture.check_dim(3)
        if self.emissive_texture is not None:
            self.emissive_texture.check_dim(3)
        if self.specular_texture is not None and self.transmission_texture is None:
            self.transmission_texture = self.specular_texture
        return self

    @property
    def texture(self) -> Texture | None:
        return self.specular_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.specular_texture = value
        self.transmission_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return any(
            t is not None and t.requires_uv
            for t in (
                self.specular_texture,
                self.diffuse_texture,
                self.transmission_texture,
                self.thickness_texture,
                self.roughness_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def update_texture(
        self,
        *,
        roughness_texture: Texture | None = None,
        normal_texture: Texture | None = None,
        emissive_texture: Texture | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        super().update_texture(force=force, **kwargs)
        self.roughness_texture = self._update_field(
            self.roughness_texture, roughness_texture, ColorTexture(color=(self.default_roughness,)), force
        )
        self.normal_texture = self._update_field(self.normal_texture, normal_texture, None, force)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


class Metal(Surface):
    """
    Metal surface.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    metal_type : str, optional
        Type of metal, indicating a specific index of refraction (IOR). Possible values are ['aluminium', 'gold',
        'copper', 'brass', 'iron', 'titanium', 'vanadium', 'lithium']. Defaults to 'iron'.
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse (basic color) texture of the surface.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    roughness: UnitInterval | None = Field(default=0.1, exclude=True, repr=False)

    metal_type: MetalType = "iron"
    diffuse_texture: Texture | None = None
    opacity_texture: Texture | None = None
    roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None

    @property
    def texture(self) -> Texture | None:
        return self.diffuse_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.diffuse_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return any(
            t is not None and t.requires_uv
            for t in (
                self.diffuse_texture,
                self.opacity_texture,
                self.roughness_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        return _make_rgba(_resolve_albedo(self.texture, self.emission), self.opacity_texture, batch)

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        self.opacity_texture = self._extract_opacity_from(
            self.diffuse_texture, self.emissive_texture, self.opacity_texture
        )
        return self

    def update_texture(
        self,
        *,
        opacity_texture: Texture | None = None,
        roughness_texture: Texture | None = None,
        normal_texture: Texture | None = None,
        emissive_texture: Texture | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        super().update_texture(force=force, **kwargs)
        self.opacity_texture = self._update_field(
            self.opacity_texture, opacity_texture, ColorTexture(color=(1.0,)), force
        )
        self.roughness_texture = self._update_field(
            self.roughness_texture, roughness_texture, ColorTexture(color=(self.default_roughness,)), force
        )
        self.normal_texture = self._update_field(self.normal_texture, normal_texture, None, force)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


class Plastic(Surface):
    """
    Plastic surface is the most basic type of surface.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    ior : float, optional
        Index of Refraction. Defaults to 1.0.
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse (basic color) texture of the surface.
    specular_texture : gs.textures.Texture | None, optional
        Specular texture of the surface.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    ior: float | None = 1.0

    diffuse_texture: Texture | None = None
    specular_texture: Texture | None = None
    opacity_texture: Texture | None = None
    roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None

    @property
    def texture(self) -> Texture | None:
        return self.diffuse_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.diffuse_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return any(
            t is not None and t.requires_uv
            for t in (
                self.diffuse_texture,
                self.specular_texture,
                self.opacity_texture,
                self.roughness_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        return _make_rgba(_resolve_albedo(self.texture, self.emission), self.opacity_texture, batch)

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        self.opacity_texture = self._extract_opacity_from(
            self.diffuse_texture, self.emissive_texture, self.opacity_texture
        )
        return self

    def update_texture(
        self,
        *,
        opacity_texture: Texture | None = None,
        roughness_texture: Texture | None = None,
        normal_texture: Texture | None = None,
        emissive_texture: Texture | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        super().update_texture(force=force, **kwargs)
        self.opacity_texture = self._update_field(
            self.opacity_texture, opacity_texture, ColorTexture(color=(1.0,)), force
        )
        self.roughness_texture = self._update_field(
            self.roughness_texture, roughness_texture, ColorTexture(color=(self.default_roughness,)), force
        )
        self.normal_texture = self._update_field(self.normal_texture, normal_texture, None, force)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


class BSDF(Surface):
    """
    Disney BSDF surface with principled shading.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    ior : float, optional
        Index of Refraction. Defaults to 1.0.
    specular_trans : float, optional
        Specular transmission. Defaults to 0.0.
    diffuse_trans : float, optional
        Diffuse transmission. Defaults to 0.0.
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse (basic color) texture of the surface.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    metallic_texture : gs.textures.Texture | None, optional
        Metallic texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    ior: float | None = 1.0

    diffuse_texture: Texture | None = None
    opacity_texture: Texture | None = None
    roughness_texture: Texture | None = None
    metallic_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None
    specular_trans: float = 0.0
    diffuse_trans: float = 0.0

    @property
    def texture(self) -> Texture | None:
        return self.diffuse_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.diffuse_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return any(
            t is not None and t.requires_uv
            for t in (
                self.diffuse_texture,
                self.opacity_texture,
                self.roughness_texture,
                self.metallic_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        return _make_rgba(_resolve_albedo(self.texture, self.emission), self.opacity_texture, batch)

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        self.opacity_texture = self._extract_opacity_from(
            self.diffuse_texture, self.emissive_texture, self.opacity_texture
        )
        return self

    def update_texture(
        self,
        *,
        opacity_texture: Texture | None = None,
        roughness_texture: Texture | None = None,
        metallic_texture: Texture | None = None,
        normal_texture: Texture | None = None,
        emissive_texture: Texture | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        super().update_texture(force=force, **kwargs)
        self.opacity_texture = self._update_field(
            self.opacity_texture, opacity_texture, ColorTexture(color=(1.0,)), force
        )
        self.roughness_texture = self._update_field(
            self.roughness_texture, roughness_texture, ColorTexture(color=(self.default_roughness,)), force
        )
        self.metallic_texture = self._update_field(self.metallic_texture, metallic_texture, None, force)
        self.normal_texture = self._update_field(self.normal_texture, normal_texture, None, force)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


class Emission(Surface):
    """
    Emission surface. This surface emits light. Note that in Genesis's ray tracing pipeline, lights are not a special
    type of objects, but simply entities with emission surfaces.

    Parameters
    ----------
    color : tuple | None, optional
        Emissive color. Shortcut for `emissive_texture` with a single color.
    emissive : tuple | None, optional
        Emissive color. Shortcut for `emissive_texture` with a single color.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    _color_target: ClassVar[str] = "emissive_texture"

    emissive_texture: Texture | None = None

    @property
    def texture(self) -> Texture | None:
        return self.emissive_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.emissive_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return self.emissive_texture is not None and self.emissive_texture.requires_uv

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        if self.emissive_texture is not None:
            self.emissive_texture.check_dim(3)
        return self

    def update_texture(self, *, emissive_texture: Texture | None = None, force: bool = False, **kwargs) -> None:
        super().update_texture(force=force, **kwargs)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


############################ Handy shortcuts ############################
class Default(BSDF):
    """
    The default surface type used in Genesis. This is an alias for `BSDF`.
    """

    pass


class Rough(Plastic):
    """
    Shortcut for a rough plastic surface.
    """

    roughness: UnitInterval | None = Field(default=1.0, exclude=True, repr=False)
    ior: float | None = 1.5


class Smooth(Plastic):
    """
    Shortcut for a smooth plastic surface.
    """

    roughness: UnitInterval | None = Field(default=0.1, exclude=True, repr=False)
    ior: float | None = 1.5


class Reflective(Plastic):
    """
    Shortcut for a reflective (smoother than `Smooth`) plastic surface.
    """

    roughness: UnitInterval | None = Field(default=0.01, exclude=True, repr=False)
    ior: float | None = 2.0


class Collision(Plastic):
    """
    Default surface type for collision geometry with a grey color by default.
    """

    color: FArrayType | None = Field(default=(0.5, 0.5, 0.5), exclude=True, repr=False)


class Water(Glass):
    """
    Shortcut for a water surface (using Glass surface with proper values).
    """

    color: FArrayType | None = Field(default=(0.61, 0.98, 0.93), exclude=True, repr=False)
    roughness: UnitInterval | None = Field(default=0.2, exclude=True, repr=False)
    ior: float | None = 1.2


class Iron(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'iron'`.
    """

    pass


class Aluminium(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'aluminium'`.
    """

    metal_type: MetalType = "aluminium"


class Copper(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'copper'`.
    """

    metal_type: MetalType = "copper"


class Gold(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'gold'`.
    """

    metal_type: MetalType = "gold"
