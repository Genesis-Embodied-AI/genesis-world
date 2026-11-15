import numpy as np
import pytest
import trimesh
from pydantic import BaseModel

import genesis as gs
import genesis.utils.mesh as mu
from genesis.options.surfaces import Surface
from genesis.options.textures import ColorTexture

from ..utils.assertions import assert_equal


@pytest.mark.required
def test_surface_shortcut_resolution():
    # Plastic family: color resolves to diffuse_texture; the Rough subclass roughness default (1.0) feeds
    # roughness_texture and default_roughness.
    rough = gs.surfaces.Rough(color=(0.4, 0.4, 0.4))
    assert rough.color == (0.4, 0.4, 0.4)
    assert rough.roughness == 1.0
    assert rough.diffuse_texture.color == (0.4, 0.4, 0.4)
    assert rough.roughness_texture.color == (1.0,)
    assert rough.default_roughness == 1.0

    # Glass: color resolves to specular_texture and the thickness shortcut is honored on the same path.
    glass = gs.surfaces.Glass(color=(0.6, 0.8, 1.0), thickness=0.02)
    assert glass.specular_texture.color == (0.6, 0.8, 1.0)
    assert glass.thickness_texture.color == (0.02,)

    # BSDF exercises multiple shortcuts at once.
    bsdf = gs.surfaces.BSDF(color=(0.2, 0.3, 0.4), roughness=0.3, metallic=0.5)
    assert bsdf.diffuse_texture.color == (0.2, 0.3, 0.4)
    assert bsdf.roughness_texture.color == (0.3,)
    assert bsdf.metallic_texture.color == (0.5,)
    assert bsdf.default_roughness == 0.3

    # Emission: color resolves to emissive_texture.
    emit = gs.surfaces.Emission(color=(1.0, 1.0, 0.0))
    assert emit.emissive_texture.color == (1.0, 1.0, 0.0)

    # Explicit default_roughness wins over the roughness shortcut.
    override = gs.surfaces.Rough(roughness=0.7, default_roughness=0.5)
    assert override.default_roughness == 0.5

    # Nesting an already-resolved surface in another Pydantic model must not re-trigger resolution.
    class Wrapper(BaseModel):
        surface: Surface

    for surface in (rough, glass, bsdf, emit):
        Wrapper(surface=surface)
    Wrapper(surface=rough)
    assert rough.diffuse_texture.color == (0.4, 0.4, 0.4)
    assert rough.roughness_texture.color == (1.0,)

    # Passing both the shortcut and its resolved texture at construction is a user error.
    with pytest.raises(Exception, match="'color' and 'diffuse_texture' cannot both be set"):
        gs.surfaces.Rough(color=(1.0, 0.0, 0.0), diffuse_texture=ColorTexture(color=(0.0, 1.0, 0.0)))
    with pytest.raises(Exception, match="'thickness' and 'thickness_texture' cannot both be set"):
        gs.surfaces.Glass(thickness=0.02, thickness_texture=ColorTexture(color=(0.05,)))


@pytest.mark.required
def test_packed_rgba_resolves_batched_fallback_per_environment():
    # Base-over-emissive selection and its black-base fallback are exercised end to end by the rasterizer emissive
    # test. Per-environment (batched) textures, however, are consumed only by the batch renderer, so their behaviour is
    # asserted here on the packed RGBA directly. A black base entry defers to its emissive while a nonblack entry in
    # the same batch keeps its base; blackness is the effective texel x factor, so a green texture masked to black by
    # a red-only factor still defers. A batch with no fallback keeps a stable identity so the packed-RGBA cache hits.
    base = gs.textures.ImageTexture(image_array=np.full((4, 4, 3), (201, 166, 105), dtype=np.uint8))
    emissive = gs.textures.ImageTexture(image_array=np.full((4, 4, 3), (76, 122, 64), dtype=np.uint8))
    masked_black = gs.textures.ImageTexture(
        image_array=np.full((4, 4, 3), (0, 255, 0), dtype=np.uint8),
        image_color=(1.0, 0.0, 0.0),
    )
    base_batch = gs.textures.BatchTexture(textures=[masked_black, base])
    emissive_batch = gs.textures.BatchTexture(textures=[emissive, emissive])
    rgba = gs.surfaces.BSDF(diffuse_texture=base_batch, emissive_texture=emissive_batch).get_rgba(batch=True)
    assert_equal(rgba.textures[0].image_array[..., :3], emissive.image_array)
    assert_equal(rgba.textures[1].image_array[..., :3], base.image_array)

    stable_surface = gs.surfaces.BSDF(diffuse_texture=gs.textures.BatchTexture(textures=[base, base]))
    assert stable_surface.get_rgba(batch=True) is stable_surface.get_rgba(batch=True)


@pytest.mark.required
def test_flat_base_with_image_emissive_uv_handling():
    # A flat base color plus an image emissive: with UVs the built visual must carry them so the emissive atlas is
    # composited, but without UVs it must fall back to a plain color visual, otherwise a material with an emissive
    # texture but no texcoords makes the shader reference an undeclared uv_0 and fail to compile.
    surface = gs.surfaces.BSDF(
        color=(0.9, 0.0, 0.0),
        emissive_texture=gs.textures.ImageTexture(image_array=np.full((4, 4, 3), (20, 20, 200), dtype=np.uint8)),
    )
    with_uvs = mu.surface_uvs_to_trimesh_visual(surface, uvs=np.zeros((3, 2), dtype=np.float32), n_verts=3)
    assert with_uvs.uv is not None
    assert with_uvs.material.emissiveTexture is not None

    without_uvs = mu.surface_uvs_to_trimesh_visual(surface, uvs=None, n_verts=3)
    assert isinstance(without_uvs, trimesh.visual.ColorVisuals)
