import enum
from importlib.util import find_spec

import pytest
from PIL import Image


from ..conftest import SKIP_NO_LUISA, SKIP_NO_MADRONA, lazy_imports

# Imported on first use (see pytest_sessionstart in tests/conftest.py)
with lazy_imports():
    import numpy as np
    import trimesh

    import genesis as gs


class RENDERER_TYPE(enum.IntEnum):
    RASTERIZER = 0
    RAYTRACER = 1
    BATCHRENDER_RASTERIZER = 2
    BATCHRENDER_RAYTRACER = 3


@pytest.fixture(scope="function")
def renderer_type():
    # Default for tests that are not parametrized over renderer types (e.g. viewer and overlay tests), so that the
    # folder-wide 'backend' and 'skip_if_not_installed' fixtures below always resolve. Parametrization overrides it.
    return RENDERER_TYPE.RASTERIZER


@pytest.fixture(scope="function")
def renderer(renderer_type):
    if renderer_type == RENDERER_TYPE.RASTERIZER:
        return gs.renderers.Rasterizer()
    if renderer_type == RENDERER_TYPE.RAYTRACER:
        return gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=15.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
            ],
        )
    return gs.renderers.BatchRenderer(
        use_rasterizer=renderer_type == RENDERER_TYPE.BATCHRENDER_RASTERIZER,
    )


@pytest.fixture(scope="function")
def backend(pytestconfig, renderer_type):
    if renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER):
        return gs.cuda

    if renderer_type == RENDERER_TYPE.RAYTRACER:
        return gs.gpu

    backend = pytestconfig.getoption("--backend") or gs.cpu
    if isinstance(backend, str):
        return getattr(gs.constants.backend, backend)
    return backend


@pytest.fixture(scope="function", autouse=True)
def skip_if_not_installed(renderer_type):
    if renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER):
        if find_spec("gs_madrona") is None:
            pytest.skip(SKIP_NO_MADRONA)
    if renderer_type == RENDERER_TYPE.RAYTRACER:
        # LuisaRenderPy is found through the path that genesis appends as it is imported, which the test module has done
        if find_spec("LuisaRenderPy") is None:
            pytest.skip(SKIP_NO_LUISA)


@pytest.fixture(scope="session")
def base_plus_emissive_glb(asset_tmp_path):
    """Path to a GLB quad whose material pairs a red base-color atlas with a blue emissive atlas."""
    base = Image.fromarray(np.broadcast_to(np.array([230, 20, 20], np.uint8), (16, 16, 3)).copy())
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=base, doubleSided=True)
    material.emissiveTexture = Image.fromarray(np.broadcast_to(np.array([20, 20, 200], np.uint8), (16, 16, 3)).copy())
    material.emissiveFactor = [1.0, 1.0, 1.0]
    path = asset_tmp_path / "base_plus_emissive.glb"
    trimesh.Trimesh(
        vertices=[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        visual=trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [1, 1], [0, 1]], material=material),
        process=False,
    ).export(path)
    return str(path)
