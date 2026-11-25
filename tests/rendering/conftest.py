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


@pytest.fixture(scope="session")
def translucent_box_obj(asset_tmp_path):
    """Path to an OBJ box of side 0.4 whose red texture has an alpha of 128."""
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 128
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    material = trimesh.visual.material.SimpleMaterial(image=Image.fromarray(rgba, "RGBA"))
    box.visual = trimesh.visual.TextureVisuals(uv=box.vertices[:, :2] + 0.5, material=material)
    path = asset_tmp_path / "translucent_box.obj"
    box.export(path)
    return str(path)


@pytest.fixture(scope="session")
def glossy_quad_obj(asset_tmp_path):
    """Path to an OBJ quad of side 1.0 facing up, whose gray material has a shininess (Ns) of 1000."""
    material = trimesh.visual.material.SimpleMaterial(diffuse=(128, 128, 128, 255), glossiness=1000.0)
    path = asset_tmp_path / "glossy_quad.obj"
    trimesh.Trimesh(
        vertices=[[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        visual=trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [1, 1], [0, 1]], material=material),
        process=False,
    ).export(path, mtl_name="glossy_quad.mtl")
    return str(path)


def build_planes_scene(planes_surface, *, plane_size, camera_pos, camera_res, split_envs, show_viewer):
    """Return an unbuilt scene of a fixed plane per surface, with a camera looking straight down at them.

    The planes stand side by side along the x axis, centered on the origin, a quarter of their width apart. The scene is
    lit uniformly without shadows, so that each plane shows the color of its surface. Returns the scene, its planes and
    its camera.
    """
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=(*camera_pos[:2], 0.0),
        ),
        vis_options=gs.options.VisOptions(
            shadow=False,
            ambient_light=(1.0, 1.0, 1.0),
            split_envs=split_envs,
        ),
        show_viewer=show_viewer,
    )
    planes_x = 1.25 * plane_size[0] * (np.arange(len(planes_surface)) - 0.5 * (len(planes_surface) - 1))
    planes = [
        scene.add_entity(
            morph=gs.morphs.Plane(
                pos=(plane_x, 0.0, 0.0),
                plane_size=plane_size,
                fixed=True,
            ),
            surface=surface,
        )
        for plane_x, surface in zip(planes_x, planes_surface)
    ]
    camera = scene.add_camera(
        pos=camera_pos,
        lookat=(*camera_pos[:2], 0.0),
        res=camera_res,
        fov=40.0,
        GUI=show_viewer,
    )
    return scene, planes, camera
