import os
import sys

import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..conftest import IS_INTERACTIVE_VIEWER_AVAILABLE, SKIP_NO_VIEWER
from ..utils.assertions import assert_allclose, assert_equal, rgb_array_to_png_bytes
from ..utils.assets import get_hf_dataset
from .conftest import RENDERER_TYPE


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
def test_batch_deformable_render(monkeypatch, png_snapshot):
    # Having many particles in the scene creates artifacts that are not deterministic between different hardware
    png_snapshot.extension._std_err_threshold = 2.0
    png_snapshot.extension._blurred_kernel_size = 3

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.01,
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1),
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=1e-2,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            visualize_mpm_boundary=True,
            visualize_sph_boundary=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(640, 480),
            run_in_thread=False,
            camera_pos=(6.0, 0.0, 4.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
            # Disable text rendering as it is messing up with pixel matching when using old CPU-based Mesa driver
            enable_help_text=False,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.0,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.5, 0.2),
            size=(0.2, 0.2, 0.2),
            euler=(30, 40, 0),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.0,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=1.0,
            pos=(0.5, 0.5, 0.5),
            euler=(180.0, 0.0, 0.0),
        ),
        material=gs.materials.PBD.Cloth(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/worm/worm.obj",
            pos=(0.3, 0.3, 0.001),
            scale=0.1,
            euler=(90, 0, 0),
        ),
        material=gs.materials.MPM.Muscle(
            E=5e5,
            nu=0.45,
            rho=10000.0,
            sampler="random",
            model="neohooken",
            n_groups=4,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.65),
            size=(0.4, 0.4, 0.4),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="random",
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )
    scene.build(n_envs=4, env_spacing=(2.0, 2.0))

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    scene.visualizer.viewer.update(auto_refresh=True, force=True)
    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )

    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.RAYTRACER])
def test_deformable_uv_textures(renderer_type, renderer, show_viewer, png_snapshot, backend):
    # FIXME: On the macOS CI runners the GPU is a virtualized "Apple Paravirtual device". This test drives that
    # virtualized Metal driver into a persistent, VM-wide broken state: afterwards newComputePipelineStateWithFunction
    # returns "Compilation failed (code=2)" for *every* kernel - even a trivial copy kernel - which is unrecoverable
    # within the VM (survives MTLCompilerService restart and process exit) and cascades to every later test in the job.
    if sys.platform == "darwin" and backend != gs.cpu and os.environ.get("QD_ENABLE_METAL", "1") != "0":
        pytest.skip("FEM implicit (PCG) solve wedges the virtualized Apple Metal GPU on macOS CI runners.")

    # Relax pixel matching because RayTracer is not deterministic between different hardware (eg RTX6000 vs H100), even
    # without denoiser.
    png_snapshot.extension._std_err_threshold = 3.0
    png_snapshot.extension._blurred_kernel_size = 3

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.04,
            substeps=6,
        ),
        fem_options=gs.options.FEMOptions(
            # Implicit solver allows for larger timestep without failure on GPU backend
            use_implicit_solver=True,
            # Reduce number of iterations to speedup runtime
            n_pcg_iterations=40,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.01,
        ),
        vis_options=gs.options.VisOptions(
            # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
            shadow=(renderer_type != RENDERER_TYPE.RASTERIZER),
        ),
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )

    # Add ground plane. For the Rasterizer, keep it small enough to stay within the camera frustum: the Apple
    # Software Renderer misrasterizes a plane whose vertices fall outside the view, breaking pixel matching. The
    # RayTracer is unaffected, so keep its default size to preserve the existing snapshot.
    scene.add_entity(
        morph=gs.morphs.Plane(
            plane_size=(1.3, 1.3) if renderer_type == RENDERER_TYPE.RASTERIZER else (1e3, 1e3),
        ),
        surface=gs.surfaces.Aluminium(
            ior=10.0,
        ),
    )

    # Add PBD cloth with checker texture
    asset_path = get_hf_dataset(pattern="uv_plane.obj")
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/uv_plane.obj",
            scale=0.4,
            pos=(-0.2, 0.0, 0.4),
        ),
        material=gs.materials.PBD.Cloth(),
        surface=gs.surfaces.Default(
            vis_mode="visual",
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/checker.png",
            ),
        ),
    )

    # Add FEM elastic object with checker texture
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.1,
            pos=(0.2, 0.0, 0.2),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.4,
        ),
        surface=gs.surfaces.Default(
            vis_mode="visual",
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/checker.png",
            ),
        ),
    )

    camera = scene.add_camera(
        res=(256, 256),
        pos=(1.5, 1.5, 1),
        lookat=(0.0, 0.0, 0.3),
        fov=45,
        spp=64,
        denoise=False,
        GUI=show_viewer,
    )

    scene.build()

    # Step simulation to deform the objects
    for _ in range(4):
        scene.step()

    # Render and verify
    rgb_arr, *_ = camera.render(rgb=True)
    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_set_vverts(renderer, show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.5, 0.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    # Keep every ground plane vertex inside the camera frustum (camera at (0, -1.5, 0.5), yfov 30 deg,
    # aspect 4:3): the Apple Software Renderer misrasterizes geometry whose vertices fall outside the view,
    # breaking the pixel comparisons below.
    plane = scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 1.8, 0.0),
            plane_size=(1.2, 2.0),
        ),
    )
    entity = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.2,
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            enable_custom_vverts=True,
        ),
    )
    cam = scene.add_camera(
        res=(160, 120),
        pos=(0.0, -1.5, 0.5),
        lookat=(0.0, 0.0, 0.5),
    )
    scene.build(n_envs=3)

    # vfaces_info covers all vgeoms in global vvert space, not just opt-in entities.
    solver = scene.sim.rigid_solver
    vfaces_idx = qd_to_numpy(solver.dyn_info.vfaces.vverts_idx)
    assert vfaces_idx.shape[0] == solver.n_vfaces
    for vg in solver.vgeoms:
        assert_equal(vfaces_idx[vg.vface_start : vg.vface_end], vg.init_vfaces + vg._vvert_start)

    scene.step()
    scene.visualizer.update_visual_states()
    fk_vverts = tensor_to_array(entity.get_vverts())

    # Render baseline through FK path before any set_vverts, to compare deformed vs FK pixels.
    rgb_baseline = tensor_to_array(cam.render(rgb=True, force_render=True)[0])

    # User-driven vverts survive step() because FK does not overwrite the custom buffer.
    entity.set_vverts(7.0)
    scene.step()
    scene.visualizer.update_visual_states()
    assert_equal(entity.get_vverts(), 7.0)

    # A strong out-of-frame override visibly changes the render through the per-env vverts path.
    entity.set_vverts((0.0, 0.0, 10.0))
    rgb_deformed = tensor_to_array(cam.render(rgb=True, force_render=True)[0])
    assert np.abs(rgb_deformed - rgb_baseline).mean() > 5.0

    # set_vverts(None) re-runs FK over the entity's vgeoms and writes the result back.
    entity.set_vverts(None)
    scene.step()
    scene.visualizer.update_visual_states()
    assert_allclose(entity.get_vverts(), fk_vverts, tol=gs.EPS)
    rgb_restored = tensor_to_array(cam.render(rgb=True, force_render=True)[0])
    if sys.platform == "darwin" and scene.visualizer.is_software:
        assert np.abs(rgb_restored.astype(np.float32) - rgb_baseline.astype(np.float32)).mean() < 1.0
    else:
        assert_equal(rgb_restored, rgb_baseline)

    # Vgeom-level write affects only the slice owned by that vgeom.
    vg = entity.vgeoms[0]
    vg.set_vverts(3.0)
    scene.step()
    scene.visualizer.update_visual_states()
    after_vg = tensor_to_array(entity.get_vverts())
    assert_equal(after_vg[..., vg.vvert_start : vg.vvert_end, :], 3.0)
    entity.set_vverts(None)

    # get_vverts returns a copy: mutating the result does not change the underlying buffer.
    copy = entity.get_vverts()
    copy[:] = 99.0
    assert (tensor_to_array(entity.get_vverts()) != 99.0).any()

    # Mix user-driven and FK-driven envs in the same entity.
    entity.set_vverts(7.0, envs_idx=0)
    entity.set_vverts(9.0, envs_idx=[2])
    scene.step()
    scene.visualizer.update_visual_states()
    v = tensor_to_array(entity.get_vverts())
    assert_equal(v[0], 7.0)
    assert_allclose(v[1], fk_vverts[1], tol=gs.EPS)
    assert_equal(v[2], 9.0)

    # set_vverts requires the entity's morph to be created with ``enable_custom_vverts=True``.
    with pytest.raises(gs.GenesisException, match="enable_custom_vverts=True"):
        plane.set_vverts(0.0)


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_set_vverts_sphere_to_box(renderer, show_viewer):
    # A grid-subdivided cube spherified onto the sphere surface gives two shapes sharing one topology, so
    # set_vverts can morph between them. Since the grid has vertices at every cube corner, the morphed silhouette
    # matches the primitive box exactly, while the spherified one differs from the primitive sphere by only a few
    # boundary pixels (two tessellations of the same outline).
    n_seg = 10
    coords = np.linspace(-1.0, 1.0, n_seg + 1)
    cube_verts_list: list[np.ndarray] = []
    vert_index: dict[tuple, int] = {}
    faces: list[list[int]] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            i, j = (axis + 1) % 3, (axis + 2) % 3
            grid = np.empty((n_seg + 1, n_seg + 1), dtype=int)
            for a in range(n_seg + 1):
                for b in range(n_seg + 1):
                    p = np.zeros(3)
                    p[axis] = sign
                    p[i] = coords[a]
                    p[j] = coords[b]
                    key = tuple(np.round(p, 8))
                    idx = vert_index.get(key)
                    if idx is None:
                        idx = len(cube_verts_list)
                        vert_index[key] = idx
                        cube_verts_list.append(p)
                    grid[a, b] = idx
            for a in range(n_seg):
                for b in range(n_seg):
                    v00, v10 = int(grid[a, b]), int(grid[a + 1, b])
                    v01, v11 = int(grid[a, b + 1]), int(grid[a + 1, b + 1])
                    if sign > 0:
                        faces.append([v00, v10, v11])
                        faces.append([v00, v11, v01])
                    else:
                        faces.append([v00, v11, v10])
                        faces.append([v00, v01, v11])

    radius = 0.2
    cube_verts = np.asarray(cube_verts_list, dtype=gs.np_float) * radius
    faces_arr = np.asarray(faces, dtype=gs.np_int)
    x, y, z = cube_verts[:, 0] / radius, cube_verts[:, 1] / radius, cube_verts[:, 2] / radius
    sphere_verts = radius * np.column_stack(
        [
            x * np.sqrt(np.maximum(0.0, 1 - y * y / 2 - z * z / 2 + y * y * z * z / 3)),
            y * np.sqrt(np.maximum(0.0, 1 - z * z / 2 - x * x / 2 + z * z * x * x / 3)),
            z * np.sqrt(np.maximum(0.0, 1 - x * x / 2 - y * y / 2 + x * x * y * y / 3)),
        ]
    )
    sphere_tri = trimesh.Trimesh(vertices=sphere_verts, faces=faces_arr, process=False)

    pos_sphere = (0.0, 0.0, 1.0)
    pos_box = (5.0, 0.0, 1.0)
    pos_deformable = (10.0, 0.0, 1.0)
    cam_dz = 1.0

    scene = gs.Scene(
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=radius,
            pos=pos_sphere,
            fixed=True,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(2.0 * radius, 2.0 * radius, 2.0 * radius),
            pos=pos_box,
            fixed=True,
        ),
    )
    deformable = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(sphere_tri,),
            pos=pos_deformable,
            fixed=True,
            enable_custom_vverts=True,
        ),
    )
    cam = scene.add_camera(
        res=(256, 256),
        pos=(pos_sphere[0], pos_sphere[1], pos_sphere[2] + cam_dz),
        lookat=pos_sphere,
        up=(0.0, 1.0, 0.0),
        GUI=show_viewer,
    )
    scene.build()

    cam.set_pose(pos=(pos_sphere[0], pos_sphere[1], pos_sphere[2] + cam_dz), lookat=pos_sphere)
    mask_sphere = tensor_to_array(cam.render(rgb=False, segmentation=True, force_render=True)[2]) > 0

    cam.set_pose(pos=(pos_box[0], pos_box[1], pos_box[2] + cam_dz), lookat=pos_box)
    mask_box = tensor_to_array(cam.render(rgb=False, segmentation=True, force_render=True)[2]) > 0

    # The reference primitive sphere uses a different tessellation than the spherified cube grid, so the two circle
    # silhouettes differ by a handful of boundary pixels - bound the symmetric difference at 0.5% of the area.
    sphere_tolerance = max(int(0.005 * mask_sphere.sum()), 1)

    cam.set_pose(pos=(pos_deformable[0], pos_deformable[1], pos_deformable[2] + cam_dz), lookat=pos_deformable)
    mask_deformable_sphere = tensor_to_array(cam.render(rgb=False, segmentation=True, force_render=True)[2]) > 0
    assert int((mask_deformable_sphere ^ mask_sphere).sum()) < sphere_tolerance

    deformable.set_vverts(cube_verts + np.array(pos_deformable, dtype=gs.np_float))
    mask_deformable_box = tensor_to_array(cam.render(rgb=False, segmentation=True, force_render=True)[2]) > 0
    assert_equal(mask_deformable_box, mask_box)

    deformable.set_vverts(None)
    mask_deformable_restored = tensor_to_array(cam.render(rgb=False, segmentation=True, force_render=True)[2]) > 0
    assert int((mask_deformable_restored ^ mask_sphere).sum()) < sphere_tolerance
