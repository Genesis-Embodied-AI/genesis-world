import os
import sys

import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.engine.render_context import DynamicMesh
from genesis.utils.misc import tensor_to_array

from ..conftest import IS_INTERACTIVE_VIEWER_AVAILABLE, SKIP_NO_VIEWER
from ..utils.assertions import assert_allclose, assert_equal, rgb_array_to_png_bytes
from ..utils.assets import get_hf_dataset
from .conftest import RENDERER_TYPE


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
def test_batch_deformable_render(monkeypatch, png_snapshot):
    # Every visible feature spans several pixels, since each renderer rasterizes sub-pixel geometry its own way: the
    # liquid holds a few hundred particles 5 pixels wide, and the wireframes of the solver boundaries are left out.

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
            particle_size=0.05,
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1),
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=1e-2,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            # Shadows are forcibly disabled on software rendering backends
            shadow=False,
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

    # The planes of the environments tile the ground and stay within the camera frustum
    scene.add_entity(
        morph=gs.morphs.Plane(
            plane_size=(2.0, 2.0),
        ),
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
    worm = scene.add_entity(
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
    liquid = scene.add_entity(
        spec=gs.entities.SPH(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.65),
                size=(0.4, 0.4, 0.4),
            ),
            material=gs.materials.SPH.Liquid(
                sampler="random",
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.8, 1.0),
            ),
            vis=gs.options.EntityVisOptions(
                debug=True,
            ),
        ),
    )
    scene.build(n_envs=4, env_spacing=(2.0, 2.0))

    # Every environment renders the liquid in a state of its own: as sampled, lifted, flattened onto its bottom, and
    # with every other particle deactivated
    liquid_pos = liquid.get_particles_pos()
    liquid_pos[1, :, 2] += 0.2
    liquid_bottom = liquid_pos[2, :, 2].min()
    liquid_pos[2, :, 2] = liquid_bottom + 0.25 * (liquid_pos[2, :, 2] - liquid_bottom)
    liquid.set_particles_pos(liquid_pos)
    is_liquid_active = liquid.get_particles_active()
    is_liquid_active[3, ::2] = False
    liquid.set_particles_active(is_liquid_active)
    # The worm tears in the third environment, its upper half lifted away from the lower one
    worm_pos = worm.get_particles_pos()
    is_worm_upper = worm_pos[2, :, 2] > worm_pos[2, :, 2].mean()
    worm_pos[2, is_worm_upper, 2] += 0.5
    worm.set_particles_pos(worm_pos)

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    scene.visualizer.viewer.update(auto_refresh=True, force=True)
    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )

    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot

    # The torn worm is drawn as the surface reconstructed from its particles in its environment, two pieces rather than
    # its skinned mesh stretched over the gap between them, which it keeps in the others.
    worm_skinned, worm_torn = (
        shape
        for shape in scene.sim.render_context.get_shapes(debug=False)
        if isinstance(shape, DynamicMesh) and shape.seg_keys.entity == worm.idx
    )
    assert_equal(worm_skinned.is_active, (True, True, False, True))
    assert_equal(worm_torn.envs_idx, (2,))
    worm_torn_mesh = trimesh.Trimesh(tensor_to_array(worm_torn.verts[0]), tensor_to_array(worm_torn.faces))
    assert len(worm_torn_mesh.split(only_watertight=False)) >= 2


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
    if renderer_type == RENDERER_TYPE.RAYTRACER:
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
    sphere_tri = trimesh.creation.icosphere(subdivisions=2, radius=0.2)
    sphere_verts = np.array(sphere_tri.vertices, dtype=gs.np_float)
    squashed_verts = sphere_verts * np.array((1.0, 1.0, 0.2))
    lifted_verts = sphere_verts + np.array((0.0, 0.0, 10.0))
    HET_POSITIONS = ((-0.5, 0.3, 0.3), (-0.5, 0.3, 0.6))

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
        spec=gs.entities.Rigid(
            morph=gs.morphs.MeshSet(
                files=(sphere_tri,),
                pos=(0.0, 0.0, 0.5),
                fixed=True,
            ),
            dynamic_vertices=gs.surfaces.DynamicVertices(
                variants={"sphere": None, "squashed": squashed_verts, "lifted": lifted_verts},
                is_batched=True,
            ),
        ),
    )
    # The entity shows DynamicVertices on its first visual geom alone, its second one following the pose of its link
    arm_spec = gs.entities.Rigid(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(0.45, 0.3, 0.2),
            fixed=True,
        ),
        dynamic_vertices=gs.surfaces.DynamicVertices(
            is_batched=True,
        ),
    )
    arm_spec.select(gs.indices.VGeom.at(1)).set(
        gs.options.VGeom(
            dynamic_vertices=None,
        )
    )
    arm = scene.add_entity(spec=arm_spec)
    # A heterogeneous entity, whose visual geom of each variant shows DynamicVertices in the environments holding it
    het = scene.add_entity(
        spec=gs.entities.Rigid(
            morph=(
                gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=HET_POSITIONS[0], fixed=True),
                gs.morphs.Box(size=(0.15, 0.15, 0.15), pos=HET_POSITIONS[1], fixed=True),
            ),
            dynamic_vertices=gs.surfaces.DynamicVertices(
                is_batched=True,
            ),
        ),
    )
    cam = scene.add_camera(
        res=(160, 120),
        pos=(0.0, -1.5, 0.5),
        lookat=(0.0, 0.0, 0.5),
    )
    scene.build(n_envs=3)

    # Every environment shows the first variant, the mesh of the asset placed at the pose of the visual geom
    sphere_vverts = sphere_verts + np.array((0.0, 0.0, 0.5))
    assert_allclose(entity.get_vverts(), sphere_vverts, tol=gs.EPS)
    rgb_baseline = tensor_to_array(cam.render(rgb=True, force_render=True)[0])

    # A variant is shown by name in some environments, and survives steps
    entity.set_vverts(envs_idx=[1], variant="squashed")
    scene.step()
    vverts = entity.get_vverts()
    assert_allclose(vverts[[0, 2]], sphere_vverts, tol=gs.EPS)
    assert_allclose(vverts[1], squashed_verts + np.array((0.0, 0.0, 0.5)), tol=gs.EPS)
    # A variant given by position is shown in every environment, which overlap in the image, so that the sphere
    # lifted out of the frame leaves it.
    entity.set_vverts(variant=2)
    rgb_lifted = tensor_to_array(cam.render(rgb=True)[0])
    assert np.abs(rgb_lifted.astype(np.float32) - rgb_baseline.astype(np.float32)).mean() > 5.0

    # The batched variant takes vertices per environment in the frame of the visual geom, which follows its link
    batched_verts = np.stack((sphere_verts * 2.0, sphere_verts * 0.5))
    entity.set_vverts(batched_verts, envs_idx=[0, 2])
    entity.set_pos((0.0, 0.0, 0.6))
    vverts = entity.get_vverts()
    assert_allclose(vverts[[0, 2]], batched_verts + np.array((0.0, 0.0, 0.6)), tol=gs.EPS)
    assert_allclose(vverts[1], lifted_verts + np.array((0.0, 0.0, 0.6)), tol=gs.EPS)

    # A reset shows the first variant again in every environment
    scene.reset()
    assert_allclose(entity.get_vverts(), sphere_vverts, tol=gs.EPS)
    rgb_restored = tensor_to_array(cam.render(rgb=True, force_render=True)[0])
    # Rendering the same geometry again may round a color to the next level
    assert_allclose(rgb_restored, rgb_baseline, atol=1)

    # get_vverts returns a copy: mutating the result does not change the vertices drawn.
    copy = entity.get_vverts()
    copy[:] = 99.0
    assert (entity.get_vverts() != 99.0).any()

    # The visual geoms of an entity show DynamicVertices or follow their link each on its own
    dynamic_vgeom, posed_vgeom = arm.vgeoms
    posed_vverts = posed_vgeom.get_vverts()
    dynamic_vgeom.set_vverts(np.zeros((dynamic_vgeom.n_vverts, 3)))
    assert_allclose(dynamic_vgeom.get_vverts(), dynamic_vgeom.get_pos()[:, None], tol=gs.EPS)
    assert_allclose(posed_vgeom.get_vverts(), posed_vverts, tol=gs.EPS)

    # The visual geom of a heterogeneous variant shows the vertices written in the environments holding it, at the pose
    # of the variant, the other environments left as they are.
    for het_vgeom, het_pos in zip(het.vgeoms, HET_POSITIONS):
        envs_held = het_vgeom.active_envs_idx
        vverts_before = tensor_to_array(het_vgeom.get_vverts())
        het_vgeom.set_vverts(np.zeros((het_vgeom.n_vverts, 3)), envs_idx=envs_held)
        vverts = tensor_to_array(het_vgeom.get_vverts())
        assert_allclose(vverts[envs_held], het_pos, tol=gs.EPS)
        envs_other = np.setdiff1d(np.arange(scene.n_envs), envs_held)
        assert_allclose(vverts[envs_other], vverts_before[envs_other], tol=gs.EPS)

    with pytest.raises(gs.GenesisException, match="DynamicVertices"):
        plane.set_vverts(None)
    with pytest.raises(gs.GenesisException, match="single visual geom"):
        arm.set_vverts(None)
    with pytest.raises(gs.GenesisException, match="No variant is named"):
        entity.set_vverts(variant="cube")
    with pytest.raises(gs.GenesisException, match="not both"):
        entity.set_vverts(squashed_verts, variant="squashed")


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
        spec=gs.entities.Rigid(
            morph=gs.morphs.MeshSet(
                files=(sphere_tri,),
                pos=pos_deformable,
                fixed=True,
            ),
            dynamic_vertices=gs.surfaces.DynamicVertices(
                variants=(None, cube_verts),
            ),
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
    mask_sphere = cam.render(rgb=False, segmentation=True, force_render=True)[2] > 0

    cam.set_pose(pos=(pos_box[0], pos_box[1], pos_box[2] + cam_dz), lookat=pos_box)
    mask_box = cam.render(rgb=False, segmentation=True, force_render=True)[2] > 0

    # The reference primitive sphere uses a different tessellation than the spherified cube grid, so the two circle
    # silhouettes differ by a handful of boundary pixels - bound the symmetric difference at 0.5% of the area.
    sphere_tolerance = max(int(0.005 * mask_sphere.sum()), 1)

    cam.set_pose(pos=(pos_deformable[0], pos_deformable[1], pos_deformable[2] + cam_dz), lookat=pos_deformable)
    mask_deformable_sphere = cam.render(rgb=False, segmentation=True, force_render=True)[2] > 0
    assert int((mask_deformable_sphere ^ mask_sphere).sum()) < sphere_tolerance

    deformable.set_vverts(variant=1)
    mask_deformable_box = cam.render(rgb=False, segmentation=True)[2] > 0
    assert_equal(mask_deformable_box, mask_box)

    deformable.set_vverts(None)
    mask_deformable_restored = cam.render(rgb=False, segmentation=True)[2] > 0
    assert int((mask_deformable_restored ^ mask_sphere).sum()) < sphere_tolerance

    # Writing vertices takes the batched variant, which these DynamicVertices leave out
    with pytest.raises(gs.GenesisException, match="is_batched=True"):
        deformable.set_vverts(cube_verts)
