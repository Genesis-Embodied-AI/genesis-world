import itertools
import sys

import numpy as np
import pytest
import torch

import av

import genesis as gs
import genesis.utils.geom as gu
from genesis.options.sensors import RasterizerCameraOptions
from genesis.utils import set_random_seed
from genesis.utils.image_exporter import FrameImageExporter, as_grayscale_image
from genesis.utils.misc import tensor_to_array
from genesis.utils.video_encoder import VideoEncoder

from ..conftest import IS_INTERACTIVE_VIEWER_AVAILABLE, SKIP_NO_VIEWER
from ..utils.assertions import assert_allclose, assert_equal, assert_pixel_match, rgb_array_to_png_bytes
from .conftest import RENDERER_TYPE


@pytest.mark.required
@pytest.mark.parametrize(
    "renderer_type",
    [
        RENDERER_TYPE.RASTERIZER,
        RENDERER_TYPE.RAYTRACER,
        RENDERER_TYPE.BATCHRENDER_RASTERIZER,
        RENDERER_TYPE.BATCHRENDER_RAYTRACER,
    ],
)
def test_render_api(show_viewer, renderer_type, renderer):
    IS_BATCHRENDER = renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER)

    scene = gs.Scene(
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    if IS_BATCHRENDER:
        scene.add_light(
            pos=(0.0, 0.0, 1.5),
            dir=(1.0, 1.0, -2.0),
            directional=True,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
        scene.add_light(
            pos=(4.0, -4.0, 4.0),
            dir=(-1.0, 1.0, -1.0),
            directional=False,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.0),
            radius=1.0,
            fixed=True,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(2.0, 0.0, 0.0),
            size=(0.5, 0.5, 0.5),
        ),
    )
    camera = scene.add_camera(
        pos=(0.0, 0.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        GUI=show_viewer,
    )
    scene.build()

    rgb_arrs, depth_arrs, seg_arrs, normal_arrs = [], [], [], []
    for rgb, depth, seg, normal in itertools.product((True, False), repeat=4):
        rgb_arr, depth_arr, seg_arr, normal_arr = camera.render(rgb=rgb, depth=depth, segmentation=seg, normal=normal)
        if rgb:
            rgb_arrs.append(tensor_to_array(rgb_arr).astype(np.float32))
        if depth:
            if renderer_type == RENDERER_TYPE.BATCHRENDER_RAYTRACER:
                depth_arr[~torch.isfinite(depth_arr)] = 0
            depth_arrs.append(tensor_to_array(depth_arr).astype(np.float32))
        if seg:
            seg_arrs.append(tensor_to_array(seg_arr).astype(np.float32))
        if normal:
            normal_arrs.append(tensor_to_array(normal_arr).astype(np.float32))

    try:
        assert_allclose(np.diff(rgb_arrs, axis=0), 0.0, tol=gs.EPS)
        assert_allclose(np.diff(seg_arrs, axis=0), 0.0, tol=gs.EPS)
        assert_allclose(np.diff(normal_arrs, axis=0), 0.0, tol=gs.EPS)

        # Depth is not matching at machine-precision because of MSAA being disabled for depth-only
        msaa_mask = [0, 1, 2, 4, 5, 6] if renderer_type == RENDERER_TYPE.RASTERIZER else slice(None)
        assert_allclose(np.diff(depth_arrs, axis=0)[msaa_mask], 0.0, tol=gs.EPS)
    except AssertionError:
        if sys.platform == "darwin" and scene.visualizer.is_software:
            pytest.xfail("Flaky on MacOS with Apple Software Renderer.")
        raise

    if renderer_type == RENDERER_TYPE.RASTERIZER:
        # An entity whose pose blew up stays drawn where it last was, and follows its pose again once it is valid
        box.set_pos((float("nan"), 0.0, 0.0))
        rgb_frozen, *_ = camera.render(rgb=True, force_render=True)
        assert_equal(rgb_frozen, rgb_arrs[0])
        box.set_pos((-2.0, 0.0, 0.0))
        rgb_moved, *_ = camera.render(rgb=True, force_render=True)
        assert (rgb_moved != rgb_frozen).any()


@pytest.mark.required
def test_emissive_composites(base_plus_emissive_glb, show_viewer, renderer):
    # The rasterizer must composite emissive on top of the base color under flat ambient light, checked on one scene
    # whose entities are separated by segmentation:
    #  - image_quad: a rigid GLB pairing a red base atlas with a blue emissive atlas keeps red dominant (the base is
    #    neither dropped to black nor replaced by the emissive map) while blue is lifted well above the base's own blue.
    #  - fem_quad: a deformable entity with a red base and a blue emissive behaves the same, exercising the non-rigid
    #    render path (which builds its material identically through surface_uvs_to_trimesh_visual).
    #  - base_green vs emissive_green: a plain green surface and a black-base surface whose green imagery lives in
    #    emissive render at the same brightness. get_rgba already packs that emissive as the albedo, so adding it again
    #    would double it; the two must match.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
        vis_options=gs.options.VisOptions(
            shadow=False,
            ambient_light=(1.0, 1.0, 1.0),
        ),
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    image_quad = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=base_plus_emissive_glb,
            pos=(-4.5, 0.0, 0.0),
            fixed=True,
            collision=False,
            file_meshes_are_zup=True,
        ),
    )
    base_green = scene.add_entity(
        morph=gs.morphs.Box(
            size=(2.0, 2.0, 0.1),
            pos=(-1.5, 0.0, 0.0),
            fixed=True,
        ),
        surface=gs.surfaces.BSDF(
            color=(0.0, 0.6, 0.0),
        ),
    )
    emissive_green = scene.add_entity(
        morph=gs.morphs.Box(
            size=(2.0, 2.0, 0.1),
            pos=(1.5, 0.0, 0.0),
            fixed=True,
        ),
        surface=gs.surfaces.BSDF(
            color=(0.0, 0.0, 0.0),
            emissive=(0.0, 0.6, 0.0),
        ),
    )
    fem_quad = scene.add_entity(
        morph=gs.morphs.Box(
            size=(2.0, 2.0, 0.5),
            pos=(4.5, 0.0, 0.0),
        ),
        material=gs.materials.FEM.Elastic(),
        surface=gs.surfaces.BSDF(
            color=(0.9, 0.0, 0.0),
            emissive=(0.0, 0.0, 0.6),
        ),
    )
    camera = scene.add_camera(
        pos=(0.0, 0.0, 16.0),
        lookat=(0.0, 0.0, 0.0),
        res=(256, 96),
        GUI=show_viewer,
    )
    scene.build()

    rgb, _, segmentation, _ = camera.render(rgb=True, segmentation=True)
    # Segmentation labels background 0 and each entity by its index plus one, giving exact per-entity pixel masks.
    entities = (image_quad, base_green, emissive_green, fem_quad)
    means = {e: rgb[segmentation == e.idx + 1].mean(axis=0) for e in entities}
    assert all(np.isfinite(mean).all() for mean in means.values())

    # Base color preserved and emissive added on top, on both a rigid and a deformable entity: red stays dominant
    # (not replaced by the blue emissive) while blue is lifted well above the base's own blue (which renders near 40).
    for e in (image_quad, fem_quad):
        assert means[e][0] > means[e][2]
        assert means[e][2] > 120.0
    # The emissive that get_rgba already packed as albedo renders once, matching the plain green surface.
    assert means[emissive_green][1] > 120.0
    assert abs(means[emissive_green][1] - means[base_green][1]) < 30.0


@pytest.mark.required
@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
@pytest.mark.parametrize("n_envs", [0, 4])
def test_render_api_advanced(tmp_path, n_envs, show_viewer, png_snapshot, renderer_type, renderer):
    # Small discrepancies between different hardware due the different physics integration
    png_snapshot.extension._std_err_threshold = 1.2

    CAM_RES = (256, 256)
    DIFF_TOL = 0.01
    NUM_STEPS = 5

    IS_BATCHRENDER = renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.04,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        vis_options=gs.options.VisOptions(
            # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
            shadow=(renderer_type != RENDERER_TYPE.RASTERIZER),
            split_envs=False,
        ),
        renderer=renderer,
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Aluminium(
            ior=10.0,
        ),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            merge_fixed_links=False,
        ),
        material=gs.materials.Rigid(
            rho=200.0,
        ),
    )
    cam_debug = scene.add_camera(
        res=(640, 480),
        pos=(1.5, 0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        debug=True,
        GUI=show_viewer,
    )
    cameras = []
    for i in range(max(1 if IS_BATCHRENDER else n_envs, 1)):
        env_idx = None if i < 1 else i
        cam_0 = scene.add_camera(
            res=CAM_RES,
            pos=(1.5, 0.5, 1.5),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            near=0.05,
            far=100.0,
            env_idx=env_idx,
            GUI=show_viewer,
        )
        cam_1 = scene.add_camera(
            res=CAM_RES,
            pos=(0.8, -0.5, 0.8),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            near=0.05,
            far=100.0,
            env_idx=env_idx,
            GUI=show_viewer,
        )
        cam_2 = scene.add_camera(
            res=CAM_RES,
            fov=45,
            env_idx=env_idx,
            near=0.05,
            far=100.0,
            GUI=show_viewer,
        )
        cameras += (cam_0, cam_1, cam_2)
    if IS_BATCHRENDER:
        scene.add_light(
            pos=(0.0, 0.0, 1.5),
            dir=(1.0, 1.0, -2.0),
            directional=True,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
        scene.add_light(
            pos=(4.0, -4.0, 4.0),
            dir=(-1.0, 1.0, -1.0),
            directional=False,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
    scene.build(n_envs=n_envs, env_spacing=(4.0, 4.0))

    # Attach cameras
    for i in range(0, len(cameras), 3):
        cameras[i + 1].follow_entity(robot)
        pose_rel = gu.trans_R_to_T(np.array([0.1, 0.0, 0.2]), np.eye(3))
        cameras[i + 2].attach(robot.get_link("Head_upper"), pose_rel)

    # Create image exporter
    exporter = FrameImageExporter(tmp_path)

    # Initialize the simulation
    set_random_seed(1)
    for i in range(max(n_envs, 1)):
        qpos = torch.zeros(robot.n_dofs, device=gs.device)
        qpos[:2] = torch.as_tensor(np.random.rand(2), dtype=gs.tc_float, device=gs.device) - 0.5
        qpos[2] = 1.0
        qpos[3:6] = 0.5 * (torch.as_tensor(np.random.rand(3), dtype=gs.tc_float, device=gs.device) - 0.5)
        qpos[6:] = torch.as_tensor(np.random.rand(robot.n_dofs - 6), dtype=gs.tc_float, device=gs.device) - 0.5
        robot.set_dofs_position(qpos, envs_idx=([i] if n_envs else None))

        qvel = torch.zeros(robot.n_dofs, device=gs.device)
        qvel[:6] = torch.as_tensor(np.random.rand(6), dtype=gs.tc_float, device=gs.device) - 0.5
        robot.set_dofs_velocity(qvel, envs_idx=([i] if n_envs else None))

    # Run a few simulation steps while monitoring the result
    cam_debug.start_recording(save_to_filename=(tmp_path / "video.mp4"))

    frames_prev = None
    for i in range(NUM_STEPS):
        # Move forward step forward in time
        scene.step()

        # Render cameras
        if IS_BATCHRENDER:
            # Note that the individual cameras is rendered alone first on purpose to make sure it works
            rgb_1, depth_1, seg_1, normal_1 = cam_1.render(
                rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True
            )
            rgb_all, depth_all, seg_all, normal_all = scene.render_all_cameras(
                rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True
            )
            assert all(isinstance(img_data, torch.Tensor) for img_data in (rgb_1, depth_1, seg_1, normal_1))
            assert all(isinstance(img_data, torch.Tensor) for img_data in (*rgb_all, *depth_all, *seg_all, *normal_all))
        else:
            # Emulate batch rendering which is not supported natively
            rgb_all, depth_all, seg_all, normal_all = zip(
                *(
                    camera.render(rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True)
                    for camera in scene._visualizer._cameras
                    if not camera.debug
                )
            )
            if n_envs > 0:
                rgb_all, depth_all, seg_all, normal_all = (
                    tuple(np.swapaxes(np.stack(img_data, axis=0).reshape((n_envs, 3, *img_data[0].shape)), 0, 1))
                    for img_data in (rgb_all, depth_all, seg_all, normal_all)
                )
            rgb_1, depth_1, seg_1, normal_1 = rgb_all[1], depth_all[1], seg_all[1], normal_all[1]

        # Check that the dimensions are valid
        batch_shape = (*((n_envs,) if n_envs else ()), *CAM_RES)
        assert len(rgb_all) == len(depth_all) == 3
        assert all(e.shape == (*batch_shape, 3) for e in (*rgb_all, *seg_all, *normal_all, rgb_1, seg_1, normal_1))
        assert all(e.shape == batch_shape for e in (*depth_all, depth_1))

        # Check that the camera whose output was rendered individually is matching batched output
        for img_data_1, img_data_2 in (
            (rgb_all[1], rgb_1),
            (depth_all[1], depth_1),
            (seg_all[1], seg_1),
            (normal_all[1], normal_1),
        ):
            assert_allclose(img_data_1, img_data_2, tol=gs.EPS)

        # Check that there is something to see here
        depth_normalized_all = tuple(as_grayscale_image(tensor_to_array(img_data)) for img_data in depth_all)
        frame_data = tuple(
            tensor_to_array(img_data).astype(np.float32)
            for img_data in (*rgb_all, *depth_normalized_all, *seg_all, *normal_all)
        )
        for img_data in frame_data:
            for img_data_i in img_data if n_envs else (img_data,):
                assert np.max(np.std(img_data_i.reshape((-1, img_data_i.shape[-1])), axis=0)) > 10.0

        # Export a few frames for later pixel-matching validation
        if i < 2:
            exporter.export_frame_all_cameras(i, rgb=rgb_all, depth=depth_all, segmentation=seg_all, normal=normal_all)
            exporter.export_frame_single_camera(
                i, cam_1.idx, rgb=rgb_1, depth=depth_1, segmentation=seg_1, normal=normal_1
            )

        # Check that cameras are recording different part of the scene
        for rgb_diff in np.diff(frame_data[:3], axis=0):
            for rgb_diff_i in rgb_diff if n_envs else (rgb_diff,):
                assert np.max(np.std(rgb_diff.reshape((-1, rgb_diff_i.shape[-1])), axis=0)) > 10.0

        # Check that images are changing over time.
        # We expect sufficient difference between two consecutive frames.
        if frames_prev is not None:
            try:
                for img_data_prev, img_data in zip(frames_prev, frame_data):
                    img_diff = np.abs(img_data_prev - img_data)
                    assert np.sum(img_diff > np.finfo(np.float32).eps) > DIFF_TOL * img_data.size
            except AssertionError:
                if sys.platform == "darwin" and scene.visualizer.is_software:
                    pytest.xfail("Flaky on MacOS with Apple Software Renderer. Successive captures are too close.")
                raise
        frames_prev = frame_data

        # Add current frame to monitor video
        rgb_debug, *_ = cam_debug.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
        assert isinstance(rgb_debug, np.ndarray)
        assert rgb_debug.shape == (480, 640, 3)

    cam_debug.stop_recording()

    # Every simulation step must have contributed exactly one frame at the default framerate, which is higher than
    # the step rate of this scene
    with av.open(tmp_path / "video.mp4") as container:
        assert sum(1 for _ in container.decode(video=0)) == NUM_STEPS

    # Verify that the output is correct pixel-wise over multiple simulation steps
    try:
        for image_file in sorted(tmp_path.rglob("*.png")):
            with open(image_file, "rb") as f:
                assert f.read() == png_snapshot
    except AssertionError:
        if sys.platform == "darwin" and scene.visualizer.is_software:
            pytest.xfail("Flaky on MacOS with Apple Software Renderer. Pixel-matching failure.")
        raise


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
def test_deterministic(tmp_path, renderer_type, renderer, show_viewer, tol):
    IS_BATCHRENDER = renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER)

    CAM_POS_LOCAL = (0.0, 0.0, 2.0)
    CAM_LOOKAT_LOCAL = (0.0, 0.0, 0.0)

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            # When env are not separated, their world pos is different, which affects lighting
            shadow=False,
            # rendered_envs_idx=(0, 1, 2),
            split_envs=False,
        ),
        renderer=renderer,
        show_viewer=False,
        show_FPS=False,
    )
    if IS_BATCHRENDER:
        scene.add_light(
            pos=(0.0, 0.0, 1.5),
            dir=(1.0, 1.0, -2.0),
            directional=True,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
        scene.add_light(
            pos=(4.0, -4.0, 4.0),
            dir=(-1.0, 1.0, -1.0),
            directional=False,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
    scene.add_entity(
        morph=gs.morphs.Plane(
            plane_size=(1.5, 1.5),
        ),
        surface=gs.surfaces.Aluminium(
            ior=10.0,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, -0.8, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(
                color=(1.0, 0.5, 0.5),
            ),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, -0.5, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, -0.2, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Smooth(
            color=(0.6, 0.8, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, 0.2, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Iron(
            color=(1.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, 0.5, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Gold(
            color=(1.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(-0.2, 0.8, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Glass(
            color=(1.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.1,
            pos=(0.2, -0.8, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 1.0, 1.0, 0.5),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.025,
            pos=(0.2, -0.5, 0.2),
            fixed=True,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.025,
            pos=(0.2, -0.2, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/checker.png",
            )
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    cam = scene.add_camera(
        pos=CAM_POS_LOCAL,
        lookat=CAM_LOOKAT_LOCAL,
        res=(500, 500),
        fov=60,
        spp=512,
        GUI=show_viewer,
    )
    scene.build(n_envs=3, env_spacing=(2.0, 2.0))

    # Apple Software Renderer is fully deterministic: successive captures do not match exactly
    if sys.platform == "darwin" and scene.visualizer.is_software:
        tol_env = tol_step = 1.0
    else:
        tol_env = 0.002
        tol_step = gs.EPS

    cam.start_recording(save_to_filename=(tmp_path / "video.mp4"))
    for _ in range(7):
        dofs_lower_bound, dofs_upper_bound = robot.get_dofs_limit()
        qpos = dofs_lower_bound + (dofs_upper_bound - dofs_lower_bound) * torch.as_tensor(
            np.random.rand(robot.n_qs), dtype=gs.tc_float, device=gs.device
        )

        steps_rgb_arrays = []
        for _ in range(2):
            scene.step()
            robot.set_qpos(qpos)

            if IS_BATCHRENDER:
                rgb_array_all, *_ = cam.render(
                    rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False, force_render=True
                )

            env_rgb_array = None
            for i_b in range(scene.n_envs):
                if IS_BATCHRENDER:
                    rgb_array = rgb_array_all[i_b]
                else:
                    # When env are not separated, scene cameras observe ALL batched environments. Moreover, scene
                    # cameras are operating in local frame of the env they bind to rather than world-frame, for
                    # consistency with camera sensors. As a result, the pos offset to apply for looking at each parallel
                    # env depends on the env the camera is bound to.
                    env_offset = scene.envs_offset[i_b] - scene.envs_offset[cam.env_idx]
                    cam.set_pose(pos=CAM_POS_LOCAL + env_offset, lookat=CAM_LOOKAT_LOCAL + env_offset)
                    rgb_array, *_ = cam.render(
                        rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False, force_render=True
                    )
                rgb_array = tensor_to_array(rgb_array).astype(np.float32)

                rgb_std = rgb_array.reshape((-1, 3)).std(axis=0).max()
                try:
                    assert rgb_std > 20.0
                except AssertionError:
                    if rgb_std < gs.EPS:
                        if sys.platform == "darwin" and scene.visualizer.is_software:
                            pytest.xfail(
                                "Flaky on MacOS with Apple Software Renderer. Nothing but the background was rendered."
                            )
                    raise
                if env_rgb_array is None:
                    env_rgb_array = rgb_array
                else:
                    rgb_env_diff = np.abs(env_rgb_array - rgb_array)
                    assert rgb_env_diff.mean() < tol_env, "Per-env renders do not match"

            steps_rgb_arrays.append(env_rgb_array)

        rgb_step_diff = np.abs(np.diff(steps_rgb_arrays, axis=0))
        assert rgb_step_diff.mean() < tol_step, "Per-step renders do not match"

    cam.stop_recording()


@pytest.mark.required
@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
@pytest.mark.parametrize("n_envs", [0, 4])
def test_renders_heterogeneous_entities(n_envs, show_viewer, png_snapshot, renderer_type, renderer):
    # A heterogeneous entity instantiates a different morph variant per environment (a sphere in some envs, the duck
    # mesh in others). Each variant must render only in the environments it is active in - and this must hold across
    # every supported renderer. The rasterizer composites all environments into a single image (the env_idx == -1
    # draw-all path), while the Madrona batch renderer returns one image per environment; both must apply the same
    # per-environment visibility. A homogeneous green box renders alongside to confirm ordinary entities are
    # unaffected by the masking. The captured snapshot is the ground truth.
    CAM_RES = (160, 120)
    IS_BATCHRENDER = renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER)

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            shadow=False,
            split_envs=False,
        ),
        renderer=renderer,
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(
            plane_size=(5.0, 5.0),
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.3, 0.3, 0.3),
        ),
        surface=gs.surfaces.Smooth(
            color=(0.3, 0.8, 0.3),
        ),
    )
    heterogeneous = scene.add_entity(
        morph=(
            gs.morphs.Sphere(
                radius=0.3,
            ),
            gs.morphs.Mesh(
                file="meshes/duck/duck.obj",
                scale=0.003,
                euler=(90.0, 0.0, 90.0),
            ),
        ),
    )
    camera = scene.add_camera(
        res=CAM_RES,
        pos=(0.0, 5.0, 2.6),
        lookat=(0.0, 0.0, 0.2),
        fov=55.0,
        GUI=show_viewer,
    )
    # The batch renderer has no built-in lighting, so add explicit lights; the rasterizer lights itself.
    if IS_BATCHRENDER:
        scene.add_light(
            pos=(0.0, 0.0, 1.5),
            dir=(1.0, 1.0, -2.0),
            directional=True,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
        scene.add_light(
            pos=(4.0, -4.0, 4.0),
            dir=(-1.0, 1.0, -1.0),
            directional=False,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )

    scene.build(n_envs=n_envs)

    if sys.platform == "darwin" and scene.visualizer.is_software:
        # Small discrepancies between different hardware due the different physics integration
        png_snapshot.extension._std_err_threshold = 3.0

    # The sphere variant fills the first environments and the duck the rest, placed on opposite sides so the duck
    # (yellow) occupies a known half of the image. The box (homogeneous) sits at the back of every environment.
    box_pos = np.array([[0.0, 1.4, 0.15], [0.4, 1.4, 0.15], [-0.4, 1.4, 0.15], [0.2, 1.4, 0.15]])
    het_pos = np.array([[-1.2, 0.3, 0.3], [-1.2, -0.3, 0.3], [1.2, 0.3, 0.3], [1.2, -0.3, 0.3]])
    if n_envs == 0:
        box.set_pos(box_pos[0])
        heterogeneous.set_pos(het_pos[0])
    else:
        box.set_pos(box_pos[:n_envs])
        heterogeneous.set_pos(het_pos[:n_envs])

    rgb = tensor_to_array(camera.render(rgb=True)[0])
    # The batch renderer returns one image per environment; the rasterizer composites all environments into one image.
    per_env = IS_BATCHRENDER and n_envs > 0
    if per_env:
        assert rgb.shape == (n_envs, *CAM_RES[::-1], 3)
        frames = list(rgb)
    else:
        assert rgb.shape == (*CAM_RES[::-1], 3)
        frames = [rgb]

    def duck_yellow(img):
        # Yellow-pixel mask isolating the (textured) duck variant from the plane, box and white sphere.
        return (img[..., 0].astype(int) > 120) & (img[..., 1].astype(int) > 120) & (img[..., 2].astype(int) < 110)

    # The homogeneous green box renders in every frame.
    for frame in frames:
        assert ((frame[..., 1].astype(int) - frame[..., 0].astype(int)) > 40).sum() > 0

    if per_env:
        # Distinct per-env variants and positions mean no two environments may render identically.
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                assert (frames[i] != frames[j]).any()
        # The duck variant is active in only a subset of environments, so it must not leak into the others: at least
        # one environment must show it and at least one must show none of it.
        counts = [int(duck_yellow(frame).sum()) for frame in frames]
        assert max(counts) > 3, f"duck variant never rendered; yellow per env={counts}"
        assert min(counts) <= max(counts) // 8, f"duck variant leaked into inactive envs; yellow per env={counts}"
    elif n_envs > 1:
        # Combined image: the duck variant (yellow) belongs to the environments on one side, so it must not be
        # duplicated into the sphere environments on the other side.
        half = rgb.shape[1] // 2
        yellow = duck_yellow(rgb)
        left, right = int(yellow[:, :half].sum()), int(yellow[:, half:].sum())
        assert max(left, right) > 3 * min(left, right), f"duck must render only in its env; yellow={left, right}"
    else:
        # Single environment: the duck variant is inactive there and must be masked out entirely; only the sphere shows.
        n_yellow = int(duck_yellow(rgb).sum())
        assert n_yellow <= 3, f"masked duck variant must not render; yellow={n_yellow}"

    for frame in frames:
        assert rgb_array_to_png_bytes(frame) == png_snapshot


@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
@pytest.mark.parametrize("segmentation_level", ["entity", "link", "geom"])
@pytest.mark.parametrize("particle_mode", ["visual", "particle"])
def test_segmentation_map(segmentation_level, particle_mode, renderer_type, renderer, show_viewer):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,  # Disable many physics features to speedup compilation
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,  # Implicit solver allows for larger timestep without failure on GPU backend
            n_pcg_iterations=40,  # Reduce number of iterations to speedup runtime
        ),
        coupler_options=gs.options.LegacyCouplerOptions(
            rigid_mpm=False,
            rigid_sph=False,
            rigid_pbd=False,
            rigid_fem=False,
            mpm_sph=False,
            mpm_pbd=False,
            fem_mpm=False,
            fem_sph=False,
        ),
        vis_options=gs.options.VisOptions(
            segmentation_level=segmentation_level,
        ),
        renderer=renderer,
        show_viewer=False,
        show_FPS=False,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(-1.0, -1.0, 0.5),
            euler=(0, 0, 90),
        ),
    )

    # We don't test "recon" for vis_mode because it is hard to install.
    materials = ((gs.materials.Rigid(), "visual"),)
    if renderer_type == RENDERER_TYPE.RASTERIZER:
        materials = (
            *materials,
            (gs.materials.Tool(), "visual"),
            (gs.materials.FEM.Elastic(), "visual"),
            (gs.materials.MPM.Elastic(), particle_mode),
            (gs.materials.PBD.Cloth(), particle_mode),
            (gs.materials.SPH.Liquid(), "particle" if particle_mode == "visual" else particle_mode),
            (gs.materials.Kinematic(), "visual"),
        )

    ducks = []
    for i, (material, vis_mode) in enumerate(materials):
        col_idx, row_idx = i // 3 - 1, i % 3 - 1
        ducks.append(
            scene.add_entity(
                material=material,
                morph=gs.morphs.Mesh(
                    file="meshes/duck.obj",
                    scale=0.1,
                    pos=(col_idx * 0.5, row_idx * 0.5, 0.5),
                ),
                surface=gs.surfaces.Default(
                    color=np.random.rand(3),
                    vis_mode=vis_mode,
                ),
            )
        )

    camera = scene.add_camera(
        # Using very low resolution to speed up rendering
        res=(128, 128),
        pos=(2.0, 0.0, 2.0),
        lookat=(0, 0, 0.5),
        fov=40,
        GUI=show_viewer,
    )
    scene.build()

    # Segmentation count: background(1) + URDF links/entity + duck materials. Rigid and Kinematic ducks use
    # add_rigid_node (tuple keys at link/geom level), other ducks use add_static_node (int keys). The URDF has
    # 2 visual links. FEM entities use per-visual-geom tuple keys (entity.idx, i_g) only at "geom" level; at "link"
    # and "entity" levels they fall back to a plain int entity.idx.
    n_rigid_like = sum(isinstance(m, gs.materials.Kinematic) for m, _ in materials)
    n_fem_geom = (
        sum(isinstance(m, gs.materials.FEM.Elastic) for m, _ in materials) if segmentation_level == "geom" else 0
    )
    seg_num = len(materials) + (2 if segmentation_level == "entity" else 3)
    idx_dict = scene.segmentation_idx_dict
    assert len(idx_dict) == seg_num
    comp_key = 0
    for seg_key in idx_dict.values():
        if isinstance(seg_key, tuple):
            comp_key += 1
    # At entity level no tuple keys; at link level: 2 URDF links + rigid-like ducks; at geom level additionally
    # one tuple per FEM visual geom (the FEM ducks have one each, so counting FEM entities suffices).
    assert comp_key == (0 if segmentation_level == "entity" else 2 + n_rigid_like + n_fem_geom)

    for i in range(2):
        scene.step()
        _, _, seg, _ = camera.render(rgb=False, depth=False, segmentation=True, colorize_seg=False, normal=False)
        seg = tensor_to_array(seg)
        assert_equal(np.sort(np.unique(seg.flat)), np.arange(0, seg_num))


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.RAYTRACER])
def test_multi_submesh_fem_render(renderer_type, renderer, show_viewer, png_snapshot):
    CAM_RES = (256, 256)

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
            shadow=(renderer_type != RENDERER_TYPE.RASTERIZER),
            segmentation_level="geom",
        ),
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    # Every vertex of the scene must stay inside the camera frustum: the Apple Software Renderer misrasterizes
    # out-of-frustum geometry.
    fem = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/trashbag_rope.glb",
            group_by_material=True,
        ),
        material=gs.materials.FEM.Cloth(),
    )
    camera = scene.add_camera(
        res=CAM_RES,
        pos=(0.6, 0.6, 0.55),
        lookat=(0.0, 0.0, 0.32),
        fov=45,
        GUI=show_viewer,
        denoise=False,
    )
    scene.build()

    # Move the entity after build so rendering exercises the per-sub-mesh vertex update path.
    fem.set_position([0.0, 0.0, 0.2])

    # The Apple Software Renderer that macOS CI falls back to differs from hardware on anti-aliased channels, and
    # RayTracer is non-deterministic across GPUs; relax the pixel match for the color-like outputs in both cases.
    if scene.visualizer.is_software or renderer_type == RENDERER_TYPE.RAYTRACER:
        png_snapshot.extension._std_err_threshold = 3.0
        png_snapshot.extension._blurred_kernel_size = 3

    if renderer_type == RENDERER_TYPE.RAYTRACER:
        # RayTracer only produces the RGB channel.
        rgb, *_ = camera.render(rgb=True)
        assert rgb_array_to_png_bytes(rgb) == png_snapshot
    else:
        rgb, depth, seg, normal = camera.render(rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True)
        assert rgb_array_to_png_bytes(rgb) == png_snapshot
        assert rgb_array_to_png_bytes(as_grayscale_image(tensor_to_array(depth))) == png_snapshot
        assert rgb_array_to_png_bytes(normal) == png_snapshot

        # Segmentation ids are integer labels rasterized without anti-aliasing, so coverage is fixed by the OpenGL
        # fill rules and identical on every renderer: assert the map bit-exactly. This is the channel that proves
        # each sub-mesh reaches the renderer with its own id.
        png_snapshot.extension._std_err_threshold = 0.0
        png_snapshot.extension._ratio_err_threshold = 0.0
        png_snapshot.extension._blurred_kernel_size = 1
        assert rgb_array_to_png_bytes(seg) == png_snapshot


@pytest.mark.required
@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
def test_point_cloud(renderer_type, renderer, show_viewer):
    N_ENVS = 2
    CAM_RES = (256, 256)
    CAMERA_DIST = 8.0
    OBJ_OFFSET = 10.0
    BOX_HALFSIZE = 1.0
    SPHERE_RADIUS = 1.0

    IS_BATCHRENDER = renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER)
    BATCH_SHAPE = (N_ENVS,) if N_ENVS > 0 and IS_BATCHRENDER else ()

    scene = gs.Scene(
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    if IS_BATCHRENDER:
        scene.add_light(
            pos=(0.0, 0.0, 1.5),
            dir=(1.0, 1.0, -2.0),
            directional=True,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
        scene.add_light(
            pos=(4.0, -4.0, 4.0),
            dir=(-1.0, 1.0, -1.0),
            directional=False,
            castshadow=True,
            cutoff=45.0,
            intensity=0.5,
        )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, OBJ_OFFSET, 0.0),
            radius=SPHERE_RADIUS,
            fixed=True,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -OBJ_OFFSET, 0.0),
            size=(2.0 * BOX_HALFSIZE, 2.0 * BOX_HALFSIZE, 2.0 * BOX_HALFSIZE),
            fixed=True,
        )
    )
    camera_sphere = scene.add_camera(
        res=CAM_RES,
        pos=(0.0, OBJ_OFFSET, CAMERA_DIST),
        lookat=(0.0, OBJ_OFFSET, 0.0),
        near=2.0,
        far=15.0,
        GUI=show_viewer,
    )
    camera_box_1 = scene.add_camera(
        res=CAM_RES,
        pos=(0.0, -OBJ_OFFSET, CAMERA_DIST),
        lookat=(0.0, -OBJ_OFFSET, 0.0),
        near=2.0,
        far=15.0,
        GUI=show_viewer,
    )
    camera_box_2 = scene.add_camera(
        res=CAM_RES,
        pos=np.array((CAMERA_DIST, CAMERA_DIST - OBJ_OFFSET, CAMERA_DIST)),
        lookat=(0.0, -OBJ_OFFSET, 0.0),
        near=2.0,
        far=15.0,
        GUI=show_viewer,
    )
    scene.build(n_envs=N_ENVS)

    if show_viewer:
        for camera in scene.visualizer.cameras:
            camera.render(rgb=True, depth=True)

    point_cloud, mask = camera_box_1.render_pointcloud(world_frame=False)
    assert point_cloud.shape == (*BATCH_SHAPE, *CAM_RES, 3)
    point_cloud = point_cloud[mask]
    assert_allclose(CAMERA_DIST - point_cloud[:, 2], BOX_HALFSIZE, atol=1e-4)
    assert np.all(-BOX_HALFSIZE <= point_cloud[:, :2].min(axis=0))
    assert np.all(point_cloud[:, :2].max(axis=0) <= BOX_HALFSIZE)

    point_cloud, mask = camera_box_2.render_pointcloud(world_frame=False)
    assert point_cloud.shape == (*BATCH_SHAPE, *CAM_RES, 3)
    point_cloud = point_cloud[mask]
    point_cloud = point_cloud @ gu.z_up_to_R(np.array((1.0, 1.0, 1.0)), np.array((0.0, 0.0, 1.0))).T
    point_cloud -= np.array((CAMERA_DIST, CAMERA_DIST, CAMERA_DIST))
    # FIXME: Tolerance must be increased whe using Apple's Software Rendering, probably due to an OpenGL bug...
    tol = 2e-4 if sys.platform == "darwin" else 1e-4
    assert_allclose(np.linalg.norm(point_cloud, ord=float("inf"), axis=-1), BOX_HALFSIZE, atol=tol)

    point_cloud, mask = camera_box_2.render_pointcloud(world_frame=True)
    assert point_cloud.shape == (*BATCH_SHAPE, *CAM_RES, 3)
    point_cloud = point_cloud[mask]
    point_cloud += np.array((0.0, OBJ_OFFSET, 0.0))
    assert_allclose(np.linalg.norm(point_cloud, ord=float("inf"), axis=-1), BOX_HALFSIZE, atol=tol)

    # It is not possible to get higher accuracy because of tesselation
    point_cloud, mask = camera_sphere.render_pointcloud(world_frame=False)
    assert point_cloud.shape == (*BATCH_SHAPE, *CAM_RES, 3)
    point_cloud = point_cloud[mask]
    assert_allclose(np.linalg.norm((0.0, 0.0, CAMERA_DIST) - point_cloud, axis=-1), SPHERE_RADIUS, atol=1e-2)


@pytest.mark.parametrize(
    "renderer_type",
    [RENDERER_TYPE.RASTERIZER, RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER],
)
def test_render_planes(tmp_path, png_snapshot, renderer_type, renderer):
    IS_BATCHRENDER = renderer_type in (RENDERER_TYPE.BATCHRENDER_RASTERIZER, RENDERER_TYPE.BATCHRENDER_RAYTRACER)

    for test_idx, (plane_size, tile_size) in enumerate(
        (
            ((3, 4.5), (0.5, 0.75)),
            ((3.0, 5.0), (5.0, 3.0)),
            ((4.0, 4.0), (1.0, 1.0)),
        )
    ):
        scene = gs.Scene(
            renderer=renderer,
            show_viewer=False,
            show_FPS=False,
        )
        if IS_BATCHRENDER:
            scene.add_light(
                pos=(0.0, 0.0, 1.5),
                dir=(1.0, 1.0, -2.0),
                directional=True,
                castshadow=True,
                cutoff=45.0,
                intensity=0.5,
            )
            scene.add_light(
                pos=(4.0, -4.0, 4.0),
                dir=(-1.0, 1.0, -1.0),
                directional=False,
                castshadow=True,
                cutoff=45.0,
                intensity=0.5,
            )
        scene.add_entity(
            gs.morphs.Plane(plane_size=plane_size, tile_size=tile_size),
        )
        camera = scene.add_camera(
            res=(256, 256),
            pos=(0.0, 0.0, 8),
            lookat=(0.0, 0.0, 0.0),
            fov=45,
            GUI=False,
        )
        scene.build()

        exporter = FrameImageExporter(tmp_path)
        rgba, depth, _, _ = camera.render(rgb=True, depth=False)
        exporter.export_frame_single_camera(test_idx, camera.idx, rgb=rgba, depth=depth)

    for image_file in sorted(tmp_path.rglob("*.png")):
        with open(image_file, "rb") as f:
            assert f.read() == png_snapshot


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
def test_render_oversized_resolution(renderer):
    # Verify that ``render_offscreen`` honors the user-requested viewport size even when it exceeds the available
    # display area, by requesting a viewer larger than GitHub-hosted Apple M1 macos-15 runners can actually allocate
    # and checking the returned image dimensions still match the request.
    requested_res = (1920, 1440)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=requested_res,
        ),
        renderer=renderer,
        show_viewer=True,
        show_FPS=False,
    )
    scene.build()
    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active
    rgb, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node,
        pyrender_viewer._renderer,
        rgb=True,
        depth=False,
        seg=False,
        normal=False,
    )
    assert rgb.shape[:2] == (requested_res[1], requested_res[0])


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_context_isolation(renderer_type):
    # Each offscreen scene owns a separate GL context, but the platform's current-context state is process/thread
    # global. Tearing down one scene's renderer while another is mid-render - which happens under cyclic GC, or
    # when the render thread deletes a retired renderer - must leave the rendering scene's context untouched. Force
    # that interleaving by destroying scene B in the middle of scene A's render, and assert A renders exactly as it
    # does without the interference, rather than rendering on B's destroyed context or losing its context entirely.
    # The two scenes must own independent renderers, hence a fresh Rasterizer each rather than a shared instance.
    scene_a = gs.Scene(
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
        show_FPS=False,
    )
    scene_a.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.5),
            size=(0.3, 0.3, 0.3),
        )
    )
    camera_a = scene_a.add_camera(
        res=(64, 64),
        pos=(1.5, 1.5, 1.0),
        lookat=(0.0, 0.0, 0.0),
    )
    scene_a.build()
    rgb_reference = camera_a.render(rgb=True)[0]

    # Distinct content so that rendering A on B's context would be visibly wrong rather than coincidentally equal.
    scene_b = gs.Scene(
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
        show_FPS=False,
    )
    scene_b.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 5.0),
            size=(0.3, 0.3, 0.3),
        )
    )
    camera_b = scene_b.add_camera(
        res=(64, 64),
        pos=(1.5, 1.5, 1.0),
        lookat=(0.0, 0.0, 0.0),
    )
    scene_b.build()
    camera_b.render(rgb=True)

    # Destroy scene B right after scene A's renderer makes its context current, i.e. mid-render, then check the current
    # context directly: a stranded context does not necessarily corrupt the rendered pixels on every driver, so
    # asserting on pixels alone is not enough to catch it. 'save_current_context' returns a restore callable for the
    # current context, or None when none is current, so a stranded context surfaces as None here.
    renderer_a = scene_a.visualizer._rasterizer._renderer
    original_make_current = renderer_a.make_current
    destroyed = False
    restore_after_destroy = None

    def make_current_then_destroy_b():
        nonlocal destroyed, restore_after_destroy
        original_make_current()
        if not destroyed:
            destroyed = True
            scene_b.destroy()
            restore_after_destroy = renderer_a.save_current_context()

    renderer_a.make_current = make_current_then_destroy_b
    rgb_after = camera_a.render(rgb=True)[0]

    assert restore_after_destroy is not None
    assert_pixel_match(
        rgb_after,
        rgb_reference,
        err_msg="Scene A rendered differently after scene B was destroyed mid-render (wrong or lost GL context).",
    )


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_camera_follow_entity(n_envs, renderer, show_viewer):
    CAM_RES = (100, 100)

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            segmentation_level="entity",
            rendered_envs_idx=[max(n_envs - 1, 0)],
        ),
        renderer=renderer,
        show_viewer=False,
        show_FPS=False,
    )
    for pos in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)):
        obj = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=pos,
            ),
        )
        cam = scene.add_camera(
            res=CAM_RES,
            pos=(0.0, 0.0, 0.0),
            lookat=(1.0, 0, 0.0),
            env_idx=1 if n_envs else None,
            GUI=show_viewer,
        )
        cam.follow_entity(obj, smoothing=None)
        cam.unfollow_entity()

    scene.build(n_envs=n_envs)

    for cam, obj in zip(scene.visualizer.cameras, scene.entities):
        cam.follow_entity(obj, smoothing=None)

    # First render
    seg_mask = None
    for entity_idx, cam in enumerate(scene.visualizer.cameras, 1):
        _, _, seg, _ = cam.render(rgb=False, segmentation=True)
        assert (np.unique(seg) == (0, entity_idx)).all()
        if seg_mask is None:
            seg_mask = seg != 0
        else:
            assert ((seg != 0) == seg_mask).all()

    # Second render - same
    for i, obj in enumerate(scene.entities):
        obj.set_pos((10.0, 0.0, i), envs_idx=([1] if n_envs else None))
    force_render = True
    for entity_idx, cam in enumerate(scene.visualizer.cameras, 1):
        _, _, seg, _ = cam.render(rgb=False, segmentation=True, force_render=force_render)
        assert (np.unique(seg) == (0, entity_idx)).all()
        assert ((seg != 0) == seg_mask).all()
        force_render = False

    # Third render - All objects but all different
    for i, obj in enumerate(scene.entities):
        obj.set_pos((0.1 * ((i // 2) % 2 - 1), 0.1 * (i % 2), 0.1 * i), envs_idx=([1] if n_envs else None))
    force_render = True
    seg_masks = []
    for cam in scene.visualizer.cameras:
        _, _, seg, _ = cam.render(rgb=False, segmentation=True, force_render=force_render)
        assert (np.unique(seg) == np.arange(len(scene.entities) + 1)).all()
        seg_masks.append(seg != 0)
        force_render = False
    assert np.diff(seg_masks, axis=0).any(axis=(1, 2)).all()

    # Track a trajectory over time
    for i in range(3):
        pos = 2.0 * (np.random.rand(3) - 0.5)
        quat = gu.rotvec_to_quat(np.pi * (np.random.rand(3) - 0.5))
        obj.set_pos(pos + np.array([10.0, 0.0, 0.0]), envs_idx=([1] if n_envs else None))
        obj.set_quat(quat, envs_idx=([1] if n_envs else None))
        _, _, seg, _ = cam.render(segmentation=True, force_render=True)
        assert (np.unique(seg) == (0, entity_idx)).all()
        assert not seg[tuple([*range(0, res // 3), *range(2 * res // 3, res)] for res in CAM_RES)].any()


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_camera_gimbal_lock_singularity(renderer, show_viewer):
    # Minimal scene
    scene = gs.Scene(
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cam = scene.add_camera(
        pos=(0.0, -1.5, 5.0),
        lookat=(0.0, 0.0, 0.0),
    )
    scene.build()

    prev_right = None

    # Move camera through singularity along y-axis: y=-1.5 to y=1.5 (singularity at y=0)
    for i in range(7):
        cam.set_pose(pos=(0.0, -1.5 + i * 0.5, 5.0), lookat=(0.0, 0.0, 0.0))

        # Get the right vector (x-axis) from camera transform
        transform = cam.get_transform()
        right = transform[:3, 0]

        # Check direction with the previous one
        if prev_right is not None:
            assert torch.dot(prev_right, right) > 0.0

        prev_right = right

    # Move camera through singularity along x-axis: x=-1.5 to x=1.5 (singularity at x=0)
    prev_right = None
    for i in range(7):
        cam.set_pose(pos=(-1.5 + i * 0.5, 0.0, 5.0), lookat=(0.0, 0.0, 0.0))

        # Get the right vector (x-axis) from camera transform
        transform = cam.get_transform()
        right = transform[:3, 0]

        # Check direction with the previous one
        if prev_right is not None:
            assert torch.dot(prev_right, right) > 0.0

        prev_right = right


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.parametrize("force_show_viewer", [False, True])
def test_rasterizer_env_separate(renderer, png_snapshot, show_viewer, force_show_viewer):
    if force_show_viewer and not IS_INTERACTIVE_VIEWER_AVAILABLE:
        pytest.skip(SKIP_NO_VIEWER)

    CAM_RES = (256, 256)
    RENDERED_ENVS = (1, 2)

    # One threshold for every frame. Marker lines land on slightly different pixels from one rasterizer to the next,
    # which the offscreen frames absorb through a blur; the viewer frame holds up without one.
    STD_ERR_THR = 1.5
    BLUR_OFFSCREEN = 3

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            show_link_frame=True,
            # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
            shadow=False,
            split_envs=True,
            rendered_envs_idx=RENDERED_ENVS,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=CAM_RES,
            run_in_thread=False,
            # Far enough back for the whole ground to fit in frame, as every camera below requires
            camera_pos=(3.0, 0.3, 2.0),
            camera_lookat=(0.0, 0.0, 0.4),
            enable_help_text=False,
            enable_default_keybinds=False,
        ),
        renderer=renderer,
        show_viewer=force_show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Plane(
            # Software rendering backends misrasterize geometry reaching outside the frustum, so every vertex of the
            # ground stays in frame from each camera below. Collision keeps its own infinite plane.
            plane_size=(1.4, 1.4),
        )
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        visualize_contact=True,
    )

    cam = scene.add_camera(
        res=CAM_RES,
        pos=(3.5, 0.0, 2.5),
        lookat=(0.0, 0.0, 0.5),
        fov=30,
        GUI=show_viewer,
    )
    cam_debug = scene.add_camera(
        res=CAM_RES,
        pos=(3.5, 0.0, 2.5),
        lookat=(0.0, 0.0, 0.5),
        fov=30,
        debug=True,
        GUI=show_viewer,
    )
    scene.build(n_envs=4, env_spacing=(0.3, 0.3))

    # The rendered environments hold an arm resting against the ground, whose contacts give the contact markers
    # something to draw, and an arm yawed a quarter turn and reaching sideways, so the two render as differently as the
    # arm's reach allows. The others hold resting arms of their own from a converged simulation.
    franka.set_dofs_position(
        [
            [-2.717, -1.763, -2.217, -2.566, 0.226, 2.933, 1.307, 0.017, 0.017],
            [1.728, -1.763, -2.157, -2.275, -0.327, 2.201, 1.833, 0.018, 0.018],
            [1.57, 1.0, 0.0, -0.5, 0.0, 1.5, 0.0, 0.04, 0.04],
            [-2.098, 1.763, -1.004, -2.421, -0.227, 2.850, -0.136, 0.020, 0.020],
        ]
    )
    scene.step()

    # Capture viewer screenshot when the interactive viewer is enabled
    if force_show_viewer:
        pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
        assert pyrender_viewer.is_active
        pyrender_viewer.switch_to()
        pyrender_viewer.on_draw()
        viewer_rgb = pyrender_viewer._renderer.jit.read_color_buf(*pyrender_viewer._viewport_size, rgba=False)

        png_snapshot.extension._std_err_threshold = STD_ERR_THR
        assert rgb_array_to_png_bytes(viewer_rgb) == png_snapshot

    # Render both cameras
    rgb, *_ = cam.render(rgb=True)
    rgb_debug, *_ = cam_debug.render(rgb=True)
    assert rgb is not None and rgb_debug is not None

    # With split_envs, renders are batched: (n_rendered_envs, H, W, 3)
    assert rgb.shape == (len(RENDERED_ENVS), *CAM_RES, 3)
    assert rgb_debug.shape == (len(RENDERED_ENVS), *CAM_RES, 3)

    # Batched set_pose: keep side view for env 0, switch to top-down view for env 1
    rgb_before = rgb.copy()
    cam.set_pose(
        pos=[(3.5, 0.0, 2.5), (0.0, 0.0, 4.0)],
        lookat=[(0.0, 0.0, 0.5), (0.0, 0.0, 0.0)],
        up=[(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
    )
    rgb_after, *_ = cam.render(rgb=True)
    assert rgb_after is not None
    assert rgb_after.shape == (len(RENDERED_ENVS), *CAM_RES, 3)

    # First env kept same pose — render should match; second env uses top-down view — render must differ
    assert np.abs(rgb_after[0].astype(np.float32) - rgb_before[0].astype(np.float32)).mean() < 1.0
    assert np.abs(rgb_after[1].astype(np.float32) - rgb_before[1].astype(np.float32)).mean() > 5.0

    # Restore original camera pose for snapshot comparisons
    cam.set_pose(
        pos=(3.5, 0.0, 2.5),
        lookat=(0.0, 0.0, 0.5),
    )
    rgb, *_ = cam.render(rgb=True)
    assert rgb is not None

    # Non-debug camera should NOT show markers — snapshot per env validates only robots are visible
    png_snapshot.extension._std_err_threshold = STD_ERR_THR
    png_snapshot.extension._blurred_kernel_size = BLUR_OFFSCREEN
    for rgb_i in rgb:
        assert rgb_array_to_png_bytes(rgb_i) == png_snapshot

    # Debug camera SHOULD show markers (frames, contact arrows) — snapshot per env validates markers
    for rgb_debug_i in rgb_debug:
        assert rgb_array_to_png_bytes(rgb_debug_i) == png_snapshot

    # Debug and non-debug must differ: markers add pixels that change between the two
    for i, (rgb_i, rgb_debug_i) in enumerate(zip(rgb, rgb_debug)):
        marker_diff = np.abs(rgb_i.astype(np.float32) - rgb_debug_i.astype(np.float32))
        n_diff_pixels = np.sum(marker_diff.max(axis=-1) > 0)
        assert n_diff_pixels > 100, (
            f"Debug and non-debug too similar for env {i} ({n_diff_pixels} pixels differ) — markers not visible"
        )

    # Per-env renders must differ since robots have different joint configurations
    for env_rgb in (rgb, rgb_debug):
        env_diff = np.abs(env_rgb[0].astype(np.float32) - env_rgb[1].astype(np.float32))
        assert env_diff.mean() > 5.0, "Per-env renders are too similar — env isolation may be broken"


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.parametrize("context_mode", ["sensor_only", "with_scene_camera", "with_viewer"])
def test_rasterizer_sensor_env_spacing_invariance(renderer, context_mode):
    if context_mode == "with_viewer" and not IS_INTERACTIVE_VIEWER_AVAILABLE:
        pytest.skip(SKIP_NO_VIEWER)

    CAM_RES = (128, 128)

    # The sensor renders each environment on its own whatever the setting, which only picks how the scene camera below
    # renders, so the window keeps the environments side by side
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            split_envs=context_mode == "with_scene_camera",
        ),
        viewer_options=gs.options.ViewerOptions(
            res=CAM_RES,
            run_in_thread=False,
            camera_pos=(2.0, -2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.3),
            enable_help_text=False,
        ),
        renderer=renderer,
        show_viewer=context_mode == "with_viewer",
        show_FPS=False,
    )
    # The environments are spaced by a length that is no multiple of the 1 m checker tile of the floor, so a floor
    # pattern anchored to the world rather than to the environment would show in the images. The floors stay clear of
    # each other and fit in every frame below, as a software renderer misrasterizes geometry outside the frustum.
    scene.add_entity(
        gs.morphs.Plane(
            plane_size=(1.4, 1.4),
        ),
    )
    scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.02,
            pos=(0.0, 0.0, 0.05),
            euler=(90.0, 90.0, 0.0),
            fixed=True,
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.4, 0.3, 0.1),
            size=(0.2, 0.2, 0.2),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.8, 0.2, 0.2),
        ),
    )
    scene.add_entity(
        gs.morphs.Sphere(
            pos=(-0.4, 0.2, 0.15),
            radius=0.15,
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.3, 0.9),
        ),
    )
    # The window draws the environments in one shared world, where a point light lights each of them differently, so
    # its frames are compared under the directional light alone
    lights = [{"type": "directional", "dir": (-1.0, -1.0, -2.0), "intensity": 3.0}]
    if context_mode != "with_viewer":
        lights.append({"type": "point", "pos": (1.0, -1.0, 2.0), "color": (1.0, 0.5, 0.25), "intensity": 5.0})
    sensor = scene.add_sensor(
        RasterizerCameraOptions(
            res=CAM_RES,
            pos=(2.0, -2.0, 1.5),
            lookat=(0.0, 0.0, 0.3),
            fov=40.0,
            lights=lights,
        )
    )
    if context_mode == "with_scene_camera":
        cam = scene.add_camera(
            res=CAM_RES,
            pos=(2.0, -2.0, 1.5),
            lookat=(0.0, 0.0, 0.3),
            fov=40.0,
            GUI=False,
        )
    scene.build(n_envs=2, env_spacing=(2.3, 2.3))

    # The environments hold the same state at different places of the grid, so their images must be the same down to
    # the shading and the shadows, which must show for the comparison to mean anything
    rgb = sensor.read().rgb
    assert rgb[0].float().std() > 10.0
    assert_equal(rgb[0], rgb[1])
    if context_mode == "with_scene_camera":
        rgb, *_ = cam.render(rgb=True)
        assert rgb[0].std() > 10.0
        assert_equal(rgb[0], rgb[1])
    if context_mode == "with_viewer":
        # The window lays the environments out side by side, each on its own floor, so from one viewpoint relative to
        # each environment it shows the same frame, up to where the shadow map is sampled. The viewpoint faces across
        # the row of environments, so the neighbour stays out of frame.
        viewer = scene.visualizer.viewer
        pyrender_viewer = viewer._pyrender_viewer
        frames = []
        for env_offset in scene.envs_offset:
            viewer.set_camera_pose(pos=(2.8, 0.0, 1.5) + env_offset, lookat=(0.0, 0.0, 0.3) + env_offset)
            frame, *_ = pyrender_viewer.render_offscreen(
                pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
            )
            frames.append(frame)
        assert frames[0].std() > 10.0
        assert_pixel_match(frames[0], frames[1])


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_video_recording_lifecycle(tmp_path, monkeypatch, renderer, show_viewer):
    DT = 0.01
    # One frame every other step, so that a span of N_STEPS resolves 2 frames and the paused span below skips 2
    FPS = 50.0
    N_STEPS = 4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        renderer=renderer,
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
    )
    cam = scene.add_camera(
        res=(64, 64),
        pos=(2.5, 0.0, 1.5),
        lookat=(0.0, 0.0, 0.5),
    )
    scene.build()

    # Encoding is lossy, so the frames handed to the encoder are intercepted to compare them exactly
    frames_encoded = []
    write = VideoEncoder.write

    def write_intercepted(self, frame):
        frames_encoded.append(frame.copy())
        write(self, frame)

    monkeypatch.setattr("genesis.utils.video_encoder.VideoEncoder.write", write_intercepted)

    with pytest.raises(gs.GenesisException, match="Recording not started"):
        cam.stop_recording()
    with pytest.raises(gs.GenesisException, match="Recording not started"):
        cam.pause_recording()

    # One frame every 1 / FPS of simulated time, whether or not 'render' is ever called by the user
    video_path = tmp_path / "paused.mp4"
    cam.start_recording(save_to_filename=video_path, fps=FPS)
    with pytest.raises(gs.GenesisException, match="Recording already started"):
        cam.start_recording()
    for _ in range(N_STEPS):
        scene.step()
        cam.render()

    # A paused span leaves the video untouched, and starting again carries on with the very same file
    cam.pause_recording()
    with pytest.raises(gs.GenesisException, match="cannot change"):
        cam.start_recording(fps=2.0 * FPS)
    for _ in range(N_STEPS):
        scene.step()
    cam.start_recording()
    for _ in range(N_STEPS):
        scene.step()
    cam.stop_recording()

    # Refreshing the visualization after the step is what puts the state that step produced in the video: the last
    # frame is the final state itself, and the one before it is genuinely different
    rgb_final, *_ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    assert_equal(frames_encoded[-1], rgb_final)
    assert not np.array_equal(frames_encoded[-2], rgb_final)

    # The two recorded spans each contribute one frame per cadence, and the paused one contributes none
    steps_per_frame = round(1.0 / (FPS * DT))
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == FPS
        assert sum(1 for _ in container.decode(stream)) == 2 * N_STEPS // steps_per_frame

    # Recording again writes a separate video, and the framerate may differ from the previous one
    video_path_fast = tmp_path / "fast.mp4"
    cam.start_recording(save_to_filename=video_path_fast, fps=2.0 * FPS)
    for _ in range(N_STEPS):
        scene.step()
    cam.stop_recording()

    with av.open(video_path_fast) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 2.0 * FPS
        # Twice the framerate resolves twice as many frames out of the same simulated span, capped at one per step
        assert sum(1 for _ in container.decode(stream)) == N_STEPS

    # Steps vetoed by another callback advance no simulated time, so they contribute no frame, and a recording that
    # never gets one leaves no file behind
    video_path_vetoed = tmp_path / "vetoed.mp4"
    cam.start_recording(save_to_filename=video_path_vetoed, fps=FPS)
    scene.register_pre_step_callback(lambda: True)
    for _ in range(N_STEPS):
        scene.step()
    cam.stop_recording()

    assert not video_path_vetoed.exists()


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_transparent_surfaces_show_what_lies_behind_them(renderer, show_viewer):
    ALPHA = 0.5
    RES = 64
    # Wholly inside the field of view, high enough above the pair to project clear of the patch sampled below. Geometry
    # straddling the frustum boundary is misrasterized by software rendering backends, so it stays clear of the edge.
    PARKED = (0.0, 0.0, 1.2)
    CENTRE = slice(RES // 2 - 3, RES // 2 + 3)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            # The boxes are placed by hand and their overlap is what is measured, so nothing may pull them down
            gravity=(0.0, 0.0, 0.0),
        ),
        vis_options=gs.options.VisOptions(
            shadow=False,
            # Black leaves each surface worth exactly its own contribution, so the composite below needs no offset
            background_color=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 0.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        renderer=renderer,
        show_viewer=show_viewer,
    )
    # Same size and lined up along the axis the camera sits on, so that the nearer one covers the farther one
    red = scene.add_entity(
        gs.morphs.Box(
            pos=(0.6, 0.0, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, ALPHA),
        ),
    )
    blue = scene.add_entity(
        gs.morphs.Box(
            pos=(-0.6, 0.0, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, ALPHA),
        ),
    )
    cam = scene.add_camera(
        res=(RES, RES),
        pos=(2.5, 0.0, 0.5),
        lookat=(0.0, 0.0, 0.5),
    )
    scene.build()

    def centre_patch(pos_red, pos_blue):
        red.set_pos(pos_red)
        blue.set_pos(pos_blue)
        scene.step()
        rgb, *_ = cam.render()
        return tensor_to_array(rgb)[CENTRE, CENTRE].reshape(-1, 3).mean(axis=0)

    # Where the two overlap, the near surface composites over whatever the far one left, so the pair is worth the near
    # one plus what survives of the far one. A far surface painted before the near one keeps that contribution; painted
    # afterwards it fails the depth test and vanishes, leaving the near box alone.
    patch_red = centre_patch((0.6, 0.0, 0.5), PARKED)
    patch_blue = centre_patch(PARKED, (-0.6, 0.0, 0.5))
    patch_pair = centre_patch((0.6, 0.0, 0.5), (-0.6, 0.0, 0.5))
    assert_allclose(patch_pair, patch_red + (1.0 - ALPHA) * patch_blue, atol=1.0)

    # Rendering the same scene again gives the same image, whichever order two surfaces at the same depth end up in
    rgb, *_ = cam.render()
    assert_equal(cam.render()[0], rgb)

    # Bringing the blue box in front of the red one swaps which of the two composites over the other
    patch_blue_ahead = centre_patch(PARKED, (1.2, 0.0, 0.5))
    patch_pair_swapped = centre_patch((0.6, 0.0, 0.5), (1.2, 0.0, 0.5))
    assert_allclose(patch_pair_swapped, patch_blue_ahead + (1.0 - ALPHA) * patch_red, atol=1.0)

    # Crossing the camera over to the other side reverses which one is in front just the same
    cam.set_pose(pos=(-2.5, 0.0, 0.5), lookat=(0.0, 0.0, 0.5))
    patch_red_across = centre_patch((0.6, 0.0, 0.5), PARKED)
    patch_blue_across = centre_patch(PARKED, (-0.6, 0.0, 0.5))
    patch_pair_across = centre_patch((0.6, 0.0, 0.5), (-0.6, 0.0, 0.5))
    assert_allclose(patch_pair_across, patch_blue_across + (1.0 - ALPHA) * patch_red_across, atol=1.0)
