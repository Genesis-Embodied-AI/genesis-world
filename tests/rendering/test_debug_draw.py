import os
import sys

import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu

from ..conftest import IS_INTERACTIVE_VIEWER_AVAILABLE, SKIP_NO_VIEWER
from ..utils.assertions import assert_allclose, rgb_array_to_png_bytes
from .conftest import RENDERER_TYPE


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_draw_debug(renderer, show_viewer):
    if "GS_DISABLE_OFFSCREEN_MARKERS" in os.environ:
        pytest.skip("Offscreen rendering of markers is forcibly disabled. Skipping...")

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0, 2],
        ),
        renderer=renderer,
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cam = scene.add_camera(
        pos=(3.5, 0.5, 2.5),
        lookat=(0.0, 0.0, 0.5),
        up=(0.0, 0.0, 1.0),
        res=(640, 640),
        env_idx=2,
        debug=True,
        GUI=show_viewer,
    )
    scene.build(n_envs=3)

    rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
    assert_allclose(np.std(rgb_array.reshape((-1, 3)), axis=0), 0.0, tol=gs.EPS)

    arrow_obj = scene.draw_debug_arrow(
        pos=(0, 0.4, 0.1),
        vec=(0, 0.3, 0.8),
        color=(1, 0, 0),
    )
    scene.draw_debug_line(
        start=(0.7, -0.3, 0.7),
        end=(0.6, 0.2, 0.7),
        radius=0.01,
        color=(1, 0, 0, 1),
    )
    sphere_obj = scene.draw_debug_sphere(
        pos=(-0.3, 0.3, 0.0),
        radius=0.15,
        color=(0, 1, 0),
    )
    frame_obj = scene.draw_debug_frame(
        T=np.array(
            [
                [1.0, 0.0, 0.0, -0.3],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, -0.2],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        axis_length=0.5,
        origin_size=0.03,
        axis_radius=0.02,
    )
    scene.visualizer.update()

    rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
    rgb_array_flat = rgb_array.reshape((-1, 3)).astype(np.int32)
    assert (np.std(rgb_array_flat, axis=0) > 10.0).any()
    rgb_array_prev = rgb_array_flat

    # An arrow drawn for every environment at once holds one shared instance while a frame and a sphere hold one
    # per environment, so placing all three again is what says an object keeps the instances it was drawn with.
    poses = gu.trans_to_T(np.zeros((2, 2, 3)))
    for i in range(2):
        poses[:, i] = gu.trans_quat_to_T(2.0 * (np.random.rand(2, 3) - 0.5), np.random.rand(2, 4))
        arrow_pose = gu.trans_quat_to_T(2.0 * (np.random.rand(3) - 0.5), np.random.rand(4))
        scene.update_debug_objects([frame_obj, sphere_obj, arrow_obj], [*poses, arrow_pose])
        scene.visualizer.update()
        rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
        rgb_array_flat = rgb_array.reshape((-1, 3)).astype(np.int32)
        assert (np.std(rgb_array_flat - rgb_array_prev, axis=0) > 10.0).any()
        rgb_array_prev = rgb_array_flat

    scene.clear_debug_objects()
    scene.visualizer.update()
    rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
    assert_allclose(np.std(rgb_array.reshape((-1, 3)), axis=0), 0.0, tol=gs.EPS)


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
def test_sensors_draw_debug(n_envs, renderer_type, renderer, png_snapshot):
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
            shadow=(renderer_type != RENDERER_TYPE.RASTERIZER),
        ),
        viewer_options=gs.options.ViewerOptions(
            # Force screen-independent low-quality resolution when running unit tests for consistency
            res=(320, 320),
            # Enable running in background thread if supported by the platform
            run_in_thread=(sys.platform == "linux"),
            camera_pos=(1.2, 1.2, 1.2),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        renderer=renderer,
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    floating_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
        )
    )
    scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=floating_box.idx,
            pos_offset=(0.0, 0.0, 0.1),
            draw_debug=True,
        )
    )

    ground_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.2, 0.1),
            pos=(-0.25, 0.0, 0.05),
        ),
        material=gs.materials.Rigid(
            rho=200.0,
        ),
    )
    scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=ground_box.idx,
            draw_debug=True,
            debug_sphere_radius=0.08,
            debug_color=(1.0, 0.5, 1.0, 1.0),
        )
    )
    scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=ground_box.idx,
            draw_debug=True,
            debug_scale=0.01,
        )
    )
    scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(
                resolution=0.2,
                size=(0.4, 0.4),
                direction=(0.0, 0.0, -1.0),
            ),
            entity_idx=floating_box.idx,
            pos_offset=(0.2, 0.0, -0.1),
            return_world_frame=True,
            draw_debug=True,
        )
    )
    scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.SphericalPattern(
                n_points=(6, 6),
                fov=(60.0, (-120.0, -60.0)),
            ),
            entity_idx=floating_box.idx,
            pos_offset=(0.0, 0.5, 0.0),
            return_world_frame=False,
            draw_debug=True,
            debug_sphere_radius=0.01,
            debug_ray_start_color=(1.0, 1.0, 0.0, 1.0),
            debug_ray_hit_color=(0.5, 1.0, 1.0, 1.0),
        )
    )

    scene.build(n_envs=n_envs)

    for _ in range(5):
        scene.step()

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active
    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )

    if sys.platform == "darwin":
        glinfo = pyrender_viewer.context.get_info()
        renderer = glinfo.get_renderer()
        if renderer == "Apple Software Renderer":
            pytest.xfail("Tile ground colors are altered on Apple Software Renderer.")

    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
def test_draw_debug_frustum_and_trajectory(n_envs, renderer_type, renderer, png_snapshot):
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
            shadow=(renderer_type != RENDERER_TYPE.RASTERIZER),
        ),
        viewer_options=gs.options.ViewerOptions(
            # Force screen-independent low-quality resolution when running unit tests for consistency
            res=(480, 320),
            # Enable running in background thread if supported by the platform
            run_in_thread=(sys.platform == "linux"),
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        renderer=renderer,
        show_viewer=True,
    )

    # Add a box inside the sensor camera frustum
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.4, 0.0, 0.7),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),
        ),
    )

    sensor_cam = scene.add_camera(
        res=(640, 480),
        pos=(0.0, 0.0, 0.9),
        lookat=(0.8, 0.0, 0.5),
        up=(0.0, 1.0, 0.0),
        fov=30,
        near=0.1,
        far=0.7,
        GUI=False,
    )

    scene.build(n_envs=n_envs)

    scene.draw_debug_frustum(sensor_cam, color=(0.0, 1.0, 0.0, 0.3))

    t = np.linspace(0, 2 * np.pi, 50)
    positions = np.column_stack([0.8 * np.cos(t), 0.8 * np.sin(t), np.full_like(t, 0.5)])
    scene.draw_debug_trajectory(positions, radius=0.02, color=(1.0, 0.5, 0.0, 1.0))

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    scene.visualizer.viewer.update(auto_refresh=True, force=True)
    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )

    # Apple Software Rendering has issues rendering sharp edges
    if sys.platform == "darwin" and scene.visualizer.is_software:
        png_snapshot.extension._blurred_kernel_size = 3
    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot
