import os
import sys

import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.render_context import Points
from genesis.engine.solvers.rigid.rigid_solver import HIBERNATION_TINT
from genesis.utils.misc import qd_to_torch

from ..conftest import IS_INTERACTIVE_VIEWER_AVAILABLE, SKIP_NO_VIEWER
from ..utils.assertions import assert_allclose, assert_equal, rgb_array_to_png_bytes
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
        GUI=show_viewer,
        debug=True,
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

    # A mesh is drawn at the scale of its transform, and placed again at the scale of the new one
    boxes_T = gu.trans_to_T(np.array(((0.0, 0.0, 0.5), (0.0, 0.0, 0.5))))
    boxes_T[:, :3, :3] *= np.array((2.0, 4.0))[:, None, None]
    box_obj = scene.draw_debug_mesh(mu.create_box(extents=(0.1, 0.1, 0.1), color=(0.0, 0.0, 1.0, 1.0)), T=boxes_T[0])
    box_widths = []
    for box_T in boxes_T:
        if box_widths:
            scene.update_debug_objects([box_obj], [box_T])
        rgb_array, *_ = cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
        box_cols = np.flatnonzero((rgb_array != rgb_array[0, 0]).any(axis=(0, 2)))
        box_widths.append(box_cols[-1] - box_cols[0])
    assert_allclose(box_widths[1] / box_widths[0], 2.0, tol=0.1)


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
            camera_pos=(2.0, 2.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        renderer=renderer,
        show_viewer=True,
    )

    # A small ground whose corners all stay in view, since a software renderer misdraws geometry that leaves the view
    scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.3, 0.0),
            plane_size=(1.0, 1.2),
        ),
    )

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
    contact_sensor = scene.add_sensor(
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
    # Behind the camera, so that moving it changes nothing in view
    kinematic_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(2.4, 2.4, 2.2),
        ),
        material=gs.materials.Kinematic(),
    )

    scene.build(n_envs=n_envs)

    for _ in range(5):
        scene.step()

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active
    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )

    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot

    # A sensor reading is a step output, so a setter moving the scene hides the debug drawing of the sensors until the
    # next step rather than showing the previous reading at the new pose. The pink sphere of the contact sensor marks
    # the contact of the ground box, and it stays hidden once a setter lifts the box, right away and after a step.
    # A setter on a solver the readings of no sensor depend on keeps the debug drawing of every sensor.
    PINK_LOWER, PINK_UPPER = (200, 60, 200), (255, 190, 255)
    assert ((PINK_LOWER <= rgb_arr) & (rgb_arr <= PINK_UPPER)).all(axis=-1).any()
    # The sphere the user hides is gone until shown again
    (contact_sphere,) = (
        marker
        for marker in scene.sim.render_context.get_markers()
        if isinstance(marker, Points) and marker.color == (1.0, 0.5, 1.0, 1.0)
    )
    pink_pixel_counts = []
    for hidden in (True, False):
        contact_sensor.render_context.set_hidden(contact_sphere, hidden)
        rgb_shown, *_ = pyrender_viewer.render_offscreen(
            pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
        )
        pink_pixel_counts.append(((PINK_LOWER <= rgb_shown) & (rgb_shown <= PINK_UPPER)).all(axis=-1).sum())
    assert pink_pixel_counts[0] < pink_pixel_counts[1]
    assert_equal(rgb_shown, rgb_arr)
    kinematic_box.set_pos((2.4, 2.4, 2.4))
    rgb_arr_kept, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )
    assert_equal(rgb_arr_kept, rgb_arr)
    ground_box.set_pos((-0.25, 0.0, 0.3))
    for _ in range(2):
        rgb_arr, *_ = pyrender_viewer.render_offscreen(
            pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
        )
        assert not ((PINK_LOWER <= rgb_arr) & (rgb_arr <= PINK_UPPER)).all(axis=-1).any()
        scene.step()


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

    assert rgb_array_to_png_bytes(rgb_arr) == png_snapshot


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_islands_and_hibernation_colors(n_envs, renderer, show_viewer):
    RES = (160, 120)
    BOX_COLOR = (0.85, 0.45, 0.2, 1.0)
    # The stacked box is the same color at half the brightness, telling the geoms of one island apart by brightness
    DARK_BOX_COLOR = (0.425, 0.225, 0.1, 1.0)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
            vis=gs.options.RigidVisOptions(
                islands=True,
                hibernation=True,
            ),
        ),
        vis_options=gs.options.VisOptions(
            # Ambient light alone shades every face of a link with its exact color, which is read back pixel by pixel
            ambient_light=(1.0, 1.0, 1.0),
            lights=[],
            split_envs=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -2.4, 1.4),
            camera_lookat=(0.3, 0.0, 0.15),
        ),
        renderer=renderer,
        show_viewer=show_viewer,
    )
    # The ground is a fixed box, which belongs to no island and stays drawn as it is. Software rendering backends
    # misrasterize geometry reaching outside the frustum, so it is sized to stay in frame.
    scene.add_entity(
        gs.morphs.Box(
            size=(2.0, 1.0, 0.02),
            pos=(0.3, 0.0, -0.01),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.6, 0.85, 0.55, 1.0),
        ),
    )
    # Two boxes stacked on the ground, one island through their contact, and a third one resting on its own
    boxes = [
        scene.add_entity(
            gs.morphs.Box(
                size=(0.2, 0.2, 0.2),
                pos=pos,
            ),
            surface=gs.surfaces.Default(
                color=color,
            ),
        )
        for pos, color in (
            ((0.0, 0.0, 0.1), BOX_COLOR),
            ((0.0, 0.0, 0.3), DARK_BOX_COLOR),
            ((0.7, 0.0, 0.1), BOX_COLOR),
        )
    ]
    # The debug camera draws the colors, and the other one the entities as they look
    cam_debug, cam_plain = (
        scene.add_camera(
            res=RES,
            pos=(0.3, -2.4, 1.4),
            lookat=(0.3, 0.0, 0.15),
            fov=45,
            debug=debug,
        )
        for debug in (True, False)
    )
    scene.build(n_envs=n_envs)
    render_context = scene.rigid_solver.render_context
    # In the second environment the third box tops the stack, so every box shares one island there
    if n_envs > 0:
        boxes[2].set_pos((0.0, 0.0, 0.5), envs_idx=1)
    links_idxc = {key[1]: idxc for idxc, key in scene.segmentation_idx_dict.items() if idxc > 0}

    def get_links_color(cam):
        rgb, _, seg, _ = cam.render(rgb=True, segmentation=True, force_render=True)
        rgb = rgb.reshape((max(n_envs, 1), RES[1], RES[0], 3))
        seg = seg.reshape((max(n_envs, 1), RES[1], RES[0]))
        # The median over the pixels of a link leaves its antialiased edges out
        return np.array(
            [
                [np.median(rgb_env[seg_env == links_idxc[link.idx]], axis=0) for link in scene.rigid_solver.links]
                for rgb_env, seg_env in zip(rgb, seg)
            ]
        )

    # Contacts climb the stack one box per step, one step later in single precision
    for _ in range(5):
        scene.step()

    # Awake, the links of one island share one hue, each at the brightness of its own material, and the islands of one
    # environment have distinct hues. The cameras showing the scene as it looks draw the materials.
    colors_awake = get_links_color(cam_debug)
    colors_plain = get_links_color(cam_plain)
    assert_allclose(colors_plain[:, (1, 3)], 255.0 * np.array(BOX_COLOR[:3]), atol=4.0)
    assert_allclose(colors_plain[:, 2].sum(axis=-1) / colors_plain[:, 1].sum(axis=-1), 0.5, atol=0.05)
    # Pixels encode the shaded color through a gamma that is no exact inverse of the sRGB decode of the material, so a
    # brightness scaling shifts their chromaticity by a few percent
    chroma_awake = colors_awake / colors_awake.sum(axis=-1, keepdims=True)
    assert_allclose(chroma_awake[:, 1], chroma_awake[:, 2], atol=0.04)
    assert_allclose(colors_awake[:, 2].sum(axis=-1) / colors_awake[:, 1].sum(axis=-1), 0.5, atol=0.05)
    assert np.abs(chroma_awake[0, 3] - chroma_awake[0, 1]).max() > 0.2
    if n_envs > 0:
        assert_allclose(colors_awake[1, 3], colors_awake[1, 1], atol=1.0)

    # Asleep, the links keep the hue of their island, faded toward grey
    for _ in range(35):
        scene.step()
    assert qd_to_torch(scene.rigid_solver.dyn_state.links.is_hibernated)[1:].all()
    colors_asleep = get_links_color(cam_debug)
    assert_allclose(get_links_color(cam_plain), colors_plain, atol=1.0)

    # Without the islands, a sleeping link is its material faded toward the same grey, and without hibernation either
    # the debug views draw the links as they look, the ground throughout
    render_context.set_islands(False)
    colors_asleep_plain = get_links_color(cam_debug)
    render_context.set_hibernation(False)
    assert_allclose(get_links_color(cam_debug), colors_plain, atol=1.0)
    for colors in (colors_awake, colors_asleep, colors_asleep_plain):
        assert_equal(colors[:, 0], colors_plain[:, 0])
    # The grey a sleeping link fades toward is read off the fade of its material, then the fade of its island color
    # must land on the same grey
    weight = HIBERNATION_TINT[3]
    grey = (colors_asleep_plain - (1.0 - weight) * colors_plain) / weight
    assert (grey[:, 1:].std(axis=-1) < 6.0).all()
    assert_allclose(colors_asleep[:, 1:], (1.0 - weight) * colors_awake[:, 1:] + weight * grey[:, 1:], atol=6.0)

    # The islands alone color the sleeping links as when they were awake
    render_context.set_islands(True)
    assert_allclose(get_links_color(cam_debug), colors_awake, atol=1.0)


@pytest.mark.required
def test_islands_colors_single_tree(renderer, show_viewer):
    RES = (160, 120)
    BOX_COLOR = (0.85, 0.45, 0.2, 1.0)

    # A single box is the only kinematic tree of the scene, which then forms a single island
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
            vis=gs.options.RigidVisOptions(
                islands=True,
                hibernation=True,
            ),
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(1.0, 1.0, 1.0),
            lights=[],
            split_envs=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.4, 1.4),
            camera_lookat=(0.0, 0.0, 0.15),
        ),
        renderer=renderer,
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=BOX_COLOR,
        ),
    )
    cam = scene.add_camera(
        res=RES,
        pos=(0.0, -2.4, 1.4),
        lookat=(0.0, 0.0, 0.15),
        fov=45,
        debug=True,
    )
    scene.build(n_envs=2)
    box_idxc = next(idxc for idxc, key in scene.segmentation_idx_dict.items() if idxc > 0 and key[1] == 1)

    def get_box_chroma():
        rgb, _, seg, _ = cam.render(rgb=True, segmentation=True, force_render=True)
        rgb = rgb.reshape((2, RES[1], RES[0], 3))
        seg = seg.reshape((2, RES[1], RES[0]))
        colors = np.array([np.median(rgb_env[seg_env == box_idxc], axis=0) for rgb_env, seg_env in zip(rgb, seg)])
        return colors / colors.sum(axis=-1, keepdims=True)

    # The box takes the hue of its island, then fades toward grey once asleep
    scene.step()
    chroma_awake = get_box_chroma()
    assert (np.abs(chroma_awake - np.array(BOX_COLOR[:3]) / sum(BOX_COLOR[:3])).max(axis=-1) > 0.2).all()
    for _ in range(50):
        scene.step()
        if qd_to_torch(scene.rigid_solver.dyn_state.links.is_hibernated)[1:].all():
            break
    assert qd_to_torch(scene.rigid_solver.dyn_state.links.is_hibernated)[1:].all()
    chroma_asleep = get_box_chroma()
    assert (np.abs(chroma_asleep - 1.0 / 3.0).max(axis=-1) < np.abs(chroma_awake - 1.0 / 3.0).max(axis=-1)).all()
