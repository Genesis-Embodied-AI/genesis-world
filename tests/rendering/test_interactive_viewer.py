import re
import sys
import threading
import time

import numpy as np
import OpenGL.error
import pytest

import genesis as gs
from genesis.options.sensors import RasterizerCameraOptions
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind, KeyMod, MouseButton
from genesis.vis.viewer_plugins.plugins.mouse_interaction import GRAB_ACC

from ..conftest import IS_INTERACTIVE_VIEWER_AVAILABLE, SKIP_NO_IMGUI_BUNDLE, SKIP_NO_VIEWER, is_imgui_bundle_supported
from ..utils.assertions import assert_allclose
from .conftest import RENDERER_TYPE

CAM_RES = (480, 320)


# Note that software emulation is so slow that it may takes minutes to render a single frame...
def wait_for_viewer_events(viewer, condition_fn, timeout=300.0, sleep_interval=0.1):
    """Utility function to wait for viewer events to be processed in a threaded viewer."""
    if not viewer.run_in_thread:
        viewer.dispatch_pending_events()
        viewer.dispatch_events()

    for _ in range(int(timeout / sleep_interval)):
        if condition_fn():
            return
        time.sleep(sleep_interval)
    else:
        raise AssertionError("Keyboard event not processed before timeout")


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
def test_disable_defaults():
    # Test with keyboard shortcuts DISABLED
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            run_in_thread=(sys.platform == "linux"),
            enable_help_text=False,
            enable_default_keybinds=False,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )
    scene.build()
    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    # Verify the flag is set correctly
    assert pyrender_viewer._enable_help_text is False
    # Verify that no keybindings are registered
    assert len(pyrender_viewer._keybindings) == 0


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
@pytest.mark.parametrize("n_envs", [0, 2])
def test_default_plugin(n_envs):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=CAM_RES,
            run_in_thread=(sys.platform == "linux"),
            camera_pos=(2.0, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            enable_help_text=True,
            enable_default_keybinds=True,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.2),
            size=(0.2, 0.2, 0.2),
            euler=(30, 40, 0),
        )
    )
    scene.build(n_envs=n_envs)

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    assert len(pyrender_viewer._keybindings) > 0, "Expected default keybindings to be registered."

    # Add a custom keybind
    flags = [False, False, False]

    def toggle_flag(idx):
        flags[idx] = not flags[idx]

    scene.viewer.register_keybinds(
        Keybind(
            name="toggle_flag_0",
            key=Key._0,
            key_action=KeyAction.PRESS,
            callback=lambda: toggle_flag(0),
        ),
        Keybind(
            name="toggle_flag_1",
            key=Key._1,
            key_action=KeyAction.PRESS,
            key_mods=(KeyMod.SHIFT, KeyMod.CTRL),
            callback=toggle_flag,
            args=(1,),
        ),
    )

    # Press key to toggle flag on
    pyrender_viewer.dispatch_event("on_key_press", Key._0, 0)
    # Press key with modifiers to toggle flag off
    pyrender_viewer.dispatch_event("on_key_press", Key._1, KeyMod.SHIFT | KeyMod.CTRL)
    # Press key toggle world frame
    pyrender_viewer.dispatch_event("on_key_release", Key.W, 0)

    wait_for_viewer_events(pyrender_viewer, lambda: flags[0] and flags[1])

    assert flags[0], "Expected custom keybind callback to toggle flag on."
    assert flags[1], "Expected custom keybind with key modifiers to toggle flag on."
    assert pyrender_viewer.gs_context.world_frame_shown, "Expected world frame to be shown after pressing 'W' key."

    # Remove the keybind and press key to verify it no longer works
    scene.viewer.remove_keybind("toggle_flag_0")
    pyrender_viewer.dispatch_event("on_key_press", Key._0, 0)
    # Remap the keybind and check it works
    scene.viewer.remap_keybind("toggle_flag_1", new_key=Key._2, new_key_mods=None)
    pyrender_viewer.dispatch_event("on_key_press", Key._2, 0)

    wait_for_viewer_events(pyrender_viewer, lambda: not flags[1])

    assert flags[0], "Keybind was not removed properly."
    assert not flags[1], "Expected rebinded keybind to toggle flag off."

    # Error when remapping non-existent keybind
    with pytest.raises(ValueError):
        scene.viewer.remap_keybind("non_existent_keybind", new_key=Key._3, new_key_mods=None)

    # Error when adding a keybind with same key
    with pytest.raises(ValueError):
        scene.viewer.register_keybinds(
            Keybind(name="conflicting_keybind", key=Key._2, key_action=KeyAction.PRESS, callback=lambda: None),
            overwrite=False,
        )

    # Force overwrite
    scene.viewer.register_keybinds(
        Keybind(name="conflicting_keybind", key=Key._2, key_action=KeyAction.PRESS, callback=lambda: None),
        overwrite=True,
    )

    # allow_overload=False: conflicts with any same-key binding; overwrite=True clears all siblings
    scene.viewer.register_keybinds(
        Keybind(name="key3_press", key=Key._3, key_action=KeyAction.PRESS, callback=lambda: None),
        Keybind(name="key3_release", key=Key._3, key_action=KeyAction.RELEASE, callback=lambda: None),
    )
    with pytest.raises(ValueError):
        scene.viewer.register_keybinds(
            Keybind(
                name="key3_exclusive",
                key=Key._3,
                key_action=KeyAction.HOLD,
                allow_overload=False,
                callback=lambda: None,
            ),
            overwrite=False,
        )
    scene.viewer.register_keybinds(
        Keybind(
            name="key3_exclusive",
            key=Key._3,
            key_action=KeyAction.PRESS,
            allow_overload=False,
            callback=lambda: None,
        ),
        overwrite=True,
    )
    assert pyrender_viewer._keybindings.get_by_name("key3_press") is None
    assert pyrender_viewer._keybindings.get_by_name("key3_release") is None


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
@pytest.mark.skipif(not is_imgui_bundle_supported, reason=SKIP_NO_IMGUI_BUNDLE)
def test_key_press(renderer_type, tmp_path, monkeypatch, renderer, png_snapshot):
    IMAGE_FILENAME = tmp_path / "screenshot.png"

    # Mock 'get_save_filename' to avoid poping up an interactive dialog
    def get_save_filename(self, file_exts):
        return IMAGE_FILENAME

    monkeypatch.setattr("genesis.ext.pyrender.viewer.Viewer._get_save_filename", get_save_filename)

    # Mock 'on_key_release' to determine whether requests have been processed
    is_done = False
    on_key_release_orig = gs.ext.pyrender.viewer.Viewer.on_key_release

    def on_key_release(self, symbol: int, modifiers: int):
        nonlocal is_done
        assert not is_done
        ret = on_key_release_orig(self, symbol, modifiers)
        is_done = True
        return ret

    monkeypatch.setattr("genesis.ext.pyrender.viewer.Viewer.on_key_release", on_key_release)

    # Create a scene
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            # Disable shadows systematically for Rasterizer because they are forcibly disabled on CPU backend anyway
            shadow=(renderer_type != RENDERER_TYPE.RASTERIZER),
        ),
        viewer_options=gs.options.ViewerOptions(
            # Force screen-independent low-quality resolution when running unit tests for consistency.
            # Still, it must be large enough since rendering text involved alpha blending, which is platform-dependent.
            res=(640, 480),
            # Enable running in background thread if supported by the platform.
            # Note that windows is not supported because it would trigger the following exception if some previous tests
            # was only using rasterizer without interactive viewer:
            # 'EventLoop.run() must be called from the same thread that imports pyglet.app'.
            run_in_thread=(sys.platform == "linux"),
        ),
        renderer=renderer,
        show_viewer=True,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.5, 0.5),
            pos=(0.0, 0.0, 0.0),
        ),
    )
    scene.build()
    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    # Try saving the current frame
    pyrender_viewer.dispatch_event("on_key_release", Key.S, 0)

    # Waiting for request completion
    if pyrender_viewer.run_in_thread:
        for i in range(100):
            if is_done:
                is_done = False
                break
            time.sleep(0.1)
        else:
            raise AssertionError("Keyboard event not processed before timeout")
    else:
        pyrender_viewer.dispatch_pending_events()
        pyrender_viewer.dispatch_events()

    # Skip the rest of the test if necessary.
    # Similarly, 'glBlitFramebuffer(..., GL_DEPTH_BUFFER_BIT, GL_NEAREST)' involved in offscreen rendering of depth map
    # with interactive viewer enabled takes ages on old CPU-based Mesa rendering driver (~15000s).
    if sys.platform == "linux":
        glinfo = pyrender_viewer.context.get_info()
        renderer = glinfo.get_renderer()
        if "llvmpipe" in renderer:
            llvm_version = re.search(r"LLVM\s+([\d.]+)", renderer).group(1)
            if llvm_version < "20":
                pytest.xfail("Text is blurry on Linux using old CPU-based Mesa rendering driver.")

    # Make sure that the result is valid
    with open(IMAGE_FILENAME, "rb") as f:
        assert f.read() == png_snapshot


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
@pytest.mark.parametrize(
    "n_envs, env_spacing, n_envs_per_row, target_env_idx, target_offset, use_visual_geom",
    [
        (0, (0.0, 0.0), None, None, (0.0, 0.0, 0.0), False),
        # Two envs spaced along x so envs_offset is non-zero. Camera is positioned over env 1, so a viewport-center
        # click must pick env 1 (exercising kernel_cast_ray's per-env offset transform) and leave env 0 untouched.
        (2, (0.5, 0.0), 1, 1, (0.25, 0.0, 0.0), False),
        (2, (0.5, 0.0), 1, 1, (0.25, 0.0, 0.0), True),
    ],
)
def test_mouse_interaction_plugin(n_envs, env_spacing, n_envs_per_row, target_env_idx, target_offset, use_visual_geom):
    DT = 0.01
    MASS = 100.0
    BOX_LENGTH = 0.2
    STEPS = 20
    DRAG_DY = 8
    SPRING_SLACK = 0.02
    CAM_FOV = 30
    # Probe lanes parked along y, clear of the drag: envs are spaced along x only, so a lane is one env's alone.
    DECOY_SIZE = 0.4
    DECOY_Z = 0.5
    TARGET_SIZE = 0.2
    TARGET_Z = 1.5
    STACK_LANE_Y = 1.5
    KINEMATIC_LANE_Y = 2.0
    OPTOUT_LANE_Y = 2.5
    EDGE_LANE_Y = 4.0
    # Kept within half the env spacing, so that the probes below reach this env's copy alone.
    EDGE_RADIUS = 0.15
    EDGE_PROBE_RANGE = 0.05
    PROBE_Z = 3.0
    CONCURRENT_PROBES = 200
    RAY_T = 0.35
    target_offset = np.asarray(target_offset, dtype=gs.np_float)
    CAM_POS = (target_offset[0], 0.6, 1.2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            # Forces odd resolution so that mouse clicks are centered on pixels
            res=(2 * (CAM_RES[0] // 2) + 1, 2 * (CAM_RES[0] // 2) + 1),
            run_in_thread=(sys.platform == "linux"),
            camera_pos=CAM_POS,
            # looking to the top of the box at the target env
            camera_lookat=(target_offset[0], target_offset[1], target_offset[2] + BOX_LENGTH),
            camera_fov=CAM_FOV,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, BOX_LENGTH / 2),
            size=(BOX_LENGTH, BOX_LENGTH, BOX_LENGTH),
        ),
        material=gs.materials.Rigid(
            use_visual_raycasting=True,
            rho=MASS / (BOX_LENGTH**3),
        ),
    )
    # One lane stacks a collision-only box under a visual-only one, so the two cast modes land on different entities.
    decoy = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, STACK_LANE_Y, DECOY_Z),
            size=(DECOY_SIZE,) * 3,
            fixed=True,
            visualization=False,
        ),
    )
    target = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, STACK_LANE_Y, TARGET_Z),
            size=(TARGET_SIZE,) * 3,
            fixed=True,
            collision=False,
        ),
        material=gs.materials.Rigid(
            use_visual_raycasting=True,
        ),
    )
    # A kinematic entity carries visual meshes only, in a separate solver from the rigid entities above.
    kinematic = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, KINEMATIC_LANE_Y, BOX_LENGTH / 2),
            size=(BOX_LENGTH,) * 3,
        ),
        material=gs.materials.Kinematic(
            use_visual_raycasting=True,
        ),
    )
    # Without the opt-in material, an entity stays unpickable.
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, OPTOUT_LANE_Y, TARGET_Z),
            size=(TARGET_SIZE,) * 3,
            fixed=True,
            collision=False,
        ),
    )
    # Rotated so that none of its triangle edges lines up with a coordinate plane, which is what puts a probe aimed at
    # an edge within rounding of it. Parked far enough along y for the probes to reach it without crossing anything,
    # and high enough that the ones coming from below still start above the ground.
    edge_target = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, EDGE_LANE_Y, EDGE_RADIUS + 2.0 * EDGE_PROBE_RANGE),
            quat=(0.5, 0.5, 0.5, 0.5),
            radius=EDGE_RADIUS,
            fixed=True,
        ),
    )
    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            use_force=True,
            spring_slack=SPRING_SLACK,
            use_visual_geom=use_visual_geom,
        )
    )
    scene.build(
        n_envs=n_envs,
        env_spacing=env_spacing,
        n_envs_per_row=n_envs_per_row,
    )

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    # Sanity-check the scene actually applies the offset we want to exercise.
    if target_env_idx is not None:
        assert_allclose(scene.envs_offset[target_env_idx], target_offset, tol=gs.EPS)

    class EventCounterHandler:
        def __init__(self):
            self.count = 0

        def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
            self.count += 1

        def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
            self.count += 1

        def on_mouse_release(self, x: int, y: int, buttons: int, modifiers: int):
            self.count += 1

    event_counter = EventCounterHandler()
    expected_count = 0

    def check_event_count():
        nonlocal expected_count
        expected_count += 1
        return lambda: event_counter.count == expected_count

    pyrender_viewer.push_handlers(event_counter)

    scene.step()

    assert_allclose(box.get_vel(), 0, tol=gs.EPS)

    initial_pos = tensor_to_array(box.get_pos())
    # Position of the env the cursor will pick (target_env_idx for multi-env, the only env otherwise).
    initial_pos_target = initial_pos if target_env_idx is None else initial_pos[target_env_idx]

    viewport_size = pyrender_viewer._viewport_size
    x, y = viewport_size[0] // 2, viewport_size[1] // 2
    plugin = next(p for p in scene.viewer.plugins if isinstance(p, gs.vis.viewer_plugins.MouseInteractionPlugin))

    # Press mouse to grab the box
    pyrender_viewer.dispatch_event("on_mouse_press", x, y, MouseButton.LEFT, 0)
    # Ensure event is processed
    wait_for_viewer_events(pyrender_viewer, check_event_count())

    # Confirm the raycaster picked the expected env
    if target_env_idx is not None:
        assert plugin._interact_env_idx == target_env_idx, (
            f"Expected mouse to pick env {target_env_idx}, got env {plugin._interact_env_idx}"
        )

    rgb_arrs = []
    for i in range(STEPS):
        y += DRAG_DY
        pyrender_viewer.dispatch_event("on_mouse_drag", x, y, 0, DRAG_DY, MouseButton.LEFT, 0)
        wait_for_viewer_events(pyrender_viewer, check_event_count())
        scene.step()
        if (i + 1) % (STEPS // 2) == 0:
            rgb_arr, *_ = pyrender_viewer.render_offscreen(
                pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
            )
            rgb_arrs.append(rgb_arr)

    assert not np.array_equal(rgb_arrs[0], rgb_arrs[1]), "Expected images to be different after dragging the object."

    final_pos = tensor_to_array(box.get_pos())
    final_vel = tensor_to_array(box.get_vel())
    final_pos_target = final_pos if target_env_idx is None else final_pos[target_env_idx]
    final_vel_target = final_vel if target_env_idx is None else final_vel[target_env_idx]

    assert_allclose(
        final_vel_target[:2],
        0.0,
        tol=0.002,
        err_msg="Final x and y velocities should be near zero since dragging only in z direction.",
    )

    distance_to_box = np.linalg.norm(initial_pos_target - CAM_POS)
    pixels_to_world = 2.0 * distance_to_box * np.tan(np.radians(CAM_FOV) / 2.0) / viewport_size[1]
    total_world_displacement = STEPS * DRAG_DY * pixels_to_world

    displacement_z = final_pos_target[2] - initial_pos_target[2]
    assert displacement_z > gs.EPS, "Box should have moved upward"
    assert displacement_z < total_world_displacement, (
        "Box displacement should be less than mouse displacement from spring lag"
    )

    # Non-target envs (multi-env case) must not have moved.
    if target_env_idx is not None:
        for i in range(n_envs):
            if i == target_env_idx:
                continue
            assert_allclose(
                final_pos[i],
                initial_pos[i],
                tol=1e-3,
                err_msg=f"Env {i} box must not move when picking env {target_env_idx}.",
            )

    pyrender_viewer.dispatch_event("on_mouse_release", x, y, MouseButton.LEFT, 0)
    scene.step()
    wait_for_viewer_events(pyrender_viewer, check_event_count())
    rgb_arr, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )
    assert not np.array_equal(rgb_arrs[-1], rgb_arr), "Expected visualization to change after releasing the object."

    # The forces from mouse spring are approximate, so use a large tolerance.
    # FIXME: Use a more accurate model to predict final velocity.
    total_sim_time = STEPS * DT
    avg_mouse_velocity = total_world_displacement / total_sim_time
    num_tau = total_sim_time * np.sqrt(GRAB_ACC / SPRING_SLACK)
    velocity_fraction = 1.0 - (1.0 + num_tau) * np.exp(-num_tau)
    expected_vel_z = avg_mouse_velocity * velocity_fraction

    assert_allclose(
        final_vel_target[2],
        expected_vel_z,
        rtol=0.5,
        err_msg="Final z velocity does not match expected value based on spring dynamics.",
    )

    probe_dir = (0.0, 0.0, -1.0)
    stack_probe = (target_offset[0], STACK_LANE_Y, PROBE_Z)
    stack_entity = target if use_visual_geom else decoy
    stack_distance = PROBE_Z - ((TARGET_Z + 0.5 * TARGET_SIZE) if use_visual_geom else (DECOY_Z + 0.5 * DECOY_SIZE))
    stack_position = (*stack_probe[:2], PROBE_Z - stack_distance)
    stack_hit = plugin._raycaster.cast(stack_probe, probe_dir)
    assert stack_hit.geom.entity is stack_entity
    assert_allclose(stack_hit.distance, stack_distance, tol=1e-6)
    assert_allclose(stack_hit.position, stack_position, tol=1e-6)

    # A hit stays the answer to its own ray once a later cast has run, both reading the same buffers.
    plugin._raycaster.cast((target_offset[0], OPTOUT_LANE_Y, PROBE_Z), probe_dir)
    assert_allclose(stack_hit.position, stack_position, tol=1e-6)
    assert stack_hit.env_idx == target_env_idx

    # The plugin casts a hover ray of its own on every frame the viewer draws, which overlaps a cast issued from here
    # wherever the viewer runs in a thread. Casting the opt-out lane alongside reproduces that overlap on every
    # platform: it answers with another entity in one cast mode and with nothing in the other, both of which the probes
    # below report should two casts share the hit they read back.
    is_hovering = True

    def hover():
        while is_hovering:
            plugin._raycaster.cast((target_offset[0], OPTOUT_LANE_Y, PROBE_Z), probe_dir)

    hover_thread = threading.Thread(target=hover)
    hover_thread.start()
    try:
        for _ in range(CONCURRENT_PROBES):
            stack_hit = plugin._raycaster.cast(stack_probe, probe_dir)
            assert stack_hit.geom.entity is stack_entity
            assert_allclose(stack_hit.distance, stack_distance, tol=1e-6)
        # A hover cast raising would leave the probes above running alone, hence unable to catch anything.
        assert hover_thread.is_alive()
    finally:
        is_hovering = False
        hover_thread.join()

    if not use_visual_geom:
        # A ray crossing the surface exactly on an edge shared by two triangles must be reported as a hit: the two
        # inside tests disagree there, so rounding can place the ray outside both triangles at once and let it through
        # the surface. Every edge of a closed mesh is shared, and aiming radially at each midpoint of a convex one
        # meets that edge first.
        edge_geom = edge_target.geoms[0]
        edge_verts = tensor_to_array(edge_geom.get_verts())
        if target_env_idx is not None:
            edge_verts = edge_verts[target_env_idx] + target_offset
        edge_center = edge_verts.mean(axis=0)
        for face in edge_geom.init_faces:
            for i_v0, i_v1 in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                midpoint = 0.5 * (edge_verts[i_v0] + edge_verts[i_v1])
                outward = midpoint - edge_center
                outward /= np.linalg.norm(outward)
                edge_hit = plugin._raycaster.cast(midpoint + EDGE_PROBE_RANGE * outward, -outward)
                assert edge_hit is not None and edge_hit.geom.entity is edge_target
                assert_allclose(edge_hit.distance, EDGE_PROBE_RANGE, tol=5e-6)
        return

    kinematic_hit = plugin._raycaster.cast((target_offset[0], KINEMATIC_LANE_Y, PROBE_Z), probe_dir)
    assert kinematic_hit.geom.entity is kinematic
    assert_allclose(kinematic_hit.distance, PROBE_Z - BOX_LENGTH, tol=1e-6)
    assert plugin._raycaster.cast((target_offset[0], OPTOUT_LANE_Y, PROBE_Z), probe_dir) is None

    # A kinematic entity carries no mass or inertia, so dragging it is forbidden. RAY_T places it that fraction of the
    # way from the camera to what it looks at, which puts it on the centre ray ahead of anything the box can reach.
    parked_pos = (0.0, CAM_POS[1] * (1.0 - RAY_T), CAM_POS[2] - RAY_T * (CAM_POS[2] - BOX_LENGTH))
    kinematic.set_pos(parked_pos)
    scene.step()
    x, y = viewport_size[0] // 2, viewport_size[1] // 2
    pyrender_viewer.dispatch_event("on_mouse_motion", x, y, 0, 0)
    pyrender_viewer.dispatch_event("on_mouse_press", x, y, MouseButton.LEFT, 0)
    wait_for_viewer_events(pyrender_viewer, check_event_count())
    pyrender_viewer.dispatch_event("on_mouse_drag", x, y + DRAG_DY, 0, DRAG_DY, MouseButton.LEFT, 0)
    wait_for_viewer_events(pyrender_viewer, check_event_count())
    scene.step()

    assert pyrender_viewer.is_active
    assert plugin._held_link is None
    assert_allclose(kinematic.get_pos(), parked_pos, tol=gs.EPS)


@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
@pytest.mark.parametrize("add_box", [False, True])
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
def test_add_camera_consistency(add_box, renderer, show_viewer):
    CAM_RES = (128, 128)
    CAM_POS = (0.0, -2.0, 1.5)
    CAM_LOOKAT = (0.0, 0.0, 0.0)
    CAM_FOV = 60.0

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            ambient_light=(0.1, 0.1, 0.1),
            lights=[
                dict(
                    type="directional",
                    dir=(-1, -1, -1),
                    color=(1.0, 1.0, 1.0),
                    intensity=5.0,
                ),
            ],
        ),
        viewer_options=gs.options.ViewerOptions(
            res=CAM_RES,
            camera_pos=CAM_POS,
            camera_lookat=CAM_LOOKAT,
            camera_fov=CAM_FOV,
        ),
        renderer=renderer,
        show_viewer=True,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    if add_box:
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.1, 0.1, 0.1),
                size=(0.1, 0.1, 0.1),
                fixed=True,
            ),
        )
    camera = scene.add_camera(
        res=CAM_RES,
        pos=CAM_POS,
        lookat=CAM_LOOKAT,
        fov=CAM_FOV,
        GUI=show_viewer,
    )
    scene.build()

    # Render from interactive viewer
    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active
    viewer_rgb, *_ = pyrender_viewer.render_offscreen(
        pyrender_viewer._camera_node, pyrender_viewer._renderer, rgb=True, depth=False, seg=False, normal=False
    )

    # Render from add_camera
    add_cam_rgb, *_ = camera.render(rgb=True)

    # Compare brightness (mean pixel value)
    viewer_brightness = viewer_rgb.mean()
    add_cam_brightness = add_cam_rgb.mean()

    brightness_ratio = add_cam_brightness / viewer_brightness
    assert 0.99 <= brightness_ratio <= 1.01, (
        f"add_camera brightness ({add_cam_brightness:.2f}) should match "
        f"interactive viewer brightness ({viewer_brightness:.2f}), "
        f"but ratio is {brightness_ratio:.2f}"
    )


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
def test_rasterizer_camera_sensor(renderer):
    # The sensor must share the interactive viewer's OpenGL context instead of creating a conflicting one.
    CAM_RES = (128, 64)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=CAM_RES,
            run_in_thread=False,
        ),
        renderer=renderer,
        show_viewer=True,
    )
    # At least one entity is needed to ensure the rendered image is not entirely blank,
    # otherwise it is not possible to verify that something was actually rendered.
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    camera_sensor = scene.add_sensor(
        RasterizerCameraOptions(
            res=CAM_RES,
        )
    )
    scene.build()

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    scene.step()

    data = camera_sensor.read()
    assert data.rgb.float().std() > 1.0, "RGB std too low, image may be blank"


@pytest.mark.required
@pytest.mark.parametrize("renderer_type", [RENDERER_TYPE.RASTERIZER])
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
def test_camera_render_honors_resolution(renderer):
    # A camera renders into its own offscreen FBO, sized to the camera resolution and independent of the
    # interactive window. With an interactive viewer shown, the rendered image must still match the camera
    # resolution, not the (different) viewer resolution, otherwise it no longer matches the camera intrinsics.
    viewer_res = (640, 480)
    camera_res = (360, 240)
    box_halfsize = 0.5
    camera_dist = 3.0
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=viewer_res,
        ),
        renderer=renderer,
        show_viewer=True,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.0),
            size=(2.0 * box_halfsize, 2.0 * box_halfsize, 2.0 * box_halfsize),
            fixed=True,
        ),
    )
    camera = scene.add_camera(
        res=camera_res,
        pos=(0.0, 0.0, camera_dist),
        lookat=(0.0, 0.0, 0.0),
        near=1.0,
        far=10.0,
    )
    scene.build()

    rgb, depth, _, _ = camera.render(rgb=True, depth=True)
    assert rgb.shape[:2] == (camera_res[1], camera_res[0])
    assert depth.shape[:2] == (camera_res[1], camera_res[0])

    # The on-axis center pixel hits the front face of the box, at metric depth camera_dist - box_halfsize.
    assert_allclose(depth[camera_res[1] // 2, camera_res[0] // 2], camera_dist - box_halfsize, atol=1e-3)


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
def test_thread_crash_reports_traceback():
    class CrashOnDrawPlugin(gs.vis.viewer_plugins.ViewerPlugin):
        def __init__(self):
            super().__init__()
            self.should_crash = False

        def on_draw(self):
            if self.should_crash:
                raise RuntimeError("Deliberate viewer thread crash.")

    run_in_thread = sys.platform == "linux"
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            run_in_thread=run_in_thread,
        ),
        show_viewer=True,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    crash_plugin = CrashOnDrawPlugin()
    scene.viewer.add_plugin(crash_plugin)
    scene.build()

    pyrender_viewer = scene.visualizer.viewer._pyrender_viewer
    assert pyrender_viewer.is_active

    # Enable crashing only after init is complete to avoid corrupting the init path
    crash_plugin.should_crash = True

    if run_in_thread:
        # Wait until the viewer thread has died, then check the traceback is preserved
        wait_for_viewer_events(pyrender_viewer, lambda: not pyrender_viewer.is_active)
        with pytest.raises(gs.GenesisException) as exc_info:
            scene.step()
        assert exc_info.type is gs.GenesisException
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "Deliberate viewer thread crash." in str(exc_info.value.__cause__)
    else:
        # Non-threaded: exception propagates directly through scene.step()
        with pytest.raises(RuntimeError, match="Deliberate viewer thread crash."):
            scene.step()
