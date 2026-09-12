"""Screenshot integration test for ImGuiOverlayPlugin."""

import os
import sys

import numpy as np
import pytest

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

from ..conftest import IS_INTERACTIVE_VIEWER_AVAILABLE, SKIP_NO_IMGUI_BUNDLE, is_imgui_bundle_supported
from ..utils.assertions import assert_allclose, assert_pixel_match, rgb_array_to_png_bytes


def _apply_deterministic_imgui_overrides(monkeypatch):
    """Make ImGui rendering and timing pixel-identical across renderers for snapshot tests."""
    from imgui_bundle import imgui

    # Pin ``on_draw`` so it resets ``_last_time`` after every call (forcing the FPS history to use the
    # deterministic 1/60 fallback instead of the wall clock) and parks the ImGui mouse cursor off-panel at
    # the start of every frame. The mouse-park must go through ``add_mouse_pos_event`` rather than a direct
    # ``io.mouse_pos`` assignment because ImGui rebuilds ``MousePos`` from the queued event stream inside
    # ``new_frame``: pyglet posts a cursor-position event whenever it processes Win32 messages, so a direct
    # write is overwritten before any widget reads it. The headless Windows runner still keeps a desktop
    # cursor position internally, and the window-local coordinates pyglet derives from it shift across runs
    # with window placement - landing on a widget on some runs and not others, which made the snapshot flaky.
    # Appending an off-panel event as the LAST entry in the queue at the start of every frame guarantees
    # that ``new_frame`` resolves ``MousePos`` to ``(-1, -1)`` regardless of what pyglet queued earlier.
    # ``_init_imgui`` is pre-called so ``self._io`` is available; the real ``on_draw`` short-circuits its
    # own init via the ``_init_attempted`` guard.
    original_on_draw = ImGuiOverlayPlugin.on_draw

    def _on_draw_deterministic(self):
        if not self._init_attempted:
            self._init_imgui()
        if self._available:
            self._io.add_mouse_pos_event(-1.0, -1.0)
        original_on_draw(self)
        self._last_time = None

    monkeypatch.setattr(ImGuiOverlayPlugin, "on_draw", _on_draw_deterministic)

    # Discard the plugin's 18 px ``ImFontConfig`` so ProggyClean loads at its native 13 px. ProggyClean is a bitmap
    # font, so glyph rasterization is a memcpy on every renderer (stb_truetype is not byte-identical across software
    # vs hardware OpenGL). The patch must run before the plugin's lazy ``_init_imgui`` so the renderer uploads the
    # font texture at the correct size from the start.
    original_add_font_default = imgui.ImFontAtlas.add_font_default
    monkeypatch.setattr(imgui.ImFontAtlas, "add_font_default", lambda atlas, _=None: original_add_font_default(atlas))

    # Disable shape anti-aliasing (lines, fills, textured-line shortcut) and baked thick-line atlas entries so window
    # borders and button rounding do not drift between renderers either.
    original_init_imgui = ImGuiOverlayPlugin._init_imgui

    def _init_imgui_deterministic(self):
        original_init_imgui(self)
        if not self._available:
            return
        style = self._imgui.get_style()
        style.anti_aliased_lines = False
        style.anti_aliased_fill = False
        style.anti_aliased_lines_use_tex = False
        # Pyglet's backend sets ``display_framebuffer_scale`` from the window's pixel ratio (2.0 on Retina macOS,
        # 1.0 on most Linux runners); ImGui scales vertex positions by that factor, so the same layout produces
        # different pixel grids across platforms. Pin to 1.0 so vertex positions are byte-identical everywhere.
        self._io.display_framebuffer_scale = (1.0, 1.0)
        self._io.fonts.flags |= self._imgui.ImFontAtlasFlags_.no_baked_lines.value

    monkeypatch.setattr(ImGuiOverlayPlugin, "_init_imgui", _init_imgui_deterministic)

    # The Scene tab prints each FileMorph's resolved path, an absolute machine-specific location, via text_wrapped.
    # Reduce it to its basename for two reasons: the captured frame must be reproducible across the snapshot host and
    # CI runners, and the panel auto-resizes its height to fit the wrapped path - so a long path (two wrapped lines)
    # makes the panel taller than the basename (one line). The substitution must happen before the panel first
    # renders, otherwise build() lays out the long path and the auto-resized window stays one frame behind when
    # captured, giving a taller, non-reproducible frame. Hook the plugin's entity capture to apply it up front.
    orig_capture = ImGuiOverlayPlugin._capture_pending_entities_kwargs

    def _capture_pending_entities_basename(self):
        orig_capture(self)
        for entity_kwargs in self._pending_entities_kwargs.values():
            morph = entity_kwargs["morph"]
            if isinstance(morph, gs.morphs.FileMorph):
                morph.file = os.path.basename(morph.file)

    monkeypatch.setattr(ImGuiOverlayPlugin, "_capture_pending_entities_kwargs", _capture_pending_entities_basename)


def _build_default_scene(*, enable_gui, run_in_thread=False):
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            shadow=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            # Keep ``res`` small enough to fit the virtual display area of GitHub-hosted Apple M1 macos-15 runners:
            # the on-screen capture below reads from the window framebuffer, whose size the OS clamps to the display.
            res=(640, 480),
            # The snapshot test keeps the default ``run_in_thread=False``: its capture path calls
            # ``pyrender_viewer.on_draw`` and reads the window framebuffer directly, which can only run on the
            # thread that owns the GL context.
            run_in_thread=run_in_thread,
            camera_pos=(4.5, -1.2, 2.5),
            camera_lookat=(0.0, -1.2, 0.5),
            # ``_render_help_text`` rasterizes "[i]: show keyboard instructions" via Genesis's own font path,
            # which is not byte-identical across software / hardware renderers; disable it so the captured
            # frame contains only the deterministic ImGui overlay.
            enable_help_text=False,
            enable_gui=enable_gui,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )
    # The ground plane is a thin fixed box rather than ``gs.morphs.Plane`` because the latter's reflection /
    # shading is not byte-identical between Apple Software Renderer and Mesa llvmpipe. Apple Software Renderer
    # also misrasterizes the plane when any of its vertices fall outside the camera frustum, so the camera is
    # pulled back below to keep all four corners visible.
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(2.0, 2.0, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.60, 0.85, 0.55, 1.0),
        ),
        name="ground",
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.15, 0.15, 0.15),
            pos=(0.0, 0.4, 0.075),
        ),
        surface=gs.surfaces.Default(
            color=(0.85, 0.45, 0.20, 1.0),
        ),
        name="cube",
    )
    # Shift the robot in the camera's right direction so the ImGui panel on the left hides less of it.
    scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.0, 0.0),
        ),
        name="panda",
    )
    return scene


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
@pytest.mark.skipif(not is_imgui_bundle_supported, reason=SKIP_NO_IMGUI_BUNDLE)
def test_control_panel(png_snapshot, monkeypatch):
    scene = _build_default_scene(enable_gui=False)

    _apply_deterministic_imgui_overrides(monkeypatch)

    # Pin the panel to a fixed width so changes in entity names / labels do not shift the layout.
    imgui_plugin = ImGuiOverlayPlugin(panel_width=420)
    scene.viewer.add_plugin(imgui_plugin)

    scene.build()

    # ``render_offscreen`` only renders the 3D scene (it is the path also used for in-scene camera captures while the
    # interactive viewer is alive), so it deliberately skips the viewer's plugin loop and the ImGui overlay never
    # appears in its output. Drive ``Viewer.on_draw`` synchronously from the test thread instead, which is only legal
    # because ``run_in_thread=False`` keeps the viewer (and the GL context it owns) on this thread.
    pyrender_viewer = scene.viewer._pyrender_viewer
    pyrender_viewer.switch_to()
    pyrender_viewer.on_draw()
    rgb = pyrender_viewer._renderer.jit.read_color_buf(*pyrender_viewer._viewport_size, rgba=False)
    assert rgb_array_to_png_bytes(rgb) == png_snapshot


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
@pytest.mark.skipif(not is_imgui_bundle_supported, reason=SKIP_NO_IMGUI_BUNDLE)
@pytest.mark.parametrize("performance_mode", [False, True])
def test_editing_controls(png_snapshot, monkeypatch):
    # The scene-editing controls (Rebuild Scene, Add Entity, per-entity scale & remove) render enabled in normal
    # mode and disabled (greyed) in performance mode, where the InteractiveScene advertises no editing features.
    # They live in the Scene tab, so select it to capture this mode-dependent gating.
    scene = _build_default_scene(enable_gui=True)

    _apply_deterministic_imgui_overrides(monkeypatch)

    plugin = next(p for p in scene.viewer.plugins if isinstance(p, ImGuiOverlayPlugin))
    plugin._panel_width = 420
    plugin._active_tab = "Scene"

    scene.build()

    pyrender_viewer = scene.viewer._pyrender_viewer
    pyrender_viewer.switch_to()
    pyrender_viewer.on_draw()
    rgb = pyrender_viewer._renderer.jit.read_color_buf(*pyrender_viewer._viewport_size, rgba=False)
    assert rgb_array_to_png_bytes(rgb) == png_snapshot


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
@pytest.mark.skipif(not is_imgui_bundle_supported, reason=SKIP_NO_IMGUI_BUNDLE)
def test_runtime_plugin_toggle_and_pause():
    from genesis.ext.pyrender.overlay.plugin import TOGGLEABLE_PLUGINS
    from genesis.vis.viewer_plugins import MouseInteractionPlugin, ViewerPlugin

    class DrawRecorder(ViewerPlugin):
        # Counts on_draw calls so the test can check the dispatch loop is not truncated by a mid-loop plugin removal.
        def __init__(self):
            super().__init__()
            self.draw_count = 0

        def on_draw(self):
            self.draw_count += 1

    class DrawTimeRemover(ViewerPlugin):
        # Detaches a target plugin from inside the on_draw dispatch loop, mirroring how the overlay's Plugins-tab
        # checkbox toggles a plugin while the viewer iterates its plugin list.
        def __init__(self, target):
            super().__init__()
            self.target = target
            self.removed = False

        def on_draw(self):
            if not self.removed:
                self.removed = True
                self.scene.viewer.remove_plugin(self.target)

    scene = _build_default_scene(enable_gui=True)
    scene.build()
    viewer = scene.viewer
    pyrender_viewer = viewer._pyrender_viewer
    pyrender_viewer.switch_to()
    overlay = next(plugin for plugin in viewer.plugins if isinstance(plugin, ImGuiOverlayPlugin))

    # The Plugins tab attaches and detaches whitelisted optional plugins on a live viewer through scene.viewer
    # add_plugin / remove_plugin; the whitelisted mouse-interaction plugin starts detached.
    assert MouseInteractionPlugin in (cls for _, cls in TOGGLEABLE_PLUGINS)
    assert not any(isinstance(plugin, MouseInteractionPlugin) for plugin in viewer.plugins)

    # Enabling attaches a fresh instance, visible to the live dispatch loop.
    mouse = viewer.add_plugin(MouseInteractionPlugin())
    assert mouse in viewer.plugins
    assert mouse in pyrender_viewer.plugins
    pyrender_viewer.on_draw()

    # The plugin acts on bodies through physics, so it is active only while the simulation advances the step counter:
    # advancing steps mark it running, while steps vetoed by an overlay pause mark it inactive and drop any held link.
    scene.step()
    scene.step()
    assert mouse._sim_running
    overlay.interactive_scene.pause()
    scene.step()
    scene.step()
    assert not mouse._sim_running
    assert mouse._held_link is None
    overlay.interactive_scene.resume()
    scene.step()
    assert mouse._sim_running

    # Disabling detaches it and tears down its interaction state and debug visuals.
    viewer.remove_plugin(mouse)
    assert mouse not in viewer.plugins
    assert mouse not in pyrender_viewer.plugins
    assert mouse._held_link is None
    assert not mouse._debug_interact_nodes
    assert mouse._debug_normal_node is None
    pyrender_viewer.on_draw()

    # Removing a plugin from inside the on_draw dispatch loop must not corrupt iteration: the plugin queued after the
    # mutator still receives on_draw and the target ends up detached, with no exception raised.
    target = MouseInteractionPlugin()
    remover = DrawTimeRemover(target)
    recorder = DrawRecorder()
    viewer.add_plugin(remover)
    viewer.add_plugin(target)
    viewer.add_plugin(recorder)
    pyrender_viewer.on_draw()
    assert recorder.draw_count == 1
    assert target not in viewer.plugins
    viewer.remove_plugin(remover)
    viewer.remove_plugin(recorder)


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason="Interactive viewer not supported on this platform.")
@pytest.mark.skipif(not is_imgui_bundle_supported, reason=SKIP_NO_IMGUI_BUNDLE)
@pytest.mark.parametrize("performance_mode", [False])
def test_scene_rebuild():
    # enable_gui makes the overlay own an InteractiveScene and rebuild the scene in place: the same Scene
    # object (and its viewer) stay valid across a rebuild, driven entirely through scene.step() with no
    # manual InteractiveScene. A Rebuild click only queues the request; scene.step() applies it on its thread.
    scene = _build_default_scene(enable_gui=True, run_in_thread=(sys.platform == "linux"))
    # A UV-mapped mesh with an image texture, so the capture comparison below catches the renderer swap invalidating
    # GL textures shared across the rebuild.
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck/duck.obj",
            scale=0.001,
            pos=(0.8, -0.8, 0.2),
            fixed=True,
        ),
        name="duck",
    )
    scene.build()

    scene_id = id(scene)
    plugin = next(p for p in scene.viewer.plugins if isinstance(p, ImGuiOverlayPlugin))
    interactive = plugin.interactive_scene
    assert interactive is not None
    names_before = [entity.name for entity in scene.entities]
    # The rebuild must reuse the live window rather than closing and reopening it.
    window_before = scene.viewer._pyrender_viewer
    # Move the camera off its default so the rebuild has to restore the exact viewpoint (including roll),
    # not reset it to the ViewerOptions default.
    scene.viewer.set_camera_pose(pos=np.array([2.3, 1.5, 1.9]), lookat=np.array([0.1, -0.2, 0.1]))
    camera_pose_before = scene.viewer.camera_pose.copy()

    # Pause so stepping only applies the rebuild without advancing the dynamics: the scene state must be identical
    # for the pre/post-rebuild captures.
    interactive.pause()
    rgb_before, *_ = window_before.render_offscreen(
        window_before._camera_node, window_before._renderer, rgb=True, depth=False, seg=False, normal=False
    )

    interactive.rebuild(entities_kwargs=plugin._pending_entities_kwargs)
    scene.step()

    assert id(scene) == scene_id
    assert scene.viewer is not None and scene.viewer.is_alive()
    assert scene.viewer._pyrender_viewer is window_before
    assert [entity.name for entity in scene.entities] == names_before
    assert_allclose(scene.viewer.camera_pose, camera_pose_before, atol=1e-4)
    qpos_before = scene.rigid_solver.get_qpos()

    rgb_after, *_ = window_before.render_offscreen(
        window_before._camera_node, window_before._renderer, rgb=True, depth=False, seg=False, normal=False
    )
    assert not window_before._retired_renderers, "Retired renderer was not deleted on the render thread."
    # The rebuilt scene is static, so the captures must match; a renderer retired after the rebuilt scene's first
    # draw would leave the duck texture invalid and shift the image. Use the same blurred pixel-match comparison
    # as the snapshot tests, which tolerates the few-pixel jitter software renderers produce on any platform while
    # still catching a stale-texture regression (which shifts a whole region).
    assert_pixel_match(rgb_after, rgb_before, err_msg="Rebuilt scene renders differently.")

    scene.step()
    assert_allclose(scene.rigid_solver.get_qpos(), qpos_before, tol=gs.EPS)

    # Switching the visual mode of an entity gives its geoms fresh nodes, which must keep following the simulation:
    # the arm sags under gravity once unpaused, so the frames before and after must differ
    interactive.set_entity_vis_mode(scene.get_entity("panda"), "collision")
    interactive.resume()
    rgb_collision_before, *_ = window_before.render_offscreen(
        window_before._camera_node, window_before._renderer, rgb=True, depth=False, seg=False, normal=False
    )
    for _ in range(20):
        scene.step()
    rgb_collision_after, *_ = window_before.render_offscreen(
        window_before._camera_node, window_before._renderer, rgb=True, depth=False, seg=False, normal=False
    )
    assert (rgb_collision_after != rgb_collision_before).any(axis=-1).mean() > 0.01


@pytest.mark.required
def test_capture_pending_entities_preserves_heterogeneous_morphs():
    scene = gs.Scene(show_viewer=False)
    single_morph = gs.morphs.Box(size=(0.1, 0.1, 0.1))
    single_entity = scene.add_entity(
        morph=single_morph,
        visualize_contact=True,
        name="single",
    )
    heterogeneous_morphs = (
        gs.morphs.Box(size=(0.2, 0.2, 0.2)),
        gs.morphs.Cylinder(radius=0.05, height=0.2),
    )
    heterogeneous_entity = scene.add_entity(
        morph=heterogeneous_morphs,
        visualize_contact=True,
        name="heterogeneous",
    )

    plugin = ImGuiOverlayPlugin.__new__(ImGuiOverlayPlugin)
    plugin.scene = scene

    plugin._capture_pending_entities_kwargs()

    single_kwargs = plugin._pending_entities_kwargs[single_entity.name]
    assert single_kwargs["morph"] is single_morph
    assert single_kwargs["material"] is single_entity.material
    assert single_kwargs["surface"] is single_entity.surface
    assert single_kwargs["visualize_contact"] is True

    heterogeneous_kwargs = plugin._pending_entities_kwargs[heterogeneous_entity.name]
    assert heterogeneous_kwargs["morph"] == heterogeneous_morphs
    assert heterogeneous_kwargs["morph"][0] is heterogeneous_morphs[0]
    assert heterogeneous_kwargs["morph"][1] is heterogeneous_morphs[1]
    assert heterogeneous_kwargs["material"] is heterogeneous_entity.material
    assert heterogeneous_kwargs["surface"] is heterogeneous_entity.surface
    assert heterogeneous_kwargs["visualize_contact"] is True
