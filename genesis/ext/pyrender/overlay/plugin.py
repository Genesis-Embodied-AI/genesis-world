"""
ImGui overlay plugin for joint control and simulation controls.

Requires the ``render`` optional extras: ``pip install 'genesis-world[render]'``.
"""

import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.interactive_scene import InteractiveFeature, InteractiveScene
from genesis.ext.pyrender.overlay.style import apply_dark_theme
from genesis.ext.pyrender.overlay.types import build_entity_joint_data, EntityCacheEntry, EntityJointData
from genesis.utils.misc import tensor_to_array
from genesis.vis.viewer_plugins import EVENT_HANDLE_STATE, EVENT_HANDLED, MouseInteractionPlugin, ViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.viewer import Viewer

_FPS_HISTORY_SIZE = 30
# Add Entity / Stage dropdown. The "File" entry covers every file-based morph (URDF / MJCF / Mesh / USD); its
# concrete type is deduced from the file extension. The others are primitive morphs built from their own params.
_ADD_ENTITY_TYPES = ["File", "Box", "Sphere", "Cylinder", "Plane"]
# Optional viewer plugins the user may attach/detach at runtime from the Plugins tab, as (label, class) pairs. Only
# plugins that are safe to build and tear down on a live viewer belong here; the always-on plugins (default keybinds,
# this overlay itself) are deliberately excluded so they can never be toggled off from their own panel.
TOGGLEABLE_PLUGINS: tuple[tuple[str, type[ViewerPlugin]], ...] = (("Mouse Interaction", MouseInteractionPlugin),)


def button_size_with_min(imgui, label: str, min_width: float) -> tuple[float, float]:
    """Return a ``(width, height)`` size tuple for ``imgui.button(label, size=...)`` that auto-fits the label but
    never drops below ``min_width``. Height stays 0 so ImGui picks its default."""
    text_width = imgui.calc_text_size(label).x
    return max(min_width, text_width + 2.0 * imgui.get_style().frame_padding.x), 0.0


def draw_separator(imgui, thickness: int = 2) -> None:
    """Draw a ``thickness`` px-tall horizontal separator at integer-pixel coordinates. ImGui's built-in
    ``Separator()`` draws a 1 px line centered on a half-pixel boundary, which different OpenGL drivers rasterize
    onto different rows; this helper uses a filled rectangle aligned to integer rows so the line is byte-identical
    on every renderer."""
    draw_list = imgui.get_window_draw_list()
    x0, y = imgui.get_cursor_screen_pos()
    x0_i, y_i = int(x0), int(y)
    x1_i = x0_i + int(imgui.get_content_region_avail().x)
    color = imgui.get_color_u32(imgui.Col_.separator.value)
    draw_list.add_rect_filled((x0_i, y_i), (x1_i, y_i + thickness), color)
    imgui.dummy((0.0, float(thickness)))


class ImGuiOverlayPlugin(ViewerPlugin):
    """
    ViewerPlugin that adds an ImGui control panel for simulation and joint control.

    Features:
    - Simulation controls: play/pause, step, reset
    - Joint sliders for each entity (editable only when paused)
    - FPS display with rolling average
    - Multi-step support
    - Custom panel registration API

    Limitations:
    - Only controls environment 0 in batched simulations

    Usage:
        scene = gs.Scene(viewer_options=gs.options.ViewerOptions(enable_gui=True), show_viewer=True)
        scene.build()

        while scene.viewer.is_alive():
            scene.step()

    The overlay drives play/pause and scene rebuild through an InteractiveScene it wraps the scene with, so
    the loop just calls ``scene.step()``; the controls take effect there on the stepping thread.
    """

    def __init__(
        self,
        controlled_env_idx: int = 0,
        free_joint_pos_limit: float = 10.0,
        panel_width: int | None = None,
    ):
        try:
            import imgui_bundle  # noqa: F401
        except ImportError:
            gs.raise_exception(
                "ImGuiOverlayPlugin requires the optional 'imgui-bundle' dependency. Install Genesis with the "
                "'render' extras (pip install 'genesis-world[render]'). Pre-built wheels are not published for every "
                "Python/OS combination (e.g. Python 3.10, Linux aarch64); on those platforms install manually via "
                "'pip install imgui-bundle', which builds from source and requires CMake."
            )

        super().__init__()
        # InteractiveScene wrapping the current scene, created and owned by this plugin in build(). It backs the
        # scene-editing controls (feature gating + rebuild) so the user never instantiates one manually.
        self.interactive_scene = None
        self._controlled_env_idx = controlled_env_idx
        self._free_joint_pos_limit = free_joint_pos_limit
        self._panel_width = panel_width
        self._imgui = None
        self._ctx = None
        self._impl = None
        self._io = None
        self._available = False
        self._init_attempted = False
        self._last_time = None
        # Number of frames a single "Step" click advances. Play/pause and stepping state itself live on the
        # InteractiveScene (the controller); this overlay only toggles them.
        self._step_count = 1
        self._entity_cache = {}
        self._user_panels = []
        self._fps_history = []
        # Name of the tab to force-select on the next frame, then cleared. None leaves the user's selection.
        self._active_tab = None

        # The Scene editor panel mutates this local mirror of the live entities; on "Rebuild Scene" it is submitted to
        # the InteractiveScene, which applies it on the stepping thread.
        self._pending_dirty = False
        self._pending_entities_kwargs: dict[str, dict] = {}
        self._add_entity_morph_type = 0  # index into _ADD_ENTITY_TYPES
        self._add_entity_file = ""
        self._add_entity_pos = [0.0, 0.0, 0.0]
        self._add_entity_scale = 1.0
        # Collision-mesh processing for file morphs (convexify replaces collision meshes with their convex hull,
        # decimate reduces their face count).
        self._add_convexify = True
        self._add_decimate = True
        # Primitive geometry params
        self._add_box_size = [0.2, 0.2, 0.2]
        self._add_sphere_radius = 0.1
        self._add_cylinder_radius = 0.05
        self._add_cylinder_height = 0.2
        self._add_entity_fixed = True
        # File browser state
        self._file_browser_open = False
        self._file_browser_dir = os.getcwd()
        self._file_browser_selected = -1
        # Gizmo state
        self._gizmo = None  # imguizmo.im_guizmo module (lazy loaded)
        self._gizmo_operation = None  # gizmo.OPERATION.translate
        self._gizmo_mode = None  # gizmo.MODE.world
        self._gizmo_entity_idx = -1  # which entity is selected for gizmo manipulation
        # Per-entity euler/quat mode: entity_idx -> "euler" or "quat"
        self._rotation_mode = {}
        # Per-entity wireframe state: entity_idx -> bool
        self._wireframe_state = {}

    def register_panel(self, callback, section="side"):
        """Register custom UI panel. callback(imgui) called each frame.

        Thread-safe: uses copy-on-write list.

        Args:
            callback: Function taking imgui module as argument, called each frame.
            section: "side" adds to main panel, "overlay" creates floating window.
        """
        new_list = list(self._user_panels) + [(callback, section)]
        self._user_panels = new_list  # Atomic reference swap

    def build(self, viewer: "Viewer", camera, scene: "Scene"):
        """Store references; ImGui initialization is deferred to on_draw (viewer thread)."""
        super().build(viewer, camera, scene)
        # Reset ImGui state so it re-initializes in the new viewer thread (needed after scene rebuild creates a new
        # viewer/OpenGL context). Don't destroy the old context here, it belonged to the old viewer thread and is
        # already invalid after scene.destroy().
        if self._init_attempted:
            self._impl = None
            self._io = None
            self._available = False
            self._init_attempted = False
            self._last_time = None
        # Non-batched scenes (n_envs == 0) expect envs_idx=None on entity setters and return 1D qpos tensors. Collapse
        # the controlled index to None so downstream code can pass it through unconditionally without branching on the
        # batched/non-batched shape.
        if scene.n_envs == 0:
            self._controlled_env_idx = None
        elif self._controlled_env_idx is not None and not (0 <= self._controlled_env_idx < scene.n_envs):
            gs.raise_exception(
                f"controlled_env_idx={self._controlled_env_idx} out of range for scene with n_envs={scene.n_envs}."
            )
        # Cache entity data now (doesn't require OpenGL)
        self._cache_entity_data()
        self._capture_pending_entities_kwargs()
        # Wrap the scene on first build. The InteractiveScene is the controller (it owns play/pause, stepping and
        # rebuild); this overlay is just a view that reads and toggles its state. On a rebuild the plugin is re-attached
        # to the reconstructed scene (same object), so the existing wrapper still applies.
        if self.interactive_scene is None:
            self.interactive_scene = InteractiveScene(scene)

    @property
    def _supported_features(self) -> frozenset[InteractiveFeature]:
        """Editing features advertised by the plugin's InteractiveScene for the current simulator mode,
        queried live. Each scene-editing control gates on its own feature and renders disabled when absent."""
        return self.interactive_scene.supported_features

    def _refresh_visuals(self):
        """Refresh render transforms after a GUI-driven mutation.

        Caller must hold the render lock. Covers both the rigid and kinematic solvers, since both manage
        KinematicEntity-based entities the browser can pose."""
        for solver in (self.scene.rigid_solver, self.scene.kinematic_solver):
            if solver.is_active:
                solver.update_vgeoms()
        ctx = self.viewer.gs_context
        ctx.update_link_frame()
        ctx.update_rigid()

    def _apply_entity_vis_mode(self, entity, mode: str):
        """Switch the entity's rendered mesh between ``"visual"`` and ``"collision"``. Removes the previous
        render nodes from the context, swaps ``entity.surface.vis_mode``, then rebuilds nodes from the
        appropriate geom set."""
        if not isinstance(entity.surface, gs.surfaces.Surface):
            return
        old_mode = entity.surface.vis_mode
        if old_mode == mode:
            return

        with self.viewer.render_lock:
            ctx = self.viewer.gs_context
            solver = entity.solver

            old_geoms = entity.vgeoms if old_mode == "visual" else entity.geoms
            for geom in old_geoms:
                ctx.remove_rigid_node(geom)

            entity.surface.vis_mode = mode
            self._refresh_visuals()

            is_visual = mode == "visual"
            geoms = entity.vgeoms if is_visual else entity.geoms
            geoms_T = ctx.rigid_geoms_T(solver, is_visual)
            for geom in geoms:
                ctx.add_geom_node(geom, geoms_T)

    def _init_imgui(self):
        """Initialize ImGui. Must be called from the viewer thread (e.g., in on_draw)."""
        if self._init_attempted:
            return
        self._init_attempted = True

        try:
            from imgui_bundle import imgui
            from imgui_bundle.python_backends import pyglet_backend

            self._imgui = imgui
            self._ctx = imgui.create_context()
            # Load default font at larger size before renderer builds the atlas
            io = imgui.get_io()
            io.fonts.clear()
            font_cfg = imgui.ImFontConfig()
            font_cfg.size_pixels = 18.0
            io.fonts.add_font_default(font_cfg)
            self._impl = pyglet_backend.create_renderer(self.viewer, attach_callbacks=False)
            # Fix: Set window reference for cursor handling (not set when attach_callbacks=False)
            self._impl._window = self.viewer
            self._io = imgui.get_io()
            self._io.set_ini_filename("")  # Don't persist window positions
            # Render the first frame as if the window is unfocused so ImGui's keyboard nav does not auto-pick the first
            # focusable widget and draw a nav highlight on top of it. Pyglet's Win32 backend reports the window as
            # focused at startup, which would otherwise leave the first entity header with a visible highlight until the
            # user interacts. Subsequent focus events from pyglet (mouse click, key press, etc.) restore normal focus
            # behavior on demand.
            self._io.add_focus_event(False)
            # Set up clipboard (pyglet backend doesn't do this by default). Pyglet caches _clipboard_str and only clears
            # it on SelectionClear events, which may not be dispatched in time. Invalidate the cache before each read so
            # we always get fresh system clipboard content.
            window_ref = self.viewer

            def _get_clipboard(_ctx):
                try:
                    window_ref._clipboard_str = None
                    text = window_ref.get_clipboard_text()
                    return text if text else ""
                except Exception:
                    return ""

            def _set_clipboard(_ctx, text):
                try:
                    window_ref.set_clipboard_text(text)
                except Exception:
                    pass

            platform_io = imgui.get_platform_io()
            platform_io.platform_get_clipboard_text_fn = _get_clipboard
            platform_io.platform_set_clipboard_text_fn = _set_clipboard
            apply_dark_theme(imgui)
            self._available = True

            # Try to load ImGuizmo for 3D gizmos
            try:
                from imgui_bundle import imguizmo

                self._gizmo = imguizmo.im_guizmo
                self._gizmo_operation = self._gizmo.OPERATION.translate
                self._gizmo_mode = self._gizmo.MODE.world
                self._gizmo.set_gizmo_size_clip_space(0.15)
                self._gizmo.allow_axis_flip(False)
            except ImportError:
                pass
        except ImportError:
            print("ImGuiOverlayPlugin: imgui-bundle not found. Install with: pip install imgui-bundle")
        except Exception as e:
            print(f"ImGuiOverlayPlugin: Failed to initialize ImGui: {e}")

    def _cache_entity_data(self):
        """Cache static joint metadata from all rigid and kinematic entities.

        RigidEntity derives from KinematicEntity and both expose the same joint API, so the entity browser controls
        them uniformly."""
        self._entity_cache.clear()
        for entity in self.scene.entities:
            if not isinstance(entity, gs.engine.entities.KinematicEntity):
                continue
            if entity.n_dofs == 0:
                # Still include for vis_mode toggle, but no joint data
                self._entity_cache[entity.idx] = EntityCacheEntry(
                    entity=entity,
                    name=entity.name,
                    joint_data=EntityJointData(
                        q_names=[],
                        q_limits=([], []),
                        q_is_quaternion=[],
                        quat_groups=[],
                        has_free_joint=False,
                        free_joint_q_start=-1,
                    ),
                    n_qs=0,
                    n_dofs=0,
                )
                continue

            jdata = build_entity_joint_data(entity, self._free_joint_pos_limit)
            if jdata.q_names:
                self._entity_cache[entity.idx] = EntityCacheEntry(
                    entity=entity,
                    name=entity.name,
                    joint_data=jdata,
                    n_qs=len(jdata.q_names),
                    n_dofs=entity.n_dofs,
                )

    def _capture_pending_entities_kwargs(self):
        """Capture current entity construction kwargs into ``self._pending_entities_kwargs`` for the Scene
        editor panel. Keyed by entity name; values are the kwargs forwarded to ``scene.add_entity``."""
        self._pending_entities_kwargs = {}
        for entity in self.scene.entities:
            # A heterogeneous entity has multiple morph variants; capture them all so a rebuild reproduces every
            # variant instead of collapsing the entity to its first morph.
            if isinstance(entity, gs.engine.entities.KinematicEntity):
                morph = tuple(entity.morphs) if len(entity.morphs) > 1 else entity.main_morph
            else:
                morph = entity.morph
            kwargs: dict[str, Any] = {"morph": morph}
            # Carry the material and surface so a rebuild preserves the entity's solver (e.g. a Kinematic entity must
            # not silently become Rigid, the add_entity default). visualize_contact is rigid-only.
            if isinstance(entity, gs.engine.entities.KinematicEntity):
                kwargs["material"] = entity.material
                kwargs["surface"] = entity.surface
                if isinstance(entity, gs.engine.entities.RigidEntity):
                    kwargs["visualize_contact"] = entity.visualize_contact
            self._pending_entities_kwargs[entity.name] = kwargs

    def _is_capturing(self) -> bool:
        """Check if ImGui or gizmo wants mouse/keyboard input."""
        if not self._available:
            return False
        return self._io.want_capture_mouse or self._io.want_capture_keyboard or self._is_gizmo_active()

    # Event handlers - forward input to ImGui and block when capturing
    def on_mouse_press(self, x, y, button, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_press(x, y, button, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_release(self, x, y, button, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_release(x, y, button, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_scroll(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        if self._available:
            # Pyglet's deltas already reflect the system's scrolling preference. Forward them directly because the
            # imgui-bundle Pyglet backend negates the vertical delta, which reverses the configured direction.
            self._io.add_mouse_wheel_event(dx, dy)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_motion(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_motion(x, y, dx, dy)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_key_press(self, symbol, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_key_press(symbol, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_key_release(self, symbol, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_key_release(symbol, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_text(self, text) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_text(text)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_resize(self, width, height) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_resize(width, height)
        return None

    def on_draw(self) -> None:
        """Render ImGui overlay after scene is drawn."""
        # Lazy initialization: must happen in viewer thread (which owns OpenGL context)
        if not self._init_attempted:
            self._init_imgui()

        if not self._available:
            return

        # Update delta time manually (avoid calling pyglet.clock.tick() which conflicts with viewer loop)
        current_time = time.perf_counter()
        if self._last_time is not None:
            self._io.delta_time = current_time - self._last_time
        else:
            self._io.delta_time = 1.0 / 60.0
        if self._io.delta_time <= 0.0:
            self._io.delta_time = 1.0 / 1000.0
        self._last_time = current_time

        # Track FPS history
        if self._io.delta_time > 0:
            self._fps_history.append(1.0 / self._io.delta_time)
            if len(self._fps_history) > _FPS_HISTORY_SIZE:
                self._fps_history = self._fps_history[-_FPS_HISTORY_SIZE:]

        self._imgui.new_frame()

        # Initialize ImGuizmo for this frame
        if self._gizmo is not None:
            self._gizmo.begin_frame()
            io = self._io
            self._gizmo.set_rect(0, 0, io.display_size.x, io.display_size.y)
            self._gizmo.set_orthographic(not self.viewer.viewer_flags["use_perspective_cam"])

        self._render_control_panel()

        # Render 3D gizmos for selected free-joint entity
        if self._gizmo is not None and self._gizmo_entity_idx >= 0:
            self._render_gizmo()

        self._imgui.render()
        self._impl.render(self._imgui.get_draw_data())

    def _render_control_panel(self):
        """Render unified control panel with all sections."""
        imgui = self._imgui
        if self._panel_width is not None:
            # Pin the width while letting the height autoresize to content.
            imgui.set_next_window_size_constraints((self._panel_width, 0.0), (self._panel_width, float("inf")))
        imgui.begin("Genesis Control Panel", flags=imgui.WindowFlags_.always_auto_resize)

        self._render_sim_controls()

        if imgui.begin_tab_bar("##main_tabs"):
            for name, render_section in (
                ("Entities", self._render_entity_browser),
                ("Visualization", self._render_visualization),
                ("Camera", self._render_camera_controls),
                ("Scene", self._render_scene_editor),
                ("Plugins", self._render_plugins),
            ):
                # Force the requested tab selected for this frame; otherwise honor the user's selection.
                flags = imgui.TabItemFlags_.set_selected.value if self._active_tab == name else 0
                if imgui.begin_tab_item(name, None, flags)[0]:
                    render_section()
                    imgui.end_tab_item()
            imgui.end_tab_bar()
        self._active_tab = None

        # Render user callback panels (side panels)
        for callback, section in self._user_panels:
            if section == "side":
                callback(imgui)

        imgui.end()

        # Render overlay panels as separate windows
        for callback, section in self._user_panels:
            if section == "overlay":
                callback(imgui)

    def _render_sim_controls(self):
        """Render simulation control buttons, time display, and FPS."""
        imgui = self._imgui
        interactive = self.interactive_scene

        # State label
        if interactive.paused:
            imgui.text_colored((1.0, 0.7, 0.0, 1.0), "Paused")
        else:
            imgui.text_colored((0.4, 0.9, 0.4, 1.0), "Running")

        # Play/Pause and Reset (always visible), Step (only when paused). Auto-fit the label but with a 60-px floor so
        # single-word verbs share a consistent baseline width and never get truncated.
        play_pause = "Pause" if not interactive.paused else "Play"
        if imgui.button(play_pause, size=button_size_with_min(imgui, play_pause, 60.0)):
            interactive.resume() if interactive.paused else interactive.pause()
        if interactive.paused:
            imgui.same_line()
            if imgui.button("Step", size=button_size_with_min(imgui, "Step", 60.0)):
                interactive.step(self._step_count)
        imgui.same_line()
        if imgui.button("Reset", size=button_size_with_min(imgui, "Reset", 60.0)):
            interactive.reset()

        # Record toggle. Drives the same on-screen video capture as the 'R' keybind; tinted red while recording so
        # its active state is obvious, and stopping prompts for a destination file.
        imgui.same_line()
        recording = interactive.recording
        record_label = "Stop Rec" if recording else "Record"
        if recording:
            imgui.push_style_color(imgui.Col_.button.value, (0.70, 0.16, 0.16, 0.90))
            imgui.push_style_color(imgui.Col_.button_hovered.value, (0.82, 0.22, 0.22, 0.95))
            imgui.push_style_color(imgui.Col_.button_active.value, (0.62, 0.12, 0.12, 1.0))
        if imgui.button(record_label, size=button_size_with_min(imgui, record_label, 60.0)):
            interactive.toggle_recording()
        if recording:
            imgui.pop_style_color(3)

        # Time display (frame count * dt = simulation time)
        step = self.scene.sim.cur_step_global
        imgui.text(f"Time: {step * self.scene.sim.dt:.3f}s  Step: {step}")

        # FPS display
        if self._fps_history:
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            imgui.same_line()
            imgui.text(f"  FPS: {avg_fps:.0f}")

        # Realtime pacing control. "Uncapped" runs the sim as fast as compute allows; otherwise the slider sets
        # how many times real time it is paced to (1.0 = real time).
        viewer = self.scene.viewer
        uncapped = viewer.realtime_factor is None
        changed_uncapped, new_uncapped = imgui.checkbox("Uncapped##realtime", uncapped)
        if changed_uncapped:
            viewer.realtime_factor = None if new_uncapped else 1.0
            uncapped = new_uncapped
        imgui.same_line()
        imgui.begin_disabled(uncapped)
        factor = 1.0 if viewer.realtime_factor is None else viewer.realtime_factor
        changed_factor, new_factor = imgui.slider_float("Realtime##realtime_factor", factor, 0.1, 4.0, "%.2fx")
        imgui.end_disabled()
        if changed_factor and not uncapped:
            viewer.realtime_factor = new_factor

        if self.scene.n_envs > 1:
            imgui.text_colored(
                (1.0, 0.7, 0.0, 1.0),
                f"Note: Controlling env {self._controlled_env_idx} of {self.scene.n_envs}",
            )

        draw_separator(imgui)

    def _render_visualization(self):
        """Render visualization toggle controls."""
        imgui = self._imgui
        render_flags = self.viewer.render_flags
        gs_context = self.viewer.gs_context

        # Shadows
        changed, new_val = imgui.checkbox("Shadows", render_flags["shadows"])
        if changed:
            render_flags["shadows"] = new_val

        # World Frame
        changed, new_val = imgui.checkbox("World Frame", gs_context.world_frame_shown)
        if changed:
            (gs_context.on_world_frame if new_val else gs_context.off_world_frame)()

        # Link Frame
        changed, new_val = imgui.checkbox("Link Frame", gs_context.link_frame_shown)
        if changed:
            (gs_context.on_link_frame if new_val else gs_context.off_link_frame)()

        # Link Frame Size slider
        link_size = gs_context.link_frame_size
        changed_size, new_size = imgui.slider_float("Frame Size##link_frame_size", link_size, 0.02, 0.5, "%.2f")
        if changed_size and link_size > 0 and new_size > 0:
            gs_context.link_frame_mesh.vertices *= new_size / link_size
            gs_context.link_frame_size = new_size
            if gs_context.link_frame_shown:
                gs_context.off_link_frame()
                gs_context.on_link_frame()

        # Camera Frustum
        changed, new_val = imgui.checkbox("Camera Frustum", gs_context.camera_frustum_shown)
        if changed:
            (gs_context.on_camera_frustum if new_val else gs_context.off_camera_frustum)()

        # Face Normals
        changed, new_val = imgui.checkbox("Face Normals", render_flags["face_normals"])
        if changed:
            render_flags["face_normals"] = new_val

        # Vertex Normals
        changed, new_val = imgui.checkbox("Vertex Normals", render_flags["vertex_normals"])
        if changed:
            render_flags["vertex_normals"] = new_val

        draw_separator(imgui)

        # Orthographic Camera
        is_ortho = not self.viewer.viewer_flags["use_perspective_cam"]
        changed, new_ortho = imgui.checkbox("Orthographic Camera", is_ortho)
        if changed:
            self.viewer.viewer_flags["use_perspective_cam"] = not new_ortho
            if new_ortho:
                self.viewer._camera_node.camera = self.viewer._default_orth_cam
            else:
                self.viewer._camera_node.camera = self.viewer._default_persp_cam

    def _render_plugins(self):
        """Render checkboxes to attach/detach the whitelisted optional viewer plugins at runtime.

        Each checkbox reflects whether an instance of that plugin class is currently registered on the viewer:
        ticking it builds and registers a fresh instance, unticking it detaches the live one. Toggling goes through
        the Genesis viewer so the registered set stays consistent across an InteractiveScene rebuild."""
        imgui = self._imgui
        viewer = self.scene.viewer
        registered = self.viewer.plugins
        for label, plugin_cls in TOGGLEABLE_PLUGINS:
            instance = next((plugin for plugin in registered if isinstance(plugin, plugin_cls)), None)
            changed, enabled = imgui.checkbox(f"{label}##plugin_{plugin_cls.__name__}", instance is not None)
            if not changed:
                continue
            if enabled:
                viewer.add_plugin(plugin_cls())
            else:
                viewer.remove_plugin(instance)

    def _render_gizmo(self):
        """Render the 3D manipulation gizmo for the selected entity, translating/rotating its base link.

        Works for fixed and floating base alike, applying ImGuizmo's resulting absolute pose via set_pos / set_quat.
        The gizmo must be fed a projection with a finite far plane: with the renderer's infinite far plane ImGuizmo's
        mouse-ray unprojection is ill-conditioned for an off-axis camera, quantizing the drag to a coarse grid."""
        gizmo = self._gizmo
        Matrix16 = gizmo.Matrix16

        data = self._entity_cache.get(self._gizmo_entity_idx)
        if data is None:
            return

        entity = data.entity

        pos = tensor_to_array(entity.get_pos(relative=False))
        quat_wxyz = tensor_to_array(entity.get_quat(relative=False))
        if self._controlled_env_idx is not None:
            pos = pos[self._controlled_env_idx]
            quat_wxyz = quat_wxyz[self._controlled_env_idx]
        rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # scipy uses x,y,z,w

        obj_mat = np.eye(4)
        obj_mat[:3, :3] = rot.as_matrix()
        obj_mat[:3, 3] = pos
        # ImGuizmo expects column-major (transpose for row-major numpy)
        object_matrix = Matrix16(obj_mat.T.flatten().tolist())

        cam_pose = self.viewer._trackball._n_pose.copy()
        camera_view = Matrix16(np.linalg.inv(cam_pose).T.flatten().tolist())

        w, h = int(self._io.display_size.x), int(self._io.display_size.y)
        if w <= 0 or h <= 0:
            return
        # A perspective camera (proj[3, 3] == 0) renders with an infinite far plane (proj[2, 2] == -1), which makes
        # ImGuizmo's mouse-ray unprojection ill-conditioned for an off-axis camera and snaps the drag to a coarse grid;
        # substitute a finite far plane (fov, aspect and near are preserved). An orthographic camera has parallel rays
        # (no such issue) and a different matrix layout, so leave its projection untouched.
        proj = self.camera.camera.get_projection_matrix(width=w, height=h).copy()
        if proj[3, 3] == 0.0:
            near = proj[2, 3] / (proj[2, 2] - 1.0)
            far = 1000.0
            proj[2, 2] = -(far + near) / (far - near)
            proj[2, 3] = -2.0 * far * near / (far - near)
        camera_proj = Matrix16(proj.T.flatten().tolist())

        modified = gizmo.manipulate(
            camera_view,
            camera_proj,
            self._gizmo_operation,
            self._gizmo_mode,
            object_matrix,
        )
        if not modified:
            return

        # ImGuizmo wrote the new absolute pose into object_matrix.
        new_mat = np.array(object_matrix.values).reshape(4, 4).T
        self.interactive_scene.pause()
        with self.viewer.render_lock:
            if self._gizmo_operation == gizmo.OPERATION.rotate:
                quat_xyzw = R.from_matrix(new_mat[:3, :3]).as_quat()  # scipy: x,y,z,w
                new_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                # set_quat must be absolute (relative=False), since the KinematicEntity default is relative.
                entity.set_quat(new_quat_wxyz, envs_idx=self._controlled_env_idx, relative=False)
            else:
                entity.set_pos(new_mat[:3, 3], envs_idx=self._controlled_env_idx, relative=False)
            self._refresh_visuals()

    def _is_gizmo_active(self):
        """Check if the gizmo is being used (for input blocking)."""
        if self._gizmo is not None:
            return self._gizmo.is_using() or self._gizmo.is_over()
        return False

    def _render_camera_controls(self):
        """Render camera position, lookat, FOV controls."""
        imgui = self._imgui
        trackball = self.viewer._trackball

        # Read current camera state from trackball
        pose = trackball._n_pose
        pos = [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])]
        # Use trackball's actual orbit center as lookat (not derived from z-axis)
        target = trackball._n_target
        lookat = [float(target[0]), float(target[1]), float(target[2])]

        # Position drag
        changed_pos, new_pos = imgui.drag_float3("Position##cam_pos", pos, 0.05, -100.0, 100.0, "%.2f")

        # Lookat drag
        changed_lookat, new_lookat = imgui.drag_float3("Lookat##cam_lookat", lookat, 0.05, -100.0, 100.0, "%.2f")

        if changed_pos or changed_lookat:
            cam_pos = np.array(list(new_pos)) if changed_pos else np.array(pos)
            cam_lookat = np.array(list(new_lookat)) if changed_lookat else np.array(lookat)
            # Build pose with fixed world-up to prevent unintuitive roll
            world_up = np.array([0.0, 0.0, 1.0])
            cam_pose = gu.pos_lookat_up_to_T(cam_pos, cam_lookat, world_up)
            self.scene.viewer._camera_up = cam_pose[:3, 1].copy()
            trackball.set_camera_pose(cam_pose)
            # Sync trackball orbit center so mouse orbiting works correctly after
            trackball._n_target = cam_lookat.copy()
            trackball._target = cam_lookat.copy()

        # FOV slider
        fov_deg = float(self.camera.camera.yfov * 180.0 / np.pi)
        changed_fov, new_fov = imgui.slider_float("FOV##cam_fov", fov_deg, 15.0, 120.0, "%.1f")
        if changed_fov:
            self.camera.camera.yfov = new_fov * np.pi / 180.0

        # Reset Camera button
        if imgui.button("Reset Camera", size=(120, 0)):
            self.viewer._reset_view()

    # Asset file extensions accepted by Add Entity, each mapped to the morph class deduced from it (matching how
    # `gs launch` selects a morph from the filename). The Add Entity menu takes only a file path and infers the type.
    _MORPH_BY_EXTENSION = {
        ".urdf": gs.morphs.URDF,
        ".xacro": gs.morphs.URDF,
        ".xml": gs.morphs.MJCF,
        ".obj": gs.morphs.Mesh,
        ".stl": gs.morphs.Mesh,
        ".ply": gs.morphs.Mesh,
        ".dae": gs.morphs.Mesh,
        ".glb": gs.morphs.Mesh,
        ".gltf": gs.morphs.Mesh,
        ".usd": gs.morphs.USD,
        ".usda": gs.morphs.USD,
        ".usdc": gs.morphs.USD,
        ".usdz": gs.morphs.USD,
    }

    def _render_file_browser(self):
        """Render a file browser popup for selecting asset files."""
        imgui = self._imgui
        if not self._file_browser_open:
            return

        imgui.open_popup("File Browser##file_popup")
        imgui.set_next_window_size((500, 400))
        if imgui.begin_popup_modal("File Browser##file_popup")[0]:
            # Current directory display with parent navigation
            if imgui.button("^##parent_dir"):
                parent = os.path.dirname(self._file_browser_dir)
                if parent != self._file_browser_dir:
                    self._file_browser_dir = parent
                    self._file_browser_selected = -1
            imgui.same_line()
            imgui.text(self._file_browser_dir)
            draw_separator(imgui)

            # List directory contents
            valid_exts = set(self._MORPH_BY_EXTENSION)
            try:
                entries = sorted(os.listdir(self._file_browser_dir))
            except OSError:
                entries = []

            dirs = [
                e for e in entries if os.path.isdir(os.path.join(self._file_browser_dir, e)) and not e.startswith(".")
            ]
            files = [
                e
                for e in entries
                if os.path.isfile(os.path.join(self._file_browser_dir, e))
                and (not valid_exts or os.path.splitext(e)[1].lower() in valid_exts)
            ]
            items = [d + "/" for d in dirs] + files

            if imgui.begin_child("file_list", size=(0, -30)):
                for idx, item in enumerate(items):
                    is_dir = item.endswith("/")
                    selected = idx == self._file_browser_selected
                    if imgui.selectable(item, selected)[0]:
                        if is_dir:
                            self._file_browser_dir = os.path.join(self._file_browser_dir, item[:-1])
                            self._file_browser_selected = -1
                        else:
                            self._file_browser_selected = idx
                    # Double-click on file to confirm
                    if not is_dir and imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                        self._add_entity_file = os.path.join(self._file_browser_dir, item)
                        self._file_browser_open = False
                        imgui.close_current_popup()
                imgui.end_child()

            # OK / Cancel buttons
            can_select = self._file_browser_selected >= 0 and self._file_browser_selected >= len(dirs)
            if imgui.button("OK", size=(80, 0)) and can_select:
                file_name = files[self._file_browser_selected - len(dirs)]
                self._add_entity_file = os.path.join(self._file_browser_dir, file_name)
                self._file_browser_open = False
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", size=(80, 0)):
                self._file_browser_open = False
                imgui.close_current_popup()

            imgui.end_popup()
        else:
            # Popup was closed (e.g. clicking outside)
            self._file_browser_open = False

    _SCENE_EDIT_DISABLED_TOOLTIP = "This action is unavailable for the current scene configuration."

    def _maybe_show_disabled_tooltip(self, disabled: bool):
        """Show the unavailable-feature tooltip when the previous item is disabled and hovered."""
        if not disabled:
            return
        imgui = self._imgui
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            imgui.set_tooltip(self._SCENE_EDIT_DISABLED_TOOLTIP)

    def _render_scene_editor(self):
        """Render scene editing controls (entity scale, add entity, rebuild).

        Each control maps to an InteractiveFeature and renders in its disabled visual state when the
        scene does not support that feature, so the panel layout stays identical across scene types.
        """
        imgui = self._imgui
        features = self._supported_features
        scale_disabled = InteractiveFeature.SCALE_ENTITY not in features
        remove_disabled = InteractiveFeature.REMOVE_ENTITY not in features
        add_disabled = InteractiveFeature.ADD_ENTITY not in features
        rebuild_disabled = InteractiveFeature.REBUILD not in features

        # Per-entity scale editing (FileMorph only; primitives carry size/radius/height instead).
        to_remove: str | None = None
        for name, kwargs in self._pending_entities_kwargs.items():
            morph = kwargs["morph"]
            morph_name = type(morph).__name__
            file_name = morph.file if isinstance(morph, gs.morphs.FileMorph) else ""

            imgui.text_wrapped(f"{name} ({morph_name}): {file_name or '(builtin)'}")

            if isinstance(morph, gs.morphs.FileMorph):
                scale = morph.scale
                scale_val = float(scale[0]) if isinstance(scale, (list, tuple, np.ndarray)) else float(scale)
                imgui.begin_disabled(scale_disabled)
                changed, new_scale = imgui.drag_float(f"Scale##scale_{name}", scale_val, 0.01, 0.01, 100.0, "%.3f")
                imgui.end_disabled()
                self._maybe_show_disabled_tooltip(scale_disabled)
                if changed and not scale_disabled:
                    morph.scale = new_scale
                    self._pending_dirty = True

                imgui.same_line()

            imgui.begin_disabled(remove_disabled)
            remove_clicked = imgui.button(f"X##remove_{name}")
            imgui.end_disabled()
            self._maybe_show_disabled_tooltip(remove_disabled)
            if remove_clicked and not remove_disabled:
                to_remove = name

            draw_separator(imgui)

        if to_remove is not None:
            del self._pending_entities_kwargs[to_remove]
            self._pending_dirty = True

        # Add entity / stage section
        imgui.begin_disabled(add_disabled)
        add_header_open = imgui.collapsing_header("Add Entity / Stage##add_entity")
        imgui.end_disabled()
        self._maybe_show_disabled_tooltip(add_disabled)
        if add_header_open:
            imgui.indent()
            imgui.begin_disabled(add_disabled)

            changed_type, self._add_entity_morph_type = imgui.combo(
                "Type##add_type", self._add_entity_morph_type, _ADD_ENTITY_TYPES
            )
            morph_type = _ADD_ENTITY_TYPES[self._add_entity_morph_type]
            is_file = morph_type == "File"
            # Default fixed=True for Plane when the type changes (a Plane is always grounded).
            if changed_type and morph_type == "Plane":
                self._add_entity_fixed = True

            # The "File" entry takes only a path; the concrete morph (URDF / MJCF / Mesh / USD) is deduced from the
            # extension, exactly as `gs launch` selects a morph from the filename it is given.
            morph_cls = None
            if is_file:
                _, self._add_entity_file = imgui.input_text("File##add_file", self._add_entity_file, 256)
                imgui.same_line()
                if imgui.button("Browse##add_browse") and not add_disabled:
                    self._file_browser_open = True
                    self._file_browser_selected = -1
                    # Start browsing from current file's directory if set
                    if self._add_entity_file:
                        parent = os.path.dirname(self._add_entity_file)
                        if os.path.isdir(parent):
                            self._file_browser_dir = parent
                self._render_file_browser()

                morph_cls = self._MORPH_BY_EXTENSION.get(os.path.splitext(self._add_entity_file)[1].lower())
                if self._add_entity_file and morph_cls is None:
                    imgui.text_colored((1.0, 0.4, 0.4, 1.0), "Unsupported file type")

                _, self._add_entity_scale = imgui.drag_float(
                    "Scale##add_scale", self._add_entity_scale, 0.01, 0.01, 100.0, "%.3f"
                )
            elif morph_type == "Box":
                _, self._add_box_size = imgui.drag_float3(
                    "Size##add_box_size", self._add_box_size, 0.01, 0.01, 100.0, "%.3f"
                )
            elif morph_type == "Sphere":
                _, self._add_sphere_radius = imgui.drag_float(
                    "Radius##add_sphere_r", self._add_sphere_radius, 0.01, 0.01, 100.0, "%.3f"
                )
            elif morph_type == "Cylinder":
                _, self._add_cylinder_radius = imgui.drag_float(
                    "Radius##add_cyl_r", self._add_cylinder_radius, 0.01, 0.01, 100.0, "%.3f"
                )
                _, self._add_cylinder_height = imgui.drag_float(
                    "Height##add_cyl_h", self._add_cylinder_height, 0.01, 0.01, 100.0, "%.3f"
                )

            # Position (all types except Plane, which is always at the origin).
            if morph_type != "Plane":
                _, self._add_entity_pos = imgui.drag_float3(
                    "Position##add_pos", self._add_entity_pos, 0.05, -100.0, 100.0, "%.2f"
                )

            # Fixed checkbox. An MJCF defines its own base joint (free or welded), so the toggle has no effect and is
            # greyed out for it. The collision-mesh processing toggles sit alongside for file morphs: convexify
            # replaces collision meshes with their convex hull and decimate reduces their face count (both default on).
            fixed_disabled = morph_cls is gs.morphs.MJCF
            imgui.begin_disabled(fixed_disabled)
            _, self._add_entity_fixed = imgui.checkbox("Fixed##add_fixed", self._add_entity_fixed)
            imgui.end_disabled()
            if fixed_disabled and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                imgui.set_tooltip("An MJCF defines its own base joint")
            if is_file:
                imgui.same_line()
                _, self._add_convexify = imgui.checkbox("Convexify##add_convexify", self._add_convexify)
                imgui.same_line()
                _, self._add_decimate = imgui.checkbox("Decimate##add_decimate", self._add_decimate)

            # A File type stays un-addable until its path resolves to a known morph.
            add_invalid = is_file and morph_cls is None
            imgui.begin_disabled(add_invalid)
            add_clicked = imgui.button("Add##add_btn")
            imgui.end_disabled()
            imgui.end_disabled()
            self._maybe_show_disabled_tooltip(add_disabled)
            if add_clicked and not add_disabled and not add_invalid:
                pos = tuple(self._add_entity_pos)
                fixed = self._add_entity_fixed
                if is_file:
                    morph_kwargs = dict(
                        file=self._add_entity_file,
                        pos=pos,
                        scale=self._add_entity_scale,
                        convexify=self._add_convexify,
                        decimate=self._add_decimate,
                    )
                    # MJCF has no 'fixed' parameter; its base joint comes from the file.
                    if morph_cls is not gs.morphs.MJCF:
                        morph_kwargs["fixed"] = fixed
                    new_morph = morph_cls(**morph_kwargs)
                    base_name = morph_cls.__name__
                elif morph_type == "Box":
                    new_morph = gs.morphs.Box(pos=pos, size=tuple(self._add_box_size), fixed=fixed)
                    base_name = "Box"
                elif morph_type == "Sphere":
                    new_morph = gs.morphs.Sphere(pos=pos, radius=self._add_sphere_radius, fixed=fixed)
                    base_name = "Sphere"
                elif morph_type == "Cylinder":
                    new_morph = gs.morphs.Cylinder(
                        pos=pos, radius=self._add_cylinder_radius, height=self._add_cylinder_height, fixed=fixed
                    )
                    base_name = "Cylinder"
                else:
                    new_morph = gs.morphs.Plane()
                    base_name = "Plane"
                # Generate a unique name based on the morph type.
                suffix = 0
                name = base_name
                while name in self._pending_entities_kwargs:
                    suffix += 1
                    name = f"{base_name}_{suffix}"
                self._pending_entities_kwargs[name] = {
                    "morph": new_morph,
                    "material": None,
                    "surface": None,
                    "visualize_contact": False,
                }
                self._pending_dirty = True
            imgui.unindent()

        # Rebuild button. The InteractiveScene performs the actual rebuild on the stepping thread; doing it here on the
        # viewer thread would destroy the OpenGL context we are rendering from.
        if self._pending_dirty:
            imgui.text_colored((1.0, 0.7, 0.0, 1.0), "Changes pending")
        imgui.begin_disabled(rebuild_disabled)
        rebuild_clicked = imgui.button("Rebuild Scene", size=(150, 0))
        imgui.end_disabled()
        self._maybe_show_disabled_tooltip(rebuild_disabled)
        if rebuild_clicked and not rebuild_disabled:
            self.interactive_scene.rebuild(entities_kwargs=self._pending_entities_kwargs)
            self._pending_dirty = False

    def _render_entity_browser(self):
        """Render entity list with joint sliders."""
        imgui = self._imgui

        if not self._entity_cache:
            imgui.text("No controllable entities")
            return

        for entity_idx, data in self._entity_cache.items():
            entity = data.entity
            # Tag each entry with the entity class (e.g. <RigidEntity>, <KinematicEntity>) so the type is visible,
            # matching the <ClassName> style used by the option reprs. Controls unavailable for kinematic entities are
            # disabled rather than removed, keeping the panel layout consistent across entity types.
            is_rigid = isinstance(entity, gs.engine.entities.RigidEntity)
            expanded = imgui.collapsing_header(
                f"{data.name}  <{type(entity).__name__}>##entity_{entity_idx}",
                flags=imgui.TreeNodeFlags_.default_open,
            )
            if not expanded:
                continue

            imgui.indent()

            # DOF count display
            imgui.text(f"DOFs: {data.n_dofs}")

            # Vis mode combo. The collision item is greyed out unless the entity has collision geometry (kinematic
            # entities have none, and rigid entities can be loaded without it), so an empty mode is never selectable.
            has_collision = is_rigid and len(entity.geoms) > 0
            current_mode = entity.surface.vis_mode
            if imgui.begin_combo(f"Vis Mode##vis_{entity_idx}", current_mode):
                for mode in ("visual", "collision"):
                    disabled = mode == "collision" and not has_collision
                    imgui.begin_disabled(disabled)
                    clicked = imgui.selectable(mode, current_mode == mode)[0]
                    imgui.end_disabled()
                    if disabled and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                        imgui.set_tooltip("No collision geometry available for this entity")
                    if clicked and mode != current_mode:
                        self._apply_entity_vis_mode(entity, mode)
                imgui.end_combo()

            # Per-entity wireframe toggle. Material lives on render primitives, so we walk the active geom set (vgeoms
            # for visual mode, geoms otherwise) and flip the flag on each.
            is_wireframe = self._wireframe_state.get(entity_idx, False)
            changed_wf, new_wf = imgui.checkbox(f"Wireframe##wf_{entity_idx}", is_wireframe)
            if changed_wf:
                self._wireframe_state[entity_idx] = new_wf
                ctx = self.viewer.gs_context
                geoms = entity.vgeoms if entity.surface.vis_mode == "visual" else entity.geoms
                for geom in geoms:
                    node = ctx.rigid_nodes.get(geom.uid)
                    if node is None:
                        continue
                    for primitive in node.mesh.primitives:
                        if primitive.material is not None:
                            primitive.material.wireframe = new_wf
                ctx._scene._meshes_updated = True

            # Visualize contact toggle. Contacts only exist for dynamically simulated rigid entities; for kinematic
            # entities the control is disabled and reads False, since they carry no visualize_contact state.
            show_contact = entity.visualize_contact if is_rigid else False
            imgui.begin_disabled(not is_rigid)
            changed_contact, new_contact = imgui.checkbox(f"Show Contacts##contact_{entity_idx}", show_contact)
            imgui.end_disabled()
            if not is_rigid and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                imgui.set_tooltip("Only available for rigid entities")
            if changed_contact:
                entity._visualize_contact = new_contact
                for link in entity.links:
                    link._visualize_contact = new_contact

            # Gizmo toggle to translate/rotate the entity's base link (fixed or floating base).
            if self._gizmo is not None:
                gizmo_active = self._gizmo_entity_idx == entity_idx
                changed_gizmo, new_gizmo = imgui.checkbox(f"Gizmo##gizmo_{entity_idx}", gizmo_active)
                if changed_gizmo:
                    self._gizmo_entity_idx = entity_idx if new_gizmo else -1
                if gizmo_active:
                    imgui.same_line()
                    gizmo = self._gizmo
                    if imgui.radio_button(
                        f"Translate##gop_{entity_idx}", self._gizmo_operation == gizmo.OPERATION.translate
                    ):
                        self._gizmo_operation = gizmo.OPERATION.translate
                    imgui.same_line()
                    if imgui.radio_button(f"Rotate##gop_{entity_idx}", self._gizmo_operation == gizmo.OPERATION.rotate):
                        self._gizmo_operation = gizmo.OPERATION.rotate

            # Joint sections only for entities with DOFs
            if data.n_dofs > 0:
                qpos = tensor_to_array(entity.get_qpos())
                if self._controlled_env_idx is not None:
                    qpos = qpos[self._controlled_env_idx]

                changed_any = False
                new_qpos = list(qpos)

                # Joint control section
                if imgui.collapsing_header(f"Joint Control##joints_{entity_idx}"):
                    imgui.indent()

                    # Euler/Quat toggle for free-joint entities
                    use_euler = False
                    if data.joint_data.has_free_joint:
                        rot_mode = self._rotation_mode.get(entity_idx, "quat")
                        if imgui.radio_button(f"Quaternion##rotmode_{entity_idx}", rot_mode == "quat"):
                            self._rotation_mode[entity_idx] = "quat"
                            rot_mode = "quat"
                        imgui.same_line()
                        if imgui.radio_button(f"Euler##rotmode_{entity_idx}", rot_mode == "euler"):
                            self._rotation_mode[entity_idx] = "euler"
                            rot_mode = "euler"
                        use_euler = rot_mode == "euler"

                    if use_euler:
                        # Euler mode: show position + euler angles from get_dofs_position
                        changed_any = self._render_joints_euler_mode(entity, data, entity_idx, qpos, new_qpos)
                    else:
                        # Quat mode: show all qpos components
                        lower, upper = data.joint_data.q_limits
                        for i, (name, val, lo, hi, is_quat) in enumerate(
                            zip(data.joint_data.q_names, qpos, lower, upper, data.joint_data.q_is_quaternion)
                        ):
                            if is_quat:
                                changed, new_val = imgui.drag_float(
                                    f"{name}##{entity_idx}_{i}", float(val), 0.01, float(lo), float(hi), "%.4f"
                                )
                            else:
                                changed, new_val = imgui.slider_float(
                                    f"{name}##{entity_idx}_{i}", float(val), float(lo), float(hi), "%.3f"
                                )
                            if changed:
                                new_qpos[i] = new_val
                                changed_any = True
                    imgui.unindent()

                if changed_any:
                    # Auto-pause when user edits joints
                    self.interactive_scene.pause()
                    if not (data.joint_data.has_free_joint and self._rotation_mode.get(entity_idx) == "euler"):
                        # Normalize any edited quaternion groups (quat mode only)
                        for qstart, qend in data.joint_data.quat_groups:
                            q = np.array(new_qpos[qstart:qend])
                            norm = np.linalg.norm(q)
                            if norm > 1e-8:
                                q /= norm
                                new_qpos[qstart:qend] = q.tolist()
                    with self.viewer.render_lock:
                        entity.set_qpos(new_qpos, envs_idx=self._controlled_env_idx)
                        self._refresh_visuals()

            imgui.unindent()

    def _render_joints_euler_mode(self, entity, data, entity_idx, qpos, new_qpos):
        """Render free joint as position + euler angles, plus remaining joints normally.

        Free joint edits are applied immediately via set_dofs_position.
        Non-free joint edits update new_qpos for the caller to apply.
        Returns True if any non-free-joint value changed (needing set_qpos).
        """
        imgui = self._imgui
        non_free_changed = False
        qs = data.joint_data.free_joint_q_start

        # Get dofs_position for euler angles
        dofs = tensor_to_array(entity.get_dofs_position())
        if self._controlled_env_idx is not None:
            dofs = dofs[self._controlled_env_idx]

        # Position (first 3 dofs = same as first 3 qpos for free joint)
        pos = [float(dofs[0]), float(dofs[1]), float(dofs[2])]
        changed_pos, new_pos = imgui.drag_float3(
            f"Position##euler_pos_{entity_idx}",
            pos,
            0.05,
            -self._free_joint_pos_limit,
            self._free_joint_pos_limit,
            "%.3f",
        )

        # Euler angles (dofs 3-5, in radians, display as degrees)
        euler_rad = [float(dofs[3]), float(dofs[4]), float(dofs[5])]
        euler_deg = [np.degrees(e) for e in euler_rad]
        changed_rot, new_euler_deg = imgui.drag_float3(
            f"Euler (deg)##euler_rot_{entity_idx}", euler_deg, 0.5, -360.0, 360.0, "%.1f"
        )

        if changed_pos or changed_rot:
            self.interactive_scene.pause()
            new_dofs = list(dofs)
            if changed_pos:
                new_dofs[0], new_dofs[1], new_dofs[2] = new_pos
            if changed_rot:
                new_dofs[3] = np.radians(new_euler_deg[0])
                new_dofs[4] = np.radians(new_euler_deg[1])
                new_dofs[5] = np.radians(new_euler_deg[2])

            # Use set_dofs_position for the whole entity (handles euler->quat internally)
            with self.viewer.render_lock:
                entity.set_dofs_position(new_dofs, envs_idx=self._controlled_env_idx)
                self._refresh_visuals()

            # Refresh new_qpos with updated free joint qpos (euler->quat conversion happened)
            fresh = tensor_to_array(entity.get_qpos())
            if self._controlled_env_idx is not None:
                fresh = fresh[self._controlled_env_idx]
            for i in range(qs, qs + 7):
                new_qpos[i] = float(fresh[i])

        # Render remaining (non-free) joints normally
        lower, upper = data.joint_data.q_limits
        free_end = qs + 7  # free joint takes 7 qpos slots
        for i, (name, val, lo, hi, is_quat) in enumerate(
            zip(data.joint_data.q_names, qpos, lower, upper, data.joint_data.q_is_quaternion)
        ):
            if qs <= i < free_end:
                continue  # Skip free joint components (handled above)
            if is_quat:
                changed, new_val = imgui.drag_float(
                    f"{name}##{entity_idx}_{i}", float(val), 0.01, float(lo), float(hi), "%.4f"
                )
            else:
                changed, new_val = imgui.slider_float(
                    f"{name}##{entity_idx}_{i}", float(val), float(lo), float(hi), "%.3f"
                )
            if changed:
                new_qpos[i] = new_val
                non_free_changed = True

        return non_free_changed

    def on_close(self) -> None:
        """Clean up ImGui resources. Idempotent: the viewer dispatches close on both the window-close event
        and scene teardown, so guard against a second call once the context is already destroyed."""
        if self._ctx is None:
            return
        # Make our context current before tearing it down; the backend shutdown and destroy_context both operate on the
        # current ImGui context.
        self._imgui.set_current_context(self._ctx)
        if self._available and self._impl is not None:
            self._impl.shutdown()
        self._imgui.destroy_context(self._ctx)
        self._ctx = None
        self._impl = None
        self._available = False
        self._init_attempted = False
