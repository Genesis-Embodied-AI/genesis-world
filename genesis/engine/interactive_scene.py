import contextlib
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Iterable

import genesis as gs
from genesis.utils.misc import with_lock
from genesis.vis.viewer_plugins.plugins import DefaultControlsPlugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.scene import Scene
    from genesis.options.sensors.options import SensorOptions


class InteractiveFeature(Enum):
    """Editing capabilities an InteractiveScene may expose to interactive frontends (e.g. the ImGui
    overlay). A frontend queries InteractiveScene.supported_features and enables the matching controls,
    without needing to know which simulator modes each feature requires."""

    SCALE_ENTITY = auto()
    ADD_ENTITY = auto()
    REMOVE_ENTITY = auto()
    REBUILD = auto()


class InteractiveScene:
    """
    Wraps a built Scene and behaves like an asynchronous scene with extra editing features. Beyond mirroring
    the scene, it offers ``pause()`` / ``resume()`` / ``step()`` / ``rebuild()``. These are not honored on the
    spot: they record intent that is applied at the start of the next underlying ``step()``, on the stepping
    thread. This lets any view (e.g. the ImGui overlay) drive the scene from any thread without performing
    main-thread-only work itself - it just calls these methods.

    ``rebuild()`` reconstructs the wrapped Scene in place - destroying and re-creating its solvers, viewer and
    entities on the same object - so external references to the Scene and its viewer stay valid across edits.

    Scene editing is unavailable in performance mode (gs.init(performance_mode=True)), where a rebuild would
    trigger systematic kernel recompilation; supported_features is then empty and rebuild() raises.
    """

    def __init__(self, scene: "Scene"):
        self._scene: "Scene" = scene
        # Pending interactive intent, applied at the next underlying step (see _pre_step).
        self._paused: bool = False
        self._pending_steps: int = 0
        self._rebuild_pending: bool = False
        self._reset_pending: bool = False
        self._entities_kwargs: dict[str, dict[str, Any]] = {}
        self._sensors_kwargs: list["SensorOptions"] = []
        scene.register_pre_step_callback(self._pre_step)
        # Capture the wrapped scene's construction so rebuild() can reconstruct it identically. The stored options
        # are already merged with the simulation options, so passing them again changes nothing.
        self._scene_kwargs: dict[str, Any] = dict(
            options=scene.options,
            show_viewer=scene.viewer is not None,
        )
        self._build_kwargs: dict[str, Any] = dict(
            n_envs=scene.n_envs,
            env_spacing=scene.env_spacing,
            n_envs_per_row=scene.n_envs_per_row,
        )

    @property
    def supported_features(self) -> frozenset[InteractiveFeature]:
        """Set of editing features available for the current simulator mode. Empty in performance mode
        since every editing operation reconstructs the scene through rebuild()."""
        if not gs.use_ndarray:
            return frozenset()
        return frozenset(InteractiveFeature)

    @property
    def paused(self) -> bool:
        """Whether the simulation is currently held paused (see ``pause()`` / ``resume()``)."""
        return self._paused

    def pause(self) -> None:
        """Hold the simulation: subsequent underlying steps do not advance until ``resume()`` or ``step()``."""
        self._paused = True

    def resume(self) -> None:
        """Let the simulation advance again on subsequent underlying steps."""
        self._paused = False

    def step(self, n: int = 1) -> None:
        """Advance ``n`` frames even while paused. Asynchronous: the frames are consumed by the next ``n``
        underlying steps rather than executed here."""
        self._pending_steps += n

    @property
    def recording(self) -> bool:
        """Whether the viewer is currently recording its on-screen output to a video file."""
        return self.viewer.recording

    def toggle_recording(self) -> bool:
        """Start or stop recording the viewer's on-screen output, returning the resulting record state.

        Unlike pause/step/reset this takes effect immediately (it is a viewer-thread action, not a queued step
        intent); stopping prompts for a destination file."""
        return self.viewer.toggle_recording()

    def _pre_step(self) -> bool:
        """Scene pre-step callback that runs on the stepping thread.

        Applies the queued intent and returns True to skip the advance this frame: after performing a pending
        rebuild or reset, or while paused with no queued steps."""
        if self._rebuild_pending:
            self._rebuild_pending = False
            self._apply_rebuild()
            return True
        if self._reset_pending:
            self._reset_pending = False
            self._apply_reset()
            return True
        if self._pending_steps > 0:
            self._pending_steps -= 1
            return False
        return self._paused

    @property
    def scene(self) -> "Scene":
        return self._scene

    @property
    def viewer(self):
        return self.scene.viewer

    @property
    def entities(self):
        return self.scene.entities

    @property
    def rigid_solver(self):
        return self.scene.rigid_solver

    @property
    def n_envs(self) -> int:
        return self.scene.n_envs

    @property
    def t(self) -> int:
        return self.scene.sim.cur_step_global

    @property
    def dt(self) -> float:
        return self.scene.sim.dt

    @property
    def is_built(self) -> bool:
        return self._scene is not None and self._scene.is_built

    @property
    def _lock(self):
        return self.scene.viewer.lock

    @property
    def _ctx(self):
        return self.scene.viewer.context

    @with_lock
    def refresh_visual_transforms(self):
        """Refresh the visual geom poses and the drawn nodes so visuals reflect the latest qpos. Idempotent."""
        self._refresh_visual_transforms_unlocked()

    def _refresh_visual_transforms_unlocked(self):
        rigid_solver = self.scene.rigid_solver
        if not rigid_solver.is_active:
            return
        rigid_solver.update_vgeoms()
        ctx = self._ctx
        ctx.update_link_frame()
        ctx.update_rigid()

    def reset(self):
        """Queue a reset of the wrapped scene to its initial state.

        Asynchronous: it is applied at the start of the next underlying step(), on the stepping thread, so it is
        safe to call from a viewer-thread GUI callback. Applying it directly would re-enter the viewer refresh (and
        thus the ImGui frame) from within the current frame."""
        self._reset_pending = True

    @with_lock
    def _apply_reset(self):
        """Reset the wrapped scene and refresh visuals, on the stepping thread.

        Clears contact arrows and other transient render nodes."""
        self.scene.reset()
        self._ctx.clear_dynamic_nodes(only_outdated=False)
        self._refresh_visual_transforms_unlocked()

    @with_lock
    def set_entity_qpos(self, entity: "RigidEntity", qpos, env_idx: int | None = None):
        """Set the entity's qpos and refresh visuals."""
        entity.set_qpos(qpos, envs_idx=env_idx)
        self._refresh_visual_transforms_unlocked()

    @with_lock
    def set_entity_dofs_position(self, entity: "RigidEntity", dofs_position, env_idx: int | None = None):
        """Set the entity's DOF positions and refresh visuals."""
        entity.set_dofs_position(dofs_position, envs_idx=env_idx)
        self._refresh_visual_transforms_unlocked()

    @with_lock
    def set_entity_vis_mode(self, entity: "RigidEntity", mode: str):
        """Switch entity rendering between ``"visual"`` and ``"collision"``."""
        if not isinstance(entity.surface, gs.surfaces.Surface):
            return
        old_mode = entity.surface.vis_mode
        if old_mode == mode:
            return

        ctx = self._ctx
        rigid_solver = self.scene.rigid_solver

        old_geoms = entity.vgeoms if old_mode == "visual" else entity.geoms
        for geom in old_geoms:
            ctx.remove_rigid_node(geom)

        entity.surface.vis_mode = mode
        rigid_solver.update_vgeoms()

        is_visual = mode == "visual"
        geoms = entity.vgeoms if is_visual else entity.geoms
        geoms_T = ctx.rigid_geoms_T(rigid_solver, is_visual)
        for geom in geoms:
            ctx.add_geom_node(geom, geoms_T)

    def rebuild(
        self,
        *,
        scene_kwargs: dict[str, Any] | None = None,
        entities_kwargs: dict[str, dict[str, Any]] | None = None,
        sensors_kwargs: Iterable["SensorOptions"] | None = None,
    ):
        """
        Queue an in-place reconstruction of the wrapped scene. Asynchronous: it is applied at the start of the
        next underlying step(), on the stepping thread, so it is safe to call from any thread (e.g. a
        viewer-thread GUI callback). The same Scene object is torn down and re-created with the same viewer
        window, so external ``scene`` / ``scene.viewer`` handles stay valid.

        Any argument left as ``None`` reuses what was supplied previously. Pass an empty ``dict`` / iterable to
        explicitly clear stored state.

        Args:
            scene_kwargs: Keyword arguments forwarded to ``gs.Scene(...)`` (sim_options, viewer_options, etc.).
            entities_kwargs: Mapping from entity name to a kwargs dict forwarded to ``scene.add_entity``
                (morph, material, surface, visualize_contact, vis_mode). The dict key becomes the entity's name.
            sensors_kwargs: Iterable of ``SensorOptions`` instances forwarded to ``scene.add_sensor``.
        """
        if InteractiveFeature.REBUILD not in self.supported_features:
            gs.raise_exception(
                "InteractiveScene.rebuild() is not supported in performance mode "
                "(gs.init(performance_mode=True)) since it would trigger systematic kernel recompilation."
            )
        if scene_kwargs is not None:
            self._scene_kwargs = dict(scene_kwargs)
        if entities_kwargs is not None:
            self._entities_kwargs = dict(entities_kwargs)
        if sensors_kwargs is not None:
            self._sensors_kwargs = list(sensors_kwargs)
        self._rebuild_pending = True

    def _apply_rebuild(self) -> None:
        """Reconstruct the wrapped scene in place from the stored setup, on the stepping thread. Preserves the
        viewer window, re-registers the pre-step callback, re-attaches non-default plugins and restores the
        camera pose."""
        scene = self._scene
        cam_pose = None
        plugins_to_reattach: list = []
        pyrender_window = None

        if scene.viewer is not None:
            viewer = scene.viewer
            # Capture the full 4x4 camera pose (position, lookat and roll/up), not just pos + lookat, so the user's
            # current viewpoint is restored exactly across the rebuild instead of resetting the roll.
            cam_pose = viewer.camera_pose.copy()
            # Skip default plugins; the rebuilt viewer recreates them based on its ViewerOptions.
            plugins_to_reattach = [p for p in viewer.plugins if not isinstance(p, DefaultControlsPlugin)]
            # Preserve the live window/GL context so the rebuild does not close and reopen it.
            pyrender_window = viewer._pyrender_viewer

        # Serialize against a threaded render loop (run_in_thread=True): holding the preserved window's render_lock
        # blocks on_draw so it never draws the scene while it is being torn down, rebuilt and re-pointed. No-op when
        # there is no window (headless) or the viewer runs on the main thread.
        with pyrender_window.render_lock if pyrender_window is not None else contextlib.nullcontext():
            if pyrender_window is not None:
                # Detach so scene.destroy() does not close the preserved window.
                scene.viewer._pyrender_viewer = None
            scene.destroy()
            # Re-initialize the SAME object in place so external references survive the rebuild.
            scene.__init__(**self._scene_kwargs)
            # Re-register the pre-step callback: the in-place re-init cleared Scene's callback list.
            scene.register_pre_step_callback(self._pre_step)
            for name, kwargs in self._entities_kwargs.items():
                # A USD morph describes a whole stage (potentially many bodies); add_stage parses and adds them and
                # takes no name. Every other morph is a single entity added by name.
                if isinstance(kwargs["morph"], gs.morphs.USD):
                    scene.add_stage(**kwargs)
                else:
                    scene.add_entity(name=name, **kwargs)
            for sensor_opts in self._sensors_kwargs:
                scene.add_sensor(sensor_opts)
            # Hand the preserved window to the new viewer so build() reuses it instead of opening a new one.
            if pyrender_window is not None and scene.viewer is not None:
                scene.viewer._pyrender_viewer = pyrender_window
            scene.build(**self._build_kwargs)

            new_viewer = scene.viewer
            if new_viewer is not None:
                # A scene built with enable_gui=True auto-attaches its own ImGui overlay. When re-attaching the previous
                # overlay (which carries user state - panel width, custom panels, pending edits), drop the fresh
                # auto-attached one of the same type so the viewer does not end up with two overlays.
                reattach_types = {type(p) for p in plugins_to_reattach}
                for plugin in [p for p in new_viewer.plugins if type(p) in reattach_types]:
                    new_viewer.remove_plugin(plugin)

                for plugin in plugins_to_reattach:
                    new_viewer.add_plugin(plugin)
                if cam_pose is not None:
                    new_viewer.set_camera_pose(pose=cam_pose)
