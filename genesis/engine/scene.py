import collections.abc
import dataclasses
import hashlib
import io
import json
import math
import os
import sys
import weakref
import zipfile
from collections import Counter
from typing import BinaryIO, Callable, Iterable, Literal, NamedTuple, TYPE_CHECKING, overload

import numpy as np
import torch

import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.base_entity import Entity, EntityDescription
from genesis.engine.entities.rigid_entity import KinematicEntity
from genesis.engine.force_fields import ForceField
from genesis.engine.materials.base import EntityT, Material
from genesis.engine.states.solvers import SimState, SimulatorCheckpoint
from genesis.options import (
    SceneOptions,
    BaseCouplerOptions,
    FEMOptions,
    KinematicOptions,
    MPMOptions,
    PBDOptions,
    ProfilingOptions,
    RigidOptions,
    SFOptions,
    SimOptions,
    SPHOptions,
    ToolOptions,
    ViewerOptions,
    VisOptions,
)
from genesis.options.morphs import Morph
from genesis.options.recorders import RecorderOptions, TrajectoryFile
from genesis.options.renderers import RendererOptions
from genesis.options.surfaces import Surface
from genesis.recorders import RecorderManager
from genesis.recorders.trajectory import Trajectory, save_checkpoint
from genesis.repr_base import RBC
from genesis.utils import serialization
from genesis.utils.array_class import check_data
from genesis.utils.misc import sanitize_index, tensor_to_array
from genesis.utils.serialization import ARRAY_MEMBER, MANIFEST_NAME, pixel_less_textures
from genesis.utils.tools import FPSTracker
from genesis.utils.warnings import warn_once
from genesis.vis import Visualizer

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.sensors.base_sensor import Sensor
    from genesis.engine.simulator import Simulator
    from genesis.options.sensors.options import SensorOptions, SensorT
    from genesis.recorders import Recorder


SCENE_FORMAT = ".gscene"


@dataclasses.dataclass
class SceneDescription:
    """Describe one scene as it was authored: the options it was created with, and every entity added to it.

    Each entity stands as its own description (see 'EntityDescription'), so a scene is created from this alone. A
    scene created from one is built by whoever loads it, with the environment layout they ask for.
    """

    options: SceneOptions
    entities: list[EntityDescription] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class EnvironmentLayout:
    """The environment layout given to 'Scene.build'. It sizes every array of the state."""

    n_envs: int
    env_spacing: tuple[float, float]
    n_envs_per_row: int
    center_envs_at_origin: bool


@dataclasses.dataclass
class SceneCheckpoint:
    """The checkpoint of a built scene, which 'Scene.__getstate__' returns and a pickled scene serializes."""

    scene: SceneDescription
    digest: str
    layout: EnvironmentLayout
    sim: SimulatorCheckpoint


class TrajectorySource(NamedTuple):
    """The simulator, scene description and environment layout a trajectory recorder reads at build."""

    sim: "Simulator"
    desc: SceneDescription
    layout: EnvironmentLayout


def description_digest(desc: SceneDescription) -> str:
    """Digest what a description states of the physics: its entities and every option group but the visual ones, which
    'Scene.load' replaces. Two scenes sharing the digest allocate the same state, so a record of one restores into the
    other. The Genesis version and sources are left out, so a record survives the code moving on.

    The digest exports the description in memory, meshes included, so it is taken once per built scene and once per
    trajectory opened, never per restore.
    """
    defaults = SceneOptions()
    visual = {
        "vis": defaults.vis,
        "viewer": defaults.viewer,
        "renderer": defaults.renderer,
        "profiling": defaults.profiling,
    }
    buffer = io.BytesIO()
    serialization.export(buffer, {"scene": dataclasses.replace(desc, options=desc.options.model_copy(update=visual))})
    with zipfile.ZipFile(buffer) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME))
        arrays = np.load(io.BytesIO(archive.read(ARRAY_MEMBER)))
        digest = hashlib.sha256(json.dumps((manifest["schema"], manifest["held"]), sort_keys=True).encode())
        for name in sorted(arrays):
            digest.update(f"{name}{arrays[name].shape}{arrays[name].dtype.str}".encode())
            digest.update(arrays[name].tobytes())
    return digest.hexdigest()


@gs.assert_initialized
class Scene(RBC):
    """
    A ``genesis.Scene`` object wraps all components in a simulation environment, including a simulator (containing multiple physics solvers), entities, and a visualizer (controlling both the viewer and all the cameras).
    Basically, everything happens inside a scene.

    Parameters
    ----------
    sim_options : gs.options.SimOptions
        The options configuring the overarching `simulator`, which in turn manages all the solvers.
    tool_options : gs.options.ToolOptions
        The options configuring the tool_solver (``scene.sim.ToolSolver``).
    rigid_options : gs.options.RigidOptions
        The options configuring the rigid_solver (``scene.sim.RigidSolver``).
    mpm_options : gs.options.MPMOptions
        The options configuring the mpm_solver (``scene.sim.MPMSolver``).
    sph_options : gs.options.SPHOptions
        The options configuring the sph_solver (``scene.sim.SPHSolver``).
    fem_options : gs.options.FEMOptions
        The options configuring the fem_solver (``scene.sim.FEMSolver``).
    sf_options : gs.options.SFOptions
        The options configuring the sf_solver (``scene.sim.SFSolver``).
    pbd_options : gs.options.PBDOptions
        The options configuring the pbd_solver (``scene.sim.PBDSolver``).
    coupler_options : gs.options.CouplerOptions
        The options configuring the `coupler` between different solvers.
    vis_options : gs.options.VisOptions
        The options configuring the visualization system (``scene.visualizer``). Visualizer controls both the interactive viewer and the cameras.
    viewer_options : gs.options.ViewerOptions
        The options configuring the viewer (``scene.visualizer.viewer``).
    renderer : gs.renderers.RendererOptions
        The renderer options used by `camera` for rendering images. This doesn't affect the behavior of the interactive viewer.
    show_viewer : bool
        Whether to show the interactive viewer. Set it to False if you only need headless rendering.
    show_FPS : bool
        Whether to show the FPS in the terminal.
    options : SceneOptions
        Every option above as one object, which is how a scene holds them. It is given alone, and passing another
        scene's ``options`` here creates a scene from what that scene was created with.
    """

    def __init__(
        self,
        sim_options: SimOptions | None = None,
        tool_options: ToolOptions | None = None,
        rigid_options: RigidOptions | None = None,
        kinematic_options: KinematicOptions | None = None,
        mpm_options: MPMOptions | None = None,
        sph_options: SPHOptions | None = None,
        fem_options: FEMOptions | None = None,
        sf_options: SFOptions | None = None,
        pbd_options: PBDOptions | None = None,
        coupler_options: BaseCouplerOptions | None = None,
        vis_options: VisOptions | None = None,
        viewer_options: ViewerOptions | None = None,
        profiling_options: ProfilingOptions | None = None,
        renderer: RendererOptions | None = None,
        show_viewer: bool | None = None,
        show_FPS: bool | None = None,  # deprecated, use Scene.options.profiling.show_FPS instead
        options: SceneOptions | None = None,
    ):
        # Delay simulator import to allow specifying Quadrants array type at init
        from genesis.engine.simulator import Simulator

        individual_options = (
            sim_options,
            tool_options,
            rigid_options,
            kinematic_options,
            mpm_options,
            sph_options,
            fem_options,
            sf_options,
            pbd_options,
            coupler_options,
            vis_options,
            viewer_options,
            profiling_options,
            renderer,
        )
        if options is not None:
            if any(option is not None for option in individual_options):
                gs.raise_exception("`options` holds every option a scene is created with, so it is given alone.")
            self.options = options
        else:
            self.options = SceneOptions(
                sim=sim_options,
                tool=tool_options,
                rigid=rigid_options,
                kinematic=kinematic_options,
                mpm=mpm_options,
                sph=sph_options,
                fem=fem_options,
                sf=sf_options,
                pbd=pbd_options,
                coupler=coupler_options,
                vis=vis_options,
                viewer=viewer_options,
                profiling=profiling_options,
                renderer=renderer,
            )
        if show_FPS is not None:
            warn_once("Scene.show_FPS is deprecated. Please use Scene.options.profiling.show_FPS")
            self.options.profiling.show_FPS = show_FPS

        # description
        self._desc = SceneDescription(options=self.options)

        # simulator
        self._sim = Simulator(scene=self, options=self.options)

        # visualizer
        self._visualizer = Visualizer(
            scene=self,
            show_viewer=show_viewer,
            vis_options=self.options.vis,
            viewer_options=self.options.viewer,
            renderer_options=self.options.renderer,
        )

        # recorders
        self._recorder_manager = RecorderManager(self._sim.dt)

        # emitters
        self._emitters = gs.List()

        self._backward_ready = False
        self._forward_ready = False

        self._uid = gs.UID()
        self._is_built = False
        self._desc_digest: str | None = None
        self._pre_step_callbacks: list = []

        gs.logger.info(f"Scene ~~~<{self._uid}>~~~ created.")

    def __del__(self):
        self.destroy()

    def destroy(self):
        # Stop tracking this scene right away
        try:
            gs._scene_registry.remove(weakref.ref(self))
        except ValueError:
            # This scene may have been destroyed previously
            pass

        # The recorders are the first to go, and a failure they raise leaves nothing of the scene alive behind it
        try:
            if getattr(self, "_recorder_manager", None) is not None and self._recorder_manager.is_recording:
                self.stop_recording()
        finally:
            self._recorder_manager = None

            if getattr(self, "_visualizer", None) is not None:
                self._visualizer.destroy()
                self._visualizer = None

            if getattr(self, "_sim", None) is not None:
                self._sim.destroy()
                self._sim = None

    @overload
    def add_entity(
        self,
        morph: Morph | Iterable[Morph],
        material: None = ...,
        surface: Surface | None = ...,
        visualize_contact: bool = ...,
        vis_mode: str | None = ...,
        name: str | None = ...,
    ) -> "RigidEntity": ...

    @overload
    def add_entity(
        self,
        morph: Morph | Iterable[Morph],
        material: Material[EntityT] = ...,
        surface: Surface | None = ...,
        visualize_contact: bool = ...,
        vis_mode: str | None = ...,
        name: str | None = ...,
    ) -> EntityT: ...

    @gs.assert_unbuilt
    def add_entity(
        self,
        morph: Morph | Iterable[Morph],
        material: Material | None = None,
        surface: Surface | None = None,
        visualize_contact: bool = False,
        vis_mode: str | None = None,
        name: str | None = None,
    ) -> "Entity":
        """
        Add an entity to the scene.

        Parameters
        ----------
        morph : gs.morphs.Morph | list[gs.morphs.Morph]
            The morph of the entity. If a list of morphs is provided, the entity will be heterogeneous
            (rigid only, single-link entities only). Each parallel environment will simulate a different
            geometry variant from the list.
        material : gs.materials.Material | None, optional
            The material of the entity. If None, use ``gs.materials.Rigid()``.
        surface : gs.surfaces.Surface | None, optional
            The surface of the entity. If None, use ``gs.surfaces.Default()``.
        visualize_contact : bool
            Whether to visualize contact forces applied to this entity as arrows in the viewer and rendered images.
            Note that this will not be displayed in images rendered by camera using the `RayTracer` renderer.
        vis_mode : str | None, optional
            The visualization mode of the entity. This is a handy shortcut for setting `surface.vis_mode` without
            explicitly creating a surface object.
        name : str | None, optional
            User-specified name for the entity. If not provided, an auto-generated name will be assigned
            based on the morph type and entity UID (e.g., "box_a1b2c3d4"). Must be unique within the scene.

        Returns
        -------
        entity : genesis.Entity
            The created entity.
        """
        if material is None:
            material = gs.materials.Rigid()

        if surface is None:
            # assign a local surface, otherwise modification will apply on global default surface
            surface = gs.surfaces.Default()

        # Handle heterogeneous morphs (any iterable of morphs, excluding Morph objects). Whether a morph, or several,
        # can be built from is for the entity created from it to say.
        if isinstance(morph, collections.abc.Iterable) and not isinstance(morph, Morph):
            morph = tuple(morph)

        # some morph should not smooth surface normal
        if isinstance(
            morph[0] if isinstance(morph, tuple) else morph, (gs.morphs.Box, gs.morphs.Cylinder, gs.morphs.Terrain)
        ):
            surface.smooth = False

        if surface.double_sided is None:
            surface.double_sided = isinstance(material, (gs.materials.PBD.Cloth, gs.materials.FEM.Cloth))

        if vis_mode is not None:
            surface.vis_mode = vis_mode
        # validate and populate default surface.vis_mode considering morph type
        if isinstance(material, (gs.materials.Kinematic, gs.materials.Tool)):
            if surface.vis_mode is None:
                surface.vis_mode = "visual"

            if surface.vis_mode not in ("visual", "collision", "sdf"):
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: "
                    "['visual', 'collision', 'sdf']."
                )

        elif isinstance(
            material,
            (
                gs.materials.PBD.Liquid,
                gs.materials.PBD.Particle,
                gs.materials.MPM.Liquid,
                gs.materials.MPM.Sand,
                gs.materials.MPM.Snow,
                gs.materials.SPH.Liquid,
            ),
        ):
            if surface.vis_mode is None:
                surface.vis_mode = "particle"

            if surface.vis_mode not in ("particle", "recon"):
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['particle', 'recon']."
                )

        elif isinstance(material, gs.materials.SF.Smoke):
            if surface.vis_mode is None:
                surface.vis_mode = "particle"

            if surface.vis_mode not in ("particle",):
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['particle', 'recon']."
                )

        elif isinstance(material, (gs.materials.PBD.Base, gs.materials.MPM.Base, gs.materials.SPH.Base)):
            if surface.vis_mode is None:
                surface.vis_mode = "visual"

            if surface.vis_mode not in ("visual", "particle", "recon"):
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['visual', 'particle', 'recon']."
                )

        elif isinstance(material, gs.materials.FEM.Base):
            if surface.vis_mode is None:
                surface.vis_mode = "visual"

            if surface.vis_mode not in ("visual",):
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['visual']."
                )

        elif isinstance(material, gs.materials.Hybrid):  # determine the visual of the outer soft part
            if surface.vis_mode is None:
                surface.vis_mode = "particle"

            if surface.vis_mode not in ("particle", "visual"):
                gs.raise_exception(
                    f"Unsupported `surface.vis_mode` for material {material}: '{surface.vis_mode}'. Expected one of: ['particle', 'visual']."
                )

        else:
            gs.raise_exception()

        entity = self._sim._add_entity(morph, material, surface, visualize_contact, name)

        return entity

    @gs.assert_unbuilt
    def add_stage(
        self,
        morph: gs.morphs.USD,
        material: Material | None = None,
        surface: Surface | None = None,
        visualize_contact: bool = False,
        vis_mode: Literal["visual", "collision"] = "visual",
    ):
        """
        Add a stage to the scene.

        Parameters
        ----------
        morph : gs.morphs.USD
            The stage to add to the scene.
        material : gs.materials.Material | None, optional
            The material of the stage. If None, use ``gs.materials.Rigid()`` for all morphs.
        surface : gs.surfaces.Surface | None, optional
            The surface of the stage. If None, use ``gs.surfaces.Default()`` for all morphs.
        visualize_contact : bool
            Whether to visualize contact forces applied to this stage as arrows in the viewer and rendered images.
            Note that this will not be displayed in images rendered by camera using the `RayTracer` renderer.
        vis_mode : str | None, optional
            The visualization mode of the stage. This is a handy shortcut for setting `surface.vis_mode` without
            explicitly creating a surface object.

        Returns
        -------
        entities : List[genesis.Entity]
            The created entities.
        """
        if not isinstance(morph, gs.morphs.USD):
            gs.raise_exception(f"Unsupported morph: {morph}.")
        from genesis.utils.usd import parse_usd_stage

        # One `gs.morphs.USD` per rigid entity of the stage, added while the stage it was split from stays open.
        with parse_usd_stage(morph) as entity_morphs:
            entities = []
            for entity_morph in entity_morphs:
                entities.append(self.add_entity(entity_morph, material, surface, visualize_contact, vis_mode))
        return entities

    @gs.assert_unbuilt
    def add_mesh_light(
        self,
        morph: Morph | None = None,
        color: "np.typing.ArrayLike | None" = (1.0, 1.0, 1.0, 1.0),
        intensity: float = 20.0,
        revert_dir: bool | None = False,
        double_sided: bool | None = False,
        cutoff: float | None = 180.0,
    ):
        """
        Add a mesh light to the scene. Only supported by RayTracer.

        Parameters
        ----------
        morph : gs.morphs.Morph
            The morph of the light. Must be an instance of `gs.morphs.Primitive` or `gs.morphs.Mesh`.
        color : tuple of float, shape (3,)
            The color of the light, specified as (r, g, b).
        intensity : float
            The intensity of the light.
        revert_dir : bool
            Whether to revert the direction of the light. If True, the light will be emitted towards the mesh's inside.
        double_sided : bool
            Whether to emit light from both sides of surface.
        cutoff : float
            The cutoff angle of the light in degrees. Range: [0.0, 180.0].
        """
        if not isinstance(self.options.renderer, gs.renderers.RayTracer):
            gs.raise_exception(
                "This method is only supported by RayTracer. Please use 'add_light' when using BatchRenderer."
            )

        if not isinstance(morph, (gs.morphs.Primitive, gs.morphs.Mesh)):
            gs.raise_exception("Light morph only supports `gs.morphs.Primitive` or `gs.morphs.Mesh`.")
        meshes = gs.Mesh.from_morph_surface(morph, gs.surfaces.Plastic(smooth=False))
        for mesh in meshes:
            self._visualizer.add_mesh_light(
                mesh, color, intensity, morph.pos, morph.quat, revert_dir, double_sided, cutoff
            )

    @gs.assert_unbuilt
    def add_light(
        self,
        pos: "np.typing.ArrayLike | None",
        dir: "np.typing.ArrayLike | None",
        color: "np.typing.ArrayLike | None" = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        directional: bool = False,
        castshadow: bool = True,
        cutoff: float = 45.0,
        attenuation: float = 0.0,
    ):
        """
        Add a light to the scene for batch renderer.

        Parameters
        ----------
        pos : tuple of float, shape (3,)
            The position of the light, specified as (x, y, z).
        dir : tuple of float, shape (3,)
            The direction of the light, specified as (x, y, z).
        color : tuple of float, shape (3,)
            The color of the light, specified as (r, g, b).
        intensity : float
            The intensity of the light.
        directional : bool
            Whether the light is directional.
        castshadow : bool
            Whether the light casts shadows.
        cutoff : float
            The cutoff angle of the light in degrees. Range: (0.0, 90.0).
        attenuation : float
            The attenuation factor of the light.
            Light intensity will attenuate by distance with (1 / (1 + attenuation * distance ^ 2))
        """
        if not isinstance(self.options.renderer, gs.renderers.BatchRenderer):
            gs.raise_exception(
                "This method is only supported by BatchRenderer. Please use 'add_mesh_light' when using RayTracer."
            )

        self.visualizer.add_light(pos, dir, color, intensity, directional, castshadow, cutoff, attenuation)

    @gs.assert_unbuilt
    def add_sensor(self, sensor_options: "SensorOptions[SensorT]") -> "SensorT":
        """
        Add a sensor to the scene.

        Sensors extract information from the scene without modifying the physics simulation.

        Parameters
        ----------
        sensor_options : SensorOptions
            The options for the sensor.
        """
        return self._sim._sensor_manager.create_sensor(sensor_options)

    @gs.assert_built
    def read_sensors(self, envs_idx=None) -> "dict[type[Sensor], torch.Tensor]":
        """
        Read every sensor in the scene as a tensor per sensor class.

        Always returns a fresh tensor independent of the internal sensor storage; the caller is free to mutate the
        result.

        Parameters
        ----------
        envs_idx : array-like | int | slice | None
            Environment selection. Defaults to all environments.

        Returns
        -------
        dict[Type[Sensor], torch.Tensor]
            For each sensor class present in the scene, a tensor of shape (B, [history,] class_cache_size).
        """
        return self._sim._sensor_manager.read_sensors(entity_idx=None, envs_idx=envs_idx)

    @gs.assert_unbuilt
    def add_recorder(self, data_func: Callable, rec_options: "RecorderOptions") -> "Recorder":
        """
        Automatically read and process data. See RecorderOptions for more details.

        Data from `data_func` is automatically read and processed using the recorder at the frequency `rec_options.hz`
        (or every step if not specified) as the scene is stepped. Recording starts with the build and every recorder
        stops with 'stop_recording'.

        Parameters
        ----------
        data_func: Callable
            A function with no arguments that returns the data to be recorded.
        rec_options : RecorderOptions
            The options for the recording.

        Returns
        -------
        recorder : Recorder
            The created recorder object.
        """
        return self._recorder_manager.add_recorder(data_func, rec_options)

    @gs.assert_unbuilt
    def start_recording(self, rec_options: TrajectoryFile) -> "Recorder":
        """Record the state of the scene at every step to a trajectory file, which 'load_trajectory' opens to seek and
        replay.

        Recording starts with the build and stops with 'stop_recording'. See 'TrajectoryFile' for what a frame holds
        and how the file is written.

        Parameters
        ----------
        rec_options : TrajectoryFile
            The file to write and the mode to record in.

        Returns
        -------
        recorder : Recorder
            The created recorder object.
        """

        def data_func() -> TrajectorySource:
            self._reject_unexportable()
            return TrajectorySource(self._sim, self.desc, self._layout)

        return self._recorder_manager.add_recorder(data_func, rec_options)

    @gs.assert_unbuilt
    def add_camera(
        self,
        model="pinhole",
        res=(320, 320),
        pos=(0.5, 2.5, 3.5),
        lookat=(0.5, 0.5, 0.5),
        up=(0.0, 0.0, 1.0),
        fov=30,
        aperture=2.0,
        focus_dist=None,
        GUI=False,
        spp=256,
        denoise=None,
        near=0.1,
        far=20.0,
        env_idx=None,
        debug=False,
    ):
        """
        Add a camera to the scene.

        The camera model can be either 'pinhole', 'thinlens' or 'fisheye':
        - The 'pinhole' model is a simple camera model that captures light rays from a single point in space.
        - The 'thinlens' model is a more complex camera model that simulates a lens with a finite aperture size,
          allowing for depth of field effects. It is only supported by the Raytracer.
        - The 'fisheye' model is a camera model that simulates a fisheye lens, allowing for wide-angle views. It is
          only supported by the BatchRenderer.

        Warning
        -------
        When 'pinhole' is used, the `aperture` and `focal_len` parameters are ignored.

        Parameters
        ----------
        model : str
            Specifies the camera model. Options are 'pinhole' or 'thinlens'.
        res : tuple of int, shape (2,)
            The resolution of the camera, specified as a tuple (width, height).
        pos : tuple of float, shape (3,)
            The position of the camera in the scene, specified as (x, y, z).
        lookat : tuple of float, shape (3,)
            The point in the scene that the camera is looking at, specified as (x, y, z).
        up : tuple of float, shape (3,)
            The up vector of the camera, defining its orientation, specified as (x, y, z).
        fov : float
            The vertical field of view of the camera in degrees.
        aperture : float
            The aperture size of the camera, controlling depth of field.
        focus_dist : float | None
            The focus distance of the camera. If None, it will be auto-computed using `pos` and `lookat`.
        GUI : bool
            Whether to display the camera's rendered image in a separate GUI window.
        spp : int, optional
            Samples per pixel. Only available when using RayTracer renderer. Defaults to 256.
        denoise : bool
            Whether to denoise the camera's rendered image. Only available when using the RayTracer renderer. Defaults
            to True on Linux, otherwise False. If OptiX denoiser is not available in your platform, consider enabling
            the OIDN denoiser option when building the RayTracer.
        near : float
            Distance from camera center to near plane in meters.
            Only available when using rasterizer in Rasterizer and BatchRender renderer. Defaults to 0.1.
        far : float
            Distance from camera center to far plane in meters.
            Only available when using rasterizer in Rasterizer and BatchRender renderer. Defaults to 20.0.
        env_idx : int, optional
            The specific environment index to bind to the camera. This option must be specified if and only if a
            non-batched renderer is being used. If provided, only this environment will be taken into account when
            following a rigid entity via 'follow_entity' and when being attached to some rigid link via 'attach'. Note
            that this option is unrelated to which environment is being rendering on the scene. Default to None for
            batched renderers (ie BatchRender), 'rendered_envs_idx[0]' otherwise (ie Raytracer or Rasterizer).
        debug : bool
            Whether to use the debug camera. It enables to create cameras that can used to monitor / debug the
            simulation without being part of the "sensors". Their output is rendered by the usual simple Rasterizer
            systematically, no matter if BatchRender and RayTracer is enabled. This way, it is possible to record the
            simulation with arbitrary resolution and camera pose, without interfering with what robots can perceive
            from their environment. Defaults to False.

        Returns
        -------
        camera : genesis.Camera
            The created camera object.
        """
        if denoise is None:
            denoise = sys.platform != "darwin"
        return self._visualizer.add_camera(
            res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise, near, far, env_idx, debug
        )

    @gs.assert_unbuilt
    def add_emitter(
        self,
        material: Material,
        max_particles=20000,
        surface: Surface | None = None,
    ):
        """
        Add a fluid emitter to the scene.

        Parameters
        ----------
        material : gs.materials.Material
            The material of the fluid to be emitted. Must be an instance of `gs.materials.MPM.Base`,
            `gs.materials.SPH.Base`, `gs.materials.PBD.Particle` or `gs.materials.PBD.Liquid`.
        max_particles : int
            The maximum number of particles that can be emitted by the emitter. Particles will be recycled once this
            limit is reached.
        surface : gs.surfaces.Surface | None, optional
            The surface of the emitter. If None, use `gs.surfaces.Default(color=(0.6, 0.8, 1.0, 1.0))`.

        Returns
        -------
        emitter : genesis.Emitter
            The created emitter object.
        """
        from genesis.engine.entities import Emitter

        if self.requires_grad:
            gs.raise_exception("Emitter is not supported in differentiable mode.")

        if not isinstance(
            material, (gs.materials.MPM.Base, gs.materials.SPH.Base, gs.materials.PBD.Particle, gs.materials.PBD.Liquid)
        ):
            gs.raise_exception(
                "Non-supported material for emitter. Supported materials are: `gs.materials.MPM.Base`, "
                "`gs.materials.SPH.Base`, `gs.materials.PBD.Particle`, `gs.materials.PBD.Liquid`."
            )

        if surface is None:
            surface = gs.surfaces.Default(color=(0.6, 0.8, 1.0, 1.0))

        if surface.vis_mode is None:
            surface.vis_mode = "particle"
        if surface.vis_mode == "visual":
            gs.raise_exception("surface.vis_mode='visual' is not supported for fluid emitters.")

        emitter = Emitter(max_particles)
        entity = self.add_entity(
            morph=gs.morphs.Nowhere(n_particles=max_particles),
            material=material,
            surface=surface,
        )
        emitter.set_entity(entity)
        self._emitters.append(emitter)
        return emitter

    @gs.assert_unbuilt
    def add_force_field(self, force_field: ForceField):
        """
        Add a force field to the scene.

        Parameters
        ----------
        force_field : gs.force_fields.ForceField
            The force field to add to the scene.

        Returns
        -------
        force_field : gs.force_fields.ForceField
            The added force field.
        """
        force_field.scene = self
        self._sim._add_force_field(force_field)
        return force_field

    @gs.assert_unbuilt
    def build(
        self,
        n_envs=0,
        env_spacing=(0.0, 0.0),
        n_envs_per_row: int | None = None,
        center_envs_at_origin=True,
    ):
        """
        Builds the scene once all entities have been added. This operation is required before running the simulation.

        Parameters
        ----------
        n_envs : int
            Number of parallel environments to create.
            If `n_envs` is 0, the scene will not have a batching dimension. When greater than 0, the first dimension of
            all the input and returned states will be the batch dimension.
        env_spacing : tuple of float, shape (2,)
            The spacing between adjacent environments in the scene.
            This is for visualization purposes only and does not change simulation-related poses.
        n_envs_per_row : int
            The number of environments per row for visualization. If None, it will be set to `sqrt(n_envs)`.
        center_envs_at_origin : bool
            Whether to put the center of all the environments at the origin (for visualization only).
        """

        # Start tracking the scene right away, so that destroy is called even if some error fires during build
        def _destroy_callback(scene_ref: weakref.ReferenceType["Scene"]):
            scene = scene_ref()
            for i, scene_ref_i in enumerate(gs._scene_registry):
                if scene is scene_ref_i():
                    del gs._scene_registry[i]
                    break

        gs._scene_registry.append(weakref.ref(self, _destroy_callback))

        with gs.logger.timer(f"Building scene ~~~<{self._uid}>~~~..."):
            self._parallelize(n_envs, env_spacing, n_envs_per_row, center_envs_at_origin)

            # simulator
            self._sim.build()

            # reset state
            self._reset()

            # The description is fixed once built, so its digest is taken once (see 'description_digest')
            self._desc_digest = description_digest(self._desc)
            self._is_built = True

        with gs.logger.timer("Compiling simulation kernels..."):
            self._sim.step()
            self._reset()

        # visualizer
        with gs.logger.timer("Building visualizer..."):
            self._visualizer.build()

        if self.options.profiling.show_FPS:
            self.FPS_tracker = FPSTracker(self.n_envs, alpha=self.options.profiling.FPS_tracker_alpha)

        # recorders
        self._recorder_manager.build()

    def _parallelize(
        self,
        n_envs: int,
        env_spacing: tuple[float, float],
        n_envs_per_row: int,
        center_envs_at_origin: bool,
    ):
        self.n_envs = n_envs
        self.env_spacing = env_spacing
        self.n_envs_per_row = n_envs_per_row
        self._center_envs_at_origin = center_envs_at_origin

        # true batch size
        self._B = max(1, self.n_envs)
        self._envs_idx = torch.arange(self._B, dtype=gs.tc_int, device=gs.device)

        if self.n_envs_per_row is None:
            self.n_envs_per_row = math.ceil(math.sqrt(self._B))

        # compute offset values for visualizing each env
        if not isinstance(env_spacing, (list, tuple)) or len(env_spacing) != 2:
            gs.raise_exception("`env_spacing` should be a tuple of length 2.")
        self._layout = EnvironmentLayout(n_envs, tuple(env_spacing), self.n_envs_per_row, center_envs_at_origin)
        offset_x = (np.arange(self._B) // self.n_envs_per_row) * self.env_spacing[0]
        offset_y = (np.arange(self._B) % self.n_envs_per_row) * self.env_spacing[1]
        offset_z = np.zeros((self._B,))
        self.envs_offset = np.stack((offset_x, offset_y, offset_z), axis=-1, dtype=gs.np_float)

        # move to center
        if center_envs_at_origin:
            center = (np.max(self.envs_offset, axis=0) + np.min(self.envs_offset, axis=0)) / 2.0
            self.envs_offset -= center

        """
        Notes:
        - When using gpu
            - for non-batched env, we only parallelize certain loops that have big loop size
            - for batched env, we parallelize all loops
        - When using cpu, we serialize everything.
            - Parallelization only provides a boost for n_envs >= num_threads.
              It is always disabled by default but can be enforced by setting the env var `GS_PARA_LEVEL=2`.
            - In order to exploit full cpu power, users are encouraged to launch multiple processes manually, so that
              each process uses a single cpu thread.
        """
        if gs.backend == gs.cpu:
            para_level = gs.PARA_LEVEL.NEVER
        elif self.n_envs <= 1:
            para_level = gs.PARA_LEVEL.PARTIAL
        else:
            para_level = gs.PARA_LEVEL.ALL
        self._para_level = int(os.environ.get("GS_PARA_LEVEL", para_level))

    @gs.assert_built
    def reset(self, state: SimState | None = None, envs_idx=None):
        """
        Resets the scene to its initial state.

        Parameters
        ----------
        state : SimState | None
            The state to reset the scene to. If None, the scene will be reset to its initial state.
            If this is given, the scene's registerered initial state will be updated to this state.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        gs.logger.debug(f"Resetting Scene ~~~<{self._uid}>~~~.")
        # The recorders finish the run first: a reset of the whole scene records the state it leaves behind, and a file
        # rotated on reset ends with it.
        if envs_idx is None:
            self._recorder_manager.record(self._sim.cur_step_global)
        self._recorder_manager.reset(envs_idx)
        self._reset(state, envs_idx=envs_idx)

    def _reset(self, state: SimState | None = None, *, envs_idx=None, keep_init: bool = False):
        if self._is_built:
            if state is None:
                state = self._init_state
            else:
                assert isinstance(state, SimState), "state must be a SimState object"
                # keep_init=True restores the state while leaving the registered init untouched, so a later bare
                # reset() still rewinds to the true initial state.
                if not keep_init:
                    self._init_state = state
            self._sim.reset(state, envs_idx)
        else:
            self._init_state = self._get_state()
        self._restart()

    def _restart(self):
        """Restart what surrounds the state: the gradient tape, the cache of the visualizer and the emitters."""
        self._forward_ready = True
        self._reset_grad()

        # Clear the entire cache of the visualizer.
        # TODO: Could be optimized to only clear cache associated the the environments being reset.
        if self._visualizer.is_built:
            self._visualizer.reset()

        # TODO: sets _next_particle = 0; not sure this is env isolation safe
        for emitter in self._emitters:
            emitter.reset()

    def _reset_grad(self):
        self._backward_ready = True

    @gs.assert_built
    def backward(self, loss: torch.Tensor, *args, **kwargs):
        """
        Differentiates `loss` through the recorded rollout and restores the pre-backward physics state.

        Unrolling the gradient tape rewinds the physics state to step 0, so this method snapshots the current state
        first, runs the backward pass, and restores the snapshot afterwards. The scene then sits at the same physics
        state as before the call, with gradients populated and forward / backward re-armed, ready to continue the
        rollout or to be reset. The registered initial state (`reset()` with no argument) is preserved.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss to differentiate. Extra positional and keyword arguments (e.g. `gradient`, `retain_graph`)
            are forwarded to `torch.autograd.backward`.

        Returns
        -------
        snapshot : SimState
            The physics state the scene was restored to.
        """
        # Snapshot the current state before the gradient-tape unroll rewinds physics to step 0. The clocks go with
        # it: the unroll winds them down step by step, and the restore below puts the physics back where they were.
        snapshot = self.get_state()
        steps = self._sim._steps.clone()
        # The sim unroll (self._backward) re-enters the torch graph from each step's queried states, so the graph
        # must survive the initial autograd pass.
        if kwargs.setdefault("retain_graph", True) is not True:
            gs.raise_exception("'retain_graph' must be left unset: scene.backward requires the graph to survive.")
        # The functional torch.autograd.backward fills torch and queried-state grads while leaving the sim unroll to
        # the explicit self._backward call below, keeping gs.Tensor.backward's automatic scene._backward out of it.
        torch.autograd.backward(loss, *args, **kwargs)
        self._backward()
        # keep_init semantics: see _reset. Restoring the snapshot puts the physics back where the forward pass left
        # it, so the clocks snapshotted above are put back with it rather than left at zero.
        self._reset(snapshot, keep_init=True)
        self._sim._steps[:] = steps
        return snapshot

    def _get_state(self):
        return self._sim.get_state()

    @gs.assert_built
    def get_state(self):
        """
        Returns the current state of the scene.

        Returns
        -------
        state : genesis.SimState
            The state of the scene at the current time step.
        """
        return self._get_state()

    def register_pre_step_callback(self, callback):
        """Register a callback invoked at the start of each ``step()``, on the stepping thread. A callback
        may run deferred work there and veto the advance of that step by returning ``True``. The scene calls
        them opaquely; use this to drive a scene from an external controller without coupling the scene to it."""
        self._pre_step_callbacks.append(callback)

    @gs.assert_built
    def step(self, update_visualizer=True, refresh_visualizer=True):
        """
        Runs a simulation step forward in time.
        """
        # Run pre-step callbacks on the stepping thread. A callback may perform deferred work and veto this
        # frame's advance by returning True. The scene treats them opaquely, without knowing what they do or who
        # registered them (e.g. an InteractiveScene driving GUI-requested rebuild/pause). The visualizer is still
        # refreshed when the advance is vetoed, so the viewer keeps rendering and stays responsive while paused.
        advance = not any([callback() for callback in tuple(self._pre_step_callbacks)])

        if advance:
            if not self._forward_ready:
                gs.raise_exception("Forward simulation not allowed after backward pass. Please reset scene state.")
            # The recorders read the state a step starts from, with the inputs the step consumes: 'Simulator.step'
            # zeroes the applied forces at its end. The pre-step callbacks ran, so their inputs are read as well.
            self._recorder_manager.step(self._sim.cur_step_global)
            self._sim.step()

        if update_visualizer:
            # Force the refresh when the sim did not advance (e.g. paused) so edits made off the step loop -
            # like a GUI joint slider calling set_qpos - are still drawn and the viewer does not appear frozen.
            self._visualizer.update(force=not advance, auto=refresh_visualizer)

        if advance:
            if self.options.profiling.show_FPS:
                self.FPS_tracker.step()
            for camera in self._visualizer.cameras:
                camera.update_recording()

    def stop_recording(self):
        # The recorders read before each step, so the state the last step left is recorded here, whatever the
        # sampling rate: it is the state a stopped run is resumed or studied from.
        try:
            self._recorder_manager.record(self._sim.cur_step_global)
        finally:
            self._recorder_manager.stop()

    def _step_grad(self):
        self._sim.collect_output_grads()
        self._sim._step_grad()

    @gs.assert_built
    def draw_debug_line(self, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws a line in the scene for visualization.

        Parameters
        ----------
        start : array_like, shape (3,)
            The starting point of the line.
        end : array_like, shape (3,)
            The ending point of the line.
        radius : float, optional
            The radius of the line (represented as a cylinder)
        color : array_like, shape (4,), optional
            The color of the line in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_line(start, end, radius, color)

    @gs.assert_built
    def draw_debug_arrow(self, pos, vec=(0, 0, 1), radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws an arrow in the scene for visualization.

        Parameters
        ----------
        pos : array_like, shape (3,)
            The starting position of the arrow.
        vec : array_like, shape (3,), optional
            The vector of the arrow.
        radius : float, optional
            The radius of the arrow body (represented as a cylinder).
        color : array_like, shape (4,), optional
            The color of the arrow in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_arrow(pos, vec, radius, color)

    @gs.assert_built
    def draw_debug_frame(self, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01, color=None):
        """
        Draws a 3-axis coordinate frame in the scene for visualization.

        Parameters
        ----------
        T : array_like, shape (4, 4)
            The transformation matrix of the frame.
        axis_length : float, optional
            The length of the axes.
        origin_size : float, optional
            The size of the origin point (represented as a sphere).
        axis_radius : float, optional
            The radius of the axes (represented as cylinders).
        color : array_like, shape (4,), optional
            Uniform RGBA color override for the entire frame. If None, uses standard RGB axis coloring.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_frame(T, axis_length, origin_size, axis_radius, color)

    @gs.assert_built
    def draw_debug_frames(self, Ts, axis_length=1.0, origin_size=0.015, axis_radius=0.01, color=None):
        """
        Draws 3-axis coordinate frames in the scene for visualization.

        Parameters
        ----------
        Ts : array_like, shape (n, 4, 4)
            The transformation matrices of frames.
        axis_length : float, optional
            The length of the axes.
        origin_size : float, optional
            The size of the origin point (represented as a sphere).
        axis_radius : float, optional
            The radius of the axes (represented as cylinders).
        color : array_like, shape (4,), optional
            Uniform RGBA color override for the entire frame. If None, uses standard RGB axis coloring.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_frames(Ts, axis_length, origin_size, axis_radius, color)

    @gs.assert_built
    def draw_debug_mesh(self, mesh, pos=np.zeros(3), T=None):
        """
        Draws a mesh in the scene for visualization.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to be drawn.
        pos : array_like, shape (3,), optional
            The position of the mesh in the scene.
        T : array_like, shape (4, 4) | None, optional
            The transformation matrix of the mesh. If None, the mesh will be drawn at the position specified by `pos`. Otherwise, `T` has a higher priority than `pos`.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_mesh(mesh, pos, T)

    @gs.assert_built
    def draw_debug_sphere(self, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws a sphere in the scene for visualization.

        Parameters
        ----------
        pos : array_like, shape (3,)
            The center position of the sphere.
        radius : float, optional
            radius of the sphere.
        color : array_like, shape (4,), optional
            The color of the sphere in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_sphere(pos, radius, color)

    @gs.assert_built
    def draw_debug_spheres(self, poss, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws multiple spheres in the scene for visualization.

        Parameters
        ----------
        poss : array_like, shape (N, 3)
            The positions of the spheres.
        radius : float, optional
            The radius of the spheres.
        color : array_like, shape (4,), optional
            The color of the spheres in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_spheres(poss, radius, color)

    @gs.assert_built
    def draw_debug_box(
        self,
        bounds,
        color=(1.0, 0.0, 0.0, 1.0),
        wireframe=True,
        wireframe_radius=0.0015,
    ):
        """
        Draws a box in the scene for visualization.

        Parameters
        ----------
        bounds : array_like, shape (2, 3)
            The bounds of the box, specified as [[min_x, min_y, min_z], [max_x, max_y, max_z]].
        color : array_like, shape (4,), optional
            The color of the box in RGBA format.
        wireframe : bool, optional
            Whether to draw the box as a wireframe.
        wireframe_radius : float, optional
            The radius of the wireframe lines.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_box(
                bounds, color, wireframe=wireframe, wireframe_radius=wireframe_radius
            )

    @gs.assert_built
    def draw_debug_points(self, poss, colors=(1.0, 0.0, 0.0, 0.5)):
        """
        Draws points in the scene for visualization.

        Parameters
        ----------
        poss : array_like, shape (N, 3)
            The positions of the points.
        colors : array_like, shape (4,), optional
            The color of the points in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_points(poss, colors)

    @gs.assert_built
    def draw_debug_frustum(self, camera, color=(1.0, 1.0, 1.0, 0.3)):
        """
        Draws a camera frustum in the scene for visualization.

        Parameters
        ----------
        camera : Camera
            The camera object whose frustum will be visualized. Works for any
            camera including sensor cameras.
        color : array_like, shape (4,), optional
            The color of the frustum in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object.
        """
        with self._visualizer.viewer_lock:
            mesh = mu.create_camera_frustum(camera, color)
            return self._visualizer.context.draw_debug_mesh(mesh, T=camera.transform)

    @gs.assert_built
    def draw_debug_trajectory(self, poss, radius=0.002, color=(1.0, 0.5, 0.0, 0.8)):
        """
        Draws a trajectory as a series of connected lines in the scene for visualization.

        Parameters
        ----------
        poss : array_like, shape (N, 3)
            The positions of the trajectory points.
        radius : float, optional
            The radius of the trajectory lines.
        color : array_like, shape (4,), optional
            The color of the trajectory in RGBA format.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object (a single merged mesh of all segments).
        """

        poss = np.asarray(poss)
        if len(poss) < 2:
            return None

        segments = [mu.create_line(poss[i], poss[i + 1], radius, color) for i in range(len(poss) - 1)]
        merged = trimesh.util.concatenate(segments)
        with self._visualizer.viewer_lock:
            return self._visualizer.context.draw_debug_mesh(merged)

    @gs.assert_built
    def draw_debug_path(self, qposs, entity, link_idx=-1, density=0.3, frame_scaling=1.0):
        """
        Draws a planned joint trajectory in the scene for visualization.

        Parameters
        ----------
        qposs : array_like, shape (N, M)
            The joint positions of the planned points.
            N is the number of configurations (i.e., trajectory points).
            M is the number of degrees of freedom for the entity (i.e., joint dimensions).
        entity : gs.engine.entities.RigidEntity
            The rigid entity whose forward kinematics are used to compute the trajectory path.
        link_idx : int, optional
            The link id of the rigid entity to visualize. Defeault is -1.
        density : float, optional
            Controls the sampling density of the trajectory points to visualize. Default is 0.3.
        frame_scaling : float, optional
            Scaling factor for the visualization frames' size. Affects the length and thickness of the debug frames.
            Default is 1.0.

        Returns
        -------
        node : genesis.ext.pyrender.mesh.Mesh
            The created debug object representing the visualized trajectory.

        Notes
        -----
        The function uses forward kinematics (FK) to convert joint positions to Cartesian space and render debug frames.
        The density parameter reduces FK computational load by sampling fewer points, with 1.0 representing the whole
        trajectory.
        """
        with self._visualizer.viewer_lock:
            N = len(qposs)
            density = np.clip(density, 0.0, 1.0)
            N_new = int(N * density)
            indices = torch.linspace(0, N - 2, N_new, dtype=int)

            Ts = np.zeros((N_new, 4, 4))
            for i in range(N_new):
                pos, quat = self.rigid_solver.forward_kinematics_query(entity, qposs[indices[i]])
                Ts[i] = tensor_to_array(gu.trans_quat_to_T(pos[link_idx], quat[link_idx]))

            return self._visualizer.context.draw_debug_frames(
                Ts, axis_length=frame_scaling * 0.1, origin_size=0.001, axis_radius=frame_scaling * 0.005
            )

    @gs.assert_built
    def render_all_cameras(
        self,
        rgb=True,
        depth=False,
        segmentation=False,
        colorize_seg=False,
        normal=False,
        antialiasing=False,
        force_render=False,
    ):
        """
        Render the scene for all cameras using the batch renderer.

        Parameters
        ----------
        rgb : bool, optional
            Whether to render the rgb image.
        depth : bool, optional
            Whether to render the depth image.
        segmentation : bool, optional
            Whether to render the segmentation image.
        normal : bool, optional
            Whether to render the normal image.
        antialiasing : bool, optional
            Whether to apply anti-aliasing.
        force_render : bool, optional
            Whether to force render the scene.

        Returns:
            A tuple of tensors of shape (n_envs, H, W, 3) if rgb is not None,
            otherwise a list of tensors of shape (n_envs, H, W) if depth is not None.
            If n_envs == 0, the first dimension of the tensor is squeezed.
        """
        if self._visualizer.batch_renderer is None:
            gs.raise_exception("Method only supported by 'BatchRenderer'")

        rgb_out, depth_out, seg_out, normal_out = self._visualizer.batch_renderer.render(
            rgb, depth, segmentation, normal, antialiasing, force_render
        )
        if segmentation and colorize_seg:
            seg_out = tuple(self._visualizer.batch_renderer.colorize_seg_idxc_arr(seg) for seg in seg_out)
        return rgb_out, depth_out, seg_out, normal_out

    @gs.assert_built
    def update_debug_objects(self, objs, poses):
        """
        Updates the poses of debug objects previously created by ``draw_debug_*`` methods.

        Parameters
        ----------
        objs : tuple of genesis.ext.pyrender.mesh.Mesh
            The debug objects to update, i.e. visualizer nodes returned by ``draw_debug_*`` methods. Currently only
            individual sphere, frame, mesh, and arrow objects (returned by ``draw_debug_sphere``, ``draw_debug_frame``,
            ``draw_debug_mesh``, and ``draw_debug_arrow`` respectively) are supported.
        poses : tuple of array_like, each of shape (4, 4)
            The new transformation matrices for each debug object.
        """
        with self._visualizer.viewer_lock:
            self._visualizer.context.update_debug_objects(objs, poses)

    @gs.assert_built
    def clear_debug_object(self, obj):
        """
        Clears the specified debug object from the scene.
        """
        with self._visualizer.viewer_lock:
            self._visualizer.context.clear_debug_object(obj)

    @gs.assert_built
    def clear_debug_objects(self):
        """
        Clears all the debug objects in the scene.
        """
        with self._visualizer.viewer_lock:
            self._visualizer.context.clear_debug_objects()

    def _backward(self):
        """
        At this point, all the scene states the simulation run should have been filled with gradients.
        Next, we run backward from scene state back to scene's internal# Quadrants variables, then back through time.
        """

        if not self._backward_ready:
            gs.raise_exception("Multiple backward calls not allowed.")

        # backward pass through time
        while self._sim.cur_step_global > 0:
            self._step_grad()

        self._backward_ready = False
        self._forward_ready = False

    @gs.assert_built
    def get_time(self, envs_idx=None):
        """
        Get the simulated time of each environment, in seconds.

        Environments are stepped and reset independently, so each one carries its own simulated time. The number of
        `scene.step()` calls is a separate scalar, `Simulator.cur_step_global`.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        time : torch.Tensor, shape (n_envs,) or scalar
            The simulated time of each environment.
        """
        return self._sim.get_time(envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- utilities --------------------------------------
    # ------------------------------------------------------------------------------------

    def _reject_unexportable(self) -> None:
        """Raise when the scene holds an entity, an emitter or a force field no description covers, and warn about what
        'export' leaves out.
        """
        # A scene holding what alters the simulation is rejected, since a file leaving it out would restore other
        # physics. Everything else is left out with a warning: what observes or draws the simulation, and every
        # callback. Every solver holds the same force fields, so one solver names them all.
        rejected_kinds = Counter(type(entity).__name__ for entity in self.entities if entity.desc is None)
        rejected_kinds.update(type(emitter).__name__ for emitter in self._emitters)
        rejected_kinds.update(type(force_field).__name__ for force_field in self._sim.solvers[0].force_fields)
        omitted_kinds = Counter(type(sensor).__name__ for sensor in self._sim._sensor_manager.sensors)
        if self._visualizer is not None:
            omitted_kinds.update(type(camera).__name__ for camera in self._visualizer.cameras)
        if self._pre_step_callbacks:
            omitted_kinds["pre_step_callback"] += len(self._pre_step_callbacks)
        surfaces = [entity.surface for entity in self.entities]
        if isinstance(self.options.renderer, gs.renderers.RayTracer) and self.options.renderer.env_surface is not None:
            surfaces.append(self.options.renderer.env_surface)
        omitted_kinds.update(name for surface in surfaces for name in pixel_less_textures(surface))
        # Visual vertices written at runtime live in the solver, and a file carries what a scene was authored from.
        omitted_kinds.update(
            f"'{entity.name}' visual vertices"
            for entity in self.entities
            if isinstance(entity, KinematicEntity) and entity._is_vverts_overridden
        )
        for kinds, report, ending in (
            (omitted_kinds, gs.logger.warning, "is exported without it, so a scene loaded from the file holds none."),
            (rejected_kinds, gs.raise_exception, "cannot be exported yet."),
        ):
            if kinds:
                listing = ", ".join(
                    f"{count} {kind}" if kind.endswith("s") else f"{count} {kind}{'s' if count > 1 else ''}"
                    for kind, count in sorted(kinds.items())
                )
                report(f"A scene holding {listing} {ending}")

    def export(self, path: str | os.PathLike) -> None:
        """Write a portable copy of this scene to a file that anyone can open.

        What is written is what the scene was authored from and what its build resolved, so the file stands on its
        own: opening it reads no mesh, no model file and no texture from disk, and creates only the options,
        descriptions and meshes Genesis declares. The file is therefore self-contained enough to attach to a bug
        report.

        A scene opened from a file stands at the configuration its entities were given rather than where the
        simulation had run to, so the simulated state has to be reproduced by stepping it again. Adding an entity
        resolves its description, so a scene is exported before it is built as readily as after.

        Only a rigid or a kinematic entity carries a description. A scene
        holding anything that alters the simulation raises, naming it: an emitter and a force field. Everything else
        a description leaves out is written without, with a warning naming it: a camera, a sensor, a callback Genesis
        calls at every step, a texture read from an HDR or EXR file, and the visual vertices an entity was given at
        runtime. A recorder neither simulates nor draws, so it is left out without a word.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write the file.
        """
        self._reject_unexportable()
        serialization.export(path, {"scene": self.desc})

    @classmethod
    def load(
        cls,
        path: str | os.PathLike | BinaryIO,
        show_viewer: bool = False,
        viewer_options: ViewerOptions | None = None,
        vis_options: VisOptions | None = None,
        renderer: RendererOptions | None = None,
    ) -> "Scene":
        """Create the scene a file holds, as written by 'Scene.export'.

        The file names what it holds rather than carrying code to run, and Genesis creates only the options,
        descriptions and meshes it declares, so a scene from a stranger is safe to open. The geometry travels in the
        file, so it opens on a machine holding none of the assets the scene was authored from.

        Parameters
        ----------
        path : str, os.PathLike or binary file
            The file to read.
        show_viewer : bool, optional
            Whether to open an interactive viewer on the scene. Defaults to False.
        viewer_options : ViewerOptions, optional
            Viewer options replacing the recorded ones. If None, the recorded ones stand. Defaults to None.
        vis_options : VisOptions, optional
            Visualizer options replacing the recorded ones. If None, the recorded ones stand. Defaults to None.
        renderer : RendererOptions, optional
            Renderer replacing the recorded one. If None, the recorded one stands. Defaults to None.

        Returns
        -------
        scene : Scene
            The scene the file describes, holding every entity it was authored with and waiting to be built.
        """
        described = serialization.load(path, {"scene": SceneDescription})["scene"]
        return cls._from_description(described, show_viewer, viewer_options, vis_options, renderer)

    @classmethod
    def _from_description(
        cls,
        described: SceneDescription,
        show_viewer: bool = False,
        viewer_options: ViewerOptions | None = None,
        vis_options: VisOptions | None = None,
        renderer: RendererOptions | None = None,
    ) -> "Scene":
        """Create the scene a description states, waiting to be built.

        The viewer, visualizer and renderer options are replaced where given. The physics options stand as described,
        since they size the arrays a recorded state fills.
        """
        replaced = (("viewer", viewer_options), ("vis", vis_options), ("renderer", renderer))
        options = described.options.model_copy(update={name: value for name, value in replaced if value is not None})
        scene = cls(show_viewer=show_viewer, options=options)
        # 'add_entity' would resolve a material and a surface the description already holds, and read the asset it
        # replaces.
        for desc in described.entities:
            scene._sim._add_entity(desc=desc)
        return scene

    @gs.assert_built
    def __getstate__(self) -> SceneCheckpoint:
        """Return the SceneCheckpoint of this built scene, for '__setstate__' to restore.

        The description travels by reference and the arrays as copies made where they live (see 'Solver.__getstate__'),
        so a save and a restore within one process cost one copy on the device. Pickling a scene reads this record,
        and unpickling creates and builds the scene, then restores the record. What a description leaves out is left
        out of a copy the same way, and reported as 'export' reports it.

        Returns
        -------
        state : SceneCheckpoint
            The scene as its description, the environment layout it was built with, and the state of its simulator.
        """
        self._reject_unexportable()
        return SceneCheckpoint(
            scene=self.desc, digest=self._desc_digest, layout=self._layout, sim=self._sim.__getstate__()
        )

    def __reduce__(self):
        return (self._from_state, (self.__getstate__(),))

    @gs.assert_built
    def save_checkpoint(self, path: str | os.PathLike) -> None:
        """Write the whole state of this scene to a file, for 'load_checkpoint' to open a copy standing where it stands.

        The file holds the scene as 'export' writes it and every array of the simulation, scratch included, so the
        exact state of a failing run is kept for inspection. It is a trajectory file of one frame (see 'TrajectoryFile'),
        and 'load_trajectory' opens it as such.

        Parameters
        ----------
        path : str or os.PathLike
            The '.gstraj' file to write.
        """
        self._reject_unexportable()
        save_checkpoint(path, TrajectorySource(self._sim, self.desc, self._layout))

    @classmethod
    def load_checkpoint(
        cls,
        path: str | os.PathLike,
        show_viewer: bool = False,
        viewer_options: ViewerOptions | None = None,
        vis_options: VisOptions | None = None,
        renderer: RendererOptions | None = None,
    ) -> "Scene":
        """Create and build the scene a checkpoint file holds, standing in the final state the file records.

        The file is one written by 'save_checkpoint' or by recording a 'TrajectoryFile' to its end. The viewer,
        visualizer and renderer options may be replaced (see 'load').

        Parameters
        ----------
        path : str or os.PathLike
            The file to read.
        show_viewer : bool, optional
            Whether to open an interactive viewer on the scene. Defaults to False.
        viewer_options : ViewerOptions, optional
            Viewer options replacing the recorded ones. Defaults to None.
        vis_options : VisOptions, optional
            Visualizer options replacing the recorded ones. Defaults to None.
        renderer : RendererOptions, optional
            Renderer replacing the recorded one. Defaults to None.

        Returns
        -------
        scene : Scene
            The built scene, in the recorded state.
        """
        trajectory = Trajectory(path, None, show_viewer, viewer_options, vis_options, renderer)
        trajectory.seek(len(trajectory) - 1)
        return trajectory.scene

    @classmethod
    def load_trajectory(
        cls,
        path: str | os.PathLike,
        show_viewer: bool = False,
        viewer_options: ViewerOptions | None = None,
        vis_options: VisOptions | None = None,
        renderer: RendererOptions | None = None,
    ) -> Trajectory:
        """Open a recorded trajectory in the scene it was recorded from, created and built here, to seek and replay.

        The file is one written by recording a 'TrajectoryFile'. The viewer, visualizer and renderer options may be
        replaced (see 'load').

        Parameters
        ----------
        path : str or os.PathLike
            The file to read.
        show_viewer : bool, optional
            Whether to open an interactive viewer on the scene. Defaults to False.
        viewer_options : ViewerOptions, optional
            Viewer options replacing the recorded ones. Defaults to None.
        vis_options : VisOptions, optional
            Visualizer options replacing the recorded ones. Defaults to None.
        renderer : RendererOptions, optional
            Renderer replacing the recorded one. Defaults to None.

        Returns
        -------
        trajectory : Trajectory
            The trajectory, holding the built scene as 'Trajectory.scene'.
        """
        return Trajectory(path, None, show_viewer, viewer_options, vis_options, renderer)

    @classmethod
    def _from_state(cls, state: SceneCheckpoint) -> "Scene":
        """Create, build and restore the scene a checkpoint describes. '__reduce__' names it as the loader."""
        scene = cls._from_description(state.scene)
        scene.build(**dataclasses.asdict(state.layout))
        scene.__setstate__(state)
        return scene

    @gs.assert_built
    def __setstate__(self, state: SceneCheckpoint) -> None:
        """Put this built scene in the state a checkpoint holds, as if it had run to there.

        The scene must be built from the description the state was read from, with the same environment layout, since
        the arrays written back are the ones that build allocates. A state from another layout is rejected, and a state
        from another build by the names of the arrays that differ. Everything around the state restarts as under
        'reset': the gradient tape, the sensors and the recorders forget the run that led here, and the visualizer
        redraws.

        Parameters
        ----------
        state : SceneCheckpoint
            The state to put back, as read by '__getstate__' or cut from a trajectory frame.
        """
        if state.layout != self._layout:
            gs.raise_exception(
                f"The state was read from a scene built with {state.layout} and this one was built with {self._layout}."
            )
        if state.digest != self._desc_digest:
            gs.raise_exception("The state was read from another scene: its entities or physics options differ.")
        # Every solver checks its record before anything is touched, so a record from another scene leaves this one as
        # it was, the recorders included
        solvers = {type(solver).__name__: solver for solver in self._sim.active_solvers}
        if set(state.sim.solvers) != set(solvers):
            gs.raise_exception(
                f"The checkpoint holds {sorted(state.sim.solvers)} where this scene simulates {sorted(solvers)}."
            )
        for name, solver in solvers.items():
            record = state.sim.solvers[name]
            check_data((item for item in solver.data if item.kind in record.kinds), record.arrays)
        # The recorders finish the run first (see 'reset'), and the restart follows the state, so the visualizer draws
        # the state put back.
        self._recorder_manager.record(self._sim.cur_step_global)
        self._recorder_manager.reset()
        self._sim.__setstate__(state.sim)
        self._restart()

    def _sanitize_envs_idx(
        self, envs_idx: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None
    ) -> torch.Tensor:
        if envs_idx is None:
            return self._envs_idx

        if self.n_envs == 0:
            gs.raise_exception("`envs_idx` is not supported for non-parallelized scene.")

        if isinstance(envs_idx, slice):
            if envs_idx.step is not None and envs_idx.step < 0:
                return sanitize_index(envs_idx, -1, self.n_envs, 0, "envs_idx")
            return self._envs_idx[envs_idx]
        if isinstance(envs_idx, range):
            if envs_idx.step < 0:
                return sanitize_index(envs_idx, -1, self.n_envs, 0, "envs_idx")
            return self._envs_idx[envs_idx]
        if isinstance(envs_idx, (int, np.integer)):
            n_envs = self.n_envs
            if not -n_envs <= envs_idx < n_envs:
                gs.raise_exception(f"`envs_idx` out of range: {envs_idx} not in [{-n_envs}, {n_envs}).")
            if envs_idx < 0:
                envs_idx = envs_idx + n_envs
            return self._envs_idx[envs_idx : envs_idx + 1]

        return sanitize_index(envs_idx, -1, self.n_envs, 0, "envs_idx")

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """The unique ID of the scene."""
        return self._uid

    @property
    def dt(self):
        """The time duration for each simulation step."""
        return self._sim.dt

    @property
    def substeps(self):
        """The number of substeps per simulation step."""
        return self._sim.substeps

    @property
    def requires_grad(self):
        """Whether the scene is in differentiable mode."""
        return self._sim.requires_grad

    @property
    def is_built(self) -> bool:
        """Whether the scene has been built."""
        return self._is_built

    @property
    def show_FPS(self):
        """Whether to print the frames per second (FPS) in the terminal."""
        warn_once("Scene.show_FPS is deprecated. Please use Scene.options.profiling.show_FPS")
        return self.options.profiling.show_FPS

    @property
    def viewer(self):
        """The viewer object for the scene."""
        return self._visualizer.viewer

    @property
    def visualizer(self):
        """The visualizer object for the scene."""
        return self._visualizer

    @property
    def sim(self):
        """The scene's top-level simulator."""
        return self._sim

    @property
    def solvers(self):
        """All the solvers managed by the scene's simulator."""
        return self._sim.solvers

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self._sim.active_solvers

    @property
    def entities(self) -> list["Entity"]:
        """All the entities in the scene."""
        return self._sim.entities

    @property
    def entity_names(self) -> tuple[str, ...]:
        """
        Get the names of all entities in the scene.

        Returns
        -------
        tuple[str, ...]
            Tuple of entity names in order of creation.
        """
        return tuple(entity.name for entity in self.entities)

    @property
    def desc(self) -> SceneDescription:
        """The description this scene is created from: its options and the description of each entity it holds.

        It suffices to recreate the scene without any asset file. Each entity's description stands here by reference,
        so what an attachment changes is visible. What the build allocates is left out, so this describes the authored
        scene rather than the state it has simulated to.
        """
        return self._desc

    def get_entity(self, name: str | None = None, *, uid: str | None = None) -> "Entity":
        """
        Get an entity by name or UID. Raises an exception if not found.

        Parameters
        ----------
        name : str, optional
            The exact name of the entity to find.
        uid : str, optional
            The short UID (7-character) of the entity to find.

        Returns
        -------
        Entity
            The matching entity.
        """
        if not ((name is None) ^ (uid is None)):
            gs.raise_exception("Please specify either one argument between `name` or `uid`.")

        if name is not None:
            try:
                return next(entity for entity in self.entities if entity.name == name)
            except StopIteration as e:
                gs.raise_exception_from(f"Entity not found for name: '{name}'.", e)
        else:  # uid is not None
            matches = [entity for entity in self.entities if entity.uid.match(uid, short_only=True)]
            if matches:
                (match,) = matches
                return match
            gs.raise_exception(f"Entity not found for uid: '{uid}'.")

    @property
    def emitters(self):
        """All the emitters in the scene."""
        return self._emitters

    @property
    def tool_solver(self):
        """The scene's `tool_solver`, managing all the `ToolEntity` in the scene."""
        return self._sim.tool_solver

    @property
    def rigid_solver(self):
        """The scene's `rigid_solver`, managing all the `RigidEntity` in the scene."""
        return self._sim.rigid_solver

    @property
    def kinematic_solver(self):
        """The scene's `kinematic_solver`, managing all the kinematic (visualization-only) entities in the scene."""
        return self._sim.kinematic_solver

    @property
    def mpm_solver(self):
        """The scene's `mpm_solver`, managing all the `MPMEntity` in the scene."""
        return self._sim.mpm_solver

    @property
    def sph_solver(self):
        """The scene's `sph_solver`, managing all the `SPHEntity` in the scene."""
        return self._sim.sph_solver

    @property
    def fem_solver(self):
        """The scene's `fem_solver`, managing all the `FEMEntity` in the scene."""
        return self._sim.fem_solver

    @property
    def pbd_solver(self):
        """The scene's `pbd_solver`, managing all the `PBDEntity` in the scene."""
        return self._sim.pbd_solver

    @property
    def segmentation_idx_dict(self):
        """
        Returns a dictionary mapping segmentation indices to scene entities.

        In the segmentation map:
        - Index 0 corresponds to the background (-1).
        - Indices > 0 correspond to scene elements, which may be represented as:
            - `entity_id`
            - `(entity_id, link_id)`
            - `(entity_id, link_id, geom_id)`
          depending on the material type and the configured segmentation level.
        """
        return self._visualizer.segmentation_idx_dict
