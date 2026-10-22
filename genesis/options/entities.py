"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field, StrictBool, field_validator

import genesis as gs
from genesis.engine.materials import FEM, MPM, PBD, SPH, Hybrid, Kinematic, Rigid, Tool
from genesis.engine.materials.base import Material

from .morphs import Morph
from .options import Options
from .solvers import FEMOptions, KinematicOptions, MPMOptions, PBDOptions, RigidOptions, SPHOptions, ToolOptions
from .surfaces import Surface


class EntityOptions(Options):
    """Options an entity is constructed from, as passed to ``Scene.add_entity(options=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits.
    The options are immutable: derive a variant with ``model_copy(update=...)``. The scene keeps its own copy of them
    as ``entity.options``, so one instance can construct several entities.

    Parameters
    ----------
    morph : Morph | tuple[Morph, ...]
        The geometry or asset of the entity. A sequence of morphs makes the entity heterogeneous: each parallel
        environment simulates one of them (see ``Scene.add_entity``).
    material : Material
        The physical properties of the entity, of a class the concrete subclass admits.
    surface : Surface | None, optional
        The appearance of the entity. If None, use ``gs.surfaces.Default()``.
    vis_mode : str | None, optional
        The visualization mode of the entity, overriding ``surface.vis_mode``. If None, the surface decides, and
        otherwise the kind of entity does.
    name : str | None, optional
        The name of the entity, unique within the scene. If None, a name is generated from the morph and the entity
        identifier.
    """

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    # The class of the scene options configuring the solver that simulates the entity, or None for an entity simulated
    # by no solver of its own, such as a hybrid entity, whose parts are entities of their own
    scene_solver_options_cls: ClassVar[type[Options] | None] = None

    morph: Morph | Annotated[tuple[Morph, ...], Field(strict=False)]
    material: Material
    surface: Surface | None = None
    vis_mode: str | None = None
    name: str | None = None


class RigidEntityOptions(EntityOptions):
    """Options of a rigid entity, which the rigid solver simulates with contacts, joints and constraints.

    Use it for every entity whose motion follows from physics. Its links collide, so every geom adds to the cost of
    collision detection and constraint solving. Parameters follow ``EntityOptions``, the material being a
    ``gs.materials.Rigid``, which defaults to ``gs.materials.Rigid()``, plus the following.

    Parameters
    ----------
    visualize_contact : bool, optional
        Whether to draw the contact forces applied to the entity as arrows in the viewer and rendered images. Each
        arrow is drawn every frame, so reserve it for the entities being inspected. Defaults to False.
    """

    scene_solver_options_cls = RigidOptions

    material: Rigid = Field(default_factory=Rigid)
    visualize_contact: StrictBool = False


class KinematicEntityOptions(EntityOptions):
    """Options of a kinematic entity, which follows the poses it is given and takes no part in physics.

    Use it for a ghost or a reference motion: forward kinematics is all it costs, while it collides with nothing and
    nothing pushes it. Parameters follow ``EntityOptions``, the material being a ``gs.materials.Kinematic``, which
    defaults to ``gs.materials.Kinematic()``. A rigid material is rejected, since it asks for physics.
    """

    scene_solver_options_cls = KinematicOptions

    material: Kinematic = Field(default_factory=Kinematic)

    @field_validator("material")
    @classmethod
    def _reject_rigid_material(cls, material: Kinematic) -> Kinematic:
        # 'Rigid' derives from 'Kinematic', so the field type alone admits it
        if isinstance(material, Rigid):
            gs.raise_exception("KinematicEntityOptions requires a kinematic material. Use RigidEntityOptions instead.")
        return material


class FEMEntityOptions(EntityOptions):
    """Options of a deformable entity, which the finite element method (FEM) solver simulates on a volume mesh.

    Use it for a solid whose deformation matters, such as a soft gripper pad or a rubber block. Every element of the
    volume mesh adds to the cost of each substep, so a coarse mesh simulates faster while a fine one resolves the
    deformation more accurately. Parameters follow ``EntityOptions``, the material being a ``gs.materials.FEM.Base``,
    such as ``gs.materials.FEM.Elastic``.
    """

    scene_solver_options_cls = FEMOptions

    material: FEM.Base


class MPMEntityOptions(EntityOptions):
    """Options of a particle entity, which the material point method (MPM) solver simulates on a background grid.

    Use it for a continuum that deforms without bound, tears or flows, such as sand, snow, dough or a liquid. Its
    accuracy and cost both follow the particle size and the grid resolution of the solver. Parameters follow
    ``EntityOptions``, the material being a ``gs.materials.MPM.Base``, such as ``gs.materials.MPM.Elastic``.
    """

    scene_solver_options_cls = MPMOptions

    material: MPM.Base


class SPHEntityOptions(EntityOptions):
    """Options of a liquid entity, which the smoothed particle hydrodynamics (SPH) solver simulates as particles.

    Use it for a liquid whose free surface and splashes matter. Each particle interacts with its neighbors, so the cost
    grows with the particle count that the particle size of the solver sets. Parameters follow ``EntityOptions``, the
    material being a ``gs.materials.SPH.Base``, which defaults to ``gs.materials.SPH.Liquid()``.
    """

    scene_solver_options_cls = SPHOptions

    material: SPH.Base = Field(default_factory=SPH.Liquid)


class PBDClothEntityOptions(EntityOptions):
    """Options of a cloth entity, which the position-based dynamics (PBD) solver simulates as a mesh of particles.

    Use it for a thin sheet that stretches and bends, such as a towel or a flag. The surface mesh is remeshed to the
    particle size of the solver, so a smaller particle size gives finer folds at a higher cost. Parameters follow
    ``EntityOptions``, the material being a ``gs.materials.PBD.Cloth``, which defaults to ``gs.materials.PBD.Cloth()``.
    """

    scene_solver_options_cls = PBDOptions

    material: PBD.Cloth = Field(default_factory=PBD.Cloth)


class PBDElasticEntityOptions(EntityOptions):
    """Options of an elastic solid, which the position-based dynamics (PBD) solver simulates on a volume mesh.

    Use it for a soft solid where speed and stability matter more than physically accurate stiffness, which the
    finite element method (FEM) solver offers at a higher cost (see ``FEMEntityOptions``). Parameters follow
    ``EntityOptions``, the material being a ``gs.materials.PBD.Elastic``, which defaults to
    ``gs.materials.PBD.Elastic()``.
    """

    scene_solver_options_cls = PBDOptions

    material: PBD.Elastic = Field(default_factory=PBD.Elastic)


class PBDLiquidEntityOptions(EntityOptions):
    """Options of a liquid entity, which the position-based dynamics (PBD) solver simulates as incompressible particles.

    Use it for a liquid where speed and stability matter more than accurate pressure and viscosity. Each particle
    interacts with its neighbors, so the cost grows with the particle count. Parameters follow ``EntityOptions``, the
    material being a ``gs.materials.PBD.Liquid``, which defaults to ``gs.materials.PBD.Liquid()``.
    """

    scene_solver_options_cls = PBDOptions

    material: PBD.Liquid = Field(default_factory=PBD.Liquid)


class PBDParticleEntityOptions(EntityOptions):
    """Options of a set of free particles, which the position-based dynamics (PBD) solver moves under external forces.

    Use it for particle effects and animations: the particles ignore each other, which makes them the cheapest
    particle entity, at the cost of any volume or cohesion. Parameters follow ``EntityOptions``, the material being a
    ``gs.materials.PBD.Particle``, which defaults to ``gs.materials.PBD.Particle()``.
    """

    scene_solver_options_cls = PBDOptions

    material: PBD.Particle = Field(default_factory=PBD.Particle)


class ToolEntityOptions(EntityOptions):
    """Options of a tool, a mesh whose pose the user drives and which material point method (MPM) entities collide with.

    Use it to cut, push or scoop an MPM entity along a prescribed motion. A tool takes no forces in return, so use
    a rigid entity instead where the reaction of the particles must move it. Parameters follow ``EntityOptions``, the
    material being a ``gs.materials.Tool``, which defaults to ``gs.materials.Tool()``.
    """

    scene_solver_options_cls = ToolOptions

    material: Tool = Field(default_factory=Tool)


class HybridEntityOptions(EntityOptions):
    """Options of a hybrid entity, a soft skin simulated by the material point method around a rigid skeleton.

    Use it for a soft robot actuated through its joints. The scene adds the rigid skeleton and the soft skin as two
    entities of their own, so the entity costs as much as both of them together. The morph is a URDF, whose links the
    skin is grown around, or a mesh, whose skeleton is extracted from the skin. Parameters follow ``EntityOptions``,
    the material being a ``gs.materials.Hybrid``.
    """

    material: Hybrid
