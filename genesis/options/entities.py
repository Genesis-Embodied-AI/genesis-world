"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field, StrictBool, field_validator

import genesis as gs
from genesis.engine.materials import Kinematic, Rigid
from genesis.engine.materials.base import Material

from .morphs import Morph
from .options import Options
from .solvers import KinematicOptions, RigidOptions
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

    # The class of the scene options configuring the solver that simulates the entity, or None for no solver
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
