"""Blocks of quantities set on selections of parts of an entity, and the settings recording them in entity options."""

from typing import Annotated, Self

from pydantic import ConfigDict, Field, StrictBool, model_validator

import genesis as gs
from genesis.engine.materials.rigid import FrictionRollingType, FrictionTorsionalType, FrictionType
from genesis.repr_base import get_public_name
from genesis.typing import ValidFloat

from . import indices
from .options import Options
from .surfaces import Surface


class _FrozenOptions(Options):
    # Frozen, since assigning a field skips the validators
    model_config = ConfigDict(frozen=True)


class Geom(_FrozenOptions):
    """Quantities of the collision geoms of a rigid entity, set on a selection of its links or geoms.

    A field given explicitly overrides, for the selected geoms, the material and the coarser selections (see
    ``Rigid.select``). Given as None, it takes the value the asset states, else the solver default. A field left unset,
    as every field is by default, keeps the value of the coarser levels. The IPC coupler rejects every block.

    Parameters
    ----------
    rho : float | None, optional
        The density (kg/m^3). It estimates the mass and the inertia of a link the asset states none for, or of every
        moving link under the 'recompute_inertia' option of the morph. An aligned body is then simulated unaligned.
    friction : float | None, optional
        The sliding friction coefficient of the contacts of the geoms (see ``gs.materials.Rigid.friction``).
    friction_torsional : float | None, optional
        The torsional friction coefficient (see ``gs.materials.Rigid.friction_torsional``).
    friction_rolling : float | None, optional
        The rolling friction coefficient (see ``gs.materials.Rigid.friction_rolling``).
    """

    rho: ValidFloat | None = None
    friction: FrictionType | None = None
    friction_torsional: FrictionTorsionalType | None = None
    friction_rolling: FrictionRollingType | None = None


class VGeom(_FrozenOptions):
    """Appearance of the visual geoms of a rigid or kinematic entity, set on a selection of its links or visual geoms.

    Parameters
    ----------
    surface : gs.surfaces.Surface | None, optional
        The appearance of the selected visual geoms, which replaces the surface of the entity for them. The textures it
        leaves unset keep those the asset and the surface of the entity give the geoms. If None, the geoms keep the
        surface of the entity. Default is None.
    """

    surface: Surface | None = None


class Vertex(_FrozenOptions):
    """Quantities of the vertices of a deformable entity, set on a selection of its vertices.

    Parameters
    ----------
    is_fixed : bool, optional
        Whether the vertices stay at their initial positions, as a hard constraint the solver enforces at every
        substep, such as a soft pad clamped by one face. They are vertex constraints from the build on, which
        ``FEMEntity.remove_vertex_constraints`` releases like any other. A Cloth material takes no fixed vertices, since
        the IPC coupler simulating it has no vertex constraints. Default is False.
    """

    is_fixed: StrictBool = False


class SelectionBlock(_FrozenOptions):
    """A block of quantities with the selection of parts of an entity it is set on, as ``select`` records it.

    Parameters
    ----------
    selectors : tuple[gs.indices.Link | gs.indices.Geom | gs.indices.VGeom | gs.indices.Vertex, ...]
        The selector of each level leading to the parts, from the coarsest to the finest, each resolving among the
        parts the previous one holds.
    block : gs.options.Geom | gs.options.VGeom | gs.options.Vertex
        The quantities set on the parts, of which the fields given explicitly apply.
    """

    selectors: Annotated[
        tuple[indices.Link | indices.Geom | indices.VGeom | indices.Vertex, ...], Field(min_length=1, strict=False)
    ]
    block: Geom | VGeom | Vertex

    @model_validator(mode="after")
    def _check_block(self) -> Self:
        if not self.block.model_fields_set:
            gs.raise_exception(f"gs.options.{type(self.block).__name__} sets no quantity. Give at least one field.")
        return self

    @property
    def selection_repr(self) -> str:
        """The selection of the block on one line, for error messages."""
        return " then ".join(
            f"{get_public_name(type(selector))}{selector.model_dump(exclude_none=True)}" for selector in self.selectors
        )
