"""Blocks of quantities set on selections of parts of an entity, and the settings recording them in entity options."""

from typing import Annotated, Self

from pydantic import ConfigDict, Field, StrictBool, model_validator

import genesis as gs
from genesis.engine.materials.rigid import FrictionRollingType, FrictionTorsionalType, FrictionType
from genesis.repr_base import get_public_name
from genesis.typing import ValidFloat

from . import indices
from .options import Options


class _FrozenOptions(Options):
    # Frozen, since assigning a field skips the validators
    model_config = ConfigDict(frozen=True)


class Geom(_FrozenOptions):
    """Quantities of the collision geoms of a rigid entity, set on a selection of its links or geoms.

    The fields given explicitly override those of the entity material for the selected geoms (see ``Rigid.select``).
    The fields left unset keep theirs.

    Parameters
    ----------
    rho : float | None, optional
        The density (kg/m^3). It estimates the mass and the inertia of a link the asset states none for, or of every
        moving link under the 'recompute_inertia' option of the morph. An aligned body is then simulated unaligned.
        The IPC coupler rejects it. If None, the geoms keep the density of the material. Default is None.
    friction : float | None, optional
        The sliding friction coefficient of the contacts of the geoms (see ``gs.materials.Rigid.friction``). If None,
        the geoms keep that of the material. Default is None.
    friction_torsional : float | None, optional
        The torsional friction coefficient (see ``gs.materials.Rigid.friction_torsional``). If None, the geoms keep
        that of the material. Default is None.
    friction_rolling : float | None, optional
        The rolling friction coefficient (see ``gs.materials.Rigid.friction_rolling``). If None, the geoms keep that of
        the material. Default is None.
    """

    rho: ValidFloat | None = None
    friction: FrictionType | None = None
    friction_torsional: FrictionTorsionalType | None = None
    friction_rolling: FrictionRollingType | None = None


class Vertex(_FrozenOptions):
    """Quantities of the vertices of a deformable entity, set on a selection of its vertices.

    Parameters
    ----------
    is_fixed : bool, optional
        Whether the vertices stay at their initial positions, as a hard constraint the solver enforces at every
        substep. It holds an entity in place by a few vertices, such as a soft pad clamped by one face. The vertices are
        vertex constraints from the build on, which ``FEMEntity.remove_vertex_constraints`` releases like any other. A
        Cloth material takes no fixed vertices, since the IPC coupler simulating it has no vertex constraints. Default is
        False.
    """

    is_fixed: StrictBool = False


class SelectionBlock(_FrozenOptions):
    """A block of quantities with the selection of parts of an entity it is set on, as ``select`` records it.

    Parameters
    ----------
    selectors : tuple[gs.indices.Link | gs.indices.Geom | gs.indices.Vertex, ...]
        The selector of each level leading to the parts, from the coarsest to the finest, each resolving among the
        parts the previous one holds.
    block : gs.options.Geom | gs.options.Vertex
        The quantities set on the parts, of which the fields given explicitly apply.
    """

    selectors: Annotated[tuple[indices.Link | indices.Geom | indices.Vertex, ...], Field(min_length=1, strict=False)]
    block: Geom | Vertex

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
