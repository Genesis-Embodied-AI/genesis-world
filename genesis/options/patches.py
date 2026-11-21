"""Patches of options set on selections of parts of an entity, and the settings recording them in an entity spec."""

from collections.abc import Iterable, Sequence
from typing import Annotated, Self

import numpy as np

from pydantic import ConfigDict, Field, StrictBool, model_validator

import genesis as gs
from genesis.engine.materials.rigid import FrictionRollingType, FrictionTorsionalType, FrictionType
from genesis.repr_base import get_public_name
from genesis.typing import ValidFloat

from . import indices
from .options import Options
from .surfaces import DynamicVertices, Surface


class _FrozenOptions(Options):
    # Frozen, since assigning a field skips the validators
    model_config = ConfigDict(frozen=True)


class Geom(_FrozenOptions):
    """Quantities of the collision geoms of a rigid entity, set on a selection of its links or geoms.

    A field given explicitly overrides, for the selected geoms, the material and the coarser selections (see
    ``Rigid.select``). Given as None, it takes the value the asset states, else the solver default. A field left unset,
    as every field is by default, keeps the value of the coarser levels. The IPC coupler rejects every patch.

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
    visualize_contact : bool | None, optional
        Whether the force and the normal of the contacts of the geoms are drawn as arrows, markers that the interactive
        viewer and the debug cameras draw, a contact being drawn when either geom of the pair shows it.
        It costs a host-side read of the contacts at each frame drawn, so reserve it for the geoms being inspected.
        Given as None, it takes the ``visualize_contact`` of the entity spec.
    """

    rho: ValidFloat | None = None
    friction: FrictionType | None = None
    friction_torsional: FrictionTorsionalType | None = None
    friction_rolling: FrictionRollingType | None = None
    visualize_contact: StrictBool | None = None


class VGeom(_FrozenOptions):
    """Appearance of the visual geoms of an entity, set on a selection of its links or visual geoms.

    Parameters
    ----------
    surface : gs.surfaces.Surface | None, optional
        The appearance of the selected visual geoms, which replaces the surface of the entity for them. The textures it
        leaves unset keep those the asset and the surface of the entity give the geoms. If None, the geoms keep the
        surface of the entity. Default is None.
    dynamic_vertices : gs.surfaces.DynamicVertices | None, optional
        The vertices of the selected visual geoms, switched at runtime between variants or written per environment (see
        ``gs.surfaces.DynamicVertices``), or None for the mesh of the asset. It replaces the ``dynamic_vertices`` of the
        entity spec for them, and only the visual geoms of rigid and kinematic entities admit it. Default is None.
    """

    surface: Surface | None = None
    dynamic_vertices: DynamicVertices | None = None


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


class Patch(_FrozenOptions):
    """Options set on a selection of parts of an entity, as ``select`` records it.

    Parameters
    ----------
    selectors : tuple[gs.indices.Link | gs.indices.Geom | gs.indices.VGeom | gs.indices.Vertex, ...]
        The selector of each level leading to the parts, from the coarsest to the finest, each resolving among the
        parts the previous one holds.
    options : gs.options.Geom | gs.options.VGeom | gs.options.Vertex
        The options set on the parts, of which the fields given explicitly apply.
    """

    selectors: Annotated[
        tuple[indices.Link | indices.Geom | indices.VGeom | indices.Vertex, ...], Field(min_length=1, strict=False)
    ]
    options: Geom | VGeom | Vertex

    @model_validator(mode="after")
    def _check_patch(self) -> Self:
        if not self.options.model_fields_set:
            gs.raise_exception(f"gs.options.{type(self.options).__name__} sets no field. Give at least one field.")
        return self

    @property
    def selection_repr(self) -> str:
        """The selection of the patch on one line, for error messages."""
        return " then ".join(
            f"{get_public_name(type(selector))}{selector.model_dump(exclude_none=True)}" for selector in self.selectors
        )


def resolve_patch_parts(
    patch: Patch,
    parts_name: list[str],
    parts_link_idx: np.ndarray | None = None,
    links_path: list[tuple[str, ...]] | None = None,
) -> np.ndarray:
    """Return whether the selection of a patch holds each part of an entity.

    Each selector resolves among the parts the selectors before it hold, an ordinal below a link selection counting the
    parts of each link apart.

    Parameters
    ----------
    patch : Patch
        The patch.
    parts_name : list[str]
        The name of each part.
    parts_link_idx : np.ndarray | None, optional
        The link of each part, which a link selector holds the parts of. Defaults to None.
    links_path : list[tuple[str, ...]] | None, optional
        The names from the tree root down to each link, which a link selector matches. Defaults to None.
    """
    is_selected = np.ones(len(parts_name), dtype=gs.np_bool)
    parts_scope = np.zeros(len(parts_name), dtype=gs.np_int)
    for selector in patch.selectors:
        if isinstance(selector, indices.Link):
            is_selected &= np.isin(parts_link_idx, selector._resolve(links_path))
            parts_scope = parts_link_idx
        else:
            held_parts_idx = np.flatnonzero(is_selected)
            is_selected[:] = False
            # A selection already empty stays empty
            if held_parts_idx.size > 0:
                held_parts_name = [parts_name[i_p] for i_p in held_parts_idx]
                held_parts_ordinal = selector._resolve(held_parts_name, parts_scope[held_parts_idx])
                is_selected[held_parts_idx[held_parts_ordinal]] = True
    return is_selected


def resolve_patches_fields(
    patches: Sequence[Patch],
    names: Iterable[str],
    parts_name: list[str],
    parts_link_idx: np.ndarray,
    links_path: list[tuple[str, ...]],
    *,
    parts_repr: str,
) -> dict[str, np.ndarray]:
    """Return, for each field name, the patch whose value each part of an entity takes, -1 where no patch sets it.

    Of the patches setting a field on a part, the one of the finest level applies, a link selection being coarser than a
    selection of the parts themselves, and the later one within a level.

    Parameters
    ----------
    patches : Sequence[Patch]
        The patches.
    names : Iterable[str]
        The names of the fields.
    parts_name : list[str]
        See 'resolve_patch_parts'.
    parts_link_idx : np.ndarray
        See 'resolve_patch_parts'.
    links_path : list[tuple[str, ...]]
        See 'resolve_patch_parts'.
    parts_repr : str
        The name of the parts and their entity in the error a patch holding none of them raises.
    """
    patches_is_selected = np.zeros((len(patches), len(parts_name)), dtype=gs.np_bool)
    patches_rank = np.arange(len(patches))
    for i_p, patch in enumerate(patches):
        patches_is_selected[i_p] = resolve_patch_parts(patch, parts_name, parts_link_idx, links_path)
        if not patches_is_selected[i_p].any():
            gs.raise_exception(
                f"The selection {patch.selection_repr} holds no {parts_repr}. Existing geoms: {parts_name}."
            )
        if not isinstance(patch.selectors[-1], indices.Link):
            patches_rank[i_p] += len(patches)
    fields_patch_idx = {}
    for name in names:
        is_setting = np.array([name in patch.options.model_fields_set for patch in patches])
        parts_rank = np.where(patches_is_selected & is_setting[:, None], patches_rank[:, None], -1)
        fields_patch_idx[name] = np.where(parts_rank.max(axis=0) >= 0, parts_rank.argmax(axis=0), -1)
    return fields_patch_idx
