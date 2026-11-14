"""Selectors of the links, geoms and vertices of an entity, which entity options select parts of it with."""

from fnmatch import fnmatchcase
from functools import reduce
from typing import Annotated, Self

import numpy as np

from pydantic import ConfigDict, Field, StrictBool, model_validator

import genesis as gs
from genesis.repr_base import get_public_name
from genesis.typing import IArrayType, PositiveFloat, StrArrayType, StrictInt, Vec3FType

from .options import Options


class _Selection(Options):
    """A selector of the parts of an entity, which entity options select with (see ``gs.entities.Rigid.select``).

    Selectors of one type combine with ``|``, holding the parts either holds, and with ``&``, holding the parts both
    hold. A combination keeps its members as given and resolves them as the entity is added.
    """

    model_config = ConfigDict(frozen=True)

    members: Annotated[tuple[Self, ...], Field(min_length=2, strict=False)] | None = None
    is_intersection: StrictBool = False

    @model_validator(mode="after")
    def _check_members(self) -> Self:
        if self.is_intersection and self.members is None:
            gs.raise_exception("An intersection of selectors takes members.")
        return self

    def __or__(self, other: Self) -> Self:
        return self._combine(other, is_intersection=False)

    def __and__(self, other: Self) -> Self:
        return self._combine(other, is_intersection=True)

    def _combine(self, other: Self, is_intersection: bool) -> Self:
        if type(other) is not type(self):
            return NotImplemented
        # A combination of the same operator flattens, so that 'a | b | c' holds three members
        members = [
            member
            for selector in (self, other)
            for member in (
                selector.members
                if selector.members is not None and selector.is_intersection == is_intersection
                else (selector,)
            )
        ]
        return type(self)(members=tuple(members), is_intersection=is_intersection)

    def _resolve_members(self, parts_idx: list[np.ndarray]) -> np.ndarray:
        """Combine the indices of the parts each member holds, as the operator of the combination does."""
        return reduce(np.intersect1d if self.is_intersection else np.union1d, parts_idx)


class Link(_Selection):
    """Select the links of an entity by name, or the links of the subtrees under some links (see ``named``)."""

    names: Annotated[StrArrayType, Field(min_length=1)] | None = None
    is_subtree: StrictBool = False

    @model_validator(mode="after")
    def _check_single_selection(self) -> Self:
        if (self.names is None) == (self.members is None):
            gs.raise_exception("gs.indices.Link takes names, or members combined with '|' or '&'.")
        return self

    @classmethod
    def named(cls, *names: str) -> Self:
        """Select the links of the given names.

        A name is an exact link name or a glob pattern such as 'finger*', matched case-sensitively whatever the
        platform. A literal '*', '?' or '[' in a link name is matched by enclosing it in brackets, so 'arm[[]1]' selects
        the link 'arm[1]'. A name matching no link raises in ``add_entity``. A link the asset parser merges into its
        parent loses its name, so keep it with the 'links_to_keep' option of the morph to select it.
        """
        return cls(names=names)

    @classmethod
    def subtree(cls, *names: str) -> Self:
        """Select the links of the given names and every link below them in the kinematic tree, such as a gripper.

        The names match as for ``named``. Of a subtree and a selection inside it setting one field of one geom, the
        finer level applies, then the later call (see ``gs.entities.Rigid.select``).
        """
        return cls(names=names, is_subtree=True)

    def _resolve(self, links_path: list[tuple[str, ...]]) -> np.ndarray:
        if self.names is None:
            assert self.members is not None
            return self._resolve_members([member._resolve(links_path) for member in self.members])
        links_name = [link_path[-1] for link_path in links_path]
        unknown_names = [name for name in self.names if not any(fnmatchcase(l_name, name) for l_name in links_name)]
        if unknown_names:
            gs.raise_exception(
                f"No link matches {unknown_names}. Existing links: {links_name}. A link the asset parser merges into "
                "its parent keeps its name with the 'links_to_keep' option of the morph."
            )
        # A subtree holds every link whose path from the root of its tree passes through one of its roots
        links_name_checked = links_path if self.is_subtree else [(link_name,) for link_name in links_name]
        is_selected = [
            any(fnmatchcase(link_name, name) for link_name in link_names for name in self.names)
            for link_names in links_name_checked
        ]
        return np.flatnonzero(is_selected)


class _OrdinalIndex(_Selection):
    """Select elements by their ordinal, as numpy indexes a range of the number of elements.

    The selection is ordinals or a span (a slice). Each subclass checks that a selection states exactly one.
    """

    ordinals: IArrayType | None = None
    span: tuple[StrictInt | None, StrictInt | None, StrictInt | None] | None = None

    @classmethod
    def at(cls, idx: "int | slice | np.typing.ArrayLike") -> Self:
        """Select the elements a numpy index picks: an ordinal, ordinals or a slice.

        The index is local to the selection it is selected in, as the ``*_idx_local`` arguments of the entity getters
        are local to the entity. Selected on the entity options, it counts the elements of the whole entity. Selected on
        a link selection, it counts the collision geoms of each selected link apart, so
        ``options.select(fingers).select(gs.indices.Geom.at(0))`` selects the first geom of every finger. The elements
        count in the order the entity lists them: its collision geoms link after link in the order of ``entity.links``,
        as the asset states them before post-processing (see ``Geom``), or the vertices of its volume mesh.
        """
        if isinstance(idx, slice):
            return cls(span=(idx.start, idx.stop, idx.step))
        idx = np.atleast_1d(idx)
        if not np.issubdtype(idx.dtype, np.integer):
            gs.raise_exception(
                f"gs.{get_public_name(cls)}.at takes integer ordinals or a slice, not {idx.dtype}. Convert a boolean "
                "mask with 'np.flatnonzero'."
            )
        return cls(ordinals=tuple(idx.tolist()))

    def _resolve_ordinals(self, n: int, domain: str) -> np.ndarray:
        """Return the sorted ordinals selected among 'n' elements, raising on an ordinal out of range."""
        if self.span is not None:
            idx = slice(*self.span)
        else:
            idx = np.array(self.ordinals)
        try:
            ordinals = np.unique(np.arange(n, dtype=gs.np_int)[idx])
        except (IndexError, ValueError) as e:
            gs.raise_exception_from(f"gs.{get_public_name(type(self))} selects no valid {domain} among {n}: {self}.", e)
        return ordinals


class _NamedOrdinalIndex(_OrdinalIndex):
    """Select geoms by their ordinal in the asset or by name (see ``named``)."""

    names: Annotated[StrArrayType, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _check_single_selection(self) -> Self:
        if sum(selection is not None for selection in (self.ordinals, self.span, self.names, self.members)) != 1:
            gs.raise_exception(
                f"gs.{get_public_name(type(self))} takes one of ordinals, a span, names and members combined with "
                "'|' or '&'."
            )
        return self

    @classmethod
    def named(cls, *names: str) -> Self:
        """Select the geoms of the given names.

        A name is matched against the names the asset gives its geoms, as an exact name or a glob pattern such as
        'pad*', case-sensitively whatever the platform. A literal '*', '?' or '[' in a geom name is matched by enclosing
        it in brackets, so 'pad[[]1]' selects the geom 'pad[1]'. A geom the asset leaves unnamed, such as an MJCF geom
        stating a class alone or any URDF geom, cannot be selected by name, so select it by ordinal (see ``at``). USD
        names a geom by its full prim path, so a pattern such as '*/collision' selects it. A name matching no geom of
        the entity raises in ``add_entity``.
        """
        return cls(names=names)

    def _resolve(self, geoms_name: list[str], geoms_scope: np.ndarray) -> np.ndarray:
        if self.members is not None:
            return self._resolve_members([member._resolve(geoms_name, geoms_scope) for member in self.members])
        if self.names is None:
            scopes_geoms_idx = [np.flatnonzero(geoms_scope == scope) for scope in np.unique(geoms_scope)]
            return np.sort(
                np.concatenate(
                    [
                        scope_geoms_idx[self._resolve_ordinals(len(scope_geoms_idx), "geom")]
                        for scope_geoms_idx in scopes_geoms_idx
                    ]
                )
            )
        # An unnamed geom, whose name is empty, matches no pattern, not even '*'
        unknown_names = [
            name for name in self.names if not any(g_name and fnmatchcase(g_name, name) for g_name in geoms_name)
        ]
        if unknown_names:
            gs.raise_exception(f"No geom matches {unknown_names}. Existing geoms: {geoms_name}.")
        is_selected = [
            bool(geom_name) and any(fnmatchcase(geom_name, name) for name in self.names) for geom_name in geoms_name
        ]
        return np.flatnonzero(is_selected)


class Geom(_NamedOrdinalIndex):
    """Select collision geoms by their ordinal in the asset or by name.

    An ordinal counts the collision geoms of the selection it is selected in (see ``at``). Post-processing splits a geom
    into convex pieces and fuses geoms with identical properties, which renumbers ``entity.geoms``, so an ordinal counts
    the geoms as the asset states them, and selecting one selects every piece of it. Geoms whose values differ never
    fuse, so a value set on a selection never reaches a geom it does not hold.

    Names match as ``named`` states.
    """


class Vertex(_OrdinalIndex):
    """Select vertices of an entity, by ordinal or by position in the frame of its morph.

    A selection resolves once, as the entity is added, on the vertices of its simulation mesh. The frame of the morph
    holds the vertices as the morph states them once scaled, relative to its 'pos' and before its 'quat' and its
    offset place them, so a selection keeps its vertices wherever the entity starts.
    """

    box: tuple[Vec3FType, Vec3FType] | None = None
    closest_pos: Vec3FType | None = None
    max_distance: PositiveFloat | None = None

    @model_validator(mode="after")
    def _check_single_selection(self) -> Self:
        selections = (self.ordinals, self.span, self.box, self.closest_pos, self.members)
        if sum(selection is not None for selection in selections) != 1:
            gs.raise_exception(
                "Select vertices with one of 'Vertex.at', 'in_box' and 'closest_to', combined with '|' or '&'."
            )
        if (self.closest_pos is None) != (self.max_distance is None):
            gs.raise_exception("'Vertex.closest_to' takes a position and a max distance together.")
        if self.box is not None and any(lower > upper for lower, upper in zip(*self.box)):
            gs.raise_exception(f"The lower corner {self.box[0]} of a box exceeds its upper corner {self.box[1]}.")
        return self

    @classmethod
    def in_box(cls, lower: Vec3FType, upper: Vec3FType) -> Self:
        """Select the vertices inside an axis-aligned box of the frame of the morph, bounds included."""
        return cls(box=(lower, upper))

    @classmethod
    def closest_to(cls, pos: Vec3FType, max_distance: float) -> Self:
        """Select the vertex closest to a point of the frame of the morph, which must lie within 'max_distance'."""
        return cls(closest_pos=pos, max_distance=max_distance)

    def _resolve(self, verts_pos: np.ndarray) -> np.ndarray:
        if self.members is not None:
            return self._resolve_members([member._resolve(verts_pos) for member in self.members])
        if self.box is not None:
            # A vertex on a bound counts as inside, within a rounding error of its position that scales with the mesh
            tol = 1e-6 * max(np.ptp(verts_pos, axis=0).max(), 1.0)
            lower, upper = np.array(self.box[0]) - tol, np.array(self.box[1]) + tol
            is_inside = np.all((lower <= verts_pos) & (verts_pos <= upper), axis=-1)
            if not is_inside.any():
                gs.raise_exception(f"No vertex lies inside the box from {self.box[0]} to {self.box[1]}.")
            return np.flatnonzero(is_inside)
        if self.closest_pos is not None:
            distances = np.linalg.norm(verts_pos - np.array(self.closest_pos), axis=-1)
            i_v = np.argmin(distances)
            if distances[i_v] > self.max_distance:
                gs.raise_exception(
                    f"The vertex closest to {self.closest_pos} lies {distances[i_v]:.4g} m away, beyond max_distance "
                    f"{self.max_distance}."
                )
            if np.count_nonzero(distances <= distances[i_v] + gs.EPS) > 1:
                gs.raise_exception(
                    f"Several vertices are the closest to {self.closest_pos}. Move the point to pick one."
                )
            return np.array([i_v], dtype=gs.np_int)
        return self._resolve_ordinals(len(verts_pos), "vertex")
