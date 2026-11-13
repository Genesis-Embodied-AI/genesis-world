"""Selections of the links, geoms and vertices of an entity, keying the options set on them."""

from fnmatch import fnmatchcase
from typing import Annotated, Any, Self

import numpy as np

from pydantic import (
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    StrictBool,
    model_serializer,
    model_validator,
)

import genesis as gs
from genesis.typing import IArrayType, PositiveFloat, StrArrayType, StrictInt, Vec3FType

from .options import Options

BArrayType = Annotated[tuple[StrictBool, ...], Field(min_length=1, strict=False)]


class _Selection(Options):
    """A selection keying the options set on part of an entity.

    In Python mode, a selection dumps to a tuple of (field, value) pairs, so that it stays hashable as a key in the dump
    of the options.
    """

    @model_serializer(mode="wrap")
    def _dump_hashable(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> tuple[tuple[str, Any], ...] | dict[str, Any]:
        dumped = handler(self)
        return tuple(dumped.items()) if info.mode == "python" else dumped


def get_links_path(links_name: list[str], links_parent_idx: list[int]) -> list[tuple[str, ...]]:
    """Return the names of the links from the root of the kinematic tree down to each link, the link included.

    'links_parent_idx' gives the index of the parent of each link among the same links, -1 for a root, every parent
    preceding its children.
    """
    links_path = []
    for link_name, parent_idx in zip(links_name, links_parent_idx, strict=True):
        links_path.append((*links_path[parent_idx], link_name) if parent_idx >= 0 else (link_name,))
    return links_path


class LinkIndex(_Selection):
    """Select the links of an entity by name, or the links of the subtrees under some links.

    Each name is an exact link name or a glob pattern such as 'finger*', and a name matching no link raises an error.
    A glob metacharacter ('*', '?' or '[') in a link name is matched literally by enclosing it in brackets, so
    'arm[[]1]' selects the link 'arm[1]'. A link merged into its parent by the asset parser has no name of its own any
    more, so a selection naming it raises an error too (see 'links_to_keep' of the morph).
    """

    model_config = ConfigDict(frozen=True)

    names: Annotated[StrArrayType, Field(min_length=1)] | None = None
    subtree_roots: Annotated[StrArrayType, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _check_single_selection(self) -> Self:
        if (self.names is None) == (self.subtree_roots is None):
            gs.raise_exception("LinkIndex takes either names or subtree roots. Use 'LinkIndex.named' or 'subtree'.")
        return self

    @classmethod
    def named(cls, *names: str) -> Self:
        """Select the links of the given names, each an exact link name or a glob pattern (see ``fnmatch``)."""
        return cls(names=names)

    @classmethod
    def subtree(cls, *names: str) -> Self:
        """Select the links of the given names and every link below them in the kinematic tree, such as a gripper.

        A selection of links stands at one specificity whatever its kind, so of a subtree and a link inside it giving
        one field of one link, the first one applies, with a warning.
        """
        return cls(subtree_roots=names)

    def matches(self, link_path: tuple[str, ...]) -> bool:
        """Whether the selection holds a link, given the names from the root of its tree down to it (see 'resolve')."""
        if self.names is not None:
            return any(fnmatchcase(link_path[-1], name) for name in self.names)
        return any(fnmatchcase(link_name, name) for link_name in link_path for name in self.subtree_roots)

    def resolve(self, links_path: list[tuple[str, ...]]) -> np.ndarray:
        """Return the indices of the selected links, raising on a name matching no link.

        'links_path' holds the names from the root of the kinematic tree down to each link (see 'get_links_path').
        """
        links_name = [link_path[-1] for link_path in links_path]
        names = self.names if self.names is not None else self.subtree_roots
        unknown_names = [name for name in names if not any(fnmatchcase(l_name, name) for l_name in links_name)]
        if unknown_names:
            gs.raise_exception(
                f"No link matches {unknown_names}. Existing links: {links_name}. A link the asset parser merges into "
                "its parent keeps its name with the 'links_to_keep' option of the morph."
            )
        return np.flatnonzero([self.matches(link_path) for link_path in links_path]).astype(gs.np_int, copy=False)


class _OrdinalIndex(_Selection):
    """Select elements by their ordinal, as numpy indexes a range of the number of elements.

    The selection is ordinals, a boolean mask over every element, or a span (a slice). Each subclass checks that a
    selection states exactly one of what it admits.
    """

    model_config = ConfigDict(frozen=True)

    ordinals: IArrayType | None = None
    mask: BArrayType | None = None
    span: tuple[StrictInt | None, StrictInt | None, StrictInt | None] | None = None

    @classmethod
    def at(cls, idx: "int | slice | np.typing.ArrayLike") -> Self:
        """Select the elements a numpy index picks: an ordinal, ordinals, a boolean mask or a slice."""
        if isinstance(idx, slice):
            return cls(span=(idx.start, idx.stop, idx.step))
        idx = np.atleast_1d(idx)
        if idx.dtype == np.bool_:
            return cls(mask=tuple(idx.tolist()))
        if not np.issubdtype(idx.dtype, np.integer):
            gs.raise_exception(f"{cls.__name__}.at takes integer ordinals, a boolean mask or a slice, not {idx.dtype}.")
        return cls(ordinals=tuple(idx.tolist()))

    def _resolve_ordinals(self, n: int, domain: str) -> np.ndarray:
        """Return the sorted ordinals selected among 'n' elements, raising on an ordinal out of range or no match."""
        if self.span is not None:
            idx = slice(*self.span)
        elif self.mask is not None:
            idx = np.array(self.mask)
        else:
            idx = np.array(self.ordinals)
        try:
            ordinals = np.unique(np.arange(n, dtype=gs.np_int)[idx])
        except (IndexError, ValueError) as e:
            gs.raise_exception_from(f"{type(self).__name__} selects no valid {domain} among {n}: {self}.", e)
        if ordinals.size == 0:
            gs.raise_exception(f"{type(self).__name__} selects no {domain}: {self}.")
        return ordinals


class _NamedOrdinalIndex(_OrdinalIndex):
    """Select geoms by their ordinal in the asset or by name.

    A name is an exact name or a glob pattern (see ``LinkIndex``).
    """

    names: Annotated[StrArrayType, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _check_single_selection(self) -> Self:
        if sum(selection is not None for selection in (self.ordinals, self.mask, self.span, self.names)) != 1:
            gs.raise_exception(f"{type(self).__name__} takes one of ordinals, a mask, a span and names.")
        return self

    @classmethod
    def at(cls, idx: "int | slice | np.typing.ArrayLike") -> Self:
        """Select the collision geoms a numpy index picks: an ordinal, ordinals, a boolean mask or a slice.

        Keying the 'geoms' of entity options, the index counts the collision geoms of the whole entity, link after link
        in the order of the asset. Keying the 'geoms' of link options, it counts the collision geoms of each selected
        link alone, so 'gs.GeomIndex.at(0)' there selects the first collision geom of every link the key of the link
        options selects.
        """
        return super().at(idx)

    @classmethod
    def named(cls, *names: str) -> Self:
        """Select the geoms of the given names, each an exact geom name or a glob pattern."""
        return cls(names=names)

    def matches(self, geom_name: str) -> bool:
        """Whether a selection by name holds the geom of a given name."""
        return any(fnmatchcase(geom_name, name) for name in self.names)

    def resolve(self, geoms_name: list[str], geoms_start: int, n_geoms: int) -> np.ndarray:
        """Return the ordinals of the selected geoms among the geoms of a link, given their names.

        An ordinal counts 'n_geoms' collision geoms, of which the geoms of the link start at 'geoms_start'. A selection
        may match no geom of a link, so that it combines with a selection of several links. A selection by name that
        matches no geom of the whole entity raises when the entity is added, and so does an ordinal out of range.
        """
        if self.names is None:
            ordinals = self._resolve_ordinals(n_geoms, "geom")
            is_in_link = (ordinals >= geoms_start) & (ordinals < geoms_start + len(geoms_name))
            return ordinals[is_in_link] - geoms_start
        return np.flatnonzero([self.matches(geom_name) for geom_name in geoms_name]).astype(gs.np_int, copy=False)


class GeomIndex(_NamedOrdinalIndex):
    """Select collision geoms by their ordinal in the asset (see ``GeomIndex.at``) or by name.

    Post-processing splits a geom into convex pieces and fuses geoms with identical properties, which renumbers them,
    so the ordinal counts the collision geoms as the asset states them, and selecting one selects every piece it is
    split into. A name is matched against the names the asset gives its geoms, as an exact name or a glob pattern (see
    ``LinkIndex``). MJCF states the name of a geom and URDF states none, while USD names a geom by its full prim path,
    so a pattern such as '*/collision' selects it.
    """


class VertexIndex(_OrdinalIndex):
    """Select vertices of an entity, by ordinal or by position in the frame of its morph.

    A selection resolves once, as the entity is added, on the vertices of its simulation mesh. The frame of the morph
    holds the vertices as the morph states them once scaled, relative to its 'pos' and before its 'quat' and its
    offset place them, so a selection keeps its vertices wherever the entity starts. Combine selections with ``|``.
    """

    box: tuple[Vec3FType, Vec3FType] | None = None
    closest_pos: Vec3FType | None = None
    max_distance: PositiveFloat | None = None
    members: Annotated[tuple["VertexIndex", ...], Field(min_length=2, strict=False)] | None = None

    @model_validator(mode="after")
    def _check_single_selection(self) -> Self:
        selections = (self.ordinals, self.mask, self.span, self.box, self.closest_pos, self.members)
        if sum(selection is not None for selection in selections) != 1:
            gs.raise_exception("Select vertices with 'VertexIndex.at', 'in_box' or 'closest_to', combined with '|'.")
        if (self.closest_pos is None) != (self.max_distance is None):
            gs.raise_exception("'VertexIndex.closest_to' takes a position and a max distance together.")
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

    def __or__(self, other: "VertexIndex") -> "VertexIndex":
        if not isinstance(other, VertexIndex):
            return NotImplemented
        members = [member for index in (self, other) for member in (index.members or (index,))]
        return VertexIndex(members=members)

    def resolve(self, verts_pos: np.ndarray) -> np.ndarray:
        """Return the sorted indices of the selected vertices, given their positions in the frame of the morph."""
        if self.members is not None:
            return np.unique(np.concatenate([member.resolve(verts_pos) for member in self.members]))
        if self.box is not None:
            # A vertex on a bound counts as inside, within a rounding error of its position that scales with the mesh
            tol = 1e-6 * max(np.ptp(verts_pos, axis=0).max(), 1.0)
            lower, upper = np.array(self.box[0]) - tol, np.array(self.box[1]) + tol
            is_inside = np.all((lower <= verts_pos) & (verts_pos <= upper), axis=-1)
            if not is_inside.any():
                gs.raise_exception(f"No vertex lies inside the box from {self.box[0]} to {self.box[1]}.")
            return np.flatnonzero(is_inside).astype(gs.np_int, copy=False)
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
