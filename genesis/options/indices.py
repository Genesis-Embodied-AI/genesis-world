"""Selections of the links, geoms and vertices of an entity, keying the options set on them."""

from fnmatch import fnmatchcase
from typing import Annotated

import numpy as np

from pydantic import ConfigDict, Field, StrictBool, model_validator
from typing_extensions import Self

import genesis as gs
from genesis.typing import IArrayType, PositiveFloat, StrArrayType, StrictInt, Vec3FType

from .options import Options

BArrayType = Annotated[tuple[StrictBool, ...], Field(min_length=1, strict=False)]


class Slice(Options):
    """A slice of a range of elements, as Python slices one (``slice(start, stop, step)``)."""

    model_config = ConfigDict(frozen=True)

    start: StrictInt | None = None
    stop: StrictInt | None = None
    step: StrictInt | None = None


class LinkIndex(Options):
    """Select the links of an entity by name, each name being an exact link name or a glob pattern such as 'finger*'.

    A name or a tuple of names keying the 'links' of entity options stands for ``LinkIndex.named(*names)``. A name that
    matches no link raises an error. A link merged into its parent by the asset parser has no name of its own any more,
    so a selection naming it raises an error too (see 'links_to_keep' of the morph).
    """

    model_config = ConfigDict(frozen=True)

    names: Annotated[StrArrayType, Field(min_length=1)]

    @classmethod
    def named(cls, *names: str) -> Self:
        """Select the links of the given names, each an exact link name or a glob pattern (see ``fnmatch``)."""
        return cls(names=names)

    def matches(self, link_name: str) -> bool:
        """Whether the selection holds the link of a given name."""
        return any(fnmatchcase(link_name, name) for name in self.names)

    def resolve(self, links_name: list[str]) -> np.ndarray:
        """Return the indices of the selected links among links of the given names, raising on a name matching none."""
        unknown_names = [name for name in self.names if not any(fnmatchcase(l_name, name) for l_name in links_name)]
        if unknown_names:
            gs.raise_exception(
                f"No link matches {unknown_names}. Existing links: {links_name}. A link the asset parser merges into "
                "its parent keeps its name with the 'links_to_keep' option of the morph."
            )
        return np.flatnonzero([self.matches(link_name) for link_name in links_name]).astype(gs.np_int, copy=False)


class _OrdinalIndex(Options):
    """Select elements by their ordinal, as numpy indexes a range of the number of elements.

    The selection is ordinals, a boolean mask over every element, or a span (a slice), at most one of them. None of them
    selects every element.
    """

    model_config = ConfigDict(frozen=True)

    ordinals: IArrayType | None = None
    mask: BArrayType | None = None
    span: Slice | None = None

    @model_validator(mode="after")
    def _check_single_selection(self) -> Self:
        if sum(selection is not None for selection in (self.ordinals, self.mask, self.span)) > 1:
            gs.raise_exception(f"{type(self).__name__} takes one of ordinals, a mask and a span.")
        return self

    @classmethod
    def at(cls, idx: "int | slice | np.typing.ArrayLike") -> Self:
        """Select the elements a numpy index picks: an ordinal, ordinals, a boolean mask or a slice."""
        if isinstance(idx, slice):
            return cls(span=Slice(start=idx.start, stop=idx.stop, step=idx.step))
        idx = np.atleast_1d(idx)
        if idx.dtype == np.bool_:
            return cls(mask=tuple(idx.tolist()))
        return cls(ordinals=tuple(idx.tolist()))

    def _resolve_ordinals(self, n: int, domain: str) -> np.ndarray:
        """Return the sorted ordinals selected among 'n' elements, raising on an ordinal out of range or no match.

        A selection of every element selects none among no element, without raising.
        """
        if self.is_all:
            return np.arange(n, dtype=gs.np_int)
        if self.span is not None:
            idx = slice(self.span.start, self.span.stop, self.span.step)
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

    @property
    def is_all(self) -> bool:
        """Whether the selection holds every element."""
        return self.ordinals is None and self.mask is None and self.span is None


class GeomIndex(_OrdinalIndex):
    """Select the collision geoms of a link by their ordinal in the asset, counted within the link.

    Post-processing splits a geom into convex pieces and fuses geoms with identical properties, which renumbers them,
    so the ordinal counts the collision geoms as the asset states them, and selecting one selects every piece it is
    split into.
    """

    @classmethod
    def all(cls) -> Self:
        """Select every collision geom."""
        return cls()

    def resolve(self, n_geoms: int) -> np.ndarray:
        """Return the ordinals of the selected geoms among 'n_geoms' collision geoms of a link."""
        return self._resolve_ordinals(n_geoms, "geom")


class VertexIndex(Options):
    """Select vertices of an entity, by ordinal or by position in the frame of its morph.

    A selection resolves once, as the entity is added, on the vertices of its simulation mesh. The frame of the morph
    holds the vertices as the morph states them once scaled, relative to its 'pos' and before its 'quat' and its
    offset place them, so a selection keeps its vertices wherever the entity starts. Combine selections with ``|``.
    """

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _reject_base(self) -> Self:
        if type(self) is VertexIndex:
            gs.raise_exception("Select vertices with 'VertexIndex.at', 'in_box' or 'closest_to'.")
        return self

    @classmethod
    def at(cls, idx: "int | slice | np.typing.ArrayLike") -> "VertexOrdinalIndex":
        """Select the vertices a numpy index picks: an ordinal, ordinals, a boolean mask or a slice."""
        return VertexOrdinalIndex.at(idx)

    @classmethod
    def in_box(cls, lower: Vec3FType, upper: Vec3FType) -> "VertexBoxIndex":
        """Select the vertices inside an axis-aligned box of the frame of the morph, bounds included."""
        return VertexBoxIndex(lower=lower, upper=upper)

    @classmethod
    def closest_to(cls, pos: Vec3FType, max_distance: float) -> "VertexClosestIndex":
        """Select the vertex closest to a point of the frame of the morph, which must lie within 'max_distance'."""
        return VertexClosestIndex(pos=pos, max_distance=max_distance)

    def __or__(self, other: "VertexIndex") -> "VertexUnionIndex":
        if not isinstance(other, VertexIndex):
            return NotImplemented
        members = []
        for index in (self, other):
            members.extend(index.members if isinstance(index, VertexUnionIndex) else (index,))
        return VertexUnionIndex(members=members)

    def resolve(self, verts_pos: np.ndarray) -> np.ndarray:
        """Return the sorted indices of the selected vertices, given their positions in the frame of the morph."""
        raise NotImplementedError


class VertexOrdinalIndex(_OrdinalIndex, VertexIndex):
    """Select vertices by ordinal (see ``VertexIndex.at``)."""

    def resolve(self, verts_pos: np.ndarray) -> np.ndarray:
        return self._resolve_ordinals(len(verts_pos), "vertex")


class VertexBoxIndex(VertexIndex):
    """Select the vertices inside an axis-aligned box (see ``VertexIndex.in_box``)."""

    lower: Vec3FType
    upper: Vec3FType

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if any(lower > upper for lower, upper in zip(self.lower, self.upper)):
            gs.raise_exception(f"The lower corner {self.lower} of a box exceeds its upper corner {self.upper}.")
        return self

    def resolve(self, verts_pos: np.ndarray) -> np.ndarray:
        # A vertex on a bound counts as inside, within the rounding error of its position, which scales with the mesh
        tol = 1e-6 * max(np.ptp(verts_pos, axis=0).max(), 1.0)
        lower, upper = np.array(self.lower) - tol, np.array(self.upper) + tol
        is_inside = np.all((lower <= verts_pos) & (verts_pos <= upper), axis=-1)
        if not is_inside.any():
            gs.raise_exception(f"No vertex lies inside the box from {self.lower} to {self.upper}.")
        return np.flatnonzero(is_inside).astype(gs.np_int, copy=False)


class VertexClosestIndex(VertexIndex):
    """Select the vertex closest to a point (see ``VertexIndex.closest_to``)."""

    pos: Vec3FType
    max_distance: PositiveFloat

    def resolve(self, verts_pos: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(verts_pos - np.array(self.pos), axis=-1)
        i_v = np.argmin(distances)
        if distances[i_v] > self.max_distance:
            gs.raise_exception(
                f"The vertex closest to {self.pos} lies {distances[i_v]:.4g} m away, beyond max_distance "
                f"{self.max_distance}."
            )
        if np.count_nonzero(distances <= distances[i_v] + gs.EPS) > 1:
            gs.raise_exception(f"Several vertices are the closest to {self.pos}. Move the point to pick one.")
        return np.array([i_v], dtype=gs.np_int)


class VertexUnionIndex(VertexIndex):
    """Select the vertices any of several selections picks (see ``VertexIndex.__or__``)."""

    members: Annotated[tuple[VertexIndex, ...], Field(min_length=2, strict=False)]

    def resolve(self, verts_pos: np.ndarray) -> np.ndarray:
        return np.unique(np.concatenate([member.resolve(verts_pos) for member in self.members]))
