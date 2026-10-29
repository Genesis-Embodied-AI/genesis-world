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


def get_links_path(links_name: list[str], links_parent_idx: list[int]) -> list[tuple[str, ...]]:
    """Return the names of the links from the root of the kinematic tree down to each link, the link included.

    'links_parent_idx' gives the index of the parent of each link among the same links, -1 for a root, every parent
    preceding its children.
    """
    links_path = []
    for link_name, parent_idx in zip(links_name, links_parent_idx, strict=True):
        links_path.append((*links_path[parent_idx], link_name) if parent_idx >= 0 else (link_name,))
    return links_path


class LinkIndex(Options):
    """Select the links of an entity by name, or the links of the subtrees under some links.

    Each name is an exact link name or a glob pattern such as 'finger*', and a name matching no link raises an error.
    A glob metacharacter ('*', '?' or '[') in a link name is matched literally by enclosing it in brackets, so
    'arm[[]1]' selects the link 'arm[1]'. A name or a tuple of names keying the 'links' of entity options stands for
    ``LinkIndex.named(*names)``. A link merged into its parent by the asset parser has no name of its own any more, so a
    selection naming it raises an error too (see 'links_to_keep' of the morph).
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

        A selection of links stands at one specificity whatever its kind, so a subtree and a link inside it giving
        one field of one link raise an error.
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


class _NamedOrdinalIndex(_OrdinalIndex):
    """Select the geoms of a link by their ordinal in the asset, counted within the link, or by name.

    A name is an exact name or a glob pattern (see ``LinkIndex``).
    """

    names: Annotated[StrArrayType, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _check_names_alone(self) -> Self:
        if self.names is not None and not super().is_all:
            gs.raise_exception(f"{type(self).__name__} takes one of ordinals, a mask, a span and names.")
        return self

    @classmethod
    def all(cls) -> Self:
        """Select every geom."""
        return cls()

    @classmethod
    def named(cls, *names: str) -> Self:
        """Select the geoms of the given names, each an exact geom name or a glob pattern."""
        return cls(names=names)

    @property
    def is_all(self) -> bool:
        """Whether the selection holds every element."""
        return super().is_all and self.names is None

    def matches(self, geom_name: str) -> bool:
        """Whether a selection by name holds the geom of a given name."""
        return any(fnmatchcase(geom_name, name) for name in self.names)

    def resolve(self, geoms_name: list[str]) -> np.ndarray:
        """Return the ordinals of the selected geoms among the geoms of a link, given their names.

        A selection by name may match no geom of a link, so that it combines with a selection of several links. A
        selection by name that matches no geom of the whole entity raises when the entity is added.
        """
        if self.names is None:
            return self._resolve_ordinals(len(geoms_name), "geom")
        return np.flatnonzero([self.matches(geom_name) for geom_name in geoms_name]).astype(gs.np_int, copy=False)


class GeomIndex(_NamedOrdinalIndex):
    """Select the collision geoms of a link by their ordinal in the asset, counted within the link, or by name.

    Post-processing splits a geom into convex pieces and fuses geoms with identical properties, which renumbers them,
    so the ordinal counts the collision geoms as the asset states them, and selecting one selects every piece it is
    split into. A name is matched against the names the asset gives its geoms, as an exact name or a glob pattern (see
    ``LinkIndex``). MJCF states the name of a geom and URDF states none, while USD names a geom by its full prim path,
    so a pattern such as '*/collision' selects it.
    """


class VGeomIndex(_NamedOrdinalIndex):
    """Select the visual geoms of a link by their ordinal in the asset, counted within the link, by name or by material.

    A mesh asset holding several materials loads as one visual geom per material, so the ordinal counts the visual
    geoms as the asset splits them. A name is matched against the names the asset gives its geoms (see ``GeomIndex``).
    A material name is matched against the name of the material the asset assigns a visual geom: the material of a
    glTF primitive, an OBJ or a URDF, the 'material' attribute of an MJCF geom, or the prim path of the material bound
    to a USD mesh. Names and material names are exact names or glob patterns (see ``LinkIndex``).
    """

    material_names: Annotated[StrArrayType, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _check_material_names_alone(self) -> Self:
        if self.material_names is not None and not super().is_all:
            gs.raise_exception("VGeomIndex takes one of ordinals, a mask, a span, names and material names.")
        return self

    @classmethod
    def with_material(cls, *names: str) -> Self:
        """Select the visual geoms whose asset material has one of the given names, each exact or a glob pattern."""
        return cls(material_names=names)

    @property
    def is_all(self) -> bool:
        """Whether the selection holds every element."""
        return super().is_all and self.material_names is None

    def matches_material(self, material_name: str) -> bool:
        """Whether a selection by material name holds the visual geom of a given asset material name."""
        return any(fnmatchcase(material_name, name) for name in self.material_names)

    def resolve(self, vgeoms_name: list[str], vgeoms_material_name: list[str]) -> np.ndarray:
        """Return the ordinals of the selected visual geoms of a link, given their names and asset material names.

        A selection by name or by material may match no visual geom of a link (see ``GeomIndex.resolve``).
        """
        if self.material_names is None:
            return super().resolve(vgeoms_name)
        is_selected = [self.matches_material(material_name) for material_name in vgeoms_material_name]
        return np.flatnonzero(is_selected).astype(gs.np_int, copy=False)


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
