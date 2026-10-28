"""Articulated geometry and state of the kinematic and rigid solvers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from genesis.utils import array_class

from . import SystemData, SystemDataDescription

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


@dataclass(frozen=True, kw_only=True, eq=False)
class ArticulatedData(SystemData):
    info: array_class.DynInfo
    state: array_class.DynState


@dataclass(frozen=True, kw_only=True, eq=False)
class LinksData(SystemData):
    data: ArticulatedData
    link_start: int
    link_end: int


@dataclass(frozen=True, kw_only=True, eq=False)
class JointsData(SystemData):
    data: ArticulatedData
    joint_start: int
    joint_end: int
    dof_start: int
    dof_end: int
    q_start: int
    q_end: int


@dataclass(frozen=True, kw_only=True, eq=False)
class GeomData(SystemData):
    """The link of a geometry and its vertex and face ranges."""

    data: ArticulatedData
    link_idx: int
    vert_start: int
    vert_end: int
    face_start: int
    face_end: int


@dataclass(frozen=True, kw_only=True, eq=False)
class VisualGeomData(GeomData):
    """The ranges index the visual vertices and faces."""


@dataclass(frozen=True, kw_only=True, eq=False)
class RigidGeomData(GeomData):
    """The ranges index the collision vertices and faces, and the vertex state is the free or the fixed one."""

    verts_state: array_class.VertsState
    verts_state_start: int
    verts_state_end: int
    edge_start: int
    edge_end: int


@dataclass(frozen=True, kw_only=True)
class ArticulatedDescription(SystemDataDescription[ArticulatedData]):
    """Collision geometry and equality counts stay at zero for a kinematic solver, which allocates none of them."""

    _B: int
    n_dofs_: int
    n_qs_: int
    n_links_: int
    n_joints_: int
    n_entities_: int
    n_vverts_: int
    n_custom_vverts_: int
    n_vfaces_: int
    n_vgeoms_: int
    n_joints_per_link_max: int
    n_geoms_: int = 0
    n_verts_: int = 0
    n_faces_: int = 0
    n_edges_: int = 0
    n_free_verts_: int = 0
    n_fixed_verts_: int = 0
    n_candidate_equalities_: int = 0
    is_dynamic: bool = False
    requires_grad: bool
    is_batch_links_info: bool
    is_batch_dofs_info: bool
    is_batch_joints_info: bool

    def allocate(self, owner: "Solver") -> ArticulatedData:
        # Leaf structs are allocated one by one before assembling the per-component aggregates, to control the
        # allocation order of the underlying buffers (see the warning in get_constraint_state).
        dofs_info = array_class.get_dofs_info(self)
        dofs_state = array_class.get_dofs_state(self)
        links_info = array_class.get_links_info(self)
        links_state = array_class.get_links_state(self)
        joints_info = array_class.get_joints_info(self)
        joints_state = array_class.get_joints_state(self)

        entities_info = array_class.get_entities_info(self)
        entities_state = array_class.get_entities_state(self)

        vverts_info = array_class.get_vverts_info(self)
        vverts_state = array_class.get_vverts_state(self)
        vfaces_info = array_class.get_vfaces_info(self)

        vgeoms_info = array_class.get_vgeoms_info(self)
        vgeoms_state = array_class.get_vgeoms_state(self)

        geoms_info = array_class.get_geoms_info(self, self.is_dynamic)
        geoms_state = array_class.get_geoms_state(self, self.is_dynamic)

        verts_info = array_class.get_verts_info(self, self.is_dynamic)
        faces_info = array_class.get_faces_info(self, self.is_dynamic)
        edges_info = array_class.get_edges_info(self, self.is_dynamic)

        free_verts_state = array_class.get_free_verts_state(self, self.is_dynamic)
        fixed_verts_state = array_class.get_fixed_verts_state(self, self.is_dynamic)

        equalities_info = array_class.get_equalities_info(self, self.is_dynamic)

        info = array_class.DynInfo(
            entities=entities_info,
            links=links_info,
            joints=joints_info,
            dofs=dofs_info,
            geoms=geoms_info,
            verts=verts_info,
            faces=faces_info,
            edges=edges_info,
            vverts=vverts_info,
            vfaces=vfaces_info,
            vgeoms=vgeoms_info,
            equalities=equalities_info,
        )
        state = array_class.DynState(
            entities=entities_state,
            links=links_state,
            joints=joints_state,
            dofs=dofs_state,
            geoms=geoms_state,
            free_verts=free_verts_state,
            fixed_verts=fixed_verts_state,
            vverts=vverts_state,
            vgeoms=vgeoms_state,
        )

        return ArticulatedData(owner=owner, info=info, state=state)
