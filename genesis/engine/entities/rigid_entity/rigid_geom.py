from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import torch

import skimage
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.mesh import Mesh
from genesis.repr_base import RBC
from genesis.utils.misc import DeprecationError, qd_to_torch, tensor_to_array

from .description import RigidGeomDescription, RigidVisGeomDescription

if TYPE_CHECKING:
    from genesis.engine.materials.rigid import Rigid as RigidMaterial
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    from .rigid_entity import RigidEntity
    from .rigid_link import RigidLink


NUM_VERTS_VISUAL_GEOM_AABB = 200


class RigidGeom(RBC):
    """
    A `RigidGeom` is the basic building block of a `RigidEntity` for collision checking. It is usually constructed
    from a single mesh. This can be accessed via `link.geoms`.
    """

    def __init__(
        self,
        link: "RigidLink",
        idx,
        vert_start: int,
        face_start: int,
        edge_start: int,
        verts_state_start: int,
        needs_coup: bool,
        desc: RigidGeomDescription,
    ):
        mesh = desc.mesh
        self.desc: RigidGeomDescription = desc
        self._link: "RigidLink" = link
        self._entity: "RigidEntity" = link.entity
        self._material: "RigidMaterial" = link.entity.material
        self._solver: "RigidSolver" = link.entity.solver

        self._uid = gs.UID()
        self._idx = idx
        self._needs_coup: bool = needs_coup
        self._is_convex: bool = mesh.is_convex
        self._vert_start: int = vert_start
        self._face_start: int = face_start
        self._edge_start: int = edge_start
        self._verts_state_start: int = verts_state_start

        self._coup_softness: float = self._material.coup_softness
        self._coup_friction: float = self._material.coup_friction
        self._coup_restitution: float = self._material.coup_restitution

        # For heterogeneous simulation: which environments this geom is active in (None = all envs)
        self.active_envs_mask: torch.Tensor | None = None
        self.active_envs_idx: np.ndarray | None = None

        self._init_verts = mesh.verts
        self._init_faces = mesh.faces
        self._init_edges = mesh.get_unique_edges()
        self._init_normals = mesh.normals
        self._uvs = mesh.uvs
        self._surface = mesh.surface
        self._metadata = mesh.metadata

        self._init_center_pos = np.repeat(
            self._init_verts.mean(0, keepdims=True), repeats=self._init_verts.shape[0], axis=0
        )

        # The solver reads a row of fixed width, so copy the described shape data into one
        self._data = np.zeros([7])
        if desc.data is not None:
            self._data[: len(desc.data)] = desc.data

        # Adjacency graph, shared by reference with sibling geoms that back the same processed geometry.
        self.vert_neighbors, self.vert_n_neighbors, self.vert_neighbor_start = mesh.get_vert_adjacency()
        assert self.n_verts == len(self.vert_n_neighbors)

    def _build(self):
        pass

    @property
    def trimesh(self):
        """
        The trimesh object of the geom.
        """
        return self.desc.mesh.trimesh

    def get_trimesh(self):
        """
        The trimesh object of the geom. Deprecated: use the 'trimesh' property instead.
        """
        gs.logger.warning("'get_trimesh' is deprecated and will be removed in a future release. Use 'trimesh' instead.")
        return self.trimesh

    @cached_property
    def sdf_trimesh(self):
        """
        The trimesh object reconstructed from the signed distance field (SDF) of the geom.
        """
        sdf_grid = self.desc.mesh.get_sdf_grid(self.desc.sdf)
        if sdf_grid.value.min() >= 0:
            gs.logger.warning("SDF is positive everywhere. Returning empty mesh.")
            return trimesh.Trimesh(vertices=[], faces=[])
        else:
            vertices, faces, _, _ = skimage.measure.marching_cubes(sdf_grid.value, level=0)
            T_sdf_to_mesh = np.linalg.inv(sdf_grid.T_mesh_to_sdf)
            vertices = (T_sdf_to_mesh @ np.hstack([vertices, np.ones([len(vertices), 1])]).T)[:3].T
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(np.array([[1.0, 1.0, 0.6, 1.0]]), [len(vertices), 1])
            )
            return mesh

    def visualize_sdf(
        self,
        pos=None,
        T=None,
        color=(1.0, 1.0, 0.3, 1.0),
        show_axis=False,
        axis_color=(1.0, 0.0, 0.0, 1.0),
        axis_length=0.3,
        show_boundary=False,
        boundary_color=(0.0, 1.0, 0.0, 0.2),
    ):
        """
        Visualizes the signed distance field (SDF) of the rigid geometry in the viewer.
        """
        if self._solver.scene.viewer is None:
            gs.raise_exception("Viewer is not available.")

        if self._solver.n_envs > 0:
            gs.raise_exception("Batched scene does not support sdf visualization.")

        sdf_mesh = self.sdf_trimesh.copy()
        sdf_mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(np.array([color]), [len(sdf_mesh.vertices), 1])
        )
        if T is None:
            if pos is None:
                T = gu.trans_quat_to_T(
                    *map(tensor_to_array, (self.get_pos(relative=False), self.get_quat(relative=False)))
                )
            else:
                T = gu.trans_to_T(np.array(pos))
        else:
            T = np.array(T)
        self._solver.scene.draw_debug_mesh(sdf_mesh, T=T)

        if show_axis:
            axis_mesh = trimesh.creation.axis(origin_size=0.01, axis_radius=0.005, axis_length=axis_length)
            axis_mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(np.array([axis_color]), [len(axis_mesh.vertices), 1])
            )
            self._solver.scene.draw_debug_mesh(axis_mesh, T=T)

        if show_boundary:
            sdf_grid = self.desc.mesh.get_sdf_grid(self.desc.sdf)
            boundary_mesh = trimesh.creation.box(extents=[1, 1, 1])
            boundary_mesh.vertices = gu.inv_transform_by_T(
                (boundary_mesh.vertices + 0.5) * np.subtract(self.desc.sdf.n_nodes, 1), sdf_grid.T_mesh_to_sdf
            )
            boundary_mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(np.array([boundary_color]), [len(boundary_mesh.vertices), 1])
            )
            self._solver.scene.draw_debug_mesh(boundary_mesh, T=T)

    @gs.assert_built
    def set_friction(self, friction):
        """
        Set the friction coefficient of this geometry.
        """
        if friction < 0:
            gs.raise_exception("`friction` must be non-negative.")
        self._solver.set_geom_friction(friction, self._idx)

    @gs.assert_built
    def get_friction(self):
        """
        Get the friction coefficient the simulation is currently using for this geom.

        Returns
        -------
        friction : torch.Tensor, shape ()
            The friction coefficient of the geom.
        """
        return self._solver.get_geoms_friction(self._idx)[0]

    @gs.assert_built
    def get_friction_torsional(self):
        """
        Get the torsional friction coefficient the simulation is currently using for this geom (see
        'gs.materials.Rigid').

        Returns
        -------
        friction_torsional : torch.Tensor, shape ()
            The torsional friction coefficient of the geom.
        """
        return self._solver.get_geoms_friction_torsional(self._idx)[0]

    @gs.assert_built
    def get_friction_rolling(self):
        """
        Get the rolling friction coefficient the simulation is currently using for this geom (see
        'gs.materials.Rigid').

        Returns
        -------
        friction_rolling : torch.Tensor, shape ()
            The rolling friction coefficient of the geom.
        """
        return self._solver.get_geoms_friction_rolling(self._idx)[0]

    @gs.assert_built
    def set_friction_torsional(self, friction_torsional):
        """
        Set the torsional friction coefficient of this geometry (see 'gs.materials.Rigid').
        """
        if friction_torsional < 0:
            gs.raise_exception("`friction_torsional` must be non-negative.")
        self._solver.set_geom_friction_torsional(friction_torsional, self._idx)

    @gs.assert_built
    def set_friction_rolling(self, friction_rolling):
        """
        Set the rolling friction coefficient of this geometry (see 'gs.materials.Rigid').
        """
        if friction_rolling < 0:
            gs.raise_exception("`friction_rolling` must be non-negative.")
        self._solver.set_geom_friction_rolling(friction_rolling, self._idx)

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None, *, relative=True):
        """
        Get the position of the geom.

        When 'relative' is True (default), the position reported is that of the authored geom origin. The internal geom
        origin used by the solver is the authored one moved by the entity's morph 'offset_pos' / 'offset_quat'.
        """
        return self._solver.get_geoms_pos(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_quat(self, envs_idx=None, *, relative=True):
        """
        Get the quaternion of the geom.

        When 'relative' is True (default), the orientation reported is that of the authored geom origin. The internal
        geom origin used by the solver is the authored one moved by the entity's morph 'offset_pos' / 'offset_quat'.
        """
        return self._solver.get_geoms_quat(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_verts(self):
        """
        Get the vertices of the geom in world frame.
        """
        self._solver.update_verts_for_geoms(self._idx)

        verts_idx = slice(self.verts_state_start, self.verts_state_end)
        if self.is_fixed and not self._entity._batch_fixed_verts:
            tensor = qd_to_torch(self._solver.dyn_state.fixed_verts.pos, verts_idx, copy=True)
        else:
            tensor = qd_to_torch(self._solver.dyn_state.free_verts.pos, None, verts_idx, transpose=True, copy=True)
            if self._solver.n_envs == 0:
                tensor = tensor[0]
        return tensor

    @gs.assert_built
    def get_AABB(self):
        """
        Get the vertex-based axis-aligned bounding box (AABB) of the geom in world frame.
        """
        verts = self.get_verts()
        return torch.stack((verts.min(dim=-2).values, verts.max(dim=-2).values), dim=-2)

    def set_sol_params(self, sol_params):
        """
        Set the solver parameters of this geometry.
        """
        if self._solver.is_built:
            self._solver.set_sol_params(sol_params, geoms_idx=self._idx, envs_idx=None)
        else:
            self.desc.sol_params = sol_params

    @gs.assert_built
    def get_sol_params(self):
        """
        Get the solver parameters the simulation is currently using for this geom.
        """
        return self._solver.get_sol_params(geoms_idx=self._idx, envs_idx=None)[0]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Get the unique ID of the geom.
        """
        return self._uid

    @property
    def idx(self) -> int:
        """
        Get the global index of the geom in RigidSolver.
        """
        return self._idx

    @property
    def type(self) -> gs.GEOM_TYPE:
        """
        Get the type of the geom.
        """
        return self.desc.type

    @property
    def data(self):
        """
        Get the additional data of the geom.
        """
        return self._data

    @property
    def metadata(self):
        """
        Get the metadata of the geom.
        """
        return self._metadata

    @property
    def link(self) -> "RigidLink":
        """
        Get the link that the geom belongs to.
        """
        return self._link

    @property
    def entity(self) -> "RigidEntity":
        """
        Get the entity that the geom belongs to.
        """
        return self._entity

    @property
    def solver(self) -> "RigidSolver":
        """
        Get the solver that the geom belongs to.s
        """
        return self._solver

    @property
    def is_convex(self) -> bool:
        """
        Get whether the geom is convex.
        """
        return self._is_convex

    @property
    def mesh(self) -> "Mesh":
        return self.desc.mesh

    @property
    def needs_coup(self) -> bool:
        """
        Get whether the geom needs coupling with other non-rigid entities.
        """
        return self._needs_coup

    @property
    def contype(self) -> int:
        """
        Get the contact type of the geometry for collision pair filtering.

        The two geoms are deemed "compatible" (i.e. collisions between them is allowed) if the 'contype' of one geom
        and the 'conaffinity' of the other geom have a common bit set to 1, i.e.
        `(geom1.contype & geom2.conaffinity) || (geom2.contype & geom1.conaffinity) == True`. This is a powerful
        mechanism borrowed from Open Dynamics Engine.
        """
        return self.desc.contype

    @property
    def conaffinity(self) -> int:
        """
        Get the contact affinity of the geometry for collision pair filtering.

        See `contype` documentation for details.
        """
        return self.desc.conaffinity

    @property
    def coup_softness(self) -> float:
        """
        Get the softness coefficient of the geom for coupling.
        """
        return self._coup_softness

    @property
    def coup_friction(self) -> float:
        """
        Get the friction coefficient of the geom for coupling.
        """
        return self._coup_friction

    @property
    def coup_restitution(self) -> float:
        """
        Get the restitution coefficient of the geom for coupling.
        """
        return self._coup_restitution

    @property
    def init_pos(self) -> np.ndarray:
        """
        Get the initial position of the geom.
        """
        return self.desc.pos

    @property
    def init_quat(self) -> np.ndarray:
        """
        Get the initial quaternion of the geom.
        """
        return self.desc.quat

    @property
    def init_verts(self):
        """
        Get the initial vertices of the geom.
        """
        return self._init_verts

    @property
    def init_faces(self):
        """
        Get the initial faces of the geom.
        """
        return self._init_faces

    @property
    def init_edges(self):
        """
        Get the initial edges of the geom.
        """
        return self._init_edges

    @property
    def init_normals(self):
        """
        Get the initial normals of the geom.
        """
        return self._init_normals

    @property
    def init_center_pos(self):
        """
        Get the initial center position of the geom.
        """
        return self._init_center_pos

    @property
    def uvs(self):
        """
        Get the UV coordinates of the geom.
        """
        return self._uvs

    @property
    def surface(self):
        """
        Get the surface object of the geom.
        """
        return self._surface

    @property
    def n_verts(self) -> int:
        """
        Number of vertices of the geom.
        """
        return len(self._init_verts)

    @property
    def n_faces(self) -> int:
        """
        Number of faces of the geom.
        """
        return len(self._init_faces)

    @property
    def n_edges(self):
        """
        Number of edges of the geom.
        """
        return len(self._init_edges)

    @property
    def vert_start(self):
        """
        Get the starting index of the geom's vertices in the rigid solver.
        """
        return self._vert_start

    @property
    def face_start(self):
        """
        Get the starting index of the geom's faces in the rigid solver.
        """
        return self._face_start

    @property
    def edge_start(self):
        """
        Get the starting index of the geom's edges in the rigid solver.
        """
        return self._edge_start

    @property
    def verts_state_start(self):
        """
        Get the starting index of the geom's vertices in the rigid solver.
        """
        return self._verts_state_start

    @property
    def vert_end(self):
        """
        Get the ending index of the geom's vertices in the rigid solver.
        """
        return self.n_verts + self.vert_start

    @property
    def verts_state_end(self):
        """
        Get the ending index of the geom's vertices in the rigid solver.
        """
        return self.n_verts + self.verts_state_start

    @property
    def face_end(self):
        """
        Get the ending index of the geom's faces in the rigid solver.
        """
        return self.n_faces + self.face_start

    @property
    def edge_end(self):
        """
        Get the ending index of the geom's edges in the rigid solver.
        """
        return self.n_edges + self.edge_start

    @property
    def is_built(self):
        """
        Whether the rigid entity the geom belongs to is built.
        """
        return self.entity.is_built

    @property
    def is_free(self):
        raise DeprecationError("This property has been removed.")

    @property
    def is_fixed(self) -> bool:
        """
        Whether this geom is fixed in the world.
        """
        return self.link.is_fixed

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{self.__repr_name__()}: {self._uid}, idx: {self._idx} (from entity {self._entity.uid}, link {self._link.uid})"


class RigidVisGeom(RBC):
    """
    A `RigidVisGeom` is a counterpart of `RigidGeom`, but for visualization purposes. This can be accessed via `link.vis_geoms`.
    """

    def __init__(self, link, idx, vvert_start, vface_start, desc: RigidVisGeomDescription):
        self.desc: RigidVisGeomDescription = desc
        self._link = link
        self._entity = link.entity
        self._material = link.entity.material
        self._solver = link.entity.solver

        # Lazy-initialize low-res geometry because it is usually unused and may be slow to compute
        self._init_pos_tc = torch.from_numpy(desc.pos).to(device=gs.device, dtype=gs.tc_float)
        self._init_quat_tc = torch.from_numpy(desc.quat).to(device=gs.device, dtype=gs.tc_float)
        self._aabb_verts: torch.Tensor | None = None

        self._uid = gs.UID()
        self._idx = idx

        self._vvert_start = vvert_start
        self._vface_start = vface_start

        # For heterogeneous simulation: which environments this vgeom is active in (None = all envs)
        self.active_envs_mask: torch.Tensor | None = None
        self.active_envs_idx: np.ndarray | None = None

    def _build(self):
        pass

    @property
    def trimesh(self):
        """
        The trimesh object of the visual geom.
        """
        return self.desc.vmesh.trimesh

    def get_trimesh(self):
        """
        The trimesh object of the visual geom. Deprecated: use the 'trimesh' property instead.
        """
        gs.logger.warning("'get_trimesh' is deprecated and will be removed in a future release. Use 'trimesh' instead.")
        return self.trimesh

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None, *, relative=True):
        """
        Get the position of the visual geom.

        When 'relative' is True (default), the position reported is that of the authored geom origin. The internal geom
        origin used by the solver is the authored one moved by the entity's morph 'offset_pos' / 'offset_quat'.
        """
        return self._solver.get_vgeoms_pos(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_quat(self, envs_idx=None, *, relative=True):
        """
        Get the quaternion of the visual geom.

        When 'relative' is True (default), the orientation reported is that of the authored geom origin. The internal
        geom origin used by the solver is the authored one moved by the entity's morph 'offset_pos' / 'offset_quat'.
        """
        return self._solver.get_vgeoms_quat(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_vAABB(self, envs_idx=None):
        """
        Get the axis-aligned bounding box (AABB) of the geom in world frame.

        This method computes the bounding box of the geometry after aggressive decimation of its convex hull. This is
        usually sufficiently accurate (<1mm), while significantly improving runtime speed and reducing memory footprint.
        """
        if self._aabb_verts is None:
            # Aggressiveness has been tuned to give sub-millimeter accuracy on Franka robot in random configurations
            aabb_mesh = self.vmesh.copy()
            aabb_mesh.convexify()
            aabb_mesh.decimate(decimate_face_num=NUM_VERTS_VISUAL_GEOM_AABB, decimate_aggressiveness=3)
            self._aabb_verts = torch.from_numpy(aabb_mesh.verts).to(dtype=gs.tc_float, device=gs.device)

        pos, quat = gu.transform_pos_quat_by_trans_quat(
            self._init_pos_tc,
            self._init_quat_tc,
            self.link.get_pos(envs_idx, relative=False),
            self.link.get_quat(envs_idx, relative=False),
        )
        vverts_pos = pos[..., None, :] + gu.transform_by_quat(self._aabb_verts, quat[..., None, :])
        return torch.stack((vverts_pos.min(dim=-2).values, vverts_pos.max(dim=-2).values), dim=-2)

    @gs.assert_built
    def set_vverts(self, vverts, envs_idx=None):
        """Override this vgeom's visual vertex positions for rendering and sensors. See
        :meth:`KinematicEntity.set_vverts` for the full behavior; this method writes only this vgeom's slice.

        Requires the owning entity's morph to be created with 'enable_custom_vverts=True'.
        """
        if not self._entity._morph.enable_custom_vverts:
            gs.raise_exception(
                "'set_vverts' requires the entity's morph to be created with 'enable_custom_vverts=True'."
            )
        self._entity._is_vverts_overridden = True
        custom_offset = self._entity._custom_vvert_start - self._entity._vvert_start
        self._entity._solver.set_vverts(
            self.vvert_start + custom_offset,
            self.vvert_end + custom_offset,
            np.array([self.idx], dtype=gs.np_int),
            vverts,
            envs_idx,
        )

    @gs.assert_built
    def get_vverts(self, envs_idx=None):
        """Return a copy of this vgeom's visual vertex positions in world space.

        Entities created with 'morph.enable_custom_vverts=True' read from the engine-owned 'vverts_state.pos' buffer.
        Other entities compute the positions on the fly from the vgeom's current world pose applied to 'init_vverts'.
        """
        if self._entity._morph.enable_custom_vverts:
            custom_offset = self._entity._custom_vvert_start - self._entity._vvert_start
            return self._entity._solver.get_vverts(
                self.vvert_start + custom_offset, self.vvert_end + custom_offset, envs_idx
            )

        self._solver.update_vgeoms()
        vgeoms_pos = qd_to_torch(self._solver.dyn_state.vgeoms.pos, envs_idx, transpose=True, copy=None)
        vgeoms_quat = qd_to_torch(self._solver.dyn_state.vgeoms.quat, envs_idx, transpose=True, copy=None)
        init = torch.as_tensor(self.init_vverts, dtype=gs.tc_float, device=gs.device)
        pos = vgeoms_pos[..., self.idx, :].unsqueeze(-2)
        quat = vgeoms_quat[..., self.idx, :].unsqueeze(-2)
        tensor = gu.transform_by_trans_quat(init, pos, quat)
        return tensor[0] if self._solver.n_envs == 0 else tensor

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Get the unique ID of the vgeom.
        """
        return self._uid

    @property
    def idx(self):
        """
        Get the global index of the vgeom in RigidSolver.
        """
        return self._idx

    @property
    def link(self):
        """
        Get the link that the vgeom belongs to.
        """
        return self._link

    @property
    def entity(self):
        """
        Get the entity that the vgeom belongs to.
        """
        return self._entity

    @property
    def vmesh(self):
        return self.desc.vmesh

    @property
    def solver(self):
        """
        Get the solver that the vgeom belongs to.
        """
        return self._solver

    @property
    def metadata(self):
        """
        Get the metadata of the vgeom.
        """
        return self.desc.vmesh.metadata

    @property
    def init_pos(self):
        """
        Get the initial position of the vgeom.
        """
        return self.desc.pos

    @property
    def init_quat(self):
        """
        Get the initial quaternion of the vgeom.
        """
        return self.desc.quat

    @property
    def init_vverts(self):
        """
        Get the initial vertices of the vgeom.
        """
        return self.desc.vmesh.verts

    @property
    def init_vfaces(self):
        """
        Get the initial faces of the vgeom.
        """
        return self.desc.vmesh.faces

    @property
    def init_vnormals(self):
        """
        Get the initial normals of the vgeom.
        """
        return self.desc.vmesh.normals

    @property
    def uvs(self):
        """
        Get the UV coordinates of the vgeom.
        """
        return self.desc.vmesh.uvs

    @property
    def surface(self):
        """
        Get the surface object of the vgeom.
        """
        return self.desc.vmesh.surface

    @property
    def n_vverts(self):
        """
        Number of vertices of the vgeom.
        """
        return len(self.desc.vmesh.verts)

    @property
    def n_vfaces(self):
        """
        Number of faces of the vgeom.
        """
        return len(self.desc.vmesh.faces)

    @property
    def vvert_start(self):
        """
        Get the starting index of the vgeom's vertices in the rigid solver.
        """
        return self._vvert_start

    @property
    def vface_start(self):
        """
        Get the starting index of the vgeom's faces in the rigid solver.
        """
        return self._vface_start

    @property
    def vvert_end(self):
        """
        Get the ending index of the vgeom's vertices in the rigid solver.
        """
        return self.n_vverts + self.vvert_start

    @property
    def vface_end(self):
        """
        Get the ending index of the vgeom's faces in the rigid solver.
        """
        return self.n_vfaces + self.vface_start

    @property
    def is_built(self):
        """
        Whether the rigid entity the vgeom belongs to is built.
        """
        return self.entity.is_built

    @property
    def is_free(self):
        raise DeprecationError("This property has been removed.")

    @property
    def is_fixed(self) -> bool:
        """
        Whether this vgeom is fixed in the world.
        """
        return self.link.is_fixed

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{self.__repr_name__()}: {self._uid}, idx: {self._idx} (from entity {self._entity.uid}, link {self._link.uid})"
