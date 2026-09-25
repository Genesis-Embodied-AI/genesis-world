from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
from genesis.constants import link_ref_frame
from genesis.repr_base import RBC
from genesis.typing import LaxPositiveFArrayType
from genesis.utils.misc import DeprecationError, qd_to_torch

from .description import KinematicLinkDescription, RigidGeomDescription, RigidLinkDescription, RigidVisGeomDescription
from .inertial import LinkInertial
from .rigid_geom import RigidGeom, RigidVisGeom

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid import RigidSolver

    from .rigid_entity import KinematicEntity, RigidEntity
    from .rigid_joint import RigidJoint


class KinematicLink(RBC):
    """
    Kinematic class. One KinematicEntity consists of multiple KinematicLinks, each of which is a rigid body and could
    consist of multiple RigidVisGeoms (`link.vgeoms` for visualization).
    """

    def __init__(
        self,
        entity: "KinematicEntity",
        idx: int,
        parent_idx: int,
        root_idx: int | None,
        joint_start: int,
        n_joints: int,
        vgeom_start: int,
        vvert_start: int,
        vface_start: int,
        desc: KinematicLinkDescription,
    ):
        self.desc: KinematicLinkDescription = desc
        self._entity: "KinematicEntity" = entity
        self._solver: "RigidSolver" = entity.solver
        self._entity_idx_in_solver = entity._idx_in_solver

        self._uid = gs.UID()
        self._idx: int = idx
        self._parent_idx: int = parent_idx  # -1 if no parent

        # 'is_fixed' attribute specifies whether the link is free to move.
        # In practice, this attributes determines whether the geometry vertices associated with the entity are stored
        # per batch-element and updated at every simulation step, or computed once at build time and shared among the
        # entire batch. This affects correct processing of collision detection and sensor raycasting as a side-effect.
        is_fixed = True
        link = self
        while True:
            is_fixed &= all(joint.type is gs.JOINT_TYPE.FIXED for joint in link.joints)
            if link.parent_idx == -1:
                break
            link = self.entity.links[link.parent_idx - self.entity.link_start]
        self._root_idx: int = link.idx if root_idx is None else root_idx
        self._is_fixed: bool = is_fixed

        self._joint_start: int = joint_start
        self._n_joints: int = n_joints

        self._vgeom_start: int = vgeom_start
        self._vvert_start: int = vvert_start
        self._vface_start: int = vface_start

        self._vgeoms: list[RigidVisGeom] = gs.List()

        # Heterogeneous variant tracking (None = not heterogeneous)
        self._variant_vgeom_ranges: list[tuple[int, int]] | None = None

    def _init_variant_tracking(self):
        """Start tracking heterogeneous variants. Records first variant from current state."""
        self._variant_vgeom_ranges = [(self._vgeom_start, self._vgeom_start + self.n_vgeoms)]

    def _record_variant_vgeom_range(self, n_new_vgeoms):
        """Record a new variant's vgeom range."""
        prev_end = self._variant_vgeom_ranges[-1][1]
        self._variant_vgeom_ranges.append((prev_end, prev_end + n_new_vgeoms))

    def _build(self):
        for vgeom in self._vgeoms:
            vgeom._build()

    def _add_vgeom(self, desc: RigidVisGeomDescription):
        vgeom = RigidVisGeom(
            link=self,
            idx=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            desc=desc,
        )
        self._vgeoms.append(vgeom)

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None, *, relative=True):
        """
        Get the position of the link.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the position. If None, get the position of all environments. Default
            is None.
        relative : bool, optional
            Whether to report the position of the authored link origin rather than of the internal link origin used by
            the solver. The internal link origin is the authored one moved by the entity's morph 'offset_pos' /
            'offset_quat' and, on a free root link with 'align=True', by the inertial alignment. Defaults to True.
        """
        return self._solver.get_links_pos(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_quat(self, envs_idx=None, *, relative=True):
        """
        Get the quaternion of the link.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the quaternion. If None, get the quaternion of all environments.
            Default is None.
        relative : bool, optional
            Whether to report the orientation of the authored link origin rather than of the internal link origin used
            by the solver. The internal link origin is the authored one moved by the entity's morph 'offset_pos' /
            'offset_quat' and, on a free root link with 'align=True', by the inertial alignment. Defaults to True.
        """
        return self._solver.get_links_quat(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_vel(self, envs_idx=None, *, relative=True) -> torch.Tensor:
        """
        Get the linear velocity of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the linear velocity. If None, get the linear velocity of all
            environments. Default is None.
        relative : bool, optional
            Whether to report the velocity of the authored link origin rather than of the internal link origin used by
            the solver. The internal link origin is the authored one moved by the world-frame vector 'd'. That
            displacement composes the entity's morph 'offset_pos' / 'offset_quat' and, on a free root link with
            'align=True', the inertial alignment. Both readings are expressed in world coordinates and differ by the
            transport 'omega x d'. Defaults to True.
        """
        return self._solver.get_links_vel(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_ang(self, envs_idx=None) -> torch.Tensor:
        """
        Get the angular velocity of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the angular velocity. If None, get the angular velocity of all environments. Default is None.
        """
        return self._solver.get_links_ang(self._idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_vAABB(self, envs_idx=None):
        """
        Get the axis-aligned bounding box (AABB) of the link's visual body in the world frame by aggregating all
        the visual geometries associated with this link (`link.vgeoms`).
        """
        if self.n_vgeoms == 0:
            gs.raise_exception("Link has no visual geometries.")

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx
        if self.entity._enable_heterogeneous:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            n_envs = len(envs_idx)
            aabb_min = torch.full((n_envs, 3), float("inf"), dtype=gs.tc_float, device=gs.device)
            aabb_max = torch.full((n_envs, 3), float("-inf"), dtype=gs.tc_float, device=gs.device)
            for vgeom in self.vgeoms:
                vgeom_aabb = vgeom.get_vAABB(envs_idx)
                active_mask = vgeom.active_envs_mask[envs_idx] if vgeom.active_envs_mask is not None else ()
                aabb_min[active_mask] = torch.minimum(aabb_min[active_mask], vgeom_aabb[active_mask, 0])
                aabb_max[active_mask] = torch.maximum(aabb_max[active_mask], vgeom_aabb[active_mask, 1])
            return torch.stack((aabb_min, aabb_max), dim=-2)

        aabbs = torch.stack([vgeom.get_vAABB(envs_idx) for vgeom in self._vgeoms], dim=-3)
        return torch.stack((aabbs[..., 0, :].min(dim=-2).values, aabbs[..., 1, :].max(dim=-2).values), dim=-2)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        The unique ID of the link.
        """
        return self._uid

    @property
    def name(self) -> str:
        """
        The name of the link.
        """
        return self.desc.name

    @property
    def entity(self) -> "KinematicEntity":
        """
        The entity that the link belongs to.
        """
        return self._entity

    @property
    def solver(self) -> "RigidSolver":
        """
        The solver that the link belongs to.
        """
        return self._solver

    @property
    def joints(self) -> list["RigidJoint"]:
        """
        The sequence of joints that connects the link to its parent link.
        """
        return self.entity.joints_by_links[self.idx_local]

    @property
    def n_joints(self):
        """
        Number of the joints that connects the link to its parent link.
        """
        return self._n_joints

    @property
    def joint_start(self):
        """
        The start index of the link's joints in the RigidSolver.
        """
        return self._joint_start

    @property
    def joint_end(self):
        """
        The end index of the link's joints in the RigidSolver.
        """
        return self._joint_start + self.n_joints

    @property
    def n_dofs(self):
        """The number of degrees of freedom (DOFs) of the entity."""
        return sum(joint.n_dofs for joint in self.joints)

    @property
    def dof_start(self):
        """The index of the link's first degree of freedom (DOF) in the scene."""
        if not self.joints:
            return -1
        return self.joints[0].dof_start

    @property
    def dof_end(self):
        """The index of the link's last degree of freedom (DOF) in the scene *plus one*."""
        if not self.joints:
            return -1
        return self.joints[-1].dof_end

    @property
    def n_qs(self):
        """Returns the number of `q` variables of the link."""
        return sum(joint.n_qs for joint in self.joints)

    @property
    def q_start(self):
        """Returns the starting index of the `q` variables of the link in the rigid solver."""
        if not self.joints:
            return -1
        return self.joints[0].q_start

    @property
    def q_end(self):
        """Returns the last index of the `q` variables of the link in the rigid solver *plus one*."""
        if not self.joints:
            return -1
        return self.joints[-1].q_end

    @property
    def idx(self):
        """
        The global index of the link in the RigidSolver.
        """
        return self._idx

    @property
    def parent_idx(self):
        """
        The global index of the link's parent link in the RigidSolver. If the link is the root link, return -1.
        """
        return self._parent_idx

    @property
    def root_idx(self):
        """
        The global index of the link's root link in the RigidSolver.
        """
        return self._root_idx

    @property
    def idx_local(self):
        """
        The local index of the link in the entity.
        """
        return self._idx - self._entity.link_start

    @property
    def is_fixed(self):
        """
        Whether the link is fixed wrt the world.
        """
        return self._is_fixed

    @property
    def aligned(self) -> bool:
        """
        Whether the build anchored the link frame on the center of mass and principal axes of its body (the 'align'
        option, on a free body that opts in or is a primitive). Only a single rigid body gets the anchor: a free root
        with an inertia and no DOF-bearing descendant. Its joint-space mass block is then exactly diagonal.
        """
        return self.desc.is_aligned

    @property
    def vgeoms(self) -> list[RigidVisGeom]:
        """
        The list of the link's visualization geometries (`RigidVisGeom`).
        """
        return self._vgeoms

    @property
    def geom_start(self) -> int:
        """Start index of collision geoms. Always 0 for KinematicLink."""
        return 0

    @property
    def geom_end(self) -> int:
        """End index of collision geoms. Always 0 for KinematicLink."""
        return 0

    @property
    def n_vgeoms(self) -> int:
        """
        Number of the link's visualization geometries (`vgeom`).
        """
        return len(self._vgeoms)

    @property
    def vgeom_start(self) -> int:
        """
        The start index of the link's vgeom in the RigidSolver.
        """
        return self._vgeom_start

    @property
    def vgeom_end(self) -> int:
        """
        The end index of the link's vgeom in the RigidSolver.
        """
        return self._vgeom_start + self.n_vgeoms

    @property
    def n_verts(self) -> int:
        """Number of collision vertices. Always 0 for KinematicLink."""
        return 0

    @property
    def n_vverts(self) -> int:
        """
        Number of vertices of all the link's vgeoms.
        """
        return sum([vgeom.n_vverts for vgeom in self._vgeoms])

    @property
    def n_vfaces(self) -> int:
        """
        Number of faces of all the link's vgeoms.
        """
        return sum([vgeom.n_vfaces for vgeom in self._vgeoms])

    @property
    def is_built(self) -> bool:
        """
        Whether the entity the link belongs to is built.
        """
        return self.entity.is_built

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{(self.__repr_name__())}: {self._uid}, name: '{self.desc.name}', idx: {self._idx}"


class RigidLink(KinematicLink):
    """
    RigidLink class. One RigidEntity consists of multiple RigidLinks, each of which is a rigid body and could consist of
    multiple RigidGeoms (`link.geoms`, for collision) and RigidVisGeoms (`link.vgeoms` for visualization).
    """

    def __init__(
        self,
        entity: "RigidEntity",
        idx: int,
        parent_idx: int,
        root_idx: int | None,
        joint_start: int,
        n_joints: int,
        geom_start: int,
        cell_start: int,
        vert_start: int,
        face_start: int,
        edge_start: int,
        free_verts_state_start: int,
        fixed_verts_state_start: int,
        vgeom_start: int,
        vvert_start: int,
        vface_start: int,
        visualize_contact: bool,
        desc: RigidLinkDescription,
    ):
        super().__init__(
            entity, idx, parent_idx, root_idx, joint_start, n_joints, vgeom_start, vvert_start, vface_start, desc
        )

        if self._is_fixed and not entity._batch_fixed_verts:
            verts_state_start = fixed_verts_state_start
        else:
            verts_state_start = free_verts_state_start

        self._geom_start: int = geom_start
        self._cell_start: int = cell_start
        self._vert_start: int = vert_start
        self._face_start: int = face_start
        self._edge_start: int = edge_start
        self._verts_state_start: int = verts_state_start

        self._visualize_contact = visualize_contact

        self._geoms: list[RigidGeom] = gs.List()

        # Heterogeneous collision-geom variant tracking (None = not heterogeneous)
        self._variant_geom_ranges: list[tuple[int, int]] | None = None

    def _init_variant_tracking(self):
        """Start tracking heterogeneous variants. Records first variant from current state."""
        super()._init_variant_tracking()
        self._variant_geom_ranges = [(self._geom_start, self._geom_start + self.n_geoms)]

    def _record_variant_geom_range(self, n_new_geoms):
        """Record a new variant's geom range."""
        prev_geom_end = self._variant_geom_ranges[-1][1]
        self._variant_geom_ranges.append((prev_geom_end, prev_geom_end + n_new_geoms))

    def _build(self):
        super()._build()

        for geom in self._geoms:
            geom._build()

        # Compute per-variant inertial for heterogeneous links
        if self._variant_geom_ranges is not None:
            self._variant_inertial = [
                LinkInertial(self.desc.mass, self.desc.inertial_pos, self.desc.inertial_quat, self.desc.inertia)
            ]
            for variant in self.entity._desc.variants[1:]:
                v_l_desc = variant.links[self.idx - self.entity._link_start]
                self._variant_inertial.append(
                    LinkInertial(v_l_desc.mass, v_l_desc.inertial_pos, v_l_desc.inertial_quat, v_l_desc.inertia)
                )

    def _add_geom(self, desc: RigidGeomDescription, needs_coup=False):
        geom = RigidGeom(
            link=self,
            idx=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            verts_state_start=self.n_verts + self._verts_state_start,
            needs_coup=needs_coup,
            desc=desc,
        )
        self._geoms.append(geom)

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_verts(self):
        """
        Get the vertices of the link's collision body (concatenation of all `link.geoms`) in the world frame.
        """
        if self.entity._enable_heterogeneous:
            gs.raise_exception("This method is not supported for heterogeneous entity.")

        geoms_idx = slice(self.geom_start, self.geom_end)
        self._solver.update_verts_for_geoms(geoms_idx)

        verts_idx = slice(self._verts_state_start, self._verts_state_start + self.n_verts)
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
        Get the vertex-based axis-aligned bounding box (AABB) of the link's collision body in the world frame by
        aggregating all the collision geometries associated with this link (`link.geoms`).
        """
        if self.n_geoms == 0:
            gs.raise_exception("Link has no collision geometries.")

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx.
        # FIXME: Remove this branch after implementing 'get_verts'.
        if self.entity._enable_heterogeneous and self._solver.n_envs > 0:
            aabb_min = torch.full((self._solver.n_envs, 3), float("inf"), dtype=gs.tc_float, device=gs.device)
            aabb_max = torch.full((self._solver.n_envs, 3), float("-inf"), dtype=gs.tc_float, device=gs.device)
            for geom in self.geoms:
                geom_aabb = geom.get_AABB()
                active_mask = geom.active_envs_mask if geom.active_envs_mask is not None else ()
                aabb_min[active_mask] = torch.minimum(aabb_min[active_mask], geom_aabb[active_mask, 0])
                aabb_max[active_mask] = torch.maximum(aabb_max[active_mask], geom_aabb[active_mask, 1])
            return torch.stack((aabb_min, aabb_max), dim=-2)

        verts = self.get_verts()
        return torch.stack((verts.min(dim=-2).values, verts.max(dim=-2).values), dim=-2)

    @gs.assert_built
    def apply_external_wrench(
        self,
        force=None,
        torque=None,
        envs_idx=None,
        *,
        pos=None,
        ref: link_ref_frame = link_ref_frame.link_origin,
        local: bool = False,
    ):
        """
        Apply an external wrench over one simulation step on the link.

        Parameters
        ----------
        force : None | array_like, optional
            The linear force to apply. None for a pure torque. Defaults to None.
        torque : None | array_like, optional
            The torque to apply, on top of the moment induced by the linear force. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        pos : None | array_like, optional
            Where the linear force is applied, which sets the moment arm of the induced torque. With `local=True`, an
            offset from the origin of the `ref` frame, so the point follows the link as it moves; otherwise a world
            position. None applies the force at the origin of the `ref` frame. Defaults to None.
        ref: gs.link_ref_frame, optional
            The reference frame: the origin of the link ('link_origin'), or its center of mass ('link_COM'). It fixes
            where the linear force acts when `pos` is None, and the axes of the input coordinates when `local=True`.
            Defaults to 'link_origin'.
        local: bool, optional
            Whether `force`, `torque` and `pos` are expressed in the coordinates of the `ref` frame rather than the
            world frame. Defaults to False.
        """
        self._solver.apply_links_external_wrench(force, torque, self._idx, envs_idx, pos=pos, ref=ref, local=local)

    def apply_external_force(
        self, force, envs_idx=None, *, pos=None, ref: link_ref_frame = link_ref_frame.link_origin, local: bool = False
    ):
        """
        Apply an external linear force over one simulation step on the link.

        Refer to `RigidLink.apply_external_wrench` for details.
        """
        self.apply_external_wrench(force, envs_idx=envs_idx, pos=pos, ref=ref, local=local)

    def apply_external_torque(
        self, torque, envs_idx=None, *, ref: link_ref_frame = link_ref_frame.link_origin, local: bool = False
    ):
        """
        Apply an external torque over one simulation step on the link.

        Refer to `RigidLink.apply_external_wrench` for details.
        """
        self.apply_external_wrench(torque=torque, envs_idx=envs_idx, ref=ref, local=local)

    @gs.assert_built
    def set_mass(self, mass: LaxPositiveFArrayType):
        """
        Set the mass of the link, in kg.

        The inertia of the link is left unchanged, so use `set_inertia` to change it too. `RigidEntity.set_mass`
        scales the inertia of every link of an entity instead, by the same factor as its mass, which is how a body of
        the same shape made heavier behaves.

        Parameters
        ----------
        mass : float | array_like, shape (n_envs,)
            The mass to set. Per-environment values require `RigidOptions.batch_links_info=True`.
        """
        if self.is_fixed:
            gs.logger.warning("Updating the mass of a link that is fixed wrt world has no effect, skipping.")
            return

        if np.ndim(mass) > 0 and not self._solver.is_links_info_batched:
            gs.raise_exception(
                f"Impossible to set per-env mass of link '{self.name}'. Please specify "
                "'RigidOptions.batch_links_info=True'."
            )

        self._solver.set_links_mass(mass, self._idx)

    @gs.assert_built
    def set_COM(self, com):
        """
        Set the center of mass (COM) of the link, as an offset in the link's local frame.

        Parameters
        ----------
        com : array_like, shape (3,) or (n_envs, 3)
            The center of mass to set. Per-environment values require `RigidOptions.batch_links_info=True`.
        """
        self._solver.set_links_COM(com, self._idx)

    @gs.assert_built
    def set_inertia(self, inertia):
        """
        Set the inertia matrix of the link, expressed in its inertial frame.

        Parameters
        ----------
        inertia : array_like, shape (3, 3) or (n_envs, 3, 3)
            The inertia matrix to set, which must be symmetric positive definite. Per-environment values require
            `RigidOptions.batch_links_info=True`.
        """
        self._solver.set_links_inertia(inertia, self._idx)

    @gs.assert_built
    def get_invweight(self, envs_idx=None):
        """
        Get the constraint inverse weights of the link: one for forces, one for torques.

        See `RigidEntity.get_links_invweight` for what an inverse weight is and where its value comes from.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        invweight : torch.Tensor, shape (2,) or (n_envs, 2)
            The translational and rotational inverse weight of the link, per environment for a batched scene whose link
            info is batched.
        """
        return self._solver.get_links_invweight(self._idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_mass(self, envs_idx=None):
        """
        Get the mass of the link, in kg.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        mass : torch.Tensor, shape (n_envs,) or scalar
            The mass of the link, per environment for a batched scene whose link info is batched.
        """
        return self._solver.get_links_mass(self._idx, envs_idx)[..., 0]

    def set_friction(self, friction):
        """
        Set the friction of all the link's geoms.
        """
        for geom in self._geoms:
            geom.set_friction(friction)

    def set_friction_torsional(self, friction_torsional):
        """
        Set the torsional friction of all the link's geoms (see 'gs.materials.Rigid').
        """
        for geom in self._geoms:
            geom.set_friction_torsional(friction_torsional)

    def set_friction_rolling(self, friction_rolling):
        """
        Set the rolling friction of all the link's geoms (see 'gs.materials.Rigid').
        """
        for geom in self._geoms:
            geom.set_friction_rolling(friction_rolling)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def visualize_contact(self) -> bool:
        """
        Whether to visualize the contact of the link.
        """
        return self._visualize_contact

    @property
    def geoms(self) -> list[RigidGeom]:
        """
        The list of the link's collision geometries (`RigidGeom`).
        """
        return self._geoms

    @property
    def n_geoms(self) -> int:
        """
        Number of the link's collision geometries.
        """
        return len(self._geoms)

    @property
    def geom_start(self) -> int:
        """
        The start index of the link's collision geometries in the RigidSolver.
        """
        return self._geom_start

    @property
    def geom_end(self) -> int:
        """
        The end index of the link's collision geometries in the RigidSolver.
        """
        return self._geom_start + self.n_geoms

    @property
    def n_cells(self):
        """
        Number of sdf cells of all the link's geoms.
        """
        return sum([geom.n_cells for geom in self._geoms])

    @property
    def n_verts(self) -> int:
        """
        Number of vertices of all the link's geoms.
        """
        return sum([geom.n_verts for geom in self._geoms])

    @property
    def n_faces(self) -> int:
        """
        Number of faces of all the link's geoms.
        """
        return sum([geom.n_faces for geom in self._geoms])

    @property
    def n_edges(self) -> int:
        """
        Number of edges of all the link's geoms.
        """
        return sum([geom.n_edges for geom in self._geoms])

    @property
    def is_free(self):
        raise DeprecationError("This property has been removed.")
