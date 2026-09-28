import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
from genesis.repr_base import RBC
from genesis.utils.misc import DeprecationError, tensor_to_array

from .description import RigidJointDescription


class RigidJoint(RBC):
    """
    Joint class for rigid body entities. Each RigidLink is connected to its parent link via a RigidJoint.
    """

    def __init__(self, entity, idx, link_idx, q_start, dof_start, desc: RigidJointDescription):
        self.desc: RigidJointDescription = desc
        self._entity = entity
        self._solver = entity.solver

        self._uid = gs.UID()
        self._idx = idx
        self._link_idx = link_idx
        self._q_start = q_start
        self._dof_start = dof_start

    def __getattr__(self, name):
        # Must be implemented to throw deprecation warnings when accessing old properties, ignoring introspection
        for name_old, name_new in (
            ("dof_idx", "dofs_idx"),
            ("dof_idx_local", "dofs_idx_local"),
            ("q_idx", "qs_idx"),
            ("q_idx_local", "qs_idx_local"),
        ):
            if name == name_old:
                gs.logger.warning(
                    f"This property is deprecated and will be removed in future release. Please use '{name_new}' instead."
                )
                getter = getattr(self, f"_{name_old}")
                return getter()
        raise AttributeError

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    def get_pos(self):
        """
        Get the position of the joint in the world frame.
        """
        raise DeprecationError(
            "This method has been removed. Please consider operating at link-level to get the cartesian position in "
            "word frame. Alternatively, 'get_anchor_pos' returns the anchor position of the joint in the world frame."
        )

    def get_quat(self):
        """
        Get the quaternion of the joint in the world frame.
        """
        raise DeprecationError(
            "This method has been removed. Please consider operating at link-level to get the cartesian orientation in "
            "word frame. Alternatively, 'get_anchor_axis' returns the anchor axis of the joint in the world frame."
        )

    @gs.assert_built
    def get_anchor_pos(self):
        """
        Get the anchor position of the joint in the world frame.

        Mathematically, the anchor point corresponds to the point that is fixed wrt parent link and is coincident with
        the joint for the neutral configuration qpos0. This means that this point moves under the effect of the
        generalized coordinates corresponding to this joint (and all its ancestors in the kinematic tree). Physically,
        the anchor point is the "output" of the joint transmission, on which the child body is welded.
        """
        return self._solver.get_joints_anchor_pos(self._idx)[..., 0, :]

    @gs.assert_built
    def get_anchor_axis(self):
        """
        Get the anchor axis of the joint in the world frame.

        See `RigidJoint.get_anchor_pos` documentation for details about the notion on anchor point.
        """
        return self._solver.get_joints_anchor_axis(self._idx)[..., 0, :]

    def set_sol_params(self, sol_params):
        """
        Set the solver parameters of this joint.
        """
        if self._solver.is_built:
            self._solver.set_sol_params(sol_params, joints_idx=self._idx, envs_idx=None)
        else:
            self.desc.sol_params = sol_params

    @gs.assert_built
    def get_sol_params(self):
        """
        Get the solver parameters the simulation is currently using for this joint.
        """
        return self._solver.get_sol_params(joints_idx=self._idx, envs_idx=None)[..., 0, :]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Returns the unique id of the joint.
        """
        return self._uid

    @property
    def name(self):
        """
        Returns the name of the joint.
        """
        return self.desc.name

    @property
    def entity(self):
        """
        Returns the entity that the joint belongs to.
        """
        return self._entity

    @property
    def solver(self):
        """
        The RigidSolver object that the joint belongs to.
        """
        return self._solver

    @property
    def link(self):
        """
        Returns the child link that of the joint.
        """
        return self._solver.links[self._link_idx]

    @property
    def idx(self):
        """
        Returns the global index of the joint in the rigid solver.
        """
        return self._idx

    @property
    def idx_local(self):
        """
        Returns the local index of the joint in the entity.
        """
        return self._idx - self._entity.joint_start

    @property
    def init_qpos(self):
        """
        Returns the initial joint position.
        """
        return self.desc.init_qpos

    @property
    def n_qs(self):
        """
        Returns the number of `q` (generalized coordinate) variables that the joint has.
        """
        return self.desc.n_qs

    @property
    def n_dofs(self):
        """
        Returns the number of dofs that the joint has.
        """
        return self.desc.n_dofs

    @property
    def type(self):
        """
        Returns the type of the joint.
        """
        return self.desc.type

    @property
    def pos(self):
        """
        Returns the initial position of the joint in the world frame.
        """
        return self.desc.pos

    @property
    def quat(self):
        """
        Returns the initial quaternion of the joint in the world frame.
        """
        return self.desc.quat

    @property
    def q_start(self):
        """
        Returns the starting index of the `q` variables of the joint in the rigid solver.
        """
        return self._q_start

    @property
    def dof_start(self):
        """
        Returns the starting index of the dofs of the joint in the rigid solver.
        """
        return self._dof_start

    @property
    def q_end(self):
        """
        Returns the ending index of the `q` variables of the joint in the rigid solver.
        """
        return self.desc.n_qs + self.q_start

    @property
    def dof_end(self):
        """
        Returns the ending index of the dofs of the joint in the rigid solver.
        """
        return self.desc.n_dofs + self.dof_start

    def _dof_idx(self):
        """
        Returns all the Degrees' of Freedom (DoF) indices of the joint in the rigid solver.

        This property either returns a list, an integer, or None depending on whether the joint has multiple DoFs, a
        single one, or none, respectively.
        """
        if self.n_dofs == 1:
            return self.dof_start
        if self.n_dofs == 0:
            return None
        return self.dofs_idx

    @property
    def dofs_idx(self):
        """
        Returns all the Degrees' of Freedom (DoF) indices of the joint in the rigid solver as a sequence.
        """
        return list(range(self.dof_start, self.dof_end))

    def _dof_idx_local(self):
        """
        Returns the local dof index of the joint in the entity.

        This property either returns a list, an integer, or None depending on whether the joint has multiple DoFs, a
        single one, or none, respectively.
        """
        if self.n_dofs == 1:
            return self.dof_start - self._entity.dof_start
        if self.n_dofs == 0:
            return None
        return self.dofs_idx_local

    @property
    def dofs_idx_local(self):
        """
        Returns the local Degrees of Freedom indices of the joint in the entity.
        """
        return list(range(self.dof_start - self._entity.dof_start, self.dof_end - self._entity.dof_start))

    def _q_idx(self):
        """
        Returns all the position indices of the joint in the rigid solver.

        This property either returns a list, an integer, or None depending on whether the joint has multiple position
        indices, a single one, or none, respectively.
        """
        if self.n_qs == 1:
            return self.q_start
        elif self.n_qs == 0:
            return None
        else:
            return self.qs_idx

    @property
    def qs_idx(self):
        """
        Returns all the position indices of the joint in the rigid solver.
        """
        return list(range(self.q_start, self.q_end))

    def _q_idx_local(self):
        """
        Returns all the local `q` indices of the joint in the entity.
        """
        if self.n_qs == 1:
            return self.q_start - self._entity.q_start
        elif self.n_qs == 0:
            return None
        else:
            return self.qs_idx_local

    @property
    def qs_idx_local(self):
        """
        Returns all the local `q` indices of the joint in the entity.
        """
        return list(range(self.q_start - self._entity.q_start, self.q_end - self._entity.q_start))

    @property
    def dofs_length(self):
        """
        Returns the characteristic length of each dof, used to put dof velocities on a common linear (m/s) scale.
        Shape is (n_dofs,), or (n_dofs, n_envs) for a heterogeneous entity whose variants differ per environment.

        A translational dof has length 1, so its velocity is already a linear speed. A rotational dof has the swept
        radius of the body as its length, so that multiplying it by an angular velocity gives the linear surface speed
        that rotation produces; this makes a single velocity tolerance meaningful across mixed dofs (e.g. for
        hibernation). By default the radius is per-axis: the largest perpendicular distance from that dof's own
        rotation axis (through the joint anchor) to the geometry. With MuJoCo compatibility it is instead MuJoCo's
        dof_length: a single bounding-sphere radius about the body COM, shared by every rotational dof. For a
        heterogeneous entity each variant's geoms are active in their own environments, so the radius is per env.
        """
        n_dofs = self.n_dofs
        enable_heterogeneous = self._solver._enable_heterogeneous
        n_envs = self._solver._B
        shape = (n_dofs, n_envs) if enable_heterogeneous else (n_dofs,)
        lengths = np.ones(shape, dtype=gs.np_float)
        if n_dofs == 0:
            return lengths

        # A kinematic/avatar link has no collision geometry to size against.
        if not isinstance(self.link, RigidLink):
            return lengths

        geoms = self.link.geoms
        if not geoms:
            return lengths

        # Reduce a per-geom radius to the body radius: one value, or one per env (maxing over the geoms active in each
        # env, which for a heterogeneous entity are only its own variant's).
        masks = [None if geom.active_envs_mask is None else tensor_to_array(geom.active_envs_mask) for geom in geoms]

        def body_radius(per_geom_radius):
            if not enable_heterogeneous:
                return max(per_geom_radius)
            radius = np.zeros(n_envs, dtype=gs.np_float)
            for value, mask in zip(per_geom_radius, masks):
                if mask is None:
                    radius = np.maximum(radius, value)
                else:
                    radius[mask] = np.maximum(radius[mask], value)
            return radius

        if self._solver._enable_mujoco_compatibility:
            # MuJoCo's dof_length: one body radius shared by every rotational dof, the largest geom bounding radius
            # plus its COM-to-geom-origin distance.
            com = self.link.desc.inertial_pos
            radii = [
                np.linalg.norm(geom.init_verts, axis=1).max() + np.linalg.norm(com - geom.init_pos) for geom in geoms
            ]
            radius = body_radius(radii)
            for i_d in range(n_dofs):
                if np.linalg.norm(self.desc.dofs_motion_ang[i_d]) >= gs.EPS:  # rotational dof
                    lengths[i_d] = radius
        else:
            # Per rotational dof, the largest perpendicular distance from its rotation axis (through the joint anchor)
            # to the geometry. Tighter than one shared sphere, and correct for an offset hinge (where the body rotates
            # about one end rather than its COM).
            anchor = self.pos
            verts = [
                gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat) - anchor for geom in geoms
            ]
            for i_d in range(n_dofs):
                axis = self.desc.dofs_motion_ang[i_d]
                axis_norm = np.linalg.norm(axis)
                if axis_norm < gs.EPS:
                    continue  # translational dof, length stays 1
                axis = axis / axis_norm
                perp = [np.linalg.norm(v - np.outer(v @ axis, axis), axis=1).max() for v in verts]
                lengths[i_d] = body_radius(perp)
        return lengths

    @property
    def is_built(self):
        """
        Returns whether the entity the joint belongs to is built.
        """
        return self.entity.is_built

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{(self.__repr_name__())}: {self._uid}, name: '{self.name}', idx: {self._idx}, type: {self.type}"
