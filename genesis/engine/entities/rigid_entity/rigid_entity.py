import inspect
from functools import wraps
from itertools import chain
from typing import TYPE_CHECKING, Hashable

import numpy as np
import torch

import genesis as gs
from genesis.constants import link_ref_frame
from genesis.engine.states.entities import RigidEntityState
from genesis.typing import UnitVec4FType, Vec3FType
from genesis.utils import geom as gu
from genesis.utils.misc import DeprecationError, broadcast_tensor, qd_to_torch, tensor_to_array

from ..base_entity import Entity
from .description import (
    KinematicAttachmentDescription,
    KinematicEntityDescription,
    RigidEntityDescription,
    RigidEqualityDescription,
    RigidLinkDescription,
)
from .inertial import RHO_MUJOCO, RHO_OBJECT, RHO_ROBOT, compose_inertial_from_g_infos, finalize_inertial
from .rigid_equality import RigidEquality
from .rigid_geom import RigidGeom
from .rigid_joint import RigidJoint
from .rigid_link import KinematicLink, RigidLink

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
    from genesis.engine.solvers.kinematic_solver import KinematicSolver


# Wrapper to track the arguments of a function and save them in the target buffer
def tracked(fun):
    sig = inspect.signature(fun)

    @wraps(fun)
    def wrapper(self, *args, **kwargs):
        if self._update_tgt_while_set:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            args_dict = dict(tuple(bound.arguments.items())[1:])
            # Key the slot by (method, dofs subset, envs subset) so same-step calls on distinct subsets (e.g. arm
            # and gripper force control, or per-environment-group commands) each keep their own entry and gradient
            # path; a coarser key would let the second call evict the first from the tape. Slices key directly when
            # hashable (Python 3.12 onward) and resolve against their dimension size otherwise.
            key = [fun.__name__]
            for indices, n in (
                (args_dict.get("dofs_idx_local"), self.n_dofs),
                (args_dict.get("envs_idx"), self._solver._B),
            ):
                if indices is None:
                    subset = None
                elif isinstance(indices, slice):
                    subset = indices if isinstance(indices, Hashable) else tuple(range(*indices.indices(n)))
                elif isinstance(indices, torch.Tensor):
                    subset = tuple(tensor_to_array(indices).reshape(-1).tolist())
                else:
                    subset = tuple(np.asarray(indices).reshape(-1).tolist())
                key.append(subset)
            self._update_tgt(tuple(key), args_dict)
        return fun(self, *args, **kwargs)

    return wrapper


class KinematicEntity(Entity):
    """
    Base entity class for articulated rigid-body systems (morphology, FK, Jacobian, IK).

    Used directly by KinematicSolver for visualization-only kinematic entities,
    and subclassed by RigidEntity as a type marker for physics-enabled entities.
    """

    # override typing
    _solver: "KinematicSolver"

    _description_cls = KinematicEntityDescription

    def __init__(self, scene: "Scene", solver: "KinematicSolver", idx: int, desc: KinematicEntityDescription):
        # Set heterogeneous support before super().__init__() because _get_morph_identifier() needs it
        self._desc = desc
        self._morph_heterogeneous = desc.morphs[1:]
        self._enable_heterogeneous = bool(self._morph_heterogeneous)

        super().__init__(idx, scene, desc.morphs[0], solver, desc.material, desc.surface, name=desc.name)
        # The scene names an entity as it is added, so the description takes the name the entity received
        desc.name = self._name

        # Where this entity's links, joints and geoms begin in the solver: right after those of the entities it holds
        self._idx_in_solver = solver.n_entities
        self._link_start: int = solver.n_links
        self._joint_start: int = solver.n_joints
        self._q_start = solver.n_qs
        self._dof_start = solver.n_dofs
        self._vgeom_start = solver.n_vgeoms
        self._vvert_start = solver.n_vverts
        self._vface_start = solver.n_vfaces
        self._custom_vvert_start = solver.n_custom_vverts
        self._custom_vface_start = solver.n_custom_vfaces

        self._is_built: bool = False
        self._is_attached: bool = False
        self._is_vverts_overridden: bool = False

        self._load_model()

        # Initialize target variables and checkpoint
        self._tgt_keys = (
            "set_pos",
            "set_quat",
            "set_dofs_velocity",
            "control_dofs_force",
            "control_dofs_velocity",
            "control_dofs_position",
            "control_dofs_position_velocity",
        )
        self._tgt = dict()
        self._tgt_buffer = list()
        self._ckpt = dict()
        self._update_tgt_while_set = self._solver._requires_grad

    def _update_tgt(self, key, value):
        # Set [self._tgt] value while keeping the insertion order between keys. When a new key is inserted or an existing
        # key is updated, the new element should be inserted at the end of the dict. This is because we need to keep
        # the insertion order to correctly pass the gradients in the backward pass.
        self._tgt.pop(key, None)
        self._tgt[key] = value

    def init_ckpt(self):
        pass

    def _add_heterogeneous_variant(self, link, v_link):
        """Give a link the geoms one variant describes for it. A kinematic entity holds visual geoms alone.

        RigidEntity overrides to additionally add collision geoms.
        """
        for g_desc in v_link.vgeoms:
            link._add_vgeom(g_desc)
        link._record_variant_vgeom_range(len(v_link.vgeoms))

    def _reassign_heterogeneous_indices(self):
        """Reassign vgeom indices for multi-link heterogeneous entities.

        RigidEntity overrides to additionally handle collision geom indices.
        """
        running_vgeom_idx = self._vgeom_start
        running_vvert = self._vvert_start
        running_vface = self._vface_start

        for link in self._links:
            for vgeom in link.vgeoms:
                vgeom._idx = running_vgeom_idx
                vgeom._vvert_start = running_vvert
                vgeom._vface_start = running_vface
                running_vgeom_idx += 1
                running_vvert += vgeom.n_vverts
                running_vface += vgeom.n_vfaces

        for link in self._links:
            if link._variant_vgeom_ranges is None:
                continue
            vgeom_counts = [end - start for start, end in link._variant_vgeom_ranges]
            vgeom_cursor = link.vgeoms[0].idx if link.vgeoms else 0
            link._variant_vgeom_ranges = []
            for count in vgeom_counts:
                link._variant_vgeom_ranges.append((vgeom_cursor, vgeom_cursor + count))
                vgeom_cursor += count

    def _load_model(self):
        """Create the links, joints and geoms the description holds (see 'KinematicEntityDescription')."""
        self._links = gs.List()
        self._joints = gs.List()
        for l_desc in self._desc.links:
            self._add_link(l_desc)

        if self._desc.variants:
            # Init variant tracking on ALL links
            for link in self._links:
                link._init_variant_tracking()
            for v_desc in self._desc.variants:
                for link, v_link in zip(self._links, v_desc.links):
                    self._add_heterogeneous_variant(link, v_link)

            # For multi-link entities, reassign indices and recompute variant ranges
            if len(self._links) > 1:
                self._reassign_heterogeneous_indices()

    def _build(self):
        for link in self._links:
            link._build()

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._vgeoms = self.vgeoms
        self._is_built = True

    def _create_joints(self, j_descs, link_idx, joint_start):
        """Create the RigidJoint objects of one link from its described joints.

        Both '_add_link' overrides call this, so a kinematic and a rigid link build their joints the same way.
        """
        joints = gs.List()
        self._joints.append(joints)
        for i_j_, j_desc in enumerate(j_descs):
            if j_desc.dofs_motion_ang is None or j_desc.dofs_motion_vel is None:
                gs.raise_exception(
                    f"Joint '{j_desc.name}' holds {j_desc.n_dofs} degrees of freedom, whose motion axes the asset "
                    "must describe."
                )

            joint = RigidJoint(
                entity=self,
                idx=joint_start + i_j_,
                link_idx=link_idx,
                q_start=self.n_qs + self._q_start,
                dof_start=self.n_dofs + self._dof_start,
                desc=j_desc,
            )
            joints.append(joint)

        return joints

    def _add_link(self, l_desc):
        """Create one link from a description the resolution completed, with the joints and geoms it carries."""
        parent_idx = l_desc.parent_idx
        if parent_idx >= 0:
            parent_idx += self._link_start
        root_idx = l_desc.root_idx
        if root_idx is not None and root_idx >= 0:
            root_idx += self._link_start
        link_idx = self.n_links + self._link_start
        joint_start = self.n_joints + self._joint_start

        joints = self._create_joints(l_desc.joints, link_idx, joint_start)

        # Add child link
        link = KinematicLink(
            entity=self,
            idx=link_idx,
            parent_idx=parent_idx,
            root_idx=root_idx,
            joint_start=joint_start,
            n_joints=len(l_desc.joints),
            vgeom_start=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            desc=l_desc,
        )
        self._links.append(link)

        # Add visual geometries
        for g_desc in l_desc.vgeoms:
            link._add_vgeom(g_desc)

        return link, joints

    @gs.assert_unbuilt
    def attach(
        self,
        parent_entity,
        parent_link_name: str | None = None,
        pos: Vec3FType | None = None,
        quat: UnitVec4FType | None = None,
    ):
        """
        Merge two entities to act as single one, by attaching the base link of this entity as a child of a given link of
        another entity.

        The merged pair is simulated as one kinematic tree, whose degrees of freedom must form one contiguous range.
        This method enforces it: instantiate attached entities consecutively, attach onto the last kinematic tree of a
        multi-tree parent, and attach all children of an entity onto the same tree.

        Parameters
        ----------
        parent_entity : genesis.Entity
            The entity in the scene that will be a parent of kinematic tree.
        parent_link_name : str
            The name of the link in the parent entity to be linked. Default to the latest link the parent kinematic
            tree.
        pos : array_like, shape (3,), optional
            Mounting position of this entity's base link relative to the parent link frame. If neither `pos` nor
            `quat` is provided, the base link keeps its current local pose, i.e. the morph `pos` / `quat` this entity
            was created with acts as the mounting transform. If only `quat` is provided, defaults to (0, 0, 0).
        quat : array_like, shape (4,), optional
            Mounting orientation (w, x, y, z) of this entity's base link relative to the parent link frame. If only
            `pos` is provided, defaults to the identity quaternion. Providing `pos` and/or `quat` overrides the pose
            inherited from the morph.
        """
        if self._is_attached:
            gs.raise_exception("Entity already attached.")
        if self._solver._requires_grad:
            gs.raise_exception("Attach is not supported yet when requires_grad is True.")

        is_mounting = pos is not None or quat is not None
        if is_mounting:
            mount_pos = gu.zero_pos() if pos is None else np.asarray(pos, dtype=gs.np_float)
            mount_quat = gu.identity_quat() if quat is None else np.asarray(quat, dtype=gs.np_float)
            if mount_pos.shape != (3,):
                gs.raise_exception(f"Mounting 'pos' must have shape (3,), got {mount_pos.shape}.")
            if mount_quat.shape != (4,):
                gs.raise_exception(f"Mounting 'quat' must have shape (4,) (w, x, y, z), got {mount_quat.shape}.")
            if np.linalg.norm(mount_quat) == 0.0:
                gs.raise_exception("Mounting 'quat' cannot be a zero-length quaternion.")
            mount_quat = gu.normalize(mount_quat)

        if not isinstance(parent_entity, KinematicEntity):
            gs.raise_exception("Parent entity must derive from 'KinematicEntity'.")

        if parent_entity is self:
            gs.raise_exception("Cannot attach entity to itself.")

        # The attach merges the pair into one kinematic tree. A tree is numbered within the solver that simulates
        # it, so one solver must simulate both entities.
        if parent_entity.solver is not self._solver:
            gs.raise_exception(
                f"Parent entity is simulated by '{type(parent_entity.solver).__name__}' while this entity is "
                f"simulated by '{type(self._solver).__name__}'. Attaching across solvers is not supported."
            )

        if parent_entity.idx > self.idx:
            gs.raise_exception("Parent entity must be instantiated before child entity.")

        # Attaching reshapes the kinematic tree, but the post-build pass that anchors each heterogeneous variant's
        # inertia and frame runs per free root and cannot follow a reparented (or absorbed) base link.
        if self._enable_heterogeneous or parent_entity._enable_heterogeneous:
            gs.raise_exception("Attaching heterogeneous entities is not supported.")

        # Check if base link was fixed but no longer is
        base_link = self.links[0]
        parent_link = parent_entity.get_link(parent_link_name)
        if base_link.is_fixed and not parent_link.is_fixed:
            if not self._batch_fixed_verts:
                gs.raise_exception(
                    "Attaching fixed-based entity to parent link requires setting Morph option 'batch_fixed_verts=True'."
                )

        # The merged kinematic tree must keep contiguous DOFs (each mass block is processed as one interval), so every
        # DOF-carrying link numbered between the target tree's root and this entity must already belong to that tree.
        # Each violation cause gets its own actionable error.
        root_link = self._solver.links[parent_link.root_idx]
        for link in self._solver.links[root_link.idx + 1 : self.link_start]:
            if link.n_dofs == 0 or link.root_idx == root_link.idx:
                continue
            foreign_root_link = self._solver.links[link.root_idx]
            if foreign_root_link.entity is root_link.entity:
                is_foreign_tree_merged = any(
                    other_link.root_idx == link.root_idx and other_link.entity is not foreign_root_link.entity
                    for other_link in self._solver.links[link.root_idx :]
                )
                if is_foreign_tree_merged:
                    gs.raise_exception(
                        "Attaching entities onto different kinematic trees of the same parent entity is not "
                        "supported. Load the parent's trees as separate entities."
                    )
                gs.raise_exception(
                    "Attaching an entity onto a kinematic tree that is not the last one declared in its file is not "
                    "supported. Declare the attached-onto tree last, or load the file's other trees as separate "
                    "entities."
                )
            gs.raise_exception(
                "Creating entities between attached entities is not supported. Instantiate attached entities "
                "consecutively."
            )

        # Remove all root joints if necessary.
        # The requires shifting joint and dof indices of all subsequent entities.
        # Note that we do not remove world link if any, but rather remove all base joints. This is to avoid altering
        # the parent entity by moving all fixed geometries to the new parent link.
        if not base_link.is_fixed:
            n_base_joints = base_link.n_joints
            n_base_dofs = base_link.n_dofs
            n_base_qs = base_link.n_qs

            base_link._n_joints = 0
            self._joints[0].clear()
            # Indexed within the solver, since a scene index counts the entities of every solver.
            for entity in self._solver.entities[(self._idx_in_solver + 1) :]:
                entity._joint_start -= n_base_joints
                entity._dof_start -= n_base_dofs
                entity._q_start -= n_base_qs
            for joint in self._solver.joints[self.joint_start :]:
                joint._idx -= n_base_joints
                joint._dof_start -= n_base_dofs
                joint._q_start -= n_base_qs
            for link in self._solver.links[(self.link_start + 1) :]:
                link._joint_start -= n_base_joints

            # Joint-equality constraints (e.g. mimic joints) reference joints by global index, which must stay
            # aligned with the dense joint ordering. Shift those references in lockstep with the joint re-indexing
            # above. Link-based equalities (connect/weld) are unaffected since no link is removed here.
            removed_joints_end = self.joint_start + n_base_joints
            for equality in self._solver.equalities:
                if equality.type == gs.EQUALITY_TYPE.JOINT:
                    if equality._eq_obj1id >= removed_joints_end:
                        equality._eq_obj1id -= n_base_joints
                    if equality._eq_obj2id >= removed_joints_end:
                        equality._eq_obj2id -= n_base_joints

        # Overwrite parent link
        base_link._parent_idx = parent_link.idx

        # Re-root the whole tree hanging from this entity's base link - scene-wide, because entities previously
        # attached into this one already share its root and must follow it (chained attaches may run in any order);
        # links of other trees declared in the same file keep their own root, as do the fixed flag and invweight.
        for link in self._solver.links:
            if link.root_idx == base_link.idx:
                was_fixed = link.is_fixed
                link._root_idx = parent_link.root_idx
                link._is_fixed &= parent_link.is_fixed

                # The attach moves this link into another kinematic tree, so its old tree's inverse weight no longer
                # applies. The sentinel makes the solver recompute it during its refresh.
                link.desc.invweight = np.full((2,), fill_value=-1.0, dtype=gs.np_float)

                # Inertial resolution skips the geometry estimate for a world-carried link (see '_describe_link'), so a
                # link that starts moving recomputes it from its geoms at the density its entity simulates. Values the
                # asset states still win over the estimate.
                if was_fixed and not link.is_fixed and isinstance(link.desc, RigidLinkDescription):
                    desc = link.desc
                    # An entity attached beneath this one shares its root, so the density comes from the link's own
                    # entity rather than from this one.
                    rho = link.entity.material.rho
                    if rho is None:
                        if self._solver._enable_mujoco_compatibility:
                            rho = RHO_MUJOCO
                        else:
                            rho = RHO_ROBOT if desc.is_robot else RHO_OBJECT
                    # A geom description holds the fields a parse states it with, so one composition serves either.
                    hint = compose_inertial_from_g_infos([vars(g_desc) for g_desc in (desc.geoms or desc.vgeoms)], rho)
                    # A fixed link holds what the asset states and None elsewhere (see 'RigidLinkDescription'), so
                    # this resolves it exactly as '_describe_link' resolves a moving link.
                    desc.mass, desc.inertial_pos, desc.inertial_quat, desc.inertia = finalize_inertial(
                        desc.mass, desc.inertial_pos, desc.inertial_quat, desc.inertia, *hint
                    )

        # The anchor of an aligned free root is the center of mass and principal axes of the body the build described,
        # and the attached links now extend that body. Its frame and qpos stay where they are, so the anchor stops
        # standing for the body: the mass block keeps its off-diagonal terms and the midpoint pass turns the body about
        # its actual center of mass (see '_init_tree_fields' and 'func_midpoint_is_aligned').
        root_link.desc.is_aligned = False

        # Apply the explicit mounting transform. Forward kinematics interprets the base link's local pose relative to
        # its (new) parent link, so overwriting it here mounts the entity at (pos, quat) in the parent link frame.
        # Compose with the morph frame offset exactly as '_align_link' does for the morph pose at load time, so the
        # relative pose getters keep stripping the offset correctly.
        if is_mounting:
            base_link.desc.pos, base_link.desc.quat = gu.transform_pos_quat_by_trans_quat(
                base_link.desc.offset_pos, base_link.desc.offset_quat, mount_pos, mount_quat
            )

        self._desc.attachment = KinematicAttachmentDescription(
            entity_name=parent_entity.name, link_name=parent_link.name
        )
        self._is_attached = True

    # ------------------------------------------------------------------------------------
    # ---------------------------------- control & io ------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim._steps_local
            self._tgt = self._tgt_buffer[index].copy()
        else:
            self._tgt_buffer.append(self._tgt.copy())

        update_tgt_while_set = self._update_tgt_while_set
        # Apply targets in the order of insertion
        for key in self._tgt.keys():
            data_kwargs = self._tgt[key]

            # We do not need zero velocity here because if it was true, [set_dofs_velocity] from zero_velocity would
            # be in [tgt]
            if "zero_velocity" in data_kwargs:
                data_kwargs["zero_velocity"] = False
            # Do not update [tgt], as input information is finalized at this point
            self._update_tgt_while_set = False

            # Every tracked setter replays uniformly from its taped kwargs, so dispatch by name (a rare legitimate
            # use of getattr, on our own vetted key set).
            if key[0] not in self._tgt_keys:
                gs.raise_exception(f"Invalid target key: {key[0]} not in {self._tgt_keys}")
            getattr(self, key[0])(**data_kwargs)

        self._tgt = dict()
        self._update_tgt_while_set = update_tgt_while_set

    def process_input_grad(self):
        index = self._sim.cur_step_local - self._sim._steps_local
        for key in reversed(self._tgt_buffer[index].keys()):
            data_kwargs = self._tgt_buffer[index][key]

            match key[0]:
                # We need to unpack the data_kwargs because [_backward_from_qd] only supports positional arguments.
                # Inputs are stored on the tape as passed by the user, so scalars and array-likes (valid setter
                # inputs that cannot carry a gradient) are filtered out with the tensor check.
                case "set_pos":
                    pos = data_kwargs.pop("pos")
                    if isinstance(pos, torch.Tensor) and pos.requires_grad:
                        pos._backward_from_qd(self.set_pos_grad, data_kwargs["envs_idx"], data_kwargs["relative"])

                case "set_quat":
                    quat = data_kwargs.pop("quat")
                    if isinstance(quat, torch.Tensor) and quat.requires_grad:
                        quat._backward_from_qd(self.set_quat_grad, data_kwargs["envs_idx"], data_kwargs["relative"])

                case "set_dofs_velocity":
                    velocity = data_kwargs.pop("velocity")
                    # [velocity] could be None when we want to zero the velocity (see set_dofs_velocity of RigidSolver)
                    if isinstance(velocity, torch.Tensor) and velocity.requires_grad:
                        velocity._backward_from_qd(
                            self.set_dofs_velocity_grad, data_kwargs["dofs_idx_local"], data_kwargs["envs_idx"]
                        )

                case "control_dofs_force":
                    force = data_kwargs.pop("force")
                    if isinstance(force, torch.Tensor) and force.requires_grad:
                        force._backward_from_qd(
                            self.set_dofs_force_grad, data_kwargs["dofs_idx_local"], data_kwargs["envs_idx"]
                        )

                case "control_dofs_velocity" | "control_dofs_position" | "control_dofs_position_velocity":
                    # PD control targets are replayed for primal correctness but have no input-gradient path.
                    for target in (data_kwargs.get("position"), data_kwargs.get("velocity")):
                        if isinstance(target, torch.Tensor) and target.requires_grad:
                            gs.raise_exception(
                                "Gradients with respect to PD control targets are not supported yet. Use "
                                "'control_dofs_force' for differentiable control inputs."
                            )

                case _:
                    gs.raise_exception(f"Invalid target key: {key[0]} not in {self._tgt_keys}")

    def save_ckpt(self, ckpt_name):
        if ckpt_name not in self._ckpt:
            self._ckpt[ckpt_name] = {}
        self._ckpt[ckpt_name]["_tgt_buffer"] = self._tgt_buffer.copy()
        self._tgt_buffer.clear()

    def load_ckpt(self, ckpt_name):
        self._tgt_buffer = self._ckpt[ckpt_name]["_tgt_buffer"].copy()

    def reset_grad(self):
        self._tgt_buffer.clear()

    @gs.assert_built
    def get_state(self):
        state = RigidEntityState(self, self._sim.cur_step_global)

        solver_state = self._solver.get_state()
        pos = solver_state.links_pos[:, self.base_link_idx]
        quat = solver_state.links_quat[:, self.base_link_idx]

        state._pos = pos
        state._quat = quat

        return state

    def _get_global_idx(self, idx_local, idx_local_max, idx_global_start=0, *, unsafe=False):
        if idx_local is None:
            idx_global = range(idx_global_start, idx_local_max + idx_global_start)
        elif isinstance(idx_local, (int, np.integer)):
            if idx_local < 0:
                idx_local = idx_local_max + idx_local
            idx_global = (idx_local + idx_global_start,)
        elif isinstance(idx_local, slice):
            start, stop, step = idx_local.indices(idx_local_max)
            idx_global = range(start + idx_global_start, stop + idx_global_start, step)
            if step < 0:
                idx_global = tuple(idx_global)
        elif isinstance(idx_local, range):
            if idx_local and -idx_local_max <= idx_local[0] < 0 and -idx_local_max <= idx_local[-1] < 0:
                # Every entry shifts by the same amount, so the constant step and the range form are preserved
                idx_global = range(
                    idx_local.start + idx_local_max + idx_global_start,
                    idx_local.stop + idx_local_max + idx_global_start,
                    idx_local.step,
                )
            elif idx_local and (idx_local[0] < 0 or idx_local[-1] < 0):
                # Mixed-sign entries wrap by different amounts, which breaks the constant step
                idx_global = [
                    i + idx_global_start + (idx_local_max if -idx_local_max <= i < 0 else 0) for i in idx_local
                ]
            else:
                idx_global = range(
                    idx_local.start + idx_global_start, idx_local.stop + idx_global_start, idx_local.step
                )
            if isinstance(idx_global, range) and idx_global.step < 0:
                idx_global = tuple(idx_global)
        elif isinstance(idx_local, (list, tuple)):
            try:
                idx_global = [
                    i + idx_global_start + (idx_local_max if -idx_local_max <= i < 0 else 0) for i in idx_local
                ]
            except TypeError:
                gs.raise_exception("Expecting a sequence of integers for `idx_local`.")
        else:
            if isinstance(idx_local, torch.Tensor):
                if idx_local.dtype == torch.bool:
                    if idx_local.shape != (idx_local_max,):
                        gs.raise_exception(f"Boolean masks for `idx_local` must have shape ({idx_local_max},).")
                    idx_local, *_ = torch.where(idx_local)
                    is_negative_wrap_required = False
                else:
                    is_negative_wrap_required = idx_local.dtype.is_signed
                    where = torch.where
            elif isinstance(idx_local, np.ndarray):
                if np.issubdtype(idx_local.dtype, np.bool_):
                    if idx_local.shape != (idx_local_max,):
                        gs.raise_exception(f"Boolean masks for `idx_local` must have shape ({idx_local_max},).")
                    idx_local, *_ = np.where(idx_local)
                    is_negative_wrap_required = False
                else:
                    is_negative_wrap_required = not np.issubdtype(idx_local.dtype, np.unsignedinteger)
                    where = np.where
            else:
                gs.raise_exception("Expecting integer indices for `idx_local`.")
            if is_negative_wrap_required:
                # Wrap valid Python-style negatives before the global offset so those entries stay local to the entity
                is_valid_negative = (-idx_local_max <= idx_local) & (idx_local < 0)
                idx_local = where(is_valid_negative, idx_local + idx_local_max, idx_local)
            # Increment may be slow when dealing with heterogenuous data, so it must be avoided if possible
            if idx_global_start > 0:
                idx_global = idx_local + idx_global_start
            else:
                idx_global = idx_local

        # Early return if unsafe
        if unsafe:
            return idx_global

        # Perform a bunch of sanity checks
        idx_global = torch.as_tensor(idx_global, dtype=gs.tc_int, device=gs.device).contiguous()
        ndim = idx_global.ndim
        if ndim == 0:
            idx_global = idx_global[None]
        elif ndim > 1:
            gs.raise_exception("Expecting a 1D tensor for local index.")

        # FIXME: This check is too expensive
        # if (idx_global < 0).any() or (idx_global >= idx_global_start + idx_local_max).any():
        #     gs.raise_exception("`idx_local` exceeds valid range.")

        return idx_global

    def get_joint(self, name=None, uid=None):
        """
        Get a RigidJoint object by name or uid.

        Parameters
        ----------
        name : str, optional
            The name of the joint. Defaults to None.
        uid : str, optional
            The uid of the joint. This can be a substring of the joint's uid. Defaults to None.

        Returns
        -------
        joint : RigidJoint
            The joint object.
        """

        if name is not None:
            for joint in self.joints:
                if joint.name == name:
                    return joint
            gs.raise_exception(
                f"Joint not found for name: {name}. Available joint names: {[joint.name for joint in self.joints]}."
            )

        elif uid is not None:
            for joint in self.joints:
                if uid in str(joint.uid):
                    return joint
            gs.raise_exception(f"Joint not found for uid: {uid}.")

        else:
            gs.raise_exception("Neither `name` nor `uid` is provided.")

    def get_link(self, name=None, uid=None):
        """
        Get a RigidLink object by name or uid.

        Parameters
        ----------
        name : str, optional
            The name of the link. Defaults to None.
        uid : str, optional
            The uid of the link. This can be a substring of the link's uid. Defaults to None.

        Returns
        -------
        link : RigidLink
            The link object.
        """

        if name is not None:
            for link in self._links:
                if link.name == name:
                    return link
            gs.raise_exception(
                f"Link not found for name: {name}. Available link names: {[link.name for link in self._links]}."
            )

        elif uid is not None:
            for link in self._links:
                if uid in str(link.uid):
                    return link
            gs.raise_exception(f"Link not found for uid: {uid}.")

        else:
            gs.raise_exception("Neither `name` nor `uid` is provided.")

    @gs.assert_built
    def get_pos(self, envs_idx=None, *, relative=True):
        """
        Returns position of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            Whether to report the position of the authored link origin rather than of the internal link origin used by
            the solver. The internal link origin is the authored one moved by the morph 'offset_pos' / 'offset_quat'
            and, on a free root link with 'align=True', by the inertial alignment. Defaults to True.

        Returns
        -------
        pos : torch.Tensor, shape (3,) or (n_envs, 3)
            The position of the entity's base link.
        """
        return self._solver.get_links_pos(self.base_link_idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_quat(self, envs_idx=None, *, relative=True):
        """
        Returns quaternion of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            Whether to report the orientation of the authored link origin rather than of the internal link origin used
            by the solver. The internal link origin is the authored one moved by the morph 'offset_pos' / 'offset_quat'
            and, on a free root link with 'align=True', by the inertial alignment. Defaults to True.

        Returns
        -------
        quat : torch.Tensor, shape (4,) or (n_envs, 4)
            The quaternion of the entity's base link.
        """
        return self._solver.get_links_quat(self.base_link_idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_vel(self, envs_idx=None, *, relative=True):
        """
        Returns linear velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            Whether to report the velocity of the authored link origin rather than of the internal link origin used by
            the solver. The internal link origin is the authored one moved by the world-frame vector 'd'. That
            displacement composes the morph 'offset_pos' / 'offset_quat' and, on a free root link with 'align=True', the
            inertial alignment. Both readings are expressed in world coordinates and differ by the transport 'omega x
            d'. Defaults to True.

        Returns
        -------
        vel : torch.Tensor, shape (3,) or (n_envs, 3)
            The linear velocity of the entity's base link.
        """
        return self._solver.get_links_vel(self.base_link_idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_ang(self, envs_idx=None):
        """
        Returns angular velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (3,) or (n_envs, 3)
            The angular velocity of the entity's base link.
        """
        return self._solver.get_links_ang(self.base_link_idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_links_pos(self, links_idx_local=None, envs_idx=None, *, relative=True):
        """
        Returns the position of a given reference point for all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            Whether to report the position of the authored link origin, which is where the morph 'pos' placed it, rather
            than of the internal link origin used by the solver. The internal link origin is the authored one moved by
            the morph 'offset_pos' / 'offset_quat' and, on a free root link with 'align=True', by the inertial
            alignment. Defaults to True.

        Returns
        -------
        pos : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The position of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_pos(links_idx, envs_idx, relative=relative)

    @gs.assert_built
    def get_links_quat(self, links_idx_local=None, envs_idx=None, *, relative=True):
        """
        Returns quaternion of all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            Whether to report the orientation of the authored link origin, which is where the morph 'quat' / 'euler'
            placed it, rather than of the internal link origin used by the solver. The internal link origin is the
            authored one moved by the morph 'offset_pos' / 'offset_quat' and, on a free root link with 'align=True', by
            the inertial alignment. Defaults to True.

        Returns
        -------
        quat : torch.Tensor, shape (n_links, 4) or (n_envs, n_links, 4)
            The quaternion of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_quat(links_idx, envs_idx, relative=relative)

    @gs.assert_built
    def get_vAABB(self, envs_idx=None):
        """
        Get the axis-aligned bounding box (AABB) of the entity in world frame by aggregating all the visual
        geometries associated with this entity.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        aabb : torch.Tensor, shape (2, 3) or (n_envs, 2, 3)
            The AABB of the entity, where `[:, 0] = min_corner (x_min, y_min, z_min)` and
            `[:, 1] = max_corner (x_max, y_max, z_max)`.
        """
        if self.n_vgeoms == 0:
            gs.raise_exception("Entity has no visual geometries.")

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx
        if self._enable_heterogeneous:
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

    @gs.assert_built
    def get_links_vel(self, links_idx_local=None, envs_idx=None, *, relative=True):
        """
        Returns linear velocity of all the entity's links expressed at a given reference position in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            Whether to report the velocity of the authored link origin rather than of the internal link origin used by
            the solver. The internal link origin is the authored one moved by the world-frame vector 'd'. That
            displacement composes the morph 'offset_pos' / 'offset_quat' and, on a free root link with 'align=True', the
            inertial alignment. Both readings are expressed in world coordinates and differ by the transport 'omega x
            d'. Defaults to True.

        Returns
        -------
        vel : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear velocity of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_vel(links_idx, envs_idx, relative=relative)

    @gs.assert_built
    def get_links_ang(self, links_idx_local=None, envs_idx=None):
        """
        Returns angular velocity of all the entity's links in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The angular velocity of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_ang(links_idx, envs_idx)

    @gs.assert_built
    @tracked
    def set_pos(self, pos, envs_idx=None, *, zero_velocity=False, relative=True, skip_forward=False):
        """
        Set position of the entity's base link.

        Parameters
        ----------
        pos : array_like
            The position to set.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to False.
        relative : bool, optional
            Whether 'pos' places the authored link origin rather than directly the internal link origin used by the
            solver. The internal link origin is the authored one moved by the morph 'offset_pos' / 'offset_quat' and, on
            a free root link with 'align=True', by the inertial alignment. Defaults to True.
        skip_forward : bool, optional
            Whether to skip forward kinematics after setting position. Defaults to False.
        """
        # Throw exception in entity no longer has a "true" base link becaused it has attached
        if self._is_attached:
            gs.raise_exception("Impossible to set position of an entity that has been attached.")
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_base_links_pos(pos, self.base_link_idx, envs_idx, relative=relative, skip_forward=skip_forward)

    @gs.assert_built
    def set_pos_grad(self, envs_idx, relative, pos_grad):
        self._solver.set_base_links_pos_grad(self.base_link_idx, envs_idx, relative, pos_grad.data)

    @gs.assert_built
    @tracked
    def set_quat(self, quat, envs_idx=None, *, zero_velocity=False, relative=True, skip_forward=False):
        """
        Set quaternion of the entity's base link.

        Parameters
        ----------
        quat : array_like
            The quaternion to set.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to False.
        relative : bool, optional
            Whether 'quat' orients the authored link origin rather than directly the internal link origin used by the
            solver. The internal link origin is the authored one moved by the morph 'offset_pos' / 'offset_quat' and, on
            a free root link with 'align=True', by the inertial alignment. Defaults to True.
        skip_forward : bool, optional
            Whether to skip forward kinematics after setting quaternion. Defaults to False.
        """
        if self._is_attached:
            gs.raise_exception("Impossible to set position of an entity that has been attached.")
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_base_links_quat(
            quat, self.base_link_idx, envs_idx, relative=relative, skip_forward=skip_forward
        )

    @gs.assert_built
    def set_quat_grad(self, envs_idx, relative, quat_grad):
        self._solver.set_base_links_quat_grad(self.base_link_idx, envs_idx, relative, quat_grad.data)

    @gs.assert_built
    def set_qpos(self, qpos, qs_idx_local=None, envs_idx=None, *, zero_velocity=False, skip_forward=False):
        """
        Set the entity's qpos.

        Parameters
        ----------
        qpos : array_like
            The qpos to set.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to False.
        """
        qs_idx = self._get_global_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=True)
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_qpos(qpos, qs_idx, envs_idx, skip_forward=skip_forward)

    @gs.assert_built
    @tracked
    def set_dofs_velocity(self, velocity=None, dofs_idx_local=None, envs_idx=None, *, skip_forward=False):
        """
        Set the entity's dofs' velocity.

        Parameters
        ----------
        velocity : array_like | None
            The velocity to set. Zero if not specified.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=skip_forward)

    @gs.assert_built
    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None, *, zero_velocity=False):
        """
        Set the entity's dofs' position.

        Parameters
        ----------
        position : array_like
            The position to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to False.
        """
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_position(position, dofs_idx, envs_idx)

    @gs.assert_built
    def get_qpos(self, qs_idx_local=None, envs_idx=None):
        """
        Get the entity's qpos.

        For a free joint, the qpos holds the world-frame pose of the (solver) link origin: it is the raw generalized
        coordinate and is never expressed in the user/offset frame, unlike the relative `get_pos`/`get_quat`.

        Parameters
        ----------
        qs_idx_local : None | array_like, optional
            The indices of the qpos to get. If None, all qpos will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        qpos : torch.Tensor, shape (n_qs,) or (n_envs, n_qs)
            The entity's qpos.
        """
        qs_idx = self._get_global_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=True)
        return self._solver.get_qpos(qs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_velocity(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' velocity.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        velocity : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' velocity.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_velocity(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_position(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' position.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        position : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' position.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_position(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the positional limits (min and max) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The lower limit of the positional limits for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The upper limit of the positional limits for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_limit(dofs_idx, envs_idx)

    @gs.assert_built
    def zero_all_dofs_velocity(self, envs_idx=None, *, skip_forward=False):
        """
        Zero the velocity of all the entity's dofs.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        self.set_dofs_velocity(None, slice(0, self._n_dofs), envs_idx, skip_forward=skip_forward)

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        if self._enable_heterogeneous:
            return "heterogeneous"
        return self._morph._identifier()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_built(self):
        """
        Whether this rigid entity is built.
        """
        return self._is_built

    @property
    def is_attached(self):
        """
        Whether this rigid entity has already been attached to another one.
        """
        return self._is_attached

    @property
    def init_qpos(self):
        """The initial qpos of the entity."""
        if self.joints:
            return np.concatenate([joint.init_qpos for joint in self.joints])
        return np.array([])

    @property
    def n_qs(self):
        """The number of `q` (generalized coordinates) of the entity."""
        if self._is_built:
            return self._n_qs
        return sum(joint.n_qs for joint in self.joints)

    @property
    def n_links(self):
        """The number of `RigidLink` in the entity."""
        return len(self._links)

    @property
    def morph(self):
        """The morph of the entity.

        Raises an exception for heterogeneous entities, which have multiple morph variants: use morphs for all
        variants, or main_morph for the first one.
        """
        if self._enable_heterogeneous:
            gs.raise_exception(
                "Heterogeneous entities have multiple morph variants. Use `.morphs` for all variants, "
                "or `.main_morph` only when explicitly using the first variant."
            )
        return self._morph

    @property
    def main_morph(self):
        """The main morph of the entity (first morph for heterogeneous entities)."""
        return self._morph

    @property
    def morphs(self):
        """All morphs of the entity (main morph + heterogeneous variants if any)."""
        return gs.List((self._morph, *self._morph_heterogeneous))

    @property
    def desc(self) -> KinematicEntityDescription:
        """The description this entity was built from, holding the values its resolution decided.

        Every link and constraint was built from one of the descriptions here and keeps it, so a change made
        through either is visible here.
        """
        return self._desc

    def _repr_morph(self):
        if self._enable_heterogeneous:
            return f"{len(self.morphs)} morph variants"
        return f"{self.main_morph}"

    @property
    def n_joints(self):
        """The number of `RigidJoint` in the entity."""
        return sum(map(len, self._joints))

    @property
    def n_dofs(self):
        """The number of degrees of freedom (DOFs) of the entity."""
        if self._is_built:
            return self._n_dofs
        return sum(joint.n_dofs for joint in self.joints)

    @property
    def n_vgeoms(self):
        """The number of vgeoms (visual geoms - `RigidVisGeom`) in the entity."""
        return sum(link.n_vgeoms for link in self._links)

    @property
    def n_vverts(self):
        """The number of vverts (visual vertices, from vgeoms) in the entity."""
        return sum([link.n_vverts for link in self._links])

    @property
    def n_vfaces(self):
        """The number of vfaces (visual faces, from vgeoms) in the entity."""
        return sum([link.n_vfaces for link in self._links])

    @property
    def base_link_idx(self):
        """The index of the entity's base link in the scene."""
        return self._link_start

    @property
    def link_start(self):
        """The index of the entity's first RigidLink in the scene."""
        return self._link_start

    @property
    def link_end(self):
        """The index of the entity's last RigidLink in the scene *plus one*."""
        return self._link_start + self.n_links

    @property
    def joint_start(self):
        """The index of the entity's first RigidJoint in the scene."""
        return self._joint_start

    @property
    def joint_end(self):
        """The index of the entity's last RigidJoint in the scene *plus one*."""
        return self._joint_start + self.n_joints

    @property
    def dof_start(self):
        """The index of the entity's first degree of freedom (DOF) in the scene."""
        return self._dof_start

    @property
    def dof_end(self):
        """The index of the entity's last degree of freedom (DOF) in the scene *plus one*."""
        return self._dof_start + self.n_dofs

    @property
    def vvert_start(self):
        """The index of the entity's first `vvert` (visual vertex) in the scene."""
        return self._vvert_start

    @property
    def vface_start(self):
        """The index of the entity's first `vface` (visual face) in the scene."""
        return self._vface_start

    @property
    def q_start(self):
        """The index of the entity's first `q` (generalized coordinates) in the scene."""
        return self._q_start

    @property
    def q_end(self):
        """The index of the entity's last `q` (generalized coordinates) in the scene *plus one*."""
        return self._q_start + self.n_qs

    @property
    def vgeoms(self):
        """The list of visual geoms (`RigidVisGeom`) in the entity."""
        if self.is_built:
            return self._vgeoms
        return gs.List(vgeom for link in self._links for vgeom in link.vgeoms)

    @gs.assert_built
    def set_vverts(self, vverts, envs_idx=None):
        """Override this entity's visual vertex positions for rendering and sensors.

        vverts is broadcast to (len(envs_idx), n_vverts, 3); scalars, (3,) and (n_vverts, 3) are accepted. vverts=None
        re-runs FK over the entity's vgeoms and writes the result back into the custom buffer. Requires the entity's
        morph to be created with enable_custom_vverts=True.
        """
        if self._enable_heterogeneous:
            gs.raise_exception("This method is not supported by heterogeneous entities.")
        if not self._morph.enable_custom_vverts:
            gs.raise_exception(
                "'set_vverts' requires the entity's morph to be created with 'enable_custom_vverts=True'."
            )
        self._is_vverts_overridden = True
        self._solver.set_vverts(
            self._custom_vvert_start,
            self._custom_vvert_start + self.n_vverts,
            np.array([vg.idx for vg in self.vgeoms], dtype=gs.np_int),
            vverts,
            envs_idx,
        )

    @gs.assert_built
    def get_vverts(self, envs_idx=None):
        """Return a copy of this entity's visual vertex positions in world space.

        For entities created with enable_custom_vverts=True the positions are read from the engine custom buffer; for
        other entities they are computed on the fly from each vgeom's current pose applied to its rest-pose init_vverts.
        """
        if self._enable_heterogeneous:
            gs.raise_exception("This method is not supported by heterogeneous entities.")
        if self._morph.enable_custom_vverts:
            return self._solver.get_vverts(self._custom_vvert_start, self._custom_vvert_start + self.n_vverts, envs_idx)

        self._solver.update_vgeoms()
        vgeoms_pos = qd_to_torch(self._solver.dyn_state.vgeoms.pos, envs_idx, transpose=True, copy=None)
        vgeoms_quat = qd_to_torch(self._solver.dyn_state.vgeoms.quat, envs_idx, transpose=True, copy=None)
        parts = []
        for vgeom in self.vgeoms:
            init = torch.as_tensor(vgeom.init_vverts, dtype=gs.tc_float, device=gs.device)
            pos = vgeoms_pos[..., vgeom.idx, :].unsqueeze(-2)
            quat = vgeoms_quat[..., vgeom.idx, :].unsqueeze(-2)
            parts.append(gu.transform_by_trans_quat(init, pos, quat))
        tensor = torch.cat(parts, dim=-2)
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @property
    def links(self) -> list[RigidLink]:
        """The list of links (`RigidLink`) in the entity."""
        return self._links

    @property
    def joints(self) -> list[RigidJoint]:
        """The list of joints (`RigidJoint`) in the entity."""
        return gs.List(chain.from_iterable(self._joints))

    @property
    def joints_by_links(self):
        """The list of joints (`RigidJoint`) in the entity grouped by parent links."""
        return self._joints

    @property
    def base_link(self) -> RigidLink:
        """The base link of the entity"""
        return self._links[0]

    @property
    def base_joint(self) -> RigidJoint:
        """The base joint of the entity"""
        return self._joints[0][0]

    @property
    @gs.assert_built
    def q_limit(self):
        """The build-time positional limits of the entity's generalised coordinates, lower row then upper row.

        A joint holding its orientation as a quaternion, a free or a ball joint, contributes the unit bounds of that
        quaternion, which is normalized rather than limited.
        """
        return self._q_limit

    def _init_q_limit(self):
        q_limit_lower, q_limit_upper = [], []
        for joint in self.joints:
            if joint.type in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.SPHERICAL):
                # A quaternion takes four of the coordinates of such a joint; whatever remains is a translation,
                # carried by the first of its degrees of freedom.
                n_translation = joint.n_qs - 4
                q_limit_lower += [joint.desc.dofs_limit[:n_translation, 0], -np.ones(4)]
                q_limit_upper += [joint.desc.dofs_limit[:n_translation, 1], np.ones(4)]
            elif joint.type != gs.JOINT_TYPE.FIXED:
                q_limit_lower.append(joint.desc.dofs_limit[:, 0])
                q_limit_upper.append(joint.desc.dofs_limit[:, 1])
        if not q_limit_lower:
            # An entity that no degree of freedom moves holds no coordinate, hence a limit of width zero.
            self._q_limit = np.zeros((2, self.n_qs), dtype=gs.np_float)
            return
        self._q_limit = np.stack(
            (np.concatenate(q_limit_lower), np.concatenate(q_limit_upper)), axis=0, dtype=gs.np_float
        )

    # ------------------------------------------------------------------------------------
    # --------------------------------- Jacobian & IK ------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_jacobian(self, link, local_point=None):
        """
        Get the spatial Jacobian for a point on a target link.

        Parameters
        ----------
        link : RigidLink
            The target link.
        local_point : torch.Tensor or None, shape (3,)
            Coordinates of the point in the link's *local* frame.
            If None, the link origin is used (back-compat).

        Returns
        -------
        jacobian : torch.Tensor
            The Jacobian matrix of shape (n_envs, 6, entity.n_dofs) or (6, entity.n_dofs) if n_envs == 0.
        """
        if self.n_dofs == 0:
            gs.raise_exception("Entity has zero dofs.")

        if local_point is not None:
            local_point = torch.as_tensor(local_point, dtype=gs.tc_float, device=gs.device)
            if local_point.shape != (3,):
                gs.raise_exception("Must be a vector of length 3")

        jacobian = self._solver.get_links_jacobian(link.idx, self._dof_start, self.n_dofs, local_point)
        if self._solver.n_envs == 0:
            jacobian = jacobian[0]

        return jacobian

    @gs.assert_built
    def inverse_kinematics(
        self,
        link,
        pos=None,
        quat=None,
        local_point=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        seed=None,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for a single target link.

        The target `pos`/`quat` are interpreted in the world frame (the morph pose offset is not applied), matching the
        world-frame link poses reported by `get_links_pos` and `get_links_quat` with `relative=False`.

        Parameters
        ----------
        link : RigidLink
            The link to be used as the end-effector.
        pos : None | array_like, shape (3,) or (n_selected_envs, 3), optional
            The target position, either shared by every selected environment or given per environment. If None,
            position error will not be considered. Defaults to None.
        quat : None | array_like, shape (4,) or (n_selected_envs, 4), optional
            The target orientation, either shared by every selected environment or given per environment. If None,
            orientation error will not be considered. Defaults to None.
        local_point : None | array_like, shape (3,), optional
            A point in the link's local frame to be positioned at `pos`. If None, the link origin is used. This is
            useful for positioning a tool center point (TCP) or fingertip that is offset from the link origin. Defaults
            to None (equivalent to [0, 0, 0]).
        init_qpos : None | array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and
            y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis
            to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        seed : None | int, optional
            Seed of the joint-limit resampling that escapes unreachable local branches. Repeated calls with the same
            seed and inputs return the same solution; vary it to explore different branches. Defaults to None (a fixed
            internal seed).
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None. This is used to specify which dofs the IK is applied to.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.
        envs_idx: None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs) or (len(envs_idx), n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6) or (len(envs_idx), 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y,
            err_rot_z]. Only returned if `return_error` is True.
        """
        ret = self.inverse_kinematics_multilink(
            links=[link],
            poss=[pos] if pos is not None else [],
            quats=[quat] if quat is not None else [],
            local_points=[local_point] if local_point is not None else [],
            init_qpos=init_qpos,
            respect_joint_limit=respect_joint_limit,
            max_samples=max_samples,
            max_solver_iters=max_solver_iters,
            damping=damping,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            pos_mask=pos_mask,
            rot_mask=rot_mask,
            max_step_size=max_step_size,
            seed=seed,
            dofs_idx_local=dofs_idx_local,
            return_error=return_error,
            envs_idx=envs_idx,
        )

        if return_error:
            qpos, error_pose = ret
            return qpos, error_pose[..., 0, :]
        return ret

    @gs.assert_built
    def inverse_kinematics_multilink(
        self,
        links,
        poss=None,
        quats=None,
        local_points=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        seed=None,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for multiple target links.

        Parameters
        ----------
        links : list of RigidLink
            List of links to be used as the end-effectors.
        poss : list, optional
            List of target positions. If empty, position error will not be considered. Defaults to None.
        quats : list, optional
            List of target orientations. If empty, orientation error will not be considered. Defaults to None.
        local_points : list, optional
            List of local points (one per link) in each link's local frame to be positioned at the corresponding target
            position. If empty or None, link origins are used. Each element should be array_like of shape (3,) or None.
            This is useful for positioning tool center points (TCP) or fingertips that are offset from the link origin.
            Defaults to None.
        init_qpos : array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and
            y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis
            to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        seed : None | int, optional
            Seed of the joint-limit resampling that escapes unreachable local branches. Repeated calls with the same
            seed and inputs return the same solution; vary it to explore different branches. Defaults to None (a fixed
            internal seed).
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None. This is used to specify which dofs the IK is applied to.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs) or (len(envs_idx), n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6) or (len(envs_idx), 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y,
            err_rot_z]. Only returned if `return_error` is True.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        if self.n_dofs == 0:
            gs.raise_exception("Entity has zero dofs.")

        n_links = len(links)
        if n_links == 0:
            gs.raise_exception("Target link not provided.")

        poss = list(poss) if poss is not None else []
        if not poss:
            poss = [None for _ in range(n_links)]
            pos_mask = [False, False, False]
        elif len(poss) != n_links:
            gs.raise_exception("Accepting only `poss` with length equal to `links` or empty list.")

        quats = list(quats) if quats is not None else []
        if not quats:
            quats = [None for _ in range(n_links)]
            rot_mask = [False, False, False]
        elif len(quats) != n_links:
            gs.raise_exception("Accepting only `quats` with length equal to `links` or empty list.")

        # Process local_points - default to origin [0, 0, 0] for each link
        local_points = list(local_points) if local_points is not None else []
        if not local_points:
            local_points = [None for _ in range(n_links)]
        elif len(local_points) != n_links:
            gs.raise_exception("Accepting only `local_points` with length equal to `links` or empty list.")
        for i, lp in enumerate(local_points):
            if lp is None:
                lp = [0.0, 0.0, 0.0]
            local_points[i] = torch.as_tensor(lp, dtype=gs.tc_float, device=gs.device)
        local_points = torch.stack(local_points, dim=0)  # (n_links, 3)

        link_pos_mask, link_rot_mask = [], []
        for i, (pos, quat) in enumerate(zip(poss, quats)):
            if pos is None and quat is None:
                gs.raise_exception("At least one of `poss` or `quats` must be provided.")
            link_pos_mask.append(pos is not None)
            poss[i] = broadcast_tensor(pos, gs.tc_float, (len(envs_idx), 3), ("envs_idx", "")).contiguous()
            link_rot_mask.append(quat is not None)
            if quat is None:
                quat = gu.identity_quat()
            quats[i] = broadcast_tensor(quat, gs.tc_float, (len(envs_idx), 4), ("envs_idx", "")).contiguous()
        link_pos_mask = torch.tensor(link_pos_mask, dtype=gs.tc_int, device=gs.device)
        link_rot_mask = torch.tensor(link_rot_mask, dtype=gs.tc_int, device=gs.device)
        poss = torch.stack(poss, dim=0)
        quats = torch.stack(quats, dim=0)

        custom_init_qpos = init_qpos is not None
        init_qpos = broadcast_tensor(
            init_qpos, gs.tc_float, (len(envs_idx), self.n_qs), ("envs_idx", "qs_idx")
        ).contiguous()

        # pos and rot mask
        pos_mask = broadcast_tensor(pos_mask, gs.tc_bool, (3,)).contiguous()
        rot_mask = broadcast_tensor(rot_mask, gs.tc_bool, (3,)).contiguous()
        if (num_axis := rot_mask.sum()) == 1:
            rot_mask = ~rot_mask if gs.tc_bool == torch.bool else 1 - rot_mask
        elif num_axis == 2:
            gs.raise_exception("You can only align 0, 1 axis or all 3 axes.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs)
        n_dofs = len(dofs_idx)
        if n_dofs == 0:
            gs.raise_exception("Target dofs not provided.")

        links_idx = torch.tensor([link.idx for link in links], dtype=gs.tc_int, device=gs.device)

        qpos, err_pose = self._solver.inverse_kinematics(
            self._idx_in_solver,
            self._q_start,
            self._link_start,
            self._joint_start,
            links_idx,
            dofs_idx,
            envs_idx,
            poss,
            quats,
            local_points,
            init_qpos,
            pos_mask,
            rot_mask,
            link_pos_mask,
            link_rot_mask,
            self.n_qs,
            self.n_dofs,
            custom_init_qpos,
            max_samples,
            max_solver_iters,
            damping,
            pos_tol,
            rot_tol,
            max_step_size,
            seed if seed is not None else 0,
            respect_joint_limit,
        )

        qpos = qpos[0] if self._solver.n_envs == 0 else qpos[envs_idx]
        if return_error:
            error_pose = err_pose[0] if self._solver.n_envs == 0 else err_pose[envs_idx]
            return qpos, error_pose
        return qpos


class RigidEntity(KinematicEntity):
    """
    Physics-enabled rigid entity (collision, constraints, dynamics).

    Inherits morphology, FK, IK, and DOF position get/set from KinematicEntity.
    Adds physics simulation methods: forces, velocities, contacts, etc.
    """

    if TYPE_CHECKING:
        material: gs.materials.Rigid
        _solver: "RigidSolver"

    _description_cls = RigidEntityDescription

    def __init__(self, scene: "Scene", solver: "RigidSolver", idx: int, desc: RigidEntityDescription):
        self._geom_start = solver.n_geoms
        self._cell_start = solver.n_cells
        self._vert_start = solver.n_verts
        self._face_start = solver.n_faces
        self._edge_start = solver.n_edges
        self._free_verts_state_start = solver.n_free_verts
        self._fixed_verts_state_start = solver.n_fixed_verts
        self._equality_start = 0
        self._free_verts_idx_local = torch.tensor([], dtype=gs.tc_int, device=gs.device)
        self._fixed_verts_idx_local = torch.tensor([], dtype=gs.tc_int, device=gs.device)
        self._visualize_contact: bool = desc.visualize_contact

        self._batch_fixed_verts: bool = desc.morphs[0].batch_fixed_verts

        super().__init__(scene, solver, idx, desc)

    def _add_heterogeneous_variant(self, link, v_link):
        # Add collision geometries
        coup_links = self.material.coup_links
        for g_desc in v_link.geoms:
            needs_coup = self.material.needs_coup and (coup_links is None or link.name in coup_links)
            link._add_geom(g_desc, needs_coup=needs_coup)

        # Add visual geoms and record vgeom range via parent
        super()._add_heterogeneous_variant(link, v_link)

        # Record geom range on the link (vgeom range already recorded by parent)
        link._record_variant_geom_range(len(v_link.geoms))

    def _reassign_heterogeneous_indices(self):
        """Reassign collision and visual geom indices for multi-link heterogeneous entities."""
        # Reassign collision geom indices sequentially
        running_idx = self._geom_start
        running_cell = self._cell_start
        running_vert = self._vert_start
        running_face = self._face_start
        running_edge = self._edge_start
        running_free_vs = self._free_verts_state_start
        running_fixed_vs = self._fixed_verts_state_start

        for link in self._links:
            for geom in link.geoms:
                geom._idx = running_idx
                geom._cell_start = running_cell
                geom._vert_start = running_vert
                geom._face_start = running_face
                geom._edge_start = running_edge
                if link.is_fixed and not self._batch_fixed_verts:
                    geom._verts_state_start = running_fixed_vs
                    running_fixed_vs += geom.n_verts
                else:
                    geom._verts_state_start = running_free_vs
                    running_free_vs += geom.n_verts
                running_idx += 1
                running_cell += geom.n_cells
                running_vert += geom.n_verts
                running_face += geom.n_faces
                running_edge += geom.n_edges

        # Reassign visual geom indices and recompute variant ranges via parent
        super()._reassign_heterogeneous_indices()

        # Recompute collision geom variant ranges from counts and reassigned indices
        for link in self._links:
            if link._variant_geom_ranges is None:
                continue
            geom_counts = [end - start for start, end in link._variant_geom_ranges]
            geom_cursor = link.geoms[0].idx if link.geoms else 0
            link._variant_geom_ranges = []
            for count in geom_counts:
                link._variant_geom_ranges.append((geom_cursor, geom_cursor + count))
                geom_cursor += count

    def _load_model(self):
        self._equalities = gs.List()
        # MJCF and USD express in-model collision filtering (MJCF '<contact><exclude>', USD CollisionGroup /
        # FilteredPairsAPI) through synthesized contype/conaffinity bitmasks. Those masks are only consistent within
        # the entity: applied across entities, they would spuriously disable collision against geoms whose default
        # masks happen not to overlap (e.g. the ground plane).
        self._is_local_collision_mask = isinstance(self._morph, (gs.morphs.MJCF, gs.morphs.USD))

        # Imported here since the couplers import the rigid entity package
        from genesis.engine.couplers import IPCCoupler

        # Make sure that the entity is not object
        if (
            isinstance(self.sim.coupler, IPCCoupler)
            and self.material.coup_type == "ipc_only"
            and any(l_desc.is_robot for l_desc in self._desc.links)
        ):
            gs.raise_exception("`RigidMaterial.coup_type='ipc_only'` only supported by rigid non-articulated objects.")

        super()._load_model()

        # Add equality constraints sequentially
        for e_desc in self._desc.equalities:
            self._add_equality(e_desc)

    def _add_link(self, l_desc):
        """Create one link from a description the resolution completed, with the joints and geoms it carries."""
        parent_idx = l_desc.parent_idx
        if parent_idx >= 0:
            parent_idx += self._link_start
        root_idx = l_desc.root_idx
        if root_idx is not None and root_idx >= 0:
            root_idx += self._link_start
        link_idx = self.n_links + self._link_start
        joint_start = self.n_joints + self._joint_start
        free_verts_start, fixed_verts_start = self._free_verts_state_start, self._fixed_verts_state_start
        for link in self.links:
            if link.is_fixed and not self._batch_fixed_verts:
                fixed_verts_start += link.n_verts
            else:
                free_verts_start += link.n_verts

        joints = self._create_joints(l_desc.joints, link_idx, joint_start)

        # Add child link
        link = RigidLink(
            entity=self,
            idx=link_idx,
            parent_idx=parent_idx,
            root_idx=root_idx,
            joint_start=joint_start,
            n_joints=len(l_desc.joints),
            geom_start=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            free_verts_state_start=free_verts_start,
            fixed_verts_state_start=fixed_verts_start,
            vgeom_start=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            visualize_contact=self.visualize_contact,
            desc=l_desc,
        )
        self._links.append(link)

        # Add visual geometries
        for g_desc in l_desc.vgeoms:
            link._add_vgeom(g_desc)
        coup_links = self.material.coup_links
        for g_desc in l_desc.geoms:
            needs_coup = self.material.needs_coup and (coup_links is None or link.name in coup_links)
            link._add_geom(g_desc, needs_coup=needs_coup)

        return link, joints

    def _build(self):
        self._n_geoms = self.n_geoms
        self._geoms = self.geoms

        # A link the world carries holds None where its asset states nothing (see 'RigidLinkDescription'). The solver
        # stores a value for every link, and the inertial of such a link never enters the dynamics, so zero it is.
        for link in self._links:
            if link.is_fixed:
                desc = link.desc
                if desc.mass is None:
                    desc.mass = 0.0
                if desc.inertial_pos is None:
                    desc.inertial_pos = gu.zero_pos()
                if desc.inertial_quat is None:
                    desc.inertial_quat = gu.identity_quat()
                if desc.inertia is None:
                    desc.inertia = np.zeros((3, 3), dtype=gs.np_float)

        super()._build()

        verts_start = 0
        free_verts_idx_local, fixed_verts_idx_local = [], []
        for link in self.links:
            verts_idx = torch.arange(verts_start, verts_start + link.n_verts, dtype=gs.tc_int, device=gs.device)
            if link.is_fixed and not self._batch_fixed_verts:
                fixed_verts_idx_local.append(verts_idx)
            else:
                free_verts_idx_local.append(verts_idx)
            verts_start += link.n_verts
        if free_verts_idx_local:
            self._free_verts_idx_local = torch.cat(free_verts_idx_local)
        if fixed_verts_idx_local:
            self._fixed_verts_idx_local = torch.cat(fixed_verts_idx_local)
        self._n_free_verts = len(self._free_verts_idx_local)
        self._n_fixed_verts = len(self._fixed_verts_idx_local)

        self._init_q_limit()

    def _add_equality(self, desc: RigidEqualityDescription):
        match desc.type:
            case gs.EQUALITY_TYPE.CONNECT | gs.EQUALITY_TYPE.WELD:
                objs_id = [self.get_link(obj_name).idx for obj_name in desc.objs_name]
            case gs.EQUALITY_TYPE.JOINT:
                objs_id = [self.get_joint(obj_name).idx if obj_name is not None else -1 for obj_name in desc.objs_name]
            case _:
                gs.raise_exception(
                    f"Equality type {desc.type} not supported. Only CONNECT, JOINT, and WELD are supported."
                )

        equality = RigidEquality(
            entity=self,
            idx=self.n_equalities + self._equality_start,
            eq_obj1id=objs_id[0],
            eq_obj2id=objs_id[1],
            desc=desc,
        )
        self._equalities.append(equality)
        return equality

    # ------------------------------------------------------------------------------------
    # --------------------------------- motion planing -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def plan_path(
        self,
        qpos_goal,
        qpos_start=None,
        max_nodes=2000,
        resolution=0.05,
        timeout=None,
        max_retry=1,
        smooth_path=True,
        num_waypoints=300,
        ignore_collision=False,
        planner="RRTConnect",
        envs_idx=None,
        return_valid_mask=False,
        *,
        ee_link_name=None,
        with_entity=None,
        **kwargs,
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters
        ----------
        qpos_goal : array_like
            The goal state. [B, Nq] or [1, Nq]
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used.
            Defaults to None. [B, Nq] or [1, Nq]
        resolution : float, optiona
            Joint-space resolution. It corresponds to the maximum distance between states to be checked
            for validity along a path segment.
        timeout : float, optional
            The max time to spend for each planning in seconds. Note that the timeout is not exact.
        max_retry : float, optional
            Maximum number of retry in case of timeout or convergence failure. Default to 1.
        smooth_path : bool, optional
            Whether to smooth the path after finding a solution. Defaults to True.
        num_waypoints : int, optional
            The number of waypoints to interpolate the path. If None, no interpolation will be performed.
            Defaults to 100.
        ignore_collision : bool, optional
            Whether to ignore collision checking during motion planning. Defaults to False.
        ignore_joint_limit : bool, optional
            This option has been deprecated and is not longer doing anything.
        planner : str, optional
            The name of the motion planning algorithm to use.
            Supported planners: 'RRT', 'RRTConnect'. Defaults to 'RRTConnect'.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        return_valid_mask: bool
            Obtain valid mask of the succesful planed path over batch.
        ee_link_name: str
            The name of the link, which we "attach" the object during the planning
        with_entity: RigidEntity
            The (non-articulated) object to "attach" during the planning

        Returns
        -------
        path : torch.Tensor
            A tensor of waypoints representing the planned path.
            Each waypoint is an array storing the entity's qpos of a single time step.
        is_invalid: torch.Tensor
            A tensor of boolean mask indicating the batch indices with failed plan.
        """
        if self._solver.n_envs > 0:
            n_envs = len(self._scene._sanitize_envs_idx(envs_idx))
        else:
            n_envs = 1

        if "ignore_joint_limit" in kwargs:
            gs.logger.warning("`ignore_joint_limit` is deprecated")

        ee_link_idx = None
        if ee_link_name is not None:
            assert with_entity is not None, "`with_entity` must be specified."
            ee_link_idx = self.get_link(ee_link_name).idx
        if with_entity is not None:
            assert ee_link_name is not None, "reference link of the robot must be specified."
            assert len(with_entity.links) == 1, "only non-articulated object is supported for now."

        # import here to avoid circular import
        from genesis.utils.path_planning import RRT, RRTConnect

        match planner:
            case "RRT":
                planner_obj = RRT(self)
            case "RRTConnect":
                planner_obj = RRTConnect(self)
            case _:
                gs.raise_exception(f"invalid planner {planner} specified.")

        path = torch.empty((num_waypoints, n_envs, self.n_qs), dtype=gs.tc_float, device=gs.device)
        is_invalid = torch.ones((n_envs,), dtype=torch.bool, device=gs.device)
        for i in range(1 + max_retry):
            retry_path, retry_is_invalid = planner_obj.plan(
                qpos_goal,
                qpos_start=qpos_start,
                resolution=resolution,
                timeout=timeout,
                max_nodes=max_nodes,
                smooth_path=smooth_path,
                num_waypoints=num_waypoints,
                ignore_collision=ignore_collision,
                envs_idx=envs_idx,
                ee_link_idx=ee_link_idx,
                obj_entity=with_entity,
            )
            # NOTE: update the previously failed path with the new results
            path[:, is_invalid] = retry_path[:, is_invalid]

            is_invalid &= retry_is_invalid
            if not is_invalid.any():
                break
            gs.logger.info(f"Planning failed. Retrying for {is_invalid.sum()} environments...")

        if self._solver.n_envs == 0:
            if return_valid_mask:
                return path.squeeze(1), ~is_invalid[0]
            return path.squeeze(1)

        if return_valid_mask:
            return path, ~is_invalid
        return path

    # ------------------------------------------------------------------------------------
    # ---------------------------------- control & io ------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_state(self):
        state = RigidEntityState(self, self._sim.cur_step_global)

        solver_state = self._solver.get_state()
        pos = solver_state.links_pos[:, self.base_link_idx]
        quat = solver_state.links_quat[:, self.base_link_idx]

        state._pos = pos
        state._quat = quat

        return state

    # ------------------------------------------------------------------------------------
    # -------------------------------- vertices / AABB -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_AABB(self, envs_idx=None, *, allow_fast_approx: bool = False):
        """
        Get the axis-aligned bounding box (AABB) of the entity in world frame by aggregating all the collision
        geometries associated with this entity.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        allow_fast_approx : bool
            Whether to allow fast approximation for efficiency if supported, i.e. 'LegacyCoupler' is enabled. In this
            case, each collision geometry is approximated by their pre-computed AABB in geometry-local frame, which is
            more efficiency but inaccurate.

        Returns
        -------
        aabb : torch.Tensor, shape (2, 3) or (n_envs, 2, 3)
            The AABB of the entity, where `[:, 0] = min_corner (x_min, y_min, z_min)` and
            `[:, 1] = max_corner (x_max, y_max, z_max)`.
        """
        from genesis.engine.couplers import LegacyCoupler

        if self.n_geoms == 0:
            gs.raise_exception("Entity has no collision geometries.")

        # Already computed internally by the solver. Let's access it directly for efficiency.
        if allow_fast_approx and isinstance(self.sim.coupler, LegacyCoupler):
            return self._solver.get_AABB(entities_idx=[self._idx_in_solver], envs_idx=envs_idx)[..., 0, :]

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx.
        # FIXME: Remove this branch after implementing 'get_verts'.
        if self._enable_heterogeneous and self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            n_envs = len(envs_idx)
            aabb_min = torch.full((n_envs, 3), float("inf"), dtype=gs.tc_float, device=gs.device)
            aabb_max = torch.full((n_envs, 3), float("-inf"), dtype=gs.tc_float, device=gs.device)
            for geom in self.geoms:
                geom_aabb = geom.get_AABB()
                active_mask = geom.active_envs_mask[envs_idx] if geom.active_envs_mask is not None else ()
                aabb_min[active_mask] = torch.minimum(aabb_min[active_mask], geom_aabb[envs_idx[active_mask], 0])
                aabb_max[active_mask] = torch.maximum(aabb_max[active_mask], geom_aabb[envs_idx[active_mask], 1])
            return torch.stack((aabb_min, aabb_max), dim=-2)

        # Compute the AABB on-the-fly based on the positions of all the vertices
        verts = self.get_verts()[envs_idx if envs_idx is not None else ()]
        return torch.stack((verts.min(dim=-2).values, verts.max(dim=-2).values), dim=-2)

    def get_aabb(self):
        raise DeprecationError("This method has been removed. Please use 'get_AABB()' instead.")

    @gs.assert_built
    def get_links_pos(
        self,
        links_idx_local=None,
        envs_idx=None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative=True,
    ):
        """
        Returns the position of a given reference point for all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: gs.link_ref_frame, optional
            The reference point used to express the position of each link: its origin ('link_origin'), its center of
            mass ('link_COM'), or the center of mass of the sub-entity it belongs to ('root_COM'). A single
            'RigidEntity' may comprise several physical sub-entities, each a kinematic sub-tree with at most one free
            joint at its root. Defaults to 'link_origin'.
        relative : bool, optional
            Whether to report the position of the authored link origin rather than of the internal link origin used by
            the solver. The internal link origin is the authored one moved by the morph 'offset_pos' / 'offset_quat'
            and, on a free root link with 'align=True', by the inertial alignment. Only
            'ref=gs.link_ref_frame.link_origin' is affected, since the offset is defined on the link origin. Defaults to
            True.

        Returns
        -------
        pos : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The position of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_pos(links_idx, envs_idx, ref=ref, relative=relative)

    @gs.assert_built
    def get_links_vel(
        self, links_idx_local=None, envs_idx=None, *, ref: link_ref_frame = link_ref_frame.link_origin, relative=True
    ):
        """
        Returns linear velocity of all the entity's links expressed at a given reference position in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: gs.link_ref_frame, optional
            The reference point used to express the velocity of each link: its origin ('link_origin') or its center of
            mass ('link_COM'). Defaults to 'link_origin'.
        relative : bool, optional
            Whether to report the velocity of the authored link origin rather than of the internal link origin used by
            the solver. The internal link origin is the authored one moved by the world-frame vector 'd'. That
            displacement composes the morph 'offset_pos' / 'offset_quat' and, on a free root link with 'align=True', the
            inertial alignment. Both readings are expressed in world coordinates and differ by the transport 'omega x
            d'. Only 'ref=gs.link_ref_frame.link_origin' is affected. Defaults to True.

        Returns
        -------
        vel : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear velocity of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_vel(links_idx, envs_idx, ref=ref, relative=relative)

    @gs.assert_built
    def get_links_acc(self, links_idx_local=None, envs_idx=None, *, relative=True):
        """
        Returns classical linear acceleration of all the entity's links expressed at their origin in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            Whether to report the acceleration of the authored link origin rather than of the internal link origin used
            by the solver. The internal link origin is the authored one moved by the world-frame vector 'd'. That
            displacement composes the morph 'offset_pos' / 'offset_quat' and, on a free root link with 'align=True', the
            inertial alignment. Both readings are expressed in world coordinates and differ by the transport 'alpha x d
            + omega x (omega x d)'. Defaults to True.

        Returns
        -------
        acc : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The classical linear acceleration of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc(links_idx, envs_idx, relative=relative)

    @gs.assert_built
    def get_links_acc_ang(self, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc_ang(links_idx, envs_idx)

    @gs.assert_built
    def apply_links_external_wrench(
        self,
        force=None,
        torque=None,
        links_idx_local=None,
        envs_idx=None,
        *,
        pos=None,
        ref: link_ref_frame = link_ref_frame.link_origin,
        local: bool = False,
    ):
        """
        Apply an external wrench over one simulation step on a set of the entity's links.

        Parameters
        ----------
        force : None | array_like, optional
            The linear force to apply. None for a pure torque. Defaults to None.
        torque : None | array_like, optional
            The torque to apply, on top of the moment induced by the linear force. Defaults to None.
        links_idx_local : None | array_like, optional
            The indices of the links. None to specify all the entity's links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        pos : None | array_like, optional
            Where the linear force is applied, which sets the moment arm of the induced torque. With `local=True`, an
            offset from the origin of the `ref` frame, so the point follows each link as it moves; otherwise a world
            position. None applies the force at the origin of the `ref` frame. Defaults to None.
        ref: gs.link_ref_frame, optional
            The reference frame: the origin of each link ('link_origin'), or its center of mass ('link_COM'). It fixes
            where the linear force acts when `pos` is None, and the axes of the input coordinates when `local=True`.
            Defaults to 'link_origin'.
        local: bool, optional
            Whether `force`, `torque` and `pos` are expressed in the coordinates of the `ref` frame rather than the
            world frame. Defaults to False.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.apply_links_external_wrench(force, torque, links_idx, envs_idx, pos=pos, ref=ref, local=local)

    # ------------------------------------------------------------------------------------
    # ----------------------------- links mass properties --------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_links_mass(self, links_idx_local=None, envs_idx=None):
        """
        Get the mass of each link, in kg.

        Parameters
        ----------
        links_idx_local : None | array_like, optional
            The indices of the links on this entity. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        mass : torch.Tensor, shape (n_links,) or (n_envs, n_links)
            The mass of each link, per environment for a scene whose link info is batched.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_mass(links_idx, envs_idx)

    @gs.assert_built
    def get_links_COM(self, links_idx_local=None, envs_idx=None):
        """
        Get the center of mass (COM) of each link, as an offset in its local frame.

        That local frame is the one specified by the morph, except for a floating-base link loaded with `align=True`:
        Genesis then moves its origin onto the center of mass and rotates its axes onto the principal axes of inertia,
        so the offset returned for such a link is zero.

        Parameters
        ----------
        links_idx_local : None | array_like, optional
            The indices of the links on this entity. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        com : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The center of mass of each link, per environment for a scene whose link info is batched.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_COM(links_idx, envs_idx)

    @gs.assert_built
    def get_links_inertia(self, links_idx_local=None, envs_idx=None):
        """
        Get the inertia matrix of each link, expressed in its inertial frame.

        Parameters
        ----------
        links_idx_local : None | array_like, optional
            The indices of the links on this entity. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        inertia : torch.Tensor, shape (n_links, 3, 3) or (n_envs, n_links, 3, 3)
            The inertia matrix of each link, per environment for a scene whose link info is batched.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_inertia(links_idx, envs_idx)

    @gs.assert_built
    def get_links_invweight(self, links_idx_local=None, envs_idx=None):
        """
        Get the constraint inverse weights of each link: one for forces, one for torques.

        The constraint solver models contacts, joint limits and equalities softly, and the regularization it adds to a
        row is proportional to the inverse weights of the links that row couples. That normalizes the row by the
        acceleration a unit impulse produces on them, so these weights change the solution the solve converges to. In
        practice, they are tuned to make the solver parameters independent of how heavy the bodies are. These
        regularization parameters are computed from the mass, the center of mass and the inertia of a link, and from the
        armature of the degrees of freedom between it and the root of its tree, and are recomputed whenever one of these
        quantities changes.

        Parameters
        ----------
        links_idx_local : None | array_like, optional
            The indices of the links on this entity. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        invweight : torch.Tensor, shape (n_links, 2) or (n_envs, n_links, 2)
            The translational and rotational inverse weight of each link, per environment for a scene whose link info
            is batched.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_invweight(links_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------- base pos/quat get/set --------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @tracked
    def set_pos(self, pos, envs_idx=None, *, zero_velocity=True, relative=True, skip_forward=False):
        """
        Set position of the entity's base link.

        Parameters
        ----------
        pos : array_like
            The position to set.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        relative : bool, optional
            Whether 'pos' places the authored link origin rather than directly the internal link origin used by the
            solver. The internal link origin is the authored one moved by the morph 'offset_pos' / 'offset_quat' and, on
            a free root link with 'align=True', by the inertial alignment. Defaults to True.
        skip_forward : bool, optional
            Whether to skip forward kinematics after setting position. Defaults to False.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type is not None and self.base_link.is_fixed:
            gs.raise_exception(
                "This method is only supported by `RigidMaterial.coup_type=None` for fixed-based rigid entities."
            )
        super().set_pos(pos, envs_idx, zero_velocity=zero_velocity, relative=relative, skip_forward=skip_forward)

    @gs.assert_built
    def set_pos_grad(self, envs_idx, relative, pos_grad):
        self._solver.set_base_links_pos_grad(self.base_link_idx, envs_idx, relative, pos_grad.data)

    @gs.assert_built
    @tracked
    def set_quat(self, quat, envs_idx=None, *, zero_velocity=True, relative=True, skip_forward=False):
        """
        Set quaternion of the entity's base link.

        Parameters
        ----------
        quat : array_like
            The quaternion to set.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        relative : bool, optional
            Whether 'quat' orients the authored link origin rather than directly the internal link origin used by the
            solver. The internal link origin is the authored one moved by the morph 'offset_pos' / 'offset_quat' and, on
            a free root link with 'align=True', by the inertial alignment. Defaults to True.
        skip_forward : bool, optional
            Whether to skip forward kinematics after setting quaternion. Defaults to False.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type is not None and self.base_link.is_fixed:
            gs.raise_exception(
                "This method is only supported by `RigidMaterial.coup_type=None` for fixed-based rigid entities."
            )
        super().set_quat(quat, envs_idx, zero_velocity=zero_velocity, relative=relative, skip_forward=skip_forward)

    @gs.assert_built
    def set_quat_grad(self, envs_idx, relative, quat_grad):
        self._solver.set_base_links_quat_grad(self.base_link_idx, envs_idx, relative, quat_grad.data)

    @gs.assert_built
    def get_verts(self):
        """
        Get the all vertices of the entity based on collision geometries.

        Returns
        -------
        verts : torch.Tensor, shape (n_envs, n_verts, 3)
            The vertices of the entity.
        """
        if self._enable_heterogeneous:
            gs.raise_exception("This method is not supported by heterogeneous entities.")

        self._solver.update_verts_for_geoms(slice(self.geom_start, self.geom_end))

        n_fixed_verts, n_free_vertices = self._n_fixed_verts, self._n_free_verts
        tensor = torch.empty((self._solver._B, n_fixed_verts + n_free_vertices, 3), dtype=gs.tc_float, device=gs.device)

        if n_fixed_verts > 0:
            verts_idx = slice(self._fixed_verts_state_start, self._fixed_verts_state_start + n_fixed_verts)
            fixed_verts_state = qd_to_torch(self._solver.dyn_state.fixed_verts.pos, verts_idx)
            tensor[:, self._fixed_verts_idx_local] = fixed_verts_state
        if n_free_vertices > 0:
            verts_idx = slice(self._free_verts_state_start, self._free_verts_state_start + n_free_vertices)
            free_verts_state = qd_to_torch(self._solver.dyn_state.free_verts.pos, None, verts_idx, transpose=True)
            tensor[:, self._free_verts_idx_local] = free_verts_state

        if self._solver.n_envs == 0:
            tensor = tensor[0]
        return tensor

    # ------------------------------------------------------------------------------------
    # --------------------------------- qpos get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_qpos(self, qpos, qs_idx_local=None, envs_idx=None, *, zero_velocity=True, skip_forward=False):
        """
        Set the entity's qpos.

        Parameters
        ----------
        qpos : array_like
            The qpos to set.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "external_articulation":
            gs.raise_exception("This method is not supported by `RigidMaterial.coup_type='external_articulation'`.")
        super().set_qpos(qpos, qs_idx_local, envs_idx, zero_velocity=zero_velocity, skip_forward=skip_forward)

    @gs.assert_built
    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' positional gains for the PD controller.

        Parameters
        ----------
        kp : array_like
            The positional gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_kp(kp, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_kv(self, kv, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' velocity gains for the PD controller.

        Parameters
        ----------
        kv : array_like
            The velocity gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_kv(kv, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_act_gain(self, act_gain, dofs_idx_local=None, envs_idx=None):
        """
        Set the actuator gain for the entity's dofs. Invalidates PD-reducibility.

        Parameters
        ----------
        act_gain : array_like
            The actuator gain values.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_act_gain(act_gain, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_act_bias(self, bias0, bias1, bias2, dofs_idx_local=None, envs_idx=None):
        """
        Set the actuator bias for the entity's dofs.

        Parameters
        ----------
        bias0 : array_like
            Constant bias term.
        bias1 : array_like
            Position coefficient.
        bias2 : array_like
            Velocity coefficient.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_act_bias(bias0, bias1, bias2, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_force_range(self, lower, upper, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' force range.

        Parameters
        ----------
        lower : array_like
            The lower bounds of the force range.
        upper : array_like
            The upper bounds of the force range.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_force_range(lower, upper, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_stiffness(self, stiffness, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_stiffness(stiffness, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_invweight(self, invweight, dofs_idx_local=None, envs_idx=None):
        raise DeprecationError(
            "This method has been removed because dof invweights are supposed to be a by-product of link properties "
            "(mass, pose, and inertia matrix), joint placements, and dof armatures. Please consider using the "
            "considering setters instead."
        )

    @gs.assert_built
    def set_dofs_armature(self, armature, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_armature(armature, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_damping(self, damping, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_damping(damping, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_frictionloss(self, frictionloss, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' friction loss.
        Parameters
        ----------
        frictionloss : array_like
            The friction loss values to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_frictionloss(frictionloss, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_velocity_grad(self, dofs_idx_local, envs_idx, velocity_grad):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_velocity_grad(dofs_idx, envs_idx, velocity_grad.data)

    @gs.assert_built
    def set_dofs_force_grad(self, dofs_idx_local, envs_idx, force_grad):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_force_grad(dofs_idx, envs_idx, force_grad.data)

    # ------------------------------------------------------------------------------------
    # ----------------------------- DOF property setters ---------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None, *, zero_velocity=True):
        """
        Set the entity's dofs' position.

        Parameters
        ----------
        position : array_like
            The position to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`,
            not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "external_articulation":
            gs.raise_exception("This method is not supported by `RigidMaterial.coup_type='external_articulation'`.")
        super().set_dofs_position(position, dofs_idx_local, envs_idx, zero_velocity=zero_velocity)

    # ------------------------------------------------------------------------------------
    # ---------------------------------- PD control --------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @tracked
    def control_dofs_force(self, force, dofs_idx_local=None, envs_idx=None):
        """
        Control the entity's dofs' motor force. This is used for force/torque control.

        Parameters
        ----------
        force : array_like
            The force to apply.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "ipc_only":
            gs.raise_exception("This method is not supported for `coup_type='ipc_only'` entities.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_force(force, dofs_idx, envs_idx)

    @gs.assert_built
    @tracked
    def control_dofs_velocity(self, velocity, dofs_idx_local=None, envs_idx=None):
        """
        Set the PD controller's target velocity for the entity's dofs. This is used for velocity control.

        Parameters
        ----------
        velocity : array_like
            The target velocity to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "ipc_only":
            gs.raise_exception("This method is not supported for `coup_type='ipc_only'` entities.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_velocity(velocity, dofs_idx, envs_idx)

    @gs.assert_built
    @tracked
    def control_dofs_position(self, position, dofs_idx_local=None, envs_idx=None):
        """
        Set the position controller's target position for the entity's dofs. The controller is a proportional term
        plus a velocity damping term (virtual friction).

        Parameters
        ----------
        position : array_like
            The target position to set.
        dofs_idx_local : array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "ipc_only":
            gs.raise_exception("This method is not supported for `coup_type='ipc_only'` entities.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_position(position, dofs_idx, envs_idx)

    @gs.assert_built
    @tracked
    def control_dofs_position_velocity(self, position, velocity, dofs_idx_local=None, envs_idx=None):
        """
        Set a PD controller's target position and velocity for the entity's dofs. This is used for position control.

        Parameters
        ----------
        position : array_like
            The target position to set.
        velocity : array_like
            The target velocity
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self.sim.coupler, IPCCoupler) and self.material.coup_type == "ipc_only":
            gs.raise_exception("This method is not supported for `coup_type='ipc_only'` entities.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_position_velocity(position, velocity, dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_control_force(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' internal control force, computed based on the position/velocity control command.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        control_force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' internal control force.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_control_force(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' internal force at the current time step.

        Note
        ----
        Different from `get_dofs_control_force`, this function returns the actual internal force experienced by all the dofs at the current time step.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' force.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_force(dofs_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------- DOF property getters ---------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_dofs_kp(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the positional gain (kp) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kp : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The positional gain (kp) for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_kp(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_kv(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the velocity gain (kv) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kv : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The velocity gain (kv) for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_kv(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_act_gain(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the actuator gain for the entity's dofs.

        Returns
        -------
        act_gain : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_act_gain(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_act_bias(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the actuator bias [constant, pos_coeff, vel_coeff] for the entity's dofs.

        Returns
        -------
        bias0, bias1, bias2 : tuple of torch.Tensor
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_act_bias(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_force_range(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the force range (min and max limits) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The lower limit of the force range for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The upper limit of the force range for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_force_range(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_stiffness(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_stiffness(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_invweight(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_invweight(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_armature(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_armature(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_damping(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_damping(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_frictionloss(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the friction loss for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        frictionloss : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The friction loss for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_frictionloss(dofs_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # -------------------------------- physics queries -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_mass_mat(self, envs_idx=None, decompose=False):
        dofs_idx = self._get_global_idx(None, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_mass_mat(dofs_idx, envs_idx, decompose)

    @gs.assert_built
    def get_kinetic_energy(self, envs_idx=None) -> torch.Tensor:
        """Get the total kinetic energy of the entity in Joules [J] (translational + rotational).

        Summed over the entity's links, each contributing ``0.5 * V^T * I * V`` for its spatial velocity ``V`` and
        spatial inertia ``I`` about the center of mass (COM) of its kinematic tree, plus the motor armature
        contribution ``0.5 * sum_d(armature_d * dq_d^2)`` of its DOFs.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kinetic_energy : torch.Tensor, shape () or (n_envs,)
        """
        links_idx = self._get_global_idx(None, self.n_links, self._link_start, unsafe=True)
        dofs_idx = self._get_global_idx(None, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_kinetic_energy(links_idx, dofs_idx, envs_idx)

    @gs.assert_built
    def get_potential_energy(self, envs_idx=None) -> torch.Tensor:
        """Get the total potential energy of the entity in Joules [J] (gravitational + joint springs).

        Gravity contributes ``-sum_i(m_i * g^T * p_i)`` over the links of the entity free to move, where ``p_i`` is the
        center-of-mass position of link *i* and ``g`` is the gravity vector obtained from the solver. Its joint springs
        contribute ``0.5 * sum_d(stiffness_d * (q_d - q0_d)^2)``, the elastic energy stored by holding each DOF away
        from its neutral position.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        potential_energy : torch.Tensor, shape () or (n_envs,)
        """
        links_idx = self._get_global_idx(None, self.n_links, self._link_start, unsafe=True)
        dofs_idx = self._get_global_idx(None, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_potential_energy(links_idx, dofs_idx, envs_idx)

    @gs.assert_built
    def get_total_energy(self, envs_idx=None) -> torch.Tensor:
        """Get the total mechanical energy of the entity in Joules [J] (kinetic + potential).

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        total_energy : torch.Tensor, shape () or (n_envs,)
        """
        return self.get_kinetic_energy(envs_idx=envs_idx) + self.get_potential_energy(envs_idx=envs_idx)

    @gs.assert_built
    def detect_collision(self, env_idx=0):
        """
        Detects collision for the entity. This only supports a single environment.

        Note
        ----
        This function re-detects real-time collision for the entity, so it doesn't rely on scene.step() and can be used for applications like motion planning, which doesn't require physical simulation during state sampling.

        Parameters
        ----------
        env_idx : int, optional
            The index of the environment. Defaults to 0.
        """

        all_collision_pairs = self._solver.detect_collision(env_idx)
        collision_pairs = all_collision_pairs[
            np.logical_and(all_collision_pairs >= self.geom_start, all_collision_pairs < self.geom_end).any(axis=1)
        ]
        return collision_pairs

    @gs.assert_built
    def get_contacts(self, with_entity=None, exclude_self_contact=False, is_padded=False):
        """
        Returns contact information computed during the most recent `scene.step()`.
        If `with_entity` is provided, only returns contact information involving the caller and the specified entity.
        Otherwise, returns all contact information involving the caller entity.
        When `with_entity` is `self`, it will return the self-collision only.

        The returned dict contains the following keys (a contact pair consists of two geoms: A and B):

        - 'geom_a'     : The global geom index of geom A in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_a])
        - 'geom_b'     : The global geom index of geom B in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_b])
        - 'link_a'     : The global link index of link A (that contains geom A) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_a])
        - 'link_b'     : The global link index of link B (that contains geom B) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_b])
        - 'position'   : The contact position in world frame.
        - 'force_a'    : The contact force applied to geom A.
        - 'force_b'    : The contact force applied to geom B.
        - 'valid_mask' : A boolean mask indicating whether the contact information is valid.
                        (Only when scene is parallelized)

        The shape of each entry is (n_envs, n_contacts, ...) for scene with parallel envs
                               and (n_contacts, ...) for non-parallelized scene.

        Parameters
        ----------
        with_entity : RigidEntity, optional
            The entity to check contact with. Defaults to None.
        exclude_self_contact: bool
            Exclude the self collision from the returning contacts. Defaults to False.
        is_padded: bool
            Return tensors padded to a fixed capacity along the contact axis instead of trimmed to the live
            contact count, with 'valid_mask' flagging the real contacts. This avoids a per-step device-to-host
            synchronization; the values are otherwise identical on every backend. Defaults to False.

        Returns
        -------
        contact_info : dict
            The contact information.
        """
        contact_data = self._solver.collider.get_contacts(as_tensor=True, to_torch=True, is_padded=is_padded)
        n_contacts = contact_data["n_contacts"] if is_padded else None
        if is_padded:
            del contact_data["n_contacts"]

        logical_operation = torch.logical_xor if exclude_self_contact else torch.logical_or
        if with_entity is not None and self.idx == with_entity.idx:
            if exclude_self_contact:
                gs.raise_exception("`with_entity` is self but `exclude_self_contact` is True.")
            logical_operation = torch.logical_and

        valid_mask = logical_operation(
            torch.logical_and(contact_data["geom_a"] >= self.geom_start, contact_data["geom_a"] < self.geom_end),
            torch.logical_and(contact_data["geom_b"] >= self.geom_start, contact_data["geom_b"] < self.geom_end),
        )
        if with_entity is not None and self.idx != with_entity.idx:
            valid_mask = torch.logical_and(
                valid_mask,
                torch.logical_or(
                    torch.logical_and(
                        contact_data["geom_a"] >= with_entity.geom_start, contact_data["geom_a"] < with_entity.geom_end
                    ),
                    torch.logical_and(
                        contact_data["geom_b"] >= with_entity.geom_start, contact_data["geom_b"] < with_entity.geom_end
                    ),
                ),
            )

        if n_contacts is not None:
            slots = torch.arange(valid_mask.shape[-1], device=valid_mask.device)
            if self._solver.n_envs == 0:
                valid_mask = torch.logical_and(valid_mask, slots < n_contacts.reshape(()))
            else:
                valid_mask = torch.logical_and(valid_mask, slots[None, :] < n_contacts[:, None])

        if self._solver.n_envs == 0 and not is_padded:
            contact_data = {key: value[valid_mask] for key, value in contact_data.items()}
        else:
            contact_data["valid_mask"] = valid_mask

        contact_data["force_a"] = -contact_data["force"]
        contact_data["force_b"] = +contact_data["force"]
        del contact_data["force"]

        return contact_data

    def get_links_net_contact_force(self, envs_idx=None):
        """
        Returns net force applied on each links due to direct external contacts.

        Returns
        -------
        entity_links_force : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The net force applied on each links due to direct external contacts.
        """
        links_idx = slice(self.link_start, self.link_end)
        tensor = qd_to_torch(self._solver.dyn_state.links.contact_force, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self._solver.n_envs == 0 else tensor

    # ------------------------------------------------------------------------------------
    # ----------------------------------- friction ---------------------------------------
    # ------------------------------------------------------------------------------------

    def set_friction_ratio(self, friction_ratio, links_idx_local=None, envs_idx=None):
        """
        Set the friction ratio of the geoms of the specified links.

        Parameters
        ----------
        friction_ratio : torch.Tensor, shape (n_envs, n_links)
            The friction ratio
        links_idx_local : array_like
            The indices of the links to set friction ratio.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx_local = self._get_global_idx(links_idx_local, self.n_links, 0, unsafe=True)

        links_n_geoms = torch.tensor(
            [self._links[i_l].n_geoms for i_l in links_idx_local], dtype=gs.tc_int, device=gs.device
        )
        links_friction_ratio = torch.as_tensor(friction_ratio, dtype=gs.tc_float, device=gs.device)
        geoms_friction_ratio = torch.repeat_interleave(links_friction_ratio, links_n_geoms, dim=-1)
        geoms_idx = [
            i_g for i_l in links_idx_local for i_g in range(self._links[i_l].geom_start, self._links[i_l].geom_end)
        ]

        self._solver.set_geoms_friction_ratio(geoms_friction_ratio, geoms_idx, envs_idx)

    def set_friction(self, friction):
        """
        Set the friction coefficient of all the links (and in turn, geometries) of the rigid entity.

        Note
        ----
        The friction coefficient associated with a pair of geometries in contact is defined as the maximum between
        their respective values, so one must be careful the set the friction coefficient properly for both of them.

        Warning
        -------
        The friction coefficient must be in range [1e-2, 5.0] for simulation stability.

        Parameters
        ----------
        friction : float
            The friction coefficient to set.
        """

        if friction < 1e-2 or friction > 5.0:
            gs.raise_exception("`friction` must be in the range [1e-2, 5.0] for simulation stability.")

        for link in self._links:
            link.set_friction(friction)

    def set_friction_torsional(self, friction_torsional):
        """
        Set the torsional friction coefficient of all the links (and in turn, geometries) of the rigid entity.

        Note
        ----
        The torsional friction coefficient associated with a pair of geometries in contact is defined as the maximum
        between their respective values (see 'gs.materials.Rigid'). Only effective when torsional friction is enabled
        at the scene level (see 'RigidOptions.enable_torsional_friction').

        Parameters
        ----------
        friction_torsional : float
            The torsional friction coefficient to set.
        """
        if friction_torsional < 0:
            gs.raise_exception("`friction_torsional` must be non-negative.")

        for link in self._links:
            link.set_friction_torsional(friction_torsional)

    def set_friction_rolling(self, friction_rolling):
        """
        Set the rolling friction coefficient of all the links (and in turn, geometries) of the rigid entity.

        Note
        ----
        The rolling friction coefficient associated with a pair of geometries in contact is defined as the maximum
        between their respective values (see 'gs.materials.Rigid'). Only effective when rolling friction is enabled
        at the scene level (see 'RigidOptions.enable_rolling_friction').

        Parameters
        ----------
        friction_rolling : float
            The rolling friction coefficient to set.
        """
        if friction_rolling < 0:
            gs.raise_exception("`friction_rolling` must be non-negative.")

        for link in self._links:
            link.set_friction_rolling(friction_rolling)

    # ------------------------------------------------------------------------------------
    # --------------------------------- mass / inertia -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_links_mass(self, mass, links_idx_local=None, envs_idx=None):
        """
        Set the mass of the given links, in kg.

        The inertia of a link is left unchanged. Use `set_links_inertia` to change it too, or `set_mass` to set the
        total mass of the entity, which keeps the ratios between the masses of its links and scales their inertia along
        with them.

        Parameters
        ----------
        mass : array_like, shape (n_links,) or (n_envs, n_links)
            The mass of each link, in kg. Per-environment values require `RigidOptions.batch_links_info=True`.
        links_idx_local : None | array_like, optional
            The indices of the links. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_mass(mass, links_idx, envs_idx)

    @gs.assert_built
    def set_links_invweight(self, invweight, links_idx_local=None, envs_idx=None):
        """
        Removed: Genesis computes the inverse weights of a link from its mass, its center of mass, its inertia and
        the armature of the degrees of freedom above it, so write one of those instead.
        """
        raise DeprecationError("This method has been removed because links invweights are supposed to be a by-product.")

    @gs.assert_built
    def set_links_COM(self, com, links_idx_local=None, envs_idx=None):
        """
        Set the center of mass (COM) of the given links, as an offset in their local frame.

        Parameters
        ----------
        com : array_like, shape (n_links, 3) or (n_envs, n_links, 3)
            The center of mass of each link. Per-environment values require `RigidOptions.batch_links_info=True`.
        links_idx_local : None | array_like, optional
            The indices of the links. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_COM(com, links_idx, envs_idx)

    @gs.assert_built
    def set_links_inertia(self, inertia, links_idx_local=None, envs_idx=None):
        """
        Set the inertia matrix of the given links, expressed in their inertial frame.

        Parameters
        ----------
        inertia : array_like, shape (n_links, 3, 3) or (n_envs, n_links, 3, 3)
            The inertia matrix of each link, which must be symmetric positive definite. Per-environment values require
            `RigidOptions.batch_links_info=True`.
        links_idx_local : None | array_like, optional
            The indices of the links. If None, all links are considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_inertia(inertia, links_idx, envs_idx)

    @gs.assert_built
    def set_mass(self, mass, envs_idx=None):
        """
        Set the total mass of the entity, in kg, keeping the ratios between the masses of its links.

        The inertia of each link is scaled by the same factor as its mass, exactly how a body of the same shape made
        heavier would behave. `RigidLink.set_mass` sets the mass of a single link and leaves its inertia unchanged.

        An entity whose links are all fixed to the world moves no mass, so setting its total mass is ill-defined and
        raises.

        Parameters
        ----------
        mass : float | array_like, shape (n_envs,)
            The mass to set. Per-environment values require `RigidOptions.batch_links_info=True`.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        # The mass of an entity is the mass it moves, so a link fixed to the world takes no part in the total nor in
        # the fractions.
        links_idx = [link.idx for link in self.links if not link.is_fixed]
        links_mass = self._solver.get_links_mass(links_idx, envs_idx)
        mass_entity = links_mass.sum(dim=-1, keepdim=True)
        dim_names = ("envs_idx", "") if self._solver.is_links_info_batched else ("",)
        mass_target = broadcast_tensor(mass, gs.tc_float, mass_entity.shape, dim_names)
        # A body brought to a mass behaves as one built at it, so the inertia of every link grows with its share.
        self._solver.set_links_mass(links_mass * mass_target / mass_entity, links_idx, envs_idx, scale_inertia=True)

    @gs.assert_built
    def get_mass(self, envs_idx=None):
        """
        Get the total mass of the entity, in kg.

        Sums the links that are not fixed to the world, a fixed link never being accelerated, causing its mass to be
        ill-defined as it is indistinguishable from the world itself.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        mass : torch.Tensor, shape (n_envs,) or scalar
            The total mass of the entity in kg.
        """
        links_idx = [link.idx for link in self.links if not link.is_fixed]
        return self._solver.get_links_mass(links_idx, envs_idx).sum(dim=-1)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def visualize_contact(self):
        """Whether to visualize contact force."""
        return self._visualize_contact

    @property
    def n_geoms(self):
        """The number of collision geom `RigidGeom` in the entity."""
        if self._is_built:
            return self._n_geoms
        return sum(link.n_geoms for link in self._links)

    @property
    def n_cells(self):
        """The number of sdf cells in the entity."""
        return sum(link.n_cells for link in self._links)

    @property
    def n_verts(self):
        """The number of vertices (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_verts for link in self._links)

    @property
    def n_faces(self):
        """The number of faces (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_faces for link in self._links)

    @property
    def n_edges(self):
        """The number of edges (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_edges for link in self._links)

    @property
    def geom_start(self):
        """The index of the entity's first RigidGeom in the scene."""
        return self._geom_start

    @property
    def geom_end(self):
        """The index of the entity's last RigidGeom in the scene *plus one*."""
        return self._geom_start + self.n_geoms

    @property
    def cell_start(self):
        """The start index the entity's sdf cells in the scene."""
        return self._cell_start

    @property
    def cell_end(self):
        """The end index the entity's sdf cells in the scene *plus one*."""
        return self._cell_start + self.n_cells

    @property
    def gravity_compensation(self):
        """Apply a force to compensate gravity. A value of 1 will make a zero-gravity behavior. Default to 0"""
        return self.material.gravity_compensation

    @property
    def vert_start(self):
        """The index of the entity's first `vert` (collision vertex) in the scene."""
        return self._vert_start

    @property
    def face_start(self):
        """The index of the entity's first `face` (collision face) in the scene."""
        return self._face_start

    @property
    def edge_start(self):
        """The index of the entity's first `edge` (collision edge) in the scene."""
        return self._edge_start

    @property
    def geoms(self) -> list[RigidGeom]:
        """The list of collision geoms (`RigidGeom`) in the entity."""
        if self.is_built:
            return self._geoms
        return gs.List(geom for link in self._links for geom in link.geoms)

    @property
    def n_equalities(self):
        """The number of equality constraints in the entity."""
        return len(self._equalities)

    @property
    def equality_start(self):
        """The index of the entity's first RigidEquality in the scene."""
        return self._equality_start

    @property
    def equality_end(self):
        """The index of the entity's last RigidEquality in the scene *plus one*."""
        return self._equality_start + self.n_equalities

    @property
    def equalities(self):
        """The list of equality constraints (`RigidEquality`) in the entity."""
        return self._equalities

    @property
    def is_free(self) -> bool:
        raise DeprecationError("This property has been removed.")

    @property
    def is_local_collision_mask(self):
        """Whether the contype and conaffinity bitmasks of this entity only applies to self-collision."""
        return self._is_local_collision_mask
