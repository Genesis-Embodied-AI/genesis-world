from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity import KinematicEntity
from genesis.engine.materials import Kinematic
from genesis.engine.solvers.rigid.abd.inverse_kinematics import (
    kernel_forward_kinematics_query,
    kernel_get_jacobian,
    kernel_get_jacobian_zero,
    kernel_inverse_kinematics_entity,
    kernel_set_ik_targets,
)
from genesis.engine.states.solvers import KinematicSolverCheckpoint, KinematicSolverState
from genesis.options.morphs import Morph
from genesis.options.solvers import KinematicOptions
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_torch,
    qd_zero_grad,
    sanitize_indexed_tensor,
)

from .base_solver import MutatedLinks, Solver, StateChange, mutates
from .rigid.abd.accessor import (
    kernel_get_kinematic_state,
    kernel_get_links_vel,
    kernel_get_state_grad,
    kernel_get_terrain_height,
    kernel_set_dofs_force_grad,
    kernel_set_dofs_position,
    kernel_set_dofs_velocity,
    kernel_set_dofs_velocity_grad,
    kernel_set_dofs_zero_velocity,
    kernel_set_kinematic_state,
    kernel_set_links_pos,
    kernel_set_links_pos_grad,
    kernel_set_links_quat,
    kernel_set_links_quat_grad,
    kernel_set_qpos,
    kernel_set_vverts,
)
from .rigid.abd.forward_kinematics import (
    kernel_forward_kinematics,
    kernel_forward_velocity,
    kernel_masked_forward_kinematics,
    kernel_masked_forward_velocity,
    kernel_update_vgeoms,
    kernel_update_vverts_for_vgeoms,
)
from .rigid.abd.misc import (
    kernel_init_dof_fields,
    kernel_init_entity_fields,
    kernel_init_joint_fields,
    kernel_init_link_fields,
    kernel_init_vgeom_fields,
    kernel_init_vvert_fields,
    kernel_update_heterogeneous_links_vgeom,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


TERRAIN_HEIGHT_QUERY_TILT_TOLERANCE = 1e-3


def _balanced_variant_mapping(n_variants, B):
    """Map N variants to B environments using balanced block assignment."""
    if B >= n_variants:
        base = B // n_variants
        extra = B % n_variants
        sizes = np.r_[np.full(extra, base + 1), np.full(n_variants - extra, base)]
        return np.repeat(np.arange(n_variants), sizes)
    else:
        return np.arange(B)


def _select_links_offset(offset, links_idx, envs_idx):
    """Index a base-link forward offset for a transposed state query.

    'offset' has shape '[n_links, dim]' for an environment-uniform offset or '[n_envs, n_links, dim]' when
    heterogeneous variants make it environment-specific. It is indexed with the same combined mask as 'qd_to_torch' so
    the result broadcasts against the '[n_envs, n_sel, dim]' state tensor, keeping the link dimension for integer
    indices.
    """
    if offset.ndim == 2:
        return offset[indices_to_mask(links_idx)]
    return offset[indices_to_mask(envs_idx, links_idx)]


def _offset_world_shift(offset_pos, offset_quat, world_quat):
    """World-frame displacement contributed by a body-frame offset position at a given world orientation.

    The authored orientation is 'world_quat' with the offset stripped, and 'offset_pos' rotates with it. Relative
    getters subtract this from the world position to recover the authored position; relative setters add it to do the
    reverse.
    """
    authored_quat = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), world_quat)
    return gu.transform_by_quat(offset_pos, authored_quat)


def _fill_base_link_geom_offsets(offset_pos, offset_quat, entity, geoms, ranges):
    """Fill the per-geom forward offset for an entity's collision or visual geoms.

    Each geom's stored offset is the morph offset (per-variant for a heterogeneous base link, else its own root link's
    offset) conjugated into the geom's own frame, so the body-frame strip in the relative getters reverts it even for
    geoms rotated/translated relative to the link. A geom's world pose is 'link_pose . G' with 'link_pose' carrying
    'link_offset = morph_offset . alignment' (G is the geom pose relative to the link); reverting only the morph offset
    from the geom's own frame needs '(link_offset . G)^-1 . morph_offset . (link_offset . G)'. This equals the morph
    offset itself when the geom is aligned with the link or the offset is a pure translation. 'ranges' are the
    per-variant geom index ranges for a heterogeneous base link, or None to use each geom's own root link offset.
    """
    for geom in geoms:
        if ranges is None:
            # An identity link offset leaves the geom frame in place, so the conjugation is skipped
            link_off_pos = geom.link.desc.offset_pos
            link_off_quat = geom.link.desc.offset_quat
            if np.allclose(link_off_pos, 0.0, atol=gs.EPS) and np.allclose(
                link_off_quat, gu.identity_quat(), atol=gs.EPS
            ):
                continue
            morph = entity._morph
        else:
            # 'ranges' only cover the heterogeneous base link's variant geoms; geoms on other (child) links carry no
            # offset, so skip them.
            i_variant = next((i for i, (start, end) in enumerate(ranges) if start <= geom.idx < end), None)
            if i_variant is None:
                continue
            link_off_pos = entity._desc.variants[i_variant].offset_pos
            link_off_quat = entity._desc.variants[i_variant].offset_quat
            morph = entity._morph if i_variant == 0 else entity._morph_heterogeneous[i_variant - 1]
        frame_pos, frame_quat = gu.transform_pos_quat_by_trans_quat(
            geom.init_pos, geom.init_quat, link_off_pos, link_off_quat
        )
        offset_frame_pos, offset_frame_quat = gu.transform_pos_quat_by_trans_quat(
            frame_pos,
            frame_quat,
            np.array(morph.offset_pos, dtype=gs.np_float),
            np.array(morph.offset_quat, dtype=gs.np_float),
        )
        offset_pos[geom.idx], offset_quat[geom.idx] = gu.inv_transform_pos_quat_by_trans_quat(
            offset_frame_pos, offset_frame_quat, frame_pos, frame_quat
        )


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class KinematicSolver(Solver):
    """
    Base solver for articulated kinematic entities (FK, rendering, state get/set).

    Provides the full build pipeline, field init methods, counter properties,
    render methods, and IO sanitization shared by both KinematicSolver and RigidSolver.

    RigidSolver extends this with physics (collision, constraints, dynamics).
    """

    material_cls = Kinematic
    # The kinds of entity this solver builds, each with the morph class it is built from, the most specific first
    _entity_classes = ((Morph, KinematicEntity),)

    def __init__(self, scene: "Scene", sim: "Simulator", options: "KinematicOptions") -> None:
        super().__init__(scene, sim, options)

        self._enable_collision = False
        self._enable_mujoco_compatibility = False
        self._requires_grad = False
        self._enable_heterogeneous = False  # Set to True when any entity has heterogeneous morphs

        self.collider = None
        self.constraint_solver = None

        self.qpos = None

        self._is_forward_pos_updated: bool = False
        self._is_forward_vel_updated: bool = False

        self._vfaces_raycast_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------------------------
    # ----------------------------------- add_entity -------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name=None, desc=None
    ) -> "KinematicEntity":
        """Create an entity from its description.

        Given no description, one is resolved from the other arguments (see 'KinematicEntityDescription'), the morph
        deciding the kind of entity among those the solver builds ('_entity_classes'). The class of a given description
        says which kind it holds. Creation from a description reads no asset.
        """
        if desc is None:
            morphs = (morph,) if isinstance(morph, Morph) else tuple(morph)
            entity_cls = next(cls for morph_cls, cls in self._entity_classes if isinstance(morphs[0], morph_cls))
            desc = entity_cls._description_cls.resolve(
                morphs, material, surface, self._options, self._enable_mujoco_compatibility, visualize_contact, name
            )
        else:
            for _, entity_cls in self._entity_classes:
                if entity_cls._description_cls is type(desc):
                    break
            else:
                gs.raise_exception(f"{type(self).__name__} simulates no entity described by {type(desc).__name__}.")
        self._enable_heterogeneous |= bool(desc.variants)
        entity = entity_cls(self._scene, self, idx, desc)
        self._entities.append(entity)
        # An attachment names an entity added before this one, which 'attach' requires, so it is restored here
        if desc.attachment is not None:
            entity.attach(self._scene.get_entity(desc.attachment.entity_name), desc.attachment.link_name)
        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------ build -----------------------------------------
    # ------------------------------------------------------------------------------------

    def build(self):
        super().build()

        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        self._para_level = self.sim._para_level

        for entity in self._entities:
            entity._build()

        # Resolve the link-reach structures of @mutates notifications; see Solver.__init__.
        self._articulated_links_idx = np.array([link.idx for link in self.links if not link.is_fixed])
        self._links_parent_idx = np.array([link.parent_idx for link in self.links])

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._n_links = self.n_links
        self._n_joints = self.n_joints
        self._n_vgeoms = self.n_vgeoms
        self._n_vfaces = self.n_vfaces
        self._n_vverts = self.n_vverts
        self._n_custom_vverts = self.n_custom_vverts
        self._n_custom_vfaces = self.n_custom_vfaces
        self._n_entities = self.n_entities

        self._vgeoms = self.vgeoms
        self._links = self.links
        self._joints = self.joints

        base_links_idx = []
        for link in self.links:
            if link.parent_idx == -1 and link.is_fixed:
                base_links_idx.append(link.idx)
        for joint in self.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                base_links_idx.append(joint.link.idx)
        self._base_links_idx = torch.tensor(base_links_idx, dtype=gs.tc_int, device=gs.device)

        # World<-authored pose offset, applied by the relative get/set methods. Links carry the morph offset composed
        # with the base-link inertial alignment (identity for non-base links); geoms and visual geoms carry only the
        # morph offset, since alignment re-expresses them into the aligned frame. Heterogeneous entities vary the
        # base-link offset per environment (one offset per variant), so the link offset is stored per-env when any such
        # entity has divergent variant offsets, and without an environment axis otherwise. The relative getters
        # recompute the inverse on the fly.
        links_offset_per_env = any(
            not all(
                np.allclose(variant.offset_pos, entity._desc.variants[0].offset_pos)
                and np.allclose(variant.offset_quat, entity._desc.variants[0].offset_quat)
                for variant in entity._desc.variants
            )
            for entity in self._entities
        )
        offset_shape = (self._B, self.n_links) if links_offset_per_env else (self.n_links,)
        links_offset_pos = np.zeros((*offset_shape, 3), dtype=gs.np_float)
        links_offset_quat = np.tile(gu.identity_quat(), (*offset_shape, 1))
        vgeoms_offset_pos = np.zeros((self.n_vgeoms, 3), dtype=gs.np_float)
        vgeoms_offset_quat = np.tile(gu.identity_quat(), (self.n_vgeoms, 1))
        for entity in self._entities:
            if links_offset_per_env and entity._desc.variants:
                variant_idx = _balanced_variant_mapping(len(entity._desc.variants), self._B)
                links_offset_pos[np.arange(self._B), entity.base_link_idx] = np.stack(
                    [entity._desc.variants[v].offset_pos for v in variant_idx]
                )
                links_offset_quat[np.arange(self._B), entity.base_link_idx] = np.stack(
                    [entity._desc.variants[v].offset_quat for v in variant_idx]
                )
                vgeoms_ranges = entity.base_link._variant_vgeom_ranges
            else:
                # Each root link carries its own offset; children inherit it through the kinematic chain and stay
                # identity. Visual geoms get the morph offset conjugated into their own frame.
                for local_idx, link in enumerate(entity.links):
                    links_offset_pos[..., entity._link_start + local_idx, :] = link.desc.offset_pos
                    links_offset_quat[..., entity._link_start + local_idx, :] = link.desc.offset_quat
                vgeoms_ranges = None
            _fill_base_link_geom_offsets(vgeoms_offset_pos, vgeoms_offset_quat, entity, entity.vgeoms, vgeoms_ranges)
        # Per-link identity masks gate the relative get/set/backward paths: a relative set on a link whose offset is
        # identity is a plain passthrough, while a non-identity offset rewrites the world position or drops the
        # composition jacobian. Kept as host-side numpy (never used in kernels) so the gates avoid a GPU->host sync.
        self._links_offset_quat_is_identity = np.all(
            np.isclose(gu.quat_to_xyz(links_offset_quat), 0.0, atol=gs.EPS), axis=-1
        )
        self._links_offset_pos_is_identity = np.all(np.isclose(links_offset_pos, 0.0, atol=gs.EPS), axis=-1)
        if links_offset_per_env:
            self._links_offset_quat_is_identity = self._links_offset_quat_is_identity.all(axis=0)
            self._links_offset_pos_is_identity = self._links_offset_pos_is_identity.all(axis=0)
        # The link offset tensors are allocated even when every link is identity, so the getter kernels can take them
        # unconditionally.
        self._links_offset_pos = torch.from_numpy(links_offset_pos).to(device=gs.device, dtype=gs.tc_float)
        self._links_offset_quat = torch.from_numpy(links_offset_quat).to(device=gs.device, dtype=gs.tc_float)
        self._vgeoms_offset_pos = self._vgeoms_offset_quat = None
        if not (
            np.allclose(vgeoms_offset_pos, 0.0, atol=gs.EPS)
            and np.allclose(gu.quat_to_xyz(vgeoms_offset_quat), 0.0, atol=gs.EPS)
        ):
            self._vgeoms_offset_pos = torch.from_numpy(vgeoms_offset_pos).to(device=gs.device, dtype=gs.tc_float)
            self._vgeoms_offset_quat = torch.from_numpy(vgeoms_offset_quat).to(device=gs.device, dtype=gs.tc_float)

        self.n_qs_ = max(1, self.n_qs)
        self.n_dofs_ = max(1, self.n_dofs)
        self.n_links_ = max(1, self.n_links)
        self.n_joints_ = max(1, self.n_joints)
        self.n_vgeoms_ = max(1, self.n_vgeoms)
        self.n_vfaces_ = max(1, self.n_vfaces)
        self.n_vverts_ = max(1, self.n_vverts)
        self.n_custom_vverts_ = max(1, self.n_custom_vverts)
        self.n_custom_vfaces_ = max(1, self.n_custom_vfaces)
        self.n_entities_ = max(1, self.n_entities)

        # The kinematic trees (see trees_root_idx in array_class.py) are the links sharing a root
        self._n_trees = len({link.root_idx for link in self.links})
        self.n_trees_ = max(1, self._n_trees)

        # batch_links_info is required when heterogeneous simulation is used.
        # We must update options because get_links_info reads from solver._options.batch_links_info.
        if self._enable_heterogeneous:
            self._options.batch_links_info = True

        self._build_static_config()
        self._create_data_manager()

        self._init_dof_fields()
        self._init_vvert_fields()
        self._init_vgeom_fields()
        self._init_link_fields()
        self._init_tree_fields()
        self._init_entity_fields()
        self._init_vfaces_raycast_mask()

        self._init_envs_offset()
        self._init_vverts_state()

    def _init_vverts_state(self):
        # Seed vverts_state.pos with one FK pass so opt-in entities show their initial pose from frame 0 even before
        # the user touches the buffer with set_vverts.
        if self.n_custom_vverts == 0:
            return
        opt_in_vgeoms_idx = np.array(
            [vgeom.idx for entity in self._entities if entity._morph.enable_custom_vverts for vgeom in entity.vgeoms],
            dtype=gs.np_int,
        )
        if opt_in_vgeoms_idx.size == 0:
            return
        self.update_vgeoms()
        self.update_vverts_for_vgeoms(opt_in_vgeoms_idx)

    def _build_static_config(self):
        # Static config with all physics disabled
        self.rigid_config = array_class.RigidSimStaticConfig(
            backend=gs.backend,
            para_level=self.sim._para_level,
            requires_grad=False,
            use_hibernation=False,
            batch_links_info=self._options.batch_links_info,
            batch_dofs_info=False,
            batch_joints_info=False,
            enable_mujoco_compatibility=False,
            enable_elliptic_friction=False,
            enable_signorini_contact=False,
            enable_multi_contact=False,
            enable_collision=False,
            enable_joint_limit=False,
            box_box_detection=False,
            sparse_solve=False,
            integrator=gs.integrator.approximate_implicitfast,
            solver_type=0,
        )

    def _create_data_manager(self):
        self.data_manager = array_class.DataManager(self, kinematic_only=True)
        self.rigid_info = self.data_manager.rigid_info
        self._rigid_adjoint_cache = self.data_manager.rigid_adjoint_cache
        self.dyn_info = self.data_manager.dyn_info
        self.dyn_state = self.data_manager.dyn_state
        self.kinematics_scratch = self.data_manager.kinematics_scratch

    @property
    def data(self) -> Iterator[array_class.DataItem]:
        yield from array_class.iter_data(self.rigid_config, "rigid_config")
        yield from array_class.iter_data(self.data_manager.errno, "errno", array_class.DataKind.STATE)
        structs = ["rigid_info", "dyn_info", "dyn_state", "rigid_adjoint_cache"]
        if self.rigid_config.requires_grad:
            structs.append("dyn_state_adjoint_cache")
        for name in structs:
            yield from array_class.iter_data(getattr(self.data_manager, name), name)

    # ------------------------------------------------------------------------------------
    # --------------------------------- hook methods -------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_joint_sol_params(self, sol_params):
        """Hook: sanitize joint constraint solver params. No-op in base (no constraints)."""
        return sol_params

    def _sanitize_geom_sol_params(self, sol_params):
        """Hook: sanitize geom constraint solver params. No-op in base (no constraints)."""
        return sol_params

    # ------------------------------------------------------------------------------------
    # --------------------------------- init methods -------------------------------------
    # ------------------------------------------------------------------------------------

    def _init_dof_fields(self):
        joints = self.joints
        has_dofs = sum(joint.n_dofs for joint in joints) > 0
        if has_dofs:
            kernel_init_dof_fields(
                np.concatenate(
                    [(joint.link._entity_idx_in_solver,) * joint.n_dofs for joint in joints if joint.n_dofs],
                    dtype=gs.np_int,
                ),
                np.concatenate([joint.desc.dofs_motion_ang for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_motion_vel for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_limit for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_invweight for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_stiffness for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_damping for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_frictionloss for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_armature for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_act_gain for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_act_bias for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_force_range for joint in joints], dtype=gs.np_float),
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

        self.dyn_state.dofs.force.fill(0)

    def _init_tree_fields(self):
        """Initialize the fields describing the kinematic trees, which the kernels walk tree by tree.

        The trees carrying a dof come first in ascending dof order, the dof-less ones after them in root order (see
        trees_root_idx in array_class.py). The links come parent first, so the dofs of a tree form one contiguous range
        and the trees are disjoint in dof space, whatever 0-dof links sit inside a span.
        """
        if self._n_trees:
            links_root_idx = np.array([link.root_idx for link in self.links], dtype=gs.np_int)
            links_n_dofs = np.array([link.n_dofs for link in self.links], dtype=gs.np_int)
            links_dof_start = np.array([link.dof_start for link in self.links], dtype=gs.np_int)
            roots_idx, links_root_pos = np.unique(links_root_idx, return_inverse=True)
            trees_n_dofs = np.zeros(self._n_trees, dtype=gs.np_int)
            np.add.at(trees_n_dofs, links_root_pos, links_n_dofs)
            trees_dof_start = np.full(self._n_trees, self.n_dofs, dtype=gs.np_int)
            dof_links = np.flatnonzero(links_n_dofs)
            np.minimum.at(trees_dof_start, links_root_pos[dof_links], links_dof_start[dof_links])
            trees_link_end = np.zeros(self._n_trees, dtype=gs.np_int)
            np.maximum.at(trees_link_end, links_root_pos, np.arange(1, self.n_links + 1, dtype=gs.np_int))
            trees_n_links = np.zeros(self._n_trees, dtype=gs.np_int)
            np.add.at(trees_n_links, links_root_pos, 1)
            trees_order = np.lexsort((np.where(trees_n_dofs > 0, trees_dof_start, roots_idx), trees_n_dofs == 0))
            trees_rank = np.empty(self._n_trees, dtype=gs.np_int)
            trees_rank[trees_order] = np.arange(self._n_trees, dtype=gs.np_int)
            self.rigid_info.trees_root_idx.from_numpy(roots_idx[trees_order])
            self.rigid_info.trees_link_end.from_numpy(trees_link_end[trees_order])
            self.rigid_info.trees_n_links.from_numpy(trees_n_links[trees_order])
            self.rigid_info.trees_dof_start.from_numpy(trees_dof_start[trees_order])
            self.rigid_info.trees_n_dofs.from_numpy(trees_n_dofs[trees_order])
            self.rigid_info.links_tree_idx.from_numpy(trees_rank[links_root_pos])

    def _init_link_fields(self):
        if self.links:
            links = self.links
            kernel_init_link_fields(
                np.array([link.parent_idx for link in links], dtype=gs.np_int),
                np.array([link.root_idx for link in links], dtype=gs.np_int),
                np.array([link.q_start for link in links], dtype=gs.np_int),
                np.array([link.dof_start for link in links], dtype=gs.np_int),
                np.array([link.joint_start for link in links], dtype=gs.np_int),
                np.array([link.q_end for link in links], dtype=gs.np_int),
                np.array([link.dof_end for link in links], dtype=gs.np_int),
                np.array([link.joint_end for link in links], dtype=gs.np_int),
                np.array([link._entity_idx_in_solver for link in links], dtype=gs.np_int),
                np.array([link.geom_start for link in links], dtype=gs.np_int),
                np.array([link.geom_end for link in links], dtype=gs.np_int),
                np.array([link.vgeom_start for link in links], dtype=gs.np_int),
                np.array([link.vgeom_end for link in links], dtype=gs.np_int),
                np.array([link.is_fixed for link in links], dtype=gs.np_bool),
                np.array([link.desc.pos for link in links], dtype=gs.np_float),
                np.array([link.desc.quat for link in links], dtype=gs.np_float),
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

        if self.joints:
            joints = self.joints
            joints_sol_params = np.array([joint.desc.sol_params for joint in joints], dtype=gs.np_float)
            joints_sol_params = self._sanitize_joint_sol_params(joints_sol_params)

            kernel_init_joint_fields(
                np.array([joint.q_start for joint in joints], dtype=gs.np_int),
                np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
                np.array([joint.q_end for joint in joints], dtype=gs.np_int),
                np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
                np.array([joint.type for joint in joints], dtype=gs.np_int),
                joints_sol_params,
                np.array([joint.pos for joint in joints], dtype=gs.np_float),
                self.dyn_info,
                self.rigid_config,
            )

        # Set initial qpos
        self.qpos = self.rigid_info.qpos
        self.qpos0 = self.rigid_info.qpos0
        if self.n_qs > 0:
            init_qpos = np.tile(np.expand_dims(self.init_qpos, -1), (1, self._B))

            # Dispatch per-variant init_qpos for heterogeneous entities
            for entity in self.entities:
                if not entity._desc.variants:
                    continue
                n_variants = len(entity._desc.variants)
                variant_idx = _balanced_variant_mapping(n_variants, self._B)
                q_s, q_e = entity.q_start, entity.q_start + entity.n_qs
                for i_b in range(self._B):
                    init_qpos[q_s:q_e, i_b] = entity._desc.variants[variant_idx[i_b]].init_qpos

            self.qpos0.from_numpy(init_qpos)
            is_init_qpos_out_of_bounds = False
            for joint in self.joints:
                if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                    is_init_qpos_out_of_bounds |= (joint.desc.dofs_limit[0, 0] > init_qpos[joint.q_start]).any()
                    is_init_qpos_out_of_bounds |= (init_qpos[joint.q_start] > joint.desc.dofs_limit[0, 1]).any()
            if is_init_qpos_out_of_bounds:
                gs.logger.warning("Neutral robot position (qpos0) exceeds joint limits.")
            self.qpos.from_numpy(init_qpos)

        self.links_T = self.rigid_info.links_T

        # Dispatch heterogeneous variant vgeom ranges per-environment
        self._dispatch_heterogeneous_vgeoms()

    def _dispatch_heterogeneous_vgeoms(self):
        """Override per-link vgeom ranges for heterogeneous variants. RigidSolver extends this."""
        for link in self.links:
            if link._variant_vgeom_ranges is None:
                continue

            n_variants = len(link._variant_vgeom_ranges)
            variant_idx = _balanced_variant_mapping(n_variants, self._B)

            vgeom_starts = np.array([link._variant_vgeom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
            vgeom_ends = np.array([link._variant_vgeom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)

            kernel_update_heterogeneous_links_vgeom(link.idx, vgeom_starts, vgeom_ends, self.dyn_info)

            for vgeom in link.vgeoms:
                active_envs_mask = (vgeom_starts <= vgeom.idx) & (vgeom.idx < vgeom_ends)
                vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                (vgeom.active_envs_idx,) = np.where(active_envs_mask)

    def _init_vvert_fields(self):
        if self.n_vverts == 0:
            return

        vgeoms = self.vgeoms
        vverts = np.concatenate([vg.init_vverts for vg in vgeoms], dtype=gs.np_float)
        vnormals = np.concatenate([vg.init_vnormals for vg in vgeoms], dtype=gs.np_float)
        vfaces = np.concatenate([vg.init_vfaces + vg._vvert_start for vg in vgeoms], dtype=gs.np_int)
        vverts_vgeom_idx = np.concatenate([np.full(vg.n_vverts, vg.idx) for vg in vgeoms], dtype=gs.np_int)
        vverts_state_idx = np.full(self.n_vverts, -1, dtype=gs.np_int)
        for entity in self._entities:
            if not entity._morph.enable_custom_vverts:
                continue
            entity_custom_offset = entity._custom_vvert_start - entity._vvert_start
            for vgeom in entity.vgeoms:
                local = np.arange(vgeom.n_vverts, dtype=gs.np_int)
                vverts_state_idx[vgeom._vvert_start + local] = vgeom._vvert_start + entity_custom_offset + local
        kernel_init_vvert_fields(
            vverts_vgeom_idx, vverts_state_idx, vverts, vfaces, vnormals, self.dyn_info, self.rigid_config
        )

    def _init_vgeom_fields(self):
        if self.n_vgeoms > 0:
            vgeoms = self.vgeoms
            kernel_init_vgeom_fields(
                np.array([vgeom.link.idx for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.vface_start for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.vvert_end for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.vface_end for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.init_pos for vgeom in vgeoms], dtype=gs.np_float),
                np.array([vgeom.init_quat for vgeom in vgeoms], dtype=gs.np_float),
                np.array([vgeom.desc.vmesh.color for vgeom in vgeoms], dtype=gs.np_float),
                self.dyn_info,
                self.rigid_config,
            )

    def _init_vfaces_raycast_mask(self):
        """Fit the static per-vface visual-raycasting opt-in mask; see the vfaces_raycast_mask property.

        Sized over the padded vgeom count so the gather also covers a solver holding no vgeom at all, whose vface
        fields still carry one padding slot: nothing opts in there, so every entry stays 0.
        """
        vgeom_enabled = torch.zeros(self.n_vgeoms_, dtype=gs.tc_bool, device=gs.device)
        for entity in self.entities:
            if not entity.material.use_visual_raycasting:
                continue
            for vgeom in entity.vgeoms:
                vgeom_enabled[vgeom.idx] = 1
        self._vfaces_raycast_mask = vgeom_enabled[qd_to_torch(self.dyn_info.vfaces.vgeom_idx)]

    def _init_entity_fields(self):
        if self._entities:
            entities = self._entities
            kernel_init_entity_fields(
                np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
                np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
                np.array([entity.link_start for entity in entities], dtype=gs.np_int),
                np.array([entity.link_end for entity in entities], dtype=gs.np_int),
                np.array([0 for entity in entities], dtype=gs.np_int),
                np.array([0 for entity in entities], dtype=gs.np_int),
                np.array([0.0 for entity in entities], dtype=gs.np_float),
                np.array([False for entity in entities], dtype=gs.np_bool),
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

    def _init_envs_offset(self):
        self.envs_offset = self.rigid_info.envs_offset
        self.envs_offset.from_numpy(self._scene.envs_offset)

    # ------------------------------------------------------------------------------------
    # -------------------------------- simulation no-ops ----------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        pass

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if not self._is_forward_pos_updated or not self._is_forward_vel_updated:
            kernel_forward_kinematics(
                self.scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True

    def substep_post_coupling_grad(self, f):
        pass

    def add_grad_from_state(self, state):
        if self.is_active:
            qpos_grad = gs.zeros_like(state.qpos)
            dofs_vel_grad = gs.zeros_like(state.dofs_vel)
            links_pos_grad = gs.zeros_like(state.links_pos)
            links_quat_grad = gs.zeros_like(state.links_quat)

            if state.qpos.grad is not None:
                qpos_grad = state.qpos.grad
            if state.dofs_vel.grad is not None:
                dofs_vel_grad = state.dofs_vel.grad
            if state.links_pos.grad is not None:
                links_pos_grad = state.links_pos.grad
            if state.links_quat.grad is not None:
                links_quat_grad = state.links_quat.grad

            kernel_get_state_grad(
                qpos_grad,
                dofs_vel_grad,
                links_pos_grad,
                links_quat_grad,
                self.dyn_state,
                self.rigid_info,
                self.rigid_config,
            )

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            assert len(self._queried_states[self._sim.cur_step_global]) == 1
            state = self._queried_states[self._sim.cur_step_global][0]
            self.add_grad_from_state(state)

    def reset_grad(self):
        # Zero every Quadrants-side adjoint buffer the solver owns so a subsequent backward pass does not accumulate on
        # top of stale residues from the previous one. The rigid/kinematic `.grad` pipeline writes through
        # `qd.atomic_add` (e.g. `kernel_get_state_grad`, the `substep_pre_coupling_grad` chain), which is the correct
        # reverse-mode accumulation rule for a single backward pass but requires the buffers to start at zero. MPM's
        # `reset_grad` already zeros its fields explicitly; the rigid/kinematic solvers historically relied on users
        # rebuilding the scene per `loss.backward()`, which breaks training loops that call `scene.reset()` between
        # backward passes.
        if self._requires_grad:
            qd_zero_grad(self.dyn_state.links)
            qd_zero_grad(self.dyn_state.dofs)
            qd_zero_grad(self.dyn_state.joints)
            qd_zero_grad(self.rigid_info)
            # One flush for the zeroing batch; see qd_zero_grad in misc.py.
            if gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()
        for entity in self._entities:
            entity.reset_grad()
        self._queried_states.clear()

    # ------------------------------------------------------------------------------------
    # -------------------------------- state get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f=None):
        if self.is_active:
            s_global = self.sim.cur_step_global
            if s_global in self._queried_states:
                return self._queried_states[s_global][0]

            state = KinematicSolverState(self._scene, s_global)

            kernel_get_kinematic_state(
                state.qpos,
                state.dofs_vel,
                state.links_pos,
                state.links_quat,
                self.dyn_state,
                self.rigid_info,
                self.rigid_config,
            )
            self._queried_states.append(state)
        else:
            state = None
        return state

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None, *, partial: bool = False) -> None:
        if not self.is_active:
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        kernel_set_kinematic_state(
            envs_idx,
            state.qpos,
            state.dofs_vel,
            state.links_pos,
            state.links_quat,
            self.dyn_state,
            self.rigid_info,
            self.rigid_config,
        )
        if not partial:
            kernel_forward_kinematics(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def __getstate__(self) -> KinematicSolverCheckpoint:
        state = super().__getstate__()
        return KinematicSolverCheckpoint(
            arrays=state.arrays,
            configs=state.configs,
            kinds=state.kinds,
            is_forward_pos_updated=self._is_forward_pos_updated,
            is_forward_vel_updated=self._is_forward_vel_updated,
        )

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def __setstate__(self, state: KinematicSolverCheckpoint) -> None:
        super().__setstate__(state)
        if array_class.DataKind.DERIVED in state.kinds:
            self._is_forward_pos_updated = state.is_forward_pos_updated
            self._is_forward_vel_updated = state.is_forward_vel_updated
        else:
            # Without the derived arrays, forward kinematics runs now so the scene is drawn where the record put it. The
            # next step recomputes the rest of the derived state.
            self._is_forward_pos_updated = self._is_forward_vel_updated = False
            self.update_forward_pos()

    # ------------------------------------------------------------------------------------
    # -------------------------------- process_input -------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        """Process input for entities (set qpos from user commands)."""
        for entity in self._entities:
            entity.process_input()

    def process_input_grad(self):
        """No-op: kinematic solver does not support gradients."""
        pass

    def save_ckpt(self, ckpt_name):
        """No-op: kinematic solver does not save checkpoints."""
        pass

    def load_ckpt(self, ckpt_name):
        """No-op: kinematic solver does not load checkpoints."""
        pass

    @property
    def is_active(self):
        return self.n_links > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_io_variables(
        self,
        tensor,
        inputs_idx,
        input_size,
        idx_name,
        envs_idx=None,
        element_shape=(),
        *,
        batched=True,
        skip_allocation=False,
    ):
        envs_idx_ = self._scene._sanitize_envs_idx(envs_idx) if batched else self._scene._envs_idx[:0]

        if self.n_envs == 0 or not batched:
            tensor_, (inputs_idx_,) = sanitize_indexed_tensor(
                tensor,
                gs.tc_float,
                (inputs_idx,),
                (-1, *element_shape),
                (input_size, *element_shape),
                (idx_name, *("" for _ in element_shape)),
                skip_allocation=skip_allocation,
            )
        else:
            tensor_, (envs_idx_, inputs_idx_) = sanitize_indexed_tensor(
                tensor,
                gs.tc_float,
                (envs_idx_, inputs_idx),
                (-1, -1, *element_shape),
                (self.n_envs, input_size, *element_shape),
                ("envs_idx", idx_name, *("" for _ in element_shape)),
                skip_allocation=skip_allocation,
            )

        return tensor_, inputs_idx_, envs_idx_

    @mutates(StateChange.GEOMETRY, links="links_idx")
    def set_base_links_pos(self, pos, links_idx=None, envs_idx=None, *, relative=False, skip_forward=False):
        if links_idx is None:
            links_idx = self._base_links_idx
        # Without any offset position on the targeted links, the authored and internal link origins coincide, so a
        # relative set is just an absolute one.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and self._links_offset_pos_is_identity[idx].all():
            relative = False
        pos, links_idx, envs_idx = self._sanitize_io_variables(
            pos, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            pos = pos[None]

        if relative:
            # Compose the body-frame offset onto the authored position while keeping the current orientation, then set
            # the resulting world position.
            cur_quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            pos = pos + _offset_world_shift(offset_pos, offset_quat, cur_quat)
            relative = False

        kernel_set_links_pos(
            links_idx, envs_idx, pos, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

        if not skip_forward:
            kernel_forward_kinematics(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def set_base_links_pos_grad(self, links_idx, envs_idx, relative, pos_grad):
        if links_idx is None:
            links_idx = self._base_links_idx
        # A relative 'set_pos' adds 'R(authored_quat) @ offset_pos', which reads the current orientation; that
        # dependency is not propagated, so with a non-zero offset position the backward pass is only supported on links
        # without one. 'relative=False' sets the world position directly and is a plain passthrough.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not self._links_offset_pos_is_identity[idx].all():
            gs.raise_exception(
                "Backward pass for 'set_pos' with 'relative=True' is only supported on links without an offset "
                "position (no inertial alignment shifting the link origin). Use 'relative=False' to set the world "
                "position."
            )
        pos_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            pos_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            pos_grad_ = pos_grad_.unsqueeze(0)
        kernel_set_links_pos_grad(
            links_idx, envs_idx, pos_grad_, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

    @mutates(StateChange.GEOMETRY, links="links_idx")
    def set_base_links_quat(self, quat, links_idx=None, envs_idx=None, *, relative=False, skip_forward=False):
        if links_idx is None:
            links_idx = self._base_links_idx
        # Without any pose offset on the targeted links, the authored and internal link origins coincide, so a relative
        # set is just an absolute one.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and (
            self._links_offset_quat_is_identity[idx].all() and self._links_offset_pos_is_identity[idx].all()
        ):
            relative = False
        relative_pos_passthrough = relative and self._links_offset_pos_is_identity[idx].all()
        quat, links_idx, envs_idx = self._sanitize_io_variables(
            quat, links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
        )
        if self.n_envs == 0:
            quat = quat[None]

        if relative:
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            if not relative_pos_passthrough:
                # The offset position rotates with the orientation, so keep the authored-frame position fixed by
                # rewriting the world position from the current authored position and the new authored orientation.
                cur_pos = qd_to_torch(self.dyn_state.links.pos, envs_idx, links_idx, transpose=True, copy=True)
                cur_quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
                offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                authored_pos = cur_pos - _offset_world_shift(offset_pos, offset_quat, cur_quat)
                world_pos = authored_pos + gu.transform_by_quat(offset_pos, quat)
                kernel_set_links_pos(
                    links_idx, envs_idx, world_pos, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
                )
            # Compose the offset onto the authored orientation, then set the resulting world orientation.
            quat = gu.transform_quat_by_quat(offset_quat, quat)
            relative = False

        kernel_set_links_quat(
            links_idx, envs_idx, quat, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

        if not skip_forward:
            kernel_forward_kinematics(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def set_base_links_quat_grad(self, links_idx, envs_idx, relative, quat_grad):
        if links_idx is None:
            links_idx = self._base_links_idx
        # A relative 'set_quat' composes the orientation offset (a constant right-multiplication) and, when the offset
        # position is non-zero, also rewrites the world position from the input orientation. Neither jacobian is
        # propagated, so the backward pass is only supported when the targeted links carry no pose offset. With
        # 'relative=False' the world orientation is set directly, which is a plain passthrough and always differentiable.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not (
            self._links_offset_quat_is_identity[idx].all() and self._links_offset_pos_is_identity[idx].all()
        ):
            gs.raise_exception(
                "Backward pass for 'set_quat' with 'relative=True' is only supported on links without a pose offset "
                "(no up-axis conversion or inertial alignment). Use 'relative=False' to set the world orientation."
            )
        quat_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            quat_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
        )
        if self.n_envs == 0:
            quat_grad_ = quat_grad_.unsqueeze(0)
        kernel_set_links_quat_grad(
            links_idx, envs_idx, quat_grad_, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

    @mutates(StateChange.GEOMETRY, links=MutatedLinks.ARTICULATED)
    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            data = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
            qs_mask = indices_to_mask(qs_idx)
            if (
                (not qs_mask or isinstance(qs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and envs_idx.dtype == torch.bool
            ):
                qs_data = data[(slice(None), *qs_mask)]
                if qpos.ndim == 2 and len(qpos) != len(qs_data):
                    # Note that it is necessary to create a new temporary view because it will be reshaped in-place
                    qs_data.masked_scatter_(envs_idx[:, None], qpos.view_as(qpos))
                else:
                    qpos = broadcast_tensor(qpos, gs.tc_float, qs_data.shape)
                    torch.where(envs_idx[:, None], qpos, qs_data, out=qs_data)
            else:
                mask = (0, *qs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *qs_mask, boolean_mask=False)
                assign_indexed_tensor(data, mask, qpos)
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            qpos, qs_idx, envs_idx = self._sanitize_io_variables(
                qpos, qs_idx, self.n_qs, "qs_idx", envs_idx, skip_allocation=True
            )
            if self.n_envs == 0:
                qpos = qpos[None]
            kernel_set_qpos(qs_idx, envs_idx, qpos, self.rigid_info, self.rigid_config)

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics
            else:
                fn = kernel_forward_kinematics
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    @mutates(StateChange.DYNAMICS, links=MutatedLinks.ARTICULATED)
    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            vel = qd_to_torch(self.dyn_state.dofs.vel, transpose=True, copy=False)
            dofs_mask = indices_to_mask(dofs_idx)
            if (
                (not dofs_mask or isinstance(dofs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and (
                    (velocity is None and (not IS_OLD_TORCH or envs_idx.dtype == torch.bool))
                    or (velocity is not None and velocity.ndim == 2 and envs_idx.dtype == torch.bool)
                )
            ):
                dofs_vel = vel[(slice(None), *dofs_mask)]
                if velocity is None:
                    if envs_idx.dtype == torch.bool:
                        dofs_vel.masked_fill_(envs_idx[:, None], 0.0)
                    else:
                        dofs_vel.scatter_(0, envs_idx[:, None].expand((-1, dofs_vel.shape[1])), 0.0)
                else:
                    if velocity.ndim == 2 and len(dofs_vel) != len(velocity):
                        # Note that it is necessary to create a new temporary view because it will be reshaped in-place
                        dofs_vel.masked_scatter_(envs_idx[:, None], velocity.view_as(velocity))
                    else:
                        velocity = broadcast_tensor(velocity, gs.tc_float, dofs_vel.shape)
                        torch.where(envs_idx[:, None], velocity, dofs_vel, out=dofs_vel)
            else:
                mask = (
                    (0, *dofs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *dofs_mask, boolean_mask=False)
                )
                if velocity is None:
                    vel[mask] = 0.0
                else:
                    assign_indexed_tensor(vel, mask, velocity)
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            if not skip_forward and not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        else:
            velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
                velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            if velocity is None:
                kernel_set_dofs_zero_velocity(dofs_idx, envs_idx, self.dyn_state, self.rigid_config)
            else:
                if self.n_envs == 0:
                    velocity = velocity[None]
                kernel_set_dofs_velocity(dofs_idx, envs_idx, velocity, self.dyn_state, self.rigid_config)

        if not skip_forward:
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_velocity
            else:
                fn = kernel_forward_velocity
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=False)
            self._is_forward_vel_updated = True
        else:
            self._is_forward_vel_updated = False

    def set_dofs_velocity_grad(self, dofs_idx, envs_idx, velocity_grad):
        velocity_grad_, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity_grad_ = velocity_grad_.unsqueeze(0)
        kernel_set_dofs_velocity_grad(dofs_idx, envs_idx, velocity_grad_, self.dyn_state, self.rigid_config)

    def set_dofs_force_grad(self, dofs_idx, envs_idx, force_grad):
        force_grad_, dofs_idx, envs_idx = self._sanitize_io_variables(
            force_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            force_grad_ = force_grad_.unsqueeze(0)
        kernel_set_dofs_force_grad(dofs_idx, envs_idx, force_grad_, self.dyn_state, self.rigid_config)

    @mutates(StateChange.GEOMETRY, links=MutatedLinks.ARTICULATED)
    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]
        kernel_set_dofs_position(
            dofs_idx, envs_idx, position, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

        kernel_forward_kinematics(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def get_terrain_height(self, positions, link_idx, envs_idx=None):
        terrain = self._links[link_idx].entity

        positions = torch.as_tensor(positions, dtype=gs.tc_float, device=gs.device)
        if positions.ndim == 0 or positions.ndim > 3 or positions.shape[-1] != 2:
            gs.raise_exception("`positions` must have shape (2,), (n_points, 2), or (n_envs, n_points, 2).")

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        n_envs = len(envs_idx)
        is_single_position = positions.ndim == 1
        n_points = 1 if is_single_position else positions.shape[-2]
        if n_points == 0:
            gs.raise_exception("`positions` must contain at least one position.")

        is_per_env = positions.ndim == 3 and positions.shape[0] != 1
        positions = broadcast_tensor(
            positions, gs.tc_float, (n_envs if is_per_env else 1, n_points, 2), ("envs_idx", "positions", "")
        )

        heights = torch.empty((n_envs, n_points), dtype=gs.tc_float, device=gs.device)
        kernel_get_terrain_height(
            envs_idx,
            link_idx,
            terrain._morph.horizontal_scale,
            positions.contiguous(),
            terrain._terrain_height_field,
            heights,
            self.dyn_state,
            self.rigid_config,
            tilt_tolerance=TERRAIN_HEIGHT_QUERY_TILT_TOLERANCE,
            is_per_env=is_per_env,
        )

        if self.n_envs == 0:
            heights = heights[0]
        if is_single_position:
            heights = heights[..., 0]
        return heights

    def get_links_pos(self, links_idx=None, envs_idx=None, *, relative=False):
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = self._sanitize_io_variables(
                None, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
        tensor = qd_to_torch(self.dyn_state.links.pos, envs_idx, links_idx, transpose=True, copy=True)
        if relative and not self._links_offset_pos_is_identity[idx].all():
            quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_quat(self, links_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not self._links_offset_quat_is_identity[idx].all():
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_joints_anchor_pos(self, joints_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.joints.xanchor, envs_idx, joints_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_joints_anchor_axis(self, joints_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.joints.xaxis, envs_idx, joints_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_vgeoms_pos(self, vgeoms_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.dyn_state.vgeoms.pos, envs_idx, vgeoms_idx, transpose=True, copy=True)
        if relative and self._vgeoms_offset_pos is not None:
            quat = qd_to_torch(self.dyn_state.vgeoms.quat, envs_idx, vgeoms_idx, transpose=True, copy=True)
            offset_pos = self._vgeoms_offset_pos if vgeoms_idx is None else self._vgeoms_offset_pos[vgeoms_idx]
            offset_quat = self._vgeoms_offset_quat if vgeoms_idx is None else self._vgeoms_offset_quat[vgeoms_idx]
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_vgeoms_quat(self, vgeoms_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.dyn_state.vgeoms.quat, envs_idx, vgeoms_idx, transpose=True, copy=True)
        if relative and self._vgeoms_offset_quat is not None:
            offset_quat = self._vgeoms_offset_quat if vgeoms_idx is None else self._vgeoms_offset_quat[vgeoms_idx]
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_vel(self, links_idx=None, envs_idx=None, *, relative=False):
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        is_relative = relative and not self._links_offset_pos_is_identity[idx].all()
        if gs.use_zerocopy:
            mask = indices_to_mask(envs_idx, links_idx)
            cd_vel = qd_to_torch(self.dyn_state.links.cd_vel, transpose=True)
            cd_ang = qd_to_torch(self.dyn_state.links.cd_ang, transpose=True)
            pos = qd_to_torch(self.dyn_state.links.pos, transpose=True)
            root_COM = qd_to_torch(self.dyn_state.links.root_COM, transpose=True)
            cpos = pos[mask] - root_COM[mask]
            if is_relative:
                quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
                offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                cpos = cpos - _offset_world_shift(offset_pos, offset_quat, quat)
            tensor = cd_vel[mask] + cd_ang[mask].cross(cpos, dim=-1)
            return tensor[0] if self.n_envs == 0 else tensor

        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        # The frame is passed as a plain int, see 'RigidSolver._sanitize_ref_frame'.
        kernel_get_links_vel(
            links_idx,
            envs_idx,
            tensor,
            self._links_offset_pos,
            self._links_offset_quat,
            self.dyn_state,
            self.rigid_config,
            ref=int(gs.link_ref_frame.link_origin),
            is_relative=is_relative,
        )
        return _tensor

    def get_links_ang(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.links.cd_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def _build_dof_to_q_map(self, dofs_idx_t):
        """Build a mapping from DOF indices to qpos indices for revolute/prismatic joints."""
        dof_to_q = torch.zeros(self.n_dofs, dtype=torch.long, device=gs.device)
        for entity in self._entities:
            for joint in entity.joints:
                if joint.n_dofs == 0:
                    continue
                for i in range(joint.n_dofs):
                    dof_to_q[joint.dof_start - entity.dof_start + i] = joint.q_start - entity.q_start + i
        return dof_to_q[dofs_idx_t]

    def get_qpos(self, qs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.qpos, envs_idx, qs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_velocity(self, dofs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.dofs.vel, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_position(self, dofs_idx=None, envs_idx=None):
        """Read current DOF positions."""
        tensor = qd_to_torch(self.dyn_state.dofs.pos, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_limit(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.limit, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def update_vgeoms(self):
        kernel_update_vgeoms(self.dyn_state, self.dyn_info, self.rigid_config)

    def update_forward_pos(self):
        """Run forward kinematics if links_state is not already up to date for the current pose."""
        if self._is_forward_pos_updated:
            return
        kernel_forward_kinematics(
            self.scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )
        self._is_forward_pos_updated = True

    def update_vverts_for_vgeoms(self, vgeoms_idx):
        """Refresh the vverts_state.pos slice for the requested vgeoms by re-running FK.

        Used by set_vverts(None, ...) and at scene-build time to initialize the custom buffer with a meaningful pose.
        The kernel is a no-op for vgeoms whose vverts have no state slot.
        """
        if self.n_custom_vverts == 0:
            return
        kernel_update_vverts_for_vgeoms(vgeoms_idx, self.dyn_state, self.dyn_info, self.rigid_config)

    @mutates(StateChange.GEOMETRY)
    def set_vverts(self, custom_vvert_start, custom_vvert_end, vgeoms_idx, vverts, envs_idx=None):
        """Write the slice [custom_vvert_start:custom_vvert_end] of vverts_state.pos.

        vverts=None re-populates the slice from FK by running update_vverts_for_vgeoms over the vgeoms owning the slice.
        Otherwise vverts is broadcast to the slice shape and written directly. vverts_state.pos is always batched;
        envs_idx selects which envs to write to.
        """
        if vverts is None:
            self.update_vverts_for_vgeoms(vgeoms_idx)
            return

        if gs.use_zerocopy:
            data = qd_to_torch(self.dyn_state.vverts.pos, transpose=True, copy=False)
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                pos_slice = data[:, custom_vvert_start:custom_vvert_end]
                if vverts.ndim == 3 and len(vverts) != len(pos_slice):
                    pos_slice.masked_scatter_(envs_idx[:, None, None], vverts.view_as(vverts))
                else:
                    vverts_b = broadcast_tensor(vverts, gs.tc_float, pos_slice.shape)
                    torch.where(envs_idx[:, None, None], vverts_b, pos_slice, out=pos_slice)
            else:
                vverts_mask = indices_to_mask(slice(custom_vvert_start, custom_vvert_end))
                pos_mask = (0, *vverts_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *vverts_mask)
                assign_indexed_tensor(data, pos_mask, vverts)
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        target_shape = (envs_idx.shape[0], custom_vvert_end - custom_vvert_start, 3)
        vverts = broadcast_tensor(vverts, gs.tc_float, target_shape, ("envs", "vverts", "xyz")).contiguous()
        kernel_set_vverts(custom_vvert_start, envs_idx, vverts, self.dyn_state, self.rigid_config)

    def get_vverts(self, custom_vvert_start, custom_vvert_end, envs_idx=None):
        """Return a copy of the vverts_state.pos slice for the given custom-vvert range.

        Shape: (len(envs_idx), custom_vvert_end - custom_vvert_start, 3). envs_idx=None returns every env.
        """
        tensor = qd_to_torch(
            self.dyn_state.vverts.pos, envs_idx, slice(custom_vvert_start, custom_vvert_end), transpose=True, copy=True
        )
        return tensor[0] if self.n_envs == 0 else tensor

    # ------------------------------------------------------------------------------------
    # --------------------------------- Jacobian & IK ------------------------------------
    # ------------------------------------------------------------------------------------

    def get_links_jacobian(self, link_idx, dof_start, n_dofs, local_point=None):
        """Spatial Jacobian of a link-local point, as (n_envs, 6, n_dofs), for the entity owning that link.

        dof_start offsets the entity degrees of freedom into the solver and n_dofs narrows the shared buffer down to
        that entity. local_point defaults to the link origin.
        """
        jacobian = self.kinematics_scratch.jacobian
        if local_point is None:
            kernel_get_jacobian_zero(
                link_idx, dof_start, jacobian, self.dyn_state, self.dyn_info, self.rigid_config, self._B
            )
        else:
            kernel_get_jacobian(
                link_idx, dof_start, local_point, jacobian, self.dyn_state, self.dyn_info, self.rigid_config, self._B
            )
        return qd_to_torch(jacobian, transpose=True, copy=True)[..., :n_dofs]

    def forward_kinematics_query(self, entity, qpos, envs_idx=None):
        """Link poses one entity would have at the given configuration, leaving the live configuration as it was.

        Distinct from the forward kinematics the step propagates: this evaluates a configuration the solver does not
        hold, and restores the one it does. Returns positions (n_envs, n_links, 3) and orientations
        (n_envs, n_links, 4), without the batch dimension when the scene is not batched.
        """
        if self.n_envs == 0:
            envs_idx = torch.zeros(1, dtype=gs.tc_int)
        else:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        qpos = broadcast_tensor(qpos, gs.tc_float, (len(envs_idx), entity.n_qs), ("envs_idx", "qs_idx")).contiguous()

        qs_idx = torch.arange(entity._q_start, entity._q_start + entity.n_qs, dtype=gs.tc_int, device=gs.device)
        links_idx = torch.arange(
            entity._link_start, entity._link_start + entity.n_links, dtype=gs.tc_int, device=gs.device
        )
        links_pos = torch.empty((len(envs_idx), entity.n_links, 3), dtype=gs.tc_float, device=gs.device)
        links_quat = torch.empty((len(envs_idx), entity.n_links, 4), dtype=gs.tc_float, device=gs.device)
        kernel_forward_kinematics_query(
            entity._idx_in_solver,
            qs_idx,
            links_idx,
            envs_idx,
            links_pos,
            links_quat,
            qpos,
            self.kinematics_scratch.qpos_cache,
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )

        if self.n_envs == 0:
            links_pos = links_pos[0]
            links_quat = links_quat[0]
        return links_pos, links_quat

    def inverse_kinematics(
        self,
        entity_idx,
        q_start,
        link_start,
        joint_start,
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
        n_qs,
        n_dofs,
        custom_init_qpos,
        max_samples,
        max_solver_iters,
        damping,
        pos_tol,
        rot_tol,
        max_step_size,
        seed,
        respect_joint_limit,
    ):
        """Damped-least-squares inverse kinematics of one entity for the given link targets.

        Returns the best configuration per environment, spanning the entity, and the stacked pose residual of every
        target it carries. Callers narrow both to the environments and targets they asked for.
        """
        n_links = len(links_idx)
        n_tgts = self._options.IK_max_targets
        if n_links > n_tgts:
            gs.raise_exception(f"Cannot solve for {n_links} targets. Raise 'IK_max_targets' above {n_tgts}.")

        ik_state, ik_fk, targets = self.data_manager.ik_scratch

        # Only the leading rows each buffer holds for this solve are written, and the kernel reads no further.
        if gs.use_zerocopy:
            for member, data in (
                (targets.links_idx, links_idx),
                (targets.local_point, local_points),
                (targets.link_pos_mask, link_pos_mask),
                (targets.link_rot_mask, link_rot_mask),
            ):
                member_t = qd_to_torch(member, copy=False)
                member_t[:n_links] = data
            dofs_idx_t = qd_to_torch(targets.dofs_idx, copy=False)
            dofs_idx_t[: len(dofs_idx)] = dofs_idx
            envs_idx_t = qd_to_torch(targets.envs_idx, copy=False)
            envs_idx_t[: len(envs_idx)] = envs_idx
            pos_mask_t = qd_to_torch(targets.pos_mask, copy=False)
            pos_mask_t[:] = pos_mask
            rot_mask_t = qd_to_torch(targets.rot_mask, copy=False)
            rot_mask_t[:] = rot_mask
            for member, data in ((targets.pos, poss), (targets.quat, quats)):
                member_t = qd_to_torch(member, copy=False)
                member_t[:n_links, envs_idx] = data
            init_qpos_t = qd_to_torch(targets.init_qpos, copy=False)
            init_qpos_t[envs_idx, :n_qs] = init_qpos
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_set_ik_targets(
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
                targets,
            )

        kernel_inverse_kinematics_entity(
            entity_idx,
            q_start,
            link_start,
            joint_start,
            self.dyn_state,
            ik_state,
            ik_fk,
            targets,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            n_qs,
            n_dofs,
            n_links,
            len(dofs_idx),
            len(envs_idx),
            custom_init_qpos,
            max_samples,
            max_solver_iters,
            damping,
            pos_tol,
            rot_tol,
            max_step_size,
            seed,
            respect_joint_limit,
        )

        qpos = qd_to_torch(ik_state.qpos_best, transpose=True, copy=True)[..., :n_qs]
        err_pose = qd_to_torch(ik_state.err_pose_best, transpose=True, copy=True).reshape((-1, n_tgts, 6))
        return qpos, err_pose[:, :n_links]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def links(self):
        if self.is_built:
            return self._links
        return gs.List(link for entity in self._entities for link in entity.links)

    @property
    def joints(self):
        if self.is_built:
            return self._joints
        return gs.List(joint for entity in self._entities for joint in entity.joints)

    @property
    def equalities(self):
        """The equality constraints the solver enforces, which a kinematic solver holds none of."""
        return gs.List()

    @property
    def geoms(self):
        if self.is_built:
            return self._geoms
        return gs.List(geom for entity in self._entities for geom in entity.geoms)

    @property
    def vgeoms(self):
        if self.is_built:
            return self._vgeoms
        return gs.List(vgeom for entity in self._entities for vgeom in entity.vgeoms)

    @property
    def vfaces_raycast_mask(self) -> torch.Tensor:
        """Per-vface mask, shape (n_vfaces,), selecting the vfaces opted into visual raycasting.

        A vface is opted in iff its owning vgeom belongs to an entity whose material has use_visual_raycasting=True.
        Both the entity materials and the vface-to-vgeom mapping are fixed once the scene is built, so the mask is
        fitted at build time and shared by every visual raycast BVH (raycaster sensors, viewer plugins) to gate which
        vfaces contribute.
        """
        return self._vfaces_raycast_mask

    @property
    def n_links(self):
        if self.is_built:
            return self._n_links
        return len(self.links)

    @property
    def n_joints(self):
        if self.is_built:
            return self._n_joints
        return len(self.joints)

    @property
    def n_vgeoms(self):
        if self.is_built:
            return self._n_vgeoms
        return len(self.vgeoms)

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def n_custom_vverts(self):
        if self.is_built:
            return self._n_custom_vverts
        return sum(entity.n_vverts for entity in self._entities if entity._morph.enable_custom_vverts)

    @property
    def n_custom_vfaces(self):
        if self.is_built:
            return self._n_custom_vfaces
        return sum(entity.n_vfaces for entity in self._entities if entity._morph.enable_custom_vverts)

    @property
    def n_qs(self):
        if self.is_built:
            return self._n_qs
        return sum(entity.n_qs for entity in self._entities)

    @property
    def is_forward_pos_updated(self) -> bool:
        """Whether the link and geom poses are current for the configuration, so the next step skips forward kinematics."""
        return self._is_forward_pos_updated

    @property
    def is_forward_vel_updated(self) -> bool:
        """Whether the link velocities are current for the generalized velocities."""
        return self._is_forward_vel_updated

    @property
    def n_dofs(self):
        if self.is_built:
            return self._n_dofs
        return sum(entity.n_dofs for entity in self._entities)

    @property
    def init_qpos(self):
        if self._entities:
            return np.concatenate([entity.init_qpos for entity in self._entities], dtype=gs.np_float)
        return np.array([], dtype=gs.np_float)
