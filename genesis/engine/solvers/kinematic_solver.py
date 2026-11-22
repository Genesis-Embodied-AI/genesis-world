from collections.abc import Iterator, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity import DroneEntity, KinematicEntity, RigidGeom, RigidJoint, RigidVisGeom
from genesis.engine.entities.rigid_entity.description import KinematicEntityDescription, _balanced_variant_mapping
from genesis.engine.entities.rigid_entity.rigid_entity import get_variants_kinematics
from genesis.engine.entities.rigid_entity.rigid_equality import RigidEquality
from genesis.engine.entities.rigid_entity.rigid_link import KinematicLink
from genesis.engine.render_context import (
    ArrayRef,
    DynamicMesh,
    EntityAsset,
    Floor,
    Frames,
    MeshAsset,
    MeshGeometry,
    MeshSource,
    PrimitiveAsset,
    ScopedRenderContext,
    Shape,
    ShapeRoles,
    StaticMeshBatch,
    TexturePaths,
    select_view_shapes,
)
from genesis.engine.solvers.rigid.abd.inverse_kinematics import (
    IKScratch,
    kernel_forward_kinematics_query,
    kernel_get_jacobian,
    kernel_get_jacobian_zero,
    kernel_inverse_kinematics_entity,
    kernel_set_ik_targets,
)
from genesis.engine.solvers.rigid.data import RigidConfig, RigidData
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import SolverCheckpoint
from genesis.options.morphs import Morph
from genesis.options.solvers import KinematicOptions
from genesis.options.surfaces import Surface
from genesis.utils.data import ConfigItem, DataItem, DataKind, iter_data
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    tensor_to_array,
)

from .base_solver import Solver
from .rigid.abd.accessor import (
    kernel_get_links_vel,
    kernel_get_terrain_height,
    kernel_set_dofs_position,
    kernel_set_dofs_velocity,
    kernel_set_dofs_zero_velocity,
    kernel_set_links_pos,
    kernel_set_links_quat,
    kernel_set_qpos,
)
from .rigid.abd.forward_kinematics import (
    kernel_forward_kinematics,
    kernel_forward_velocity,
    kernel_masked_forward_kinematics,
    kernel_masked_forward_velocity,
)
from .rigid.abd.misc import (
    kernel_init_dof_fields,
    kernel_init_entity_fields,
    kernel_init_joint_fields,
    kernel_init_link_fields,
)
from .rigid.render import (
    ASSET_VARIANT_IDX,
    VARIANTS,
    RigidRenderData,
    kernel_add_propellers_revs,
    kernel_reset_propellers_revs,
    kernel_reset_vverts_variant,
    kernel_set_batched_vverts,
    kernel_set_vverts_variant,
    kernel_update_dynamic_vverts,
    kernel_update_render_vgeoms,
)

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator
    from genesis.typing import IndexType


TERRAIN_HEIGHT_QUERY_TILT_TOLERANCE = 1e-3


def _select_links_offset(offset, links_idx, envs_idx):
    """Index a base-link forward offset for a transposed state query.

    'offset' has shape '[n_links, dim]' for an environment-uniform offset or '[n_envs, n_links, dim]' when
    heterogeneous variants make it environment-specific. It is indexed with the same combined mask as 'qd_to_torch' so
    the result broadcasts against the '[n_envs, n_sel, dim]' state tensor, keeping the link dimension for integer
    indices.
    """
    if offset.ndim == 2:
        return offset[indices_to_mask(links_idx)]
    return offset[indices_to_mask(envs_idx, links_idx, boolean_mask=False)]


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


class KinematicSolver(Solver):
    """
    Base solver for articulated kinematic entities (FK, rendering, state get/set).

    Provides the full build pipeline, field init methods, counter properties,
    render methods, and IO sanitization shared by both KinematicSolver and RigidSolver.

    RigidSolver extends this with physics (collision, constraints, dynamics).
    """

    entity_spec_cls: ClassVar[type[gs.entities.Spec]] = gs.entities.Kinematic

    # Its whole state lives in the arrays 'data' yields (see 'CheckpointMixin')
    is_checkpointable = True

    # Kinematic entities carry no dynamics, so their state stays outside the autograd graph of a differentiable scene.
    # The rigid solver deriving from this one reverses its own dynamics.
    is_differentiable = True

    # The kinds of entity this solver builds, each with the morph class it is built from, the most specific first
    _entity_classes = ((Morph, KinematicEntity),)

    def __init__(self, sim: "Simulator", options: "KinematicOptions") -> None:
        super().__init__(sim, options)

        self._enable_collision = False
        self._enable_mujoco_compatibility = False
        # See 'is_differentiable' above
        self._requires_grad = False
        self._enable_heterogeneous = False  # Set to True when any entity has heterogeneous morphs

        # The structure of the entities added so far, which 'add_entity' and 'attach' keep up to date
        self._links: list[KinematicLink] = gs.List()
        self._joints: list[RigidJoint] = gs.List()
        self._vgeoms: list[RigidVisGeom] = gs.List()
        self._n_qs = 0
        self._n_dofs = 0
        self._n_vverts = 0
        self._n_vfaces = 0

        self.collider = None
        self.constraint_solver = None

        self.qpos = None
        self._ik_scratch: IKScratch | None = None

        self._is_forward_pos_updated: bool = False
        self._is_forward_vel_updated: bool = False

    # ------------------------------------------------------------------------------------
    # ----------------------------------- add_entity -------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(
        self,
        idx: int,
        surface: Surface | None,
        spec: "gs.entities.Kinematic | gs.entities.Rigid | None",
        desc: KinematicEntityDescription | None = None,
    ) -> KinematicEntity:
        """Create an entity from its description.

        Given no description, one is resolved from the entity spec and its resolved surface (see
        'KinematicEntityDescription'), the primary morph deciding the kind of entity ('_entity_classes'). A given
        description says the kind by its class, and creating from it reads no asset.
        """
        if desc is None:
            entity_cls = next(cls for morph_cls, cls in self._entity_classes if isinstance(spec.morphs[0], morph_cls))
            desc = entity_cls._description_cls.resolve(spec, surface, self._options, self._enable_mujoco_compatibility)
        else:
            for _, entity_cls in self._entity_classes:
                if entity_cls._description_cls is type(desc):
                    break
            else:
                gs.raise_exception(f"{type(self).__name__} simulates no entity described by {type(desc).__name__}.")
        self._enable_heterogeneous |= bool(desc.variants)
        entity = entity_cls(self._sim.scene, self, idx, desc)
        self._entities.append(entity)
        self._links.extend(entity.links)
        self._joints.extend(entity.joints)
        self._vgeoms.extend(entity.vgeoms)
        self._n_qs += entity.n_qs
        self._n_dofs += entity.n_dofs
        self._n_vverts += entity.n_vverts
        self._n_vfaces += entity.n_vfaces
        # An attachment names an entity added before this one, which 'attach' requires, so it is restored here
        if desc.attachment is not None:
            entity.attach(self._sim.get_entity(desc.attachment.entity_name), desc.attachment.link_name)
        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------ build -----------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_unbuilt
    def build(self, n_envs: int) -> None:
        super().build(n_envs)

        # Resolve the link-reach structures of @mutates notifications; see Solver.__init__.
        joints = self.joints
        joints_link_idx = np.array([joint.link.idx for joint in joints], dtype=gs.np_int)
        joints_n_dofs = np.array([joint.n_dofs for joint in joints], dtype=gs.np_int)
        self._articulated_links_idx = np.unique(joints_link_idx[joints_n_dofs > 0])
        self._links_parent_idx = np.array([link.parent_idx for link in self.links], dtype=gs.np_int)
        self._indices_link_idx["dofs_idx"] = np.repeat(joints_link_idx, joints_n_dofs)
        self._indices_link_idx["qs_idx"] = np.repeat(joints_link_idx, [joint.n_qs for joint in joints])

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
        offset_shape = (self.n_envs_, self.n_links) if links_offset_per_env else (self.n_links,)
        links_offset_pos = np.zeros((*offset_shape, 3), dtype=gs.np_float)
        links_offset_quat = np.tile(gu.identity_quat(), (*offset_shape, 1))
        vgeoms_offset_pos = np.zeros((self.n_vgeoms, 3), dtype=gs.np_float)
        vgeoms_offset_quat = np.tile(gu.identity_quat(), (self.n_vgeoms, 1))
        for entity in self._entities:
            if links_offset_per_env and entity._desc.variants:
                variant_idx = _balanced_variant_mapping(len(entity._desc.variants), self.n_envs_)
                links_offset_pos[np.arange(self.n_envs_), entity.base_link_idx] = np.stack(
                    [entity._desc.variants[v].offset_pos for v in variant_idx]
                )
                links_offset_quat[np.arange(self.n_envs_), entity.base_link_idx] = np.stack(
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
        self.n_entities_ = max(1, self.n_entities)

        # The kinematic roots and trees, see roots_link_idx and trees_root_idx in rigid/data.py. A link no dof moves,
        # its own or an ancestor's, is static and belongs to no tree. A moving link roots the tree of its parent, or its
        # own when its parent is static or absent. The parents precede the children.
        self._n_roots = len({link.root_idx for link in self.links})
        self.n_roots_ = max(1, self._n_roots)
        # The pose of a fixed root link is state, held apart from the poses forward kinematics derives (see
        # FixedRootsState in rigid/data.py).
        links_is_fixed_root = np.array([link.parent_idx == -1 and link.is_fixed for link in self.links], dtype=bool)
        self._links_fixed_root_idx = np.where(
            links_is_fixed_root, np.cumsum(links_is_fixed_root, dtype=gs.np_int) - 1, -1
        )
        self.n_fixed_roots_ = max(1, np.count_nonzero(links_is_fixed_root))
        links_is_static = np.ones(self.n_links, dtype=bool)
        links_tree_root_idx = np.full(self.n_links, -1, dtype=gs.np_int)
        for link in self.links:
            is_parent_static = link.parent_idx == -1 or links_is_static[link.parent_idx]
            links_is_static[link.idx] = is_parent_static and link.n_dofs == 0
            if links_is_static[link.idx]:
                continue
            links_tree_root_idx[link.idx] = link.idx if is_parent_static else links_tree_root_idx[link.parent_idx]
        self._links_tree_root_idx = links_tree_root_idx
        self._n_trees = np.unique(links_tree_root_idx[links_tree_root_idx >= 0]).size
        self.n_trees_ = max(1, self._n_trees)

        # The level of a link in its root and in its tree is its number of ancestors below the root link of either (see
        # trees_levels_links_idx in rigid/data.py). Each pass moves every link one ancestor up, so the depth of a link
        # is the number of passes it takes to run out of parents.
        links_depth = np.zeros(self.n_links, dtype=gs.np_int)
        links_ancestor_idx = self._links_parent_idx
        while (links_ancestor_idx >= 0).any():
            has_ancestor = links_ancestor_idx >= 0
            links_depth += has_ancestor
            links_ancestor_idx = np.where(has_ancestor, self._links_parent_idx[links_ancestor_idx], -1)
        self._links_root_idx = np.array([link.root_idx for link in self.links], dtype=gs.np_int)
        self._links_root_level = links_depth - links_depth[self._links_root_idx]
        self._links_tree_level = np.where(links_tree_root_idx >= 0, links_depth - links_depth[links_tree_root_idx], -1)

        # Heterogeneous simulation stores the link, joint and dof info per environment, every variant stating its own
        # (see '_dispatch_heterogeneous_kinematics'). We must update options because the getters read from them.
        if self._enable_heterogeneous:
            self._options.batch_links_info = True
            self._options.batch_joints_info = True
            self._options.batch_dofs_info = True

        self._build_config()
        self._create_data_manager()

        self._init_dof_fields()
        self._init_link_fields()
        self._init_tree_fields()
        self._init_entity_fields()

        # The entities, their links, joints and geoms read the data of the solver, which exists from now on. The rest
        # of the build of the rigid solver calls their accessors, and theirs alike.
        for entity in self._entities:
            entity._build()
        self._is_built = True

    def _build_config(self):
        # Static config with all physics disabled
        self.rigid_config = RigidConfig(
            backend=gs.backend,
            para_level=self.sim.para_level,
            requires_grad=False,
            use_hibernation=False,
            batch_links_info=self._options.batch_links_info,
            batch_dofs_info=self._options.batch_dofs_info,
            batch_joints_info=self._options.batch_joints_info,
            enable_mujoco_compatibility=False,
            enable_elliptic_friction=False,
            enable_signorini_contact=False,
            enable_multi_contact=False,
            enable_collision=False,
            enable_joint_limit=False,
            box_box_detection=False,
            sparse_solve=False,
            integrator=gs.integrator.approximate_implicitfast,
            solver_type=gs.constraint_solver.CG,
        )

    def _create_data_manager(self):
        self.data_manager = RigidData(self, kinematic_only=True)
        self._errno = self.data_manager.errno
        self.rigid_info = self.data_manager.rigid_info
        self.dyn_info = self.data_manager.dyn_info
        self.dyn_state = self.data_manager.dyn_state
        self.kinematics_scratch = self.data_manager.kinematics_scratch

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        yield from iter_data(self, "rigid_config")
        yield from iter_data(self.data_manager, "errno", DataKind.STATE)
        for name in ("rigid_info", "dyn_info", "dyn_state"):
            yield from iter_data(self.data_manager, name)
        # The flags describe the derived poses and velocities, so they travel with them
        yield from iter_data(self, "_is_forward_pos_updated", DataKind.DERIVED)
        yield from iter_data(self, "_is_forward_vel_updated", DataKind.DERIVED)

    # ------------------------------------------------------------------------------------
    # --------------------------------- hook methods -------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_joint_sol_params(self, sol_params):
        """Hook: sanitize joint constraint solver params. No-op in base (no constraints)."""
        return sol_params

    # ------------------------------------------------------------------------------------
    # --------------------------------- init methods -------------------------------------
    # ------------------------------------------------------------------------------------

    def _init_dof_fields(self):
        joints = self.joints
        has_dofs = sum(joint.n_dofs for joint in joints) > 0
        if has_dofs:
            kernel_init_dof_fields(
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
        """Initialize the fields describing the kinematic roots and trees (see roots_link_idx and trees_root_idx in
        rigid/data.py).

        The links come parent first and each branch occupies a contiguous index range, so the dofs of a tree form one
        contiguous range and the trees are disjoint in dof space.
        """
        # A static link belongs to no tree, and a scene without any tree keeps its padded tree slot empty (see
        # trees_root_idx in rigid/data.py)
        links_tree_idx = np.full(self.n_links_, -1, dtype=gs.np_int)
        if self._n_roots:
            roots_link_idx, links_root_rank = np.unique(self._links_root_idx, return_inverse=True)
            roots_link_end = np.zeros(self._n_roots, dtype=gs.np_int)
            np.maximum.at(roots_link_end, links_root_rank, np.arange(1, self.n_links + 1, dtype=gs.np_int))
            self.rigid_info.roots_link_idx.from_numpy(roots_link_idx)
            self.rigid_info.links_root_end.from_numpy(roots_link_end[links_root_rank])
        if self._n_trees:
            links_n_dofs = np.array([link.n_dofs for link in self.links], dtype=gs.np_int)
            links_dof_start = np.array([link.dof_start for link in self.links], dtype=gs.np_int)
            links_dof_end = np.array([link.dof_end for link in self.links], dtype=gs.np_int)
            tree_links = np.flatnonzero(self._links_tree_root_idx >= 0)
            trees_root_idx, links_tree_rank = np.unique(self._links_tree_root_idx[tree_links], return_inverse=True)
            trees_n_dofs = np.zeros(self._n_trees, dtype=gs.np_int)
            np.add.at(trees_n_dofs, links_tree_rank, links_n_dofs[tree_links])
            # A dof-less link carries no dof range of its own
            dof_links = np.flatnonzero(links_n_dofs[tree_links])
            trees_dof_start = np.full(self._n_trees, self.n_dofs, dtype=gs.np_int)
            np.minimum.at(trees_dof_start, links_tree_rank[dof_links], links_dof_start[tree_links][dof_links])
            trees_dof_end = np.zeros(self._n_trees, dtype=gs.np_int)
            np.maximum.at(trees_dof_end, links_tree_rank[dof_links], links_dof_end[tree_links][dof_links])
            if (trees_dof_end - trees_dof_start != trees_n_dofs).any():
                gs.raise_exception("The dofs of a kinematic tree must be contiguous.")
            trees_link_end = np.zeros(self._n_trees, dtype=gs.np_int)
            np.maximum.at(trees_link_end, links_tree_rank, tree_links + 1)
            trees_n_links = np.zeros(self._n_trees, dtype=gs.np_int)
            np.add.at(trees_n_links, links_tree_rank, 1)
            trees_order = np.argsort(trees_dof_start)
            trees_rank = np.empty(self._n_trees, dtype=gs.np_int)
            trees_rank[trees_order] = np.arange(self._n_trees, dtype=gs.np_int)
            links_tree_idx[tree_links] = trees_rank[links_tree_rank]
            self.rigid_info.trees_root_idx.from_numpy(trees_root_idx[trees_order])
            self.rigid_info.trees_link_end.from_numpy(trees_link_end[trees_order])
            self.rigid_info.trees_n_links.from_numpy(trees_n_links[trees_order])
            self.rigid_info.trees_dof_start.from_numpy(trees_dof_start[trees_order])
            self.rigid_info.trees_n_dofs.from_numpy(trees_n_dofs[trees_order])
        else:
            self.rigid_info.trees_root_idx.fill(0)
            self.rigid_info.trees_link_end.fill(0)
        self.rigid_info.links_tree_idx.from_numpy(links_tree_idx)

        # Fill the level tables of every tree and root for the level sweep (see trees_levels_links_idx in
        # rigid/data.py). A scene without any tree keeps its padded tree slot at zero levels.
        if self._n_roots and self.rigid_config.enable_level_sweep:
            for links_group_idx, links_group_level, n_groups, groups_fields in (
                (
                    links_tree_idx,
                    self._links_tree_level,
                    self.n_trees_,
                    (
                        self.rigid_info.trees_level_start,
                        self.rigid_info.trees_n_levels,
                        self.rigid_info.trees_levels_links_idx,
                        None,
                        self.rigid_info.trees_levels_links_end,
                    ),
                ),
                (
                    links_root_rank,
                    self._links_root_level,
                    self._n_roots,
                    (
                        self.rigid_info.roots_level_start,
                        self.rigid_info.roots_n_levels,
                        self.rigid_info.roots_levels_links_idx,
                        self.rigid_info.roots_levels_links_start,
                        self.rigid_info.roots_levels_links_end,
                    ),
                ),
            ):
                group_links = np.arange(self.n_links, dtype=gs.np_int)[links_group_idx >= 0]
                links_group = links_group_idx[group_links]
                links_level = links_group_level[group_links]
                groups_n_levels = np.zeros(n_groups, dtype=gs.np_int)
                np.maximum.at(groups_n_levels, links_group, links_level + 1)
                groups_level_start = np.zeros(n_groups, dtype=gs.np_int)
                np.cumsum(np.bincount(links_group, minlength=n_groups)[:-1], out=groups_level_start[1:])
                # The links of every group, group after group, level after level, and each level in ascending order
                entries_order = np.lexsort((group_links, links_level, links_group))
                groups_levels_links_idx = group_links[entries_order]
                entries_group = links_group[entries_order]
                entries_level = links_level[entries_order]
                entries_idx = np.arange(len(groups_levels_links_idx), dtype=gs.np_int)
                # An entry ends its level where the next entry starts another level or another group
                is_level_last = np.ones(len(groups_levels_links_idx), dtype=bool)
                is_level_last[:-1] = (np.diff(entries_group) != 0) | (np.diff(entries_level) != 0)
                levels_last_entry = entries_idx[is_level_last]
                groups_levels_links_end = levels_last_entry[np.searchsorted(levels_last_entry, entries_idx)] + 1
                is_level_first = np.ones(len(groups_levels_links_idx), dtype=bool)
                is_level_first[1:] = is_level_last[:-1]
                levels_first_entry = entries_idx[is_level_first]
                groups_levels_links_start = levels_first_entry[
                    np.searchsorted(levels_first_entry, entries_idx, side="right") - 1
                ]
                for field, value in zip(
                    groups_fields,
                    (
                        groups_level_start,
                        groups_n_levels,
                        groups_levels_links_idx,
                        groups_levels_links_start,
                        groups_levels_links_end,
                    ),
                ):
                    if field is not None and len(value):
                        field.from_numpy(value)
            child_links = np.arange(self.n_links, dtype=gs.np_int)[self._links_parent_idx >= 0]
            links_child_start = np.zeros(self.n_links_ + 1, dtype=gs.np_int)
            np.cumsum(
                np.bincount(self._links_parent_idx[child_links], minlength=self.n_links_), out=links_child_start[1:]
            )
            self.rigid_info.links_child_start.from_numpy(links_child_start)
            if len(child_links):
                self.rigid_info.links_child_idx.from_numpy(
                    child_links[np.lexsort((-child_links, self._links_parent_idx[child_links]))]
                )

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
                np.array([link.is_fixed for link in links], dtype=gs.np_bool),
                self._links_fixed_root_idx,
                np.array([link.desc.pos for link in links], dtype=gs.np_float),
                np.array([link.desc.quat for link in links], dtype=gs.np_float),
                self.dyn_state,
                self.dyn_info,
                self.rigid_config,
            )

        if self.joints:
            joints = self.joints
            joints_sol_params = np.array([joint.desc.sol_params for joint in joints], dtype=gs.np_float)
            joints_sol_params = self._sanitize_joint_sol_params(joints_sol_params)

            kernel_init_joint_fields(
                np.array([joint.q_start for joint in joints], dtype=gs.np_int),
                np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
                np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
                np.array([joint.type for joint in joints], dtype=gs.np_int),
                joints_sol_params,
                np.array([joint.pos for joint in joints], dtype=gs.np_float),
                self.dyn_info,
                self.rigid_config,
            )

        self._dispatch_heterogeneous_kinematics()

        # Set initial qpos
        self.qpos = self.rigid_info.qpos
        self.qpos0 = self.rigid_info.qpos0
        if self.n_qs > 0:
            init_qpos = np.tile(np.expand_dims(self.init_qpos, -1), (1, self.n_envs_))

            # Dispatch per-variant init_qpos for heterogeneous entities
            for entity in self.entities:
                if not entity._desc.variants:
                    continue
                _, _, variants_qpos0, envs_variant_idx = get_variants_kinematics(entity, n_envs=self.n_envs_)
                init_qpos[entity.q_start : entity.q_end] = np.array(variants_qpos0)[envs_variant_idx].T

            self.qpos0.from_numpy(init_qpos)
            dofs_limit = np.broadcast_to(
                qd_to_numpy(self.dyn_info.dofs.limit, transpose=True), (self.n_envs_, self.n_dofs_, 2)
            )
            is_init_qpos_out_of_bounds = False
            for joint in self.joints:
                if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                    is_init_qpos_out_of_bounds |= (dofs_limit[:, joint.dof_start, 0] > init_qpos[joint.q_start]).any()
                    is_init_qpos_out_of_bounds |= (init_qpos[joint.q_start] > dofs_limit[:, joint.dof_start, 1]).any()
            if is_init_qpos_out_of_bounds:
                gs.logger.warning("Neutral robot position (qpos0) exceeds joint limits.")
            self.qpos.from_numpy(init_qpos)

        # Dispatch heterogeneous variant vgeom ranges per-environment
        self._dispatch_heterogeneous_vgeoms()

    def _dispatch_heterogeneous_kinematics(self):
        """Write the link frames and the joint and dof parameters of each heterogeneous variant to the envs carrying it.

        The init kernels broadcast those of the primary to every environment, while heterogeneous simulation batches
        the link, joint and dof info (see 'build').
        """
        entities = [entity for entity in self._entities if entity._desc.variants]
        if not entities:
            return

        # Field layout, batch second, since the arrays are written back as fields
        links_pos = qd_to_numpy(self.dyn_info.links.pos, transpose=False, copy=True)
        links_quat = qd_to_numpy(self.dyn_info.links.quat, transpose=False, copy=True)
        joints_pos = qd_to_numpy(self.dyn_info.joints.pos, transpose=False, copy=True)
        joints_sol_params = qd_to_numpy(self.dyn_info.joints.sol_params, transpose=False, copy=True)
        dofs_info = self.dyn_info.dofs
        dofs_fields = (
            (dofs_info.motion_ang, "dofs_motion_ang"),
            (dofs_info.motion_vel, "dofs_motion_vel"),
            (dofs_info.limit, "dofs_limit"),
            (dofs_info.stiffness, "dofs_stiffness"),
            (dofs_info.damping, "dofs_damping"),
            (dofs_info.frictionloss, "dofs_frictionloss"),
            (dofs_info.armature, "dofs_armature"),
            (dofs_info.act_gain, "dofs_act_gain"),
            (dofs_info.act_bias, "dofs_act_bias"),
            (dofs_info.force_range, "dofs_force_range"),
        )
        dofs_values = [qd_to_numpy(field, transpose=False, copy=True) for field, _ in dofs_fields]

        for entity in entities:
            variants_links, variants_j_descs, _, envs_variant_idx = get_variants_kinematics(entity, n_envs=self.n_envs_)
            links_slice = slice(entity.link_start, entity.link_end)
            variants_links_pos = np.stack([[l_desc.pos for l_desc in links] for links in variants_links])
            variants_links_quat = np.stack([[l_desc.quat for l_desc in links] for links in variants_links])
            links_pos[links_slice] = variants_links_pos[envs_variant_idx].swapaxes(0, 1)
            links_quat[links_slice] = variants_links_quat[envs_variant_idx].swapaxes(0, 1)

            # A file morph drops its joints without degrees of freedom, so a fixed one may leave the entity without any
            if entity.n_joints == 0:
                continue
            variants_joints = [[asdict(j_desc) for j_desc in j_descs] for j_descs in variants_j_descs]
            joints_slice = slice(entity.joint_start, entity.joint_end)
            variants_joints_pos = np.stack([[j_fields["pos"] for j_fields in joints] for joints in variants_joints])
            joints_pos[joints_slice] = variants_joints_pos[envs_variant_idx].swapaxes(0, 1)
            variants_sol_params = np.stack(
                [[j_fields["sol_params"] for j_fields in joints] for joints in variants_joints]
            )
            variants_sol_params = self._sanitize_joint_sol_params(variants_sol_params).reshape(
                variants_sol_params.shape
            )
            joints_sol_params[joints_slice] = variants_sol_params[envs_variant_idx].swapaxes(0, 1)
            dofs_slice = slice(entity.dof_start, entity.dof_end)
            for dofs_value, (_, name) in zip(dofs_values, dofs_fields):
                variants_value = np.stack(
                    [np.concatenate([j_fields[name] for j_fields in joints]) for joints in variants_joints]
                )
                dofs_value[dofs_slice] = variants_value[envs_variant_idx].swapaxes(0, 1)

        self.dyn_info.links.pos.from_numpy(links_pos)
        self.dyn_info.links.quat.from_numpy(links_quat)
        self.dyn_info.joints.pos.from_numpy(joints_pos)
        self.dyn_info.joints.sol_params.from_numpy(joints_sol_params)
        for (field, _), dofs_value in zip(dofs_fields, dofs_values):
            field.from_numpy(dofs_value)

        # The pose of a fixed root link starts at the frame of the variant its environment carries (see
        # kernel_init_link_fields).
        links_is_fixed_root = self._links_fixed_root_idx >= 0
        if links_is_fixed_root.any():
            self.dyn_state.fixed_roots.pos.from_numpy(links_pos[links_is_fixed_root])
            self.dyn_state.fixed_roots.quat.from_numpy(links_quat[links_is_fixed_root])

    def _dispatch_heterogeneous_vgeoms(self):
        """Mark the environments holding each visual geom of a heterogeneous variant."""
        for link in self.links:
            if link._variant_vgeom_ranges is None:
                continue

            n_variants = len(link._variant_vgeom_ranges)
            variant_idx = _balanced_variant_mapping(n_variants, self.n_envs_)

            vgeom_starts = np.array([link._variant_vgeom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
            vgeom_ends = np.array([link._variant_vgeom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)
            for vgeom in link.vgeoms:
                active_envs_mask = (vgeom_starts <= vgeom.idx) & (vgeom.idx < vgeom_ends)
                vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                (vgeom.active_envs_idx,) = np.where(active_envs_mask)

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
                self.dyn_state,
                self.dyn_info,
                self.rigid_config,
            )

    # ------------------------------------------------------------------------------------
    # -------------------------------- simulation no-ops ----------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self):
        pass

    def substep_post_coupling(self):
        if not self._is_forward_pos_updated or not self._is_forward_vel_updated:
            kernel_forward_kinematics(self._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True

    # ------------------------------------------------------------------------------------
    # -------------------------------- state get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self) -> SolverCheckpoint:
        # A restored state runs forward kinematics (see 'set_state'), so a state is captured with the poses of the links
        # its configuration implies, which the step loop leaves one integration behind whenever it defers the refresh
        # that follows it.
        self.update_forward_pos()
        return super().get_state()

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(
        self,
        state: SolverCheckpoint,
        envs_idx: int | Sequence[int] | np.ndarray | torch.Tensor | None = None,
        *,
        partial: bool = False,
    ) -> None:
        """
        Write back the state 'get_state' returned, for the given environments, all of them if None.

        With `partial`, the forward kinematics of the restored configuration is left to the next step, which spares it
        to a reset followed by setters. The link poses read before that step are those of the previous configuration.
        """
        super().set_state(state, envs_idx)
        if partial:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False
        else:
            # FIXME: quadrants#1086 - fast math reassociates sums by code layout, so the kinematics recomputed here can
            # differ in the last bits from those the step left, and a replay from a saved state matches up to rounding.
            kernel_forward_kinematics(
                self._sim.sanitize_envs_idx(envs_idx),
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True

    def load_checkpoint(
        self,
        checkpoint: SolverCheckpoint,
        *,
        boundary_only: bool = False,
        envs_idx: int | Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().load_checkpoint(checkpoint, boundary_only=boundary_only, envs_idx=envs_idx)
        if DataKind.DERIVED not in checkpoint.kinds:
            # Without the derived arrays, the next step recomputes them from the state. A restored record runs forward
            # kinematics at once, since the scene may be drawn from it, while the replay of a step restoring a boundary
            # checkpoint recomputes them with the same kernels as its forward pass.
            self._is_forward_pos_updated = self._is_forward_vel_updated = False
            if not boundary_only:
                self.update_forward_pos()

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def __setstate__(self, state: SolverCheckpoint) -> None:
        self.load_checkpoint(state)

    @property
    def is_active(self):
        return self.n_links > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

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
            cur_quat = self._read_data(self.dyn_state.links.quat, envs_idx, links_idx)
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
                cur_pos, cur_quat = self._read_data(
                    (self.dyn_state.links.pos, self.dyn_state.links.quat), envs_idx, links_idx
                )
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

    @mutates(StateChange.GEOMETRY, links="qs_idx")
    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            data = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
            mask = (0, *indices_to_mask(qs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, qs_idx)
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
                envs_idx = self._sim.sanitize_envs_idx(envs_idx)
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

    @mutates(StateChange.DYNAMICS, links="dofs_idx")
    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            vel = qd_to_torch(self.dyn_state.dofs.vel, transpose=True, copy=False)
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            assign_indexed_tensor(vel, mask, value=0.0 if velocity is None else velocity)
            if mask and isinstance(mask[0], torch.Tensor):
                envs_idx = mask[0].reshape((-1,))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            if not skip_forward and not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._sim.sanitize_envs_idx(envs_idx)
        else:
            velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
                velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            if self.n_envs == 0 and velocity is not None:
                velocity = velocity[None]
            if velocity is None:
                kernel_set_dofs_zero_velocity(dofs_idx, envs_idx, self.dyn_state, self.rigid_config)
            else:
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

    @mutates(StateChange.GEOMETRY, links="dofs_idx")
    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]
        # The positions of the other dofs are read from the state, so that the gradient of a recorded write reaches both
        # the positions given and the state the others are read from (see 'kernel_set_dofs_position')
        positions = self._read_data(self.dyn_state.dofs.pos, envs_idx, None, copy=True)
        positions[:, dofs_idx] = position
        is_dof_set = torch.zeros((self.n_dofs,), dtype=gs.tc_bool, device=gs.device)
        is_dof_set[dofs_idx] = True
        kernel_set_dofs_position(is_dof_set, envs_idx, positions, self.dyn_info, self.rigid_info, self.rigid_config)

        kernel_forward_kinematics(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def get_terrain_height(self, positions, link_idx, envs_idx=None):
        terrain = self._links[link_idx].entity

        positions = torch.as_tensor(positions, dtype=gs.tc_float, device=gs.device)
        if positions.ndim == 0 or positions.ndim > 3 or positions.shape[-1] != 2:
            gs.raise_exception("`positions` must have shape (2,), (n_points, 2), or (n_envs, n_points, 2).")

        envs_idx = self._sim.sanitize_envs_idx(envs_idx)
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

    def get_links_pos(self, links_idx=None, envs_idx=None, *, relative=False, copy: bool | None = True):
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = self._sanitize_io_variables(
                None, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
        tensor = self._read_data(self.dyn_state.links.pos, envs_idx, links_idx, copy=copy)
        if relative and not self._links_offset_pos_is_identity[idx].all():
            quat = self._read_data(self.dyn_state.links.quat, envs_idx, links_idx)
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor = tensor - _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_quat(self, links_idx=None, envs_idx=None, *, relative=False, copy: bool | None = True):
        tensor = self._read_data(self.dyn_state.links.quat, envs_idx, links_idx, copy=copy)
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not self._links_offset_quat_is_identity[idx].all():
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_joints_anchor_pos(self, joints_idx=None, envs_idx=None, *, copy: bool | None = True):
        tensor = qd_to_torch(self.dyn_state.joints.xanchor, envs_idx, joints_idx, transpose=True, copy=copy)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_joints_anchor_axis(self, joints_idx=None, envs_idx=None, *, copy: bool | None = True):
        tensor = qd_to_torch(self.dyn_state.joints.xaxis, envs_idx, joints_idx, transpose=True, copy=copy)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_vgeoms_pos(self, vgeoms_idx=None, envs_idx=None, *, relative=False):
        tensor = self.render_context.get_vgeoms_pose(vgeoms_idx, envs_idx, is_quat=False)
        if relative and self._vgeoms_offset_pos is not None:
            quat = self.render_context.get_vgeoms_pose(vgeoms_idx, envs_idx, is_quat=True)
            offset_pos = self._vgeoms_offset_pos if vgeoms_idx is None else self._vgeoms_offset_pos[vgeoms_idx]
            offset_quat = self._vgeoms_offset_quat if vgeoms_idx is None else self._vgeoms_offset_quat[vgeoms_idx]
            tensor = tensor - _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_vgeoms_quat(self, vgeoms_idx=None, envs_idx=None, *, relative=False):
        tensor = self.render_context.get_vgeoms_pose(vgeoms_idx, envs_idx, is_quat=True)
        if relative and self._vgeoms_offset_quat is not None:
            offset_quat = self._vgeoms_offset_quat if vgeoms_idx is None else self._vgeoms_offset_quat[vgeoms_idx]
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_vel(self, links_idx=None, envs_idx=None, *, relative=False):
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        is_relative = relative and not self._links_offset_pos_is_identity[idx].all()
        # The velocity of the links is derived from their pose, so the recorded queries of a differentiable scene take
        # the torch path, whatever the backend
        if gs.use_zerocopy or self.is_recording:
            links_state = self.dyn_state.links
            cd_vel, cd_ang, pos, root_COM = self._read_data(
                (links_state.cd_vel, links_state.cd_ang, links_state.pos, links_state.root_COM), envs_idx, links_idx
            )
            cpos = pos - root_COM
            if is_relative:
                quat = self._read_data(links_state.quat, envs_idx, links_idx)
                offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                cpos = cpos - _offset_world_shift(offset_pos, offset_quat, quat)
            tensor = cd_vel + cd_ang.cross(cpos, dim=-1)
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

    def get_links_ang(self, links_idx=None, envs_idx=None, *, copy: bool | None = True):
        tensor = self._read_data(self.dyn_state.links.cd_ang, envs_idx, links_idx, copy=copy)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_qpos(self, qs_idx=None, envs_idx=None, *, copy: bool | None = True):
        tensor = self._read_data(self.qpos, envs_idx, qs_idx, copy=copy)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_velocity(self, dofs_idx=None, envs_idx=None, *, copy: bool | None = True):
        tensor = self._read_data(self.dyn_state.dofs.vel, envs_idx, dofs_idx, copy=copy)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_position(self, dofs_idx=None, envs_idx=None, *, copy: bool | None = True):
        """Read current DOF positions."""
        tensor = self._read_data(self.dyn_state.dofs.pos, envs_idx, dofs_idx, copy=copy)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_limit(self, dofs_idx=None, envs_idx=None, *, copy: bool | None = True):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.limit, envs_idx, dofs_idx, transpose=True, copy=copy)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def update_forward_pos(self):
        """Run forward kinematics if links_state is not already up to date for the current pose."""
        if self._is_forward_pos_updated:
            return
        kernel_forward_kinematics(self._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        self._is_forward_pos_updated = True

    # ------------------------------------------------------------------------------------
    # --------------------------------- Jacobian & IK ------------------------------------
    # ------------------------------------------------------------------------------------

    def get_links_jacobian(self, link_idx, dof_start, n_dofs, local_point=None, *, copy: bool | None = True):
        """Spatial Jacobian of a link-local point, as (n_envs, 6, n_dofs), for the entity owning that link.

        dof_start offsets the entity degrees of freedom into the solver and n_dofs narrows the shared buffer down to
        that entity. local_point defaults to the link origin.
        """
        jacobian = self.kinematics_scratch.jacobian
        if local_point is None:
            kernel_get_jacobian_zero(
                link_idx, dof_start, jacobian, self.dyn_state, self.dyn_info, self.rigid_config, self.n_envs_
            )
        else:
            kernel_get_jacobian(
                link_idx,
                dof_start,
                local_point,
                jacobian,
                self.dyn_state,
                self.dyn_info,
                self.rigid_config,
                self.n_envs_,
            )
        return qd_to_torch(jacobian, transpose=True, copy=copy)[..., :n_dofs]

    def forward_kinematics_query(self, entity, qpos, envs_idx=None):
        """Link poses one entity would have at the given configuration, leaving the live configuration as it was.

        Distinct from the forward kinematics the step propagates: this evaluates a configuration the solver does not
        hold, and restores the one it does. Returns positions (n_envs, n_links, 3) and orientations
        (n_envs, n_links, 4), without the batch dimension when the scene is not batched.
        """
        envs_idx = self._sim.sanitize_envs_idx(envs_idx)
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

        # The damped normal matrix of inverse kinematics is quadratic in RigidOptions.IK_max_targets, so the scratch is
        # allocated by the first solve and a scene that never solves inverse kinematics allocates none of it
        if self._ik_scratch is None:
            entities = self.entities
            self._ik_scratch = IKScratch(
                sizes=dict(
                    envs=self.n_envs_,
                    entity_qs=max([entity.n_qs for entity in entities], default=0),
                    entity_dofs=max([entity.n_dofs for entity in entities], default=0),
                    entity_links=max([entity.n_links for entity in entities], default=0),
                    entity_joints=max([entity.n_joints for entity in entities], default=0),
                    ik_targets=n_tgts,
                ),
                # The initial configuration of an environment is read as a whole
                env_first={IKScratch.targets.init_qpos: True},
            )
        ik_state, ik_fk, targets = self._ik_scratch.state, self._ik_scratch.fk, self._ik_scratch.targets

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
            init_qpos_t[:n_qs, envs_idx] = init_qpos.T
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
    def links(self) -> gs.List[KinematicLink]:
        """The links of the solver, as a copy."""
        return gs.List(self._links)

    @property
    def joints(self) -> gs.List[RigidJoint]:
        """The joints of the solver, as a copy."""
        return gs.List(self._joints)

    @property
    def equalities(self) -> gs.List[RigidEquality]:
        """The equality constraints the solver enforces, which a kinematic solver holds none of."""
        return gs.List()

    @property
    def geoms(self) -> gs.List[RigidGeom]:
        """The collision geoms of the solver, which a kinematic solver holds none of."""
        return gs.List()

    @property
    def vgeoms(self) -> gs.List[RigidVisGeom]:
        """The visual geoms of the solver, as a copy."""
        return gs.List(self._vgeoms)

    @property
    def n_links(self) -> int:
        return len(self._links)

    @property
    def n_joints(self) -> int:
        return len(self._joints)

    @property
    def n_vgeoms(self) -> int:
        return len(self._vgeoms)

    @property
    def n_vverts(self) -> int:
        return self._n_vverts

    @property
    def n_vfaces(self) -> int:
        return self._n_vfaces

    @property
    def n_qs(self) -> int:
        return self._n_qs

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    @property
    def init_qpos(self):
        if self._entities:
            return np.concatenate([entity.init_qpos for entity in self._entities], dtype=gs.np_float)
        return np.array([], dtype=gs.np_float)


class KinematicScopedRenderContext(ScopedRenderContext[KinematicSolver]):
    """The visual meshes of the visual geoms of a kinematic or rigid solver, posed by their link, and its link frames.

    'set_vverts', 'get_vverts', 'get_vgeoms_pose' and 'add_propellers_rpm' are the interface of the visual geoms and the
    drones of the solver to the appearance the scoped context holds.
    """

    def build(self) -> None:
        solver = self.owner
        # The visual geoms take a slot each, those posed by their link first, so that their poses form one contiguous
        # block, the visual geoms of the entities a deprecated 'vis_mode' shows in their debug representation after the
        # others, so that the block of the scene as it looks leaves them out, then those showing DynamicVertices, which
        # own a slice of the dynamic vertices: the vertices of the variant each environment shows placed at the pose of
        # the visual geom.
        vgeoms = solver.vgeoms
        vgeoms_order = [
            (vgeom.desc.dynamic_vertices is not None, vgeom.entity.surface.vis_mode is not None, vgeom.idx)
            for vgeom in vgeoms
        ]
        self._slots_vgeom = [vgeoms[i_vg] for *_, i_vg in sorted(vgeoms_order)]
        self._n_posed_vgeoms = sum(not is_dynamic for is_dynamic, *_ in vgeoms_order)
        self._n_drawn_vgeoms = sum(not (is_dynamic or is_pinned) for is_dynamic, is_pinned, _ in vgeoms_order)
        self._vgeoms_slot = np.empty(len(vgeoms), dtype=gs.np_int)
        self._vgeoms_slot[[vgeom.idx for vgeom in self._slots_vgeom]] = np.arange(len(vgeoms), dtype=gs.np_int)
        # The start of the slice of each visual geom among the dynamic vertices of the solver, -1 for a posed one, and
        # among the vertices of the batched variants, -1 for a visual geom without.
        self._vgeoms_dynamic_vvert_start = np.full(len(vgeoms), -1, dtype=gs.np_int)
        self._vgeoms_batched_vvert_start = np.full(len(vgeoms), -1, dtype=gs.np_int)
        dynamic_vgeoms = self._slots_vgeom[self._n_posed_vgeoms :]
        # The variants of the vertices of the visual geoms showing DynamicVertices, None for the mesh of the asset
        variants_vverts: list[np.ndarray | None] = []
        vgeoms_variant_start = []
        n_dynamic_vverts = n_batched_vverts = 0
        for vgeom in dynamic_vgeoms:
            if isinstance(vgeom.entity.main_morph, gs.morphs.Plane):
                gs.raise_exception("A plane cannot show DynamicVertices.")
            self._vgeoms_dynamic_vvert_start[vgeom.idx] = n_dynamic_vverts
            n_dynamic_vverts += vgeom.n_vverts
            if vgeom.desc.dynamic_vertices.is_batched:
                self._vgeoms_batched_vvert_start[vgeom.idx] = n_batched_vverts
                n_batched_vverts += vgeom.n_vverts
            variants = vgeom.desc.dynamic_vertices.variants
            vgeoms_variant_start.append(len(variants_vverts))
            for variant in variants.values() if isinstance(variants, dict) else variants:
                if variant is not None and variant.shape != (vgeom.n_vverts, 3):
                    gs.raise_exception(
                        f"A variant of the DynamicVertices of a visual geom of entity '{vgeom.entity.name}' holds the "
                        f"position of each of its {vgeom.n_vverts} vertices, of shape ({vgeom.n_vverts}, 3), got "
                        f"{variant.shape}."
                    )
                variants_vverts.append(variant)
        self._has_variants = any(vgeom.active_envs_mask is not None for vgeom in vgeoms)
        self._data = RigidRenderData(
            sizes=dict(
                vgeoms=len(vgeoms),
                vverts=solver.n_vverts,
                vfaces=solver.n_vfaces,
                propellers=sum(
                    entity.n_propellers
                    for entity in solver.entities
                    if isinstance(entity, DroneEntity) and entity.propellers_vgeom_idx is not None
                ),
                dynamic_vgeoms=len(dynamic_vgeoms),
                dynamic_vverts=n_dynamic_vverts,
                batched_vverts=n_batched_vverts,
                vgeoms_variants=len(variants_vverts),
                variants_vverts=sum(len(variant) for variant in variants_vverts if variant is not None),
                envs=solver.n_envs_,
            ),
            unused={VARIANTS: not self._has_variants},
        )
        # The first propeller of each drone whose propellers turn as drawn
        self._drones_propeller_start: dict[DroneEntity, int] = {}
        # The frames of the links, drawn while 'link_frames' holds
        links = solver.dyn_state.links
        self._link_frames = Frames(ArrayRef(links.pos), ArrayRef(links.quat), solver.options.vis.link_frame_size)
        self._is_link_frames_shown = solver.options.vis.link_frames and solver.n_links > 0
        # The shapes of the visual geoms
        self._shapes: tuple[Shape, ...] = ()
        if not vgeoms:
            return
        data = self._data

        slots_propeller_idx = np.full(len(vgeoms), -1, dtype=gs.np_int)
        slots_propeller_rad_per_rev = np.zeros(len(vgeoms), dtype=gs.np_float)
        n_propellers = 0
        for drone in solver.entities:
            if isinstance(drone, DroneEntity) and drone.propellers_vgeom_idx is not None:
                self._drones_propeller_start[drone] = n_propellers
                propellers_slot = self._vgeoms_slot[tensor_to_array(drone.propellers_vgeom_idx)]
                slots_propeller_idx[propellers_slot] = n_propellers + np.arange(drone.n_propellers)
                # The propellers accumulate their speed in revolutions per minute once per step
                slots_propeller_rad_per_rev[propellers_slot] = (
                    tensor_to_array(drone.propellers_spin) * solver.sim.substep_dt * np.pi / 30.0
                )
                n_propellers += drone.n_propellers
        slots_n_vverts = np.array([vgeom.n_vverts for vgeom in self._slots_vgeom], dtype=gs.np_int)
        self._slots_vvert_start = np.cumsum(slots_n_vverts, dtype=gs.np_int) - slots_n_vverts
        slots_n_vfaces = np.array([vgeom.n_vfaces for vgeom in self._slots_vgeom], dtype=gs.np_int)
        self._slots_vface_start = np.cumsum(slots_n_vfaces, dtype=gs.np_int) - slots_n_vfaces
        data.vgeoms_info.pos.from_numpy(np.array([vgeom.init_pos for vgeom in self._slots_vgeom], dtype=gs.np_float))
        data.vgeoms_info.quat.from_numpy(np.array([vgeom.init_quat for vgeom in self._slots_vgeom], dtype=gs.np_float))
        data.vgeoms_info.vvert_start.from_numpy(self._slots_vvert_start)
        data.vgeoms_info.vvert_end.from_numpy(self._slots_vvert_start + slots_n_vverts)
        data.vgeoms_info.link_idx.from_numpy(np.array([vgeom.link.idx for vgeom in self._slots_vgeom], dtype=gs.np_int))
        data.vgeoms_info.propeller_idx.from_numpy(slots_propeller_idx)
        data.vgeoms_info.propeller_rad_per_rev.from_numpy(slots_propeller_rad_per_rev)
        data.vverts_info.init_pos.from_numpy(
            np.concatenate([vgeom.init_vverts for vgeom in self._slots_vgeom], dtype=gs.np_float)
        )
        data.vverts_info.init_normal.from_numpy(
            np.concatenate([vgeom.init_vnormals for vgeom in self._slots_vgeom], dtype=gs.np_float)
        )
        data.vfaces_info.vverts_idx.from_numpy(
            np.concatenate(
                [
                    vgeom.init_vfaces + vvert_start
                    for vgeom, vvert_start in zip(self._slots_vgeom, self._slots_vvert_start)
                ],
                dtype=gs.np_int,
            )
        )
        if self._has_variants:
            slots_is_active = np.ones((len(vgeoms), solver.n_envs_), dtype=gs.np_bool)
            for i_s, vgeom in enumerate(self._slots_vgeom):
                if vgeom.active_envs_mask is not None:
                    slots_is_active[i_s] = tensor_to_array(vgeom.active_envs_mask)
            data.vgeoms.is_active.from_numpy(slots_is_active)
        if dynamic_vgeoms:
            dynamic_vverts = data.dynamic_vverts
            dynamic_vverts.vgeoms_slot.from_numpy(np.arange(self._n_posed_vgeoms, len(vgeoms), dtype=gs.np_int))
            dynamic_vgeoms_idx = [vgeom.idx for vgeom in dynamic_vgeoms]
            dynamic_vverts.vgeoms_vvert_start.from_numpy(self._vgeoms_dynamic_vvert_start[dynamic_vgeoms_idx])
            dynamic_vverts.vgeoms_batched_vvert_start.from_numpy(self._vgeoms_batched_vvert_start[dynamic_vgeoms_idx])
            dynamic_vverts.vgeoms_variant_start.from_numpy(np.array(vgeoms_variant_start, dtype=gs.np_int))
            variants_size = np.array([0 if variant is None else len(variant) for variant in variants_vverts])
            variants_vvert_start = np.where(variants_size > 0, np.cumsum(variants_size) - variants_size, -1)
            dynamic_vverts.variants_vvert_start.from_numpy(variants_vvert_start.astype(gs.np_int))
            if variants_size.any():
                dynamic_vverts.variants_pos.from_numpy(
                    np.concatenate([variant for variant in variants_vverts if variant is not None], dtype=gs.np_float)
                )
            # The batched variants start as the mesh of the asset in every environment
            if n_batched_vverts > 0:
                init_vverts = np.concatenate(
                    [vgeom.init_vverts for vgeom in dynamic_vgeoms if vgeom.desc.dynamic_vertices.is_batched],
                    dtype=gs.np_float,
                )
                dynamic_vverts.batched_pos.from_numpy(
                    np.ascontiguousarray(np.broadcast_to(init_vverts[:, None], (n_batched_vverts, solver.n_envs_, 3)))
                )
        # The file each entity drawn as loaded is loaded from, which a visual geom of a surface of its own or showing
        # DynamicVertices draws apart.
        entities_asset: dict[int, EntityAsset | None] = {}
        for entity in solver.entities:
            morph = entity.main_morph
            entities_asset[entity.idx] = None
            is_drawn_apart = any(isinstance(patch.options, gs.options.VGeom) for patch in entity.spec.patches) or any(
                vgeom.desc.dynamic_vertices is not None for vgeom in entity.vgeoms
            )
            if isinstance(morph, gs.morphs.FileMorph) and entity.vgeoms and not is_drawn_apart:
                is_zup = bool(entity.vgeoms[0].metadata.get("imported_as_zup", False))
                is_merged = False
                if isinstance(morph, (gs.morphs.URDF, gs.morphs.Drone)):
                    is_zup, is_merged = morph.file_meshes_are_zup, morph.merge_fixed_links
                entities_asset[entity.idx] = EntityAsset(
                    morph.file,
                    is_zup,
                    is_merged,
                    scale=(morph.scale,) * 3 if isinstance(morph.scale, float) else tuple(morph.scale),
                    pos=(0.0, 0.0, 0.0) if morph.pos is None else tuple(morph.pos),
                    quat=(1.0, 0.0, 0.0, 0.0) if morph.quat is None else tuple(morph.quat),
                )
        shapes: list[Shape] = []
        geometries = []
        for i_s, vgeom in enumerate(self._slots_vgeom):
            entity = vgeom.entity
            morph = entity.main_morph
            seg_keys = vgeom.seg_keys
            is_ray_target = entity.spec.use_visual_raycasting
            if i_s < self._n_drawn_vgeoms:
                # The origin of the vertices of the mesh, None for a mesh computed in memory, such as the meshes an
                # MJCF parser makes of its geoms or a terrain.
                metadata = vgeom.metadata
                textures_path = TexturePaths(
                    metadata.get("texture_path"),
                    metadata.get("normal_texture_path"),
                    metadata.get("emissive_texture_path"),
                )
                is_visual_overwritten = bool(metadata.get("is_visual_overwritten", False))
                primitive = None
                if isinstance(morph, gs.morphs.Box):
                    primitive = PrimitiveAsset(gs.GEOM_TYPE.BOX, tuple(morph.size))
                elif isinstance(morph, gs.morphs.Sphere):
                    primitive = PrimitiveAsset(gs.GEOM_TYPE.SPHERE, (morph.radius,))
                elif isinstance(morph, gs.morphs.Cylinder):
                    primitive = PrimitiveAsset(gs.GEOM_TYPE.CYLINDER, (morph.radius, morph.height))
                elif isinstance(morph, gs.morphs.Plane):
                    primitive = PrimitiveAsset(gs.GEOM_TYPE.PLANE, tuple(morph.plane_size))
                source = None
                if primitive is not None:
                    source = MeshSource(
                        file=None,
                        is_zup=False,
                        primitive=primitive,
                        scale=(1.0, 1.0, 1.0),
                        is_visual_overwritten=is_visual_overwritten,
                        textures_path=textures_path,
                    )
                elif not isinstance(morph, gs.morphs.MJCF) and "mesh_path" in metadata:
                    scale = (morph.scale,) * 3 if isinstance(morph.scale, float) else tuple(morph.scale)
                    is_zup = bool(metadata.get("imported_as_zup", False))
                    source = MeshSource(
                        metadata["mesh_path"],
                        is_zup,
                        primitive=None,
                        scale=scale,
                        is_visual_overwritten=is_visual_overwritten,
                        textures_path=textures_path,
                    )
                entity_asset = entities_asset[entity.idx]
                vvert_start, vface_start = self._slots_vvert_start[i_s], self._slots_vface_start[i_s]
                geometries.append(
                    MeshGeometry(
                        vgeom.surface,
                        vgeom.uvs,
                        vgeom.trimesh.visual,
                        seg_keys,
                        source,
                        asset=None
                        if entity_asset is None
                        else MeshAsset(entity_asset, vgeom.link.name, vgeom.link.vgeoms.index(vgeom)),
                        floor=Floor(morph.normal, morph.plane_size) if isinstance(morph, gs.morphs.Plane) else None,
                        verts_ref=ArrayRef(data.vverts_info.init_pos, slice(vvert_start, vvert_start + vgeom.n_vverts)),
                        normals_ref=ArrayRef(
                            data.vverts_info.init_normal, slice(vvert_start, vvert_start + vgeom.n_vverts)
                        ),
                        faces_ref=ArrayRef(
                            data.vfaces_info.vverts_idx, slice(vface_start, vface_start + vgeom.n_vfaces)
                        ),
                        is_ray_target=is_ray_target,
                    )
                )
            elif i_s >= self._n_posed_vgeoms and entity.surface.vis_mode is None:
                dynamic_vvert_start = self._vgeoms_dynamic_vvert_start[vgeom.idx]
                shapes.append(
                    DynamicMesh(
                        vgeom.init_vfaces,
                        vgeom.surface,
                        ArrayRef(
                            data.dynamic_vverts.pos, slice(dynamic_vvert_start, dynamic_vvert_start + vgeom.n_vverts)
                        ),
                        vgeom.uvs,
                        seg_keys,
                        is_active=None if vgeom.active_envs_mask is None else ArrayRef(data.vgeoms.is_active, i_s),
                        is_ray_target=is_ray_target,
                        roles=ShapeRoles.of(entity.idx, is_looks=True, is_debug=False),
                    )
                )
        if geometries:
            vgeoms_cols = slice(0, self._n_drawn_vgeoms)
            shapes.insert(
                0,
                StaticMeshBatch(
                    tuple(geometries),
                    ArrayRef(data.vgeoms.pos, vgeoms_cols),
                    ArrayRef(data.vgeoms.quat, vgeoms_cols),
                    is_active=ArrayRef(data.vgeoms.is_active, vgeoms_cols) if self._has_variants else None,
                    roles=ShapeRoles.of(
                        [vgeom.entity.idx for vgeom in self._slots_vgeom[: self._n_drawn_vgeoms]],
                        is_looks=True,
                        is_debug=False,
                    ),
                ),
            )
        self._shapes = tuple(shapes)

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Show the first variant of the DynamicVertices of every visual geom again, and stop the propellers."""
        if self._n_posed_vgeoms < len(self._slots_vgeom):
            kernel_reset_vverts_variant(envs_idx, self._data.dynamic_vverts)
        if self._drones_propeller_start:
            kernel_reset_propellers_revs(envs_idx, self._data.propellers)

    def update_shapes(self, envs_idx: torch.Tensor | None, view: str) -> tuple[Shape, ...]:
        solver = self.owner
        if envs_idx is not None:
            # The visual and collision geoms follow the poses of the links, current once forward kinematics ran
            solver.update_forward_pos()
            # The poses and the dynamic vertices of every environment are placed at once, passing a selection costing as
            # much as a placement this light.
            if self._slots_vgeom:
                kernel_update_render_vgeoms(solver.dyn_state.links, self._data)
                if self._n_posed_vgeoms < len(self._slots_vgeom):
                    kernel_update_dynamic_vverts(self._data)
        return select_view_shapes(self._shapes, view)

    def update_markers(self, envs_idx: torch.Tensor) -> tuple[Shape, ...]:
        if not self._is_link_frames_shown:
            return ()
        # The frames of the links view their poses, up to date once forward kinematics ran, which hidden frames skip
        if not self.is_hidden(self._link_frames):
            self.owner.update_forward_pos()
        return (self._link_frames,)

    @property
    def link_frames(self) -> bool:
        """Whether the frames of the links are drawn (see ``KinematicVisOptions.link_frames``)."""
        return self._is_link_frames_shown

    def set_link_frames(self, link_frames: bool) -> None:
        """Draw the frames of the links, or stop drawing them (see ``KinematicVisOptions.link_frames``)."""
        self._is_link_frames_shown = link_frames and self.owner.n_links > 0
        self._context.invalidate((self,))

    @property
    def link_frame_size(self) -> float:
        """The length of the axes of the frames of the links (see ``KinematicVisOptions.link_frame_size``)."""
        return self._link_frames.length

    def set_link_frame_size(self, link_frame_size: float) -> None:
        """Set the length of the axes of the frames of the links (see ``KinematicVisOptions.link_frame_size``)."""
        # The frames keep their identity, and so their resources in the renderers, which redraw them at their new size
        object.__setattr__(self._link_frames, "length", link_frame_size)
        self._context.invalidate((self,))

    def add_propellers_rpm(self, drone: DroneEntity, propellers_rpm: torch.Tensor) -> None:
        """Add the speed of the propellers of a drone in revolutions per minute to the revolutions they accumulated.

        A drone drawing no propeller is left as is.

        Parameters
        ----------
        drone : DroneEntity
            The drone.
        propellers_rpm : torch.Tensor
            The speed of each propeller, one row per environment (see 'DroneEntity.set_propellers_rpm').
        """
        propeller_start = self._drones_propeller_start.get(drone)
        if propeller_start is None:
            return
        revs_period = 60.0 / self.owner.sim.dt
        if gs.use_zerocopy:
            revs = qd_to_torch(self._data.propellers.revs, transpose=True, copy=False)
            drone_revs = revs[:, propeller_start : propeller_start + drone.n_propellers]
            torch.remainder(drone_revs + propellers_rpm, revs_period, out=drone_revs)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            propellers_rpm = propellers_rpm.contiguous()
            kernel_add_propellers_revs(propeller_start, propellers_rpm, self._data.propellers, revs_period=revs_period)

    def set_vverts(
        self,
        vgeom: "RigidVisGeom",
        vverts: "np.typing.ArrayLike | None",
        envs_idx: "IndexType | None",
        variant: "int | str | Sequence[int | str] | None",
    ) -> None:
        """Write the batched variant of the DynamicVertices of a visual geom and show it, or show one of its variants,
        in some environments.

        See 'RigidVisGeom.set_vverts'. The subscribers of the render context are notified of the change of appearance.
        """
        solver = self.owner
        if gs.use_zerocopy and solver.n_envs == 0 and envs_idx is not None:
            gs.raise_exception("A scene without parallel environments takes no 'envs_idx'.")
        dynamic_vverts = self._data.dynamic_vverts
        i_dg = int(self._vgeoms_slot[vgeom.idx]) - self._n_posed_vgeoms
        variants = vgeom.desc.dynamic_vertices.variants
        if vverts is None:
            variants_name = tuple(variants) if isinstance(variants, dict) else ()
            variants_idx = []
            for variant_given in np.atleast_1d(np.array(variant, dtype=object)).tolist():
                if variant_given is None:
                    variant_idx = ASSET_VARIANT_IDX
                elif isinstance(variant_given, str):
                    if variant_given not in variants_name:
                        gs.raise_exception(
                            f"No variant is named '{variant_given}'. Existing names: {list(variants_name)}."
                        )
                    variant_idx = variants_name.index(variant_given)
                elif not isinstance(variant_given, (int, np.integer)) or isinstance(variant_given, bool):
                    gs.raise_exception(
                        "A variant of DynamicVertices is given by an integer position or a name, got "
                        f"{variant_given!r}."
                    )
                elif not 0 <= variant_given < len(variants):
                    gs.raise_exception(f"Variant {variant_given} is out of range for {len(variants)} variants.")
                else:
                    variant_idx = variant_given
                variants_idx.append(variant_idx)
            # A single variant is shown in every selected environment
            variants_idx = variants_idx if len(variants_idx) > 1 else variants_idx[0]
            if gs.use_zerocopy:
                variant_view = qd_to_torch(dynamic_vverts.vgeoms_variant_idx, transpose=True, copy=False)
                mask = (0, i_dg) if solver.n_envs == 0 else indices_to_mask(envs_idx, i_dg)
                assign_indexed_tensor(variant_view, mask, variants_idx)
            else:
                envs_idx = solver.sim.sanitize_envs_idx(envs_idx)
                variants_idx = broadcast_tensor(variants_idx, gs.tc_int, envs_idx.shape).contiguous()
                kernel_set_vverts_variant(envs_idx, i_dg, variants_idx, dynamic_vverts)
        else:
            vvert_start = int(self._vgeoms_batched_vvert_start[vgeom.idx])
            if vvert_start < 0:
                gs.raise_exception(
                    "Writing the vertices of a visual geom requires its batched variant, which "
                    "'gs.surfaces.DynamicVertices(is_batched=True)' allocates. Show one of its variants instead."
                )
            vverts_mask = indices_to_mask(slice(vvert_start, vvert_start + vgeom.n_vverts))
            if gs.use_zerocopy:
                variant_view = qd_to_torch(dynamic_vverts.vgeoms_variant_idx, transpose=True, copy=False)
                pos_view = qd_to_torch(dynamic_vverts.batched_pos, transpose=True, copy=False)
                if solver.n_envs == 0:
                    variant_mask, pos_mask = (0, i_dg), (0, *vverts_mask)
                else:
                    variant_mask, pos_mask = indices_to_mask(envs_idx, i_dg), indices_to_mask(envs_idx, *vverts_mask)
                assign_indexed_tensor(pos_view, pos_mask, vverts)
                assign_indexed_tensor(variant_view, variant_mask, -1)
            else:
                envs_idx = solver.sim.sanitize_envs_idx(envs_idx)
                target_shape = (envs_idx.shape[0], vgeom.n_vverts, 3)
                vverts = broadcast_tensor(vverts, gs.tc_float, target_shape, ("envs", "vverts", "xyz")).contiguous()
                kernel_set_batched_vverts(envs_idx, i_dg, vvert_start, vverts, dynamic_vverts)
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()
        self._context.invalidate((self,), change=StateChange.VISUAL)

    def get_vgeoms_pose(
        self,
        vgeoms_idx: "IndexType | None" = None,
        envs_idx: "IndexType | None" = None,
        *,
        is_quat: bool,
        copy: bool | None = True,
    ) -> torch.Tensor:
        """Return the world position or orientation of some visual geoms of the solver.

        Parameters
        ----------
        vgeoms_idx : None | int | array_like, optional
            The visual geoms, all of them if None. Defaults to None.
        envs_idx : None | int | array_like, optional
            The environments, all of them if None. Defaults to None.
        is_quat : bool
            Whether the orientation is returned, as a quaternion, rather than the position.
        copy : bool | None, optional
            Whether to return a copy, a view where zero-copy allows it for None (see 'qd_to_torch'). Defaults to True.
        """
        # The arrays of the visual geoms are refreshed first
        self.get_shapes()
        if vgeoms_idx is None:
            vgeoms_slot = self._vgeoms_slot
        elif isinstance(vgeoms_idx, int):
            vgeoms_slot = int(self._vgeoms_slot[vgeoms_idx])
        else:
            vgeoms_slot = self._vgeoms_slot[tensor_to_array(vgeoms_idx)]
        if not isinstance(vgeoms_slot, int):
            vgeoms_slot = torch.as_tensor(vgeoms_slot, device=gs.device)
        vgeoms = self._data.vgeoms
        return qd_to_torch(
            value=vgeoms.quat if is_quat else vgeoms.pos,
            row_mask=envs_idx,
            col_mask=vgeoms_slot,
            transpose=True,
            copy=copy,
        )

    def get_vverts(self, vgeoms: "Sequence[RigidVisGeom]", envs_idx: "IndexType | None" = None) -> torch.Tensor:
        """Return a copy of the world position of the vertices of some visual geoms of the solver, as drawn, those of
        each visual geom following those of the previous one."""
        vgeoms_idx = np.array([vgeom.idx for vgeom in vgeoms], dtype=gs.np_int)
        vgeoms_n_vverts = np.array([vgeom.n_vverts for vgeom in vgeoms], dtype=gs.np_int)
        vgeoms_is_dynamic = self._vgeoms_dynamic_vvert_start[vgeoms_idx] >= 0
        vverts_is_dynamic = torch.as_tensor(np.repeat(vgeoms_is_dynamic, vgeoms_n_vverts), device=gs.device)
        # The vertices posed by their visual geom are placed at once, each vertex taking the pose of its visual geom
        vgeoms_pos = self.get_vgeoms_pose(vgeoms_idx[~vgeoms_is_dynamic], envs_idx, is_quat=False)
        vgeoms_quat = self.get_vgeoms_pose(vgeoms_idx[~vgeoms_is_dynamic], envs_idx, is_quat=True)
        static_n_vverts = torch.as_tensor(vgeoms_n_vverts[~vgeoms_is_dynamic], device=gs.device)
        static_init_vverts = np.zeros((0, 3), dtype=gs.np_float)
        if not vgeoms_is_dynamic.all():
            static_init_vverts = np.concatenate(
                [vgeom.init_vverts for vgeom, is_dynamic in zip(vgeoms, vgeoms_is_dynamic) if not is_dynamic]
            )
        static_init_vverts = torch.as_tensor(static_init_vverts, dtype=gs.tc_float, device=gs.device)
        vverts = vgeoms_pos.new_empty((len(vgeoms_pos), int(vgeoms_n_vverts.sum()), 3))
        vverts[:, ~vverts_is_dynamic] = gu.transform_by_trans_quat(
            static_init_vverts,
            vgeoms_pos.repeat_interleave(static_n_vverts, dim=-2),
            vgeoms_quat.repeat_interleave(static_n_vverts, dim=-2),
        )
        if vgeoms_is_dynamic.any():
            dynamic_vverts_idx = np.concatenate(
                [
                    np.arange(vvert_start, vvert_start + n_vverts)
                    for vvert_start, n_vverts in zip(
                        self._vgeoms_dynamic_vvert_start[vgeoms_idx[vgeoms_is_dynamic]],
                        vgeoms_n_vverts[vgeoms_is_dynamic],
                    )
                ]
            )
            dynamic_vverts_idx = torch.as_tensor(dynamic_vverts_idx, device=gs.device)
            vverts[:, vverts_is_dynamic] = qd_to_torch(
                self._data.dynamic_vverts.pos, envs_idx, dynamic_vverts_idx, transpose=True
            )
        return vverts
