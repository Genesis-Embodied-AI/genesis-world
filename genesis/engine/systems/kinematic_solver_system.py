"""System of the kinematic solver, which owns its data and its forward kinematics.

The initialization of the kinematic data is shared with RigidSolverSystem, whose rigid data extends it, through the
functions of this module. Each takes the System that owns the data, which reads the model of the entities from its
own attributes once adopt_model has taken it over from the solver.
"""

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import System, register_system
from genesis.engine.solvers.kinematic_solver import _balanced_variant_mapping
from genesis.engine.solvers.rigid.abd.forward_kinematics import (
    kernel_forward_kinematics,
    kernel_update_vgeoms,
    kernel_update_vverts_for_vgeoms,
)
from genesis.engine.solvers.rigid.abd.misc import (
    kernel_init_dof_fields,
    kernel_init_entity_fields,
    kernel_init_joint_fields,
    kernel_init_link_fields,
    kernel_init_vgeom_fields,
    kernel_init_vvert_fields,
    kernel_update_heterogeneous_links_vgeom,
)
from genesis.utils.misc import qd_to_numpy, qd_to_torch

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# The model of the entities a kinematic solver has built and the options it has resolved, which the user-facing solver
# keeps resolving from its entities before the scene is built and which stay fixed once it is
KINEMATIC_MODEL = (
    "_entities",
    "entities",
    "links",
    "joints",
    "vgeoms",
    "equalities",
    "init_qpos",
    "n_entities",
    "n_entities_",
    "_n_entities",
    "n_links",
    "n_links_",
    "_n_links",
    "n_joints",
    "n_joints_",
    "_n_joints",
    "n_dofs",
    "n_dofs_",
    "_n_dofs",
    "n_qs",
    "n_qs_",
    "_n_qs",
    "n_vgeoms",
    "n_vgeoms_",
    "_n_vgeoms",
    "n_vverts",
    "n_vverts_",
    "_n_vverts",
    "n_vfaces",
    "n_vfaces_",
    "_n_vfaces",
    "n_custom_vverts",
    "n_custom_vverts_",
    "_n_custom_vverts",
    "n_custom_vfaces",
    "n_custom_vfaces_",
    "_n_custom_vfaces",
    "_n_roots",
    "n_roots_",
    "_n_trees",
    "n_trees_",
    "_links_parent_idx",
    "_links_root_idx",
    "_links_root_level",
    "_links_tree_level",
    "_links_tree_root_idx",
    "sim",
    "_sim",
    "scene",
    "_B",
    "n_envs",
    "_para_level",
    "_options",
    "_enable_collision",
    "_enable_mujoco_compatibility",
    "_requires_grad",
    "_enable_heterogeneous",
)


def adopt_model(system, model: tuple[str, ...]) -> None:
    """Take over the given model from the solver of the System, the only reads of the solver past its own build."""
    for name in model:
        setattr(system, name, getattr(system.solver, name))


class KinematicSolverSystem(System):
    """System owning the data of the kinematic solver of the scene, which reaches it through forwarding attributes.

    The Initializer has it allocate and initialize the data from the entities the solver has built. The data exists
    whether or not the solver holds an entity, since other code reads it regardless.
    """

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.kinematic_solver
        # Whether the solver holds an entity to advance, known once the scene has added its entities
        self.is_active = self.solver.is_active

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    def initialize_data(self) -> None:
        """Take over the model of the entities from the solver, then allocate and initialize the kinematic data, which
        the solver forwards to from here on."""
        adopt_model(self, KINEMATIC_MODEL)
        self.solver._system = self
        self._is_forward_pos_updated = False
        self._is_forward_vel_updated = False
        # A kinematic scene detects no collision and solves no constraint
        self.collider_data = None
        self.constraint_solver = None
        build_kinematic_static_config(self)
        create_kinematic_data_manager(self)
        init_dof_fields(self)
        init_vvert_fields(self)
        init_vgeom_fields(self)
        init_link_fields(self, dispatch_heterogeneous_vgeoms)
        init_tree_fields(self)
        init_entity_fields(self)
        init_vfaces_raycast_mask(self)
        init_vverts_state(self)

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.substep_post_coupling

    def substep_pre_coupling(self, f: int) -> None:
        """The kinematic solver integrates nothing ahead of the coupling."""

    def substep_post_coupling(self, f: int) -> None:
        """Update the forward kinematics of substep f unless the cartesian space and the velocities are up to date."""
        if not self._is_forward_pos_updated or not self._is_forward_vel_updated:
            kernel_forward_kinematics(
                self.scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True

    def sanitize_joint_sol_params(self, sol_params):
        """Leave the solver parameters of the joints as they are, since a kinematic scene solves no constraint."""
        return sol_params


@register_system(SimulatorEngine)
def create_kinematic_solver_system(scene: "Scene") -> KinematicSolverSystem:
    """Create the System owning the data of the kinematic solver, which every scene has."""
    return KinematicSolverSystem(scene)


def build_kinematic_static_config(system):
    # Static config with all physics disabled
    system.rigid_config = array_class.RigidSimStaticConfig(
        backend=gs.backend,
        para_level=system.sim._para_level,
        requires_grad=False,
        use_hibernation=False,
        batch_links_info=system._options.batch_links_info,
        batch_dofs_info=system._options.batch_dofs_info,
        batch_joints_info=system._options.batch_joints_info,
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


def create_kinematic_data_manager(system):
    system.data_manager = array_class.DataManager(system, kinematic_only=True)
    system.rigid_info = system.data_manager.rigid_info
    system._rigid_adjoint_cache = system.data_manager.rigid_adjoint_cache
    system.dyn_info = system.data_manager.dyn_info
    system.dyn_state = system.data_manager.dyn_state
    system.kinematics_scratch = system.data_manager.kinematics_scratch


def init_dof_fields(system):
    joints = system.joints
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
            system.dyn_state,
            system.dyn_info,
            system.rigid_info,
            system.rigid_config,
        )

    system.dyn_state.dofs.force.fill(0)


def init_tree_fields(system):
    """Initialize the fields describing the kinematic roots and trees (see roots_link_idx and trees_root_idx in
    array_class.py).

    The links come parent first and each branch occupies a contiguous index range, so the dofs of a tree form one
    contiguous range and the trees are disjoint in dof space.
    """
    # A static link belongs to no tree, and a scene without any tree keeps its padded tree slot empty (see
    # trees_root_idx in array_class.py)
    links_tree_idx = np.full(system.n_links_, -1, dtype=gs.np_int)
    if system._n_roots:
        roots_link_idx, links_root_rank = np.unique(system._links_root_idx, return_inverse=True)
        roots_link_end = np.zeros(system._n_roots, dtype=gs.np_int)
        np.maximum.at(roots_link_end, links_root_rank, np.arange(1, system.n_links + 1, dtype=gs.np_int))
        system.rigid_info.roots_link_idx.from_numpy(roots_link_idx)
        system.rigid_info.links_root_end.from_numpy(roots_link_end[links_root_rank])
    if system._n_trees:
        links_n_dofs = np.array([link.n_dofs for link in system.links], dtype=gs.np_int)
        links_dof_start = np.array([link.dof_start for link in system.links], dtype=gs.np_int)
        links_dof_end = np.array([link.dof_end for link in system.links], dtype=gs.np_int)
        tree_links = np.flatnonzero(system._links_tree_root_idx >= 0)
        trees_root_idx, links_tree_rank = np.unique(system._links_tree_root_idx[tree_links], return_inverse=True)
        trees_n_dofs = np.zeros(system._n_trees, dtype=gs.np_int)
        np.add.at(trees_n_dofs, links_tree_rank, links_n_dofs[tree_links])
        # A dof-less link carries no dof range of its own
        dof_links = np.flatnonzero(links_n_dofs[tree_links])
        trees_dof_start = np.full(system._n_trees, system.n_dofs, dtype=gs.np_int)
        np.minimum.at(trees_dof_start, links_tree_rank[dof_links], links_dof_start[tree_links][dof_links])
        trees_dof_end = np.zeros(system._n_trees, dtype=gs.np_int)
        np.maximum.at(trees_dof_end, links_tree_rank[dof_links], links_dof_end[tree_links][dof_links])
        if (trees_dof_end - trees_dof_start != trees_n_dofs).any():
            gs.raise_exception("The dofs of a kinematic tree must be contiguous.")
        trees_link_end = np.zeros(system._n_trees, dtype=gs.np_int)
        np.maximum.at(trees_link_end, links_tree_rank, tree_links + 1)
        trees_n_links = np.zeros(system._n_trees, dtype=gs.np_int)
        np.add.at(trees_n_links, links_tree_rank, 1)
        trees_order = np.argsort(trees_dof_start)
        trees_rank = np.empty(system._n_trees, dtype=gs.np_int)
        trees_rank[trees_order] = np.arange(system._n_trees, dtype=gs.np_int)
        links_tree_idx[tree_links] = trees_rank[links_tree_rank]
        system.rigid_info.trees_root_idx.from_numpy(trees_root_idx[trees_order])
        system.rigid_info.trees_link_end.from_numpy(trees_link_end[trees_order])
        system.rigid_info.trees_n_links.from_numpy(trees_n_links[trees_order])
        system.rigid_info.trees_dof_start.from_numpy(trees_dof_start[trees_order])
        system.rigid_info.trees_n_dofs.from_numpy(trees_n_dofs[trees_order])
    else:
        system.rigid_info.trees_root_idx.fill(0)
        system.rigid_info.trees_link_end.fill(0)
    system.rigid_info.links_tree_idx.from_numpy(links_tree_idx)

    # Fill the level tables of every tree and root for the level sweep (see trees_levels_links_idx in
    # array_class.py). A scene without any tree keeps its padded tree slot at zero levels.
    if system._n_roots and system.rigid_config.enable_level_sweep:
        for links_group_idx, links_group_level, n_groups, groups_fields in (
            (
                links_tree_idx,
                system._links_tree_level,
                system.n_trees_,
                (
                    system.rigid_info.trees_level_start,
                    system.rigid_info.trees_n_levels,
                    system.rigid_info.trees_levels_links_idx,
                    None,
                    system.rigid_info.trees_levels_links_end,
                ),
            ),
            (
                links_root_rank,
                system._links_root_level,
                system._n_roots,
                (
                    system.rigid_info.roots_level_start,
                    system.rigid_info.roots_n_levels,
                    system.rigid_info.roots_levels_links_idx,
                    system.rigid_info.roots_levels_links_start,
                    system.rigid_info.roots_levels_links_end,
                ),
            ),
        ):
            group_links = np.arange(system.n_links, dtype=gs.np_int)[links_group_idx >= 0]
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
        child_links = np.arange(system.n_links, dtype=gs.np_int)[system._links_parent_idx >= 0]
        links_child_start = np.zeros(system.n_links_ + 1, dtype=gs.np_int)
        np.cumsum(
            np.bincount(system._links_parent_idx[child_links], minlength=system.n_links_), out=links_child_start[1:]
        )
        system.rigid_info.links_child_start.from_numpy(links_child_start)
        if len(child_links):
            system.rigid_info.links_child_idx.from_numpy(
                child_links[np.lexsort((-child_links, system._links_parent_idx[child_links]))]
            )


def init_link_fields(system, dispatch_vgeoms):
    if system.links:
        links = system.links
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
            system.dyn_state,
            system.dyn_info,
            system.rigid_info,
            system.rigid_config,
        )

    if system.joints:
        joints = system.joints
        joints_sol_params = np.array([joint.desc.sol_params for joint in joints], dtype=gs.np_float)
        joints_sol_params = system.sanitize_joint_sol_params(joints_sol_params)

        kernel_init_joint_fields(
            np.array([joint.q_start for joint in joints], dtype=gs.np_int),
            np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
            np.array([joint.q_end for joint in joints], dtype=gs.np_int),
            np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
            np.array([joint.type for joint in joints], dtype=gs.np_int),
            joints_sol_params,
            np.array([joint.pos for joint in joints], dtype=gs.np_float),
            system.dyn_info,
            system.rigid_config,
        )

    dispatch_heterogeneous_kinematics(system)

    # Set initial qpos
    system.qpos = system.rigid_info.qpos
    system.qpos0 = system.rigid_info.qpos0
    if system.n_qs > 0:
        init_qpos = np.tile(np.expand_dims(system.init_qpos, -1), (1, system._B))

        # Dispatch per-variant init_qpos for heterogeneous entities
        for entity in system.entities:
            if not entity._desc.variants:
                continue
            n_variants = len(entity._desc.variants)
            variant_idx = _balanced_variant_mapping(n_variants, system._B)
            q_s, q_e = entity.q_start, entity.q_start + entity.n_qs
            for i_b in range(system._B):
                init_qpos[q_s:q_e, i_b] = entity._desc.variants[variant_idx[i_b]].init_qpos

        system.qpos0.from_numpy(init_qpos)
        dofs_limit = np.broadcast_to(
            qd_to_numpy(system.dyn_info.dofs.limit, transpose=True), (system._B, system.n_dofs_, 2)
        )
        is_init_qpos_out_of_bounds = False
        for joint in system.joints:
            if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                is_init_qpos_out_of_bounds |= (dofs_limit[:, joint.dof_start, 0] > init_qpos[joint.q_start]).any()
                is_init_qpos_out_of_bounds |= (init_qpos[joint.q_start] > dofs_limit[:, joint.dof_start, 1]).any()
        if is_init_qpos_out_of_bounds:
            gs.logger.warning("Neutral robot position (qpos0) exceeds joint limits.")
        system.qpos.from_numpy(init_qpos)

    system.links_T = system.rigid_info.links_T

    # Dispatch heterogeneous variant vgeom ranges per-environment
    dispatch_vgeoms(system)


def dispatch_heterogeneous_kinematics(system):
    """Write the link frames and the joint and dof parameters of each heterogeneous variant to the envs carrying it.

    The init kernels broadcast those of the primary to every environment, while heterogeneous simulation batches
    the link, joint and dof info (see 'build').
    """
    entities = [entity for entity in system._entities if entity._desc.variants]
    if not entities:
        return

    # Field layout, batch second, since the arrays are written back as fields
    links_pos = qd_to_numpy(system.dyn_info.links.pos, transpose=False, copy=True)
    links_quat = qd_to_numpy(system.dyn_info.links.quat, transpose=False, copy=True)
    joints_pos = qd_to_numpy(system.dyn_info.joints.pos, transpose=False, copy=True)
    joints_sol_params = qd_to_numpy(system.dyn_info.joints.sol_params, transpose=False, copy=True)
    dofs_info = system.dyn_info.dofs
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
        variants_links = [entity._desc.links, *(variant.links for variant in entity._desc.variants[1:])]
        envs_variant_idx = _balanced_variant_mapping(len(variants_links), system._B)
        links_slice = slice(entity.link_start, entity.link_end)
        variants_links_pos = np.stack([[l_desc.pos for l_desc in links] for links in variants_links])
        variants_links_quat = np.stack([[l_desc.quat for l_desc in links] for links in variants_links])
        links_pos[links_slice] = variants_links_pos[envs_variant_idx].swapaxes(0, 1)
        links_quat[links_slice] = variants_links_quat[envs_variant_idx].swapaxes(0, 1)

        # A file morph drops its joints without degrees of freedom, so a fixed one may leave the entity without any
        if entity.n_joints == 0:
            continue
        variants_joints = [[asdict(j_desc) for l_desc in links for j_desc in l_desc.joints] for links in variants_links]
        joints_slice = slice(entity.joint_start, entity.joint_end)
        variants_joints_pos = np.stack([[j_fields["pos"] for j_fields in joints] for joints in variants_joints])
        joints_pos[joints_slice] = variants_joints_pos[envs_variant_idx].swapaxes(0, 1)
        variants_sol_params = np.stack([[j_fields["sol_params"] for j_fields in joints] for joints in variants_joints])
        variants_sol_params = system.sanitize_joint_sol_params(variants_sol_params).reshape(variants_sol_params.shape)
        joints_sol_params[joints_slice] = variants_sol_params[envs_variant_idx].swapaxes(0, 1)
        dofs_slice = slice(entity.dof_start, entity.dof_end)
        for dofs_value, (_, name) in zip(dofs_values, dofs_fields):
            variants_value = np.stack(
                [np.concatenate([j_fields[name] for j_fields in joints]) for joints in variants_joints]
            )
            dofs_value[dofs_slice] = variants_value[envs_variant_idx].swapaxes(0, 1)

    system.dyn_info.links.pos.from_numpy(links_pos)
    system.dyn_info.links.quat.from_numpy(links_quat)
    system.dyn_info.joints.pos.from_numpy(joints_pos)
    system.dyn_info.joints.sol_params.from_numpy(joints_sol_params)
    for (field, _), dofs_value in zip(dofs_fields, dofs_values):
        field.from_numpy(dofs_value)

    # The state of a fixed root link is set at init alone (see kernel_init_link_fields), so it takes the frame of
    # the variant its environment carries too
    fixed_roots_idx = [entity.link_start for entity in entities if entity.base_link.is_fixed]
    if fixed_roots_idx:
        links_state_pos = qd_to_numpy(system.dyn_state.links.pos, transpose=False, copy=True)
        links_state_quat = qd_to_numpy(system.dyn_state.links.quat, transpose=False, copy=True)
        links_state_pos[fixed_roots_idx] = links_pos[fixed_roots_idx]
        links_state_quat[fixed_roots_idx] = links_quat[fixed_roots_idx]
        system.dyn_state.links.pos.from_numpy(links_state_pos)
        system.dyn_state.links.quat.from_numpy(links_state_quat)


def dispatch_heterogeneous_vgeoms(system):
    """Override per-link vgeom ranges for heterogeneous variants, which rigid_dispatch_heterogeneous_vgeoms extends."""
    for link in system.links:
        if link._variant_vgeom_ranges is None:
            continue

        n_variants = len(link._variant_vgeom_ranges)
        variant_idx = _balanced_variant_mapping(n_variants, system._B)

        vgeom_starts = np.array([link._variant_vgeom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
        vgeom_ends = np.array([link._variant_vgeom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)

        kernel_update_heterogeneous_links_vgeom(link.idx, vgeom_starts, vgeom_ends, system.dyn_info)

        for vgeom in link.vgeoms:
            active_envs_mask = (vgeom_starts <= vgeom.idx) & (vgeom.idx < vgeom_ends)
            vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
            (vgeom.active_envs_idx,) = np.where(active_envs_mask)


def init_vvert_fields(system):
    if system.n_vverts == 0:
        return

    vgeoms = system.vgeoms
    vverts = np.concatenate([vg.init_vverts for vg in vgeoms], dtype=gs.np_float)
    vnormals = np.concatenate([vg.init_vnormals for vg in vgeoms], dtype=gs.np_float)
    vfaces = np.concatenate([vg.init_vfaces + vg._vvert_start for vg in vgeoms], dtype=gs.np_int)
    vverts_vgeom_idx = np.concatenate([np.full(vg.n_vverts, vg.idx) for vg in vgeoms], dtype=gs.np_int)
    vverts_state_idx = np.full(system.n_vverts, -1, dtype=gs.np_int)
    for entity in system._entities:
        if not entity._morph.enable_custom_vverts:
            continue
        entity_custom_offset = entity._custom_vvert_start - entity._vvert_start
        for vgeom in entity.vgeoms:
            local = np.arange(vgeom.n_vverts, dtype=gs.np_int)
            vverts_state_idx[vgeom._vvert_start + local] = vgeom._vvert_start + entity_custom_offset + local
    kernel_init_vvert_fields(
        vverts_vgeom_idx, vverts_state_idx, vverts, vfaces, vnormals, system.dyn_info, system.rigid_config
    )


def init_vgeom_fields(system):
    if system.n_vgeoms > 0:
        vgeoms = system.vgeoms
        kernel_init_vgeom_fields(
            np.array([vgeom.link.idx for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.vface_start for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.vvert_end for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.vface_end for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.init_pos for vgeom in vgeoms], dtype=gs.np_float),
            np.array([vgeom.init_quat for vgeom in vgeoms], dtype=gs.np_float),
            np.array([vgeom.desc.vmesh.color for vgeom in vgeoms], dtype=gs.np_float),
            system.dyn_info,
            system.rigid_config,
        )


def init_vfaces_raycast_mask(system):
    """Fit the static per-vface visual-raycasting opt-in mask; see the vfaces_raycast_mask property.

    Sized over the padded vgeom count so the gather also covers a solver holding no vgeom at all, whose vface
    fields still carry one padding slot: nothing opts in there, so every entry stays 0.
    """
    vgeom_enabled = torch.zeros(system.n_vgeoms_, dtype=gs.tc_bool, device=gs.device)
    for entity in system.entities:
        if not entity.material.use_visual_raycasting:
            continue
        for vgeom in entity.vgeoms:
            vgeom_enabled[vgeom.idx] = 1
    system._vfaces_raycast_mask = vgeom_enabled[qd_to_torch(system.dyn_info.vfaces.vgeom_idx)]


def init_entity_fields(system):
    if system._entities:
        entities = system._entities
        kernel_init_entity_fields(
            np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
            np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
            np.array([entity.link_start for entity in entities], dtype=gs.np_int),
            np.array([entity.link_end for entity in entities], dtype=gs.np_int),
            np.array([0 for entity in entities], dtype=gs.np_int),
            np.array([0 for entity in entities], dtype=gs.np_int),
            np.array([0.0 for entity in entities], dtype=gs.np_float),
            np.array([False for entity in entities], dtype=gs.np_bool),
            system.dyn_state,
            system.dyn_info,
            system.rigid_info,
            system.rigid_config,
        )


def init_vverts_state(system):
    # Seed vverts_state.pos with one FK pass so opt-in entities show their initial pose from frame 0 even before
    # the user touches the buffer with set_vverts.
    if system.n_custom_vverts == 0:
        return
    opt_in_vgeoms_idx = np.array(
        [vgeom.idx for entity in system._entities if entity._morph.enable_custom_vverts for vgeom in entity.vgeoms],
        dtype=gs.np_int,
    )
    if opt_in_vgeoms_idx.size == 0:
        return
    kernel_update_vgeoms(system.dyn_state, system.dyn_info, system.rigid_config)
    kernel_update_vverts_for_vgeoms(opt_in_vgeoms_idx, system.dyn_state, system.dyn_info, system.rigid_config)
