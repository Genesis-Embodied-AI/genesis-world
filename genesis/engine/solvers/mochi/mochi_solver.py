# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math
import sys
from typing import TYPE_CHECKING, NamedTuple

import igl
import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.mochi_entity import MochiEntity, MochiSoftEntity
from genesis.engine.materials import Mochi
from genesis.engine.states.solvers import MochiSolverState
from genesis.options.morphs import Morph
from genesis.options.solvers import MochiOptions
from genesis.utils import array_class
from genesis.utils.misc import get_gpu_shared_tile_sizes, qd_to_numpy, qd_to_torch, tensor_to_array
from genesis.utils.sdf import SDF

from ..base_solver import GravityMixin, StateChange, Subscriber, TimeBasedMixin
from ..kinematic_solver import KinematicSolver
from ..rigid.abd.accessor import (
    kernel_control_dofs_force,
    kernel_control_dofs_position,
    kernel_control_dofs_position_velocity,
    kernel_control_dofs_velocity,
    kernel_get_dofs_control_force,
    kernel_get_kinematic_state,
    kernel_set_dofs_armature,
    kernel_set_dofs_damping,
    kernel_set_dofs_kp,
    kernel_set_dofs_kv,
    kernel_set_dofs_limit,
    kernel_set_dofs_stiffness,
)
from ..rigid.abd.forward_kinematics import kernel_forward_kinematics, kernel_update_geoms
from ..rigid.abd.misc import (
    kernel_bit_reduction,
    kernel_init_entity_fields,
    kernel_init_geom_fields,
    kernel_init_link_dynamics,
    kernel_init_vert_fields,
)
from .articulated import kernel_assemble_joints, kernel_project_links_residual, kernel_update_conv_weights
from .colliders import query_collider
from .contact import (
    TRIANGLE_QUADRATURES,
    build_geom_samples,
    kernel_broadphase_pairs,
    kernel_conservative_bounds,
    kernel_contact_eval,
    kernel_count_contact_records,
    kernel_gather_contact_records,
    kernel_init_mochi_fields,
    kernel_pairs_to_blocks,
    kernel_set_links_pair_enabled,
    kernel_zero_assembly,
)
from .data import (
    COLLIDER_TYPE,
    FRICTION_MODEL,
    INTEGRATOR,
    LINEAR_TOLERANCE,
    LINESEARCH,
    N_HISTORY,
    MochiInfo,
    MochiState,
    MochiStaticConfig,
    get_mochi_contact_state,
    get_mochi_hit_readback,
    get_mochi_info,
    get_mochi_island_state,
    get_mochi_soft_info,
    get_mochi_soft_state,
    get_mochi_state,
)
from .equalities import (
    get_mochi_equalities_info,
    get_mochi_equalities_state,
    kernel_assemble_equalities,
    kernel_equalities_stage_start,
)
from .integration import (
    kernel_post_stage,
    kernel_reset_history,
    kernel_step_start,
    kernel_store_stage_start_poses,
)
from .islands import kernel_build_islands, kernel_cholesky_solve_islands
from .kinematics import kernel_update_geom_aabbs, kernel_update_kinematics
from .linear_solver import (
    kernel_condense_dense,
    kernel_pcg_any_active,
    kernel_pcg_init,
    kernel_pcg_iter,
)
from .linear_solver_tiled import kernel_cholesky_solve_tiled
from .newton import (
    kernel_any_active,
    kernel_apply_increment,
    kernel_convergence_check,
    kernel_linesearch_begin,
    kernel_linesearch_decide,
    kernel_reset_newton,
    kernel_residual_norms,
    kernel_store_initial_norms,
    kernel_update_linear_tolerance,
)
from .rigid_assembly import kernel_assemble_links
from .sample_tree import build_sample_tree
from .soft import (
    ENTITY_PARAMS,
    SOFT_KIND_ROD,
    SOFT_KIND_SHELL,
    SOFT_KIND_SOLID,
    build_soft_samples,
    kernel_assemble_attachments,
    kernel_attachments_stage_start,
    kernel_init_rod_fields,
    kernel_init_shell_fields,
    kernel_init_soft_fields,
    kernel_pc_collider_eval,
    kernel_pc_hash_build,
    kernel_rod_apply_increment,
    kernel_rod_assemble,
    kernel_rod_get_state_render,
    kernel_rod_init_render,
    kernel_rod_post_stage,
    kernel_rod_step_start,
    kernel_rod_store_ls_ref,
    kernel_rod_update_conv_weights,
    kernel_shell_assemble,
    kernel_shell_stage_start,
    kernel_soft_apply_increment,
    kernel_soft_assemble_elements,
    kernel_soft_broadphase,
    kernel_soft_collider_eval,
    kernel_soft_condense_dense,
    kernel_soft_conservative_bounds,
    kernel_soft_contact_eval,
    kernel_soft_dirichlet,
    kernel_soft_get_entity_state,
    kernel_soft_get_state,
    kernel_soft_get_state_render,
    kernel_soft_get_vertices_field,
    kernel_soft_init_render,
    kernel_soft_pairs_to_blocks,
    kernel_soft_post_stage,
    kernel_soft_set_entity_contact_params,
    kernel_soft_set_links_pair_enabled,
    kernel_soft_set_pair_enabled,
    kernel_soft_set_state,
    kernel_soft_set_vertices_fixed,
    kernel_soft_set_vertices_positions,
    kernel_soft_set_vertices_target,
    kernel_soft_set_vertices_velocities,
    kernel_soft_step_start,
    kernel_soft_store_ls_ref,
    kernel_soft_update_conv_weights,
    kernel_soft_zero_assembly,
    kernel_tet_tree_refit,
)
from .soft_materials import ELASTIC_MODEL_BY_NAME
from .step_graph import kernel_step_graph
from .step_monolith import kernel_step_monolith
from .tet_tree import build_tet_tree

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# Conjugate gradient iterations between two host-side checks of the per-environment convergence flags.
PCG_CHECK_PERIOD = 8

_COLLIDER_TYPE_BY_NAME = {
    "none": COLLIDER_TYPE.NONE,
    "plane": COLLIDER_TYPE.PLANE,
    "sphere": COLLIDER_TYPE.SPHERE,
    "box": COLLIDER_TYPE.BOX,
    "sdf": COLLIDER_TYPE.GRID,
}
# Rest-shape signed distance grid of a deformable collider: cell size relative to the mean boundary edge length,
# padding around the rest bounds and minimum resolution per axis.
SOFT_SDF_CELL_RATIO = 0.25
SOFT_SDF_PADDING = 5e-3
SOFT_SDF_MIN_RES = 6

_AUTO_COLLIDER_TYPE_BY_GEOM_TYPE = {
    gs.GEOM_TYPE.PLANE: COLLIDER_TYPE.PLANE,
    gs.GEOM_TYPE.SPHERE: COLLIDER_TYPE.SPHERE,
    gs.GEOM_TYPE.BOX: COLLIDER_TYPE.BOX,
}


class SoftCSRLayout(NamedTuple):
    """Scalar compressed sparse row (CSR) sparsity of the deformable Hessian and the CSR index tables of the element
    blocks (see `MochiSolver._soft_csr_layout`)."""

    start: np.ndarray
    col: np.ndarray
    elems: np.ndarray
    shell: np.ndarray
    rod_elems: np.ndarray
    rod_stencils: np.ndarray
    elems_block: np.ndarray
    shell_block: np.ndarray


class RodBandLayout(NamedTuple):
    """Band ordering of the open rods' degrees of freedom and the rod ranges per entity (see
    `MochiSolver._rod_band_layout`)."""

    dofs_row: np.ndarray
    rows_dof: np.ndarray
    rows_entity: np.ndarray
    band_start: np.ndarray
    band_n: np.ndarray
    rod_elem_start: np.ndarray
    rod_elem_end: np.ndarray
    rod_stencil_start: np.ndarray
    rod_stencil_end: np.ndarray


def _next_power_of_two(n):
    return 1 << max(0, int(n) - 1).bit_length()


class MochiSolver(GravityMixin, TimeBasedMixin, KinematicSolver):
    """
    Fully-implicit multi-physics solver with smooth penalty contact.

    Every substep solves one nonlinear system over the generalized coordinates of all rigid and articulated bodies and
    the vertex positions of all deformable bodies, whose residual is the gradient of an incremental potential (inertia,
    gravity, damping, elasticity and the contact penalty with its friction and damping terms) and whose Hessian is
    assembled from per-link, per-contact-pair and per-tetrahedron blocks. The solve is a damped Newton iteration with a
    line search; contact is re-detected at every iterate.
    """

    material_cls = Mochi.Base
    _entity_classes = ((Morph, MochiEntity),)

    def __init__(self, scene: "Scene", sim: "Simulator", options: MochiOptions) -> None:
        super().__init__(scene, sim, options)
        self._options: MochiOptions = options
        self.sdf: SDF | None = None
        self._errno = None
        self._is_contacts_recorded = False
        # Environments whose kinematic state was changed from outside the solver; their multistep history is rebuilt
        # at the start of the next step. Allocated at build, once the batch size is known.
        self._external_state_dirty_mask = None
        self._external_state_subscriber = Subscriber(
            frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS}), callback=self._on_external_state_change
        )
        # Deformable bodies live outside the kinematic tree.
        self._soft_entities = gs.List()
        self.soft_info = None
        self.soft_state = None
        self._soft_vverts_render = None
        self._soft_vverts_vert_idx = None

    # ------------------------------------------------------------------------------------
    # ----------------------------------- add_entity -------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(self, idx, material, morph, surface, visualize_contact=False, name=None, desc=None):
        morphs = desc.morphs if desc is not None else ((morph,) if isinstance(morph, Morph) else tuple(morph))
        if len(morphs) > 1:
            gs.raise_exception("Heterogeneous morphs are not supported by the MochiSolver.")
        if isinstance(morphs[0], (gs.morphs.Terrain, gs.morphs.USD, gs.morphs.Drone)):
            gs.raise_exception(f"Morph {type(morphs[0]).__name__} is not supported by the MochiSolver.")

        if isinstance(material, (gs.materials.Mochi.Elastic, gs.materials.Mochi.Shell, gs.materials.Mochi.Rod)):
            is_rod = isinstance(material, gs.materials.Mochi.Rod)
            if not is_rod and not isinstance(
                morph, (gs.morphs.Box, gs.morphs.Sphere, gs.morphs.Cylinder, gs.morphs.Mesh)
            ):
                gs.raise_exception(
                    f"Morph {type(morph).__name__} is not supported for deformable Mochi bodies (Box, Sphere, "
                    "Cylinder or Mesh expected)."
                )
            if is_rod and not isinstance(morph, gs.morphs.Rod):
                gs.raise_exception("A `gs.materials.Mochi.Rod` material requires a `gs.morphs.Rod` morph.")
            entity = MochiSoftEntity(
                scene=self._scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                idx=idx,
                idx_in_solver=self.n_soft_entities,
                v_start=self.n_soft_verts,
                vvert_start=self.n_soft_vverts,
                vface_start=self.n_soft_vfaces,
                name=name,
            )
            self._soft_entities.append(entity)
            return entity

        return super().add_entity(idx, material, morph, surface, visualize_contact, name, desc)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ build -----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_attachments(self):
        """Number of rigid-deformable vertex attachments registered on the deformable entities."""
        return sum(len(att["verts_idx"]) for entity in self._soft_entities for att in entity.attachments)

    def _resolve_para_level(self):
        # On a CPU backend initialized with more than one quadrants thread (`gs.init(cpu_threads=...)` or
        # QD_NUM_THREADS), the solver's item loops run in parallel even at one environment. The scene-global level
        # and the other solvers are unaffected: this solver builds its own static configs.
        if gs.backend == gs.cpu and gs.cpu_threads_effective > 1:
            return max(self.sim._para_level, gs.PARA_LEVEL.PARTIAL)
        return self.sim._para_level

    def build(self):
        if self.is_active:
            other_solvers = [
                solver
                for solver in self.sim.solvers
                if solver is not self and solver.is_active and type(solver) is not KinematicSolver
            ]
            if other_solvers:
                gs.raise_exception(
                    "MochiSolver cannot run alongside other physics solvers: "
                    f"{[type(solver).__name__ for solver in other_solvers]}."
                )
            if self.sim.requires_grad:
                gs.raise_exception("MochiSolver does not support differentiable simulation.")

        self._n_geoms = self.n_geoms
        self._n_cells = self.n_cells
        self._n_verts = self.n_verts
        self._n_free_verts = self.n_free_verts
        self._n_fixed_verts = self.n_fixed_verts
        self._n_faces = self.n_faces
        self._n_edges = self.n_edges
        self._geoms = self.geoms

        self.n_geoms_ = max(1, self.n_geoms)
        self.n_cells_ = max(1, self.n_cells)
        self.n_verts_ = max(1, self.n_verts)
        self.n_faces_ = max(1, self.n_faces)
        self.n_edges_ = max(1, self.n_edges)
        self.n_free_verts_ = max(1, self.n_free_verts)
        self.n_fixed_verts_ = max(1, self.n_fixed_verts)

        super().build()
        self._init_default_armature()
        self._external_state_dirty_mask = np.zeros((self._B,), dtype=bool)
        # Which of the two linear arms have environments to solve; refreshed at every substep by _select_linear_arms.
        self._has_dense_envs = False
        self._has_pcg_envs = False

        if not self.is_active:
            return

        self.n_soft_entities_ = max(1, self.n_soft_entities)
        self.n_soft_verts_ = max(1, self.n_soft_verts)
        self.n_soft_elems_ = max(1, self.n_soft_elems)
        self.n_shell_elems_ = max(1, self.n_shell_elems)
        self.n_rod_elems_ = max(1, self.n_rod_elems)
        self.n_rod_stencils_ = max(1, self.n_rod_stencils)
        self.n_attachments_ = max(1, self.n_attachments)
        self.n_dofs_total_ = max(1, self.n_dofs_total)

        if gs.qd_float == qd.f32 and any(
            entity.material.penalty_coefficient >= 1e8 for entity in (*self._entities, *self._soft_entities)
        ):
            gs.logger.warning(
                "MochiSolver runs in single precision with a contact stiffness of 1e8 Pa/m or more: the Newton system "
                "is ill-conditioned and contact accuracy degrades. Consider `gs.init(precision='64')`."
            )

        self._init_vert_fields()
        self._init_geom_fields()
        kernel_update_geoms(
            self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, True
        )
        self._init_mochi()
        self._init_soft()
        self._step_kernel = self._resolve_step_kernel()
        # loop counters of the graph step kernel (the same physical arrays every launch)
        self._newton_counter = qd.ndarray(qd.i32, shape=())
        self._round_counter = qd.ndarray(qd.i32, shape=())
        self._pcg_counter = qd.ndarray(qd.i32, shape=())
        # The contact points recorded for readback are allocated at the first readback (see `_record_contacts`).
        self.hit_readback = get_mochi_hit_readback(self, 1, 1, 1, 1)
        self._is_hit_readback_allocated = False
        self._links_entity_idx = np.array([link.entity.idx for link in self.links] or [-1], dtype=gs.np_int)
        self._geoms_link_idx = np.array([geom.link.idx for geom in self.geoms] or [-1], dtype=gs.np_int)
        self._soft_entities_idx = np.array([entity.idx for entity in self._soft_entities] or [-1], dtype=gs.np_int)
        self.island_state = get_mochi_island_state(self, *self._island_nodes())
        self._init_sdf()
        self.subscribe(self._external_state_subscriber)
        kernel_reset_history(
            self._scene._envs_idx, self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.rigid_config
        )
        self._forward_kinematics()

    def _create_data_manager(self):
        super()._create_data_manager()
        # The kinematic data manager allocates the collision-related leaves as scalar dummies; the contact model
        # needs the real geoms and verts.
        data_manager = self.data_manager
        data_manager.dyn_info = dataclasses.replace(
            data_manager.dyn_info,
            geoms=array_class.get_geoms_info(self),
            verts=array_class.get_verts_info(self),
            faces=array_class.get_faces_info(self),
            edges=array_class.get_edges_info(self),
        )
        data_manager.dyn_state = dataclasses.replace(
            data_manager.dyn_state,
            geoms=array_class.get_geoms_state(self),
            free_verts=array_class.get_free_verts_state(self),
            fixed_verts=array_class.get_fixed_verts_state(self),
        )
        self.dyn_info = data_manager.dyn_info
        self.dyn_state = data_manager.dyn_state
        self.geoms_init_AABB = array_class.V_VEC(3, dtype=gs.qd_float, shape=(self.n_geoms_, 8))
        self._errno = data_manager.errno

    def _init_default_armature(self):
        """Fill in the default rotor inertia of the rotor joints whose armature the model leaves unset.

        A rotor joint is the single revolute or prismatic joint of a link. Only a scene or robot description file yields
        one, and its morph states the default armature (see 'KinematicVariantDescription').
        """
        dofs_idx, dofs_default = [], []
        for entity in self._entities:
            for link in entity.links:
                if link.n_dofs == 1 and link.joints[0].type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                    if entity.main_morph.default_armature:
                        dofs_idx.append(link.dof_start)
                        dofs_default.append(entity.main_morph.default_armature)
        if not dofs_idx:
            return
        dofs_idx, dofs_default = np.array(dofs_idx), np.array(dofs_default)
        # Field layout, batch last, since the array is written back as a field
        dofs_armature = qd_to_numpy(self.dyn_info.dofs.armature, transpose=False, copy=True)
        is_default = (np.atleast_2d(dofs_armature[dofs_idx].T) <= 0.0).all(axis=0)
        if self._options.batch_dofs_info:
            dofs_armature[dofs_idx[is_default]] = dofs_default[is_default, None]
        else:
            dofs_armature[dofs_idx[is_default]] = dofs_default[is_default]
        self.dyn_info.dofs.armature.from_numpy(dofs_armature)

    def _init_link_fields(self):
        if self.links:
            links = self.links
            kernel_init_link_dynamics(
                np.array([link.desc.invweight for link in links], dtype=gs.np_float),
                np.array([link.desc.inertial_pos for link in links], dtype=gs.np_float),
                np.array([link.desc.inertial_quat for link in links], dtype=gs.np_float),
                np.array([link.desc.inertia for link in links], dtype=gs.np_float),
                np.array([link.desc.mass for link in links], dtype=gs.np_float),
                self.dyn_info,
            )
        super()._init_link_fields()

    def _init_vert_fields(self):
        if self.n_verts > 0:
            geoms = self.geoms
            kernel_init_vert_fields(
                np.concatenate([np.full(geom.n_verts, geom.idx) for geom in geoms], dtype=gs.np_int),
                np.concatenate(
                    [np.arange(geom.verts_state_start, geom.verts_state_start + geom.n_verts) for geom in geoms],
                    dtype=gs.np_int,
                ),
                np.concatenate([geom.init_verts for geom in geoms], dtype=gs.np_float),
                np.concatenate([geom.init_faces + geom.vert_start for geom in geoms], dtype=gs.np_int),
                np.concatenate([geom.init_edges + geom.vert_start for geom in geoms], dtype=gs.np_int),
                np.concatenate([geom.init_normals for geom in geoms], dtype=gs.np_float),
                np.concatenate([geom.init_center_pos for geom in geoms], dtype=gs.np_float),
                np.concatenate(
                    [np.full(geom.n_verts, geom.is_fixed and not geom.entity._batch_fixed_verts) for geom in geoms],
                    dtype=gs.np_bool,
                ),
                self.dyn_info,
                self.rigid_config,
            )

    def _init_geom_fields(self):
        if self.n_geoms == 0:
            return
        geoms = self.geoms
        geoms_center = []
        for geom in geoms:
            tmesh = geom.mesh.trimesh
            geoms_center.append(tmesh.center_mass if tmesh.is_watertight else np.mean(tmesh.vertices, axis=0))
        kernel_init_geom_fields(
            np.array([geom.link.idx for geom in geoms], dtype=gs.np_int),
            np.array([geom.vert_start for geom in geoms], dtype=gs.np_int),
            np.array([geom.face_start for geom in geoms], dtype=gs.np_int),
            np.array([geom.edge_start for geom in geoms], dtype=gs.np_int),
            np.array([geom.verts_state_start for geom in geoms], dtype=gs.np_int),
            np.array([geom.vert_end for geom in geoms], dtype=gs.np_int),
            np.array([geom.face_end for geom in geoms], dtype=gs.np_int),
            np.array([geom.edge_end for geom in geoms], dtype=gs.np_int),
            np.array([geom.verts_state_end for geom in geoms], dtype=gs.np_int),
            np.array([geom.init_pos for geom in geoms], dtype=gs.np_float),
            np.array(geoms_center, dtype=gs.np_float),
            np.array([geom.init_quat for geom in geoms], dtype=gs.np_float),
            np.array([geom.type for geom in geoms], dtype=gs.np_int),
            np.array([geom.desc.friction for geom in geoms], dtype=gs.np_float),
            np.array([geom.desc.friction_torsional for geom in geoms], dtype=gs.np_float),
            np.array([geom.desc.friction_rolling for geom in geoms], dtype=gs.np_float),
            np.array([geom.desc.sol_params for geom in geoms], dtype=gs.np_float),
            np.array([geom.data for geom in geoms], dtype=gs.np_float),
            np.array([geom.is_convex for geom in geoms], dtype=gs.np_bool),
            np.array([geom.needs_coup for geom in geoms], dtype=gs.np_int),
            np.array([geom.contype for geom in geoms], dtype=np.int32),
            np.array([geom.conaffinity for geom in geoms], dtype=np.int32),
            np.array([geom.coup_softness for geom in geoms], dtype=gs.np_float),
            np.array([geom.coup_friction for geom in geoms], dtype=gs.np_float),
            np.array([geom.coup_restitution for geom in geoms], dtype=gs.np_float),
            np.array([geom.is_fixed for geom in geoms], dtype=gs.np_bool),
            np.array([geom.metadata.get("decomposed", False) for geom in geoms], dtype=gs.np_bool),
            np.zeros(self.n_geoms, dtype=gs.np_bool),
            self.geoms_init_AABB,
            self.dyn_state,
            self.dyn_info,
            self.rigid_config,
        )

    def _init_entity_fields(self):
        if not self._entities:
            return
        entities = self._entities
        kernel_init_entity_fields(
            np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
            np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
            np.array([entity.link_start for entity in entities], dtype=gs.np_int),
            np.array([entity.link_end for entity in entities], dtype=gs.np_int),
            np.array([entity.geom_start for entity in entities], dtype=gs.np_int),
            np.array([entity.geom_end for entity in entities], dtype=gs.np_int),
            np.zeros(len(entities), dtype=gs.np_float),
            np.array([entity.is_local_collision_mask for entity in entities], dtype=gs.np_bool),
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )

    def _resolve_collider_type(self, geom):
        collider_type = geom.entity.material.collider_type
        if collider_type != "auto":
            return _COLLIDER_TYPE_BY_NAME[collider_type]
        return _AUTO_COLLIDER_TYPE_BY_GEOM_TYPE.get(geom.type, COLLIDER_TYPE.GRID)

    def _init_mochi(self):
        options = self._options
        links = self.links
        geoms = self.geoms

        # Equality constraints (connect, weld, joint couplings) are enforced as stiff penalties.
        self._equalities = [equality for entity in self._entities for equality in entity.equalities]
        dofs_entity_mass = np.zeros(self.n_dofs_total_, dtype=gs.np_float)
        for entity in self._entities:
            dofs_entity_mass[entity.dof_start : entity.dof_end] = sum(link.desc.mass for link in entity.links)
        for entity in self._soft_entities:
            dof_start = self.n_dofs + 3 * entity.v_start
            dofs_entity_mass[dof_start : dof_start + entity.n_dofs] = entity.mass

        self._layers = sorted({entity.material.contact_layer for entity in (*self._entities, *self._soft_entities)})
        n_layers = len(self._layers)
        self._layers_pair_enabled = np.ones((n_layers, n_layers), dtype=bool)
        self._entities_pair_enabled = np.ones((self.n_entities, self.n_entities), dtype=bool)
        self._soft_entities_pair_enabled = np.ones((self.n_soft_entities, self.n_soft_entities), dtype=bool)
        self._soft_rigid_entities_pair_enabled = np.ones((self.n_soft_entities, self.n_entities), dtype=bool)

        geoms_collider_type = np.array([self._resolve_collider_type(geom) for geom in geoms], dtype=gs.np_int)
        self._has_grid_colliders = bool((geoms_collider_type == COLLIDER_TYPE.GRID).any())
        geoms_contact_params = np.array(
            [
                (
                    geom.entity.material.penalty_coefficient,
                    geom.entity.material.penalty_smoothing_half_distance,
                    geom.entity.material.penalty_threshold,
                    geom.desc.friction,
                    geom.entity.material.friction_falloff_vel,
                    geom.entity.material.viscous_friction,
                    geom.entity.material.normal_viscous_damping,
                    geom.entity.material.max_alignment_normals,
                )
                for geom in geoms
            ],
            dtype=gs.np_float,
        ).reshape((-1, 8))

        # Contact samples of every collision geom that is not an unbounded plane, gathered per link.
        samples_pos, samples_normal, samples_weight, samples_link_idx, samples_geom_idx = [], [], [], [], []
        links_sample_start = np.zeros(self.n_links, dtype=gs.np_int)
        links_sample_end = np.zeros(self.n_links, dtype=gs.np_int)
        links_samples_aabb_min = np.zeros((self.n_links, 3), dtype=gs.np_float)
        links_samples_aabb_max = np.zeros((self.n_links, 3), dtype=gs.np_float)
        n_samples = 0
        for link in links:
            links_sample_start[link.idx] = n_samples
            for geom in link.geoms:
                if geom.type == gs.GEOM_TYPE.PLANE or not (geom.contype or geom.conaffinity):
                    continue
                pos, normal, weight = build_geom_samples(geom, options.boundary_element_type)
                samples_pos.append(pos)
                samples_normal.append(normal)
                samples_weight.append(weight)
                samples_link_idx.append(np.full(len(weight), link.idx, dtype=gs.np_int))
                samples_geom_idx.append(np.full(len(weight), geom.idx, dtype=gs.np_int))
                n_samples += len(weight)
            links_sample_end[link.idx] = n_samples
        # Bounding boxes of the sample clouds, per link.
        if n_samples > 0:
            samples_pos = np.concatenate(samples_pos).astype(gs.np_float)
            samples_normal = np.concatenate(samples_normal).astype(gs.np_float)
            samples_weight = np.concatenate(samples_weight).astype(gs.np_float)
            samples_link_idx = np.concatenate(samples_link_idx)
            samples_geom_idx = np.concatenate(samples_geom_idx)
            for link in links:
                start, end = links_sample_start[link.idx], links_sample_end[link.idx]
                if end > start:
                    links_samples_aabb_min[link.idx] = samples_pos[start:end].min(axis=0)
                    links_samples_aabb_max[link.idx] = samples_pos[start:end].max(axis=0)
        else:
            samples_pos = np.zeros((0, 3), dtype=gs.np_float)
            samples_normal = np.zeros((0, 3), dtype=gs.np_float)
            samples_weight = np.zeros((0,), dtype=gs.np_float)
            samples_link_idx = np.zeros((0,), dtype=gs.np_int)
            samples_geom_idx = np.zeros((0,), dtype=gs.np_int)
        # Bounding-sphere hierarchy of every link's samples; the samples of a link are permuted so that the samples of
        # a leaf are contiguous.
        tree_arrays = [[] for _ in range(6)]
        links_tree_start = np.zeros(self.n_links, dtype=gs.np_int)
        links_tree_end = np.zeros(self.n_links, dtype=gs.np_int)
        n_tree_nodes = 0
        for link in links:
            start, end = links_sample_start[link.idx], links_sample_end[link.idx]
            links_tree_start[link.idx] = n_tree_nodes
            if end > start:
                order, centers, radii, first, count, escape, is_leaf = build_sample_tree(samples_pos[start:end])
                for arrays in (samples_pos, samples_normal, samples_weight, samples_link_idx, samples_geom_idx):
                    arrays[start:end] = arrays[start:end][order]
                for values, chunk in zip(
                    tree_arrays, (centers, radii, first + start, count, escape + n_tree_nodes, is_leaf)
                ):
                    values.append(chunk)
                n_tree_nodes += len(radii)
            links_tree_end[link.idx] = n_tree_nodes
        self._n_samples = n_samples
        self.n_samples_ = max(1, n_samples)
        self.n_tree_nodes_ = max(1, n_tree_nodes)

        n_collider_geoms = int((geoms_collider_type != COLLIDER_TYPE.NONE).sum())
        n_links_with_samples = int((links_sample_end > links_sample_start).sum())
        self._max_pairs = options.max_contact_pairs_per_env
        if self._max_pairs is None:
            self._max_pairs = max(1, n_links_with_samples * n_collider_geoms)
        self._max_hits = max(1, 2 * n_samples) if options.record_contacts else 1

        # The dense matrix is allocated for systems up to `dense_matrix_max_dofs`; an environment is solved directly,
        # island by island, when its largest island fits `dense_solver_max_dofs` (every island under "ldlt").
        has_dense = options.linear_solver != "pcg" and self.n_dofs_total <= options.dense_matrix_max_dofs
        if options.linear_solver == "ldlt" and not has_dense:
            gs.raise_exception(
                f"The system has {self.n_dofs_total} degrees of freedom, more than 'dense_matrix_max_dofs' "
                f"({options.dense_matrix_max_dofs}): increase it or use the 'pcg' linear solver."
            )
        self._dense_max_dofs = self.n_dofs_total if options.linear_solver == "ldlt" else options.dense_solver_max_dofs
        # GPU: fused tiled factorization and solve of the whole matrix when the factor fits in shared memory.
        cholesky_tile_size = 16 if (self.n_dofs_total <= 16 or 32 < self.n_dofs_total <= 48) else 32
        tiled_n_dofs = max(math.ceil(self.n_dofs_total / cholesky_tile_size), 1) * cholesky_tile_size
        use_tiled_cholesky = (
            has_dense
            and gs.backend != gs.cpu
            and self.n_dofs_total >= 16
            and tiled_n_dofs <= get_gpu_shared_tile_sizes(max_n_sizes=1)[-1]
        )
        self._n_pcg_iterations = options.n_pcg_iterations
        if self._n_pcg_iterations is None:
            self._n_pcg_iterations = min(max(1, self.n_dofs_total), 1000)

        self._resolve_soft_collider_flags()
        self.mochi_config = MochiStaticConfig(
            backend=gs.backend,
            para_level=self._resolve_para_level(),
            integrator=INTEGRATOR.BDF2 if options.integrator == "bdf2" else INTEGRATOR.BACKWARD_EULER,
            use_newton_euler_inertia=options.use_newton_euler_inertia,
            friction_model=FRICTION_MODEL.CINF if options.friction_model == "cinf" else FRICTION_MODEL.C1,
            linesearch_type={"none": LINESEARCH.NONE, "residual_norm": LINESEARCH.RESIDUAL_NORM}.get(
                options.linesearch_type, LINESEARCH.ARMIJO
            ),
            linear_tolerance=(
                LINEAR_TOLERANCE.ADAPTIVE
                if options.linear_tolerance_strategy == "adaptive"
                else LINEAR_TOLERANCE.CONSTANT
            ),
            use_fitted_friction_hessian=options.use_fitted_friction_hessian,
            friction_with_collider_normal=options.friction_with_collider_normal,
            fade_friction=options.fade_friction,
            implicit_normal_force_for_dissipation=options.implicit_normal_force_for_dissipation,
            has_dense=has_dense,
            use_tiled_cholesky=use_tiled_cholesky,
            cholesky_tile_size=cholesky_tile_size,
            tiled_n_dofs=tiled_n_dofs,
            has_grid_colliders=self._has_grid_colliders,
            record_contacts=options.record_contacts,
            batch_links_info=self._options.batch_links_info,
            has_soft=self.has_soft,
            has_tets=self.n_soft_elems > 0,
            has_pc_colliders=self._has_pc_colliders,
            has_rod_band=any(e.is_rod and not e.morph.is_closed_loop for e in self._soft_entities),
            has_soft_colliders=self._has_soft_colliders,
            tet_tree_levels=self.n_tet_levels,
            has_equalities=len(self._equalities) > 0,
            has_attachments=self.n_attachments > 0,
        )
        self.mochi_info = get_mochi_info(self)
        self._build_gravity(self.mochi_info.gravity)
        self.mochi_state = get_mochi_state(self, self._max_pairs, has_dense)
        self.mochi_state.all_envs.from_numpy(np.arange(self._B, dtype=gs.np_int))
        self.contact_state = get_mochi_contact_state(self, self._max_pairs)
        self.eq_info = get_mochi_equalities_info(self, self._equalities)
        self.eq_state = get_mochi_equalities_state(self, len(self._equalities))

        kernel_init_mochi_fields(
            np.array([not link.is_fixed for link in links], dtype=gs.np_bool),
            np.array([link.entity.material.has_gravity for link in links], dtype=gs.np_bool),
            np.array([link.desc.mass for link in links], dtype=gs.np_float),
            np.array([link.desc.inertia for link in links], dtype=gs.np_float).reshape((-1, 3, 3)),
            links_sample_start,
            links_sample_end,
            links_samples_aabb_min,
            links_samples_aabb_max,
            geoms_collider_type,
            geoms_contact_params,
            samples_pos,
            samples_normal,
            samples_weight,
            samples_link_idx,
            samples_geom_idx,
            self._compute_links_pair_enabled(),
            dofs_entity_mass,
            self.mochi_info,
            self.rigid_config,
        )
        if self.n_links > 0:
            self.mochi_info.links.tree_start.from_numpy(links_tree_start)
            self.mochi_info.links.tree_end.from_numpy(links_tree_end)
        if n_tree_nodes > 0:
            tree_center, tree_radius, tree_first, tree_count, tree_escape, tree_is_leaf = (
                np.concatenate(values) for values in tree_arrays
            )
            self.mochi_info.samples.tree_center.from_numpy(tree_center)
            self.mochi_info.samples.tree_radius.from_numpy(tree_radius)
            self.mochi_info.samples.tree_first.from_numpy(tree_first)
            self.mochi_info.samples.tree_count.from_numpy(tree_count)
            self.mochi_info.samples.tree_escape.from_numpy(tree_escape)
            self.mochi_info.samples.tree_is_leaf.from_numpy(tree_is_leaf)

    def _init_soft(self):
        """Rest data, contact samples and material parameters of the deformable bodies."""
        options = self._options
        entities = self._soft_entities
        _B = self._B

        if self.has_soft:
            verts_rest = np.concatenate([entity.init_positions for entity in entities]).astype(gs.np_float)
            verts_entity_idx = np.concatenate(
                [np.full(entity.n_vertices, entity.idx_in_solver, dtype=gs.np_int) for entity in entities]
            )
            solids = [entity for entity in entities if not (entity.is_shell or entity.is_rod)]
            shells = [entity for entity in entities if entity.is_shell]
            rods = [entity for entity in entities if entity.is_rod]
            rod_elems_v = np.concatenate(
                [np.zeros((0, 2), dtype=gs.np_int)] + [entity.elems + entity.v_start for entity in rods]
            ).astype(gs.np_int)
            rod_elems_entity_idx = np.concatenate(
                [np.zeros((0,), dtype=gs.np_int)]
                + [np.full(entity.n_elements, entity.idx_in_solver, dtype=gs.np_int) for entity in rods]
            )
            rod_elems_axis_ref = np.concatenate(
                [np.zeros((0, 3), dtype=gs.np_float)] + [entity.rod_axes_ref for entity in rods]
            ).astype(gs.np_float)
            rod_stencils_v, rod_stencils_e = self._rod_stencils(rods)
            elems_v = np.concatenate(
                [np.zeros((0, 4), dtype=gs.np_int)] + [entity.elems + entity.v_start for entity in solids]
            ).astype(gs.np_int)
            elems_entity_idx = np.concatenate(
                [np.zeros((0,), dtype=gs.np_int)]
                + [np.full(entity.n_elements, entity.idx_in_solver, dtype=gs.np_int) for entity in solids]
            )
            shell_elems_v = np.concatenate(
                [np.zeros((0, 3), dtype=gs.np_int)] + [entity.elems + entity.v_start for entity in shells]
            ).astype(gs.np_int)
            shell_elems_hinge = np.concatenate(
                [np.zeros((0, 3), dtype=gs.np_int)] + [self._shell_hinges(entity) for entity in shells]
            ).astype(gs.np_int)
            shell_elems_entity_idx = np.concatenate(
                [np.zeros((0,), dtype=gs.np_int)]
                + [np.full(entity.n_elements, entity.idx_in_solver, dtype=gs.np_int) for entity in shells]
            )
            bary, ref_weights = TRIANGLE_QUADRATURES[options.boundary_element_type]
            samples_tri, samples_bary, samples_weight, samples_entity_idx = [], [], [], []
            entities_sample_range = np.zeros((len(entities), 2), dtype=gs.np_int)
            n_samples = 0
            for entity in entities:
                if entity.is_rod:
                    tri, sample_bary, weight = self._rod_samples(entity)
                else:
                    tri, sample_bary, weight = build_soft_samples(
                        entity.init_positions, entity.surface_triangles, bary, ref_weights
                    )
                samples_tri.append(tri + entity.v_start)
                samples_bary.append(sample_bary)
                samples_weight.append(weight)
                samples_entity_idx.append(np.full(len(weight), entity.idx_in_solver, dtype=gs.np_int))
                entities_sample_range[entity.idx_in_solver] = (n_samples, n_samples + len(weight))
                n_samples += len(weight)
            samples_tri = np.concatenate(samples_tri).astype(gs.np_int)
            samples_bary = np.concatenate(samples_bary).astype(gs.np_float)
            samples_weight = np.concatenate(samples_weight).astype(gs.np_float)
            samples_entity_idx = np.concatenate(samples_entity_idx)
            sdf_grids = [self._build_soft_sdf_grid(entity) for entity in entities]
            sdf_values = np.concatenate([grid["values"] for grid in sdf_grids]).astype(gs.np_float)
            sdf_starts = np.cumsum([0] + [len(grid["values"]) for grid in sdf_grids[:-1]])
            entities_params = np.array(
                [
                    [
                        entity.mass,
                        float(entity.material.has_gravity),
                        *self._solid_material_params(entity),
                        entity.material.mass_damping,
                        entity.material.stiffness_damping,
                        entity.material.penalty_coefficient,
                        entity.material.penalty_smoothing_half_distance,
                        entity.material.penalty_threshold,
                        entity.material.friction,
                        entity.material.friction_falloff_vel,
                        entity.material.viscous_friction,
                        entity.material.normal_viscous_damping,
                        entity.material.max_alignment_normals,
                        entity.v_start,
                        entity.v_end,
                        entities_sample_range[entity.idx_in_solver, 0],
                        entities_sample_range[entity.idx_in_solver, 1],
                        float(grid["collider_type"]),
                        float(sdf_start),
                        *grid["res"],
                        *grid["origin"],
                        *grid["cell"],
                        float(
                            SOFT_KIND_ROD if entity.is_rod else SOFT_KIND_SHELL if entity.is_shell else SOFT_KIND_SOLID
                        ),
                        *self._shell_material_params(entity),
                        *self._rod_material_params(entity),
                        *self._self_contact_params(entity),
                    ]
                    for entity, grid, sdf_start in zip(entities, sdf_grids, sdf_starts)
                ],
                dtype=gs.np_float,
            ).reshape((-1, len(ENTITY_PARAMS)))
        else:
            verts_rest = np.zeros((0, 3), dtype=gs.np_float)
            verts_entity_idx = np.zeros((0,), dtype=gs.np_int)
            elems_v = np.zeros((0, 4), dtype=gs.np_int)
            elems_entity_idx = np.zeros((0,), dtype=gs.np_int)
            shell_elems_v = np.zeros((0, 3), dtype=gs.np_int)
            shell_elems_hinge = np.zeros((0, 3), dtype=gs.np_int)
            shell_elems_entity_idx = np.zeros((0,), dtype=gs.np_int)
            rod_elems_v = np.zeros((0, 2), dtype=gs.np_int)
            rod_elems_entity_idx = np.zeros((0,), dtype=gs.np_int)
            rod_elems_axis_ref = np.zeros((0, 3), dtype=gs.np_float)
            rod_stencils_v = np.zeros((0, 3), dtype=gs.np_int)
            rod_stencils_e = np.zeros((0, 2), dtype=gs.np_int)
            samples_tri = np.zeros((0, 3), dtype=gs.np_int)
            samples_bary = np.zeros((0, 3), dtype=gs.np_float)
            samples_weight = np.zeros((0,), dtype=gs.np_float)
            samples_entity_idx = np.zeros((0,), dtype=gs.np_int)
            entities_params = np.zeros((0, len(ENTITY_PARAMS)), dtype=gs.np_float)
            entities_sample_range = np.zeros((0, 2), dtype=gs.np_int)
            sdf_values = np.zeros((0,), dtype=gs.np_float)
            n_samples = 0
        self.n_soft_samples_ = max(1, n_samples)
        self._max_samples_per_soft_entity = int(
            max(1, (entities_sample_range[:, 1] - entities_sample_range[:, 0]).max(initial=0))
        )

        n_collider_geoms = sum(1 for geom in self.geoms if self._resolve_collider_type(geom) != COLLIDER_TYPE.NONE)
        self._max_soft_pairs = options.max_contact_pairs_per_env
        if self._max_soft_pairs is None:
            self._max_soft_pairs = max(1, len(entities) * n_collider_geoms)
        self._max_soft_hits = max(1, options.max_soft_hits_per_sample * n_samples)
        self.n_soft_sdf_voxels_ = max(1, len(sdf_values))

        # Deformable colliders: the query list holds the rigid samples (dynamic links query the deformed tetrahedra
        # only) followed by the deformable samples (tetrahedra of other entities; spheres for point-cloud entities),
        # located through structures rebuilt at every assembly (the flags are resolved in `_init_mochi`).
        self._n_soft_queries = self.n_samples + n_samples
        self._max_sc_hits = (
            max(1, options.max_deformable_collider_hits_per_query * self._n_soft_queries)
            if self._has_soft_colliders
            else 1
        )
        # A deformable sample under self-contact sees the spheres of the opposing layer of its own body (up to
        # about a dozen within the contact range); only point-cloud entities query spheres.
        has_self_contact = any((e.is_shell or e.is_rod) and e.material.self_contact for e in entities)
        pc_hits_per_query = options.max_point_cloud_hits_per_query
        if pc_hits_per_query is None:
            pc_hits_per_query = 4 if has_self_contact else 2
        self._max_pc_hits = max(1, pc_hits_per_query * n_samples) if self._has_pc_colliders else 1
        bins_per_item = options.spatial_hash_bins_per_item
        # every item occupies up to eight entries (the cells its bounds overlap)
        self.n_pc_bins_ = _next_power_of_two(bins_per_item * 8 * self.n_soft_verts) if self._has_pc_colliders else 1
        # The hash cell of the spheres is the largest diameter of the contact range of a point-cloud collider (radius
        # plus penalty threshold, as mochi's, with a rounding margin): the range of a sphere then overlaps at most two
        # cells per axis, and a sample within it lies in one of them.
        pc_bands = [
            (e.rod_radius if e.is_rod else e.material.collider_radius) + e.material.penalty_threshold
            for e in entities
            if self._soft_collider_kind(e) == COLLIDER_TYPE.POINT_CLOUD
        ]
        self._pc_hash_cell = 2.0 * max(pc_bands) * (1.0 + 1e-3) if pc_bands else 1.0

        band = self._rod_band_layout()
        self.n_band_rows_ = max(1, len(band.rows_dof))
        csr = self._soft_csr_layout(
            elems_v, shell_elems_v, shell_elems_hinge, rod_elems_v, rod_stencils_v, rod_stencils_e
        )
        self.n_soft_dofs_ = max(1, len(csr.start) - 1)
        self.n_csr_ = max(1, len(csr.col))
        self.soft_info = get_mochi_soft_info(self)
        self.soft_state = get_mochi_soft_state(
            self, self._max_soft_pairs, self._max_soft_hits, self._max_sc_hits, self._max_pc_hits
        )
        self.soft_info.dofs_band_row.from_numpy(band.dofs_row)
        if self._tet_tree is not None:
            self.soft_info.tet_tree_first.from_numpy(self._tet_tree.first)
            self.soft_info.tet_tree_count.from_numpy(self._tet_tree.count)
            self.soft_info.tet_tree_escape.from_numpy(self._tet_tree.escape)
            self.soft_info.tet_tree_is_leaf.from_numpy(self._tet_tree.is_leaf)
            self.soft_info.tet_tree_level_nodes.from_numpy(self._tet_tree.level_nodes)
            self.soft_info.tet_tree_level_start.from_numpy(self._tet_tree.level_start)
            self.soft_info.tet_tree_elems.from_numpy(self._tet_tree_elems)
        if len(csr.col) > 0:
            self.soft_info.csr_start.from_numpy(csr.start)
            self.soft_info.csr_col.from_numpy(csr.col)
            for field, table in (
                (self.soft_info.elems_csr_block, csr.elems_block),
                (self.soft_info.shell_csr_block, csr.shell_block),
                (self.soft_info.rod_elems_csr, csr.rod_elems),
                (self.soft_info.rod_stencils_csr, csr.rod_stencils),
            ):
                if len(table) > 0:
                    field.from_numpy(table)
        if len(band.rows_dof) > 0:
            self.soft_info.band_rows_dof.from_numpy(band.rows_dof)
            self.soft_info.band_rows_entity.from_numpy(band.rows_entity)
        if self.n_soft_entities > 0:
            self.soft_info.entities_band_start.from_numpy(band.band_start)
            self.soft_info.entities_band_n.from_numpy(band.band_n)
            self.soft_info.entities_rod_elem_start.from_numpy(band.rod_elem_start)
            self.soft_info.entities_rod_elem_end.from_numpy(band.rod_elem_end)
            self.soft_info.entities_rod_stencil_start.from_numpy(band.rod_stencil_start)
            self.soft_info.entities_rod_stencil_end.from_numpy(band.rod_stencil_end)
        kernel_init_soft_fields(
            verts_rest,
            verts_entity_idx,
            elems_v,
            elems_entity_idx,
            samples_tri,
            samples_bary,
            samples_weight,
            samples_entity_idx,
            entities_params,
            self._compute_soft_links_pair_enabled(),
            self._compute_soft_pair_enabled(),
            sdf_values,
            self.soft_info,
            self.soft_state,
            self.rigid_config,
        )
        # A sample only queries the colliders it can hit: the tetrahedra of other entities, and, for point-cloud
        # entities only, the spheres of other point-cloud entities or of its own body under self-contact (mochi's
        # rule: point-cloud colliders only collide with each other, never with rigid bodies or solids).
        kinds = [self._soft_collider_kind(entity) for entity in entities]
        queries_tets = np.zeros((self.n_soft_entities_,), dtype=gs.np_int)
        queries_spheres = np.zeros((self.n_soft_entities_,), dtype=gs.np_int)
        for i, entity in enumerate(entities):
            others = [kind for j, kind in enumerate(kinds) if j != i]
            queries_tets[i] = int(any(kind == COLLIDER_TYPE.GRID for kind in others))
            queries_spheres[i] = int(
                kinds[i] == COLLIDER_TYPE.POINT_CLOUD
                and (
                    any(kind == COLLIDER_TYPE.POINT_CLOUD for kind in others)
                    or ((entity.is_shell or entity.is_rod) and entity.material.self_contact)
                )
            )
        self.soft_info.entities_queries_tets.from_numpy(queries_tets)
        self.soft_info.entities_queries_spheres.from_numpy(queries_spheres)
        if self.n_shell_elems > 0:
            kernel_init_shell_fields(
                shell_elems_v, shell_elems_hinge, shell_elems_entity_idx, self.soft_info, self.rigid_config
            )
        if self.n_rod_elems > 0:
            kernel_init_rod_fields(
                rod_elems_v,
                rod_elems_entity_idx,
                rod_elems_axis_ref,
                rod_stencils_v,
                rod_stencils_e,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
            )

        # Render geometry of the deformable surfaces. The render-vertex offsets are renormalized here because an
        # entity's visual geoms may have been replaced after later entities were added (`set_visual_mesh`).
        vvert_start, vface_start = 0, 0
        for entity in entities:
            entity._vvert_start, entity._vface_start = vvert_start, vface_start
            for vgeom in entity.vgeoms:
                vgeom._vvert_start, vgeom._vface_start = vvert_start, vface_start
                vvert_start += vgeom.n_vverts
                vface_start += vgeom.n_vfaces
        n_soft_vverts_ = max(1, self.n_soft_vverts)
        self._soft_vverts_render = array_class.V_VEC(3, dtype=qd.f32, shape=(n_soft_vverts_, _B))
        self._soft_vverts_vert_idx = array_class.V(dtype=gs.qd_int, shape=(n_soft_vverts_,))
        self._soft_vverts_elem = array_class.V(dtype=gs.qd_int, shape=(n_soft_vverts_,))
        self._soft_vverts_bary = array_class.V_VEC(4, dtype=gs.qd_float, shape=(n_soft_vverts_,))
        if self.n_soft_vverts > 0:
            # Global tetrahedron offset of every solid entity, in the concatenation order of the element tables.
            tet_start, n_tets = {}, 0
            for entity in entities:
                if not (entity.is_shell or entity.is_rod):
                    tet_start[entity.idx_in_solver] = n_tets
                    n_tets += entity.n_elements
            vert_idx, elems_idx, bary = [], [], []
            for entity in entities:
                for vgeom in entity.vgeoms:
                    if vgeom.elems_idx is not None:
                        vert_idx.append(np.zeros(vgeom.n_vverts, dtype=gs.np_int))
                        elems_idx.append(vgeom.elems_idx + tet_start[entity.idx_in_solver])
                        bary.append(vgeom.bary)
                    else:
                        vert_idx.append(vgeom.sim_verts_idx + entity.v_start)
                        elems_idx.append(np.full(vgeom.n_vverts, -1, dtype=gs.np_int))
                        bary.append(np.zeros((vgeom.n_vverts, 4), dtype=gs.np_float))
            kernel_soft_init_render(
                np.concatenate(vert_idx).astype(gs.np_int),
                np.concatenate(elems_idx).astype(gs.np_int),
                np.concatenate(bary).astype(gs.np_float),
                self._soft_vverts_vert_idx,
                self._soft_vverts_elem,
                self._soft_vverts_bary,
            )
        rods = [entity for entity in entities if entity.is_rod]
        self._n_rod_vverts = sum(entity.n_vverts for entity in rods)
        n_rod_vverts_ = max(1, self._n_rod_vverts)
        self._rod_vverts_vvert = array_class.V(dtype=gs.qd_int, shape=(n_rod_vverts_,))
        self._rod_vverts_node = array_class.V(dtype=gs.qd_int, shape=(n_rod_vverts_,))
        self._rod_vverts_elem = array_class.V(dtype=gs.qd_int, shape=(n_rod_vverts_,))
        self._rod_vverts_offset = array_class.V(dtype=gs.qd_vec2, shape=(n_rod_vverts_,))
        if self._n_rod_vverts > 0:
            rod_elem_start = {}
            n_rod_elems = 0
            for entity in rods:
                rod_elem_start[entity.idx_in_solver] = n_rod_elems
                n_rod_elems += entity.n_elements
            kernel_rod_init_render(
                np.concatenate(
                    [np.arange(entity.vvert_start, entity.vvert_start + entity.n_vverts) for entity in rods]
                ).astype(gs.np_int),
                np.concatenate([entity.rod_vverts_node + entity.v_start for entity in rods]).astype(gs.np_int),
                np.concatenate(
                    [entity.rod_vverts_elem + rod_elem_start[entity.idx_in_solver] for entity in rods]
                ).astype(gs.np_int),
                np.concatenate([entity.rod_vverts_offset for entity in rods]).astype(gs.np_float),
                self._rod_vverts_vvert,
                self._rod_vverts_node,
                self._rod_vverts_elem,
                self._rod_vverts_offset,
            )
        self._envs_offset = np.asarray(self._scene.envs_offset, dtype=gs.np_float).reshape((_B, 3))
        if self.n_attachments > 0:
            self._init_attachments(verts_rest)

    @staticmethod
    def _shell_hinges(entity):
        """Opposite vertex (global index) of the neighboring triangle across each edge of every shell triangle, -1
        across a boundary edge. Edge e is the edge opposite the local vertex e."""
        faces = np.asarray(entity.elems, dtype=np.int64)
        edge_to_faces = {}
        for i_f, face in enumerate(faces):
            for e in range(3):
                key = tuple(sorted((int(face[(e + 1) % 3]), int(face[(e + 2) % 3]))))
                edge_to_faces.setdefault(key, []).append(i_f)
        hinges = np.full((len(faces), 3), -1, dtype=gs.np_int)
        for i_f, face in enumerate(faces):
            for e in range(3):
                key = tuple(sorted((int(face[(e + 1) % 3]), int(face[(e + 2) % 3]))))
                for j_f in edge_to_faces[key]:
                    if j_f != i_f:
                        other = [int(v) for v in faces[j_f] if int(v) not in key]
                        hinges[i_f, e] = other[0] + entity.v_start
                        break
        return hinges

    @staticmethod
    def _rod_samples(entity):
        """Centerline contact samples of a rod: 3-point Gauss-Legendre quadrature of every segment, weighted by the rest
        segment length, expressed as degenerate triangles (v0, v1, v1) with barycentric coordinates (1 - s, s, 0)."""
        elems = np.asarray(entity.elems, dtype=gs.np_int)
        lengths = np.linalg.norm(entity.init_positions[elems[:, 1]] - entity.init_positions[elems[:, 0]], axis=1)
        points = np.array([0.5 * (1.0 - np.sqrt(0.6)), 0.5, 0.5 * (1.0 + np.sqrt(0.6))])
        weights = np.array([2.5 / 9.0, 4.0 / 9.0, 2.5 / 9.0])
        tri = np.repeat(np.stack([elems[:, 0], elems[:, 1], elems[:, 1]], axis=-1), 3, axis=0)
        bary = np.tile(np.stack([1.0 - points, points, np.zeros(3)], axis=-1), (len(elems), 1))
        sample_weights = (weights[None, :] * lengths[:, None]).reshape((-1,))
        return tri.astype(gs.np_int), bary.astype(gs.np_float), sample_weights.astype(gs.np_float)

    def _init_attachments(self, verts_rest):
        """Static tables of the rigid-deformable attachments, with the anchors expressed in the link frames of the
        build pose so that every attachment starts at zero violation."""
        # The link poses of the build configuration (the deformable rest state is already in world coordinates).
        self._forward_kinematics(with_velocity=False)
        att_vert, att_link, att_is_dynamic, att_stiffness, att_damping = [], [], [], [], []
        for entity in self._soft_entities:
            for att in entity.attachments:
                verts_idx = np.asarray(att["verts_idx"], dtype=gs.np_int)
                link = att["link"]
                if not (0 <= link.idx < self.n_links):
                    gs.raise_exception(f"Attachment link '{link.name}' is not simulated by this solver.")
                att_vert.append(verts_idx + entity.v_start)
                att_link.append(np.full(len(verts_idx), link.idx, dtype=gs.np_int))
                att_is_dynamic.append(np.full(len(verts_idx), 0 if link.is_fixed else 1, dtype=gs.np_int))
                att_stiffness.append(np.full(len(verts_idx), att["stiffness"], dtype=gs.np_float))
                att_damping.append(np.full(len(verts_idx), att["damping"], dtype=gs.np_float))
        att_vert = np.concatenate(att_vert)
        att_link = np.concatenate(att_link)
        links_pos = qd_to_numpy(self.dyn_state.links.pos, row_mask=0, col_mask=att_link, keepdim=False, transpose=True)
        links_quat = qd_to_numpy(
            self.dyn_state.links.quat, row_mask=0, col_mask=att_link, keepdim=False, transpose=True
        )
        pos_local = gu.inv_transform_by_trans_quat(verts_rest[att_vert], links_pos, links_quat)
        self.soft_info.att_vert.from_numpy(att_vert.astype(gs.np_int))
        self.soft_info.att_link.from_numpy(att_link)
        self.soft_info.att_link_is_dynamic.from_numpy(np.concatenate(att_is_dynamic))
        self.soft_info.att_pos_local.from_numpy(pos_local.astype(gs.np_float))
        self.soft_info.att_stiffness.from_numpy(np.concatenate(att_stiffness))
        self.soft_info.att_damping.from_numpy(np.concatenate(att_damping))

    def _soft_csr_layout(self, elems_v, shell_elems_v, shell_elems_hinge, rod_elems_v, rod_stencils_v, rod_stencils_e):
        """Scalar CSR sparsity of the deformable Hessian (local dofs: 3 i_v + k for the vertices, then one twist dof
        per rod segment) and, for every element kind, the CSR index of each entry of the element block."""
        n_dofs = 3 * self.n_soft_verts + self.n_rod_elems

        def vertex_dofs(verts):
            return (3 * verts[..., None] + np.arange(3)).reshape(len(verts), 3 * verts.shape[1])

        def element_dofs():
            yield "elems", vertex_dofs(elems_v.astype(np.int64))
            nodes = np.concatenate([shell_elems_v, shell_elems_hinge], axis=1).astype(np.int64)
            dofs = vertex_dofs(nodes)
            dofs[np.repeat(nodes < 0, 3, axis=1)] = -1
            yield "shell", dofs
            yield "rod_elems", vertex_dofs(rod_elems_v.astype(np.int64))
            v, e = rod_stencils_v.astype(np.int64), rod_stencils_e.astype(np.int64)
            dofs = np.empty((len(v), 11), dtype=np.int64)
            for a in range(3):
                dofs[:, 4 * a : 4 * a + 3] = 3 * v[:, a : a + 1] + np.arange(3)
            dofs[:, 3] = 3 * self.n_soft_verts + e[:, 0]
            dofs[:, 7] = 3 * self.n_soft_verts + e[:, 1]
            yield "rod_stencils", dofs

        # diagonal of every dof, and the whole 3x3 block of every vertex so that its three rows share one column
        # sequence (the matvec walks it once per vertex)
        keys = [np.arange(n_dofs, dtype=np.int64) * (n_dofs + 1)]
        if self.n_soft_verts > 0:
            v_dofs = 3 * np.arange(self.n_soft_verts, dtype=np.int64)[:, None] + np.arange(3)
            keys.append((v_dofs[:, :, None] * n_dofs + v_dofs[:, None, :]).ravel())
        blocks = {}
        for name, dofs in element_dofs():
            rows = np.repeat(dofs[:, :, None], dofs.shape[1], axis=2)
            cols = np.repeat(dofs[:, None, :], dofs.shape[1], axis=1)
            valid = (rows >= 0) & (cols >= 0)
            block_keys = np.where(valid, rows * n_dofs + cols, -1)
            blocks[name] = (block_keys, valid)
            keys.append(block_keys[valid])
        unique_keys = np.unique(np.concatenate(keys))
        start = np.searchsorted(unique_keys // n_dofs, np.arange(n_dofs + 1))
        col = unique_keys % n_dofs
        if self.n_soft_verts > 0:
            # the three rows of a vertex must hold the same columns in the same order
            lengths = np.diff(start)[: 3 * self.n_soft_verts].reshape(-1, 3)
            assert (lengths[:, 1:] == lengths[:, :1]).all(), "vertex rows with different lengths"
            n_cols = lengths[:, 0]
            entries_offset = np.arange(n_cols.sum()) - np.repeat(np.cumsum(n_cols) - n_cols, n_cols)
            entries_first = np.repeat(start[0 : 3 * self.n_soft_verts : 3], n_cols) + entries_offset
            for k in (1, 2):
                entries_row = np.repeat(start[k : 3 * self.n_soft_verts : 3], n_cols) + entries_offset
                assert (col[entries_first] == col[entries_row]).all(), "vertex rows with different columns"
        tables = {}
        for name, (block_keys, valid) in blocks.items():
            index = np.searchsorted(unique_keys, np.where(valid, block_keys, 0))
            n_block = block_keys.shape[1] * block_keys.shape[2]
            tables[name] = np.where(valid, index, -1).reshape(len(block_keys), n_block)
        # Tetrahedra and shells scatter per 3x3 vertex block: the position of the block's first column in the shared
        # column sequence of the row vertex (the scalar index is csr_start[3 f + r] + position + c).
        for name, n_nodes in (("elems", 4), ("shell", 6)):
            table = tables[name].reshape(-1, 3 * n_nodes, 3 * n_nodes)
            first = table[:, ::3, ::3]  # entry (3 f, 3 g) of every block
            row_dof = np.where(first >= 0, unique_keys[np.maximum(first, 0)] // n_dofs, 0)
            position = np.where(first >= 0, first - start[row_dof], -1)
            tables[f"{name}_block"] = position.reshape(len(table), n_nodes * n_nodes)
        return SoftCSRLayout(
            start=start.astype(gs.np_int),
            col=col.astype(gs.np_int),
            **{name: table.astype(gs.np_int) for name, table in tables.items()},
        )

    def _rod_band_layout(self):
        """Node-interleaved ordering [x_0, theta_0, x_1, theta_1, ..., x_(n-1)] of every open rod's degrees of freedom
        (band rows), the inverse map for all degrees of freedom, and the rod element / stencil ranges per entity."""
        n_entities = self.n_soft_entities
        dofs_row = np.full(self.n_dofs_total_, -1, dtype=gs.np_int)
        rows_dof = []
        rows_entity = []
        band_start, band_n, rod_elem_start, rod_elem_end, rod_stencil_start, rod_stencil_end = np.zeros(
            (6, max(1, n_entities)), dtype=gs.np_int
        )
        elem_offset, stencil_offset = 0, 0
        for entity in self._soft_entities:
            if not entity.is_rod:
                continue
            i_e = entity.idx_in_solver
            n_nodes, n_elems = entity.n_vertices, entity.n_elements
            n_stencils = n_nodes if entity.morph.is_closed_loop else n_nodes - 2
            rod_elem_start[i_e], rod_elem_end[i_e] = elem_offset, elem_offset + n_elems
            rod_stencil_start[i_e], rod_stencil_end[i_e] = stencil_offset, stencil_offset + n_stencils
            if not entity.morph.is_closed_loop:
                band_start[i_e] = len(rows_dof)
                for i_n in range(n_nodes):
                    for k in range(3):
                        rows_dof.append(self.n_dofs + 3 * (entity.v_start + i_n) + k)
                    if i_n < n_elems:
                        rows_dof.append(self.n_dofs + 3 * self.n_soft_verts + elem_offset + i_n)
                band_n[i_e] = len(rows_dof) - band_start[i_e]
                rows_entity.extend([i_e] * band_n[i_e])
            elem_offset += n_elems
            stencil_offset += n_stencils
        rows_dof = np.array(rows_dof, dtype=gs.np_int)
        dofs_row[rows_dof] = np.arange(len(rows_dof), dtype=gs.np_int)
        return RodBandLayout(
            dofs_row=dofs_row,
            rows_dof=rows_dof,
            rows_entity=np.array(rows_entity, dtype=gs.np_int),
            band_start=band_start,
            band_n=band_n,
            rod_elem_start=rod_elem_start,
            rod_elem_end=rod_elem_end,
            rod_stencil_start=rod_stencil_start,
            rod_stencil_end=rod_stencil_end,
        )

    @staticmethod
    def _rod_stencils(rods):
        """Interior-node stencils of the rods: the three vertices and the two segments meeting at the node."""
        stencils_v, stencils_e = [], []
        elem_start = 0
        for entity in rods:
            n_nodes = entity.n_vertices
            n_elems = entity.n_elements
            nodes = range(n_nodes) if entity.morph.is_closed_loop else range(1, n_nodes - 1)
            for i_n in nodes:
                stencils_v.append(
                    [
                        (i_n - 1) % n_nodes + entity.v_start,
                        i_n + entity.v_start,
                        (i_n + 1) % n_nodes + entity.v_start,
                    ]
                )
                stencils_e.append([(i_n - 1) % n_elems + elem_start, i_n % n_elems + elem_start])
            elem_start += n_elems
        return (
            np.array(stencils_v, dtype=gs.np_int).reshape((-1, 3)),
            np.array(stencils_e, dtype=gs.np_int).reshape((-1, 2)),
        )

    @staticmethod
    def _rod_material_params(entity):
        """(axial stiffness, torsional stiffness, linear rotational inertia) columns of a rod, zeros otherwise."""
        if entity.is_rod:
            params = entity.material.resolve(entity.rod_radius)
            return (params["axial_stiffness"], params["torsional_stiffness"], params["linear_rotational_inertia"])
        return (0.0, 0.0, 0.0)

    def _island_nodes(self):
        """Island node of every link and of every degree of freedom: the rigid entities first (all links of an
        articulation share one node), then the deformable entities."""
        rigid_node = {entity.idx: i for i, entity in enumerate(self._entities)}
        links_node = np.array([rigid_node[link.entity.idx] for link in self.links], dtype=gs.np_int)
        dofs_node = np.zeros(self.n_dofs_total_, dtype=gs.np_int)
        for link in self.links:
            dofs_node[link.dof_start : link.dof_end] = rigid_node[link.entity.idx]
        n_rigid = len(self._entities)
        offset = self.n_dofs
        for i_e, entity in enumerate(self._soft_entities):
            dofs_node[offset : offset + 3 * entity.n_vertices] = n_rigid + i_e
            offset += 3 * entity.n_vertices
        for i_e, entity in enumerate(self._soft_entities):
            if entity.is_rod:
                dofs_node[offset : offset + len(entity.elems)] = n_rigid + i_e
                offset += len(entity.elems)
        return links_node, dofs_node

    @staticmethod
    def _self_contact_params(entity):
        """(self-contact flag, rest-configuration exclusion ratio) columns of a point-cloud collider."""
        if entity.is_shell or entity.is_rod:
            return (float(entity.material.self_contact), entity.material.self_contact_exclusion_ratio)
        return (0.0, 1.5)

    @staticmethod
    def _solid_material_params(entity):
        """(model, mu, lambda, density) columns: the elastic model of a solid, the areal density of a shell."""
        material = entity.material
        if entity.is_rod:
            return (0.0, 0.0, 0.0, material.resolve(entity.rod_radius)["linear_density"])
        if entity.is_shell:
            return (0.0, 0.0, 0.0, material.areal_density)
        return (float(ELASTIC_MODEL_BY_NAME[material.model]), material.mu, material.lam, material.rho)

    @staticmethod
    def _shell_material_params(entity):
        """(membrane mu, membrane lambda, bending alpha, bending beta, collider radius) of a shell; a rod stores its two
        flexural stiffnesses in the first two columns and its radius in the last."""
        material = entity.material
        if entity.is_rod:
            params = material.resolve(entity.rod_radius)
            return (params["flexural_stiffness"], params["flexural_stiffness"], 0.0, 0.0, entity.rod_radius)
        if entity.is_shell:
            return (
                material.membrane_mu,
                material.membrane_lambda,
                material.bending_alpha,
                material.bending_beta,
                material.collider_radius,
            )
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    def _build_soft_sdf_grid(self, entity):
        """Rest-shape signed distance grid of a deformable solid collider (igl over the boundary triangles); shells are
        point-cloud colliders and bodies that do not act as colliders get an empty grid."""
        if entity.material.collider_type == "none" or entity.is_shell or entity.is_rod:
            collider_type = COLLIDER_TYPE.NONE
            if (entity.is_shell or entity.is_rod) and entity.material.collider_type != "none":
                collider_type = COLLIDER_TYPE.POINT_CLOUD
            return {
                "collider_type": collider_type,
                "values": np.zeros((0,), dtype=gs.np_float),
                "res": (1, 1, 1),
                "origin": (0.0, 0.0, 0.0),
                "cell": (1.0, 1.0, 1.0),
            }
        verts = np.asarray(entity.init_positions, dtype=np.float64)
        faces = np.asarray(entity.surface_triangles, dtype=np.int64)
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        mean_edge = float(np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1).mean())
        lower = verts.min(axis=0) - SOFT_SDF_PADDING
        upper = verts.max(axis=0) + SOFT_SDF_PADDING
        # Voxels of at most a quarter of the mean boundary edge, and at least SOFT_SDF_MIN_RES per axis so that thin
        # bodies are resolved through their thickness.
        res = np.maximum(np.ceil((upper - lower) / (SOFT_SDF_CELL_RATIO * mean_edge)).astype(int) + 1, SOFT_SDF_MIN_RES)
        cell = (upper - lower) / (res - 1)
        axes = [lower[k] + cell[k] * np.arange(res[k]) for k in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
        values, *_ = igl.signed_distance(grid, verts, faces.astype(np.int32))
        return {
            "collider_type": COLLIDER_TYPE.GRID,
            "values": np.asarray(values, dtype=gs.np_float),
            "res": tuple(int(r) for r in res),
            "origin": tuple(float(x) for x in lower),
            "cell": tuple(float(c) for c in cell),
        }

    @staticmethod
    def _soft_collider_kind(entity):
        """Collider of a deformable entity: none, the spheres of its vertices (shells and rods) or the signed distance
        field of its rest shape sampled in its deformed tetrahedra (solids)."""
        if entity.material.collider_type == "none":
            return COLLIDER_TYPE.NONE
        if entity.is_shell or entity.is_rod:
            return COLLIDER_TYPE.POINT_CLOUD
        return COLLIDER_TYPE.GRID

    def _resolve_soft_collider_flags(self):
        """Whether the scene has deformable colliders that something can query. Tetrahedral colliders are queried by
        the samples of dynamic rigid links and of other deformable entities; point-cloud colliders only collide with
        each other (mochi's rule): they are queried solely by the samples of other point-cloud entities, or by their
        own under self-contact, never by rigid bodies or solids."""
        entities = self._soft_entities
        kinds = [self._soft_collider_kind(entity) for entity in entities]
        has_self_contact = any((e.is_shell or e.is_rod) and e.material.self_contact for e in entities)
        has_queries = self.n_samples > 0 or len(entities) > 1
        self._has_soft_colliders = has_queries and any(kind == COLLIDER_TYPE.GRID for kind in kinds)
        n_pc = sum(1 for kind in kinds if kind == COLLIDER_TYPE.POINT_CLOUD)
        self._has_pc_colliders = n_pc > 1 or (n_pc > 0 and has_self_contact)
        self._build_tet_tree(kinds)

    def _build_tet_tree(self, kinds):
        """Bounding-box hierarchy over the collider tetrahedra, built over the rest shape (see tet_tree.py); the
        element indices follow the solids' concatenation order of the element arrays."""
        self._tet_tree = None
        self._tet_tree_elems = np.zeros((0,), dtype=gs.np_int)
        aabb_min, aabb_max, elems = [], [], []
        e_start = 0
        for entity, kind in zip(self._soft_entities, kinds):
            if entity.is_shell or entity.is_rod:
                continue
            if self._has_soft_colliders and kind == COLLIDER_TYPE.GRID and entity.n_elements > 0:
                rest = np.asarray(entity.init_positions, dtype=np.float64)[np.asarray(entity.elems)]
                aabb_min.append(rest.min(axis=1))
                aabb_max.append(rest.max(axis=1))
                elems.append(e_start + np.arange(entity.n_elements))
            e_start += entity.n_elements
        if elems:
            self._tet_tree = build_tet_tree(np.concatenate(aabb_min), np.concatenate(aabb_max))
            self._tet_tree_elems = np.concatenate(elems)[self._tet_tree.order].astype(gs.np_int)
        self.n_tet_levels = self._tet_tree.n_levels if self._tet_tree is not None else 0
        self.n_tet_nodes_ = max(1, len(self._tet_tree.first)) if self._tet_tree is not None else 1
        self.n_tet_tree_elems_ = max(1, len(self._tet_tree_elems))

    def _compute_soft_pair_enabled(self):
        """Deformable entity pair contact filter from the contact layers (an entity never collides with itself)."""
        enabled = np.ones((self.n_soft_entities_, self.n_soft_entities_), dtype=bool)
        if self.has_soft:
            entities_layer = np.array(
                [self._layers.index(entity.material.contact_layer) for entity in self._soft_entities], dtype=int
            )
            n = self.n_soft_entities
            enabled[:n, :n] = self._layers_pair_enabled[entities_layer[:, None], entities_layer[None, :]]
            enabled[:n, :n] &= self._soft_entities_pair_enabled
            np.fill_diagonal(enabled, False)
        return enabled

    def _compute_soft_links_pair_enabled(self):
        """Deformable entity / rigid link contact filter from the contact layers."""
        links = self.links
        enabled = np.ones((self.n_soft_entities_, self.n_links_), dtype=bool)
        if self.has_soft and self.n_links > 0:
            entities_layer = np.array(
                [self._layers.index(entity.material.contact_layer) for entity in self._soft_entities], dtype=int
            )
            links_layer = np.array(
                [self._layers.index(link.entity.material.contact_layer) for link in links], dtype=int
            )
            links_entity = np.array([link._entity_idx_in_solver for link in links], dtype=int)
            enabled[: self.n_soft_entities, : self.n_links] = self._layers_pair_enabled[
                entities_layer[:, None], links_layer[None, :]
            ]
            enabled[: self.n_soft_entities, : self.n_links] &= self._soft_rigid_entities_pair_enabled[:, links_entity]
        return enabled

    def _init_sdf(self):
        if not self._has_grid_colliders:
            self.sdf = SDF(self)
            return
        for geom in self.geoms:
            geom._preprocess()
        self.sdf = SDF(self)
        self.sdf.activate()

    def _compute_links_pair_enabled(self):
        """Link pair filter: layers, entity pairs, and never a link against itself or another link of its entity."""
        links = self.links
        links_layer = np.array([self._layers.index(link.entity.material.contact_layer) for link in links], dtype=int)
        links_entity = np.array([link._entity_idx_in_solver for link in links], dtype=int)
        enabled = self._layers_pair_enabled[links_layer[:, None], links_layer[None, :]]
        enabled &= self._entities_pair_enabled[links_entity[:, None], links_entity[None, :]]
        enabled &= links_entity[:, None] != links_entity[None, :]
        return enabled

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def _forward_kinematics(self, with_velocity=True):
        """Link and geom poses (and bounds) of the current joint coordinates; the joint-space velocities only when the
        kinematic state is final (the Newton iterates never read them)."""
        if with_velocity:
            kernel_forward_kinematics(
                self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
            )
            kernel_update_geoms(
                self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, False
            )
            kernel_update_geom_aabbs(self.geoms_init_AABB, self.dyn_state, self.dyn_info, self.rigid_config)
        else:
            kernel_update_kinematics(
                self._scene._envs_idx,
                self.geoms_init_AABB,
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

    def _on_external_state_change(self, change, envs_idx):
        self._mark_external_state_dirty(self._scene._sanitize_envs_idx(envs_idx))

    def _mark_external_state_dirty(self, envs_idx):
        """Record the environments whose state was set from outside the solver, from an already sanitized index."""
        self._external_state_dirty_mask[tensor_to_array(envs_idx)] = True

    def _sync_external_state(self):
        # State set from outside the solver invalidates the multistep history and the rotation-derivative correction
        # of the environments it touched. Untouched environments keep theirs, so that setting the state of one
        # environment leaves the trajectories of the others unchanged.
        if self._external_state_dirty_mask.any():
            envs_idx = (
                self._scene._envs_idx
                if self._external_state_dirty_mask.all()
                else self._scene._sanitize_envs_idx(np.nonzero(self._external_state_dirty_mask)[0])
            )
            kernel_reset_history(
                envs_idx,
                self.dyn_state,
                self.dyn_info,
                self.mochi_info,
                self.mochi_state,
                self.rigid_config,
            )
            self._external_state_dirty_mask[:] = False

    def _resolve_step_kernel(self):
        """How a step runs: "monolith" (one kernel, one thread per environment; the CPU, and small rigid scenes on
        the GPU), "pipeline" (one kernel per stage, the host driving the loops; deformable and large scenes on the
        GPU) or "graph" (one graph-launched kernel whose loops run on the device, parallel over items and
        environments). The graph kernel is opt-in: on GPUs without device-side graph conditionals it replays its
        loop bodies from the host and runs at the pipeline's speed, while compiling as one module several times
        slower than the pipeline's kernels."""
        step_kernel = self._options.step_kernel
        if step_kernel != "auto":
            return step_kernel
        if gs.backend == gs.cpu:
            # With several quadrants CPU threads, a deformable scene with enough work per loop runs the pipeline so
            # its element and contact loops parallelize. The threshold is empirical: each offloaded task costs
            # ~15-70 us of fork/join on the CPU pool, so a scene whose conjugate-gradient iteration does less serial
            # work than that overhead (the duck: 0.21 ms across ~20 tasks) is faster in the single-launch monolith,
            # while a larger cloth (the t-shirt: 0.71 ms per iteration) gains ~1.2x.
            if gs.cpu_threads_effective > 1 and self.has_soft and self.n_soft_verts >= 3000:
                return "pipeline"
            return "monolith"
        if self.n_dofs_total <= 64 and not self.has_soft:
            return "monolith"
        return "pipeline"

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        self._sync_external_state()
        if self._step_kernel == "graph":
            options = self._options
            n_linesearch = (
                0 if self.mochi_config.linesearch_type == LINESEARCH.NONE else options.n_linesearch_iterations
            )
            pcg_unroll = options.graph_pcg_unroll
            kernel_step_graph(
                self._newton_counter,
                self._round_counter,
                self._pcg_counter,
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.sdf._sdf_info,
                self.geoms_init_AABB,
                self.mochi_info,
                self.mochi_state,
                self.contact_state,
                self.hit_readback,
                self.island_state,
                self.eq_info,
                self.eq_state,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
                self.mochi_config,
                self.n_shell_elems > 0,
                self.n_rod_elems > 0,
                pcg_unroll,
                max(1, n_linesearch),
                self._dense_max_dofs,
                len(self._entities),
                options.n_newton_iterations,
                self._n_pcg_iterations,
                -(-self._n_pcg_iterations // pcg_unroll),
                self._max_samples_per_soft_entity,
                self._errno,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
            self._is_contacts_recorded = False
            return
        if self._step_kernel == "monolith":
            options = self._options
            n_linesearch = (
                0 if self.mochi_config.linesearch_type == LINESEARCH.NONE else options.n_linesearch_iterations
            )
            kernel_step_monolith(
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.sdf._sdf_info,
                self.geoms_init_AABB,
                self.mochi_info,
                self.mochi_state,
                self.contact_state,
                self.hit_readback,
                self.island_state,
                self.eq_info,
                self.eq_state,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
                self.mochi_config,
                self.n_shell_elems > 0,
                self.n_rod_elems > 0,
                max(1, n_linesearch),
                self._dense_max_dofs,
                len(self._entities),
                options.n_newton_iterations,
                self._n_pcg_iterations,
                self._max_samples_per_soft_entity,
                self._errno,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
            self._is_contacts_recorded = False
            return

        kernel_step_start(
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.mochi_info,
            self.mochi_state,
            self.rigid_config,
            self.mochi_config,
        )
        if self.has_soft:
            kernel_soft_step_start(
                self.mochi_state, self.soft_info, self.soft_state, self.rigid_config, self.mochi_config
            )
            if self.n_shell_elems > 0:
                kernel_shell_stage_start(self.mochi_state, self.soft_info, self.soft_state, self.rigid_config)
            if self.n_rod_elems > 0:
                kernel_rod_step_start(
                    self.mochi_state, self.soft_info, self.soft_state, self.rigid_config, self.mochi_config
                )
        self._forward_kinematics(with_velocity=False)
        kernel_store_stage_start_poses(self.dyn_state, self.mochi_state, self.rigid_config)
        if self.mochi_config.has_equalities:
            kernel_equalities_stage_start(
                self.dyn_info,
                self.rigid_info,
                self.mochi_info,
                self.mochi_state,
                self.eq_info,
                self.eq_state,
                self.rigid_config,
            )
        if self.mochi_config.has_attachments:
            kernel_attachments_stage_start(self.mochi_state, self.soft_info, self.soft_state, self.rigid_config)
        kernel_reset_newton(self.mochi_state, self.rigid_config)
        kernel_conservative_bounds(
            self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.contact_state, self.rigid_config
        )
        kernel_broadphase_pairs(
            self.dyn_state,
            self.dyn_info,
            self.mochi_info,
            self.mochi_state,
            self.contact_state,
            self.rigid_config,
            self._errno,
        )
        if self.has_soft:
            kernel_soft_conservative_bounds(
                self.mochi_state, self.mochi_info, self.soft_info, self.soft_state, self.rigid_config
            )
            kernel_soft_broadphase(
                self.dyn_state,
                self.dyn_info,
                self.mochi_info,
                self.mochi_state,
                self.contact_state,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
                self._errno,
            )
        kernel_build_islands(
            self.dyn_info,
            self.mochi_info,
            self.mochi_state,
            self.contact_state,
            self.soft_info,
            self.soft_state,
            self.island_state,
            self.eq_info,
            self._dense_max_dofs,
            len(self._entities),
            self.rigid_config,
            self.has_soft,
            self.mochi_config.has_dense,
            self.mochi_config.has_equalities,
        )
        self._select_linear_arms()
        self._newton_solve()
        kernel_post_stage(
            self.dyn_state, self.dyn_info, self.rigid_info, self.mochi_info, self.mochi_state, self.rigid_config
        )
        if self.has_soft:
            kernel_soft_post_stage(self.mochi_state, self.soft_info, self.soft_state, self.rigid_config)
            if self.n_rod_elems > 0:
                kernel_rod_post_stage(self.mochi_state, self.soft_info, self.soft_state, self.rigid_config)
        self._forward_kinematics()
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True
        self._is_contacts_recorded = False

    def _assemble(self, *, assem_res, assem_dres, skip_ls_done, record=False):
        assem_obj = self.mochi_config.linesearch_type == LINESEARCH.ARMIJO
        kernel_zero_assembly(
            self.dyn_state,
            self.mochi_state,
            self.contact_state,
            self.hit_readback,
            self.rigid_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
            record,
        )
        if self.has_soft:
            kernel_soft_zero_assembly(
                self.mochi_state, self.soft_state, self.rigid_config, assem_dres, skip_ls_done, record
            )
        kernel_contact_eval(
            self.dyn_state,
            self.dyn_info,
            self.sdf._sdf_info,
            self.mochi_info,
            self.mochi_state,
            self.contact_state,
            self.hit_readback,
            self.rigid_config,
            self.mochi_config,
            assem_dres,
            skip_ls_done,
            record,
            self._errno,
        )
        if self.has_soft:
            kernel_soft_contact_eval(
                self.dyn_state,
                self.dyn_info,
                self.sdf._sdf_info,
                self.mochi_info,
                self.mochi_state,
                self.soft_info,
                self.soft_state,
                self.hit_readback,
                self.rigid_config,
                self.mochi_config,
                self._max_samples_per_soft_entity,
                assem_res,
                assem_dres,
                skip_ls_done,
                record,
                self._errno,
            )
            if self._has_pc_colliders:
                kernel_pc_hash_build(self.mochi_state, self.soft_info, self.soft_state, self.rigid_config, skip_ls_done)
                kernel_pc_collider_eval(
                    self.dyn_state,
                    self.dyn_info,
                    self.mochi_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.hit_readback,
                    self.rigid_config,
                    self.mochi_config,
                    assem_obj,
                    assem_res,
                    assem_dres,
                    skip_ls_done,
                    record,
                    self._errno,
                )
            if self._has_soft_colliders:
                kernel_tet_tree_refit(
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.rigid_config,
                    self.mochi_config,
                    skip_ls_done,
                )
                kernel_soft_collider_eval(
                    self.dyn_state,
                    self.dyn_info,
                    self.mochi_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.hit_readback,
                    self.rigid_config,
                    self.mochi_config,
                    assem_obj,
                    assem_res,
                    assem_dres,
                    skip_ls_done,
                    record,
                    self._errno,
                )
        kernel_pairs_to_blocks(
            self.dyn_state,
            self.dyn_info,
            self.mochi_info,
            self.mochi_state,
            self.contact_state,
            self.rigid_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
        )
        kernel_assemble_links(
            self.dyn_state,
            self.dyn_info,
            self.mochi_info,
            self.mochi_state,
            self.rigid_config,
            self.mochi_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
        )
        kernel_assemble_joints(
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.mochi_info,
            self.mochi_state,
            self.rigid_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
        )
        if self.mochi_config.has_equalities:
            kernel_assemble_equalities(
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.mochi_info,
                self.mochi_state,
                self.eq_info,
                self.eq_state,
                self.rigid_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
            )
        if self.mochi_config.has_attachments:
            kernel_assemble_attachments(
                self.dyn_state,
                self.mochi_state,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
            )
        if self.has_soft:
            kernel_soft_pairs_to_blocks(
                self.dyn_state,
                self.mochi_info,
                self.mochi_state,
                self.soft_state,
                self.rigid_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
            )
            if self.n_soft_elems > 0:
                kernel_soft_assemble_elements(
                    self.mochi_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.rigid_config,
                    assem_obj,
                    assem_res,
                    assem_dres,
                    skip_ls_done,
                )
            if self.n_shell_elems > 0:
                kernel_shell_assemble(
                    self.mochi_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.rigid_config,
                    assem_obj,
                    assem_res,
                    assem_dres,
                    skip_ls_done,
                )
            if self.n_rod_elems > 0:
                kernel_rod_assemble(
                    self.mochi_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.rigid_config,
                    assem_obj,
                    assem_res,
                    assem_dres,
                    skip_ls_done,
                )
        if assem_res:
            if self.has_soft:
                kernel_soft_dirichlet(
                    self.mochi_state, self.soft_info, self.soft_state, self.rigid_config, skip_ls_done
                )
            kernel_project_links_residual(
                self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.rigid_config, skip_ls_done
            )
            kernel_residual_norms(self.mochi_state, self.island_state, self.rigid_config, skip_ls_done)

    def _select_linear_arms(self):
        """Decide, once per substep, which of the two linear arms have work to do. Islands are rebuilt at the start of
        the substep and the split they induce holds for every Newton iteration, so a single readback spares the dead
        arm all of its dispatches (and, on the conjugate gradient side, its per-iteration convergence readbacks)."""
        if self._options.linear_solver == "auto" and self.mochi_config.has_dense:
            uses_dense = qd_to_numpy(self.island_state.uses_dense)
            self._has_dense_envs = bool((uses_dense != 0).any())
            self._has_pcg_envs = bool((uses_dense == 0).any())
        else:
            self._has_dense_envs = self.mochi_config.has_dense
            self._has_pcg_envs = not self.mochi_config.has_dense

    def _linear_solve(self):
        # Environments whose largest island fits the dense limit are solved island by island by a direct
        # factorization, the others by the matrix-free conjugate gradient.
        if self._has_pcg_envs:
            kernel_pcg_init(
                self.dyn_state,
                self.dyn_info,
                self.mochi_info,
                self.mochi_state,
                self.soft_info,
                self.soft_state,
                self.island_state,
                self.rigid_config,
                self.mochi_config,
            )
            for i_iter in range(self._n_pcg_iterations):
                kernel_pcg_iter(
                    self.dyn_state,
                    self.dyn_info,
                    self.mochi_info,
                    self.mochi_state,
                    self.contact_state,
                    self.soft_info,
                    self.soft_state,
                    self.eq_info,
                    self.eq_state,
                    self.rigid_config,
                    self.mochi_config,
                )
                if (i_iter + 1) % PCG_CHECK_PERIOD == 0 and kernel_pcg_any_active(self.mochi_state) == 0:
                    break
        if self._has_dense_envs:
            kernel_condense_dense(
                self.dyn_state,
                self.dyn_info,
                self.mochi_info,
                self.mochi_state,
                self.contact_state,
                self.island_state,
                self.eq_info,
                self.eq_state,
                self.rigid_config,
                self.mochi_config.has_equalities,
            )
            if self.has_soft:
                kernel_soft_condense_dense(
                    self.dyn_state,
                    self.dyn_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.island_state,
                    self.rigid_config,
                )
            if self.mochi_config.use_tiled_cholesky:
                kernel_cholesky_solve_tiled(self.mochi_info, self.mochi_state, self.island_state, self.mochi_config)
            else:
                kernel_cholesky_solve_islands(self.mochi_info, self.mochi_state, self.island_state, self.rigid_config)

    def _newton_solve(self):
        options = self._options
        kernel_update_conv_weights(self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.rigid_config)
        if self.has_soft:
            kernel_soft_update_conv_weights(
                self.mochi_info, self.mochi_state, self.soft_info, self.soft_state, self.rigid_config
            )
            if self.n_rod_elems > 0:
                kernel_rod_update_conv_weights(
                    self.mochi_info, self.mochi_state, self.soft_info, self.soft_state, self.rigid_config
                )
        self._assemble(assem_res=True, assem_dres=True, skip_ls_done=False)
        kernel_store_initial_norms(self.rigid_info, self.mochi_state, self.island_state, self.rigid_config)
        if self.has_soft:
            kernel_soft_store_ls_ref(self.mochi_state, self.soft_state, self.rigid_config, False)
            if self.n_rod_elems > 0:
                kernel_rod_store_ls_ref(self.mochi_state, self.soft_state, self.rigid_config, False)
        kernel_convergence_check(
            self.mochi_info, self.mochi_state, self.island_state, self.rigid_config, False, self._errno
        )

        n_linesearch = options.n_linesearch_iterations
        if self.mochi_config.linesearch_type == LINESEARCH.NONE:
            n_linesearch = 0
        for i_iter in range(options.n_newton_iterations):
            if i_iter > 0:
                self._assemble(assem_res=True, assem_dres=True, skip_ls_done=False)
                kernel_convergence_check(
                    self.mochi_info, self.mochi_state, self.island_state, self.rigid_config, False, self._errno
                )
            if kernel_any_active(self.mochi_state, False) == 0:
                break
            if self._has_pcg_envs:
                kernel_update_linear_tolerance(self.mochi_info, self.mochi_state, self.rigid_config, self.mochi_config)
            self._linear_solve()
            kernel_linesearch_begin(self.rigid_info, self.mochi_state, self.rigid_config)
            if self.has_soft:
                kernel_soft_store_ls_ref(self.mochi_state, self.soft_state, self.rigid_config, False)
                if self.n_rod_elems > 0:
                    kernel_rod_store_ls_ref(self.mochi_state, self.soft_state, self.rigid_config, False)
            n_trials = max(1, n_linesearch)
            for i_ls in range(n_trials):
                kernel_apply_increment(
                    self.dyn_info, self.rigid_info, self.mochi_info, self.mochi_state, self.rigid_config
                )
                if self.has_soft:
                    kernel_soft_apply_increment(self.mochi_state, self.soft_info, self.soft_state, self.rigid_config)
                    if self.n_rod_elems > 0:
                        kernel_rod_apply_increment(self.mochi_state, self.soft_info, self.soft_state, self.rigid_config)
                self._forward_kinematics(with_velocity=False)
                self._assemble(assem_res=True, assem_dres=False, skip_ls_done=True)
                kernel_linesearch_decide(
                    self.rigid_info,
                    self.mochi_info,
                    self.mochi_state,
                    self.rigid_config,
                    self.mochi_config,
                    i_ls == n_trials - 1,
                )
                if self.has_soft:
                    kernel_soft_store_ls_ref(self.mochi_state, self.soft_state, self.rigid_config, True)
                    if self.n_rod_elems > 0:
                        kernel_rod_store_ls_ref(self.mochi_state, self.soft_state, self.rigid_config, True)
                # Every remaining trial would re-evaluate the accepted iterate of every environment, which leaves the
                # state it writes unchanged; the readback is worth it because a trial costs a full assembly.
                if i_ls + 1 < n_trials and kernel_any_active(self.mochi_state, True) == 0:
                    break
            kernel_convergence_check(
                self.mochi_info, self.mochi_state, self.island_state, self.rigid_config, True, self._errno
            )

    def substep_post_coupling(self, f):
        pass

    def check_errno(self):
        if gs.use_zerocopy or sys.platform == "darwin":
            errno = np.bitwise_or.reduce(qd_to_numpy(self._errno))
        else:
            errno = kernel_bit_reduction(self._errno)
        if errno & array_class.ErrorCode.OVERFLOW_MOCHI_CONTACT_PAIRS:
            gs.raise_exception(
                f"Exceeding max number of contact pairs ({self._max_pairs}). Please increase the value of "
                "MochiSolver's option 'max_contact_pairs_per_env'."
            )
        if errno & array_class.ErrorCode.OVERFLOW_MOCHI_CONTACTS:
            gs.raise_exception(
                "Exceeding the capacity of a contact list: increase MochiSolver's option 'max_soft_hits_per_sample', "
                "'max_deformable_collider_hits_per_query' or 'max_point_cloud_hits_per_query' (see "
                "`get_contact_capacity_usage()`); or too many tetrahedra grew past the hash cell of the deformable "
                "colliders (a solid stretched to several times its rest size)."
            )
        if errno & array_class.ErrorCode.MOCHI_DIVERGED:
            gs.raise_exception(
                "MochiSolver diverged: the residual of the implicit solve blew up. Reduce the time step or the "
                "contact stiffness, or increase 'n_newton_iterations'."
            )

    # ------------------------------------------------------------------------------------
    # ------------------------------------ render ----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_soft_state_render(self, f):
        """Environment-offset render vertex positions of the deformable surfaces, shape (n_vverts, B), as (positions,
        None, None) (UVs and faces are read from the visual geoms)."""
        if not self.has_soft or self.n_soft_vverts == 0:
            return None, None, None
        kernel_soft_get_state_render(
            self._soft_vverts_render,
            self._soft_vverts_vert_idx,
            self._soft_vverts_elem,
            self._soft_vverts_bary,
            self._envs_offset,
            self.soft_info,
            self.soft_state,
            self.rigid_config,
        )
        if self._n_rod_vverts > 0:
            kernel_rod_get_state_render(
                self._soft_vverts_render,
                self._rod_vverts_vvert,
                self._rod_vverts_node,
                self._rod_vverts_elem,
                self._rod_vverts_offset,
                self._envs_offset,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
            )
        return self._soft_vverts_render, None, None

    # ------------------------------------------------------------------------------------
    # -------------------------------- state get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f=None):
        if not self.is_active:
            return None
        s_global = self.sim.cur_step_global
        if s_global in self._queried_states:
            return self._queried_states[s_global][0]
        state = MochiSolverState(self._scene, s_global)
        kernel_get_kinematic_state(
            state.qpos,
            state.dofs_vel,
            state.links_pos,
            state.links_quat,
            self.dyn_state,
            self.rigid_info,
            self.rigid_config,
        )
        state.qpos_prev = qd_to_torch(self.mochi_state.qpos_prev, copy=True).permute((2, 0, 1)).contiguous()
        state.dofs_vel_prev = qd_to_torch(self.mochi_state.dofs_vel_prev, copy=True).permute((2, 0, 1)).contiguous()
        state.links_vel = qd_to_torch(self.mochi_state.links_vel, copy=True).permute((1, 0, 2)).contiguous()
        state.links_ang = qd_to_torch(self.mochi_state.links_ang, copy=True).permute((1, 0, 2)).contiguous()
        state.links_vsym = qd_to_torch(self.mochi_state.links_vsym, copy=True).permute((1, 0, 2, 3)).contiguous()
        state.links_vel_prev = (
            qd_to_torch(self.mochi_state.links_vel_prev, copy=True).permute((2, 0, 1, 3)).contiguous()
        )
        state.links_ang_prev = (
            qd_to_torch(self.mochi_state.links_ang_prev, copy=True).permute((2, 0, 1, 3)).contiguous()
        )
        state.links_vsym_prev = (
            qd_to_torch(self.mochi_state.links_vsym_prev, copy=True).permute((2, 0, 1, 3, 4)).contiguous()
        )
        state.n_hist = qd_to_torch(self.mochi_state.n_hist, copy=True)
        if self.has_soft:
            kernel_soft_get_state(
                state.soft_pos,
                state.soft_vel,
                state.soft_pos_prev,
                state.soft_vel_prev,
                self.soft_state,
                self.rigid_config,
            )
        self._queried_states.append(state)
        return state

    def set_state(self, f, state, envs_idx=None, *, partial: bool = False) -> None:
        if not self.is_active:
            return
        # The kinematic state is restored (and subscribers notified) by the base class; the integration history is
        # restored on top, leaving nothing for the external-state sync of the next step to rebuild.
        super().set_state(f, state, envs_idx, partial=False)
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        _kernel_set_history(
            envs_idx,
            state.qpos_prev,
            state.dofs_vel_prev,
            state.links_vel,
            state.links_ang,
            state.links_vsym,
            state.links_vel_prev,
            state.links_ang_prev,
            state.links_vsym_prev,
            state.n_hist,
            self.mochi_state,
            self.rigid_config,
        )
        kernel_update_geoms(
            self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, False
        )
        kernel_update_geom_aabbs(self.geoms_init_AABB, self.dyn_state, self.dyn_info, self.rigid_config)
        if self.has_soft:
            kernel_soft_set_state(
                envs_idx,
                state.soft_pos,
                state.soft_vel,
                state.soft_pos_prev,
                state.soft_vel_prev,
                self.soft_state,
                self.rigid_config,
            )
        self._external_state_dirty_mask[tensor_to_array(envs_idx)] = False
        self._is_contacts_recorded = False

    # ------------------------------------------------------------------------------------
    # ----------------------------------- contact API ------------------------------------
    # ------------------------------------------------------------------------------------

    def _record_contacts(self):
        if self._is_contacts_recorded:
            return
        if not self._options.record_contacts:
            gs.raise_exception("Contact readback requires MochiSolver option 'record_contacts=True'.")
        if not self._is_hit_readback_allocated:
            self.hit_readback = get_mochi_hit_readback(
                self, self._max_hits, self._max_soft_hits, self._max_sc_hits, self._max_pc_hits
            )
            self._is_hit_readback_allocated = True
        _kernel_activate_all_envs(self.mochi_state, self.rigid_config)
        self._assemble(assem_res=False, assem_dres=False, skip_ls_done=False, record=True)
        self._is_contacts_recorded = True

    def get_contacts(self, as_tensor: bool = True, to_torch: bool = True, is_padded: bool = False):
        """
        Contact points of the current state, as a dict of arrays laid out (n_envs, n_contacts, ...) (see
        `MochiEntity.get_contacts` and `MochiSoftEntity.get_contacts`). The contact axis holds the largest
        per-environment count; shorter environments carry -1 indices and zeros in their unused slots.
        """
        self._record_contacts()
        n_contacts = torch.zeros((self._B,), dtype=gs.tc_int, device=gs.device)
        kernel_count_contact_records(self.hit_readback, self.soft_state, n_contacts, self.rigid_config, self.has_soft)
        n_max = max(1, int(n_contacts.max()))

        def ints(*shape):
            return torch.full((self._B, n_max, *shape), -1, dtype=gs.tc_int, device=gs.device)

        def floats(*shape):
            return torch.zeros((self._B, n_max, *shape), dtype=gs.tc_float, device=gs.device)

        contact_data = {
            "entity_a": ints(),
            "entity_b": ints(),
            "link_a": ints(),
            "link_b": ints(),
            "geom_a": ints(),
            "geom_b": ints(),
            "verts_a": ints(3),
            "bary_a": floats(3),
            "verts_b": ints(4),
            "bary_b": floats(4),
            "position": floats(3),
            "normal": floats(3),
            "distance": floats(),
            "force_a": floats(3),
            "weight": floats(),
        }
        kernel_gather_contact_records(
            self._links_entity_idx,
            self._geoms_link_idx,
            self._soft_entities_idx,
            self.mochi_info.samples,
            self.hit_readback,
            self.soft_info,
            self.soft_state,
            self.rigid_config,
            self.has_soft,
            contact_data["entity_a"],
            contact_data["entity_b"],
            contact_data["link_a"],
            contact_data["link_b"],
            contact_data["geom_a"],
            contact_data["geom_b"],
            contact_data["verts_a"],
            contact_data["bary_a"],
            contact_data["verts_b"],
            contact_data["bary_b"],
            contact_data["position"],
            contact_data["normal"],
            contact_data["force_a"],
            contact_data["distance"],
            contact_data["weight"],
        )
        contact_data["force_b"] = -contact_data["force_a"]
        if is_padded:
            contact_data["n_contacts"] = n_contacts if self.n_envs > 0 else n_contacts[0]
        if self.n_envs == 0:
            contact_data = {key: value[0] for key, value in contact_data.items()}
        if not to_torch:
            contact_data = {key: tensor_to_array(value) for key, value in contact_data.items()}
        return contact_data

    def get_collider_distances(self, geom_idx, points_world, envs_idx=None):
        """
        Signed distance and distance gradient of world points to a collider geom, in the world frame, as seen by the
        contact model (a grid collider only answers inside its grid; other points are flagged invalid).

        Returns
        -------
        (distances, gradients, is_valid) : tuple of arrays of shape (n_points,), (n_points, 3), (n_points,)
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        points = torch.as_tensor(points_world, dtype=gs.tc_float, device=gs.device).reshape((-1, 3)).contiguous()
        distances = torch.zeros((points.shape[0],), dtype=gs.tc_float, device=gs.device)
        gradients = torch.zeros((points.shape[0], 3), dtype=gs.tc_float, device=gs.device)
        is_valid = torch.zeros((points.shape[0],), dtype=torch.bool, device=gs.device)
        _kernel_get_collider_distances(
            int(geom_idx),
            int(envs_idx[0]),
            points,
            distances,
            gradients,
            is_valid,
            self.dyn_state,
            self.dyn_info,
            self.sdf._sdf_info,
            self.mochi_info,
            self.mochi_config,
        )
        return distances, gradients, is_valid

    def get_convergence_info(self):
        """Per-environment outcome of the last solve: number of Newton iterations, number of conjugate-gradient
        iterations summed over the Newton iterations, status (0 running, 1 converged, 2 stopped at the iteration budget,
        3 diverged), and the plain residual norm before and after the solve."""
        return {
            "n_iter": qd_to_torch(self.mochi_state.n_iter, copy=True),
            "n_pcg_iter": qd_to_torch(self.mochi_state.n_pcg_iter, copy=True),
            "status": qd_to_torch(self.mochi_state.status, copy=True),
            "res_norm0": qd_to_torch(self.mochi_state.res_norm0, copy=True),
            "res_norm": qd_to_torch(self.mochi_state.res_norm_sq, copy=True).sqrt(),
        }

    def get_contact_capacity_usage(self):
        """Per bounded contact list, the largest per-environment count of the last full assembly (the one the last
        linear solve used) and the capacity of the list (what the `max_*` options size), to tune the capacities of a
        scene."""
        usage = {"contact_pairs": (int(qd_to_numpy(self.contact_state.n_pairs).max()), self._max_pairs)}
        if self.has_soft:
            usage["soft_hits"] = (int(qd_to_numpy(self.soft_state.n_soft_hits_max)), self._max_soft_hits)
            usage["soft_collider_hits"] = (int(qd_to_numpy(self.soft_state.n_sc_hits_max)), self._max_sc_hits)
            usage["point_cloud_hits"] = (int(qd_to_numpy(self.soft_state.n_pc_hits_max)), self._max_pc_hits)
        return usage

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def _set_dofs_info(self, tensor_list, dofs_idx, name, envs_idx=None):
        tensor_list = list(tensor_list)
        for j, tensor in enumerate(tensor_list):
            tensor, dofs_idx, envs_idx_ = self._sanitize_io_variables(
                tensor,
                dofs_idx,
                self.n_dofs,
                "dofs_idx",
                envs_idx,
                batched=self._options.batch_dofs_info,
                skip_allocation=True,
            )
            if self.n_envs == 0 and self._options.batch_dofs_info:
                tensor = tensor[None]
            tensor_list[j] = tensor
        if name == "kp":
            kernel_set_dofs_kp(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "kv":
            kernel_set_dofs_kv(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "stiffness":
            kernel_set_dofs_stiffness(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "armature":
            kernel_set_dofs_armature(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "damping":
            kernel_set_dofs_damping(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "limit":
            kernel_set_dofs_limit(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def set_dofs_kp(self, kp, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kp], dofs_idx, "kp", envs_idx)

    def set_dofs_kv(self, kv, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kv], dofs_idx, "kv", envs_idx)

    def set_dofs_stiffness(self, stiffness, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([stiffness], dofs_idx, "stiffness", envs_idx)

    def set_dofs_armature(self, armature, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([armature], dofs_idx, "armature", envs_idx)

    def set_dofs_damping(self, damping, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([damping], dofs_idx, "damping", envs_idx)

    def set_dofs_limit(self, lower, upper, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx)

    def _sanitize_control(self, tensor, dofs_idx, envs_idx):
        tensor, dofs_idx, envs_idx = self._sanitize_io_variables(
            tensor, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            tensor = tensor[None]
        return tensor, dofs_idx, envs_idx

    def control_dofs_force(self, force, dofs_idx=None, envs_idx=None):
        force, dofs_idx, envs_idx = self._sanitize_control(force, dofs_idx, envs_idx)
        kernel_control_dofs_force(dofs_idx, envs_idx, force, self.dyn_state, self.rigid_config)

    def control_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None):
        velocity, dofs_idx, envs_idx = self._sanitize_control(velocity, dofs_idx, envs_idx)
        kernel_control_dofs_velocity(dofs_idx, envs_idx, velocity, self.dyn_state, self.rigid_config)

    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        position, dofs_idx, envs_idx = self._sanitize_control(position, dofs_idx, envs_idx)
        kernel_control_dofs_position(dofs_idx, envs_idx, position, self.dyn_state, self.rigid_config)

    def control_dofs_position_velocity(self, position, velocity, dofs_idx=None, envs_idx=None):
        position, dofs_idx, envs_idx_ = self._sanitize_control(position, dofs_idx, envs_idx)
        velocity, _, _ = self._sanitize_control(velocity, dofs_idx, envs_idx)
        kernel_control_dofs_position_velocity(
            dofs_idx, envs_idx_, position, velocity, self.dyn_state, self.rigid_config
        )

    def get_dofs_control_force(self, dofs_idx=None, envs_idx=None):
        _tensor, dofs_idx, envs_idx = self._sanitize_io_variables(None, dofs_idx, self.n_dofs, "dofs_idx", envs_idx)
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_dofs_control_force(dofs_idx, envs_idx, tensor, self.dyn_state, self.dyn_info, self.rigid_config)
        return _tensor

    def set_links_contact_params(self, links_idx, **params):
        links_idx = tensor_to_array(links_idx).reshape((-1,))
        geoms_idx = np.concatenate(
            [np.arange(self.links[i_l].geom_start, self.links[i_l].geom_end) for i_l in links_idx], dtype=gs.np_int
        )
        for name, value in params.items():
            if value is None:
                continue
            field = self.mochi_info.geoms.__getattribute__(name)
            values = qd_to_numpy(field)
            values[geoms_idx] = value
            field.from_numpy(values)

    @property
    def is_links_info_batched(self) -> bool:
        """Whether the inertial properties of a link are stored per environment rather than shared by the batch."""
        return self._options.batch_links_info

    def get_links_mass(self, links_idx=None, envs_idx=None):
        """The mass of each link, as the solver currently uses it."""
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.inertial_mass, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_links_inertia(self, links_idx=None, envs_idx=None):
        """The inertia matrix of each link, in its inertial frame, as the solver currently uses it."""
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.inertial_i, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_links_COM(self, links_idx=None, envs_idx=None):
        """The center of mass (COM) of each link, as an offset from the origin of its local frame."""
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.inertial_pos, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_links_invweight(self, links_idx=None, envs_idx=None):
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.invweight, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def set_links_mass(self, mass, links_idx=None, envs_idx=None, scale_inertia=False):
        gs.raise_exception("`set_links_mass` is not supported by the MochiSolver: link masses are fixed at build.")

    def set_links_inertia(self, inertia, links_idx=None, envs_idx=None):
        gs.raise_exception("`set_links_inertia` is not supported by the MochiSolver: link inertias are fixed at build.")

    def set_links_COM(self, com, links_idx=None, envs_idx=None):
        gs.raise_exception("`set_links_COM` is not supported by the MochiSolver: link inertials are fixed at build.")

    def set_links_has_gravity(self, links_idx, has_gravity):
        links_idx = tensor_to_array(links_idx).reshape((-1,))
        values = qd_to_numpy(self.mochi_info.links.has_gravity)
        values[links_idx] = has_gravity
        self.mochi_info.links.has_gravity.from_numpy(values)

    def enable_layer_contact(self, layer_a: str, layer_b: str, is_enabled: bool = True):
        """Enable or disable contact between two contact layers (symmetric)."""
        i_a, i_b = self._layers.index(layer_a), self._layers.index(layer_b)
        self._layers_pair_enabled[i_a, i_b] = self._layers_pair_enabled[i_b, i_a] = is_enabled
        kernel_set_links_pair_enabled(self._compute_links_pair_enabled(), self.mochi_info, self.rigid_config)
        if self.has_soft:
            kernel_soft_set_links_pair_enabled(
                self._compute_soft_links_pair_enabled(), self.soft_info, self.rigid_config
            )
            kernel_soft_set_pair_enabled(self._compute_soft_pair_enabled(), self.soft_info, self.rigid_config)

    def enable_entity_contact(self, entity_a, entity_b, is_enabled: bool = True):
        """Enable or disable contact between two entities (symmetric)."""
        is_soft_a = isinstance(entity_a, MochiSoftEntity)
        is_soft_b = isinstance(entity_b, MochiSoftEntity)
        i_a, i_b = entity_a._idx_in_solver, entity_b._idx_in_solver
        if is_soft_a and is_soft_b:
            self._soft_entities_pair_enabled[i_a, i_b] = self._soft_entities_pair_enabled[i_b, i_a] = is_enabled
        elif is_soft_a:
            self._soft_rigid_entities_pair_enabled[i_a, i_b] = is_enabled
        elif is_soft_b:
            self._soft_rigid_entities_pair_enabled[i_b, i_a] = is_enabled
        else:
            self._entities_pair_enabled[i_a, i_b] = self._entities_pair_enabled[i_b, i_a] = is_enabled
        kernel_set_links_pair_enabled(self._compute_links_pair_enabled(), self.mochi_info, self.rigid_config)
        if self.has_soft:
            kernel_soft_set_links_pair_enabled(
                self._compute_soft_links_pair_enabled(), self.soft_info, self.rigid_config
            )
            kernel_soft_set_pair_enabled(self._compute_soft_pair_enabled(), self.soft_info, self.rigid_config)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_geoms(self):
        if self.is_built:
            return self._n_geoms
        return len(self.geoms)

    @property
    def n_cells(self):
        if self.is_built:
            return self._n_cells
        return sum(entity.n_cells for entity in self._entities)

    @property
    def n_verts(self):
        if self.is_built:
            return self._n_verts
        return sum(entity.n_verts for entity in self._entities)

    @property
    def n_free_verts(self):
        if self.is_built:
            return self._n_free_verts
        return sum(link.n_verts if not link.is_fixed or link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_fixed_verts(self):
        if self.is_built:
            return self._n_fixed_verts
        return sum(link.n_verts if link.is_fixed and not link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_faces(self):
        if self.is_built:
            return self._n_faces
        return sum(entity.n_faces for entity in self._entities)

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        return sum(entity.n_edges for entity in self._entities)

    @property
    def n_equalities(self):
        if self.is_built:
            return len(self._equalities)
        return sum(entity.n_equalities for entity in self._entities)

    @property
    def equalities(self):
        if self.is_built:
            return self._equalities
        return gs.List(equality for entity in self._entities for equality in entity.equalities)

    @property
    def n_samples(self):
        return self._n_samples

    # ------------------------------------------------------------------------------------
    # --------------------------------- deformable bodies --------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_links > 0 or self.n_soft_entities > 0

    @property
    def has_soft(self):
        return self.n_soft_entities > 0

    @property
    def soft_entities(self):
        return self._soft_entities

    @property
    def n_soft_entities(self):
        return len(self._soft_entities)

    @property
    def n_soft_verts(self):
        return sum(entity.n_vertices for entity in self._soft_entities)

    @property
    def n_soft_elems(self):
        return sum(entity.n_elements for entity in self._soft_entities if not (entity.is_shell or entity.is_rod))

    @property
    def n_shell_elems(self):
        return sum(entity.n_elements for entity in self._soft_entities if entity.is_shell)

    @property
    def n_rod_elems(self):
        return sum(entity.n_elements for entity in self._soft_entities if entity.is_rod)

    @property
    def n_rod_stencils(self):
        return sum(entity.n_rod_stencils for entity in self._soft_entities)

    @property
    def n_soft_vverts(self):
        return sum(entity.n_vverts for entity in self._soft_entities)

    @property
    def n_soft_vfaces(self):
        return sum(entity.n_vfaces for entity in self._soft_entities)

    @property
    def n_dofs_total(self):
        """Degrees of freedom of the Newton system: rigid, then 3 per deformable vertex, then one twist per rod segment."""
        return self.n_dofs + 3 * self.n_soft_verts + self.n_rod_elems

    def _soft_field_of_entity(self, entity, field, envs_idx):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        out = torch.empty((len(envs_idx), entity.n_vertices, 3), dtype=gs.tc_float, device=gs.device)
        kernel_soft_get_vertices_field(envs_idx, entity.v_start, out, field, self.rigid_config)
        return out[0] if self.n_envs == 0 else out

    def _soft_values_of_entity(self, entity, values, envs_idx):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        values = torch.as_tensor(values, dtype=gs.tc_float, device=gs.device)
        if values.ndim == 2:
            values = values[None].expand(len(envs_idx), -1, -1)
        if values.shape != (len(envs_idx), entity.n_vertices, 3):
            gs.raise_exception(
                f"Expected an array of shape ({len(envs_idx)}, {entity.n_vertices}, 3) or ({entity.n_vertices}, 3), "
                f"got {tuple(values.shape)}."
            )
        return envs_idx, values.contiguous()

    def get_soft_entity_state(self, entity, pos, vel):
        kernel_soft_get_entity_state(entity.v_start, pos, vel, self.soft_state, self.rigid_config)

    def get_soft_vertices_position(self, entity, envs_idx=None):
        return self._soft_field_of_entity(entity, self.soft_state.verts_pos, envs_idx)

    def get_soft_vertices_velocity(self, entity, envs_idx=None):
        return self._soft_field_of_entity(entity, self.soft_state.verts_vel, envs_idx)

    def get_soft_vertices_contact_force(self, entity, envs_idx=None):
        self._record_contacts()
        return self._soft_field_of_entity(entity, self.soft_state.verts_contact_force, envs_idx)

    def set_soft_vertices_position(self, entity, pos, envs_idx=None):
        envs_idx, pos = self._soft_values_of_entity(entity, pos, envs_idx)
        kernel_soft_set_vertices_positions(envs_idx, entity.v_start, pos, self.soft_state, self.rigid_config)
        self._mark_external_state_dirty(envs_idx)
        self._is_contacts_recorded = False

    def set_soft_vertices_velocity(self, entity, vel, envs_idx=None):
        envs_idx, vel = self._soft_values_of_entity(entity, vel, envs_idx)
        kernel_soft_set_vertices_velocities(envs_idx, entity.v_start, vel, self.soft_state, self.rigid_config)
        self._mark_external_state_dirty(envs_idx)

    def set_soft_vertices_fixed(self, entity, verts_idx, is_fixed, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = np.asarray(verts_idx, dtype=gs.np_int) + entity.v_start
        kernel_soft_set_vertices_fixed(envs_idx, verts_idx, int(bool(is_fixed)), self.soft_state, self.rigid_config)
        self._mark_external_state_dirty(envs_idx)

    def set_soft_vertices_target(self, entity, verts_idx, pos, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = np.asarray(verts_idx, dtype=gs.np_int) + entity.v_start
        pos = torch.as_tensor(tensor_to_array(pos), dtype=gs.tc_float, device=gs.device)
        if pos.shape == (len(verts_idx), 3):
            pos = pos[None].expand(len(envs_idx), -1, -1)
        if pos.shape != (len(envs_idx), len(verts_idx), 3):
            gs.raise_exception(
                f"Expected an array of shape ({len(envs_idx)}, {len(verts_idx)}, 3) or ({len(verts_idx)}, 3), got "
                f"{tuple(pos.shape)}."
            )
        kernel_soft_set_vertices_target(envs_idx, verts_idx, pos.contiguous(), self.soft_state, self.rigid_config)

    def set_soft_entity_contact_params(self, entity, **params):
        material = entity.material
        keys = (
            "penalty_coefficient",
            "friction",
            "penalty_smoothing_half_distance",
            "penalty_threshold",
            "friction_falloff_vel",
            "viscous_friction",
            "normal_viscous_damping",
        )
        values = []
        for key in keys:
            value = params.get(key)
            if value is None:
                value = getattr(material, key)
            else:
                setattr(material, key, value)
            values.append(float(value))
        kernel_soft_set_entity_contact_params(entity.idx_in_solver, np.array(values, dtype=gs.np_float), self.soft_info)
        self._is_contacts_recorded = False


@qd.kernel
def _kernel_activate_all_envs(mochi_state: MochiState, rigid_config: qd.template()):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(mochi_state.is_active.shape[0]):
        mochi_state.is_active[i_b] = True
        mochi_state.ls_is_done[i_b] = False


@qd.kernel
def _kernel_set_history(
    envs_idx: qd.types.ndarray(),
    qpos_prev: qd.types.ndarray(),
    dofs_vel_prev: qd.types.ndarray(),
    links_vel: qd.types.ndarray(),
    links_ang: qd.types.ndarray(),
    links_vsym: qd.types.ndarray(),
    links_vel_prev: qd.types.ndarray(),
    links_ang_prev: qd.types.ndarray(),
    links_vsym_prev: qd.types.ndarray(),
    n_hist: qd.types.ndarray(),
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    n_qs = mochi_state.qpos_prev.shape[1]
    n_dofs = mochi_state.dofs_vel_prev.shape[1]
    n_links = mochi_state.links_vsym.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        mochi_state.n_hist[envs_idx[i_b_]] = n_hist[envs_idx[i_b_]]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_b_ in qd.ndrange(n_qs, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for k in qd.static(range(N_HISTORY)):
            mochi_state.qpos_prev[k, i_q, i_b] = qpos_prev[i_b, k, i_q]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b_ in qd.ndrange(n_dofs, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for k in qd.static(range(N_HISTORY)):
            mochi_state.dofs_vel_prev[k, i_d, i_b] = dofs_vel_prev[i_b, k, i_d]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b_ in qd.ndrange(n_links, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            mochi_state.links_vel[i_l, i_b][j] = links_vel[i_b, i_l, j]
            mochi_state.links_ang[i_l, i_b][j] = links_ang[i_b, i_l, j]
            for k in qd.static(range(N_HISTORY)):
                mochi_state.links_vel_prev[k, i_l, i_b][j] = links_vel_prev[i_b, k, i_l, j]
                mochi_state.links_ang_prev[k, i_l, i_b][j] = links_ang_prev[i_b, k, i_l, j]
        for j, l in qd.static(qd.ndrange(3, 3)):
            mochi_state.links_vsym[i_l, i_b][j, l] = links_vsym[i_b, i_l, j, l]
            for k in qd.static(range(N_HISTORY)):
                mochi_state.links_vsym_prev[k, i_l, i_b][j, l] = links_vsym_prev[i_b, k, i_l, j, l]


@qd.kernel
def _kernel_get_collider_distances(
    geom_idx: qd.i32,
    env_idx: qd.i32,
    points: qd.types.ndarray(),
    distances: qd.types.ndarray(),
    gradients: qd.types.ndarray(),
    is_valid: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    mochi_info: MochiInfo,
    mochi_config: qd.template(),
):
    for i_p in range(points.shape[0]):
        pos_world = qd.Vector([points[i_p, 0], points[i_p, 1], points[i_p, 2]], dt=gs.qd_float)
        pos_g = dyn_state.geoms.pos[geom_idx, env_idx]
        quat_g = dyn_state.geoms.quat[geom_idx, env_idx]
        pos_geom = gu.qd_inv_transform_by_trans_quat(pos_world, pos_g, quat_g)
        valid, sd, grad = query_collider(geom_idx, pos_geom, dyn_info.geoms, mochi_info.geoms, sdf_info, mochi_config)
        grad_world = gu.qd_transform_by_quat(grad, quat_g)
        distances[i_p] = sd
        is_valid[i_p] = valid
        for k in qd.static(range(3)):
            gradients[i_p, k] = grad_world[k]
