import logging
import os
import sys
import tempfile
import weakref
from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity.inertial import RHO_MUJOCO, RHO_OBJECT, RHO_ROBOT
from genesis.engine.materials.FEM.cloth import Cloth
from genesis.options.solvers import IPCCouplerOptions, RigidOptions
from genesis.repr_base import RBC
from genesis.utils.misc import geometric_mean, harmonic_mean, qd_to_numpy, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities import FEMEntity, RigidEntity
    from genesis.engine.entities.rigid_entity import RigidJoint, RigidLink
    from genesis.engine.simulator import Simulator
    from genesis.engine.solvers import FEMSolver, RigidSolver

# Check if libuipc is available
try:
    import uipc

    UIPC_AVAILABLE = True
except ImportError:
    UIPC_AVAILABLE = False

if TYPE_CHECKING or UIPC_AVAILABLE:
    import polyscope as ps
    from uipc.backend import SceneVisitor
    from uipc.constitution import (
        AffineBodyConstitution,
        AffineBodyPrismaticJoint,
        AffineBodyRevoluteJoint,
        DiscreteShellBending,
        ElasticModuli,
        ElasticModuli2D,
        ExternalArticulationConstraint,
        SoftTransformConstraint,
        StableNeoHookean,
        StrainLimitingBaraffWitkinShell,
    )
    from uipc.core import Engine, World, Scene, AffineBodyStateAccessorFeature, ContactElement, SubsceneElement
    from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexSlot
    from uipc.gui import SceneGUI

    from .data import COUPLING_TYPE, ABDLinkEntry, ArticulatedEntityData, IPCCouplingData
    from .utils import (
        build_ipc_scene_config,
        compute_link_to_link_transform,
        default_coup_type,
        find_target_link_for_fixed_merge,
        has_articulation_dofs,
        read_ipc_geometry_metadata,
        update_coupling_forces,
    )


ABD_KAPPA = 100.0  # MPa unit
# TODO: consider deriving from Genesis joint properties instead of hardcoding.
STIFFNESS_DEFAULT = 1e4
JOINT_STRENGTH_RATIO = 100.0


def _animate_rigid_link(coupler_ref, link, env_idx, info):
    """Animator callback for a soft-constraint coupled rigid link.

    Uses a weakref to the coupler to avoid preventing garbage collection.
    """
    coupler = coupler_ref()
    if coupler is None:
        gs.raise_exception("IPCCoupler was garbage collected while animator callback is still active.")

    geom_slots = info.geo_slots()
    if not geom_slots:
        return
    geom = geom_slots[0].geometry()

    # Enable constraint and set target transform (q_genesis^n)
    is_constrained_attr = geom.instances().find(uipc.builtin.is_constrained)
    aim_transform_attr = geom.instances().find(uipc.builtin.aim_transform)
    assert is_constrained_attr and aim_transform_attr
    uipc.view(is_constrained_attr)[0] = 1
    uipc.view(aim_transform_attr)[:] = coupler._abd_transforms_by_link[link][env_idx]


class IPCCoupler(RBC):
    """
    Coupler class for handling Incremental Potential Contact (IPC) simulation coupling.

    This coupler manages the communication between Genesis solvers and the IPC system,
    including rigid bodies (as ABD objects) and FEM bodies in a unified contact framework.
    """

    def __init__(self, simulator: "Simulator", options: IPCCouplerOptions) -> None:
        """
        Initialize IPC Coupler.

        Parameters
        ----------
        simulator : Simulator
            The simulator containing all solvers
        options : IPCCouplerOptions
            IPC configuration options
        """
        # Check if uipc is available
        if not UIPC_AVAILABLE:
            raise ImportError(
                "Python module 'uipc' is required by IPCCoupler but is not installed. Please install it via "
                "`pip install pyuipc`."
            )

        self.sim = simulator
        self.options = options

        # Define some proxies for convenience
        self.rigid_solver: "RigidSolver" = self.sim.rigid_solver
        self.fem_solver: "FEMSolver" = self.sim.fem_solver

        # ==== IPC System Infrastructure ====
        # The soft-constraint strengths coupling a rigid body to its Genesis pose, 'constraint_strength_translation'
        # and 'constraint_strength_rotation', are scaled by the inverse square of the substep interval to reach the
        # units the IPC world expects. That interval is only known once every solver has derived its substeps, so the
        # initialization of both of them, and of the IPC scene, is postponed to build time.
        self._constraint_strength_translation_scaled: float | None = None
        self._constraint_strength_rotation_scaled: float | None = None
        self._ipc_engine: Engine | None = None
        self._ipc_world: World | None = None
        self._ipc_scene: Scene | None = None
        self._ipc_subscenes: list[SubsceneElement] = []
        self._ipc_constitution_tabular = None
        self._ipc_contact_tabular = None
        self._ipc_subscene_tabular = None
        self._ipc_objects = None
        self._ipc_animator = None

        # ==== IPC Constitutions ====
        self._ipc_abd: AffineBodyConstitution | None = None
        self._ipc_stk: StableNeoHookean | None = None
        self._ipc_stc: SoftTransformConstraint | None = None
        self._ipc_nks: StrainLimitingBaraffWitkinShell | None = None
        self._ipc_dsb: DiscreteShellBending | None = None
        self._ipc_eac: ExternalArticulationConstraint | None = None

        # ==== IPC Contact Elements ====
        self._ipc_no_collision_contact: ContactElement | None = None
        self._ipc_fem_contacts: dict["FEMEntity", ContactElement] = {}
        self._ipc_cloth_contacts: dict["FEMEntity", ContactElement] = {}
        self._ipc_abd_contacts: dict["RigidEntity", ContactElement] = {}
        self._ipc_ground_contacts: dict["RigidEntity", ContactElement] = {}

        # ==== Entity Coupling Configuration ====
        self._coup_type_by_entity: dict["RigidEntity", COUPLING_TYPE] = {}
        self._coup_links: dict["RigidEntity", set["RigidLink"]] = {}  # Used for "two_way_soft_constraint"
        self._coupling_collision_settings: dict["RigidEntity", dict["RigidLink", bool]] = {}
        self._entities_by_coup_type: dict[COUPLING_TYPE, list["RigidEntity"]] = {}

        # ==== ABD Geometry & State ====
        self._abd_slots_by_link: dict["RigidLink", list[GeometrySlot]] = {}
        self._abd_state_feature: AffineBodyStateAccessorFeature | None = None
        self._abd_state_geom: SimplicialComplex | None = None  # Geometry for batch data transfer
        self._abd_data_by_link: dict["RigidLink", list[ABDLinkEntry]] = {}

        # ==== Two-Way Coupling State ====
        self._abd_transforms_by_link: dict["RigidLink", list[np.ndarray]] = {}
        self._coupling_data: IPCCouplingData | None = None

        # ==== GUI ====
        self._ipc_gui: SceneGUI | None = None  # polyscope viewer, only when _show_ipc_gui=True

        # ==== External Articulation ====
        self._articulation_non_fixed_base_entities: list["RigidEntity"] = []  # entities with non-fixed base
        self._articulation_data_by_entity: dict["RigidEntity", ArticulatedEntityData] = {}

    # ============================================================
    # Section 1: Configuration API
    # ============================================================

    def build(self) -> None:
        """Build IPC system"""
        # IPC coupler builds a single IPC scene shared across all envs, so it requires
        # identical geometry topology (links, joints, geoms) across environments.
        # Batched info options allow per-env topology which is incompatible.
        if self.rigid_solver.is_active:
            rigid_options = cast(RigidOptions, self.rigid_solver._options)
            if rigid_options.batch_links_info or rigid_options.batch_dofs_info or rigid_options.batch_joints_info:
                gs.raise_exception(
                    "IPC coupler does not support batched rigid info (batch_links_info, batch_dofs_info, "
                    "batch_joints_info). Please disable these options when using IPC coupling."
                )

        # The IPC world is built with one gravity, the scene's, and applies it to every body it couples, whether the
        # coupling runs one way or two. A solver asking for a different one is therefore refused, while a solver asking
        # for the scene's own, explicitly or by inheriting it, gets what the world already applies.
        # FIXME: 'Solver.set_gravity' is not honored either, and goes through: it writes a solver buffer the IPC world
        # never reads, and there is nothing to detect it by, the world being built long before the write.
        gravity_scene = self.sim.options.gravity
        for solver in (self.rigid_solver, self.fem_solver):
            gravity_solver = solver._options.gravity
            if solver.is_active and not np.allclose(gravity_solver, gravity_scene, atol=gs.EPS):
                gs.raise_exception(
                    f"{type(solver).__name__} specifies a gravity of {tuple(gravity_solver)}, which the IPC coupler "
                    f"cannot honor: every body it couples falls under the {tuple(gravity_scene)} of 'SimOptions'."
                )

        self._init_ipc()
        self._setup_coupling_config()
        self._add_objects_to_ipc()
        self._finalize_ipc()
        self._init_accessors()

        if self.options._show_ipc_gui:
            self._init_ipc_gui()

    def _setup_coupling_config(self):
        """Read coup_type, coup_links, and collision settings from entity materials."""
        assert gs.logger is not None

        entity: "RigidEntity"
        for i_e, entity in enumerate(cast(list["RigidEntity"], self.rigid_solver.entities)):
            if not entity.material.needs_coup:
                continue
            coup_type = entity.material.coup_type
            if coup_type is None:
                coup_type = default_coup_type(entity)

            self._coup_type_by_entity[entity] = coup_type = getattr(COUPLING_TYPE, coup_type.upper())
            if coup_type == COUPLING_TYPE.EXTERNAL_ARTICULATION:
                if not entity.base_link.is_fixed:
                    gs.raise_exception(
                        f"Rigid entity {i_e} is not fixed. Coupling type 'external_articulation' is not supported."
                    )
                if not has_articulation_dofs(entity):
                    gs.raise_exception(
                        f"Rigid entity {i_e} has no articulation DOFs. "
                        f"Coupling type 'external_articulation' is not supported."
                    )
            gs.logger.debug(f"Rigid entity {i_e}: coupling type '{coup_type.name.lower()}'")

            # Resolve link filter from material
            link_filter_names = entity.material.coup_links
            if link_filter_names is not None:
                self._coup_links[entity] = set(map(entity.get_link, link_filter_names))
                gs.logger.debug(f"Rigid entity {i_e}: IPC link filter set to {len(link_filter_names)} link(s)")

            # Resolve collision settings from material
            if not entity.material.enable_coup_collision:
                # Disable collision for all links
                self._coupling_collision_settings[entity] = {link: False for link in entity.links}
                gs.logger.debug(f"Rigid entity {i_e}: IPC collision disabled for all links")
            elif entity.material.coup_collision_links is not None:
                # Positive filter: only named links get collision, others disabled
                allowed = set(entity.material.coup_collision_links)
                self._coupling_collision_settings[entity] = {
                    link: False for link in entity.links if link.name not in allowed
                }
                gs.logger.debug(f"Rigid entity {i_e}: IPC collision limited to {allowed}")

        # Categorize entities by coupling type
        for entity, coup_type in self._coup_type_by_entity.items():
            self._entities_by_coup_type.setdefault(coup_type, []).append(entity)

    def _init_ipc(self) -> None:
        """Initialize IPC system components"""
        assert gs.logger is not None

        if gs.logger.level <= logging.DEBUG:
            uipc.Logger.set_level(uipc.Logger.Level.Info)
            uipc.Timer.enable_all()
        else:
            uipc.Logger.set_level(uipc.Logger.Level.Error)
            uipc.Timer.disable_all()

        # Create workspace directory for IPC output, named after scene UID.
        workspace = os.path.join(tempfile.gettempdir(), f"genesis_ipc_{self.sim.scene.uid.full()}")
        os.makedirs(workspace, exist_ok=False)

        # Note: gpu_device option may need to be set via CUDA environment variables (CUDA_VISIBLE_DEVICES)
        # before Genesis initialization, as libuipc Engine does not expose device selection in constructor
        self._ipc_engine = Engine("cuda", workspace)
        self._ipc_world = World(self._ipc_engine)

        substep_dt = self.sim.substep_dt
        self._constraint_strength_translation_scaled = self.options.constraint_strength_translation / substep_dt**2
        self._constraint_strength_rotation_scaled = self.options.constraint_strength_rotation / substep_dt**2

        self._ipc_scene = Scene(build_ipc_scene_config(self.options, self.sim.options, substep_dt))
        self._ipc_constitution_tabular = self._ipc_scene.constitution_tabular()
        self._ipc_contact_tabular = self._ipc_scene.contact_tabular()
        self._ipc_subscene_tabular = self._ipc_scene.subscene_tabular()
        self._ipc_objects = self._ipc_scene.objects()
        self._ipc_animator = self._ipc_scene.animator()
        self._ipc_no_collision_contact = self._ipc_contact_tabular.create("no_collision_contact")

        # Set up sub-scenes for multi-environment to isolate per-environment contacts if batched
        for env_idx in range(self.sim._B):
            ipc_subscene = self._ipc_subscene_tabular.create(f"subscene_{env_idx}")
            for other_ipc_subscene in self._ipc_subscenes:
                self._ipc_subscene_tabular.insert(other_ipc_subscene, ipc_subscene, False)
            self._ipc_subscenes.append(ipc_subscene)

    def _add_objects_to_ipc(self) -> None:
        """Add objects from solvers to IPC system"""
        # Add FEM entities to IPC
        if self.fem_solver.is_active:
            self._add_fem_entities_to_ipc()

        # Add rigid geoms and articulated entities to IPC based on per-entity coupling types
        if self.rigid_solver.is_active:
            self._add_rigid_geoms_to_ipc()
            self._add_articulation_entities_to_ipc()

        # Register all per-entity contact pair models with per-material friction
        self._register_contact_pairs()

    def _add_fem_entities_to_ipc(self) -> None:
        """Add FEM entities to the existing IPC scene (includes both volumetric FEM and cloth)"""

        # Create constitutions based on entity types present
        entity: "FEMEntity"
        for env_idx in range(self.sim._B):
            for i_e, entity in enumerate(cast(list["FEMEntity"], self.fem_solver.entities)):
                is_cloth = isinstance(entity.material, Cloth)
                solver_type = "cloth" if is_cloth else "fem"

                # Create object in IPC
                fem_obj = self._ipc_objects.create(f"{solver_type}_{i_e}_{env_idx}")

                # ---- Create mesh ----
                # trimesh for cloth (2D shell), tetmesh for volumetric FEM (3D)
                if is_cloth:
                    verts = tensor_to_array(entity.init_positions).astype(np.float64, copy=False)
                    faces = entity.surface_triangles.astype(np.int32, copy=False)
                    mesh = uipc.geometry.trimesh(verts, faces)
                else:
                    mesh = uipc.geometry.tetmesh(tensor_to_array(entity.init_positions), entity.elems)
                uipc.geometry.label_surface(mesh)

                # ---- Apply constitutions ----
                # Add to contact subscene (only for multi-environment)
                if self.sim.n_envs > 0:
                    self._ipc_subscenes[env_idx].apply_to(mesh)

                # Apply per-entity contact element (created once per entity on first env iteration)
                if is_cloth:
                    if entity not in self._ipc_cloth_contacts:
                        self._ipc_cloth_contacts[entity] = self._ipc_contact_tabular.create(f"cloth_contact_{i_e}")
                    self._ipc_cloth_contacts[entity].apply_to(mesh)
                else:
                    if entity not in self._ipc_fem_contacts:
                        self._ipc_fem_contacts[entity] = self._ipc_contact_tabular.create(f"fem_contact_{i_e}")
                    self._ipc_fem_contacts[entity].apply_to(mesh)

                # Apply material constitution based on type
                if is_cloth:
                    if self._ipc_nks is None:
                        self._ipc_nks = StrainLimitingBaraffWitkinShell()
                        self._ipc_constitution_tabular.insert(self._ipc_nks)

                    # Apply shell material for cloth
                    moduli = ElasticModuli2D.youngs_poisson(entity.material.E, entity.material.nu)
                    self._ipc_nks.apply_to(
                        mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                    )

                    # Apply bending stiffness if specified
                    if entity.material.bending_stiffness is not None:
                        if self._ipc_dsb is None:
                            self._ipc_dsb = DiscreteShellBending()
                            self._ipc_constitution_tabular.insert(self._ipc_dsb)

                        self._ipc_dsb.apply_to(mesh, bending_stiffness=entity.material.bending_stiffness)
                else:
                    if self._ipc_stk is None:
                        self._ipc_stk = StableNeoHookean()
                        self._ipc_constitution_tabular.insert(self._ipc_stk)

                    # Apply volumetric material for FEM
                    moduli = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
                    self._ipc_stk.apply_to(mesh, moduli, mass_density=entity.material.rho)

                # ---- Apply subscene and metadata ----
                meta_attrs = mesh.meta()
                meta_attrs.create("solver_type", solver_type)
                meta_attrs.create("entity_idx", str(i_e))
                meta_attrs.create("env_idx", str(env_idx))

                # ---- Create IPC object and geometry slot ----
                fem_obj.geometries().create(mesh)

    def _add_rigid_geoms_to_ipc(self) -> None:
        """Add rigid geoms to the IPC scene as ABD objects, merging geoms by link."""
        assert gs.logger is not None

        gs.logger.debug(f"Registered entity coupling types: {set(self._coup_type_by_entity.values())}")

        # ========== Pre-compute link groups (env-independent) ==========
        # Group links by fixed-joint merge target, matching mjcf.py behavior where geoms from fixed-joint children are
        # merged into the parent body's mesh.
        target_groups: dict["RigidLink", list["RigidLink"]] = {}  # target_link_idx -> [source_link_idx, ...]
        merge_transforms: dict["RigidLink", tuple[np.ndarray, np.ndarray]] = {
            # source_link_idx -> (R, t) relative to target frame
        }
        for link in self.rigid_solver.links:
            entity = link.entity

            coup_type = self._coup_type_by_entity.get(entity)
            if coup_type is None:
                continue

            # Link filter for two_way_soft_constraint
            if coup_type == COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT:
                link_filter = self._coup_links.get(entity)
                if link_filter is not None and link not in link_filter:
                    continue

            target_link = find_target_link_for_fixed_merge(link)
            target_groups.setdefault(target_link, []).append(link)

            if target_link is not link:
                merge_transforms[link] = compute_link_to_link_transform(link, target_link)
                gs.logger.debug(f"Fixed-merge: link {link.idx} ({link.name}) -> {target_link.idx} ({target_link.name})")

        # ========== Process each environment ==========
        links_pos = qd_to_numpy(self.rigid_solver.dyn_state.links.pos, transpose=True)
        links_quat = qd_to_numpy(self.rigid_solver.dyn_state.links.quat, transpose=True)

        for env_idx in range(self.sim._B):
            for target_link, source_links in target_groups.items():
                entity = target_link.entity
                entity_coup_type = self._coup_type_by_entity[entity]
                i_e = entity._idx_in_solver

                # ---- Collect geom meshes ----
                meshes = []
                for source_link in source_links:
                    for geom in source_link.geoms:
                        if geom.type == gs.GEOM_TYPE.PLANE:
                            if entity_coup_type != COUPLING_TYPE.IPC_ONLY:
                                gs.raise_exception(
                                    f"Plane entity (solver idx={i_e}) has coup_type='{entity_coup_type}', "
                                    f"but only 'ipc_only' is supported for plane geoms."
                                )

                            local_normal = geom.data[:3].astype(np.float64, copy=False)
                            normal = gu.transform_by_quat(local_normal, geom.init_quat)
                            normal = normal / np.linalg.norm(normal)
                            height = np.dot(geom.init_pos, normal)
                            plane_geom = uipc.geometry.ground(height, normal)

                            if entity not in self._ipc_ground_contacts:
                                plane_contact = self._ipc_contact_tabular.create(f"ground_contact_{i_e}")
                                self._ipc_ground_contacts[entity] = plane_contact
                            self._ipc_ground_contacts[entity].apply_to(plane_geom)

                            plane_obj = self._ipc_objects.create(f"rigid_plane_{geom.idx}_{env_idx}")

                            if self.sim.n_envs > 0:
                                self._ipc_subscenes[env_idx].apply_to(plane_geom)

                            plane_obj.geometries().create(plane_geom)
                        elif geom.n_verts:
                            # Apply geom transform to vertices
                            geom_verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)

                            # Apply additional transform for fixed joint merging
                            if source_link is not target_link:
                                geom_verts = gu.transform_by_trans_quat(geom_verts, *merge_transforms[source_link])

                            try:
                                mesh = uipc.geometry.trimesh(
                                    geom_verts.astype(np.float64, copy=False),
                                    geom.init_faces.astype(np.int32, copy=False),
                                )
                            except RuntimeError as e:
                                gs.raise_exception_from(f"Failed to process geom {geom.idx} for IPC.", e)

                            meshes.append(mesh)

                if not meshes:
                    continue

                # ---- Merge meshes and apply world transform ----
                rigid_link_geom = meshes[0] if len(meshes) == 1 else uipc.geometry.merge(meshes)
                uipc.geometry.label_surface(rigid_link_geom)

                link_T = gu.trans_quat_to_T(links_pos[env_idx, target_link.idx], links_quat[env_idx, target_link.idx])
                trans_view = uipc.view(rigid_link_geom.transforms())
                trans_view[0] = link_T

                # ---- Determine coupling behavior ----
                is_ipc_only = entity_coup_type == COUPLING_TYPE.IPC_ONLY
                is_free_base = (
                    entity_coup_type == COUPLING_TYPE.EXTERNAL_ARTICULATION
                    and target_link is entity.base_link
                    and not entity.base_link.is_fixed
                )
                is_soft_constraint_target = entity_coup_type == COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT or (
                    is_free_base and not self.options.free_base_driven_by_ipc
                )
                is_free_base_ipc_driven = is_free_base and self.options.free_base_driven_by_ipc

                # ---- Apply constitutions ----
                abd_obj = self._ipc_objects.create(f"rigid_link_{target_link.idx}_{env_idx}")

                if self.sim.n_envs > 0:
                    self._ipc_subscenes[env_idx].apply_to(rigid_link_geom)

                # Apply per-entity contact element or no-collision marker
                if self._coupling_collision_settings.get(entity, {}).get(target_link, True):
                    if entity not in self._ipc_abd_contacts:
                        abd_contact = self._ipc_contact_tabular.create(f"abd_contact_{i_e}")
                        self._ipc_abd_contacts[entity] = abd_contact
                    self._ipc_abd_contacts[entity].apply_to(rigid_link_geom)
                else:
                    self._ipc_no_collision_contact.apply_to(rigid_link_geom)

                # Apply ABD constitution
                if self._ipc_abd is None:
                    self._ipc_abd = AffineBodyConstitution()
                    self._ipc_constitution_tabular.insert(self._ipc_abd)

                rho = entity.material.rho
                if rho is None:
                    if entity.solver._enable_mujoco_compatibility:
                        rho = RHO_MUJOCO
                    else:
                        rho = RHO_ROBOT if target_link.desc.is_robot else RHO_OBJECT
                self._ipc_abd.apply_to(rigid_link_geom, kappa=ABD_KAPPA * uipc.unit.MPa, mass_density=rho)

                # Apply SoftTransformConstraint and animator for coupled links
                if is_soft_constraint_target:
                    if self._ipc_stc is None:
                        self._ipc_stc = SoftTransformConstraint()
                        self._ipc_constitution_tabular.insert(self._ipc_stc)

                    constraint_strength = np.array(
                        [self.options.constraint_strength_translation, self.options.constraint_strength_rotation],
                        dtype=np.float64,
                    )
                    self._ipc_stc.apply_to(rigid_link_geom, constraint_strength)
                    self._ipc_animator.insert(
                        abd_obj, partial(_animate_rigid_link, weakref.ref(self), target_link, env_idx)
                    )

                # ---- Set geometry attributes ----
                # external_kinetic: 1 = driven by rigid solver, 0 = IPC-only or IPC-driven free base
                external_kinetic_attr = rigid_link_geom.instances().find(uipc.builtin.external_kinetic)
                uipc.view(external_kinetic_attr)[:] = int(not is_free_base_ipc_driven and not is_ipc_only)

                is_fixed_attr = rigid_link_geom.instances().find(uipc.builtin.is_fixed)
                uipc.view(is_fixed_attr)[:] = int(target_link.is_fixed)

                # For external_articulation, store reference DOF for articulation constraint sync
                if entity_coup_type == COUPLING_TYPE.EXTERNAL_ARTICULATION and self.options.enable_rigid_dofs_sync:
                    ref_dof_prev_attr = rigid_link_geom.instances().create("ref_dof_prev", uipc.Vector12.Zero())
                    uipc.view(ref_dof_prev_attr)[:] = uipc.geometry.affine_body.transform_to_q(link_T)

                # set metadata attributes
                meta_attrs = rigid_link_geom.meta()
                meta_attrs.create("solver_type", "rigid")
                meta_attrs.create("link_idx", str(target_link.idx))
                meta_attrs.create("env_idx", str(env_idx))

                # ---- Create IPC object and geometry slot ----
                abd_geom_slot, _ = abd_obj.geometries().create(rigid_link_geom)

                # ---- Store slot mappings ----
                self._abd_slots_by_link.setdefault(target_link, []).append(abd_geom_slot)

    def _add_articulation_entities_to_ipc(self) -> None:
        """
        Add articulated robot entities to IPC using ExternalArticulationConstraint.

        This enables joint-level coupling between Genesis and IPC.
        """
        assert gs.logger is not None

        if COUPLING_TYPE.EXTERNAL_ARTICULATION not in self._coup_type_by_entity.values():
            return

        self._ipc_eac = ExternalArticulationConstraint()
        self._ipc_constitution_tabular.insert(self._ipc_eac)

        joints_xaxis = qd_to_numpy(self.rigid_solver.dyn_state.joints.xaxis, transpose=True)
        joints_xanchor = qd_to_numpy(self.rigid_solver.dyn_state.joints.xanchor, transpose=True)

        # Process each rigid entity with external_articulation coupling type
        for i_e, entity in enumerate(cast(list["RigidEntity"], self.rigid_solver.entities)):
            # Only process entities with external_articulation coupling type
            if self._coup_type_by_entity.get(entity) != COUPLING_TYPE.EXTERNAL_ARTICULATION:
                continue

            # Detect non-fixed base for handling base link separately via SoftTransformConstraint
            gs.logger.debug(f"Adding articulated entity {i_e} with {entity.n_joints} joints")

            mass_matrix = np.diag(np.full((entity.n_dofs,), fill_value=STIFFNESS_DEFAULT, dtype=np.float64))

            # ---- Collect joint info (env-independent) ----
            joints: list[tuple["RigidJoint", type, bool, "RigidLink", "RigidLink"]] = []
            for joint in entity.joints[(0 if entity.base_link.is_fixed else 1) :]:
                if joint.type == gs.JOINT_TYPE.FIXED:
                    continue
                elif joint.type == gs.constants.JOINT_TYPE.REVOLUTE:
                    joint_constitution = AffineBodyRevoluteJoint
                    reverse_verts = True
                elif joint.type == gs.constants.JOINT_TYPE.PRISMATIC:
                    joint_constitution = AffineBodyPrismaticJoint
                    reverse_verts = False
                else:
                    gs.raise_exception(f"Unsupported joint type: {joint.type}")

                child_link = joint.link
                parent_link = entity.links[max(joint.link.parent_idx, 0) - entity.link_start]
                parent_link = find_target_link_for_fixed_merge(parent_link)
                if parent_link not in self._abd_slots_by_link or child_link not in self._abd_slots_by_link:
                    gs.raise_exception(
                        "Rigid link has no collision geometry. Coupling type 'external_articulation' is not supported."
                    )
                joints.append((joint, joint_constitution, reverse_verts, parent_link, child_link))

            # ---- Create joint geometries per environment ----
            articulation_geom_slots: list[GeometrySlot] = []
            for env_idx in range(self.sim._B):
                joint_geom_slots: list[GeometrySlot] = []
                for joint, joint_constitution, reverse_verts, parent_link, child_link in joints:
                    joint_axis = joints_xaxis[env_idx, joint.idx]
                    joint_pos = joints_xanchor[env_idx, joint.idx]

                    v1 = joint_pos - 0.5 * joint_axis
                    v2 = joint_pos + 0.5 * joint_axis
                    vertices = np.array([v2, v1] if reverse_verts else [v1, v2], dtype=np.float64)
                    edges = np.array([[0, 1]], dtype=np.int32)
                    joint_geom = uipc.geometry.linemesh(vertices, edges)
                    if self.sim.n_envs > 0:
                        self._ipc_subscenes[env_idx].apply_to(joint_geom)

                    parent_abd_slot = self._abd_slots_by_link[parent_link][env_idx]
                    child_abd_slot = self._abd_slots_by_link[child_link][env_idx]
                    joint_constitution().apply_to(
                        joint_geom, [parent_abd_slot], [0], [child_abd_slot], [0], [JOINT_STRENGTH_RATIO]
                    )

                    joint_obj = self._ipc_objects.create(f"joint_{joint.idx}_{env_idx}")
                    joint_geom_slot, _ = joint_obj.geometries().create(joint_geom)
                    joint_geom_slots.append(joint_geom_slot)

                articulation_geom = self._ipc_eac.create_geometry(joint_geom_slots, [0] * len(joint_geom_slots))
                if self.sim.n_envs > 0:
                    self._ipc_subscenes[env_idx].apply_to(articulation_geom)

                mass_attr = articulation_geom["joint_joint"].find("mass")
                uipc.view(mass_attr).flat[:] = mass_matrix

                articulation_obj = self._ipc_objects.create(f"articulation_entity_{i_e}_{env_idx}")
                articulation_geom_slot, _ = articulation_obj.geometries().create(articulation_geom)
                articulation_geom_slots.append(articulation_geom_slot)

            # Store articulation data
            self._articulation_data_by_entity[entity] = ArticulatedEntityData(
                joints_child_link=[j.link for j, *_ in joints],
                joints_q_idx_local=[j.qs_idx_local[0] for j, *_ in joints],
                articulation_slots=articulation_geom_slots,
                ref_dof_prev=np.zeros((self.sim._B, entity.n_qs), dtype=np.float64),
                qpos_stored=np.zeros((self.sim._B, entity.n_qs), dtype=np.float64),
                qpos_current=np.zeros((self.sim._B, entity.n_qs), dtype=np.float64),
                qpos_new=np.zeros((self.sim._B, entity.n_qs), dtype=np.float64),
                delta_theta_tilde=np.zeros((self.sim._B, len(joints)), dtype=np.float64),
                delta_theta_ipc=np.zeros((self.sim._B, len(joints)), dtype=np.float64),
                prev_links_transform=[[None for _ in range(self.sim._B)] for _ in joints],
            )

            # Add to cache list if non-fixed base for '_retrieve_rigid_states' in couple()
            if not entity.base_link.is_fixed:
                self._articulation_non_fixed_base_entities.append(entity)

            gs.logger.debug(f"Successfully added articulated rigid entity {i_e} to IPC.")

    def _register_contact_pairs(self) -> None:
        """Register pairwise contact models for all entity contact elements.

        Friction is combined by geometric mean, resistance by harmonic mean (series spring).
        When an entity material does not define
        ``contact_resistance``, ``options.contact_resistance`` is used as the per-entity fallback.
        Ground pairs combine entity parameters with the plane entity's material friction.
        """
        # Collect (ContactElement, friction_mu, resistance, is_abd) for all entity contact elements
        contact_infos: list[tuple[ContactElement, float, float, bool]] = []
        for entity, elem in (*self._ipc_cloth_contacts.items(), *self._ipc_fem_contacts.items()):
            friction = entity.material.friction_mu
            resistance = entity.material.contact_resistance or self.options.contact_resistance
            contact_infos.append((elem, friction, resistance, False))
        for entity, elem in self._ipc_abd_contacts.items():
            friction = entity.material.coup_friction
            resistance = entity.material.contact_resistance or self.options.contact_resistance
            contact_infos.append((elem, friction, resistance, True))

        # Register entity-entity pairs (upper triangle including self-pairs)
        for i, (elem_i, friction_i, resistance_i, is_abd_i) in enumerate(contact_infos):
            for elem_j, friction_j, resistance_j, is_abd_j in contact_infos[i:]:
                friction_ij = geometric_mean(friction_i, friction_j)
                resistance_ij = harmonic_mean(resistance_i, resistance_j)
                enabled = not (is_abd_i and is_abd_j) or self.options.enable_rigid_rigid_contact
                self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, enabled)

        # Register per-plane ground contact pairs
        for entity, ground_elem in self._ipc_ground_contacts.items():
            plane_friction = entity.material.coup_friction
            plane_resistance = entity.material.contact_resistance or self.options.contact_resistance
            for elem, friction, resistance, is_abd in contact_infos:
                friction_ground = geometric_mean(friction, plane_friction)
                resistance_ground = harmonic_mean(resistance, plane_resistance)
                enabled = not is_abd or self.options.enable_rigid_ground_contact
                self._ipc_contact_tabular.insert(ground_elem, elem, friction_ground, resistance_ground, enabled)
            self._ipc_contact_tabular.insert(self._ipc_no_collision_contact, ground_elem, 0.0, 0.0, False)

        # Register no_collision pairs (always disabled)
        for elem, *_ in contact_infos:
            self._ipc_contact_tabular.insert(self._ipc_no_collision_contact, elem, 0.0, 0.0, False)
        self._ipc_contact_tabular.insert(
            self._ipc_no_collision_contact, self._ipc_no_collision_contact, 0.0, 0.0, False
        )

    def _finalize_ipc(self):
        """Finalize IPC setup and initialize AffineBodyStateAccessorFeature"""
        assert gs.logger is not None
        assert self._ipc_world is not None
        self._ipc_world.init(self._ipc_scene)
        # Checkpoint frame 0 so that recover(0) works in reset().
        self._ipc_world.dump()
        gs.logger.info("IPC world initialized successfully")

    def _init_accessors(self):
        assert gs.logger is not None
        assert self._ipc_world is not None

        # No ABD bodies, feature not needed
        abd_links = list(self._abd_slots_by_link.keys())
        n_abd_links = len(abd_links)
        if not abd_links:
            return

        self._abd_state_feature = cast(
            AffineBodyStateAccessorFeature, self._ipc_world.features().find(AffineBodyStateAccessorFeature)
        )
        body_count = self._abd_state_feature.body_count()

        # Verify the count matches IPC's ABD body count
        if body_count != n_abd_links * self.sim._B:
            gs.raise_exception(f"ABD body count mismatch: got {body_count}.")

        # Pre-allocate rigid link transform
        for link in abd_links:
            self._abd_transforms_by_link[link] = [np.eye(4, dtype=gs.np_float) for _ in range(self.sim._B)]

        # Create state geometry for batch data transfer
        self._abd_state_geom = self._abd_state_feature.create_geometry()
        self._abd_state_geom.instances().create(uipc.builtin.transform, np.eye(4, dtype=np.float64))
        self._abd_state_geom.instances().create(uipc.builtin.velocity, np.zeros((4, 4), dtype=np.float64))

        rigid_retrieve_entities = set(
            self._entities_by_coup_type.get(COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT, [])
            + self._entities_by_coup_type.get(COUPLING_TYPE.IPC_ONLY, [])
            + self._articulation_non_fixed_base_entities
        )
        self._abd_data_by_link = {
            link: [
                ABDLinkEntry(transform=np.eye(4, dtype=gs.np_float), velocity=np.zeros((4, 4), dtype=gs.np_float))
                for _ in range(self.sim._B)
            ]
            for link in abd_links
            if link.entity in rigid_retrieve_entities
        }

        # Pre-allocate coupling data
        coupling_links = list(self._abd_data_by_link.keys())
        abd_body_idx_by_link = {
            link: [env_idx * n_abd_links + abd_links.index(link) for env_idx in range(self.sim._B)]
            for link in coupling_links
        }
        self._coupling_data = IPCCouplingData(coupling_links, abd_body_idx_by_link, self.sim._B)

        gs.logger.debug(f"IPC coupling data created: {len(coupling_links)} links.")

    def _init_ipc_gui(self):
        """Initialize polyscope-based IPC GUI viewer."""
        try:
            if not ps.is_initialized():
                # Use EGL on Linux to match Genesis offscreen renderer and avoid context conflicts.
                ps.init("openGL3_egl" if sys.platform == "linux" else "")
            self._ipc_gui = SceneGUI(self._ipc_scene, "split")
            self._ipc_gui.register()  # also sets up_dir and ground_plane_height from scene
            ps.show(forFrames=1)
            gs.logger.info("IPC GUI initialized successfully")
        except Exception as e:
            gs.logger.warning(f"IPC GUI unavailable: {e}. Continuing without IPC GUI.")
            self._ipc_gui = None

    # ============================================================
    # Section 2: Core implementation
    # ============================================================

    def preprocess(self, f):
        """Preprocessing step before coupling"""
        pass

    def couple(self, f):
        """
        Execute IPC coupling step with per-entity coupling types.

        This unified coupling flow handles all entity types:
        - 'two_way_soft_constraint': Uses Animator + SoftTransformConstraint
        - 'external_articulation': Uses ExternalArticulationConstraint at joint level
        - 'ipc_only': One-way coupling, IPC controls rigid body transforms

        Flow:
        1. Store Genesis rigid states (common)
        2. Pre-advance processing (per entity type)
        3. IPC advance + retrieve (common, only once)
        4. Retrieve FEM states (common)
        5. Post-advance processing (per entity type)
        """
        assert self._ipc_world is not None

        if not self.is_active:
            return

        # Step 1: Store Genesis rigid states (common)
        self._store_gs_rigid_states()

        # Step 2: Pre-advance processing (per entity type)
        self._pre_advance_external_articulation()

        # Step 3: IPC advance + retrieve (common)
        self._ipc_world.advance()
        self._ipc_world.retrieve()

        # Step 4: Retrieve states
        self._retrieve_fem_states()
        self._retrieve_rigid_states()

        # Step 5: Post-advance processing (per entity type)
        self._apply_abd_coupling_forces()
        self._post_advance_external_articulation()
        self._post_advance_ipc_only()

        # Step 6: Update GUI if enabled
        if self._ipc_gui is not None:
            ps.frame_tick()
            self._ipc_gui.update()

    def couple_grad(self, f):
        """Gradient computation for coupling"""
        # IPC doesn't support gradients yet
        pass

    def reset(self, envs_idx=None):
        """Reset coupling state"""
        assert gs.logger is not None
        assert self._ipc_world is not None
        assert envs_idx is None

        gs.logger.debug("Resetting IPC coupler state")
        self._ipc_world.recover(0)
        self._ipc_world.retrieve()

    @property
    def is_active(self) -> bool:
        """Check if IPC coupling is active"""
        return self._ipc_world is not None

    @property
    def has_any_rigid_coupling(self) -> bool:
        """
        Check if any rigid entity is coupled to IPC.

        Returns
        -------
        bool
            True if at least one rigid entity has a coupling type (two_way_soft_constraint,
            external_articulation, or ipc_only).
        """
        return bool(self._coup_type_by_entity)

    # ============================================================
    # Section 3: Helpers
    # ============================================================

    def _apply_base_link_velocity_from_ipc(self, entity):
        envs_vel = np.empty((self.sim._B, 6), dtype=gs.np_float)
        for env_idx in range(self.sim._B):
            abd_entry = self._abd_data_by_link[entity.base_link][env_idx]
            envs_vel[env_idx, :3] = abd_entry.velocity[:3, 3]

            # omega_skew = dR/dt @ R^T
            omega_skew = abd_entry.velocity[:3, :3] @ abd_entry.transform[:3, :3].T
            envs_vel[env_idx, 3:] = (
                (omega_skew[2, 1] - omega_skew[1, 2]) / 2.0,
                (omega_skew[0, 2] - omega_skew[2, 0]) / 2.0,
                (omega_skew[1, 0] - omega_skew[0, 1]) / 2.0,
            )

        self.rigid_solver.set_dofs_velocity(
            envs_vel if self.sim.n_envs > 0 else envs_vel[0],
            dofs_idx=slice(entity.dof_start, entity.dof_start + 6),
            skip_forward=True,
        )

    def _pre_advance_external_articulation(self):
        """
        Pre-advance processing for external_articulation entities.
        Prepares articulation data and updates IPC geometry before advance().
        """
        if COUPLING_TYPE.EXTERNAL_ARTICULATION not in self._entities_by_coup_type:
            return

        mass_matrix = qd_to_numpy(self.rigid_solver.mass_mat, transpose=True)

        for entity, ad in self._articulation_data_by_entity.items():
            # Copy stored qpos to articulation_data.qpos_current
            ad.qpos_current[:] = ad.qpos_stored

            # Compute delta_theta_tilde = qpos_current - ref_dof_prev (per joint)
            ad.delta_theta_tilde[:] = (
                ad.qpos_current[..., ad.joints_q_idx_local] - ad.ref_dof_prev[..., ad.joints_q_idx_local]
            )

            # Update IPC geometry for each articulated entity
            for env_idx in range(self.sim._B):
                articulation_slot = ad.articulation_slots[env_idx]
                articulation_geom = articulation_slot.geometry()

                # Update ref_dof_prev on all ABD instances
                if self.options.enable_rigid_dofs_sync:
                    for child_link, prev_link_transform in zip(ad.joints_child_link, ad.prev_links_transform):
                        link_transform = prev_link_transform[env_idx]
                        if link_transform is None:
                            link_transform = self._abd_transforms_by_link[child_link][env_idx]

                        abd_geom_slot = self._abd_slots_by_link[child_link][env_idx]
                        abd_geom = abd_geom_slot.geometry()
                        ref_dof_prev_attr = abd_geom.instances().find("ref_dof_prev")
                        uipc.view(ref_dof_prev_attr)[:] = uipc.geometry.affine_body.transform_to_q(link_transform)

                # Set delta_theta_tilde to IPC geometry
                delta_theta_tilde_attr = articulation_geom["joint"].find("delta_theta_tilde")
                uipc.view(delta_theta_tilde_attr)[:] = ad.delta_theta_tilde[env_idx]

                # Extract and transfer mass matrix from Genesis to IPC
                dofs_idx = slice(entity.dof_start, entity.dof_end)
                mass_matrix_attr = articulation_geom["joint_joint"].find("mass")
                uipc.view(mass_matrix_attr).flat[:] = mass_matrix[env_idx, dofs_idx, dofs_idx]

    def _post_advance_external_articulation(self):
        """
        Post-advance processing for external_articulation entities.
        Reads delta_theta from IPC and updates Genesis qpos.
        """
        if COUPLING_TYPE.EXTERNAL_ARTICULATION not in self._entities_by_coup_type:
            return

        for entity, ad in self._articulation_data_by_entity.items():
            # Read 'delta_theta_ipc' from IPC
            for env_idx in range(self.sim._B):
                scene_art_geom = ad.articulation_slots[env_idx].geometry()
                delta_theta_attr = scene_art_geom["joint"].find("delta_theta")
                ad.delta_theta_ipc[env_idx] = delta_theta_attr.view()

            # Compute qpos_new: copy ref_dof_prev then scatter joint deltas
            ad.qpos_new[:] = ad.ref_dof_prev
            ad.qpos_new[..., ad.joints_q_idx_local] += ad.delta_theta_ipc

            # Set qpos for all DOFs.
            # For non-fixed base robots, apply base link transform from IPC.
            qpos_new = ad.qpos_new.astype(dtype=gs.np_float, copy=(not entity.base_link.is_fixed))
            if not entity.base_link.is_fixed:
                abd_entry = self._abd_data_by_link[entity.base_link]
                for env_idx in range(self.sim._B):
                    qpos_new[env_idx, :3], qpos_new[env_idx, 3:7] = gu.T_to_trans_quat(abd_entry[env_idx].transform)

            self.rigid_solver.set_qpos(
                qpos_new if self.sim.n_envs > 0 else qpos_new[0],
                qs_idx=slice(entity.q_start, entity.q_end),
                skip_forward=False,
            )

            # Set base link velocities from IPC if available
            if not entity.base_link.is_fixed:
                self._apply_base_link_velocity_from_ipc(entity)

            # Update ref_dof_prev for next timestep
            ad.ref_dof_prev[:] = ad.qpos_new

            # Store current link transforms to prev_links_transform
            for env_idx in range(self.sim._B):
                for child_link, prev_link_transform in zip(ad.joints_child_link, ad.prev_links_transform):
                    link_transform = self._abd_transforms_by_link[child_link][env_idx]
                    prev_link_transform[env_idx] = link_transform.copy()

    def _post_advance_ipc_only(self):
        """
        Post-advance processing for 'ipc_only' entities.

        This method directly sets Genesis transforms from IPC results. It only handles rigid objects.
        """
        if COUPLING_TYPE.IPC_ONLY not in self._entities_by_coup_type:
            return

        envs_qpos = np.empty((self.sim._B, 7), dtype=gs.np_float)
        for entity in self._entities_by_coup_type[COUPLING_TYPE.IPC_ONLY]:
            if entity.base_link.is_fixed:
                continue

            for env_idx in range(self.sim._B):
                abd_entry = self._abd_data_by_link[entity.base_link][env_idx]
                envs_qpos[env_idx, :3], envs_qpos[env_idx, 3:7] = gu.T_to_trans_quat(abd_entry.transform)

            self.rigid_solver.set_qpos(
                envs_qpos if self.sim.n_envs > 0 else envs_qpos[0],
                qs_idx=slice(entity.q_start, entity.q_start + 7),
                skip_forward=True,
            )

            # FIXME: It is currently necessary to enforce zero velocity to avoid double time integration by Rigid solver
            # self._apply_base_link_velocity_from_ipc(entity)
            self.rigid_solver.set_dofs_velocity(
                velocity=None, dofs_idx=slice(entity.dof_start, entity.dof_start + 6), skip_forward=True
            )

    def _retrieve_fem_states(self):
        # IPC world advance/retrieve is handled at Scene level
        # This method handles both volumetric FEM (3D) and cloth (2D) post-processing

        if not self.fem_solver.is_active:
            return

        # Gather FEM states (both volumetric and cloth) using metadata filtering
        visitor = SceneVisitor(self._ipc_scene)

        # Collect FEM and cloth geometries using metadata
        fem_positions_by_entity: dict["FEMEntity", list[np.ndarray]] = {
            entity: [np.array([]) for _ in range(self.sim._B)] for entity in self.fem_solver.entities
        }
        for fem_geom_slot in visitor.geometries():
            if not isinstance(fem_geom_slot, SimplicialComplexSlot):
                continue

            fem_geom = fem_geom_slot.geometry()
            if fem_geom.dim() not in (2, 3):
                continue
            meta = read_ipc_geometry_metadata(fem_geom)
            if meta is None:
                continue
            solver_type, env_idx, i_e = meta
            if solver_type not in ("fem", "cloth"):
                continue

            entity = cast("FEMEntity", self.fem_solver.entities[i_e])
            (transformed_geom,) = uipc.geometry.apply_transform(fem_geom)
            fem_positions_by_entity[entity][env_idx] = transformed_geom.positions().view().reshape(-1, 3)

        # Update FEM entities using filtered geometries
        for entity, geom_positions in fem_positions_by_entity.items():
            geom_positions = np.stack(geom_positions, axis=0, dtype=gs.np_float)
            entity.set_pos(0, geom_positions)

    def _retrieve_rigid_states(self):
        """
        Retrieve ABD transforms/affine matrices after IPC step using AffineBodyStateAccessorFeature.

        O(num_rigid_bodies) instead of O(total_geometries).
        Also populates data arrays for force computation.
        """
        if self._abd_state_feature is None:
            return

        # Single batch copy of ALL ABD states from IPC
        assert self._abd_state_geom is not None
        self._abd_state_feature.copy_to(self._abd_state_geom)

        # Get all transforms at once (array view)
        trans_attr = self._abd_state_geom.instances().find(uipc.builtin.transform)
        transforms = trans_attr.view()  # Shape: (num_bodies, 4, 4)

        # Get velocities (4x4 matrix representing transform derivative)
        vel_attr = self._abd_state_geom.instances().find(uipc.builtin.velocity)
        velocities = vel_attr.view()  # Shape: (num_bodies, 4, 4)

        assert self._coupling_data is not None
        for i_link, link in enumerate(self._coupling_data.links):
            for env_idx, abd_body_idx in enumerate(self._coupling_data.abd_body_idx_by_link[link]):
                self._abd_data_by_link[link][env_idx].transform[:] = transforms[abd_body_idx]
                self._abd_data_by_link[link][env_idx].velocity[:] = velocities[abd_body_idx]

                self._coupling_data.ipc_transforms[env_idx, i_link] = transforms[abd_body_idx]
                self._coupling_data.aim_transforms[env_idx, i_link] = self._abd_transforms_by_link[link][env_idx]

    def _store_gs_rigid_states(self):
        """
        Store current Genesis rigid body states before IPC advance.

        These stored states will be used by:
        1. Animator: to set aim_transform for IPC soft constraints
        2. Force computation: to ensure action-reaction force consistency
        3. User modification detection: to detect if user called set_qpos
        """
        if not self.rigid_solver.is_active:
            return

        # Store qpos for all entities. It will be used by 'external_articulation' coupling mode
        assert self.rigid_solver.qpos is not None
        entities_qpos = qd_to_numpy(self.rigid_solver.qpos, transpose=True)
        for entity, articulation_data in self._articulation_data_by_entity.items():
            articulation_data.qpos_stored[:] = entities_qpos[..., entity.q_start : entity.q_end]

        # Store transforms for all rigid links
        links_pos = qd_to_numpy(self.rigid_solver.dyn_state.links.pos, transpose=True)
        links_quat = qd_to_numpy(self.rigid_solver.dyn_state.links.quat, transpose=True)
        links_transform = cast(np.ndarray, gu.trans_quat_to_T(links_pos, links_quat))
        for link, transforms in self._abd_transforms_by_link.items():
            for env_idx in range(self.sim._B):
                transforms[env_idx][:] = links_transform[env_idx, link.idx]

    def _apply_abd_coupling_forces(self):
        """
        Apply coupling forces from IPC ABD constraint to Genesis rigid bodies.

        Data has already been populated in data by _retrieve_rigid_states, so this function computes forces and applies
        the results.

        This ensures action-reaction force consistency:
        - IPC constraint force: G_ipc = M * (q_ipc^{n+1} - q_genesis^n)
        - Genesis reaction force: F_genesis = M * (q_ipc^{n+1} - q_genesis^n) = G_ipc
        """
        if (
            not self.options.two_way_coupling
            or COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT not in self._entities_by_coup_type
            or not self._abd_data_by_link
        ):
            return

        assert self._coupling_data is not None
        update_coupling_forces(
            self._coupling_data.ipc_transforms,
            self._coupling_data.aim_transforms,
            self._coupling_data.links_mass,
            self._coupling_data.links_inertia_i,
            self._constraint_strength_translation_scaled,
            self._constraint_strength_rotation_scaled,
            self._coupling_data.out_forces,
            self._coupling_data.out_torques,
        )

        if np.isnan(self._coupling_data.out_forces).any() or np.isnan(self._coupling_data.out_torques).any():
            gs.raise_exception(
                "Invalid coupling forces/torques causing 'nan'. This indicates numerical instability. Please decrease "
                "the simulation timestep."
            )

        self.rigid_solver.apply_links_external_wrench(
            self._coupling_data.out_forces if self.sim.n_envs > 0 else self._coupling_data.out_forces[0],
            self._coupling_data.out_torques if self.sim.n_envs > 0 else self._coupling_data.out_torques[0],
            links_idx=self._coupling_data.links_idx,
        )
