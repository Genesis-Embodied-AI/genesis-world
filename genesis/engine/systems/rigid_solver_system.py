"""Manager System of the rigid solver, which owns its data and runs one substep as three graphs of stage Actions around
a host solve.

The initialization of the rigid data extends that of the kinematic data (see kinematic_solver_system.py) through the
functions at the end of this module.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.core import Data, HostAction, Require, StageAction, System, register_system
from genesis.engine.couplers import SAPCoupler
from genesis.engine.solvers.kinematic_solver import _balanced_variant_mapping
from genesis.engine.solvers.rigid.abd.diff import kernel_save_adjoint_cache
from genesis.engine.solvers.rigid.abd.forward_dynamics import (
    kernel_refresh_invweight_and_meaninertia,
    update_qacc_from_qvel_delta,
    update_qvel,
)
from genesis.engine.solvers.rigid.abd.forward_kinematics import kernel_update_cartesian_space
from genesis.engine.solvers.rigid.abd.misc import (
    kernel_init_entity_fields,
    kernel_init_equality_fields,
    kernel_init_geom_fields,
    kernel_init_link_dynamics,
    kernel_init_vert_fields,
    kernel_update_heterogeneous_link_info,
    kernel_wakeup_coupled_links,
)
from genesis.engine.solvers.rigid.collider.collider import create_collider_data
from genesis.engine.solvers.rigid.constraint import ConstraintSolver
from genesis.engine.solvers.rigid.rigid_solver import (
    _sanitize_sol_params,
    func_step_1,
    func_step_2,
    kernel_step_1,
    kernel_step_2,
)
from genesis.utils.misc import get_gpu_core_count, get_gpu_shared_tile_sizes, qd_to_numpy

from .kinematic_solver_system import (
    KINEMATIC_MODEL,
    adopt_model,
    init_dof_fields,
    init_link_fields,
    init_tree_fields,
    init_vfaces_raycast_mask,
    init_vgeom_fields,
    init_vverts_state,
    init_vvert_fields,
)
from .rigid_sdf_system import RigidSdfSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# The model of the rigid entities beyond the kinematic one (see KINEMATIC_MODEL), with the options the rigid solver has
# resolved
RIGID_MODEL = (
    "geoms",
    "n_geoms",
    "n_geoms_",
    "_n_geoms",
    "n_cells",
    "n_cells_",
    "_n_cells",
    "n_verts",
    "n_verts_",
    "_n_verts",
    "n_free_verts",
    "n_free_verts_",
    "_n_free_verts",
    "n_fixed_verts",
    "n_fixed_verts_",
    "_n_fixed_verts",
    "n_faces",
    "n_faces_",
    "_n_faces",
    "n_edges",
    "n_edges_",
    "_n_edges",
    "n_equalities",
    "_n_equalities",
    "n_candidate_equalities_",
    "_enable_multi_contact",
    "_enable_joint_limit",
    "_enable_self_collision",
    "_enable_neutral_collision",
    "_enable_adjacent_collision",
    "_disable_constraint",
    "_max_collision_pairs",
    "max_collision_pairs",
    "_integrator",
    "_box_box_detection",
    "_use_hibernation",
    "_hibernation_thresh_vel",
    "_sol_min_timeconst",
    "_sol_default_timeconst",
    "_contact_resolution",
    "_substep_dt",
)


# WORKAROUND: Quadrants discovers the tensors of the array_class structs reached from a kernel parameter only through a
# @qd.data_oriented object, so the structs a substep reads and writes are gathered here rather than bound one by one.
@qd.data_oriented
class RigidSolverData(Data):
    """The array_class structs of the rigid data, its collision detection and its constraint solve that a substep
    reads and writes."""

    def __init__(self, system: "RigidSolverSystem") -> None:
        collider, constraint_solver = system.collider_data, system.constraint_solver
        self.geoms_init_AABB = system.geoms_init_AABB
        self.dyn_state = system.dyn_state
        self.dyn_info = system.dyn_info
        self.rigid_info = system.rigid_info
        self.collider_state = collider.collider_state
        self.collider_info = collider.collider_info
        self.mpr_state = collider.mpr.mpr_state
        self.contact0_mpr_state = collider.mpr.contact0_mpr_state
        self.multicontact_mpr_state = collider.mpr.multicontact_mpr_state
        self.gjk_state = collider.gjk.gjk_state
        self.contact0_gjk_state = collider.gjk.contact0_gjk_state
        self.multicontact_gjk_state = collider.gjk.multicontact_gjk_state
        self.constraint_state = constraint_solver.constraint_state
        self.errno = system._errno


class RigidSolverSystem(System):
    """Manager of the rigid substep, owning the data of the rigid solver of the scene, which reaches it through
    forwarding attributes.

    The Initializer has it allocate and initialize the data from the entities the solver has built, including the
    collision detection and the constraint solve, whose state the rigid substep reads whether or not the scene enables
    them. The data exists whether or not the solver holds an entity, since the couplers and the MPM solver pass it to
    their kernels regardless.

    A substep runs three graphs around a host solve, which selects its kernels at run time:
        graph 1  the forward dynamics, then every equality assembly
        graph 2  every detection, then every inequality assembly
        host     every solve
        graph 3  every constraint force update, then the integration
    The work before the solve is cut ahead of the detection, since the compile time of a kernel grows faster than its
    size while one more graph replay costs next to nothing. The rigid-rigid contact and the constraint solve join
    through on_detect and on_constrain, from Systems the scene adds only when it enables them.
    """

    # The collision detection embeds the signed distance fields of the geoms in its collider info
    rigid_sdf_system = Require(RigidSdfSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.rigid_solver
        # Whether the solver holds an entity to advance, known once the scene has added its entities
        self.is_active = self.solver.is_active

    @property
    def sdf(self):
        """The signed distance fields of the rigid geoms (see RigidSdfSystem)."""
        return self.rigid_sdf_system.sdf

    def sanitize_joint_sol_params(self, sol_params):
        """Floor the time constant of the solver parameters of the joints at the one the substep resolves."""
        return _sanitize_sol_params(sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

    def refresh_invweight_and_meaninertia(self, envs_idx=None, *, force_update=True, in_place=False):
        """Recompute the inverse weights and the mean inertia from the inertial properties of the links."""
        # Every kinematic tree of the solver is recomputed. A change limited to some links is handled by the setter
        # writing them, which recomputes their own tree only.
        # Early return if no DoFs. This is essential to avoid segfault on CUDA.
        if self._n_dofs == 0:
            return

        # Handling default arguments
        batched = self._options.batch_dofs_info or self._options.batch_links_info
        if not batched and envs_idx is not None:
            gs.raise_exception(
                "Links and dofs must be batched to selectively update invweight and meaninertia for some environment."
            )
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        # The computation needs one vector of size n_dofs and one mass-matrix solve against it. Build time borrows
        # both from the constraint solver, so that a scene never writing an inertial property allocates no scratch of
        # its own.
        if in_place:
            constraint_state = self.constraint_solver.constraint_state
            jac_row, solve_out = constraint_state.grad, constraint_state.Mgrad
        else:
            jac_row, solve_out = self.data_manager.weight_scratch.jac_row, self.data_manager.weight_scratch.solve_out
        kernel_refresh_invweight_and_meaninertia(
            envs_idx,
            jac_row,
            solve_out,
            self.dyn_state,
            self.constraint_solver.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            force_update,
            self._is_forward_pos_updated,
            self._is_forward_vel_updated,
        )

    @System.action(kind="host")
    def initialize_model(self):
        return self.adopt_model

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    def adopt_model(self) -> None:
        """Take over the model of the entities from the solver, which the signed distance fields and the rigid data
        are sized from."""
        adopt_model(self, KINEMATIC_MODEL + RIGID_MODEL)
        self.solver._system = self
        self._is_forward_pos_updated = False
        self._is_forward_vel_updated = False
        self._is_backward = False

    def initialize_data(self) -> None:
        """Allocate and initialize the rigid data, which the solver forwards to from here on."""
        # The static configuration every graph of the substep is compiled against
        rigid_build_static_config(self)
        rigid_create_data_manager(self)
        init_dof_fields(self)
        init_vvert_fields(self)
        init_vgeom_fields(self)
        rigid_init_link_fields(self)
        rigid_init_tree_fields(self)
        rigid_init_entity_fields(self)
        init_vfaces_raycast_mask(self)
        init_vverts_state(self)
        rigid_init_dynamics(self)
        self.data = RigidSolverData(self)

    @System.action_collection(kind="host")
    def preparations(self):
        """Host Actions run before the first graph of every substep."""

    @System.action_collection(kind="stage")
    def equality_assemblies(self):
        """Stage Actions that follow the forward dynamics in the first graph."""

    @System.action_collection(kind="stage")
    def detections(self):
        """Stage Actions that open the second graph."""

    @System.action_collection(kind="stage")
    def inequality_assemblies(self):
        """Stage Actions that follow every detection in the second graph."""

    @System.action_collection(kind="host")
    def solves(self):
        """Host Actions run between the second and the third graph."""

    @System.action_collection(kind="stage")
    def force_updates(self):
        """Stage Actions that open the third graph, ahead of the integration."""

    @System.action_collection(kind="host")
    def kernel_detections(self):
        """Host Actions that detect the collisions of a substep launched kernel by kernel (see substep_kernels)."""

    @System.action_collection(kind="host", call_args=("envs_idx",))
    def resets(self):
        """Host Actions that restart what the given environments carry over from the previous substeps, such as the
        warm start of the collision detection, once the user sets their configuration."""

    @System.action_collection(kind="host", call_args=("envs_idx",))
    def clears(self):
        """Host Actions that clear everything the given environments carry over, once the user sets their state."""

    @System.protocol
    def on_detect(self, *, prepare: HostAction, detect: StageAction, detect_kernels: HostAction):
        """Join the rigid substep with a collision detection, calling it once from build().

        prepare         HostAction   `function(*bound_args)`, before the first graph
        detect          StageAction  `function(*bound_args)`, opening the second graph, once the forward dynamics has
                                     updated the cartesian space
        detect_kernels  HostAction   `function(*bound_args)`, in place of detect when substep_kernels launches the
                                     substep kernel by kernel

        No graph runs backward, since a scene tracking gradients advances the rigid solver through substep_kernels.
        """
        self.preparations += prepare
        self.detections += detect
        self.kernel_detections += detect_kernels

    @System.protocol
    def on_reset(self, *, reset: HostAction, clear: HostAction):
        """Join the reset of the rigid data with what a participant carries over between substeps, calling it once
        from build().

        reset  HostAction  `function(*bound_args, envs_idx)`, when the user sets the configuration of the environments
        clear  HostAction  `function(*bound_args, envs_idx)`, when the user sets the whole state of the environments
        """
        self.resets += reset
        self.clears += clear

    @System.protocol
    def on_constrain(
        self,
        *,
        prepare: HostAction,
        assemble_equalities: StageAction,
        assemble_inequalities: StageAction,
        solve: HostAction,
        update_forces: StageAction,
    ):
        """Join the rigid substep with a constraint solve, calling it once from build().

        prepare                HostAction   `function(*bound_args)`, before the first graph
        assemble_equalities    StageAction  `function(*bound_args)`, in the first graph, after the forward dynamics
        assemble_inequalities  StageAction  `function(*bound_args)`, in the second graph, after every detection
        solve                  HostAction   `function(*bound_args)`, between the second and the third graph
        update_forces          StageAction  `function(*bound_args)`, in the third graph, before the integration

        No graph runs backward, since a scene tracking gradients advances the rigid solver through substep_kernels.
        """
        self.preparations += prepare
        self.equality_assemblies += assemble_equalities
        self.inequality_assemblies += assemble_inequalities
        self.solves += solve
        self.force_updates += update_forces

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.substep

    @System.action(kind="host")
    def post_coupling(self):
        return self.substep_post_coupling

    def reset(self, envs_idx=None) -> None:
        """Restart what the given environments carry over from the previous substeps, once their configuration is
        set."""
        for action in self.resets.actions:
            action.invoke((envs_idx,))

    def clear(self, envs_idx=None) -> None:
        """Clear everything the given environments carry over, once their whole state is set."""
        for action in self.clears.actions:
            action.invoke((envs_idx,))

    def substep(self, f: int) -> None:
        """Advance the rigid solver by substep f."""
        if isinstance(self.sim.coupler, SAPCoupler) or self._requires_grad:
            self.substep_kernels(f)
            return
        self.wakeup_coupled_links()
        for action in self.preparations.actions:
            action.invoke()
        kernel_substep_dynamics(self, self._is_forward_pos_updated, self._is_forward_vel_updated)
        kernel_substep_collision(self)
        for action in self.solves.actions:
            action.invoke()
        kernel_substep_post(self)
        self.mark_forward_updated()

    # FIXME: quadrants#946 - a graph refuses the ndarrays that own a gradient, so a scene tracking gradients launches
    # the substep kernels one by one.
    def substep_kernels(self, f: int) -> None:
        """Advance substep f by launching its kernels one by one.

        A SAP-coupled substep replaces the constraint solve by the coupler's own, so it stops once the velocities are
        updated, and substep_post_coupling completes it.
        """
        if self._requires_grad and f == 0:
            kernel_save_adjoint_cache(f, self.dyn_state, self._rigid_adjoint_cache, self.rigid_info, self.rigid_config)

        self.wakeup_coupled_links()
        kernel_step_1(
            self.dyn_state,
            self.constraint_solver.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            self._is_forward_pos_updated,
            self._is_forward_vel_updated,
            self._is_backward,
        )
        if isinstance(self.sim.coupler, SAPCoupler):
            update_qvel(self.dyn_state, self.rigid_info, self.rigid_config)
            return
        self.constraint_force()
        kernel_step_2(
            self.dyn_state,
            self.constraint_solver.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            self._is_backward,
            self._errno,
        )
        self.mark_forward_updated()
        if self._requires_grad:
            kernel_save_adjoint_cache(
                f + 1, self.dyn_state, self._rigid_adjoint_cache, self.rigid_info, self.rigid_config
            )

    def constraint_force(self) -> None:
        """Assemble and solve the constraints of a substep kernel by kernel, from the collisions it detects."""
        if not self._disable_constraint:
            self.constraint_solver.add_equality_constraints()

        for action in self.kernel_detections.actions:
            action.invoke()

        if not self._disable_constraint:
            self.constraint_solver.add_inequality_constraints()
            self.constraint_solver.resolve()

    def substep_post_coupling(self, f: int) -> None:
        """Complete substep f from the velocities a SAP coupler has solved for, which substep_kernels stops short of."""
        if isinstance(self.sim.coupler, SAPCoupler):
            update_qacc_from_qvel_delta(self.dyn_state, self.rigid_info, self.rigid_config)
            kernel_step_2(
                self.dyn_state,
                self.constraint_solver.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                self._is_backward,
                self._errno,
            )

    def wakeup_coupled_links(self) -> None:
        """Wake up the hibernated links that the coupling forces of the previous coupling phase target.

        See kernel_wakeup_coupled_links in abd/misc.py. Such forces only exist when another solver is active.
        """
        if self._use_hibernation and len(self.sim.active_solvers) > 1:
            kernel_wakeup_coupled_links(
                self.dyn_state,
                self.constraint_solver.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

    def mark_forward_updated(self) -> None:
        """Record that the integration has updated the cartesian space and the velocities the next substep starts
        from, which the MuJoCo-compatible mode recomputes at the start of every substep instead."""
        self._is_forward_pos_updated = not self._enable_mujoco_compatibility
        self._is_forward_vel_updated = not self._enable_mujoco_compatibility


@register_system(SimulatorEngine)
def create_rigid_solver_system(scene: "Scene") -> RigidSolverSystem:
    """Create the manager of the rigid substep, owning the rigid data, which every scene has."""
    return RigidSolverSystem(scene)


@qd.func(requires_top_level=True)
def func_substep_forward_dynamics(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
):
    """Run the forward dynamics of a substep, after updating the cartesian space and the velocities unless they are
    already up to date."""
    func_step_1(
        rigid_data.dyn_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
        is_forward_pos_updated,
        is_forward_vel_updated,
        False,
    )


@qd.func(requires_top_level=True)
def func_substep_integrate(rigid_data: qd.template(), rigid_config: qd.template()):
    """Integrate a substep, then update the cartesian space and the velocities the next substep starts from."""
    func_step_2(
        rigid_data.dyn_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
        False,
        rigid_data.errno,
    )


# Each graph is a kernel taking the System, since Quadrants discovers the Data of every Action through it, and unrolls
# its frozen collections at compile time.
# FIXME: quadrants#966 - fastcache keys a kernel on its arguments without the constants an Action binds, so it could
# load a kernel compiled for another configuration of the solver. These kernels stay out of it until then.
@qd.kernel(graph=True)
def kernel_substep_dynamics(
    rigid_solver_system: qd.template(), is_forward_pos_updated: qd.template(), is_forward_vel_updated: qd.template()
):
    func_substep_forward_dynamics(
        rigid_solver_system.data, rigid_solver_system.rigid_config, is_forward_pos_updated, is_forward_vel_updated
    )
    for action in qd.static(rigid_solver_system.equality_assemblies.actions):
        action.invoke()


@qd.kernel(graph=True)
def kernel_substep_collision(rigid_solver_system: qd.template()):
    for action in qd.static(rigid_solver_system.detections.actions):
        action.invoke()
    for action in qd.static(rigid_solver_system.inequality_assemblies.actions):
        action.invoke()


@qd.kernel(graph=True)
def kernel_substep_post(rigid_solver_system: qd.template()):
    for action in qd.static(rigid_solver_system.force_updates.actions):
        action.invoke()
    func_substep_integrate(rigid_solver_system.data, rigid_solver_system.rigid_config)


def rigid_build_static_config(system):
    # The scene has multi-island block structure when it holds several independent DOF-carrying bodies or free
    # joints (the Hessian then splits into per-island blocks instead of one dense tree), where the sparse Jacobian
    # of the CPU solve pays off; a single dense-coupled tree (e.g. one big robot) gains nothing from it.
    n_dof_entities = sum(entity.n_dofs > 0 for entity in system.entities)
    n_free_joints = sum(joint.type == gs.JOINT_TYPE.FREE for joint in system.joints)
    has_multi_island_structure = n_dof_entities >= 2 or n_free_joints >= 2

    # sparse_solve=None resolves automatically: the sparse Jacobian makes the per-iteration Jacobian-vector
    # products, the constraint-to-island lookup and the per-island Hessian assembly cost O(nonzeros) instead of
    # O(n_constraints * n_dofs), which pays off on CPU when the scene has block structure, whereas a single
    # dense-coupled tree gains nothing. An explicit value overrides this on CPU. On GPU the dense tiled path is
    # faster, so the sparse representation is dropped there whatever the option says.
    if system._options.sparse_solve is None:
        sparse_solve = gs.backend == gs.cpu and not system._enable_mujoco_compatibility and has_multi_island_structure
    else:
        sparse_solve = system._options.sparse_solve
        if sparse_solve and gs.backend != gs.cpu:
            gs.logger.warning(
                "'sparse_solve' is ignored on the GPU backend, where the dense tiled factorization is faster."
            )
    sparse_solve = sparse_solve and gs.backend == gs.cpu

    # Under the elliptic cone any middle-zone contact invalidates the Cholesky factor every Newton iteration, so
    # a contact-dense solve rebuilds it nearly every iteration; persisting the cone-free assembled Hessian turns
    # each rebuild into an envelope copy (see the nt_H declaration in array_class.py for the packed storage).
    # Confined to the CPU per-island skyline path that owns per-iteration rebuilds, the GPU arms rebuilding through
    # the tiled assembly.
    enable_cone_free_hessian_reuse = (
        system._options.friction_cone == gs.friction_cone.elliptic
        and gs.backend == gs.cpu
        and sparse_solve
        and system._options.constraint_solver == gs.constraint_solver.Newton
    )

    # The tiled per-island seed (see enable_tiled_island_seed in array_class.py) replaces a per-env thread walking
    # O(n^3) dependent loads, slower than the whole Newton iteration above the core count. The cooperative body
    # kernels run one warp per env, which pays off once an env carries enough dofs to keep its lanes busy, and stops
    # paying once the envs alone saturate the GPU. Both are dense-only. Outside performance mode the ndarray kernels
    # serve every scene shape, so the static config drops the scene bounds whose generic value costs little.
    is_generic = gs.use_ndarray
    enable_tiled_island_seed = (
        gs.backend != gs.cpu
        and not system.sim.options.requires_grad
        and not sparse_solve
        and (system.n_dofs >= 16 or (is_generic and system.n_dofs > 0))
    )
    enable_cooperative_constraint_kernels = enable_tiled_island_seed and (
        is_generic or system._sim._B <= get_gpu_core_count()
    )
    # The noslip sweep of an island is the last one-thread process of the cooperative regime, so its block-per-island
    # variant takes the same bound (see func_noslip in noslip.py).
    enable_cooperative_noslip = enable_cooperative_constraint_kernels and system._options.noslip_iterations > 0
    # Dofs per kinematic tree, counted from the links here since rigid_init_tree_fields runs once the fields this config
    # sizes are allocated. A scene holding one tree forms at most one island per env, and the per-island passes
    # read the env's plain ranges.
    tree_links = np.flatnonzero(system._links_tree_root_idx >= 0)
    trees_n_dofs = np.bincount(
        system._links_tree_root_idx[tree_links], np.array([link.n_dofs for link in system.links])[tree_links]
    )
    is_single_island = system._n_trees == 1 and not is_generic
    # Above the cooperative bound one thread per env saturates the GPU, where the scalar dense Cholesky of an env's
    # one block beats the tiled factor, so the seed kernel assembles the block and the monolith factors it.
    has_scalar_seed_factor = enable_tiled_island_seed and is_single_island and not enable_cooperative_constraint_kernels
    constraint_layout_batch_first = enable_cooperative_constraint_kernels or system.sim._para_level < gs.PARA_LEVEL.ALL

    # The level sweep (see func_sweep_links_by_level) syncs the lanes of a block, which neither the CPU nor the
    # differentiable runs offer. The serial walks run one thread per tree or root and env through all the links of
    # its group, so they take as long as the longest group. The lanes of an env step through a level in chunks of
    # their count, so the sweep stays within that time while its lane steps over all the levels and envs fit in the
    # cores times the longest group. Past that bound it only adds instructions to a busy GPU, whether the links are
    # grouped by tree or by root.
    enable_level_sweep = gs.backend != gs.cpu and not system.sim.options.requires_grad
    if enable_level_sweep and not is_generic:
        n_lanes_per_env = array_class.RigidSimStaticConfig.level_sweep_n_lanes_per_env
        for links_group_idx, links_level in (
            (system._links_tree_root_idx, system._links_tree_level),
            (system._links_root_idx, system._links_root_level),
        ):
            group_links = np.flatnonzero(links_level >= 0)
            if group_links.size:
                _, groups_n_links = np.unique(links_group_idx[group_links], return_counts=True)
                _, levels_n_links = np.unique(
                    links_group_idx[group_links] * system.n_links + links_level[group_links], return_counts=True
                )
                n_lane_steps = n_lanes_per_env * (-(-levels_n_links // n_lanes_per_env)).sum() * system._sim._B
                enable_level_sweep &= bool(n_lane_steps <= groups_n_links.max() * get_gpu_core_count())

    rigid_config = dict(
        backend=gs.backend,
        para_level=system.sim._para_level,
        requires_grad=system.sim.options.requires_grad,
        use_hibernation=system._use_hibernation,
        batch_links_info=system._options.batch_links_info,
        batch_dofs_info=system._options.batch_dofs_info,
        batch_joints_info=system._options.batch_joints_info,
        enable_mujoco_compatibility=system._enable_mujoco_compatibility,
        enable_elliptic_friction=system._options.friction_cone == gs.friction_cone.elliptic,
        enable_signorini_contact=system._contact_resolution == gs.contact_resolution.signorini,
        enable_torsional_friction=system._options.enable_torsional_friction,
        enable_rolling_friction=system._options.enable_rolling_friction,
        enable_multi_contact=system._enable_multi_contact,
        enable_collision=system._enable_collision,
        enable_joint_limit=system._enable_joint_limit,
        box_box_detection=system._box_box_detection,
        sparse_solve=sparse_solve,
        enable_cone_free_hessian_reuse=enable_cone_free_hessian_reuse,
        integrator=system._integrator,
        solver_type=system._options.constraint_solver,
        broadphase_traversal=resolve_broadphase_traversal(system),
        # Parallelize init over (constraints, envs) when envs alone don't saturate the GPU.
        parallel_init=(
            gs.backend != gs.cpu
            and not system.sim.options.requires_grad
            and (is_generic or system.n_envs <= get_gpu_core_count())
        ),
        enable_level_sweep=enable_level_sweep,
        enable_tiled_island_seed=enable_tiled_island_seed,
        enable_cooperative_constraint_kernels=enable_cooperative_constraint_kernels,
        enable_cooperative_noslip=enable_cooperative_noslip,
        is_single_island=is_single_island,
        has_scalar_seed_factor=has_scalar_seed_factor,
        constraint_layout_batch_first=constraint_layout_batch_first,
    )

    # Prefer the monolith solver on CPU (always faster there, perf dispatch is a waste of effort)
    if gs.backend == gs.cpu or system.sim.options.requires_grad:
        rigid_config["prefer_decomposed_solver"] = 0

    # Per-DOF mass-block bounds (see dofs_mass_block_start in array_class.py), computed here because the tiled
    # factor arms below are sized for the largest block; rigid_init_tree_fields uploads them.
    links_by_idx = {link.idx: link for link in system.links}
    dofs_mass_block_start = np.arange(system.n_dofs_, dtype=gs.np_int)
    for link in system.links:
        if link.n_dofs == 0:
            continue
        root_link = link
        node = link
        while node.parent_idx != -1:
            node = links_by_idx[node.parent_idx]
            if node.n_dofs > 0:
                root_link = node
        dofs_mass_block_start[link.dof_start : link.dof_end] = root_link.dof_start
    dofs_mass_block_end = np.empty(system.n_dofs_, dtype=gs.np_int)
    for i_d in range(system.n_dofs_):
        dofs_mass_block_end[dofs_mass_block_start[i_d]] = i_d + 1
    dofs_mass_block_end = dofs_mass_block_end[dofs_mass_block_start]

    # Blocks form clean intervals: attach() is the only source of cross-entity blocks and rejects any layout that
    # would interleave foreign DOFs inside a merged one (see RigidEntity.attach).
    system._dofs_mass_block_start = dofs_mass_block_start
    system._dofs_mass_block_end = dofs_mass_block_end

    if system.is_active:
        # The tiled and cooperative Cholesky kernels trade per-env serial work for cross-lane parallelism, so they
        # only help while envs alone do not already saturate the GPU. Above that env count one-thread-per-env keeps
        # every core busy and the scalar path wins; below it the parallel kernels hide latency by swapping warps.
        # The crossover is also hardware- and kernel-dependent, so the env threshold (GPU core count) is a heuristic
        # and a dynamic timer-based selection would be more accurate still.
        # Largest mass block in DOFs, the unit the per-block factor is sized for: merged entities can make a block
        # exceed any single entity (see attach).
        max_block_dofs = int((dofs_mass_block_end - dofs_mass_block_start).max()) if system.n_dofs else 0
        if gs.backend != gs.cpu:
            max_tiled_envs = get_gpu_core_count()
            envs_undersaturate = system.n_envs <= max_tiled_envs

            # Register tile of the mass-matrix Cholesky kernels and unit of the island tile caps, see
            # cholesky_tile_size_for in array_class.py.
            cholesky_tile_size = 32 if is_generic else array_class.cholesky_tile_size_for(system.n_dofs)

            # A shared tile holds the dofs it factors rounded up to the next size of get_gpu_shared_tile_sizes,
            # which bounds the sizes compiled while keeping the occupancy of the tightest fit. A mass block above
            # the largest size factors outside the shared tile, which leaves the tile size unused.
            shared_tile_sizes = get_gpu_shared_tile_sizes(2 if is_generic else 8)
            mass_matrix_fits_shared = max_block_dofs <= shared_tile_sizes[-1]
            tiled_n_dofs_per_block = min(
                (size for size in shared_tile_sizes if size >= max_block_dofs), default=shared_tile_sizes[-1]
            )

            # The cooperative per-island solve stages one island's tile in shared memory, in size classes (see
            # island_tile_cap_first in array_class.py): the last cap is the shared tile of the dof count, or the
            # largest one where the dofs exceed it. An island holds at least one tree, so a class below the
            # smallest tree never holds an island: the first cap starts at that tree.
            island_tile_cap_last = min(
                (size for size in shared_tile_sizes if size >= system.n_dofs), default=shared_tile_sizes[-1]
            )
            min_tree_dofs = trees_n_dofs[trees_n_dofs > 0].min() if system.n_dofs else 0
            island_tile_cap_first = cholesky_tile_size
            while island_tile_cap_first < min(min_tree_dofs, island_tile_cap_last):
                island_tile_cap_first *= 2
            island_tile_cap_first = min(island_tile_cap_first, island_tile_cap_last)

            # The cooperative in-place LDL^T has no cap; the shared-memory tile is faster but capped. Same env logic
            # as the Hessian: tile from the largest block >= 8 DOFs, drop the env guard above the cap where the
            # scalar O(n_block_dofs^3) per-(block, env) factor is always worse. Outside performance mode every scene
            # holding dofs takes the tiled factor, whatever its block size and env count.
            enable_tiled_cholesky_mass_matrix = (
                max_block_dofs > 0
                if is_generic
                else max_block_dofs >= 8 and (not mass_matrix_fits_shared or envs_undersaturate)
            )

            # Register-streaming tiled mass factor for the >shared-cap forward GPU path: factors each mass
            # block in registers via the same primitive as the Hessian, faster than and numerically matching the
            # cooperative LDL^T. Reuses cholesky_tile_size (always 32 here).
            enable_register_tiled_mass = (
                enable_tiled_cholesky_mass_matrix and not mass_matrix_fits_shared and not system._requires_grad
            )

            # The shared-memory mass factor solves the smooth acceleration while it still holds the factor. A
            # differentiable scene keeps the separate solve, which kernel_manual_compute_qacc_bw reverses.
            enable_fused_smooth_acc_solve = (
                enable_tiled_cholesky_mass_matrix and mass_matrix_fits_shared and not system._requires_grad
            )

            rigid_config.update(
                enable_tiled_cholesky_mass_matrix=enable_tiled_cholesky_mass_matrix,
                mass_matrix_fits_shared=mass_matrix_fits_shared,
                enable_register_tiled_mass=enable_register_tiled_mass,
                enable_fused_smooth_acc_solve=enable_fused_smooth_acc_solve,
                cholesky_tile_size=cholesky_tile_size,
                tiled_n_dofs_per_block=tiled_n_dofs_per_block,
                island_tile_cap_first=island_tile_cap_first,
                island_tile_cap_last=island_tile_cap_last,
                # An island holds at most every dof of the scene, so below this bound the factor paths above the
                # last cap (see func_island_assemble_factor_solve_tiled) are dead code and stay uncompiled.
                has_island_above_tile_cap=system.n_dofs > island_tile_cap_last,
            )

            # One thread block assembles and factors an island above the last cap for the Newton solver (see
            # func_island_assemble_factor_solve_tiled), at a cost growing with the cube of its dof count whatever
            # the environment count, where the skyline factor of the CPU backend exploits the sparsity of the
            # contact graph.
            if (
                system._options.constraint_solver == gs.constraint_solver.Newton
                and system.n_dofs > 4 * island_tile_cap_last
            ):
                gs.logger.warning(
                    f"The scene holds {system.n_dofs} dofs. A contact island wider than {4 * island_tile_cap_last} "
                    "dofs takes seconds per step with the Newton constraint solver on the GPU backend. If the "
                    "bodies can pile up into one island, use the CPU backend or the CG constraint solver."
                )

            # Manually pin the solve arm only where the winner is determinable in advance AND confirmed across
            # CUDA + Metal; genuinely backend-dependent cases fall through to the per-step autotuner.
            if not enable_cooperative_constraint_kernels:
                # No cooperative layout (n_envs > 8192 already saturates the GPU, or n_dofs < 16): the decomposed
                # arm has nothing to exploit, so the scalar one-thread-per-env monolith is the clear winner.
                rigid_config["prefer_decomposed_solver"] = 0

            # The autotuner picks between two numerically distinct arms by timing them as the simulation runs, so
            # which one runs at a given step follows the machine rather than the scene. Pinning the arm is what
            # makes a trajectory reproducible; the decomposed one takes whatever the static scene description
            # leaves open, every case that description settles in the monolith's favor being pinned above already.
            if gs.use_deterministic_algorithms and rigid_config.get("prefer_decomposed_solver", -1) == -1:
                rigid_config["prefer_decomposed_solver"] = 1

    # Jacobi equilibration of the Newton system (see nt_jacobi in array_class.py): every factor, incremental
    # update and solve of every arm rides the scaled coordinates; the reference behaviour keeps the raw factor.
    # Enabled when the model's mass-diagonal spread bound (_jacobi_mass_spread_bound) exceeds what the working
    # precision's Cholesky factor conditions accurately; the double-precision threshold carries the mantissa
    # headroom over single, past any scene built from physical units.
    mass_spread_threshold = 1e4 if gs.qd_float == qd.f32 else 5e12
    rigid_config["enable_jacobi_equilibration"] = (
        not system._enable_mujoco_compatibility
        and rigid_config["solver_type"] == gs.constraint_solver.Newton
        and jacobi_mass_spread_bound(system) > mass_spread_threshold
    )
    system.rigid_config = array_class.RigidSimStaticConfig(**rigid_config)

    if system.rigid_config.requires_grad:
        if system.rigid_config.use_hibernation:
            gs.raise_exception("Hibernation is not supported yet when requires_grad is True")
        if system.rigid_config.integrator != gs.integrator.approximate_implicitfast:
            gs.raise_exception("Only approximate_implicitfast integrator is supported yet when requires_grad is True.")
        from genesis.engine.couplers import SAPCoupler, IPCCoupler

        if isinstance(system.sim.coupler, (SAPCoupler, IPCCoupler)):
            gs.raise_exception(f"{type(system.sim.coupler).__name__} is not supported yet when requires_grad is True.")

        if system._options.noslip_iterations > 0:
            gs.raise_exception("Noslip is not supported yet when requires_grad is True.")
        if system._options.enable_torsional_friction or system._options.enable_rolling_friction:
            gs.raise_exception("Torsional and rolling friction are not supported yet when requires_grad is True.")


def rigid_create_data_manager(system):
    # We initialize data even if the solver is not active because the coupler needs arguments like
    # rigid_solver.dyn_state.links, etc. regardless of the solver is active or not.
    solver = system.solver
    system.data_manager = array_class.DataManager(system, kinematic_only=False)
    system._errno = system.data_manager.errno

    system.rigid_info = system.data_manager.rigid_info
    system._rigid_adjoint_cache = system.data_manager.rigid_adjoint_cache
    system.dyn_info = system.data_manager.dyn_info
    system.dyn_state = system.data_manager.dyn_state
    system.kinematics_scratch = system.data_manager.kinematics_scratch
    if system._use_hibernation:
        system.n_awake_dofs = system.rigid_info.n_awake_dofs
    if system._requires_grad:
        system.dyn_state_adjoint_cache = system.data_manager.dyn_state_adjoint_cache

    # Gravity lives with the rigid arrays, which the kernels read, so that array is the one handed over to hold it.
    solver._build_gravity(system.rigid_info.gravity)


def rigid_init_tree_fields(system):
    """Initialize the fields describing the kinematic trees, the structure of the joint-space mass matrix
    included."""
    init_tree_fields(system)

    system.mass_mat = system.rigid_info.mass_mat
    system.mass_mat_L = system.rigid_info.mass_mat_L
    system.mass_mat_D_inv = system.rigid_info.mass_mat_D_inv
    system.meaninertia = system.rigid_info.meaninertia

    # tree structure information
    mass_parent_mask = np.zeros((system.n_dofs_, system.n_dofs_), dtype=gs.np_float)
    for i_l in range(system.n_links):
        j_l = i_l
        while j_l != -1:
            for i_d, j_d in qd.ndrange(
                (system.links[i_l].dof_start, system.links[i_l].dof_end),
                (system.links[j_l].dof_start, system.links[j_l].dof_end),
            ):
                mass_parent_mask[i_d, j_d] = 1.0
            j_l = system.links[j_l].parent_idx

    # Per-DOF mass-block bounds, computed by rigid_build_static_config (which sizes the tiled factor arms from them).
    dofs_mass_block_start = system._dofs_mass_block_start
    dofs_mass_block_end = system._dofs_mass_block_end

    # An aligned free body whose only DOFs are its own free joint has a diagonal joint-space mass block, so zero its
    # within-link off-diagonal mask to make the assembled mass exactly diagonal (else ~1e-6 round-off once it
    # rotates) and the skyline envelope tighter. A DOF-bearing (articulated) descendant adds off-diagonal base
    # coupling, so the block must be exactly the root's own DOFs for the diagonalization to be valid.
    # Writing a center of mass or an inertia on such a link is rejected, so the block stays valid for good (see
    # _set_links_info).
    for link in system.links:
        # 'aligned' already implies a free joint; the block bounds must additionally be exactly the link's own DOFs
        # (no DOF-bearing ancestor or descendant), otherwise the coupled block is not diagonal.
        if (
            not link.aligned
            or dofs_mass_block_start[link.dof_start] != link.dof_start
            or dofs_mass_block_end[link.dof_start] != link.dof_end
        ):
            continue
        for i_d in range(link.dof_start, link.dof_end):
            for j_d in range(link.dof_start, link.dof_end):
                if i_d != j_d:
                    mass_parent_mask[i_d, j_d] = 0.0
            dofs_mass_block_start[i_d] = i_d
            dofs_mass_block_end[i_d] = i_d + 1

    # Smallest dof structurally coupled to each dof by the mass matrix (itself when none lies below it), read by the
    # skyline envelope of the per-island solver (see dof_env_start_local in array_class.py).
    dofs_mass_envelope_start = ((mass_parent_mask + mass_parent_mask.T) > 0.5).argmax(axis=1).astype(gs.np_int)
    system.rigid_info.mass_parent_mask.from_numpy(mass_parent_mask)
    system.rigid_info.dofs_mass_envelope_start.from_numpy(dofs_mass_envelope_start)
    system.rigid_info.dofs_mass_block_start.from_numpy(dofs_mass_block_start)
    system.rigid_info.dofs_mass_block_end.from_numpy(dofs_mass_block_end)


def rigid_dispatch_heterogeneous_vgeoms(system):
    """
    Dispatch per-environment geom/vgeom ranges and inertial properties for heterogeneous links.

    Extends the base class (which handles vgeom-only dispatch) to also dispatch collision geom
    ranges and per-variant inertial properties. Per-variant inertial is pre-computed during
    link._build() from actual geom objects, using analytic formulas for primitives.
    """
    for link in system.links:
        if link._variant_vgeom_ranges is None:
            continue

        n_variants = len(link._variant_vgeom_ranges)
        variant_idx = _balanced_variant_mapping(n_variants, system._B)

        # Build per-env arrays from link's variant data
        geom_starts = np.array([link._variant_geom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
        geom_ends = np.array([link._variant_geom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)
        vgeom_starts = np.array([link._variant_vgeom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
        vgeom_ends = np.array([link._variant_vgeom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)

        # Build per-env inertial arrays from pre-computed per-variant inertial
        links_inertial_mass = np.array([link._variant_inertial[v][0] for v in variant_idx], dtype=gs.np_float)
        links_inertial_pos = np.array([link._variant_inertial[v][1] for v in variant_idx], dtype=gs.np_float)
        links_inertial_quat = np.array([link._variant_inertial[v][2] for v in variant_idx], dtype=gs.np_float)
        links_inertial_i = np.array([link._variant_inertial[v][3] for v in variant_idx], dtype=gs.np_float)

        # Update links_info with per-environment values
        # Note: when batch_links_info is True, the shape is (n_links, B)
        kernel_update_heterogeneous_link_info(
            link.idx,
            geom_starts,
            geom_ends,
            vgeom_starts,
            vgeom_ends,
            links_inertial_mass,
            links_inertial_pos,
            links_inertial_quat,
            links_inertial_i,
            system.dyn_info,
        )

        # Set active_envs on geoms — indicates which environments each geom is active in
        for geom in link.geoms:
            active_envs_mask = (geom_starts <= geom.idx) & (geom.idx < geom_ends)
            geom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
            (geom.active_envs_idx,) = np.where(active_envs_mask)

        # Set active_envs on vgeoms
        for vgeom in link.vgeoms:
            active_envs_mask = (vgeom_starts <= vgeom.idx) & (vgeom.idx < vgeom_ends)
            vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
            (vgeom.active_envs_idx,) = np.where(active_envs_mask)


def rigid_init_link_fields(system):
    # The base initialization ends by dispatching each heterogeneous variant's inertial per environment, which
    # must write last, so this runs first.
    if system.links:
        links = system.links
        kernel_init_link_dynamics(
            np.array([link.desc.invweight for link in links], dtype=gs.np_float),
            np.array([link.desc.inertial_pos for link in links], dtype=gs.np_float),
            np.array([link.desc.inertial_quat for link in links], dtype=gs.np_float),
            np.array([link.desc.inertia for link in links], dtype=gs.np_float),
            np.array([link.desc.mass for link in links], dtype=gs.np_float),
            system.dyn_info,
        )

    init_link_fields(system, rigid_dispatch_heterogeneous_vgeoms)


def rigid_init_vert_fields(system):
    if system.n_verts > 0:
        geoms = system.geoms
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
            system.dyn_info,
            system.rigid_config,
        )


def rigid_init_dof_length(system):
    # Characteristic length of each dof (1 for translation, the body radius for rotation), used to weight dof
    # velocities in the hibernation rest test. Computed here, after geom dispatch, because a heterogeneous entity's
    # per-variant geoms (and hence body radius) are only assigned to environments at that point. Only needed when
    # hibernation is on.
    if not system._use_hibernation:
        return

    joints = system.joints
    if sum(joint.n_dofs for joint in joints) == 0:
        return

    # dofs_length is per-env only for a heterogeneous entity; broadcast the shared row across envs when dofs_info
    # is batched (always so for a heterogeneous entity, optionally for a homogeneous one).
    dof_length = np.concatenate([joint.dofs_length for joint in joints], axis=0)
    if system._options.batch_dofs_info and dof_length.ndim == 1:
        dof_length = np.broadcast_to(dof_length[:, None], (len(dof_length), system._B))
    system.dyn_info.dofs.dof_length.from_numpy(dof_length)


def rigid_init_geom_fields(system):
    system.geoms_init_AABB = system.rigid_info.geoms_init_AABB

    if system.n_geoms > 0:
        geoms = system.geoms
        geoms_sol_params = np.array([geom.desc.sol_params for geom in geoms], dtype=gs.np_float)
        # A geom keeps the time constant the model states; the floor lands on the value a contact mixes out of its
        # two geoms (see func_set_contact_data), the value the solver actually consumes. Flooring per geom would
        # raise the mix of a pair that is already above the floor, distorting the stated stiffness with no
        # stability benefit.
        _sanitize_sol_params(
            geoms_sol_params, system._sol_min_timeconst, system._sol_default_timeconst, floor_timeconst=False
        )

        # Accurately compute the center of mass of each geometry if possible.
        # Note that the mean vertex position is a bad approximation, which is impeding the ability of MPR to
        # estimate the exact contact information.
        geoms_center = []
        for geom in geoms:
            tmesh = geom.mesh.trimesh
            if tmesh.is_watertight:
                geoms_center.append(tmesh.center_mass)
            else:
                # Still fallback to mean vertex position if no better option...
                geoms_center.append(np.mean(tmesh.vertices, axis=0))

        # A geom is hollow when its own center lies in a cavity rather than inside its material (bowl, mug,
        # nut), i.e. its own SDF is positive at its center. This is a static property of the collision
        # geometry, precomputed here so the narrowphase never has to probe it at runtime. SPHERE/PLANE/TERRAIN
        # SDFs are analytic and convex geoms enclose their own center, so neither is ever hollow and probing
        # them would only force their SDF grid to be built.
        geoms_is_hollow = []
        for geom, center in zip(geoms, geoms_center):
            is_hollow = False
            if geom.type not in (gs.GEOM_TYPE.SPHERE, gs.GEOM_TYPE.PLANE, gs.GEOM_TYPE.TERRAIN) and not geom.is_convex:
                grid_pos = geom.T_mesh_to_sdf[:3, :3] @ center + geom.T_mesh_to_sdf[:3, 3]
                cell = np.minimum(np.maximum(np.floor(grid_pos).astype(gs.np_int), 0), geom.sdf_res - 2)
                frac = grid_pos - cell
                corners = geom.sdf_val[cell[0] : cell[0] + 2, cell[1] : cell[1] + 2, cell[2] : cell[2] + 2]
                weights_x, weights_y, weights_z = ([1.0 - frac[i], frac[i]] for i in range(3))
                sd_center = np.einsum("i,j,k,ijk->", weights_x, weights_y, weights_z, corners)
                is_hollow = sd_center > gs.EPS
            geoms_is_hollow.append(is_hollow)

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
            geoms_sol_params,
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
            np.array(geoms_is_hollow, dtype=gs.np_bool),
            system.geoms_init_AABB,
            system.dyn_state,
            system.dyn_info,
            system.rigid_config,
        )


def rigid_init_entity_fields(system):
    if system._entities:
        entities = system._entities
        kernel_init_entity_fields(
            np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
            np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
            np.array([entity.link_start for entity in entities], dtype=gs.np_int),
            np.array([entity.link_end for entity in entities], dtype=gs.np_int),
            np.array([entity.geom_start for entity in entities], dtype=gs.np_int),
            np.array([entity.geom_end for entity in entities], dtype=gs.np_int),
            np.array([entity.gravity_compensation for entity in entities], dtype=gs.np_float),
            np.array([entity.is_local_collision_mask for entity in entities], dtype=gs.np_bool),
            system.dyn_state,
            system.dyn_info,
            system.rigid_info,
            system.rigid_config,
        )


def rigid_init_equality_fields(system):
    if system.n_equalities > 0:
        equalities = system.equalities

        equalities_sol_params = np.array([equality.desc.sol_params for equality in equalities], dtype=gs.np_float)
        _sanitize_sol_params(equalities_sol_params, system._sol_min_timeconst, system._sol_default_timeconst)

        kernel_init_equality_fields(
            np.array([equality.type for equality in equalities], dtype=gs.np_int),
            np.array([equality.eq_obj1id for equality in equalities], dtype=gs.np_int),
            np.array([equality.eq_obj2id for equality in equalities], dtype=gs.np_int),
            np.array([equality.eq_data for equality in equalities], dtype=gs.np_float),
            np.array([equality.type for equality in equalities], dtype=gs.np_int),
            equalities_sol_params,
            system.dyn_info,
            system.rigid_config,
        )


def rigid_init_dynamics(system):
    """Initialize the collision geometry, the equality constraints and the dynamics, construct the collision
    detection and the constraint solve, then settle the inverse weights and the default rotor inertia."""
    rigid_init_vert_fields(system)
    rigid_init_geom_fields(system)
    rigid_init_equality_fields(system)
    rigid_init_dof_length(system)

    # Compute the pose of every link in the neutral configuration and update the pose of all the geometries,
    # including inactive variants for a given env: the step updates the active variant alone, so this is the only
    # pose the inactive ones get (see func_update_geoms_link). The collider built next relies on these updated poses
    # to determine whether some collision pairs must be filtered out.
    kernel_update_cartesian_space(
        system.dyn_state,
        system.dyn_info,
        system.rigid_info,
        system.rigid_config,
        force_update_all_geoms=True,
        is_backward=False,
    )

    # WORKAROUND: the rigid data rather than RigidRigidContactSystem owns the collision data (see ColliderData)
    system.collider_data = create_collider_data(system, system.rigid_sdf_system.sdf)
    system.constraint_solver = ConstraintSolver(system)
    system.refresh_invweight_and_meaninertia(force_update=False, in_place=True)

    # Fill in the default rotor inertia (see 'KinematicVariantDescription'), one variant per environment. It is
    # written to the armature field from the host once the parsed inverse weights stand, and a second refresh
    # recomputes the ones it changes: those of every degree of freedom and link of the kinematic trees holding a
    # defaulted joint. The parsed inverse weights stand everywhere else.
    dofs_idx, dofs_default, dofs_link = [], [], []
    for entity in system._entities:
        rotor_links = [
            link
            for link in entity.links
            if link.n_dofs == 1 and link.joints[0].type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
        ]
        if not rotor_links:
            continue
        # Only a scene or robot description file yields a rotor link, and those morphs state a default armature.
        if entity.desc.variants:
            variants_default = np.array([variant.default_armature or 0.0 for variant in entity.desc.variants])
        else:
            variants_default = np.array([entity.main_morph.default_armature or 0.0])
        envs_default = variants_default[_balanced_variant_mapping(variants_default.size, system._B)]
        if (envs_default <= 0.0).all():
            continue
        dofs_idx.extend(link.dof_start for link in rotor_links)
        dofs_default.extend([envs_default] * len(rotor_links))
        dofs_link.extend(rotor_links)
    if dofs_idx:
        # Field layout, batch last, since the arrays are written back as fields.
        dofs_armature = qd_to_numpy(system.dyn_info.dofs.armature, transpose=False, copy=True)
        is_default = (np.atleast_2d(dofs_armature[dofs_idx].T) <= 0.0).all(axis=0)
        if is_default.any():
            dofs_idx = np.array(dofs_idx)[is_default]
            default_armature = np.stack(dofs_default, axis=1)[:, is_default]
            dofs_armature[dofs_idx] = default_armature.T if system._options.batch_dofs_info else default_armature[0]
            system.dyn_info.dofs.armature.from_numpy(dofs_armature)
            roots_idx = {link.root_idx for link, is_dof_default in zip(dofs_link, is_default) if is_dof_default}
            trees_links = [link for link in system.links if link.root_idx in roots_idx]
            dofs_invweight = qd_to_numpy(system.dyn_info.dofs.invweight, transpose=False, copy=True)
            dofs_invweight[[i_d for link in trees_links for i_d in range(link.dof_start, link.dof_end)]] = -1.0
            system.dyn_info.dofs.invweight.from_numpy(dofs_invweight)
            links_idx = [link.idx for link in trees_links]
            links_invweight = qd_to_numpy(system.dyn_info.links.invweight, transpose=False, copy=True)
            links_invweight[links_idx] = -1.0
            system.dyn_info.links.invweight.from_numpy(links_invweight)
            system.refresh_invweight_and_meaninertia(force_update=False, in_place=True)

    # The constraint solver decides it has converged from quantities summed over the whole scene, and every DOF of
    # a link contributes a cost of the order of the link's mass. A link whose mass is a tolerance-fraction of the
    # scene total therefore contributes less than the tolerance, and the solve stops while that link still carries
    # residual. The floor below follows the tolerance linearly, down to the resolution of the working precision in
    # kilograms. The scene total is summed before the loop below folds each link into its parent, after which every
    # entry holds the mass of a whole subtree, which is what the floor is compared against.
    links_subtree_mass = np.atleast_2d(qd_to_numpy(system.dyn_info.links.inertial_mass, transpose=True, copy=True))
    movable_links_idx = np.flatnonzero([link.n_dofs > 0 for link in system.links])
    if movable_links_idx.size:
        total_mass = links_subtree_mass[:, movable_links_idx].sum(axis=1)
        for i_l in reversed(range(system._n_links)):
            if system.links[i_l].parent_idx >= 0:
                links_subtree_mass[:, system.links[i_l].parent_idx] += links_subtree_mass[:, i_l]
        movable_links_mass = links_subtree_mass[:, movable_links_idx]
        mass_floor = np.maximum(system._options.tolerance * total_mass, 0.2 * system._options.tolerance)[:, None]
        i_b_min, i_l_min = np.unravel_index((movable_links_mass / mass_floor).argmin(), movable_links_mass.shape)
        mass_min = movable_links_mass[i_b_min, i_l_min]
        if mass_min < mass_floor[i_b_min, 0]:
            link = system.links[movable_links_idx[i_l_min]]
            if gs.qd_float == qd.f32:
                remedy = "Use 64-bit simulation precision or tighten the solver tolerance."
            else:
                remedy = "Tighten the solver tolerance."
            gs.logger.warning(
                f"Link '{link.name}' has mass {mass_min:.1e}, too small for the constraint solver to be "
                f"numerically stable. {remedy} Note that increasing the solver iterations would not help."
            )


def resolve_broadphase_traversal(system):
    if system._options.broadphase_traversal is not None:
        return system._options.broadphase_traversal
    # The valid pairs are shared by every environment, while a heterogeneous environment carries one variant of each
    # entity. ALL_VS_ALL tests the pairs of every variant in every environment, those of the variants it does not
    # carry failing on their empty axis-aligned bounding box (see func_update_geom_aabbs), whereas sweep-and-prune
    # (SAP) only sweeps the geoms each environment carries.
    if gs.backend == gs.cpu or system._enable_heterogeneous:
        return gs.broadphase_traversal.SAP
    return gs.broadphase_traversal.ALL_VS_ALL


def jacobi_mass_spread_bound(system):
    """Upper-bound the spread of the mass matrix diagonal over every configuration, from the model alone.

    Each entry gets a lower and an upper bound that hold in every configuration. A translational entry is armature
    plus subtree mass exactly. A rotational entry is at least armature plus the link inertia about its axis (the
    smallest principal inertia for free and spherical DOFs), and at most the same with the largest principal inertia
    plus, per descendant, its largest principal inertia and its mass times the squared anchor-to-COM distance no
    configuration exceeds (frame offsets along the chain, each anchor twice, prismatic spans, infinite when
    unlimited). The variants of a heterogeneous entity each bring their own inertial, link frames, joints and
    build-time default armature, which adds its value to the entries it fills in, and any variant may be dispatched
    to any environment, so the bounds are kept per variant of each entity and the spread is the largest over every
    combination of variants across entities. That spread is at least the true one, so the gate never leaves a scene
    that needs equilibration without it, its only error being to enable it at a small cost for one that does not.
    """
    children = {}
    for entity in system._entities:
        for link in entity.links:
            children.setdefault(link.parent_idx, []).append(link)

    # Per entity, the smallest lower bound and the largest upper bound over its entries, per variant.
    entities_lower, entities_upper = [], []
    for entity in system._entities:
        rotor_links = [
            link
            for link in entity.links
            if link.n_dofs == 1 and link.joints[0].type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
        ]
        # Only a scene or robot description file yields a rotor link, and those morphs state a default armature.
        if entity.desc.variants:
            variants_default = np.array([variant.default_armature or 0.0 for variant in entity.desc.variants])
        elif rotor_links:
            variants_default = np.array([entity.main_morph.default_armature or 0.0])
        else:
            variants_default = np.zeros(1)
        variants_links = [entity.desc.links, *(variant.links for variant in entity.desc.variants[1:])]
        variants_lower, variants_upper = [], []
        for variant_default, variant_links in zip(variants_default, variants_links):
            # The subtree of a link reaches the entities attached to it, which have no variants of their own
            links_desc = {link.idx: l_desc for link, l_desc in zip(entity.links, variant_links)}
            lower, upper = [], []
            for link in entity.links:
                if link.is_fixed:
                    continue
                l_desc = links_desc[link.idx]
                is_rotor = link in rotor_links
                for joint, j_desc in zip(link.joints, l_desc.joints):
                    if joint.type == gs.JOINT_TYPE.FIXED:
                        continue
                    anchor = j_desc.pos
                    # Anchor-to-COM distance bound per subtree link (see the docstring); the carrying link's own
                    # term is exact only when this joint is its sole joint, since later chained joints re-rotate the
                    # link about their own anchors.
                    is_sole_joint = len(link.joints) == 1
                    if is_sole_joint:
                        dist_com = {link.idx: np.linalg.norm(l_desc.inertial_pos - anchor)}
                    else:
                        dist_com = {link.idx: np.linalg.norm(anchor) + np.linalg.norm(l_desc.inertial_pos)}
                    dist_origin = {link.idx: np.linalg.norm(anchor)}
                    sub, stack = [], [link]
                    while stack:
                        cur = stack.pop()
                        sub.append(cur)
                        for child in children.get(cur.idx, []):
                            child_desc = links_desc.get(child.idx, child.desc)
                            hop = np.linalg.norm(child_desc.pos) + 2.0 * sum(
                                np.linalg.norm(j.pos) for j in child_desc.joints
                            )
                            # A prismatic descendant carries the subtree outward by up to its full travel span
                            # (which covers the offset from any zero configuration within limits); an unlimited
                            # slide makes the upper bound infinite and the gate enables.
                            hop += sum(
                                np.ptp(j.dofs_limit) for j in child_desc.joints if j.type == gs.JOINT_TYPE.PRISMATIC
                            )
                            dist_origin[child.idx] = dist_origin[cur.idx] + hop
                            dist_com[child.idx] = dist_origin[child.idx] + np.linalg.norm(child_desc.inertial_pos)
                            stack.append(child)
                    sub_desc = [links_desc.get(sub_link.idx, sub_link.desc) for sub_link in sub]
                    sub_mass = sum(sub_link_desc.mass for sub_link_desc in sub_desc)
                    eigvals = np.linalg.eigvalsh(l_desc.inertia)
                    rot_desc_upper = sum(
                        np.linalg.eigvalsh(sub_link_desc.inertia)[-1] + sub_link_desc.mass * dist_com[sub_link.idx] ** 2
                        for sub_link, sub_link_desc in zip(sub, sub_desc)
                        if sub_link is not link
                    )
                    # Carrying link's inertia about the anchor: exact along a fixed axis, otherwise between its
                    # smallest and largest principal inertia.
                    if is_sole_joint:
                        R_inertial = gu.quat_to_R(l_desc.inertial_quat)
                        inertia_com = R_inertial @ l_desc.inertia @ R_inertial.T
                        offset_com = l_desc.inertial_pos - anchor
                        rot_self_lower = eigvals[0]
                        rot_self_upper = eigvals[-1] + l_desc.mass * np.dot(offset_com, offset_com)
                    else:
                        inertia_com = None
                        offset_com = None
                        rot_self_lower = eigvals[0]
                        rot_self_upper = eigvals[-1] + l_desc.mass * dist_com[link.idx] ** 2
                    for i_d, armature_d in enumerate(j_desc.dofs_armature):
                        armature = variant_default if is_rotor and armature_d <= 0.0 else armature_d
                        if joint.type == gs.JOINT_TYPE.PRISMATIC or (joint.type == gs.JOINT_TYPE.FREE and i_d < 3):
                            lower.append(armature + sub_mass)
                            upper.append(armature + sub_mass)
                        elif joint.type == gs.JOINT_TYPE.REVOLUTE and is_sole_joint:
                            axis = j_desc.dofs_motion_ang[i_d]
                            lever = np.cross(axis, offset_com)
                            rot_self = axis @ inertia_com @ axis + l_desc.mass * np.dot(lever, lever)
                            lower.append(armature + rot_self)
                            upper.append(armature + rot_self + rot_desc_upper)
                        else:
                            lower.append(armature + max(rot_self_lower, 0.0))
                            upper.append(armature + rot_self_upper + rot_desc_upper)
            variants_lower.append(lower)
            variants_upper.append(upper)
        if variants_lower[0]:
            lower, upper = np.array(variants_lower).T, np.array(variants_upper).T
            entities_lower.append(np.where(lower > 0.0, lower, np.inf).min(axis=0))
            entities_upper.append(upper.max(axis=0))
    if not entities_lower:
        return 0.0

    # The largest upper bound of a variant pairs with the smallest lower bound reachable alongside it: its own, or
    # the one of any variant of another entity. A variant without a positive lower bound has no spread to speak of.
    lower_min = np.array([entity_lower.min() for entity_lower in entities_lower])
    mass_spread = 0.0
    for i_e, (entity_lower, entity_upper) in enumerate(zip(entities_lower, entities_upper)):
        lower_paired = np.minimum(entity_lower, np.delete(lower_min, i_e).min(initial=np.inf))
        is_bounded = np.isfinite(lower_paired)
        mass_spread_entity = np.zeros_like(entity_upper)
        np.divide(entity_upper, lower_paired, out=mass_spread_entity, where=is_bounded)
        mass_spread = max(mass_spread, mass_spread_entity.max())
    return mass_spread
