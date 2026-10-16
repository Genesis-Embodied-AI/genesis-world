from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.constants import link_ref_frame
from genesis.engine.entities import DroneEntity, RigidEntity, TerrainEntity
from genesis.engine.materials import Rigid
from genesis.engine.states import RigidSolverState, SolverCheckpoint
from genesis.options.morphs import Drone, Morph, Terrain
from genesis.options.solvers import RigidOptions
from genesis.utils.misc import (
    DeprecationError,
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    qd_zero_grad,
    tensor_to_array,
)

from ..base_solver import GravityMixin, MutatedLinks, StateChange, TimeBasedMixin, mutates
from ..kinematic_solver import (
    KinematicSolver,
    _fill_base_link_geom_offsets,
    _offset_world_shift,
    _select_links_offset,
    forward_to_system,
)
from .abd.accessor import (
    ConstraintType,
    kernel_control_dofs_force,
    kernel_control_dofs_position,
    kernel_control_dofs_position_velocity,
    kernel_control_dofs_velocity,
    kernel_get_dofs_control_force,
    kernel_get_links_acc,
    kernel_get_links_vel,
    kernel_get_state,
    kernel_set_dofs_act_bias,
    kernel_set_dofs_act_gain,
    kernel_set_dofs_armature,
    kernel_set_dofs_damping,
    kernel_set_dofs_force_range,
    kernel_set_dofs_frictionloss,
    kernel_set_dofs_kp,
    kernel_set_dofs_kv,
    kernel_set_dofs_limit,
    kernel_set_dofs_position,
    kernel_set_dofs_stiffness,
    kernel_set_drone_rpm,
    kernel_set_geom_friction,
    kernel_set_geom_friction_rolling,
    kernel_set_geom_friction_torsional,
    kernel_set_geoms_friction,
    kernel_set_geoms_friction_ratio,
    kernel_set_geoms_friction_rolling,
    kernel_set_geoms_friction_torsional,
    kernel_set_global_sol_params,
    kernel_set_links_COM,
    kernel_set_links_inertia,
    kernel_set_links_mass,
    kernel_set_links_pos,
    kernel_set_links_quat,
    kernel_set_qpos,
    kernel_set_sol_params,
    kernel_set_state,
    kernel_update_drone_propeller_vgeoms,
    kernel_wake_up_entities_by_dofs,
    kernel_wake_up_entities_by_links,
    kernel_wake_up_entities_by_qs,
)
from .abd.diff import (
    func_copy_next_to_curr,
    func_integrate_dq_entity,
    kernel_begin_backward_substep,
    kernel_copy_acc,
    kernel_copy_next_to_curr_no_check,
    kernel_prepare_backward_substep,
)
from .abd.forward_dynamics import (
    func_forward_dynamics,
    func_implicit_damping,
    func_integrate,
    func_update_acc,
    func_vel_at_point,
    kernel_forward_dynamics,
    kernel_forward_dynamics_without_qacc,
    kernel_update_acc,
)
from .abd.forward_kinematics import (
    func_forward_kinematics_root,
    func_forward_velocity,
    func_update_all_verts,
    func_update_cartesian_space,
    func_update_geoms_root,
    kernel_COM_links_replay,
    kernel_forward_kinematics_links_geoms,
    kernel_forward_kinematics_replay,
    kernel_forward_velocity,
    kernel_masked_forward_kinematics_links_geoms,
    kernel_update_all_verts,
    kernel_update_cartesian_space,
    kernel_update_geoms_replay,
    kernel_update_verts_for_geoms,
)
from .abd.manual_bw import (
    kernel_manual_compute_qacc_bw,
    kernel_manual_forward_kinematics_bw,
    kernel_manual_forward_velocity_bw,
)
from .abd.misc import (
    func_apply_coupling_force,
    kernel_apply_links_external_wrench,
    kernel_apply_links_external_wrench_at_pos,
    kernel_bit_reduction,
    kernel_clear_external_force,
    kernel_reset_hibernation,
    kernel_set_zero,
)
from .collider.collider import Collider
from .constraint.backward import (
    kernel_accumulate_constraint_solver_grads,
    kernel_load_dL_dqacc_from_acc_grad,
    kernel_manual_add_collision_constraints_bw,
    kernel_manual_add_equality_constraints_bw,
    kernel_manual_add_frictionloss_constraints_bw,
    kernel_manual_add_joint_limit_constraints_bw,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)

# minimum constraint impedance
IMP_MIN = 0.0001
# maximum constraint impedance
IMP_MAX = 0.9999

# Minimum ratio between simulation timestep `_substep_dt` and time constant of constraints
TIME_CONSTANT_SAFETY_FACTOR = 2.0


def _sanitize_sol_params(
    sol_params, min_timeconst: float, default_timeconst: float | None = None, *, floor_timeconst: bool = True
):
    timeconst, dampratio, dmin, dmax, width, mid, power = sol_params.reshape((-1, 7)).T
    direct_mask = timeconst < 0.0
    if direct_mask.any():
        gs.logger.warning(
            "Constraint solver `timeconst` is negative, which parameterizes the constraint by its stiffness and its "
            "damping directly. Genesis does not support it for now, so the default time constant is used instead."
        )
        timeconst[direct_mask] = 0.0
        dampratio[direct_mask] = gu.default_solver_params()[1]
    if (timeconst < gs.EPS).any() and default_timeconst is not None:
        gs.logger.debug(
            f"Constraint solver time constant not specified. Using default value (`{default_timeconst:0.6g}`)."
        )
    invalid_mask = (timeconst > gs.EPS) & (timeconst + gs.EPS < min_timeconst)
    if invalid_mask.any():
        gs.logger.warning(
            "Constraint solver time constant should be greater than 2*substep_dt. timeconst is changed from "
            f"`{min(timeconst[invalid_mask]):0.6g}` to `{min_timeconst:0.6g}`). Decrease simulation timestep or "
            "increase timeconst to avoid altering the original value."
        )
    # An unspecified time constant takes the default, when there is one; a caller that passes none leaves it bare for
    # the floor to handle. The floor itself may be deferred, which contact assembly does so that it applies to the
    # mixed value of the two geoms rather than to each of them.
    if default_timeconst is not None:
        timeconst[timeconst < gs.EPS] = default_timeconst
    if floor_timeconst:
        timeconst[:] = timeconst.clip(min_timeconst)
    if (dampratio < gs.EPS).any():
        gs.raise_exception(
            "Constraint solver `dampratio` must be strictly positive. Despite its name, it controls spring stiffness, "
            "not damping. See `genesis.utils.geom.default_solver_params` for details."
        )
    dmin[:] = dmin.clip(IMP_MIN, IMP_MAX)
    dmax[:] = dmax.clip(IMP_MIN, IMP_MAX)
    mid[:] = mid.clip(IMP_MIN, IMP_MAX)
    width[:] = width.clip(0.0)
    power[:] = power.clip(1)
    return sol_params


class RigidSolver(GravityMixin, TimeBasedMixin, KinematicSolver):
    material_cls = Rigid
    _entity_classes = ((Drone, DroneEntity), (Terrain, TerrainEntity), (Morph, RigidEntity))
    # override typing
    _entities: list[RigidEntity] = gs.List()

    # The rigid data beyond the kinematic one, which RigidSolverSystem owns
    _errno = forward_to_system("_errno")
    n_awake_dofs = forward_to_system("n_awake_dofs")
    dyn_state_adjoint_cache = forward_to_system("dyn_state_adjoint_cache")
    mass_mat = forward_to_system("mass_mat")
    mass_mat_L = forward_to_system("mass_mat_L")
    mass_mat_D_inv = forward_to_system("mass_mat_D_inv")
    meaninertia = forward_to_system("meaninertia")
    geoms_init_AABB = forward_to_system("geoms_init_AABB")
    _dofs_mass_block_start = forward_to_system("_dofs_mass_block_start")
    _dofs_mass_block_end = forward_to_system("_dofs_mass_block_end")
    # The signed distance fields of the rigid geoms, which RigidSdfSystem owns
    sdf = forward_to_system("sdf")

    @property
    def collider(self) -> "Collider | None":
        """User-facing access to the collision detection of the rigid geoms, once the scene is built (see
        Collider)."""
        if self._system is None:
            return None
        if self._collider is None:
            self._collider = Collider(self._system)
        return self._collider

    _is_backward = forward_to_system("_is_backward", False, settable=True)

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: RigidOptions) -> None:
        super().__init__(scene, sim, options)

        self._enable_collision = options.enable_collision
        self._enable_multi_contact = options.enable_multi_contact
        self._enable_mujoco_compatibility = options.enable_mujoco_compatibility
        self._enable_joint_limit = options.enable_joint_limit
        self._enable_self_collision = options.enable_self_collision
        self._enable_neutral_collision = options.enable_neutral_collision
        self._enable_adjacent_collision = options.enable_adjacent_collision
        self._disable_constraint = options.disable_constraint
        self._max_collision_pairs = options.max_collision_pairs
        self._integrator = options.integrator
        self._box_box_detection = options.box_box_detection
        self._requires_grad = self._sim.options.requires_grad
        self._enable_heterogeneous = False  # Set to True when any entity has heterogeneous morphs

        # The differentiable solve keeps every body in its iteration trace, so hibernation stays off there (see the
        # use_hibernation option).
        self._use_hibernation = options.use_hibernation and not self._requires_grad

        # Resolve the hibernation velocity tolerance. MuJoCo compatibility uses MuJoCo's own default (1e-4); otherwise
        # use a coarser floor that a body reliably settles below across float precisions and dense contact piles, where
        # the contact solve leaves a larger residual resting-velocity jitter.
        if options.hibernation_thresh_vel is None:
            self._hibernation_thresh_vel = 1e-4 if self._enable_mujoco_compatibility else 2e-3
        else:
            self._hibernation_thresh_vel = options.hibernation_thresh_vel

        self._sol_min_timeconst = TIME_CONSTANT_SAFETY_FACTOR * self._substep_dt
        self._sol_default_timeconst = (
            None if options.constraint_timeconst is None else max(options.constraint_timeconst, self._sol_min_timeconst)
        )

        if options.friction_cone == gs.friction_cone.elliptic and self._requires_grad:
            gs.raise_exception("The elliptic friction cone is not supported yet when 'requires_grad' is True.")

        # Bounding friction against the developed normal force needs the contact to split into a normal row and a
        # friction disc, which only the elliptic cone provides. MuJoCo compatibility keeps the coupled cone regardless,
        # since letting sliding inflate the normal force is part of the behaviour being reproduced. The disc radius is
        # relatched every iteration, making the solve a successive approximation whose objective moves underneath the
        # solver; only Newton re-derives its curvature each iteration and lands on the fixed point, while conjugate
        # gradient carries a search history that the moving objective invalidates, leaving friction short.
        signorini_blocker = ""
        if self._enable_mujoco_compatibility:
            signorini_blocker = "'enable_mujoco_compatibility' is True"
        elif options.friction_cone != gs.friction_cone.elliptic:
            signorini_blocker = "it requires 'friction_cone' to be 'gs.friction_cone.elliptic'"
        elif options.constraint_solver != gs.constraint_solver.Newton:
            signorini_blocker = "it requires 'constraint_solver' to be 'gs.constraint_solver.Newton'"
        if options.contact_resolution is None:
            options.contact_resolution = (
                gs.contact_resolution.convex if signorini_blocker else gs.contact_resolution.signorini
            )
        elif options.contact_resolution == gs.contact_resolution.signorini and signorini_blocker:
            gs.raise_exception(
                f"'contact_resolution' cannot be 'gs.contact_resolution.signorini' when {signorini_blocker}."
            )
        self._contact_resolution = options.contact_resolution

        # A high tangential-to-normal impedance ratio suppresses the tangential creep of regularized friction that
        # lets resting structures slowly slide apart under their own weight. With the elliptic cone the tangential
        # rows are stiffened independently, so it resolves to a high ratio - except under MuJoCo compatibility, where
        # behavioral parity with MuJoCo (whose own default is 1) takes priority. The pyramidal cone mixes the normal
        # direction into every row, so it always keeps the neutral default of 1.
        if options.impratio is None:
            if options.friction_cone == gs.friction_cone.elliptic and not self._enable_mujoco_compatibility:
                options.impratio = 100.0
            else:
                options.impratio = 1.0

        self._ckpt = dict()

        # The user-facing access to the collision detection, created on first use once the scene is built
        self._collider: Collider | None = None

    def init_ckpt(self):
        pass

    def build(self):
        self._n_geoms = self.n_geoms
        self._n_cells = self.n_cells
        self._n_verts = self.n_verts
        self._n_free_verts = self.n_free_verts
        self._n_fixed_verts = self.n_fixed_verts
        self._n_faces = self.n_faces
        self._n_edges = self.n_edges
        self._n_equalities = self.n_equalities

        self._geoms = self.geoms
        self._equalities = self.equalities

        self.n_geoms_ = max(1, self.n_geoms)
        self.n_cells_ = max(1, self.n_cells)
        self.n_verts_ = max(1, self.n_verts)
        self.n_faces_ = max(1, self.n_faces)
        self.n_edges_ = max(1, self.n_edges)
        self.n_free_verts_ = max(1, self.n_free_verts)
        self.n_fixed_verts_ = max(1, self.n_fixed_verts)
        self.n_candidate_equalities_ = max(1, self.n_equalities + self._options.max_dynamic_constraints)

        # Resolve precision-dependent tolerance default. The convergence thresholds reference the scene's free-motion
        # cost (see func_exit_decision in linesearch.py), which stands an order of magnitude above the bare inertia
        # for a metre-scale scene under standard gravity, so the ratio drops by as much to leave the thresholds where
        # they stood. Reproducing the reference behaviour compares against the inertia and keeps its value.
        if self._options.tolerance is None:
            self._options.tolerance = 1e-5 if gs.qd_float == qd.f32 else 1e-8
            if not self._enable_mujoco_compatibility:
                self._options.tolerance *= 0.1

        super().build()

        # Morph pose offset of each collision geom, conjugated into the geom's own frame so the relative getters
        # revert it for geoms rotated relative to the link. Each root link carries its own offset; child-link geoms
        # inherit it through the kinematic chain and keep an identity offset. Forward offset device tensors, None when
        # everything is identity; the relative geom getters recompute the inverse.
        geoms_offset_pos = np.zeros((self.n_geoms, 3), dtype=gs.np_float)
        geoms_offset_quat = np.tile(gu.identity_quat(), (self.n_geoms, 1))
        for entity in self._entities:
            ranges = entity.base_link._variant_geom_ranges if entity._desc.variants else None
            _fill_base_link_geom_offsets(geoms_offset_pos, geoms_offset_quat, entity, entity.geoms, ranges)
        self._geoms_offset_pos = self._geoms_offset_quat = None
        if not (
            np.allclose(geoms_offset_pos, 0.0, atol=gs.EPS)
            and np.allclose(gu.quat_to_xyz(geoms_offset_quat), 0.0, atol=gs.EPS)
        ):
            self._geoms_offset_pos = torch.from_numpy(geoms_offset_pos).to(device=gs.device, dtype=gs.tc_float)
            self._geoms_offset_quat = torch.from_numpy(geoms_offset_quat).to(device=gs.device, dtype=gs.tc_float)

        # FIXME: when the migration is finished, we will remove the about two lines
        self._func_vel_at_point = func_vel_at_point
        self._func_apply_coupling_force = func_apply_coupling_force

    def update_forward_pos(self):
        """Run forward kinematics over links and geoms if they are not already up to date for the current pose.

        Geoms are refreshed alongside links, unlike the geom-less base solver: the flag this sets also authorizes the
        next step to skip its own Cartesian-space update, which covers geoms too. Refreshing links alone would leave
        collision - and any raycast deriving its vertices from geom poses - reading a one-step-stale pose.
        """
        if self._is_forward_pos_updated:
            return
        kernel_forward_kinematics_links_geoms(
            self.scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )
        self._is_forward_pos_updated = True

    def substep(self, f):
        """Advance the rigid solver by substep f (see RigidSolverSystem.substep)."""
        self._system.substep(f)

    def get_error_envs_mask(self):
        return qd_to_torch(self._errno) > 0

    def check_errno(self):
        if gs.use_zerocopy:
            errno = np.bitwise_or.reduce(qd_to_numpy(self._errno))
        else:
            errno = kernel_bit_reduction(self._errno)

        if errno & array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS:
            max_collision_pairs_broad = self.collider.collider_info.max_collision_pairs_broad[None]
            gs.raise_exception(
                f"Exceeding max number of broad phase candidate contact pairs ({max_collision_pairs_broad}). "
                f"Please increase the value of RigidSolver's option 'multiplier_collision_broad_phase'."
            )
        if errno & array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS:
            max_candidate_contacts = self.collider.collider_info.max_candidate_contacts[None]
            gs.raise_exception(
                f"Exceeding max number of candidate contact points ({max_candidate_contacts}). Please increase the "
                "value of RigidSolver's option 'max_collision_pairs'."
            )
        if errno & array_class.ErrorCode.OVERFLOW_CONTACTS:
            max_contacts = self.collider.collider_info.max_contacts[None]
            gs.raise_exception(
                f"Exceeding max number of post-pruning contact points ({max_contacts}) supported by the constraint "
                "solver. Please increase the value of RigidSolver's option 'max_contacts'."
            )
        if errno & array_class.ErrorCode.INVALID_CONTACT_NAN:
            gs.raise_exception(
                "Collision detection reported a contact whose position, normal or penetration is not finite. This is a "
                "solver-internal error, please report it."
            )
        if errno & array_class.ErrorCode.INVALID_FORCE_NAN:
            gs.raise_exception("Invalid constraint forces causing 'nan'. Please decrease Rigid simulation timestep.")
        if errno & array_class.ErrorCode.INVALID_ACC_NAN:
            gs.raise_exception("Invalid accelerations causing 'nan'. Please decrease Rigid simulation timestep.")

    def _kernel_detect_collision(self):
        self.collider.clear()
        self.collider.detection()

    def detect_collision(self, env_idx=0):
        # TODO: support batching
        self._kernel_detect_collision()

        n_collision = qd_to_numpy(self.collider.collider_state.n_contacts)[env_idx]
        collision_pairs = np.empty((n_collision, 2), dtype=np.int32)
        collision_pairs[:, 0] = qd_to_numpy(self.collider.collider_state.contact_data.geom_a)[:n_collision, env_idx]
        collision_pairs[:, 1] = qd_to_numpy(self.collider.collider_state.contact_data.geom_b)[:n_collision, env_idx]

        return collision_pairs

    def _func_constraint_force(self):
        """Assemble and solve the constraints of the current substep (see RigidSolverSystem.constraint_force)."""
        self._system.constraint_force()

    def _constraint_force_grad(self):
        """Backward pass for the constraint solver: seed dL_dqacc from acc.grad, run the adjoint solve, fold its
        outputs back into the autodiff grad fields, then reverse the constraint-row assembly."""
        constraint_state = self.constraint_solver.constraint_state
        # Pure grad shuffles: in-place through zero-copy views when supported, kernel dispatch otherwise (see
        # "Pure read-write data accessors on the hot path" in CLAUDE.md). gs.use_zerocopy encodes every zero-copy
        # availability condition for these buffers (standalone dense allocations with DLPack-supported dtypes; the
        # platform gates live in gs.init). torch (MPS) and quadrants do not share a compute stream on Metal, so the
        # writes are flushed before the next kernels (see set_base_links_quat).
        if gs.use_zerocopy:
            acc_grad = qd_to_torch(self.dyn_state.dofs.acc.grad, copy=False)
            dL_dqacc = qd_to_torch(constraint_state.dL_dqacc, copy=False)
            dL_dqacc.copy_(acc_grad)
            acc_grad.zero_()
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_load_dL_dqacc_from_acc_grad(self.dyn_state, constraint_state, self.rigid_config)
        self.constraint_solver.backward()
        if gs.use_zerocopy:
            force_grad = qd_to_torch(self.dyn_state.dofs.force.grad, copy=False)
            dL_dforce = qd_to_torch(constraint_state.dL_dforce, copy=False)
            force_grad.add_(dL_dforce)
            mass_mat_grad = qd_to_torch(self.rigid_info.mass_mat.grad, copy=False)
            dL_dM = qd_to_torch(constraint_state.dL_dM, copy=False)
            mass_mat_grad.add_(dL_dM)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_accumulate_constraint_solver_grads(
                self.dyn_state, constraint_state, self.rigid_info, self.rigid_config
            )

        kernel_manual_add_equality_constraints_bw(
            self.dyn_state, constraint_state, self.dyn_info, self.rigid_info, self.rigid_config
        )
        kernel_manual_add_frictionloss_constraints_bw(
            self.dyn_state, constraint_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

        if self._enable_collision:
            collider_state = self.collider.collider_state
            qd_zero_grad(collider_state.contact_data.pos)
            qd_zero_grad(collider_state.contact_data.normal)
            qd_zero_grad(collider_state.contact_data.penetration)
            # One flush for the zeroing batch; see qd_zero_grad in misc.py.
            if gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()
            kernel_manual_add_collision_constraints_bw(
                self.dyn_state,
                collider_state,
                constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )
            self.collider.backward_narrowphase()

        if self._enable_joint_limit:
            kernel_manual_add_joint_limit_constraints_bw(
                self.dyn_state,
                self.collider.collider_state,
                constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                enable_collision=self._enable_collision,
            )

    def _func_forward_dynamics(self):
        kernel_forward_dynamics(
            self.dyn_state,
            self.constraint_solver.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )

    def _func_update_acc(self):
        kernel_update_acc(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)

    def _func_integrate_dq_entity(self, dq, i_e, i_b, respect_joint_limit):
        func_integrate_dq_entity(i_e, i_b, dq, self.dyn_info, self.rigid_info, self.rigid_config, respect_joint_limit)

    def apply_links_external_wrench(
        self,
        force=None,
        torque=None,
        links_idx=None,
        envs_idx=None,
        *,
        pos=None,
        ref: link_ref_frame = link_ref_frame.link_origin,
        local: bool = False,
    ):
        """
        Apply an external wrench over one simulation step on a set of links.

        Parameters
        ----------
        force : None | array_like, optional
            The linear force to apply. None for a pure torque. Defaults to None.
        torque : None | array_like, optional
            The torque to apply, on top of the moment induced by the linear force. Defaults to None.
        links_idx : None | array_like, optional
            The indices of the links on which to apply the wrench. None to specify all links. Default to None.
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
        if force is None and torque is None:
            gs.raise_exception("Either 'force' or 'torque' must be specified.")
        if force is None and pos is not None:
            gs.raise_exception("'pos' requires 'force', as a torque acts the same wherever it is applied.")
        ref_frame = self._sanitize_ref_frame(ref, has_root_COM=False)

        if pos is not None:
            pos, _, _ = self._sanitize_io_variables(
                pos, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
        # An absent wrench component contributes nothing, which the kernel spells out as zeros.
        force = 0.0 if force is None else force
        torque = 0.0 if torque is None else torque
        force, _, _ = self._sanitize_io_variables(
            force, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        torque, links_idx, envs_idx = self._sanitize_io_variables(
            torque, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            force, torque = force[None], torque[None]
            if pos is not None:
                pos = pos[None]

        # A wrench on a sleeping body must revive it, otherwise the input is silently dropped.
        if self._use_hibernation:
            kernel_wake_up_entities_by_links(
                links_idx,
                envs_idx,
                self.dyn_state,
                self.constraint_solver.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

        if pos is None:
            kernel_apply_links_external_wrench(
                links_idx, envs_idx, force, torque, self.dyn_state, self.rigid_config, ref_frame, local
            )
        else:
            kernel_apply_links_external_wrench_at_pos(
                links_idx, envs_idx, pos, force, torque, self.dyn_state, self.rigid_config, ref_frame, local
            )

    def substep_pre_coupling(self, f):
        if self.is_active:
            # Skip rigid body computation when using IPCCoupler (IPC handles rigid simulation)
            from genesis.engine.couplers import IPCCoupler

            if isinstance(self.sim.coupler, IPCCoupler):
                # If any rigid entity is coupled to IPC, skip pre-coupling rigid simulation
                # The rigid simulation will be done in post-coupling phase instead
                if self.sim.coupler.has_any_rigid_coupling:
                    return

            # Run Genesis rigid simulation step for non-IPC couplers
            self._system.substep(f)

    def reset_grad(self):
        # Rigid additionally owns `geoms_state`, `entities_state`, and the `*_adjoint_cache` structs written by the
        # backward substep chain. All carry `needs_grad=True` fields that accumulate via `atomic_add` during backward,
        # so they must start at zero between consecutive `loss.backward()`s.
        super().reset_grad()
        if self._requires_grad:
            qd_zero_grad(self.dyn_state.geoms)
            qd_zero_grad(self.dyn_state.entities)
            qd_zero_grad(self.dyn_state_adjoint_cache.dofs)
            qd_zero_grad(self.dyn_state_adjoint_cache.links)
            qd_zero_grad(self.dyn_state_adjoint_cache.joints)
            qd_zero_grad(self.dyn_state_adjoint_cache.geoms)
            qd_zero_grad(self._rigid_adjoint_cache)
            # One flush for the zeroing batch; see qd_zero_grad in misc.py.
            if gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()

    def _update_cartesian_grad(self, envs_idx):
        """Forward-replay the post-integrate cartesian-space update (FK -> COM -> geom poses -> velocity) under
        is_backward=True, then reverse it stage by stage.

        Velocity and forward kinematics are reversed manually (kernel_manual_*_bw in manual_bw.py), while COM and
        the link->geom transform are reversed by Quadrants autodiff (.grad). Shared by the post-integrate reverse
        and the first-substep initial-state reverse in substep_pre_coupling_grad.
        """
        # Forward replay in dependency order (FK -> COM -> geoms -> velocity).
        kernel_forward_kinematics_replay(
            envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=True
        )
        kernel_COM_links_replay(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        kernel_update_geoms_replay(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=True)
        kernel_forward_velocity(
            envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=True
        )

        # Reverse the stages: velocity first, forward kinematics last. COM and geoms both consume only FK
        # outputs, so their mutual order is free.
        kernel_manual_forward_velocity_bw(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        kernel_COM_links_replay.grad(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        kernel_update_geoms_replay.grad(
            self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=True
        )
        kernel_manual_forward_kinematics_bw(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)

    def substep_pre_coupling_grad(self, f):
        # Change to backward mode
        self._is_backward = True

        # Run forward substep again to restore this step's information, this is needed because we do not store info
        # of every substep.
        kernel_prepare_backward_substep(
            f,
            self.dyn_state,
            self.dyn_state_adjoint_cache,
            self._rigid_adjoint_cache,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )
        self.substep(f)
        # =================== Backward substep ======================
        envs_idx = self._scene._sanitize_envs_idx(None)
        if not self._enable_mujoco_compatibility:
            # The FK backward below builds its Jacobian at the post-integrate qpos / vel, so copy the integrator's
            # _next outputs into the current slots first.
            kernel_copy_next_to_curr_no_check(self.dyn_state, self.rigid_info, self.rigid_config)
            self._update_cartesian_grad(envs_idx)

        is_grad_valid = kernel_begin_backward_substep(
            f,
            self.dyn_state,
            self.dyn_state_adjoint_cache,
            self._rigid_adjoint_cache,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )
        if not is_grad_valid:
            gs.raise_exception(f"Nan grad in qpos or dofs_vel found at step {self._sim.cur_step_global}")

        kernel_step_2.grad(
            self.dyn_state,
            self.constraint_solver.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            is_backward=True,
            errno=self._errno,
        )

        # Mirror the forward branch in _func_constraint_force:
        #   (A) _disable_constraint=True: the forward never calls the constraint solver; acc is the smooth-dynamics
        #       result. Reverse via kernel_manual_compute_qacc_bw (implicit function theorem through M).
        #   (B) _disable_constraint=False: the forward always calls constraint_solver.resolve. Reverse via
        #       _constraint_force_grad.
        if not self._disable_constraint:
            self._constraint_force_grad()
        else:
            kernel_manual_compute_qacc_bw(self.dyn_state, self.rigid_info, self.rigid_config)
        kernel_copy_acc(f, self.dyn_state, self._rigid_adjoint_cache, self.rigid_config)

        kernel_forward_dynamics_without_qacc.grad(
            self.dyn_state,
            self.constraint_solver.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            is_backward=True,
        )

        # If it was the very first substep, we need to backpropagate through the initial update of the cartesian space
        if self._enable_mujoco_compatibility or self._sim.cur_substep_global == 0:
            self._update_cartesian_grad(envs_idx)

        # Change back to forward mode
        self._is_backward = False

    def substep_post_coupling(self, f):
        from genesis.engine.couplers import SAPCoupler, IPCCoupler

        if not self.is_active:
            return

        if isinstance(self.sim.coupler, SAPCoupler):
            self._system.substep_post_coupling(f)
        elif isinstance(self.sim.coupler, IPCCoupler):
            # If any rigid entity is coupled to IPC, perform rigid simulation in post-coupling phase.
            # Collision exclusion for IPC-coupled links is handled in the collider at build time.
            if self.sim.coupler.has_any_rigid_coupling:
                self._system.substep(f)

    # ------------------------------------------------------------------------------------
    # -------------------------------- state get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f=None):
        s_global = self.sim.cur_step_global
        if self.is_active:
            if s_global in self._queried_states:
                return self._queried_states[s_global][0]

            state = RigidSolverState(self._scene, s_global)

            # A captured state must be self-consistent: links_pos / links_quat have to be the Cartesian pose implied by
            # the qpos captured alongside them, otherwise restoring it does not reproduce the configuration it was taken
            # from. The step loop leaves that pose one integration behind qpos whenever it defers the post-integrate
            # refresh, so catch up here - a no-op when the pose is already current, which is the common case.
            self.update_forward_pos()

            kernel_get_state(
                state.qpos,
                state.dofs_vel,
                state.dofs_acc,
                state.links_pos,
                state.links_quat,
                state.friction_ratio,
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

        if partial:
            self._system.reset(envs_idx)
            self.constraint_solver.reset(envs_idx)
        else:
            self._system.clear(envs_idx)
            self.constraint_solver.clear(envs_idx)

        if (
            not self._requires_grad
            and gs.use_zerocopy
            and (not isinstance(envs_idx, torch.Tensor) or (not IS_OLD_TORCH or envs_idx.dtype == torch.bool))
        ):
            errno = qd_to_torch(self._errno, copy=False)
            qpos_dst = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
            vel_dst = qd_to_torch(self.dyn_state.dofs.vel, transpose=True, copy=False)
            acc_dst = qd_to_torch(self.dyn_state.dofs.acc, transpose=True, copy=False)
            ctrl_force_dst = qd_to_torch(self.dyn_state.dofs.ctrl_force, transpose=True, copy=False)
            ctrl_mode_dst = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            pos_dst = qd_to_torch(self.dyn_state.links.pos, transpose=True, copy=False)
            quat_dst = qd_to_torch(self.dyn_state.links.quat, transpose=True, copy=False)
            cfrc_vel_dst = qd_to_torch(self.dyn_state.links.cfrc_applied_vel, transpose=True, copy=False)
            cfrc_ang_dst = qd_to_torch(self.dyn_state.links.cfrc_applied_ang, transpose=True, copy=False)
            fric_dst = qd_to_torch(self.dyn_state.geoms.friction_ratio, transpose=True, copy=False)
            # Setting the state is a discontinuity: wake every body in the affected envs (a body left hibernated would
            # stay frozen), restoring the flags and the awake-dof count alongside the other state buffers.
            if self._use_hibernation:
                links_hibernated_dst = qd_to_torch(self.dyn_state.links.is_hibernated, transpose=True, copy=False)
                awake_steps_dst = qd_to_torch(self.dyn_state.links.awake_steps, transpose=True, copy=False)
                dofs_hibernated_dst = qd_to_torch(self.dyn_state.dofs.is_hibernated, transpose=True, copy=False)
                geoms_hibernated_dst = qd_to_torch(self.dyn_state.geoms.is_hibernated, transpose=True, copy=False)
                entities_hibernated_dst = qd_to_torch(self.dyn_state.entities.is_hibernated, transpose=True, copy=False)
                islands_hibernated_dst = qd_to_torch(
                    self.constraint_solver.constraint_state.island.is_hibernated, transpose=True, copy=False
                )
                islands_next_link_dst = qd_to_torch(
                    self.constraint_solver.constraint_state.island.hibernated_next_link, transpose=True, copy=False
                )
                n_awake_dofs_dst = qd_to_torch(self.rigid_info.n_awake_dofs, copy=False)

            if envs_idx is not None and not isinstance(envs_idx, torch.Tensor):
                (envs_idx,) = indices_to_mask(envs_idx)
            if isinstance(envs_idx, torch.Tensor):
                if envs_idx.dtype == torch.bool:
                    envs_mask = envs_idx
                else:
                    envs_mask = torch.zeros(self._B, dtype=torch.bool, device=gs.device)
                    envs_mask[envs_idx] = True

                errno.masked_fill_(envs_mask, 0)
                if self.n_qs:
                    torch.where(envs_mask[:, None], state.qpos, qpos_dst, out=qpos_dst)
                    torch.where(envs_mask[:, None], state.dofs_vel, vel_dst, out=vel_dst)
                    torch.where(envs_mask[:, None], state.dofs_acc, acc_dst, out=acc_dst)
                    ctrl_force_dst.masked_fill_(envs_mask[:, None], 0.0)
                    ctrl_mode_dst.masked_fill_(envs_mask[:, None], gs.CTRL_MODE.FORCE)
                torch.where(envs_mask[:, None, None], state.links_pos, pos_dst, out=pos_dst)
                torch.where(envs_mask[:, None, None], state.links_quat, quat_dst, out=quat_dst)
                cfrc_vel_dst.masked_fill_(envs_mask[:, None, None], 0.0)
                cfrc_ang_dst.masked_fill_(envs_mask[:, None, None], 0.0)
                if self.n_geoms:
                    torch.where(envs_mask[:, None], state.friction_ratio, fric_dst, out=fric_dst)
                if self._use_hibernation:
                    links_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    awake_steps_dst.masked_fill_(envs_mask[:, None], 0)
                    dofs_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    geoms_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    entities_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    islands_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    islands_next_link_dst.masked_fill_(envs_mask[:, None], -1)
                    n_awake_dofs_dst.masked_fill_(envs_mask, self.n_dofs)
            else:
                if self.n_qs:
                    errno[envs_idx] = 0
                    qpos_dst[envs_idx] = state.qpos[envs_idx]
                    vel_dst[envs_idx] = state.dofs_vel[envs_idx]
                    acc_dst[envs_idx] = state.dofs_acc[envs_idx]
                    ctrl_force_dst[envs_idx] = 0.0
                    ctrl_mode_dst[envs_idx] = gs.CTRL_MODE.FORCE
                pos_dst[envs_idx] = state.links_pos[envs_idx]
                quat_dst[envs_idx] = state.links_quat[envs_idx]
                cfrc_vel_dst[envs_idx] = 0.0
                cfrc_ang_dst[envs_idx] = 0.0
                if self.n_geoms:
                    fric_dst[envs_idx] = state.friction_ratio[envs_idx]
                if self._use_hibernation:
                    links_hibernated_dst[envs_idx] = 0
                    awake_steps_dst[envs_idx] = 0
                    dofs_hibernated_dst[envs_idx] = 0
                    geoms_hibernated_dst[envs_idx] = 0
                    entities_hibernated_dst[envs_idx] = 0
                    islands_hibernated_dst[envs_idx] = 0
                    islands_next_link_dst[envs_idx] = -1
                    n_awake_dofs_dst[envs_idx] = self.n_dofs
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            kernel_set_zero(envs_idx, self._errno)
            kernel_set_state(
                envs_idx,
                state.qpos,
                state.dofs_vel,
                state.dofs_acc,
                state.links_pos,
                state.links_quat,
                state.friction_ratio,
                self.dyn_state,
                self.rigid_info,
                self.rigid_config,
            )
            if self._use_hibernation:
                kernel_reset_hibernation(
                    envs_idx,
                    self.dyn_state,
                    self.constraint_solver.constraint_state,
                    self.dyn_info,
                    self.rigid_info,
                    self.rigid_config,
                )

        if not partial:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

        self._restart()

    @property
    def data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        yield from super().data
        yield from self.collider.data
        yield from self.constraint_solver.data

    def _restart(self):
        """Clear the contact and equality caches of the last query and re-arm the once-per-step propeller guard of
        each drone (see 'set_propellers_rpm').
        """
        self.collider._contact_data_cache.clear()
        self.constraint_solver._eq_const_info_cache.clear()
        for entity in self.entities:
            if isinstance(entity, DroneEntity):
                entity._prev_prop_t = -1

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def __setstate__(self, state: SolverCheckpoint) -> None:
        super().__setstate__(state)
        self._restart()

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities:
            entity.process_input_grad()

    def save_ckpt(self, ckpt_name):
        # Save ckpt only if we need gradients, because this operation is costly
        if self._requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()

            # copy=True required: with the zerocopy backend qd_to_numpy returns a view, so later substeps would
            # overwrite this ckpt's buffer in place.
            self._ckpt[ckpt_name]["qpos"] = qd_to_numpy(self._rigid_adjoint_cache.qpos, copy=True)
            self._ckpt[ckpt_name]["dofs_vel"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_vel, copy=True)
            self._ckpt[ckpt_name]["dofs_acc"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_acc, copy=True)

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

    def load_ckpt(self, ckpt_name):
        # Set first frame
        self.rigid_info.qpos.from_numpy(self._ckpt[ckpt_name]["qpos"][0])
        self.dyn_state.dofs.vel.from_numpy(self._ckpt[ckpt_name]["dofs_vel"][0])
        self.dyn_state.dofs.acc.from_numpy(self._ckpt[ckpt_name]["dofs_acc"][0])

        if not self._enable_mujoco_compatibility:
            # Mirror the post-integrate refresh of kernel_step_2: the replayed substeps skip their own cartesian /
            # velocity updates (is_forward_pos_updated / is_forward_vel_updated), so both must be recomputed here
            # from the restored qpos / vel. A stale link velocity would corrupt the velocity-product terms of every
            # backward primal in the window.
            kernel_update_cartesian_space(
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                force_update_all_geoms=False,
                is_backward=False,
            )
            kernel_forward_velocity(
                self._scene._sanitize_envs_idx(None),
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                is_backward=False,
            )

        for entity in self._entities:
            entity.load_ckpt(ckpt_name)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def set_links_pos(self, pos, links_idx=None, envs_idx=None):
        raise DeprecationError("This method has been removed. Please use 'set_base_links_pos' instead.")

    @mutates(StateChange.GEOMETRY, links="links_idx")
    def set_base_links_pos(self, pos, links_idx=None, envs_idx=None, *, relative=False, skip_forward=False):
        if links_idx is None:
            links_idx = self._base_links_idx

        # Without any offset position on the targeted links, the authored and internal link origins coincide, so a
        # relative set is just an absolute one.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and self._links_offset_pos_is_identity[idx].all():
            relative = False

        # Map a single base link's authored position to world here (keeping the current orientation) so the zero-copy
        # in-place write below still applies, both for an environment-uniform and a per-environment offset. Multi-link
        # relative sets are composed in the kernel branch instead.
        if relative and isinstance(links_idx, int):
            cur_quat = self.get_links_quat(links_idx, envs_idx, relative=False)[..., 0, :]
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)[..., 0, :]
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)[..., 0, :]
            pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device) + _offset_world_shift(
                offset_pos, offset_quat, cur_quat
            )
            relative = False

        # Zero-copy fast path: single base link, non-relative. Write the position buffer in place instead of
        # launching a kernel. The kernel path below handles relative or multi-link updates, as well as waking up
        # hibernated entities, which is required whenever hibernation is enabled.
        if gs.use_zerocopy and not relative and isinstance(links_idx, int) and not self._use_hibernation:
            link = self.links[links_idx]
            if link.is_fixed:
                data = qd_to_torch(self.dyn_state.links.pos, transpose=True, copy=False)
                target = data[:, links_idx]
            else:
                data = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
                target = data[:, link.q_start : link.q_start + 3]
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                if pos.ndim == 2 and len(pos) not in (1, len(target)):
                    # A fresh source view is needed because masked_scatter_ may reshape it in-place. Metal
                    # mis-scatters a stride-0 broadcast mask, so it must be materialized to a dense mask there.
                    envs_mask = envs_idx[:, None]
                    if gs.backend == gs.metal:
                        envs_mask = envs_mask.expand_as(target).contiguous()
                    target.masked_scatter_(envs_mask, pos.view_as(pos))
                else:
                    pos = broadcast_tensor(pos, gs.tc_float, target.shape)
                    torch.where(envs_idx[:, None], pos, target, out=target)
            else:
                # Fixed links with at least one geom and non-batched vertices cannot take env-specific positions
                if link.is_fixed and link.geoms and not link.entity._batch_fixed_verts:
                    pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
                    same_pos = pos.ndim < 2 or len(pos) == 1 or (torch.diff(pos, dim=0).abs() < gs.EPS).all()
                    set_all_envs = envs_idx is None or torch.equal(
                        torch.sort(self._scene._sanitize_envs_idx(envs_idx)).values, self._scene._envs_idx
                    )
                    if not (set_all_envs and same_pos):
                        gs.raise_exception(
                            "Specifying env-specific pos for fixed links with at least one geometry requires "
                            "setting morph option 'batch_fixed_verts=True'."
                        )
                mask = (0,) if self.n_envs == 0 else indices_to_mask(envs_idx)
                assign_indexed_tensor(target, mask, pos)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            pos, links_idx, envs_idx = self._sanitize_io_variables(
                pos, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
            if self.n_envs == 0:
                pos = pos[None]

            if relative:
                # Compose the body-frame offset onto the authored position, keeping the current orientation, then set
                # the resulting world position absolutely.
                cur_quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
                offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                pos = pos + _offset_world_shift(offset_pos, offset_quat, cur_quat)
                relative = False

            # Raise exception for fixed links with at least one geom and non-batched fixed vertices, except if setting
            # same location for all envs at once
            set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
            has_fixed_verts = any(
                link.is_fixed and link.geoms and not link.entity._batch_fixed_verts
                for link in (self.links[i_l] for i_l in links_idx)
            )
            if has_fixed_verts and not (set_all_envs and (torch.diff(pos, dim=0).abs() < gs.EPS).all()):
                gs.raise_exception(
                    "Specifying env-specific pos for fixed links with at least one geometry requires setting morph "
                    "option 'batch_fixed_verts=True'."
                )

            # Wake up hibernated entities before setting position (fixed links don't need wake-up)
            if self._use_hibernation and not all(self.links[i_l].is_fixed for i_l in links_idx):
                kernel_wake_up_entities_by_links(
                    links_idx,
                    envs_idx,
                    self.dyn_state,
                    self.constraint_solver.constraint_state,
                    self.dyn_info,
                    self.rigid_info,
                    self.rigid_config,
                )

            kernel_set_links_pos(
                links_idx, envs_idx, pos, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
            )

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def set_links_quat(self, quat, links_idx=None, envs_idx=None):
        raise DeprecationError("This method has been removed. Please use 'set_base_links_quat' instead.")

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

        # Reused by the int fast path and the kernel branch below.
        relative_pos_passthrough = relative and self._links_offset_pos_is_identity[idx].all()

        # Compose a single base link's authored orientation to world here so the zero-copy in-place write below still
        # applies, both for an environment-uniform and a per-environment offset. This only preserves the authored-frame
        # position when the offset position is identity; a non-zero offset position rotates with the orientation and
        # is handled (together with multi-link relative sets) in the kernel branch instead.
        if isinstance(links_idx, int) and relative_pos_passthrough:
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)[..., 0, :]
            quat = gu.transform_quat_by_quat(offset_quat, torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device))
            relative = False

        # Zero-copy fast path: single base link, non-relative. Write the quaternion buffer in place instead of
        # launching a kernel. The kernel path below handles relative or multi-link updates, as well as waking up
        # hibernated entities, which is required whenever hibernation is enabled.
        if gs.use_zerocopy and not relative and isinstance(links_idx, int) and not self._use_hibernation:
            link = self.links[links_idx]
            if link.is_fixed:
                data = qd_to_torch(self.dyn_state.links.quat, transpose=True, copy=False)
                target = data[:, links_idx]
            else:
                data = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
                target = data[:, link.q_start + 3 : link.q_start + 7]
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                if quat.ndim == 2 and len(quat) not in (1, len(target)):
                    # A fresh source view is needed because masked_scatter_ may reshape it in-place. Metal
                    # mis-scatters a stride-0 broadcast mask, so it must be materialized to a dense mask there.
                    envs_mask = envs_idx[:, None]
                    if gs.backend == gs.metal:
                        envs_mask = envs_mask.expand_as(target).contiguous()
                    target.masked_scatter_(envs_mask, quat.view_as(quat))
                else:
                    quat = broadcast_tensor(quat, gs.tc_float, target.shape)
                    torch.where(envs_idx[:, None], quat, target, out=target)
            else:
                # Fixed links with at least one geom and non-batched vertices cannot take env-specific orientations
                if link.is_fixed and link.geoms and not link.entity._batch_fixed_verts:
                    quat = torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device)
                    same_quat = quat.ndim < 2 or len(quat) == 1 or (torch.diff(quat, dim=0).abs() < gs.EPS).all()
                    set_all_envs = envs_idx is None or torch.equal(
                        torch.sort(self._scene._sanitize_envs_idx(envs_idx)).values, self._scene._envs_idx
                    )
                    if not (set_all_envs and same_quat):
                        gs.raise_exception(
                            "Impossible to set env-specific quat for fixed links with at least one geometry."
                        )
                mask = (0,) if self.n_envs == 0 else indices_to_mask(envs_idx)
                assign_indexed_tensor(target, mask, quat)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
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
                        links_idx,
                        envs_idx,
                        world_pos,
                        self.dyn_state,
                        self.dyn_info,
                        self.rigid_info,
                        self.rigid_config,
                    )
                # Compose the offset onto the authored orientation, then set the resulting world orientation absolutely.
                quat = gu.transform_quat_by_quat(offset_quat, quat)
                relative = False

            set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
            has_fixed_verts = any(
                link.is_fixed and link.geoms and not link.entity._batch_fixed_verts
                for link in (self.links[i_l] for i_l in links_idx)
            )
            if has_fixed_verts and not (set_all_envs and (torch.diff(quat, dim=0).abs() < gs.EPS).all()):
                gs.raise_exception("Impossible to set env-specific quat for fixed links with at least one geometry.")

            # Wake up hibernated entities before setting quaternion (fixed links don't need wake-up)
            if self._use_hibernation and not all(self.links[i_l].is_fixed for i_l in links_idx):
                kernel_wake_up_entities_by_links(
                    links_idx,
                    envs_idx,
                    self.dyn_state,
                    self.constraint_solver.constraint_state,
                    self.dyn_info,
                    self.rigid_info,
                    self.rigid_config,
                )

            kernel_set_links_quat(
                links_idx, envs_idx, quat, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
            )

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    @mutates(StateChange.MODEL)
    def _set_links_info(self, values, links_idx, name, envs_idx=None, *, scale_inertia=False):
        """Write one inertial property of the given links, then recompute the inverse weights of their trees once.

        The property is the mass, the center of mass or the inertia, selected by `name`. The inverse weights are per
        environment even when the property itself is shared by the whole batch, so writing shared link info recomputes
        them for every environment.
        """
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        shape = {"mass": (), "COM": (3,), "inertia": (3, 3)}[name]
        # A value of fewer dimensions than the property holds would be spread over it, which for an inertia means a
        # tensor of one number repeated: no inertia of any body.
        if shape and np.shape(values)[-len(shape) :] != shape:
            gs.raise_exception(f"{name} is {shape} per link, and {np.shape(values)} cannot be read as that.")
        # The anchor check below reads the written indices as given, in any form the sanitizer accepts. A tensor on a
        # GPU device is read back in debug mode only, since that stalls the GPU.
        links_idx_ = None
        if links_idx is None:
            links_idx_ = range(self.n_links)
        elif not isinstance(links_idx, torch.Tensor) or gs.debug or links_idx.device.type == "cpu":
            (links_idx_,) = indices_to_mask(links_idx, keepdim=False, to_torch=False, boolean_mask=False)
            if isinstance(links_idx_, slice):
                links_idx_ = range(*links_idx_.indices(self.n_links))
            elif isinstance(links_idx_, torch.Tensor):
                links_idx_ = tensor_to_array(links_idx_)
            if np.ndim(links_idx_) == 0:
                links_idx_ = (links_idx_,)
        values, links_idx, envs_idx = self._sanitize_io_variables(
            values,
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
            element_shape=shape,
            batched=self._options.batch_links_info,
            skip_allocation=True,
        )
        if links_idx_ is not None:
            # TODO: the anchor of an aligned root keeps its mass block diagonal (see rigid_init_tree_fields). A center
            # of mass or an inertia written on any link of the body moves the anchor. A mass write keeps it only as one
            # rescale of every link of the body, inertia included. A frame move shifts the local pose of every geom
            # ('GeomsInfo.pos' / 'quat', one entry per geom for all environments). It also shifts 'qpos', quoted in the
            # anchored frame, and the origin a fixed child reports. A per-environment value has no frame to go to, so
            # the guard stays until 'GeomsInfo' gets a batch dimension.
            links_anchor = []
            for link in self.links:
                anchor = link
                while anchor.parent_idx != -1 and all(joint.type == gs.JOINT_TYPE.FIXED for joint in anchor.joints):
                    anchor = self.links[anchor.parent_idx]
                links_anchor.append(anchor if anchor.aligned else None)
            values_idx_by_link = {i_l: i_col for i_col, i_l in enumerate(links_idx_)}
            for i_l in values_idx_by_link:
                anchor = links_anchor[i_l]
                if anchor is None:
                    continue
                links_idx_body = [link.idx for link in self.links if links_anchor[link.idx] is anchor]
                if name == "mass" and len(links_idx_body) == 1:
                    continue
                if name == "mass" and scale_inertia and all(i_b in values_idx_by_link for i_b in links_idx_body):
                    # The rescale check reads the values back and stalls the GPU, so it runs in debug mode only.
                    if not gs.debug:
                        continue
                    ratios = values[..., [values_idx_by_link[i_b] for i_b in links_idx_body]]
                    ratios = ratios / self.get_links_mass(links_idx_body, envs_idx)
                    if torch.allclose(ratios, ratios[..., :1], rtol=gs.EPS, atol=gs.EPS):
                        continue
                link = self.links[i_l]
                remedy = (
                    "Load the entity with 'align=False' to write inertial properties at runtime."
                    if isinstance(link.entity.main_morph, gs.options.morphs.FileMorph)
                    else "Writing the inertial properties of a primitive morph is not supported yet."
                )
                what = {"mass": "mass", "COM": "center of mass", "inertia": "inertia"}[name]
                gs.raise_exception(
                    f"Cannot set the {what} of link '{link.name}': the frame of its body is anchored on the center of "
                    f"mass and principal axes of all its links. Set the mass of the whole entity instead. {remedy}"
                )

        envs_idx = envs_idx if self._options.batch_links_info else self._scene._envs_idx
        if not self._options.batch_links_info or self.n_envs == 0:
            values = values[None]
        # The weights are solved again rather than scaled by the mass ratio: a weight follows the mass alone for a body
        # whose degrees of freedom carry no armature, which a model loaded from a file has.
        if name == "mass":
            kernel_set_links_mass(
                links_idx,
                envs_idx,
                values,
                self.data_manager.weight_scratch.jac_row,
                self.data_manager.weight_scratch.solve_out,
                self.dyn_state,
                self.constraint_solver.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                scale_inertia,
                self._is_forward_pos_updated,
                self._is_forward_vel_updated,
            )
        elif name == "COM":
            kernel_set_links_COM(
                links_idx,
                envs_idx,
                values,
                self.data_manager.weight_scratch.jac_row,
                self.data_manager.weight_scratch.solve_out,
                self.dyn_state,
                self.constraint_solver.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                self._is_forward_pos_updated,
                self._is_forward_vel_updated,
            )
        else:
            kernel_set_links_inertia(
                links_idx,
                envs_idx,
                values,
                self.data_manager.weight_scratch.jac_row,
                self.data_manager.weight_scratch.solve_out,
                self.dyn_state,
                self.constraint_solver.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                refresh_position=self._is_forward_pos_updated,
            )

    def set_links_mass(self, mass, links_idx=None, envs_idx=None, scale_inertia=False):
        """Set the mass of the given links, in kg.

        Set 'scale_inertia' to scale their inertia by the same factor, which gives what a body of the same shape made
        heavier behaves like, its inertia being proportional to its mass. Without it, the mass of a link changes and
        the inertia it was given stays as it is.

        Every mass must be finite and strictly positive, which is not checked.

        The value written survives `scene.reset()`: a reset restores the configuration the scene was built at and
        leaves inertial properties alone.
        """
        # Neither the masses nor the links they are written to are read back for checking: a condition drawn from
        # values on the device forces a synchronization at every call. The mass is spread over the links given, so
        # they have to carry some between them. A selection already on the device is left as it is, for that reason.
        if not isinstance(links_idx, torch.Tensor):
            if links_idx is None:
                links = self.links
            elif isinstance(links_idx, slice):
                links = self.links[links_idx]
            elif isinstance(links_idx, (int, np.integer)):
                links = (self.links[links_idx],)
            else:
                links = [self.links[i_l] for i_l in links_idx]
            if sum(0.0 if link.is_fixed else link.desc.mass for link in links) <= gs.EPS:
                gs.raise_exception(
                    "None of the links being set holds a mass to spread out, so there is no mass to set."
                )

        self._set_links_info(mass, links_idx, "mass", envs_idx, scale_inertia=scale_inertia)

    def set_links_inertia(self, inertia, links_idx=None, envs_idx=None):
        """Set the inertia tensor of the given links, expressed in their inertial frame.

        The caller is responsible for making sure that the inertia specified is physically sound, that is symmetric
        positive definite. There is no runtime check of this precondition, for the sake of efficiency.
        """
        self._set_links_info(inertia, links_idx, "inertia", envs_idx)

    def set_links_COM(self, com, links_idx=None, envs_idx=None):
        """Set the center of mass (COM) of the given links, as an offset from the origin of their local frame."""
        self._set_links_info(com, links_idx, "COM", envs_idx)

    @mutates(StateChange.MODEL)
    def set_geoms_friction_ratio(self, friction_ratio, geoms_idx=None, envs_idx=None):
        friction_ratio, geoms_idx, envs_idx = self._sanitize_io_variables(
            friction_ratio, geoms_idx, self.n_geoms, "geoms_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            friction_ratio = friction_ratio[None]
        kernel_set_geoms_friction_ratio(geoms_idx, envs_idx, friction_ratio, self.dyn_state, self.rigid_config)

    @mutates(StateChange.GEOMETRY, links=MutatedLinks.ARTICULATED)
    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False):
        if self._system is not None:
            self._system.reset(envs_idx)
        if self.constraint_solver is not None:
            self.constraint_solver.reset(envs_idx)

        if gs.use_zerocopy:
            data = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
            errno = qd_to_torch(self._errno, copy=False)
            qs_mask = indices_to_mask(qs_idx)
            if (
                (not qs_mask or isinstance(qs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and envs_idx.dtype == torch.bool
            ):
                qs_data = data[(slice(None), *qs_mask)]
                if qpos.ndim == 2 and len(qpos) not in (1, len(qs_data)):
                    # A fresh source view is needed because masked_scatter_ may reshape it in-place. Metal mis-scatters
                    # a stride-0 broadcast mask, so it must be materialized to a dense full-shape mask there.
                    envs_mask = envs_idx[:, None]
                    if gs.backend == gs.metal:
                        envs_mask = envs_mask.expand_as(qs_data).contiguous()
                    qs_data.masked_scatter_(envs_mask, qpos.view_as(qpos))
                else:
                    qpos = broadcast_tensor(qpos, gs.tc_float, qs_data.shape)
                    torch.where(envs_idx[:, None], qpos, qs_data, out=qs_data)
                errno.masked_fill_(envs_idx, 0.0)
            else:
                mask = (0, *qs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *qs_mask, boolean_mask=False)
                assign_indexed_tensor(data, mask, qpos)
                errno[envs_idx] = 0
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

            # Teleporting a sleeping body must revive it, otherwise the new pose is silently dropped. Nothing can be
            # hibernated while the scene is still being built.
            if self._use_hibernation and self.is_built:
                kernel_wake_up_entities_by_qs(
                    qs_idx,
                    envs_idx,
                    self.dyn_state,
                    self.constraint_solver.constraint_state,
                    self.dyn_info,
                    self.rigid_info,
                    self.rigid_config,
                )

            kernel_set_qpos(qs_idx, envs_idx, qpos, self.rigid_info, self.rigid_config)
            kernel_set_zero(envs_idx, self._errno)

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    @mutates(StateChange.MODEL)
    def set_global_sol_params(self, sol_params):
        """
        Set constraint solver parameters.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: Tuple[float] | List[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        """
        sol_params_ = broadcast_tensor(sol_params, gs.tc_float, (7,), ("",))
        sol_params_ = _sanitize_sol_params(sol_params_.clone(), self._sol_min_timeconst)
        kernel_set_global_sol_params(sol_params_, self.dyn_info, self.rigid_config)

    @mutates(StateChange.MODEL)
    def set_sol_params(self, sol_params, geoms_idx=None, envs_idx=None, *, joints_idx=None, eqs_idx=None):
        """
        Set constraint solver parameters.

        See :func:`genesis.utils.geom.default_solver_params` for the parameter semantics, in particular the
        relationship between ``dampratio``, spring stiffness, and velocity damping.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: Tuple[float] | List[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        """
        # Make sure that a single constraint type has been selected at once
        if sum(inputs_idx is not None for inputs_idx in (geoms_idx, joints_idx, eqs_idx)) > 1:
            gs.raise_exception("Cannot set more than one constraint type at once.")

        # Select the right input type
        if eqs_idx is not None:
            constraint_type = ConstraintType.EQUALITY
            idx_name = "eqs_idx"
            inputs_idx = eqs_idx
            inputs_length = self.n_equalities
            batched = True
        elif joints_idx is not None:
            constraint_type = ConstraintType.JOINT
            idx_name = "joints_idx"
            inputs_idx = joints_idx
            inputs_length = self.n_joints
            batched = self._options.batch_joints_info
        else:
            constraint_type = ConstraintType.GEOM
            idx_name = "geoms_idx"
            inputs_idx = geoms_idx
            inputs_length = self.n_geoms
            batched = False

        # Sanitize input arguments
        sol_params_, inputs_idx, envs_idx = self._sanitize_io_variables(
            sol_params, inputs_idx, inputs_length, idx_name, envs_idx, (7,), batched=batched, skip_allocation=True
        )
        # Geom values resolve an unspecified time constant to the default, then defer the floor to contact
        # assembly, which floors the mixed pair value (see func_set_contact_data), mirroring the build-time
        # sanitization; joint and equality values are consumed unmixed, so they are floored here.
        sol_params_ = _sanitize_sol_params(
            sol_params_.clone(),
            self._sol_min_timeconst,
            self._sol_default_timeconst if constraint_type == ConstraintType.GEOM else None,
            floor_timeconst=constraint_type != ConstraintType.GEOM,
        )
        if self.n_envs == 0 and batched:
            sol_params_ = sol_params_[None]

        kernel_set_sol_params(inputs_idx, envs_idx, sol_params_, self.dyn_info, self.rigid_config, int(constraint_type))

    @mutates(StateChange.MODEL)
    def _set_dofs_info(self, tensor_list, dofs_idx, name, envs_idx=None):
        if gs.use_zerocopy and name in {
            "kp",
            "kv",
            "act_gain",
            "act_bias",
            "force_range",
            "stiffness",
            "damping",
            "frictionloss",
            "limit",
        }:
            mask = indices_to_mask(*((envs_idx, dofs_idx) if self._options.batch_dofs_info else (dofs_idx,)))
            if name == "kp":
                # kp sets act_gain, act_bias[0] = 0, act_bias[1] = -kp (full PD reset)
                kp = torch.as_tensor(tensor_list[0], dtype=gs.tc_float, device=gs.device)
                gain = qd_to_torch(self.dyn_info.dofs.act_gain, transpose=True, copy=False)
                assign_indexed_tensor(gain, mask, kp)
                bias = qd_to_torch(self.dyn_info.dofs.act_bias, transpose=True, copy=False)
                bias[(*mask, ..., 0)] = 0.0
                assign_indexed_tensor(bias, (*mask, ..., 1), -kp)
            elif name == "kv":
                # kv sets act_bias[..., 2] = -kv
                kv = torch.as_tensor(tensor_list[0], dtype=gs.tc_float, device=gs.device)
                bias = qd_to_torch(self.dyn_info.dofs.act_bias, transpose=True, copy=False)
                assign_indexed_tensor(bias, (*mask, ..., 2), -kv)
            else:
                data = qd_to_torch(getattr(self.dyn_info.dofs, name), transpose=True, copy=False)
                num_values = len(tensor_list)
                for j, mask_j in enumerate(((*mask, ..., j) for j in range(num_values)) if num_values > 1 else (mask,)):
                    assign_indexed_tensor(data, mask_j, tensor_list[j])
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

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
        elif name == "force_range":
            kernel_set_dofs_force_range(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "stiffness":
            kernel_set_dofs_stiffness(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "armature":
            kernel_set_dofs_armature(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
            self._system.refresh_invweight_and_meaninertia(envs_idx=envs_idx, force_update=True)
        elif name == "damping":
            kernel_set_dofs_damping(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "frictionloss":
            kernel_set_dofs_frictionloss(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "limit":
            kernel_set_dofs_limit(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "act_gain":
            kernel_set_dofs_act_gain(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "act_bias":
            kernel_set_dofs_act_bias(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def set_dofs_kp(self, kp, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kp], dofs_idx, "kp", envs_idx)

    def set_dofs_kv(self, kv, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kv], dofs_idx, "kv", envs_idx)

    def set_dofs_act_gain(self, act_gain, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([act_gain], dofs_idx, "act_gain", envs_idx)

    def set_dofs_act_bias(self, bias0, bias1, bias2, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([bias0, bias1, bias2], dofs_idx, "act_bias", envs_idx)

    def set_dofs_force_range(self, lower, upper, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "force_range", envs_idx)

    def set_dofs_stiffness(self, stiffness, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([stiffness], dofs_idx, "stiffness", envs_idx)

    def set_dofs_armature(self, armature, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([armature], dofs_idx, "armature", envs_idx)

    def set_dofs_damping(self, damping, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([damping], dofs_idx, "damping", envs_idx)

    def set_dofs_frictionloss(self, frictionloss, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([frictionloss], dofs_idx, "frictionloss", envs_idx)

    def set_dofs_limit(self, lower, upper, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx)

    @mutates(StateChange.GEOMETRY, links=MutatedLinks.ARTICULATED)
    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        self._system.reset(envs_idx)
        self.constraint_solver.reset(envs_idx)

        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]

        self._wake_dofs(dofs_idx, envs_idx)

        kernel_set_dofs_position(
            dofs_idx, envs_idx, position, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

        if gs.use_zerocopy:
            errno = qd_to_torch(self._errno, copy=False)
            errno[envs_idx] = 0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_set_zero(envs_idx, self._errno)

        kernel_forward_kinematics_links_geoms(
            envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def _wake_dofs(self, dofs_idx, envs_idx):
        # Revive any hibernated entity owning these (already sanitized) dofs before a state is written to them;
        # forward dynamics and integration act only on awake dofs, so a state written to a sleeping body would
        # otherwise be silently dropped until it is woken by some other means.
        if self._use_hibernation:
            kernel_wake_up_entities_by_dofs(
                dofs_idx,
                envs_idx,
                self.dyn_state,
                self.constraint_solver.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False):
        # Wake the owning entities before delegating to the base setter, which re-sanitizes and applies the write.
        if self._use_hibernation:
            _, wake_dofs_idx, wake_envs_idx = self._sanitize_io_variables(
                velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            self._wake_dofs(wake_dofs_idx, wake_envs_idx)
        super().set_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=skip_forward)

    def control_dofs_force(self, force, dofs_idx=None, envs_idx=None):
        # A control target is consumed by the actuation pass of the next step, which wakes a sleeping link it actuates
        # (see func_torque_and_passive_force), so the target is written without waking anything here.
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.FORCE
            ctrl_force = qd_to_torch(self.dyn_state.dofs.ctrl_force, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_force, mask, force)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        force, dofs_idx, envs_idx = self._sanitize_io_variables(
            force, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            force = force[None]

        kernel_control_dofs_force(dofs_idx, envs_idx, force, self.dyn_state, self.rigid_config)

    def control_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.VELOCITY
            ctrl_pos = qd_to_torch(self.dyn_state.dofs.ctrl_pos, transpose=True, copy=False)
            ctrl_pos[mask] = 0.0
            ctrl_vel = qd_to_torch(self.dyn_state.dofs.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity = velocity[None]

        kernel_control_dofs_velocity(dofs_idx, envs_idx, velocity, self.dyn_state, self.rigid_config)

    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(self.dyn_state.dofs.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(self.dyn_state.dofs.ctrl_vel, transpose=True, copy=False)
            ctrl_vel[mask] = 0.0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]

        kernel_control_dofs_position(dofs_idx, envs_idx, position, self.dyn_state, self.rigid_config)

    def control_dofs_position_velocity(self, position, velocity, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(self.dyn_state.dofs.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(self.dyn_state.dofs.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        position, dofs_idx, _ = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]
            velocity = velocity[None]

        kernel_control_dofs_position_velocity(dofs_idx, envs_idx, position, velocity, self.dyn_state, self.rigid_config)

    def get_sol_params(self, geoms_idx=None, envs_idx=None, *, joints_idx=None, eqs_idx=None):
        """
        Get constraint solver parameters.
        """
        if eqs_idx is not None:
            # Always batched
            tensor = qd_to_torch(self.dyn_info.equalities.sol_params, envs_idx, eqs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
                tensor = tensor[0]
        elif joints_idx is not None:
            # Conditionally batched
            assert envs_idx is None
            # batch_shape = (envs_idx, joints_idx) if self._options.batch_joints_info else (joints_idx,) tensor =
            # qd_to_torch(self.dyn_info.joints.sol_params, *batch_shape, transpose=True)
            tensor = qd_to_torch(self.dyn_info.joints.sol_params, envs_idx, joints_idx, transpose=True, copy=True)
            if self.n_envs == 0 and self._options.batch_joints_info:
                tensor = tensor[0]
        else:  # geoms_idx is not None
            # Never batched
            assert envs_idx is None
            tensor = qd_to_torch(self.dyn_info.geoms.sol_params, geoms_idx, transpose=True, copy=True)
        return tensor

    @staticmethod
    def _sanitize_ref_frame(ref: link_ref_frame, *, has_root_COM: bool = True) -> int:
        """Check that a reference frame is supported and return it as a plain int.

        The value is returned as a plain int rather than the enum member itself because quadrants' fastcache holds a
        weak reference to template arguments, which enum members do not support.
        """
        frames = tuple(link_ref_frame) if has_root_COM else (link_ref_frame.link_origin, link_ref_frame.link_COM)
        if not isinstance(ref, link_ref_frame) or ref not in frames:
            refs = ", ".join(f"'gs.link_ref_frame.{frame.name}'" for frame in frames)
            gs.raise_exception(f"'ref' must be one of {refs}.")
        return int(ref)

    def get_links_pos(
        self,
        links_idx=None,
        envs_idx=None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative=False,
    ):
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = self._sanitize_io_variables(
                None, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )

        ref_frame = self._sanitize_ref_frame(ref)
        if ref_frame == link_ref_frame.root_COM:
            tensor = qd_to_torch(self.dyn_state.links.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        elif ref_frame == link_ref_frame.link_COM:
            i_pos = qd_to_torch(self.dyn_state.links.i_pos, envs_idx, links_idx, transpose=True)
            root_COM = qd_to_torch(self.dyn_state.links.root_COM, envs_idx, links_idx, transpose=True)
            tensor = i_pos + root_COM
        else:
            tensor = qd_to_torch(self.dyn_state.links.pos, envs_idx, links_idx, transpose=True, copy=True)

        # The pose offset is defined on the link origin, so it is only stripped for the 'link_origin' reference.
        if relative and ref_frame == link_ref_frame.link_origin and not self._links_offset_pos_is_identity[idx].all():
            quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)

        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_vel(
        self, links_idx=None, envs_idx=None, *, ref: link_ref_frame = link_ref_frame.link_origin, relative=False
    ):
        ref_frame = self._sanitize_ref_frame(ref, has_root_COM=False)
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        # Only the 'link_origin' reference carries the offset, as in 'get_links_pos'.
        is_relative = (
            relative and ref_frame == link_ref_frame.link_origin and not self._links_offset_pos_is_identity[idx].all()
        )
        if gs.use_zerocopy:
            mask = indices_to_mask(envs_idx, links_idx)
            cd_vel = qd_to_torch(self.dyn_state.links.cd_vel, transpose=True)
            cd_ang = qd_to_torch(self.dyn_state.links.cd_ang, transpose=True)
            if ref_frame == link_ref_frame.link_COM:
                i_pos = qd_to_torch(self.dyn_state.links.i_pos, transpose=True)
                delta = i_pos[mask]
            else:
                pos = qd_to_torch(self.dyn_state.links.pos, transpose=True)
                root_COM = qd_to_torch(self.dyn_state.links.root_COM, transpose=True)
                delta = pos[mask] - root_COM[mask]
                if is_relative:
                    quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
                    offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                    offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                    delta = delta - _offset_world_shift(offset_pos, offset_quat, quat)
            tensor = cd_vel[mask] + cd_ang[mask].cross(delta, dim=-1)
            return tensor[0] if self.n_envs == 0 else tensor

        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_links_vel(
            links_idx,
            envs_idx,
            tensor,
            self._links_offset_pos,
            self._links_offset_quat,
            self.dyn_state,
            self.rigid_config,
            ref_frame,
            is_relative=is_relative,
        )
        return _tensor

    def get_links_acc(self, links_idx=None, envs_idx=None, *, relative=False):
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_links_acc(
            links_idx,
            envs_idx,
            tensor,
            self._links_offset_pos,
            self._links_offset_quat,
            self.dyn_state,
            self.rigid_config,
            is_relative=relative and not self._links_offset_pos_is_identity[idx].all(),
        )
        return _tensor

    def get_links_acc_ang(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.links.cacc_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_root_COM(self, links_idx=None, envs_idx=None):
        """
        Returns the center of mass (COM) of the entire kinematic tree to which the specified links belong.

        This corresponds to the global COM of each entity, assuming a single-rooted structure - that is, as long as no
        two successive links are connected by a free-floating joint (ie a joint that allows all 6 degrees of freedom).
        """
        tensor = qd_to_torch(self.dyn_state.links.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_COM(self, links_idx=None, envs_idx=None):
        """The center of mass (COM) of each link, as an offset from the origin of its local frame."""
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.inertial_pos, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

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

    def get_links_invweight(self, links_idx=None, envs_idx=None):
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.invweight, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_geoms_friction_ratio(self, geoms_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.geoms.friction_ratio, envs_idx, geoms_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_geoms_pos(self, geoms_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.dyn_state.geoms.pos, envs_idx, geoms_idx, transpose=True, copy=True)
        if relative and self._geoms_offset_pos is not None:
            quat = qd_to_torch(self.dyn_state.geoms.quat, envs_idx, geoms_idx, transpose=True, copy=True)
            offset_pos = self._geoms_offset_pos if geoms_idx is None else self._geoms_offset_pos[geoms_idx]
            offset_quat = self._geoms_offset_quat if geoms_idx is None else self._geoms_offset_quat[geoms_idx]
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_geoms_quat(self, geoms_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.dyn_state.geoms.quat, envs_idx, geoms_idx, transpose=True, copy=True)
        if relative and self._geoms_offset_quat is not None:
            offset_quat = self._geoms_offset_quat if geoms_idx is None else self._geoms_offset_quat[geoms_idx]
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_control_force(self, dofs_idx=None, envs_idx=None):
        _tensor, dofs_idx, envs_idx = self._sanitize_io_variables(None, dofs_idx, self.n_dofs, "dofs_idx", envs_idx)
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_dofs_control_force(dofs_idx, envs_idx, tensor, self.dyn_state, self.dyn_info, self.rigid_config)
        return _tensor

    def get_dofs_actuator_force(self, dofs_idx=None, envs_idx=None):
        """
        Generalized effort transmitted to each DOF at the actuator output (torque for revolute DOFs, force for
        prismatic DOFs), accounting for the gearbox losses between the motor and the joint.

        Computed as qf_applied - armature * qacc + qf_frictionloss + qf_passive: the commanded effort from
        get_dofs_control_force minus the armature-inertia load, plus the dissipative frictionloss and passive damping
        efforts. Contact, Coriolis and gravity loads are captured implicitly through the constraint-solved acceleration.
        """
        qf_applied = qd_to_torch(self.dyn_state.dofs.qf_applied, envs_idx, transpose=True)
        qacc = qd_to_torch(self.constraint_solver.qacc, envs_idx, transpose=True)
        qf_passive = qd_to_torch(self.dyn_state.dofs.qf_passive, envs_idx, transpose=True)
        if self._options.batch_dofs_info:
            armature = qd_to_torch(self.dyn_info.dofs.armature, envs_idx, transpose=True)
            frictionloss = qd_to_torch(self.dyn_info.dofs.frictionloss, envs_idx, transpose=True)
        else:
            armature = qd_to_torch(self.dyn_info.dofs.armature, transpose=True)
            frictionloss = qd_to_torch(self.dyn_info.dofs.frictionloss, transpose=True)

        # Frictionloss constraint forces mapped back to DOF space. Frictionloss constraints occupy the contiguous block
        # [n_constraints_equality, n_constraints_equality + n_constraints_frictionloss) of the constraint list and have
        # an identity Jacobian, so `efc_force` at a frictionloss row is exactly the DOF-space frictionloss effort. The
        # assembly loop appends them in ascending DOF order (it iterates links -> joints -> DOFs serially within each
        # env, matching the global DOF numbering), so the k-th frictionloss row is the k-th DOF with nonzero
        # frictionloss. Its row index is therefore `n_constraints_equality + rank`, with `rank` the running count of
        # frictionloss-enabled DOFs (-1 for DOFs without frictionloss, which contribute zero).
        efc_force = qd_to_torch(self.constraint_solver.efc_force, envs_idx, transpose=True)
        n_constraints_equality = qd_to_torch(self.constraint_solver.n_constraints_equality, envs_idx)
        has_frictionloss = frictionloss > gs.EPS
        rank = torch.cumsum(has_frictionloss, dim=-1) - 1
        gather_idx = (n_constraints_equality[:, None] + rank).clamp_(min=0)
        qf_frictionloss = torch.gather(efc_force, 1, gather_idx) * has_frictionloss

        actuator_force = qf_applied - armature * qacc + qf_frictionloss + qf_passive
        if dofs_idx is not None:
            actuator_force = actuator_force[indices_to_mask(None, dofs_idx)]
        return actuator_force[0] if self.n_envs == 0 else actuator_force

    def get_dofs_force(self, dofs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.dofs.force, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_acc(self, dofs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.dofs.acc, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_kp(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        gain = qd_to_torch(self.dyn_info.dofs.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        bias = qd_to_torch(self.dyn_info.dofs.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            gain, bias = gain[0], bias[0]
        if not (torch.abs(gain + bias[..., 1]) < gs.EPS * torch.clamp(torch.abs(gain), min=1.0)).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_gain != -act_bias[1]). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        if not (torch.abs(bias[..., 0]) < gs.EPS).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_bias[0] != 0). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        return gain

    def get_dofs_kv(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        gain = qd_to_torch(self.dyn_info.dofs.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        bias = qd_to_torch(self.dyn_info.dofs.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            gain, bias = gain[0], bias[0]
        if not (torch.abs(gain + bias[..., 1]) < gs.EPS * torch.clamp(torch.abs(gain), min=1.0)).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_gain != -act_bias[1]). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        if not (torch.abs(bias[..., 0]) < gs.EPS).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_bias[0] != 0). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        return -bias[..., 2]

    def get_dofs_act_gain(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_act_bias(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1], tensor[..., 2]

    def get_dofs_force_range(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.force_range, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_stiffness(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.stiffness, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_invweight(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.invweight, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_armature(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.armature, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_damping(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.damping, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_frictionloss(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.frictionloss, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_mass_mat(self, dofs_idx=None, envs_idx=None, decompose=False):
        tensor = qd_to_torch(self.mass_mat_L if decompose else self.mass_mat, envs_idx, transpose=True, copy=True)
        if dofs_idx is not None:
            tensor = tensor[indices_to_mask(None, dofs_idx, dofs_idx)]
        if self.n_envs == 0:
            tensor = tensor[0]

        if decompose:
            mass_mat_D_inv = qd_to_torch(self.rigid_info.mass_mat_D_inv, envs_idx, dofs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
                mass_mat_D_inv = mass_mat_D_inv[0]
            return tensor, mass_mat_D_inv

        return tensor

    def get_kinetic_energy(self, links_idx=None, dofs_idx=None, envs_idx=None):
        """Get the kinetic energy of the specified links and DOFs in Joules [J] (translational + rotational).

        Summed over the links, each contributing ``0.5 * V^T * I * V`` for its spatial velocity ``V`` and spatial
        inertia ``I`` about the center of mass (COM) of its kinematic tree, plus the motor armature contribution
        ``0.5 * sum_d(armature_d * dq_d^2)`` of the DOFs. This equals the joint-space form ``0.5 * dq^T * M(q) * dq``
        while reading the link velocities and inertias that forward kinematics already maintains, so it needs no mass
        matrix and stays consistent with the current configuration whatever the integrator.

        A link and the DOFs driving it contribute independently, so selecting one without the other is meaningful only
        to isolate that one term.

        Parameters
        ----------
        links_idx : None | array_like, optional
            The indices of the links. If None, all links will be considered. Defaults to None.
        dofs_idx : None | array_like, optional
            The indices of the degrees of freedom. If None, all of them will be considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kinetic_energy : torch.Tensor, shape () or (n_envs,)
        """
        # Spatial velocity and inertia are both referenced to the tree COM, so the quadratic form expands as
        # 0.5 * m * |v|^2 + v . (w x (m * c)) + 0.5 * w . (I * w), with `cinr_pos` holding the first moment m * c.
        cd_ang = qd_to_torch(self.dyn_state.links.cd_ang, envs_idx, links_idx, transpose=True)
        cd_vel = qd_to_torch(self.dyn_state.links.cd_vel, envs_idx, links_idx, transpose=True)
        cinr_inertial = qd_to_torch(self.dyn_state.links.cinr_inertial, envs_idx, links_idx, transpose=True)
        cinr_pos = qd_to_torch(self.dyn_state.links.cinr_pos, envs_idx, links_idx, transpose=True)
        cinr_mass = qd_to_torch(self.dyn_state.links.cinr_mass, envs_idx, links_idx, transpose=True)
        translational = cinr_mass * torch.sum(cd_vel * cd_vel, dim=-1)
        coupling = torch.sum(cd_vel * torch.cross(cd_ang, cinr_pos, dim=-1), dim=-1)
        rotational = torch.sum(cd_ang * torch.matmul(cinr_inertial, cd_ang.unsqueeze(-1)).squeeze(-1), dim=-1)
        kinetic_energy = torch.sum(0.5 * (translational + rotational) + coupling, dim=-1)

        dofs_vel = qd_to_torch(self.dyn_state.dofs.vel, envs_idx, dofs_idx, transpose=True)
        armature = self.get_dofs_armature(dofs_idx, envs_idx if self._options.batch_dofs_info else None)
        kinetic_energy += 0.5 * torch.sum(armature * dofs_vel * dofs_vel, dim=-1)

        return kinetic_energy[0] if self.n_envs == 0 else kinetic_energy

    def get_potential_energy(self, links_idx=None, dofs_idx=None, envs_idx=None):
        """Get the potential energy of the specified links and DOFs in Joules [J] (gravitational + joint springs).

        Gravity contributes ``-sum_i(m_i * g^T * p_i)`` over the links free to move, where ``p_i`` is the center of
        mass (COM) position of link *i*. A link fixed to the world is left out, as it is of the mass of an entity: its
        potential is a constant of the scene. Joint springs contribute ``0.5 * sum_d(stiffness_d * (q_d - q0_d)^2)``,
        the elastic energy stored by holding each DOF away from its neutral position. Both are state functions, so
        their sum with the kinetic energy is conserved by a passive, frictionless, contact-free model.

        Contacts contribute nothing: they are resolved by the constraint solver, which stabilizes a penetration
        rather than storing it as an elastic potential.

        Parameters
        ----------
        links_idx : None | array_like, optional
            The indices of the links. If None, all links will be considered. Defaults to None.
        dofs_idx : None | array_like, optional
            The indices of the degrees of freedom. If None, all of them will be considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        potential_energy : torch.Tensor, shape () or (n_envs,)
        """
        gravity = self.get_gravity(envs_idx=envs_idx)  # (3,) or (n_envs, 3)
        links_pos = self.get_links_pos(links_idx, envs_idx, ref=link_ref_frame.link_COM)  # (..., n_links, 3)
        links_envs_idx = envs_idx if self._options.batch_links_info else None
        links_mass = qd_to_torch(self.dyn_info.links.inertial_mass, links_envs_idx, links_idx, transpose=True)
        links_is_fixed = qd_to_torch(self.dyn_info.links.is_fixed, links_envs_idx, links_idx, transpose=True)
        links_mass = links_mass.masked_fill(links_is_fixed.to(torch.bool), 0.0)
        if self.n_envs == 0 and self._options.batch_links_info:
            links_mass = links_mass[0]

        # PE_i = m_i * g^T * p_i => PE = sum_i(m_i * (g . p_i))
        # g is (..., 3), links_pos is (..., n_links, 3) -> broadcast g to (..., 1, 3)
        g_dot_p = torch.sum(gravity.unsqueeze(-2) * links_pos, dim=-1)  # (..., n_links)
        potential_energy = -torch.sum(links_mass * g_dot_p, dim=-1)

        # `dofs.pos` holds the deflection `qpos - qpos0` that the spring force is proportional to, so the elastic
        # energy integrates directly against it whatever the neutral configuration.
        dofs_pos = qd_to_torch(self.dyn_state.dofs.pos, envs_idx, dofs_idx, transpose=True)
        stiffness = self.get_dofs_stiffness(dofs_idx, envs_idx if self._options.batch_dofs_info else None)
        spring_energy = 0.5 * torch.sum(stiffness * dofs_pos * dofs_pos, dim=-1)
        if self.n_envs == 0:
            spring_energy = spring_energy[0]

        return potential_energy + spring_energy

    def get_total_energy(self, envs_idx=None):
        """Get the total mechanical energy of all entities in Joules [J] (kinetic + potential).

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        total_energy : torch.Tensor, shape () or (n_envs,)
        """
        return self.get_kinetic_energy(envs_idx=envs_idx) + self.get_potential_energy(envs_idx=envs_idx)

    def get_geoms_friction(self, geoms_idx=None):
        return qd_to_torch(self.dyn_info.geoms.friction, geoms_idx, copy=True)

    def get_geoms_friction_torsional(self, geoms_idx=None):
        return qd_to_torch(self.dyn_info.geoms.friction_torsional, geoms_idx, copy=True)

    def get_geoms_friction_rolling(self, geoms_idx=None):
        return qd_to_torch(self.dyn_info.geoms.friction_rolling, geoms_idx, copy=True)

    def get_AABB(self, entities_idx=None, envs_idx=None):
        from genesis.engine.couplers import LegacyCoupler

        if not isinstance(self.sim.coupler, LegacyCoupler):
            gs.raise_exception("Method only supported when using 'LegacyCoupler' coupler type.")

        aabb_min = qd_to_torch(self.dyn_state.geoms.aabb_min, envs_idx, transpose=True)
        aabb_max = qd_to_torch(self.dyn_state.geoms.aabb_max, envs_idx, transpose=True)

        aabb = torch.stack([aabb_min, aabb_max], dim=-2)

        if entities_idx is not None:
            entity_geom_starts = []
            entity_geom_ends = []
            for entity_idx in entities_idx:
                entity = self._entities[entity_idx]
                entity_geom_starts.append(entity._geom_start)
                entity_geom_ends.append(entity._geom_start + entity.n_geoms)

            entity_aabbs = []
            for start, end in zip(entity_geom_starts, entity_geom_ends):
                if start < end:
                    entity_geoms_aabb = aabb[..., start:end, :, :]
                    entity_min = entity_geoms_aabb[..., :, 0, :].min(dim=-2)[0]
                    entity_max = entity_geoms_aabb[..., :, 1, :].max(dim=-2)[0]
                    entity_aabb = torch.stack([entity_min, entity_max], dim=-2)
                else:
                    entity_aabb = torch.zeros_like(aabb[..., 0:1, :, :])
                entity_aabbs.append(entity_aabb)

            aabb = torch.stack(entity_aabbs, dim=-2)

        return aabb[0] if self.n_envs == 0 else aabb

    @mutates(StateChange.MODEL)
    def set_geom_friction(self, friction, geoms_idx):
        kernel_set_geom_friction(geoms_idx, friction, self.dyn_info)

    @mutates(StateChange.MODEL)
    def set_geom_friction_torsional(self, friction_torsional, geoms_idx):
        kernel_set_geom_friction_torsional(geoms_idx, friction_torsional, self.dyn_info)

    @mutates(StateChange.MODEL)
    def set_geom_friction_rolling(self, friction_rolling, geoms_idx):
        kernel_set_geom_friction_rolling(geoms_idx, friction_rolling, self.dyn_info)

    @mutates(StateChange.MODEL)
    def set_geoms_friction(self, friction, geoms_idx=None):
        friction, geoms_idx, _ = self._sanitize_io_variables(
            friction, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, batched=False, skip_allocation=True
        )
        kernel_set_geoms_friction(geoms_idx, friction, self.dyn_info, self.rigid_config)

    @mutates(StateChange.MODEL)
    def set_geoms_friction_torsional(self, friction_torsional, geoms_idx=None):
        friction_torsional, geoms_idx, _ = self._sanitize_io_variables(
            friction_torsional, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, batched=False, skip_allocation=True
        )
        kernel_set_geoms_friction_torsional(geoms_idx, friction_torsional, self.dyn_info, self.rigid_config)

    @mutates(StateChange.MODEL)
    def set_geoms_friction_rolling(self, friction_rolling, geoms_idx=None):
        friction_rolling, geoms_idx, _ = self._sanitize_io_variables(
            friction_rolling, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, batched=False, skip_allocation=True
        )
        kernel_set_geoms_friction_rolling(geoms_idx, friction_rolling, self.dyn_info, self.rigid_config)

    @mutates(StateChange.MODEL)
    def add_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        return self.constraint_solver.add_weld_constraint(link1_idx, link2_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def delete_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        return self.constraint_solver.delete_weld_constraint(link1_idx, link2_idx, envs_idx)

    def get_weld_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_weld_constraints(as_tensor, to_torch)

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_equality_constraints(as_tensor, to_torch)

    def clear_external_force(self):
        if gs.use_zerocopy:
            for tensor in (self.dyn_state.links.cfrc_applied_ang, self.dyn_state.links.cfrc_applied_vel):
                out = qd_to_torch(tensor, copy=False)
                out.zero_()
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        kernel_clear_external_force(self.dyn_state, self.rigid_info, self.rigid_config)

    def update_drone_propeller_vgeoms(self, propellers_vgeom_idxs, propellers_revs, propellers_spin):
        kernel_update_drone_propeller_vgeoms(
            propellers_vgeom_idxs, propellers_revs, propellers_spin, self.dyn_state, self.rigid_info, self.rigid_config
        )

    def set_drone_rpm(self, propellers_link_idx, kf, km, propellers_rpm, propellers_spin, invert):
        kernel_set_drone_rpm(
            propellers_link_idx, kf, km, propellers_rpm, propellers_spin, self.dyn_state, self.rigid_config, invert
        )

    def update_verts_for_geoms(self, geoms_idx):
        _, geoms_idx, _ = self._sanitize_io_variables(
            None, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, skip_allocation=True
        )
        kernel_update_verts_for_geoms(geoms_idx, self.dyn_state, self.dyn_info, self.rigid_config)

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
    def max_collision_pairs(self):
        return self._max_collision_pairs

    @property
    def n_equalities(self):
        if self.is_built:
            return self._n_equalities
        return sum(entity.n_equalities for entity in self._entities)

    @property
    def equalities(self):
        if self.is_built:
            return self._equalities
        return gs.List(equality for entity in self._entities for equality in entity.equalities)


@qd.func
def func_step_1(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
    is_backward: qd.template(),
):
    if qd.static(not is_forward_pos_updated):
        func_update_cartesian_space(
            dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms=False, is_backward=is_backward
        )

    if qd.static(not is_forward_vel_updated):
        func_forward_velocity(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)

    func_forward_dynamics(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.kernel(fastcache=True)
def kernel_step_1(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
    is_backward: qd.template(),
):
    """Run the first half of a substep on its own (see func_step_1)."""
    func_step_1(
        dyn_state,
        constraint_state,
        dyn_info,
        rigid_info,
        rigid_config,
        is_forward_pos_updated,
        is_forward_vel_updated,
        is_backward,
    )


@qd.func
def func_step_2(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
    errno: qd.Tensor,
):
    # Position, Velocity and Acceleration data must be consistent when computing links acceleration, otherwise it
    # would not corresponds to anyting physical. There is no other way than doing this right before integration,
    # because the acceleration at the end of the step is unknown for now as it may change discontinuous between
    # before and after integration under the effect of external forces and constraints. This means that
    # acceleration data will be shifted one timestep in the past, but there isn't really any way around.
    func_update_acc(dyn_state, dyn_info, rigid_info, rigid_config, update_cacc=True, is_backward=is_backward)

    if qd.static(rigid_config.integrator != gs.integrator.approximate_implicitfast):
        func_implicit_damping(dyn_state, dyn_info, rigid_info, rigid_config)

    func_integrate(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)

    if qd.static(not is_backward):
        func_copy_next_to_curr(dyn_state, rigid_info, rigid_config, errno)

        if qd.static(not rigid_config.enable_mujoco_compatibility):
            func_update_cartesian_space(
                dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms=False, is_backward=is_backward
            )
            func_forward_velocity(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.kernel(fastcache=True)
def kernel_step_2(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
    errno: qd.Tensor,
):
    """Run the second half of a substep on its own (see func_step_2)."""
    func_step_2(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_backward, errno)
