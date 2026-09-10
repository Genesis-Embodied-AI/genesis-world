from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch
from frozendict import frozendict

import genesis as gs

import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd import func_solve_mass_batch
from genesis.engine.solvers.rigid.abd.misc import linear_to_lower_tri
from genesis.utils.misc import qd_to_torch, indices_to_mask, assign_indexed_tensor

from .island import (
    func_build_islands,
    func_build_islands_coop,
    func_group_constraints_by_island,
    func_group_constraints_by_island_coop,
    func_sort_contacts,
    func_sort_contacts_coop,
)
from . import backward as backward_constraint_solver
from . import linesearch
from . import noslip as constraint_noslip


@qd.func
def _append_relevant_dof(
    i_con: qd.int32,
    i_d: qd.int32,
    i_b: qd.int32,
    n: qd.int32,
    dedup: qd.int32,
    constraint_state: array_class.ConstraintState,
):
    """Append dof i_d to jac_dofs_idx[i_con, :n, i_b] unless already present, returning the new count.

    A row coupling two links of the same kinematic tree walks both ancestor chains, so shared ancestor DOFs come up
    twice: every sparse consumer (J.v / J^T.v products, Hessian assembly, noslip residuals) treats the list as a
    set, and appending duplicates blindly can push the count past the row capacity (n_dofs), spilling into the next
    row. The serialized CPU assembly rebuilds rows in index order and self-heals the spill, but the parallel GPU
    assembly does not, leaving clobbered supports. Duplicates only ever arise while walking the second chain of a
    row whose links share a kinematic root, so callers pass dedup=False everywhere else and the O(n) scan - which
    costs >10% on contact-heavy free-body scenes - is skipped.
    """
    is_new = True
    if dedup:
        for j in range(n):
            if constraint_state.jac_dofs_idx[i_con, j, i_b] == i_d:
                is_new = False
    if is_new:
        constraint_state.jac_dofs_idx[i_con, n, i_b] = i_d
        n = n + 1
    return n


@qd.func
def _sort_relevant_dofs_descending(
    i_con: qd.int32,
    i_b: qd.int32,
    n: qd.int32,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
):
    """Insertion sort jac_dofs_idx[i_con, :n, i_b] in descending order.

    Only the sparse skyline / incremental Cholesky relies on globally descending DOF order; the J.v / J^T.v products
    and the per-island solves are order-independent. So the sort is skipped unless sparse_solve is set - it is a
    serial (data-dependent) loop that would otherwise serialize the parallel contact-assembly kernel on GPU. The
    array is typically <= 14 elements, so O(n^2) is fine.
    """
    if qd.static(rigid_config.sparse_solve):
        for i in range(1, n):
            key = constraint_state.jac_dofs_idx[i_con, i, i_b]
            j = i - 1
            while j >= 0 and constraint_state.jac_dofs_idx[i_con, j, i_b] < key:
                constraint_state.jac_dofs_idx[i_con, j + 1, i_b] = constraint_state.jac_dofs_idx[i_con, j, i_b]
                j -= 1
            constraint_state.jac_dofs_idx[i_con, j + 1, i_b] = key


if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class ConstraintSolver:
    def __init__(self, rigid_solver: "RigidSolver"):
        self._solver = rigid_solver
        self._collider = rigid_solver.collider
        self._B = rigid_solver._B
        self._para_level = rigid_solver._para_level

        self._solver_type = rigid_solver._options.constraint_solver
        self._n_iterations = int(rigid_solver._options.iterations)
        self.tolerance = rigid_solver._options.tolerance
        self.ls_iterations = rigid_solver._options.ls_iterations
        self.ls_tolerance = rigid_solver._options.ls_tolerance
        # Effective (CPU-gated) sparsity flag, resolved in the static config; the raw option may differ on GPU.
        self.sparse_solve = rigid_solver.rigid_config.sparse_solve

        # Note that it must be over-estimated because friction parameters and joint limits may be updated dynamically.
        # * rows_per_contact constraints per contact, bounded by the post-pruning contact budget enforced by the
        #   collider
        # * 1 constraint per 1DoF joint limit (upper and lower, if not inf)
        # * 1 constraint per dof frictionloss
        # * up to 6 constraints per equality (weld)
        # When 'max_contacts' is set, it overrides the post-pruning contact budget enforced by the collider.
        # Resolve the max_contacts option in place: from the collider's post-pruning budget when unset, else clamped
        # to the candidate budget and written back so the collider honors the user's cap. Downstream reads the option.
        collider_info = rigid_solver.collider.collider_info
        if rigid_solver._options.max_contacts is None:
            rigid_solver._options.max_contacts = collider_info.max_contacts[None]
        else:
            rigid_solver._options.max_contacts = min(
                rigid_solver._options.max_contacts, collider_info.max_candidate_contacts[None]
            )
            collider_info.max_contacts[None] = rigid_solver._options.max_contacts
        rows_per_contact = rigid_solver.rigid_config.rows_per_contact
        self.len_constraints = (
            rows_per_contact * rigid_solver._options.max_contacts
            + sum(joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC) for joint in self._solver.joints)
            + self._solver.n_dofs
            + self._solver.n_candidate_equalities_ * 6
        )
        self.len_constraints_ = max(1, self.len_constraints)
        # Max cone rows (rows_per_contact per contact); sizes the elliptic-only previous-residual buffer to exactly
        # the contact rows, below the full constraint count that also covers equalities, joint limits, and
        # frictionloss.
        self.n_cone_constraints_ = max(1, rows_per_contact * rigid_solver._options.max_contacts)

        self.constraint_state = array_class.get_constraint_state(self, self._solver, self._collider)
        self.constraint_state.qd_n_equalities.from_numpy(
            np.full((self._solver._B,), self._solver.n_equalities, dtype=gs.np_int)
        )

        self._eq_const_info_cache = {}

        cs = self.constraint_state
        self.qd_n_equalities = cs.qd_n_equalities
        self.jac = cs.jac
        self.diag = cs.diag
        self.aref = cs.aref
        self.jac_n_dofs = cs.jac_n_dofs
        self.jac_dofs_idx = cs.jac_dofs_idx
        self.n_constraints = cs.n_constraints
        self.n_constraints_equality = cs.n_constraints_equality
        self.n_constraints_frictionloss = cs.n_constraints_frictionloss
        self.n_constraints_cone = cs.n_constraints_cone
        self.improved = cs.improved
        self.Jaref = cs.Jaref
        self.Ma = cs.Ma
        self.Ma_ws = cs.Ma_ws
        self.grad = cs.grad
        self.Mgrad = cs.Mgrad
        self.search = cs.search
        self.efc_D = cs.efc_D
        self.efc_force = cs.efc_force
        self.active = cs.active
        self.prev_active = cs.prev_active
        self.qfrc_constraint = cs.qfrc_constraint
        self.qacc = cs.qacc
        self.qacc_ws = cs.qacc_ws
        self.qacc_prev = cs.qacc_prev
        self.cost_ws = cs.cost_ws
        self.cost = cs.cost
        self.mv = cs.mv
        self.jv = cs.jv
        if self._solver_type == gs.constraint_solver.CG:
            self.cg_prev_grad = cs.cg_prev_grad
            self.cg_prev_Mgrad = cs.cg_prev_Mgrad
        if self._solver_type == gs.constraint_solver.Newton:
            self.nt_H = cs.nt_H
            self.nt_vec = cs.nt_vec

        self.reset()

        # The hibernated-island daisy chain must start empty (-1 = no successor); it persists across steps, written
        # when an island hibernates and cleared on wakeup.
        if self._solver._use_hibernation:
            self.constraint_state.island.hibernated_next_link.fill(-1)

    @property
    def data(self) -> Iterator[array_class.DataItem]:
        """Yield every array of the constraint solver, tagged by kind (see 'Solver.data')."""
        yield from array_class.iter_data(self.constraint_state, "constraint_state")

    def reset(self, envs_idx=None):
        self._eq_const_info_cache.clear()

        if gs.use_zerocopy:
            envs_mask = indices_to_mask(envs_idx)
            dofs_mask = (slice(None), *envs_mask)
            is_warmstart = qd_to_torch(self.constraint_state.is_warmstart, copy=False)
            qacc_ws = qd_to_torch(self.constraint_state.qacc_ws, copy=False)
            assign_indexed_tensor(is_warmstart, envs_mask, False)
            assign_indexed_tensor(qacc_ws, dofs_mask, 0.0)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        constraint_solver_kernel_reset(envs_idx, self.constraint_state, self._solver.rigid_config)

    def clear(self, envs_idx=None):
        self.reset(envs_idx)

        if gs.use_zerocopy and (
            not isinstance(envs_idx, torch.Tensor) or (not IS_OLD_TORCH or envs_idx.dtype == torch.bool)
        ):
            n_constraints = qd_to_torch(self.constraint_state.n_constraints, copy=False)
            n_constraints_equality = qd_to_torch(self.constraint_state.n_constraints_equality, copy=False)
            n_constraints_frictionloss = qd_to_torch(self.constraint_state.n_constraints_frictionloss, copy=False)
            n_constraints_cone = qd_to_torch(self.constraint_state.n_constraints_cone, copy=False)
            qd_n_equalities = qd_to_torch(self.constraint_state.qd_n_equalities, copy=False)
            n_eq = self._solver._n_equalities
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                n_constraints.masked_fill_(envs_idx, 0)
                n_constraints_equality.masked_fill_(envs_idx, 0)
                n_constraints_frictionloss.masked_fill_(envs_idx, 0)
                n_constraints_cone.masked_fill_(envs_idx, 0)
                qd_n_equalities.masked_fill_(envs_idx, n_eq)
            elif isinstance(envs_idx, torch.Tensor):
                n_constraints.scatter_(0, envs_idx, 0)
                n_constraints_equality.scatter_(0, envs_idx, 0)
                n_constraints_frictionloss.scatter_(0, envs_idx, 0)
                n_constraints_cone.scatter_(0, envs_idx, 0)
                qd_n_equalities.scatter_(0, envs_idx, n_eq)
            else:
                env_mask = indices_to_mask(envs_idx)
                assign_indexed_tensor(n_constraints, env_mask, 0)
                assign_indexed_tensor(n_constraints_equality, env_mask, 0)
                assign_indexed_tensor(n_constraints_frictionloss, env_mask, 0)
                assign_indexed_tensor(n_constraints_cone, env_mask, 0)
                assign_indexed_tensor(qd_n_equalities, env_mask, n_eq)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            fn = constraint_solver_kernel_masked_clear
        else:
            fn = constraint_solver_kernel_clear
        fn(envs_idx, self.constraint_state, self._solver.rigid_info, self._solver.rigid_config)

    def add_equality_constraints(self):
        self._eq_const_info_cache.clear()

        add_equality_constraints(
            self._solver.dyn_state,
            self._collider.collider_state,
            self.constraint_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
        )

    def add_inequality_constraints(self):
        add_inequality_constraints(
            self._solver.dyn_state,
            self._collider.collider_state,
            self.constraint_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
            self._collider.collider_config,
        )

    def resolve(self):
        # func_solve_init is launched by each dispatch entrypoint (func_solve_body_monolith / func_solve_decomposed),
        # not here: only the entrypoint statically knows its arm, which determines whether the init factor/gradient is
        # done (monolith) or skipped (decomposed re-factors in-loop).
        func_solve_body(
            self._solver.dyn_state,
            self.constraint_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
            self._n_iterations,
        )

        func_update_qacc(self._solver.dyn_state, self.constraint_state, self._solver.rigid_config, self._solver._errno)

        if self._solver._options.noslip_iterations > 0:
            self.noslip()

        func_update_contact_force(
            self._solver.dyn_state,
            self._collider.collider_state,
            self.constraint_state,
            self._solver.dyn_info,
            self._solver.rigid_config,
        )

    def noslip(self):
        constraint_noslip.kernel_noslip(
            self._solver.dyn_state,
            self._collider.collider_state,
            self.constraint_state,
            self._solver.rigid_info,
            self._solver.rigid_config,
        )

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        # Early return if already pre-computed
        eq_const_info = self._eq_const_info_cache.get((as_tensor, to_torch))
        if eq_const_info is not None:
            return eq_const_info.copy()

        n_eqs = tuple(self.constraint_state.qd_n_equalities.to_numpy())
        n_envs = len(n_eqs)
        n_eqs_max = max(n_eqs)

        if as_tensor:
            out_size = n_envs * n_eqs_max
        else:
            *n_eqs_starts, out_size = np.cumsum(n_eqs)

        if to_torch:
            iout = torch.full((out_size, 3), -1, dtype=gs.tc_int, device=gs.device)
            fout = torch.zeros((out_size, 6), dtype=gs.tc_float, device=gs.device)
        else:
            iout = np.full((out_size, 3), -1, dtype=gs.np_int)
            fout = np.zeros((out_size, 6), dtype=gs.np_float)

        if n_eqs_max > 0:
            kernel_get_equality_constraints(
                iout, fout, self.constraint_state, self._solver.dyn_info, self._solver.rigid_config, as_tensor
            )

        if as_tensor:
            iout = iout.reshape((n_envs, n_eqs_max, 3))
            eq_type, obj_a, obj_b = (iout[..., i] for i in range(3))
            efc_force = fout.reshape((n_envs, n_eqs_max, 6))
            values = (eq_type, obj_a, obj_b, fout)
        else:
            if to_torch:
                iout_chunks = torch.split(iout, n_eqs)
                efc_force = torch.split(fout, n_eqs)
            else:
                iout_chunks = np.split(iout, n_eqs_starts)
                efc_force = np.split(fout, n_eqs_starts)
            eq_type, obj_a, obj_b = tuple(zip(*([data[..., i] for i in range(3)] for data in iout_chunks)))

        values = (eq_type, obj_a, obj_b, efc_force)
        eq_const_info = dict(zip(("type", "obj_a", "obj_b", "force"), values))

        # Cache equality constraint information before returning
        self._eq_const_info_cache[(as_tensor, to_torch)] = eq_const_info

        return eq_const_info.copy()

    def get_weld_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        eq_const_info = self.get_equality_constraints(as_tensor, to_torch)
        eq_type = eq_const_info.pop("type")

        weld_const_info = {}
        if as_tensor:
            weld_mask = eq_type == gs.EQUALITY_TYPE.WELD
            n_envs = len(weld_mask)
            n_welds = weld_mask.sum(dim=-1) if to_torch else np.sum(weld_mask, axis=-1)
            n_welds_max = max(n_welds)
            for key, value in eq_const_info.items():
                shape = (n_envs, n_welds_max, *value.shape[2:])
                if to_torch:
                    if torch.is_floating_point(value):
                        weld_const_info[key] = torch.zeros(shape, dtype=value.dtype, device=value.device)
                    else:
                        weld_const_info[key] = torch.full(shape, -1, dtype=value.dtype, device=value.device)
                else:
                    if np.issubdtype(value.dtype, np.floating):
                        weld_const_info[key] = np.zeros(shape, dtype=value.dtype)
                    else:
                        weld_const_info[key] = np.full(shape, -1, dtype=value.dtype)
            for i_b, (n_welds_i, weld_mask_i) in enumerate(zip(n_welds, weld_mask)):
                for eq_value, weld_value in zip(eq_const_info.values(), weld_const_info.values()):
                    weld_value[i_b, :n_welds_i] = eq_value[i_b, weld_mask_i]
        else:
            weld_mask_chunks = tuple(eq_type_i == gs.EQUALITY_TYPE.WELD for eq_type_i in eq_type)
            for key, value in eq_const_info.items():
                weld_const_info[key] = tuple(data[weld_mask] for weld_mask, data in zip(weld_mask_chunks, value))

        weld_const_info["link_a"] = weld_const_info.pop("obj_a")
        weld_const_info["link_b"] = weld_const_info.pop("obj_b")

        return weld_const_info

    def add_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        link1_idx, link2_idx = int(link1_idx), int(link2_idx)

        assert link1_idx >= 0 and link2_idx >= 0
        weld_const_info = self.get_weld_constraints(as_tensor=True, to_torch=True)
        link_a = weld_const_info["link_a"]
        link_b = weld_const_info["link_b"]
        assert not (
            ((link_a == link1_idx) | (link_b == link1_idx)) & ((link_a == link2_idx) | (link_b == link2_idx))
        ).any()

        self._eq_const_info_cache.clear()
        overflow = kernel_add_weld_constraint(
            link1_idx,
            link2_idx,
            envs_idx,
            self._solver.dyn_state,
            self.constraint_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
        )
        if overflow:
            gs.logger.warning(
                "Ignoring dynamically registered weld constraint to avoid exceeding max number of equality constraints"
                f"({self.rigid_info.n_candidate_equalities.to_numpy()}). Please increase the value of "
                "RigidSolver's option 'max_dynamic_constraints'."
            )

    def delete_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        self._eq_const_info_cache.clear()
        kernel_delete_weld_constraint(
            int(link1_idx),
            int(link2_idx),
            envs_idx,
            self.constraint_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
        )

    def backward(self):
        """Adjoint solve of the constraint force computation.

        The caller must pre-populate the upstream gradient constraint_state.dL_dqacc (see
        RigidSolver._constraint_force_grad)."""
        if not self._solver._requires_grad:
            gs.raise_exception("Please set `requires_grad` to True in SimOptions to enable differentiable mode.")

        # 1. We first need to find a solution to A^T * u = g system.
        backward_constraint_solver.kernel_solve_adjoint_u(
            self.constraint_state, self._solver.dyn_info, self._solver.rigid_info, self._solver.rigid_config
        )

        # 2. Using the solution u, we can compute the gradients of the input variables.
        backward_constraint_solver.kernel_compute_gradients(self.constraint_state, self._solver.rigid_info)


# =====================================================================================================================
# ================================================= Getters / Setters =================================================
# =====================================================================================================================


@qd.kernel(fastcache=True)
def kernel_get_equality_constraints(
    iout: qd.types.ndarray(),
    fout: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    is_padded: qd.template(),
):
    _B = constraint_state.qd_n_equalities.shape[0]
    n_eqs_max = gs.qd_int(0)

    # this is a reduction operation (global max), we have to serialize it
    # TODO: a good unittest and a better implementation from Quadrants for this kind of reduction
    qd.loop_config(serialize=True)
    for i_b in range(_B):
        n_eqs = constraint_state.qd_n_equalities[i_b]
        if n_eqs > n_eqs_max:
            n_eqs_max = n_eqs

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        i_c_start = gs.qd_int(0)
        i_e_start = gs.qd_int(0)
        if qd.static(is_padded):
            i_e_start = i_b * n_eqs_max
        else:
            for j_b in range(i_b):
                i_e_start = i_e_start + constraint_state.qd_n_equalities[j_b]

        for i_e_ in range(constraint_state.qd_n_equalities[i_b]):
            i_e = i_e_start + i_e_

            iout[i_e, 0] = dyn_info.equalities.eq_type[i_e_, i_b]
            iout[i_e, 1] = dyn_info.equalities.eq_obj1id[i_e_, i_b]
            iout[i_e, 2] = dyn_info.equalities.eq_obj2id[i_e_, i_b]

            if dyn_info.equalities.eq_type[i_e_, i_b] == gs.EQUALITY_TYPE.CONNECT:
                for i_c_ in qd.static(range(3)):
                    i_c = i_c_start + i_c_
                    fout[i_e, i_c_] = constraint_state.efc_force[i_c, i_b]
                i_c_start = i_c_start + 3
            elif dyn_info.equalities.eq_type[i_e_, i_b] == gs.EQUALITY_TYPE.WELD:
                for i_c_ in qd.static(range(6)):
                    i_c = i_c_start + i_c_
                    fout[i_e, i_c_] = constraint_state.efc_force[i_c, i_b]
                i_c_start = i_c_start + 6
            elif dyn_info.equalities.eq_type[i_e_, i_b] == gs.EQUALITY_TYPE.JOINT:
                fout[i_e, 0] = constraint_state.efc_force[i_c_start, i_b]
                i_c_start = i_c_start + 1


# =====================================================================================================================
# =================================================== Problem Setup ===================================================
# =====================================================================================================================

# ====================================== Reset and Clear Constraint Solver State ======================================


@qd.kernel(fastcache=True)
def constraint_solver_kernel_reset(
    envs_idx: qd.types.ndarray(), constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    n_dofs = constraint_state.qacc_ws.shape[0]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        constraint_state.is_warmstart[i_b] = False
        for i_d in range(n_dofs):
            constraint_state.qacc_ws[i_d, i_b] = 0.0


@qd.func
def func_clear_constraint_at_env(
    i_b,
    n_dofs,
    len_constraints,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    constraint_state.n_constraints[i_b] = 0
    constraint_state.n_constraints_equality[i_b] = 0
    constraint_state.n_constraints_frictionloss[i_b] = 0
    constraint_state.n_constraints_cone[i_b] = 0
    constraint_state.qd_n_equalities[i_b] = rigid_info.n_equalities[None]
    for i_d, i_c in qd.ndrange(n_dofs, len_constraints):
        constraint_state.jac[i_c, i_d, i_b] = 0.0
    for i_c in range(len_constraints):
        constraint_state.jac_n_dofs[i_c, i_b] = 0


@qd.kernel(fastcache=True)
def constraint_solver_kernel_clear(
    envs_idx: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_dofs = constraint_state.qacc_ws.shape[0]
    len_constraints = constraint_state.jac.shape[0]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        func_clear_constraint_at_env(i_b, n_dofs, len_constraints, constraint_state, rigid_info, rigid_config)


@qd.kernel(fastcache=True)
def constraint_solver_kernel_masked_clear(
    envs_mask: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_dofs = constraint_state.qacc_ws.shape[0]
    len_constraints = constraint_state.jac.shape[0]

    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_clear_constraint_at_env(i_b, n_dofs, len_constraints, constraint_state, rigid_info, rigid_config)


# ========================================= Register Pre-Defined Constraints ==========================================


@qd.func
def _func_contact_row_direction(
    i_friction,
    normal,
    d1,
    d2,
    friction,
    friction_torsional,
    friction_rolling,
    rigid_config: qd.template(),
):
    """Direction of collision row i_friction, as a translational part n and an angular part n_ang.

    Elliptic rows [normal, t1, t2(, spin)(, roll1, roll2)] follow the contact frame unmixed (the cone couples them in
    the solver; the spin and rolling rows' friction strength lives in their regularization, see the diag computation
    in _add_friction_constraint). Pyramidal rows are 2 opposing friction-mixed edges per tangent axis, then the
    torsional and rolling pairs mixing the spin and tangent axes through the angular jacobian.
    """
    n = -normal
    n_ang = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(rigid_config.enable_elliptic_friction):
        if i_friction == 1:
            n = d1
        elif i_friction == 2:
            n = d2
        if qd.static(rigid_config.enable_torsional_friction):
            # The spin axis opposes the contact normal like the normal row. A zero coefficient must zero the whole
            # row: the cone never bounds a zero-weighted axis, so its regularization alone would leak a spurious
            # viscous torque.
            if i_friction == 3:
                n = qd.Vector.zero(gs.qd_float, 3)
                if friction_torsional > 0.0:
                    n_ang = -normal
        if qd.static(rigid_config.enable_rolling_friction):
            # The rolling axes follow the tangent rows' frame; a zero coefficient zeroes them like the spin row.
            if i_friction >= 4:
                n = qd.Vector.zero(gs.qd_float, 3)
                if friction_rolling > 0.0:
                    n_ang = d1 if i_friction == 4 else d2
    else:
        d = (2 * (i_friction % 2) - 1) * (d1 if i_friction < 2 else d2)
        n = d * friction - normal
        if qd.static(rigid_config.enable_torsional_friction):
            # A zero coefficient zeroes the pair (aref alongside, see _add_friction_constraint): the rows would
            # otherwise degenerate to two extra pure-normal rows that stiffen the normal response and report phantom
            # contact force.
            if i_friction >= 4 and i_friction < 6:
                n = qd.Vector.zero(gs.qd_float, 3)
                if friction_torsional > 0.0:
                    n = -normal
                    n_ang = ((2 * (i_friction % 2) - 1) * friction_torsional) * normal
        if qd.static(rigid_config.enable_rolling_friction):
            # The rolling pairs mix the tangent axes; a zero coefficient zeroes them like the torsional pair.
            if i_friction >= 6:
                n = qd.Vector.zero(gs.qd_float, 3)
                if friction_rolling > 0.0:
                    n = -normal
                    n_ang = ((2 * (i_friction % 2) - 1) * friction_rolling) * (d1 if i_friction < 8 else d2)
    return n, n_ang


@qd.func
def _is_contact_inert(
    link_a,
    link_b,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
) -> bool:
    """Whether a contact carries no constraint because neither endpoint is an awake dynamic body.

    A sleeper struck by an awake body is revived before the constraints are assembled
    (kernel_wake_up_entities_on_new_contact), so only hibernated-fixed pairs reach this state.
    """
    link_a_maybe_batch = [link_a, i_b] if qd.static(rigid_config.batch_links_info) else link_a
    link_b_maybe_batch = [link_b, i_b] if qd.static(rigid_config.batch_links_info) else link_b
    is_a_awake = not (dyn_info.links.is_fixed[link_a_maybe_batch] or dyn_state.links.is_hibernated[link_a, i_b])
    is_b_awake = link_b >= 0 and not (
        dyn_info.links.is_fixed[link_b_maybe_batch] or dyn_state.links.is_hibernated[link_b, i_b]
    )
    return not is_a_awake and not is_b_awake


@qd.func
def _clear_inert_collision_row(n_con, i_b, constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Write an inert (force-free) collision row in slot n_con.

    The slots are reused by index across steps, so a contact that carries no constraint must actively clear its
    slots and mark them inert: leaving the stale jacobian of a prior step (when those dofs were awake and in contact)
    would leak that contact force into the qfrc_constraint of a since-woken body that now shares the slot. The dof
    support is emptied too, so every sparse consumer (jv products, island resolve, noslip) skips the row.
    """
    if qd.static(rigid_config.sparse_solve):
        for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
            i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
    else:
        for i_d in range(constraint_state.jac.shape[1]):
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
    constraint_state.jac_n_dofs[n_con, i_b] = 0
    constraint_state.diag[n_con, i_b] = gs.qd_float(1.0)
    constraint_state.aref[n_con, i_b] = gs.qd_float(0.0)
    # The elliptic cone reads efc_D as con_mu = friction * sqrt(d0 / d1). A zero would give sqrt(0 / 0) = NaN that the
    # cleared jacobian cannot mask (0 * NaN = NaN), poisoning the solve. A finite efc_D = 1 / diag keeps con_mu finite,
    # so the zero residuals classify this inert row as inactive. The pyramidal path is unaffected by efc_D once its
    # jacobian is zero, so it keeps 0.
    if qd.static(rigid_config.enable_elliptic_friction):
        constraint_state.efc_D[n_con, i_b] = 1.0
    else:
        constraint_state.efc_D[n_con, i_b] = 0.0


@qd.func
def _add_friction_constraint(
    i_b,
    i_col_,
    i_friction,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Add one collision row to the constraint Jacobian and write its matching diag/aref/efc_D scalars.

    Pyramidal: one of the rows_per_contact friction-basis rows. Elliptic: one cone row (normal at i_friction 0),
    coupled by the Newton solver into a single second-order friction cone. Torsional friction appends the spin axis
    to either basis.
    """
    EPS = rigid_info.EPS[None]
    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]

    collision_con_start = constraint_state.n_constraints[i_b]

    i_col = collider_state.contact_sort_idx[i_col_, i_b]
    contact_data_link_a = collider_state.contact_data.link_a[i_col, i_b]
    contact_data_link_b = collider_state.contact_data.link_b[i_col, i_b]

    contact_data_pos = collider_state.contact_data.pos[i_col, i_b]
    contact_data_normal = collider_state.contact_data.normal[i_col, i_b]
    contact_data_friction = collider_state.contact_data.friction[i_col, i_b]
    contact_data_sol_params = collider_state.contact_data.sol_params[i_col, i_b]
    contact_data_penetration = collider_state.contact_data.penetration[i_col, i_b]

    link_a = contact_data_link_a
    link_b = contact_data_link_b
    link_a_maybe_batch = [link_a, i_b] if qd.static(rigid_config.batch_links_info) else link_a
    link_b_maybe_batch = [link_b, i_b] if qd.static(rigid_config.batch_links_info) else link_b

    # One mark per contact: the driver launches one thread per friction row
    if i_friction == 0:
        dyn_state.links.is_constrained[link_a, i_b] = True
        if link_b > -1:
            dyn_state.links.is_constrained[link_b, i_b] = True

    d1, d2 = gu.qd_orthogonals(contact_data_normal)

    invweight = dyn_info.links.invweight[link_a_maybe_batch][0]
    if link_b > -1:
        invweight = invweight + dyn_info.links.invweight[link_b_maybe_batch][0]

    contact_data_friction_torsional = gs.qd_float(0.0)
    if qd.static(rigid_config.enable_torsional_friction):
        contact_data_friction_torsional = collider_state.contact_data.friction_torsional[i_col, i_b]
    contact_data_friction_rolling = gs.qd_float(0.0)
    if qd.static(rigid_config.enable_rolling_friction):
        contact_data_friction_rolling = collider_state.contact_data.friction_rolling[i_col, i_b]
    n, n_ang = _func_contact_row_direction(
        i_friction,
        contact_data_normal,
        d1,
        d2,
        contact_data_friction,
        contact_data_friction_torsional,
        contact_data_friction_rolling,
        rigid_config,
    )

    rows_per_contact = qd.static(rigid_config.rows_per_contact)
    n_con = collision_con_start + i_col_ * rows_per_contact + i_friction
    if qd.static(rigid_config.sparse_solve):
        for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
            i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
    else:
        for i_d in range(n_dofs):
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

    same_root = (
        link_b > -1 and dyn_info.links.root_idx[link_a_maybe_batch] == dyn_info.links.root_idx[link_b_maybe_batch]
    )
    con_n_dofs = 0
    jac_qvel = gs.qd_float(0.0)
    for i_ab in range(2):
        sign = gs.qd_float(-1.0)
        link = link_a
        if i_ab == 1:
            sign = gs.qd_float(1.0)
            link = link_b

        while link > -1:
            link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link

            # reverse order to make sure dofs in each row of self.jac_dofs_idx are strictly descending
            for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_

                cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                cdot_vel = dyn_state.dofs.cdof_vel[i_d, i_b]

                t_quat = gu.qd_identity_quat()
                t_pos = contact_data_pos - dyn_state.links.root_COM[link, i_b]
                _, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                diff = sign * vel
                jac = diff @ n
                if qd.static(rigid_config.enable_torsional_friction):
                    # Unconditional fma even though n_ang is zero on the tangential rows: a per-row gate would
                    # diverge the per-friction kernel's warps, whose adjacent lanes hold different rows.
                    jac = jac + (sign * cdof_ang) @ n_ang
                jac_qvel = jac_qvel + jac * dyn_state.dofs.vel[i_d, i_b]
                constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                con_n_dofs = _append_relevant_dof(
                    n_con, i_d, i_b, con_n_dofs, i_ab == 1 and same_root, constraint_state
                )

            link = dyn_info.links.parent_idx[link_maybe_batch]

    constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
    _sort_relevant_dofs_descending(n_con, i_b, con_n_dofs, constraint_state, rigid_config)

    diag = gs.qd_float(0.0)
    aref = gs.qd_float(0.0)
    if qd.static(rigid_config.enable_elliptic_friction):
        # Friction rows carry no positional error (pure damping reference); the normal row references the penetration
        # depth. The impedance is shared (it depends only on penetration), and the friction rows are impratio times
        # stiffer than the normal row (R_t = R_n / impratio), matching MuJoCo's elliptic cone.
        pos_ref = -contact_data_penetration if i_friction == 0 else 0.0
        imp, aref = gu.imp_aref(contact_data_sol_params, -contact_data_penetration, jac_qvel, pos_ref)
        diag = invweight * (1 - imp) / imp
        if i_friction > 0:
            diag = diag / rigid_info.impratio[None]
        # The cone solver reads the contact friction coefficient off the head (normal) row, and the torsional and
        # rolling ones off their own rows.
        efc_frictionloss = contact_data_friction if i_friction == 0 else gs.qd_float(0.0)
        if qd.static(rigid_config.enable_torsional_friction):
            # The mu ratio lives in the spin row's regularization (jacobian unscaled), matching MuJoCo's elliptic
            # cone: it is what bounds the torsional torque at friction_torsional times the normal force once every
            # axis shares one ellipsoid. The 'signorini' resolution bounds each block by its own coefficient
            # explicitly, so rescaling there would only distort the row's stick stiffness. A zero coefficient keeps
            # the shared regularization: the row is zeroed at the source, so the value just has to stay finite.
            if i_friction == 3:
                if qd.static(not rigid_config.enable_signorini_contact):
                    if contact_data_friction_torsional > 0.0:
                        diag = diag * contact_data_friction**2 / contact_data_friction_torsional**2
                efc_frictionloss = contact_data_friction_torsional
        if qd.static(rigid_config.enable_rolling_friction):
            # The rolling rows carry the mu ratio in their regularization like the spin row.
            if i_friction >= 4:
                if qd.static(not rigid_config.enable_signorini_contact):
                    if contact_data_friction_rolling > 0.0:
                        diag = diag * contact_data_friction**2 / contact_data_friction_rolling**2
                efc_frictionloss = contact_data_friction_rolling
        constraint_state.efc_frictionloss[n_con, i_b] = efc_frictionloss
    else:
        imp, aref = gu.imp_aref(contact_data_sol_params, -contact_data_penetration, jac_qvel, -contact_data_penetration)
        # MuJoCo's regularized pyramid impedance: impratio shrinks the effective cone coefficient (mu_reg^2 = mu^2 /
        # impratio), stiffening the friction-mixed rows. Because every pyramid row mixes the normal direction, a high
        # ratio also stiffens the normal response - the reason to raise impratio with the elliptic cone instead.
        friction_sq_reg = contact_data_friction * contact_data_friction / rigid_info.impratio[None]
        diag = invweight + friction_sq_reg * invweight
        diag = diag * 2 * friction_sq_reg * (1 - imp) / imp
        if qd.static(rigid_config.enable_torsional_friction):
            # A zeroed torsional pair (see _func_contact_row_direction) must also drop the positional reference:
            # with it, the zero-jacobian rows would carry a phantom force that pollutes the reported contact force.
            if i_friction >= 4 and i_friction < 6 and contact_data_friction_torsional <= 0.0:
                aref = gs.qd_float(0.0)
        if qd.static(rigid_config.enable_rolling_friction):
            # A zeroed rolling pair drops its positional reference like the torsional pair.
            if i_friction >= 6 and contact_data_friction_rolling <= 0.0:
                aref = gs.qd_float(0.0)
    diag = qd.max(diag, EPS)
    constraint_state.diag[n_con, i_b] = diag
    constraint_state.aref[n_con, i_b] = aref
    constraint_state.efc_D[n_con, i_b] = 1 / diag


@qd.func
def _add_collision_constraints_per_friction(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Build all collision-contact constraints with one GPU thread per friction-basis constraint.

    Per-friction threading: rows_per_contact times more threads than the legacy path; adjacent lanes vary the
    friction slot i_col_ * rows_per_contact + i_friction so within a warp adjacent threads write adjacent n_con
    values. Under the flipped jac layout (_B, n_dofs, n_constraints), n_con is stride-1, so jac writes coalesce.
    """
    _B = dyn_state.dofs.ctrl_mode.shape[1]
    max_candidate_contacts = collider_state.contact_data.link_a.shape[0]
    rows_per_contact = qd.static(rigid_config.rows_per_contact)

    qd.loop_config(name="add_collision_constraints", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for flat_idx in range(_B * max_candidate_contacts * rows_per_contact):
        slot = flat_idx % (max_candidate_contacts * rows_per_contact)
        i_b = flat_idx // (max_candidate_contacts * rows_per_contact)
        i_col_ = slot // rows_per_contact
        i_friction = slot % rows_per_contact
        if i_col_ < collider_state.n_contacts[i_b]:
            is_inert = False
            if qd.static(rigid_config.use_hibernation):
                i_col = collider_state.contact_sort_idx[i_col_, i_b]
                link_a = collider_state.contact_data.link_a[i_col, i_b]
                link_b = collider_state.contact_data.link_b[i_col, i_b]
                is_inert = _is_contact_inert(link_a, link_b, i_b, dyn_state, dyn_info, rigid_config)
            if is_inert:
                n_con = constraint_state.n_constraints[i_b] + i_col_ * rows_per_contact + i_friction
                _clear_inert_collision_row(n_con, i_b, constraint_state, rigid_config)
            else:
                _add_friction_constraint(
                    i_b,
                    i_col_,
                    i_friction,
                    dyn_state,
                    collider_state,
                    constraint_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                )


@qd.func
def _add_collision_constraints_per_contact(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Build all collision-contact constraints with one GPU thread per contact."""
    EPS = rigid_info.EPS[None]
    _B = dyn_state.dofs.ctrl_mode.shape[1]
    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]
    max_candidate_contacts = collider_state.contact_data.link_a.shape[0]
    rows_per_contact = qd.static(rigid_config.rows_per_contact)

    # Iteration order follows the jac layout: batch-outer keeps every write within one env's batch-first block, while
    # the batch-inner order keeps consecutive GPU threads on consecutive envs (coalesced batch-last).
    qd.loop_config(name="add_collision_constraints", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_col_, i_b in qd.ndrange(
        max_candidate_contacts, _B, axes=qd.static((1, 0) if rigid_config.constraint_layout_batch_first else None)
    ):
        if i_col_ < collider_state.n_contacts[i_b]:
            collision_con_start = constraint_state.n_constraints[i_b]

            i_col = collider_state.contact_sort_idx[i_col_, i_b]
            contact_data_link_a = collider_state.contact_data.link_a[i_col, i_b]
            contact_data_link_b = collider_state.contact_data.link_b[i_col, i_b]

            contact_data_pos = collider_state.contact_data.pos[i_col, i_b]
            contact_data_normal = collider_state.contact_data.normal[i_col, i_b]
            contact_data_friction = collider_state.contact_data.friction[i_col, i_b]
            contact_data_sol_params = collider_state.contact_data.sol_params[i_col, i_b]
            contact_data_penetration = collider_state.contact_data.penetration[i_col, i_b]

            link_a = contact_data_link_a
            link_b = contact_data_link_b
            link_a_maybe_batch = [link_a, i_b] if qd.static(rigid_config.batch_links_info) else link_a
            link_b_maybe_batch = [link_b, i_b] if qd.static(rigid_config.batch_links_info) else link_b

            if qd.static(rigid_config.use_hibernation):
                if _is_contact_inert(link_a, link_b, i_b, dyn_state, dyn_info, rigid_config):
                    for i_friction in range(rows_per_contact):
                        n_con = collision_con_start + i_col_ * rows_per_contact + i_friction
                        _clear_inert_collision_row(n_con, i_b, constraint_state, rigid_config)
                    continue

            dyn_state.links.is_constrained[link_a, i_b] = True
            if link_b > -1:
                dyn_state.links.is_constrained[link_b, i_b] = True

            # FIXME: The reference engine anchors the tangent frame of a plane-capsule contact to the capsule axis,
            # while this frame comes from the normal alone, so the friction rows of plane-capsule pairs do not match
            # it. Anchoring the frame would make the rows depend on the capsule quaternion, whose adjoint the manual
            # backward pass (kernel_manual_add_collision_constraints_bw) must then derive by hand.
            d1, d2 = gu.qd_orthogonals(contact_data_normal)

            invweight = dyn_info.links.invweight[link_a_maybe_batch][0]
            if link_b > -1:
                invweight = invweight + dyn_info.links.invweight[link_b_maybe_batch][0]

            contact_data_friction_torsional = gs.qd_float(0.0)
            if qd.static(rigid_config.enable_torsional_friction):
                contact_data_friction_torsional = collider_state.contact_data.friction_torsional[i_col, i_b]
            contact_data_friction_rolling = gs.qd_float(0.0)
            if qd.static(rigid_config.enable_rolling_friction):
                contact_data_friction_rolling = collider_state.contact_data.friction_rolling[i_col, i_b]

            for i_friction in range(rows_per_contact):
                n, n_ang = _func_contact_row_direction(
                    i_friction,
                    contact_data_normal,
                    d1,
                    d2,
                    contact_data_friction,
                    contact_data_friction_torsional,
                    contact_data_friction_rolling,
                    rigid_config,
                )

                n_con = collision_con_start + i_col_ * rows_per_contact + i_friction
                if qd.static(rigid_config.sparse_solve):
                    for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                        i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
                        constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
                else:
                    for i_d in range(n_dofs):
                        constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

                same_root = (
                    link_b > -1
                    and dyn_info.links.root_idx[link_a_maybe_batch] == dyn_info.links.root_idx[link_b_maybe_batch]
                )
                con_n_dofs = 0
                jac_qvel = gs.qd_float(0.0)
                for i_ab in range(2):
                    sign = gs.qd_float(-1.0)
                    link = link_a
                    if i_ab == 1:
                        sign = gs.qd_float(1.0)
                        link = link_b

                    while link > -1:
                        link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link

                        # reverse order to make sure dofs in each row of self.jac_dofs_idx are strictly descending
                        for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                            i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_

                            cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                            cdot_vel = dyn_state.dofs.cdof_vel[i_d, i_b]

                            t_quat = gu.qd_identity_quat()
                            t_pos = contact_data_pos - dyn_state.links.root_COM[link, i_b]
                            _, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                            diff = sign * vel
                            jac = diff @ n
                            if qd.static(rigid_config.enable_torsional_friction):
                                # Unconditional fma on zero n_ang rows: see _add_friction_constraint.
                                jac = jac + (sign * cdof_ang) @ n_ang
                            jac_qvel = jac_qvel + jac * dyn_state.dofs.vel[i_d, i_b]
                            constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                            con_n_dofs = _append_relevant_dof(
                                n_con, i_d, i_b, con_n_dofs, i_ab == 1 and same_root, constraint_state
                            )

                        link = dyn_info.links.parent_idx[link_maybe_batch]

                constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
                _sort_relevant_dofs_descending(n_con, i_b, con_n_dofs, constraint_state, rigid_config)

                diag = gs.qd_float(0.0)
                aref = gs.qd_float(0.0)
                if qd.static(rigid_config.enable_elliptic_friction):
                    # Friction rows carry no positional error (pure damping reference) and are impratio times stiffer
                    # than the normal row; the head (normal) row stores the contact friction for the cone solver.
                    pos_ref = -contact_data_penetration if i_friction == 0 else 0.0
                    imp, aref = gu.imp_aref(contact_data_sol_params, -contact_data_penetration, jac_qvel, pos_ref)
                    diag = invweight * (1 - imp) / imp
                    if i_friction > 0:
                        diag = diag / rigid_info.impratio[None]
                    efc_frictionloss = contact_data_friction if i_friction == 0 else gs.qd_float(0.0)
                    if qd.static(rigid_config.enable_torsional_friction):
                        # Spin-row regularization and torsional coefficient storage: see _add_friction_constraint.
                        if i_friction == 3:
                            if qd.static(not rigid_config.enable_signorini_contact):
                                if contact_data_friction_torsional > 0.0:
                                    diag = diag * contact_data_friction**2 / contact_data_friction_torsional**2
                            efc_frictionloss = contact_data_friction_torsional
                    if qd.static(rigid_config.enable_rolling_friction):
                        # Rolling-row regularization and coefficient storage: see _add_friction_constraint.
                        if i_friction >= 4:
                            if qd.static(not rigid_config.enable_signorini_contact):
                                if contact_data_friction_rolling > 0.0:
                                    diag = diag * contact_data_friction**2 / contact_data_friction_rolling**2
                            efc_frictionloss = contact_data_friction_rolling
                    constraint_state.efc_frictionloss[n_con, i_b] = efc_frictionloss
                else:
                    imp, aref = gu.imp_aref(
                        contact_data_sol_params, -contact_data_penetration, jac_qvel, -contact_data_penetration
                    )
                    # MuJoCo's regularized pyramid impedance: impratio shrinks the effective cone coefficient
                    # (mu_reg^2 = mu^2 / impratio), stiffening the friction-mixed rows.
                    friction_sq_reg = contact_data_friction * contact_data_friction / rigid_info.impratio[None]
                    diag = invweight + friction_sq_reg * invweight
                    diag = diag * 2 * friction_sq_reg * (1 - imp) / imp
                    if qd.static(rigid_config.enable_torsional_friction):
                        # A zeroed torsional pair drops its positional reference: see _add_friction_constraint.
                        if i_friction >= 4 and i_friction < 6 and contact_data_friction_torsional <= 0.0:
                            aref = gs.qd_float(0.0)
                    if qd.static(rigid_config.enable_rolling_friction):
                        # A zeroed rolling pair drops its positional reference: see _add_friction_constraint.
                        if i_friction >= 6 and contact_data_friction_rolling <= 0.0:
                            aref = gs.qd_float(0.0)
                diag = qd.max(diag, EPS)
                constraint_state.diag[n_con, i_b] = diag
                constraint_state.aref[n_con, i_b] = aref
                constraint_state.efc_D[n_con, i_b] = 1 / diag


@qd.func
def add_collision_constraints(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    _B = dyn_state.dofs.ctrl_mode.shape[1]

    if qd.static(rigid_config.enable_cooperative_constraint_kernels):
        _add_collision_constraints_per_friction(
            dyn_state, collider_state, constraint_state, dyn_info, rigid_info, rigid_config
        )
    else:
        _add_collision_constraints_per_contact(
            dyn_state, collider_state, constraint_state, dyn_info, rigid_info, rigid_config
        )

    rows_per_contact = qd.static(rigid_config.rows_per_contact)
    qd.loop_config(name="add_collision_count", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        n_collision_rows = collider_state.n_contacts[i_b] * rows_per_contact
        constraint_state.n_constraints[i_b] = constraint_state.n_constraints[i_b] + n_collision_rows
        # The elliptic cone rows are the whole collision segment (rows_per_contact contiguous per contact); joint
        # limits follow.
        if qd.static(rigid_config.enable_elliptic_friction):
            constraint_state.n_constraints_cone[i_b] = n_collision_rows
        else:
            constraint_state.n_constraints_cone[i_b] = 0


@qd.func
def func_equality_jdotv(
    i_b,
    link,
    anchor_pos,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Jacobian-derivative bias Jdot @ qvel of a link-attached anchor point.

    Contracting the per-dof columns of the anchor Jacobian time derivative with qvel collapses to link-level
    quantities: the ancestor-chain sum of vel * cdofd is the link's velocity-product spatial acceleration, and the
    link's angular velocity cd_ang contracts the per-column cdof correction terms. The per-column quaternion-dof
    correction (full-body velocity in place of the accumulated prefix) contracts with qvel to a self-cross that
    vanishes, so the stored cdofd is exact here. Returns (jdotv, cddb_ang): the linear Jdot @ qvel at the anchor and
    the angular Jdot @ qvel of the chain (the weld rotation rows need the latter). Both are zero for the world
    (link == -1).
    """
    jdotv = qd.Vector.zero(gs.qd_float, 3)
    cddb_ang = qd.Vector.zero(gs.qd_float, 3)
    if link > -1:
        cddb_vel = qd.Vector.zero(gs.qd_float, 3)
        i_l = link
        while i_l > -1:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                cddb_ang = cddb_ang + dyn_state.dofs.cdofd_ang[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
                cddb_vel = cddb_vel + dyn_state.dofs.cdofd_vel[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
            i_l = dyn_info.links.parent_idx[I_l]
        offset = anchor_pos - dyn_state.links.root_COM[link, i_b]
        pvel = dyn_state.links.cd_vel[link, i_b] + dyn_state.links.cd_ang[link, i_b].cross(offset)
        jdotv = cddb_vel + cddb_ang.cross(offset) + dyn_state.links.cd_ang[link, i_b].cross(pvel)
    return jdotv, cddb_ang


@qd.func
def func_equality_connect(
    i_b,
    i_e,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    EPS = rigid_info.EPS[None]

    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]

    link1_idx = dyn_info.equalities.eq_obj1id[i_e, i_b]
    link2_idx = dyn_info.equalities.eq_obj2id[i_e, i_b]
    link_a_maybe_batch = [link1_idx, i_b] if qd.static(rigid_config.batch_links_info) else link1_idx
    link_b_maybe_batch = [link2_idx, i_b] if qd.static(rigid_config.batch_links_info) else link2_idx
    anchor1_pos = gs.qd_vec3(
        [
            dyn_info.equalities.eq_data[i_e, i_b][0],
            dyn_info.equalities.eq_data[i_e, i_b][1],
            dyn_info.equalities.eq_data[i_e, i_b][2],
        ]
    )
    anchor2_pos = gs.qd_vec3(
        [
            dyn_info.equalities.eq_data[i_e, i_b][3],
            dyn_info.equalities.eq_data[i_e, i_b][4],
            dyn_info.equalities.eq_data[i_e, i_b][5],
        ]
    )
    sol_params = dyn_info.equalities.sol_params[i_e, i_b]

    # Transform anchor positions to global coordinates
    global_anchor1 = gu.qd_transform_by_trans_quat(
        pos=anchor1_pos, trans=dyn_state.links.pos[link1_idx, i_b], quat=dyn_state.links.quat[link1_idx, i_b]
    )
    global_anchor2 = gu.qd_transform_by_trans_quat(
        pos=anchor2_pos, trans=dyn_state.links.pos[link2_idx, i_b], quat=dyn_state.links.quat[link2_idx, i_b]
    )

    invweight = dyn_info.links.invweight[link_a_maybe_batch][0] + dyn_info.links.invweight[link_b_maybe_batch][0]

    # The reference acceleration must track the true relative anchor acceleration, so the centripetal/Coriolis bias
    # Jdot @ qvel is subtracted from aref below; without it closed chains (e.g. four-bar linkages) accumulate
    # constraint violation.
    jdotv1, _cddb1_ang = func_equality_jdotv(i_b, link1_idx, global_anchor1, dyn_state, dyn_info, rigid_config)
    jdotv2, _cddb2_ang = func_equality_jdotv(i_b, link2_idx, global_anchor2, dyn_state, dyn_info, rigid_config)
    jdotv = jdotv1 - jdotv2

    dyn_state.links.is_constrained[link1_idx, i_b] = True
    if link2_idx > -1:
        dyn_state.links.is_constrained[link2_idx, i_b] = True

    for i_3 in range(3):
        n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
        qd.atomic_add(constraint_state.n_constraints_equality[i_b], 1)
        con_n_dofs = 0

        if qd.static(rigid_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
        else:
            for i_d in range(n_dofs):
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

        same_root = (
            link2_idx > -1
            and dyn_info.links.root_idx[link_a_maybe_batch] == dyn_info.links.root_idx[link_b_maybe_batch]
        )
        jac_qvel = gs.qd_float(0.0)
        for i_ab in range(2):
            sign = gs.qd_float(1.0)
            link = link1_idx
            pos = global_anchor1
            if i_ab == 1:
                sign = gs.qd_float(-1.0)
                link = link2_idx
                pos = global_anchor2

            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link

                for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                    i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_

                    cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                    cdot_vel = dyn_state.dofs.cdof_vel[i_d, i_b]

                    t_quat = gu.qd_identity_quat()
                    t_pos = pos - dyn_state.links.root_COM[link, i_b]
                    ang, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                    diff = sign * vel
                    jac = diff[i_3]
                    jac_qvel = jac_qvel + jac * dyn_state.dofs.vel[i_d, i_b]
                    constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                    con_n_dofs = _append_relevant_dof(
                        n_con, i_d, i_b, con_n_dofs, i_ab == 1 and same_root, constraint_state
                    )

                link = dyn_info.links.parent_idx[link_maybe_batch]

        constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
        # Sort needed: DOFs from two entities are only descending within each
        # entity. Incremental Cholesky requires globally descending order.
        _sort_relevant_dofs_descending(n_con, i_b, con_n_dofs, constraint_state, rigid_config)

        pos_diff = global_anchor1 - global_anchor2
        penetration = pos_diff.norm()

        imp, aref = gu.imp_aref(sol_params, -penetration, jac_qvel, pos_diff[i_3])

        diag = qd.max(invweight * (1.0 - imp) / imp, EPS)

        constraint_state.diag[n_con, i_b] = diag
        constraint_state.aref[n_con, i_b] = aref - jdotv[i_3]
        constraint_state.efc_D[n_con, i_b] = 1.0 / diag


@qd.func
def func_equality_joint(
    i_b,
    i_e,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    EPS = rigid_info.EPS[None]

    n_dofs = constraint_state.jac.shape[1]

    sol_params = dyn_info.equalities.sol_params[i_e, i_b]

    I_joint1 = (
        [dyn_info.equalities.eq_obj1id[i_e, i_b], i_b]
        if qd.static(rigid_config.batch_joints_info)
        else dyn_info.equalities.eq_obj1id[i_e, i_b]
    )
    I_joint2 = (
        [dyn_info.equalities.eq_obj2id[i_e, i_b], i_b]
        if qd.static(rigid_config.batch_joints_info)
        else dyn_info.equalities.eq_obj2id[i_e, i_b]
    )
    i_qpos1 = dyn_info.joints.q_start[I_joint1]
    i_qpos2 = dyn_info.joints.q_start[I_joint2]
    i_dof1 = dyn_info.joints.dof_start[I_joint1]
    i_dof2 = dyn_info.joints.dof_start[I_joint2]
    I_dof1 = [i_dof1, i_b] if qd.static(rigid_config.batch_dofs_info) else i_dof1
    I_dof2 = [i_dof2, i_b] if qd.static(rigid_config.batch_dofs_info) else i_dof2

    n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
    qd.atomic_add(constraint_state.n_constraints_equality[i_b], 1)

    if qd.static(rigid_config.sparse_solve):
        for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
            i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
    else:
        for i_d in range(n_dofs):
            constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

    pos1 = rigid_info.qpos[i_qpos1, i_b]
    pos2 = rigid_info.qpos[i_qpos2, i_b]
    ref1 = rigid_info.qpos0[i_qpos1, i_b]
    ref2 = rigid_info.qpos0[i_qpos2, i_b]

    # TODO: zero objid2
    diff = pos2 - ref2
    pos = pos1 - ref1
    deriv = gs.qd_float(0.0)

    # y - y0 = a0 + a1 * (x-x0) + a2 * (x-x0)^2 + a3 * (x-fx0)^3 + a4 * (x-x0)^4
    for i_5 in range(5):
        diff_power = diff**i_5
        pos = pos - diff_power * dyn_info.equalities.eq_data[i_e, i_b][i_5]
        if i_5 < 4:
            deriv = deriv + dyn_info.equalities.eq_data[i_e, i_b][i_5 + 1] * diff_power * (i_5 + 1)

    constraint_state.jac[n_con, i_dof1, i_b] = gs.qd_float(1.0)
    constraint_state.jac[n_con, i_dof2, i_b] = -deriv
    jac_qvel = (
        constraint_state.jac[n_con, i_dof1, i_b] * dyn_state.dofs.vel[i_dof1, i_b]
        + constraint_state.jac[n_con, i_dof2, i_b] * dyn_state.dofs.vel[i_dof2, i_b]
    )
    invweight = dyn_info.dofs.invweight[I_dof1] + dyn_info.dofs.invweight[I_dof2]

    imp, aref = gu.imp_aref(sol_params, -qd.abs(pos), jac_qvel, pos)

    diag = qd.max(invweight * (1.0 - imp) / imp, EPS)

    constraint_state.diag[n_con, i_b] = diag
    constraint_state.aref[n_con, i_b] = aref
    constraint_state.efc_D[n_con, i_b] = 1.0 / diag

    # Populate jac_dofs_idx for this joint-equality constraint, so the sparse-Jacobian iterations see its relevant
    # DOFs (otherwise they would see 0 and produce zero forces, leading to NaN in the solver).
    con_n_dofs = 0
    constraint_state.jac_dofs_idx[n_con, con_n_dofs, i_b] = i_dof1
    con_n_dofs += 1
    if i_dof2 != i_dof1:
        constraint_state.jac_dofs_idx[n_con, con_n_dofs, i_b] = i_dof2
        con_n_dofs += 1
    constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
    _sort_relevant_dofs_descending(n_con, i_b, con_n_dofs, constraint_state, rigid_config)


@qd.kernel(fastcache=True)
def add_equality_constraints(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    _B = dyn_state.dofs.ctrl_mode.shape[1]

    n_links = dyn_state.links.pos.shape[0]

    # Reset the per-link constraint involvement (see is_constrained in array_class.py); the equality funcs below and
    # the collision assembly mark it back.
    qd.loop_config(name="clear_links_constrained", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        dyn_state.links.is_constrained[i_l, i_b] = False

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        constraint_state.n_constraints[i_b] = 0
        constraint_state.n_constraints_equality[i_b] = 0

        for i_e in range(constraint_state.qd_n_equalities[i_b]):
            if dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.CONNECT:
                func_equality_connect(i_b, i_e, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)

            elif dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.WELD:
                func_equality_weld(i_b, i_e, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
            elif dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.JOINT:
                func_equality_joint(i_b, i_e, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)


@qd.func
def _sort_contacts_and_build_islands(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """Order the contacts of every env (see add_inequality_constraints) and build its island partition, the two per-env
    steps sharing one launch.

    Where the cooperative kernels run, a block serves each env: the lanes sort together (func_sort_contacts_coop), then
    build the partition together (func_build_islands_coop); elsewhere one thread per env does both. The order and the
    partition are the same whichever way they are built, so the constraint order the caller assembles is too.
    """
    _B = constraint_state.jac.shape[2]
    if qd.static(rigid_config.enable_cooperative_constraint_kernels):
        _K = qd.static(32)
        # Reset the per-class (env, island) work-list counters before the per-env builds append to them
        N_CLASSES = qd.static(len(array_class.island_tile_caps(rigid_config)))
        for i_class in range(N_CLASSES):
            constraint_state.island.factor_worklist_size[i_class] = 0
        qd.loop_config(name="sort_contacts_and_build_islands", block_dim=_K)
        for i_flat in range(_B * _K):
            tid = i_flat % _K
            i_b = i_flat // _K
            if qd.static(collider_static_config.spatial_sort_supported):
                func_sort_contacts_coop(i_b, tid, dyn_state, collider_state, constraint_state)
                qd.simt.block.sync()
            func_build_islands_coop(
                i_b, tid, dyn_state, collider_state, constraint_state, dyn_info, rigid_info, rigid_config
            )
            # Append this env's islands to the work-list of their size class, the smallest tile cap holding the
            # island's dofs and the last class for the islands above every cap (see island_tile_caps).
            CAPS = qd.static(array_class.island_tile_caps(rigid_config))
            region = constraint_state.island.factor_worklist_i_b.shape[0] // N_CLASSES
            i_island = tid
            while i_island < constraint_state.island.n_islands[i_b]:
                n_island_dofs = constraint_state.island.dof_slices.n[i_island, i_b]
                i_class = N_CLASSES - 1
                for k in qd.static(range(N_CLASSES - 2, -1, -1)):
                    if n_island_dofs <= CAPS[k]:
                        i_class = k
                i_slot = i_class * region + qd.atomic_add(constraint_state.island.factor_worklist_size[i_class], 1)
                constraint_state.island.factor_worklist_i_b[i_slot] = i_b
                constraint_state.island.factor_worklist_i_island[i_slot] = i_island
                i_island = i_island + _K
    else:
        qd.loop_config(
            name="sort_contacts_and_build_islands", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL)
        )
        for i_b in range(_B):
            if qd.static(collider_static_config.spatial_sort_supported):
                func_sort_contacts(
                    i_b,
                    collider_state.contact_sort_idx,
                    collider_state.n_contacts[i_b],
                    collider_state.contact_data.pos,
                    collider_state.contact_data.geom_a,
                    collider_state.contact_data.geom_b,
                    dyn_state.geoms.pos,
                    dyn_state.geoms.quat,
                )
            func_build_islands(i_b, dyn_state, collider_state, constraint_state, dyn_info, rigid_info, rigid_config)


@qd.kernel(fastcache=True)
def add_inequality_constraints(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    # Order the contacts deterministically BEFORE assembling the contact constraints below: the contact-constraint
    # index i_c follows the logical contact order (contact_sort_idx), so fixing that order here makes both the solve
    # order and get_contacts deterministic despite the racy atomic_add narrowphase layout. Done here rather than in the
    # collider (which has no notion of constraints) or in func_solve_init (too late - contacts are consumed just below).
    # The collider's compacted contact_sort_idx is sorted in place. The island partition of the env rides the same
    # launch (see _sort_contacts_and_build_islands); the constraint grouping it feeds waits for the assembled
    # constraints, in func_solve_init.
    _sort_contacts_and_build_islands(
        dyn_state, collider_state, constraint_state, dyn_info, rigid_info, rigid_config, collider_static_config
    )

    add_frictionloss_constraints(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
    if qd.static(rigid_config.enable_collision):
        add_collision_constraints(dyn_state, collider_state, constraint_state, dyn_info, rigid_info, rigid_config)
    if qd.static(rigid_config.enable_joint_limit):
        add_joint_limit_constraints(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)


@qd.func
def func_equality_weld(
    i_b,
    i_e,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    EPS = rigid_info.EPS[None]

    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]

    # Get equality info for this constraint
    link1_idx = dyn_info.equalities.eq_obj1id[i_e, i_b]
    link2_idx = dyn_info.equalities.eq_obj2id[i_e, i_b]
    link_a_maybe_batch = [link1_idx, i_b] if qd.static(rigid_config.batch_links_info) else link1_idx
    link_b_maybe_batch = [link2_idx, i_b] if qd.static(rigid_config.batch_links_info) else link2_idx

    # For weld, eq_data layout:
    # [0:3]  : anchor2 (local pos in body2)
    # [3:6]  : anchor1 (local pos in body1)
    # [6:10] : relative pose (quat) of body 2 related to body 1 to match orientations
    # [10]   : torquescale
    anchor1_pos = gs.qd_vec3(
        [
            dyn_info.equalities.eq_data[i_e, i_b][3],
            dyn_info.equalities.eq_data[i_e, i_b][4],
            dyn_info.equalities.eq_data[i_e, i_b][5],
        ]
    )
    anchor2_pos = gs.qd_vec3(
        [
            dyn_info.equalities.eq_data[i_e, i_b][0],
            dyn_info.equalities.eq_data[i_e, i_b][1],
            dyn_info.equalities.eq_data[i_e, i_b][2],
        ]
    )
    relpose = gs.qd_vec4(
        [
            dyn_info.equalities.eq_data[i_e, i_b][6],
            dyn_info.equalities.eq_data[i_e, i_b][7],
            dyn_info.equalities.eq_data[i_e, i_b][8],
            dyn_info.equalities.eq_data[i_e, i_b][9],
        ]
    )
    torquescale = dyn_info.equalities.eq_data[i_e, i_b][10]
    sol_params = dyn_info.equalities.sol_params[i_e, i_b]

    # Transform anchor positions to global coordinates
    global_anchor1 = gu.qd_transform_by_trans_quat(
        pos=anchor1_pos, trans=dyn_state.links.pos[link1_idx, i_b], quat=dyn_state.links.quat[link1_idx, i_b]
    )
    global_anchor2 = gu.qd_transform_by_trans_quat(
        pos=anchor2_pos, trans=dyn_state.links.pos[link2_idx, i_b], quat=dyn_state.links.quat[link2_idx, i_b]
    )

    pos_error = global_anchor1 - global_anchor2

    # Compute orientation error.
    # For weld: compute q = body1_quat * relpose, then error = (inv(body2_quat) * q)
    quat_body1 = dyn_state.links.quat[link1_idx, i_b]
    quat_body2 = dyn_state.links.quat[link2_idx, i_b]
    q = gu.qd_quat_mul(quat_body1, relpose)
    inv_quat_body2 = gu.qd_inv_quat(quat_body2)
    error_quat = gu.qd_quat_mul(inv_quat_body2, q)
    # Take the vector (axis) part and scale by torquescale.
    rot_error = gs.qd_vec3([error_quat[1], error_quat[2], error_quat[3]]) * torquescale

    all_error = gs.qd_vec6([pos_error[0], pos_error[1], pos_error[2], rot_error[0], rot_error[1], rot_error[2]])
    pos_imp = all_error.norm()

    # Compute inverse weight from both bodies.
    invweight = dyn_info.links.invweight[link_a_maybe_batch] + dyn_info.links.invweight[link_b_maybe_batch]

    # Centripetal/Coriolis bias of the anchor pair: see func_equality_connect for the rationale of subtracting
    # Jdot @ qvel from aref. The chains' angular parts feed the rotation rows' bias below.
    jdotv1, cddb1_ang = func_equality_jdotv(i_b, link1_idx, global_anchor1, dyn_state, dyn_info, rigid_config)
    jdotv2, cddb2_ang = func_equality_jdotv(i_b, link2_idx, global_anchor2, dyn_state, dyn_info, rigid_config)
    jdotv = jdotv1 - jdotv2

    dyn_state.links.is_constrained[link1_idx, i_b] = True
    dyn_state.links.is_constrained[link2_idx, i_b] = True

    # --- Position part (first 3 constraints) ---
    same_root = (
        link2_idx > -1 and dyn_info.links.root_idx[link_a_maybe_batch] == dyn_info.links.root_idx[link_b_maybe_batch]
    )
    for i in range(3):
        n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
        qd.atomic_add(constraint_state.n_constraints_equality[i_b], 1)
        con_n_dofs = 0

        if qd.static(rigid_config.sparse_solve):
            for i_d_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                i_d = constraint_state.jac_dofs_idx[n_con, i_d_, i_b]
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)
        else:
            for i_d in range(n_dofs):
                constraint_state.jac[n_con, i_d, i_b] = gs.qd_float(0.0)

        jac_qvel = gs.qd_float(0.0)
        for i_ab in range(2):
            sign = gs.qd_float(1.0) if i_ab == 0 else gs.qd_float(-1.0)
            link = link1_idx if i_ab == 0 else link2_idx
            pos_anchor = global_anchor1 if i_ab == 0 else global_anchor2

            # Accumulate jacobian contributions along the kinematic chain.
            # (Assuming similar structure to equality_connect.)
            while link > -1:
                link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link

                for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                    i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_
                    cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                    cdot_vel = dyn_state.dofs.cdof_vel[i_d, i_b]
                    t_pos = pos_anchor - dyn_state.links.root_COM[link, i_b]
                    # t_quat = gu.qd_identity_quat()
                    # _ang, vel = gu.qd_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)
                    vel = cdot_vel - t_pos.cross(cdof_ang)
                    diff = sign * vel
                    jac = diff[i]
                    jac_qvel = jac_qvel + jac * dyn_state.dofs.vel[i_d, i_b]
                    constraint_state.jac[n_con, i_d, i_b] = constraint_state.jac[n_con, i_d, i_b] + jac

                    con_n_dofs = _append_relevant_dof(
                        n_con, i_d, i_b, con_n_dofs, i_ab == 1 and same_root, constraint_state
                    )
                link = dyn_info.links.parent_idx[link_maybe_batch]

        constraint_state.jac_n_dofs[n_con, i_b] = con_n_dofs
        _sort_relevant_dofs_descending(n_con, i_b, con_n_dofs, constraint_state, rigid_config)

        imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel, pos_error[i])
        diag = qd.max(invweight[0] * (1 - imp) / imp, EPS)

        constraint_state.diag[n_con, i_b] = diag
        constraint_state.aref[n_con, i_b] = aref - jdotv[i]
        constraint_state.efc_D[n_con, i_b] = 1.0 / diag

    # --- Orientation part (next 3 constraints) ---
    n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 3)
    qd.atomic_add(constraint_state.n_constraints_equality[i_b], 3)
    con_n_dofs = 0
    for i_con in range(n_con, n_con + 3):
        for i_d in range(n_dofs):
            constraint_state.jac[i_con, i_d, i_b] = gs.qd_float(0.0)

    for i_ab in range(2):
        sign = gs.qd_float(1.0) if i_ab == 0 else gs.qd_float(-1.0)
        link = link1_idx if i_ab == 0 else link2_idx
        # For rotation, we use the body's orientation (here we use its quaternion)
        # and a suitable reference frame. (You may need a more detailed implementation.)
        while link > -1:
            link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link

            for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_
                jac = sign * dyn_state.dofs.cdof_ang[i_d, i_b]

                for i_con in range(n_con, n_con + 3):
                    constraint_state.jac[i_con, i_d, i_b] = constraint_state.jac[i_con, i_d, i_b] + jac[i_con - n_con]

                # The 3 orientation constraints share the same support (the DOFs along both kinematic chains); record
                # it so sparse assembly does not drop them. (The position part above does the same per constraint.)
                n_dofs_new = con_n_dofs
                for i_con in range(n_con, n_con + 3):
                    n_dofs_new = _append_relevant_dof(
                        i_con, i_d, i_b, con_n_dofs, i_ab == 1 and same_root, constraint_state
                    )
                con_n_dofs = n_dofs_new
            link = dyn_info.links.parent_idx[link_maybe_batch]

    jac_qvel = qd.Vector([0.0, 0.0, 0.0])
    for i_d in range(n_dofs):
        # quat2 = neg(q1)*(jac0-jac1)
        # quat3 = neg(q1)*(jac0-jac1)*q0*relpose
        jac_diff_r = qd.Vector(
            [
                constraint_state.jac[n_con, i_d, i_b],
                constraint_state.jac[n_con + 1, i_d, i_b],
                constraint_state.jac[n_con + 2, i_d, i_b],
            ]
        )
        quat2 = gu.qd_quat_mul_axis(inv_quat_body2, jac_diff_r)
        quat3 = gu.qd_quat_mul(quat2, q)

        for i_con in range(n_con, n_con + 3):
            constraint_state.jac[i_con, i_d, i_b] = 0.5 * quat3[i_con - n_con + 1] * torquescale
            jac_qvel[i_con - n_con] = (
                jac_qvel[i_con - n_con] + constraint_state.jac[i_con, i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
            )

    for i_con in range(n_con, n_con + 3):
        constraint_state.jac_n_dofs[i_con, i_b] = con_n_dofs
        _sort_relevant_dofs_descending(i_con, i_b, con_n_dofs, constraint_state, rigid_config)

    # Rotational Jdot @ qvel: differentiate the rotation rows 0.5 * neg(q1) * (Jr0 - Jr1) @ qvel * q0 * relpose in
    # time; the product rule yields three terms, with quaternion derivatives qdot = 0.5 * (0, omega) * quat.
    omega1 = dyn_state.links.cd_ang[link1_idx, i_b]
    omega2 = dyn_state.links.cd_ang[link2_idx, i_b]
    domega = omega1 - omega2
    qdot_body1 = 0.5 * gu.qd_quat_mul(gs.qd_vec4([0.0, omega1[0], omega1[1], omega1[2]]), quat_body1)
    qdot0r = gu.qd_quat_mul(qdot_body1, relpose)
    qdot_body2 = 0.5 * gu.qd_quat_mul(gs.qd_vec4([0.0, omega2[0], omega2[1], omega2[2]]), quat_body2)
    djrdv = cddb1_ang - cddb2_ang
    t1 = gu.qd_quat_mul(gu.qd_quat_mul_axis(gu.qd_inv_quat(qdot_body2), domega), q)
    t2 = gu.qd_quat_mul(gu.qd_quat_mul_axis(inv_quat_body2, djrdv), q)
    t3 = gu.qd_quat_mul(gu.qd_quat_mul_axis(inv_quat_body2, domega), qdot0r)

    for i_con_ in qd.static(range(3)):
        i_con = i_con_ + n_con
        imp, aref = gu.imp_aref(sol_params, -pos_imp, jac_qvel[i_con_], rot_error[i_con_])
        diag = qd.max(invweight[1] * (1.0 - imp) / imp, EPS)

        constraint_state.diag[i_con, i_b] = diag
        constraint_state.aref[i_con, i_b] = (
            aref - 0.5 * (t1[i_con_ + 1] + t2[i_con_ + 1] + t3[i_con_ + 1]) * torquescale
        )
        constraint_state.efc_D[i_con, i_b] = 1.0 / diag


@qd.func
def add_joint_limit_constraints(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    EPS = rigid_info.EPS[None]

    _B = constraint_state.jac.shape[2]
    n_links = dyn_info.links.root_idx.shape[0]
    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]

    # TODO: sparse mode
    qd.loop_config(name="add_joint_limit_constraints", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

            for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j

                if (
                    dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE
                    or dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC
                ):
                    i_q = dyn_info.joints.q_start[I_j]
                    i_d = dyn_info.joints.dof_start[I_j]
                    I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                    pos_delta_min = rigid_info.qpos[i_q, i_b] - dyn_info.dofs.limit[I_d][0]
                    pos_delta_max = dyn_info.dofs.limit[I_d][1] - rigid_info.qpos[i_q, i_b]
                    pos_delta = qd.min(pos_delta_min, pos_delta_max)

                    if pos_delta < 0:
                        jac = (pos_delta_min < pos_delta_max) * 2 - 1
                        jac_qvel = jac * dyn_state.dofs.vel[i_d, i_b]
                        imp, aref = gu.imp_aref(dyn_info.joints.sol_params[I_j], pos_delta, jac_qvel, pos_delta)
                        diag = qd.max(dyn_info.dofs.invweight[I_d] * (1 - imp) / imp, EPS)

                        n_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
                        constraint_state.diag[n_con, i_b] = diag
                        constraint_state.aref[n_con, i_b] = aref
                        constraint_state.efc_D[n_con, i_b] = 1 / diag

                        if qd.static(rigid_config.sparse_solve):
                            for i_d2_ in range(constraint_state.jac_n_dofs[n_con, i_b]):
                                i_d2 = constraint_state.jac_dofs_idx[n_con, i_d2_, i_b]
                                constraint_state.jac[n_con, i_d2, i_b] = gs.qd_float(0.0)
                        else:
                            for i_d2 in range(n_dofs):
                                constraint_state.jac[n_con, i_d2, i_b] = gs.qd_float(0.0)
                        constraint_state.jac[n_con, i_d, i_b] = jac

                        constraint_state.jac_n_dofs[n_con, i_b] = 1
                        constraint_state.jac_dofs_idx[n_con, 0, i_b] = i_d


@qd.func
def add_frictionloss_constraints(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    EPS = rigid_info.EPS[None]

    _B = constraint_state.jac.shape[2]
    n_links = dyn_info.links.root_idx.shape[0]
    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]

    # TODO: sparse mode
    # FIXME: The condition `if dofs_info.frictionloss[I_d] > EPS:` is not correctly evaluated on Apple Metal
    # if `serialize=True`...
    qd.loop_config(
        name="add_frictionloss_constraints",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL and rigid_config.backend != gs.metal),
    )
    for i_b in range(_B):
        constraint_state.n_constraints_frictionloss[i_b] = 0

        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

            for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j

                for i_d in range(dyn_info.joints.dof_start[I_j], dyn_info.joints.dof_end[I_j]):
                    I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d

                    if dyn_info.dofs.frictionloss[I_d] > EPS:
                        jac = 1.0
                        jac_qvel = jac * dyn_state.dofs.vel[i_d, i_b]
                        imp, aref = gu.imp_aref(dyn_info.joints.sol_params[I_j], 0.0, jac_qvel, 0.0)
                        diag = qd.max(dyn_info.dofs.invweight[I_d] * (1.0 - imp) / imp, EPS)

                        i_con = qd.atomic_add(constraint_state.n_constraints[i_b], 1)
                        qd.atomic_add(constraint_state.n_constraints_frictionloss[i_b], 1)

                        constraint_state.diag[i_con, i_b] = diag
                        constraint_state.aref[i_con, i_b] = aref
                        constraint_state.efc_D[i_con, i_b] = 1.0 / diag
                        constraint_state.efc_frictionloss[i_con, i_b] = dyn_info.dofs.frictionloss[I_d]
                        for i_d2 in range(n_dofs):
                            constraint_state.jac[i_con, i_d2, i_b] = gs.qd_float(0.0)
                        constraint_state.jac[i_con, i_d, i_b] = jac

                        constraint_state.jac_dofs_idx[i_con, 0, i_b] = i_d
                        constraint_state.jac_n_dofs[i_con, i_b] = 1


# ====================================== Runtime User-Specified Weld Constraints ======================================


@qd.kernel(fastcache=True)
def kernel_add_weld_constraint(
    link1_idx: qd.i32,
    link2_idx: qd.i32,
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
) -> qd.i32:
    overflow = gs.qd_bool(False)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_e = constraint_state.qd_n_equalities[i_b]
        if i_e == rigid_info.n_candidate_equalities[None]:
            overflow = True
        else:
            shared_pos = dyn_state.links.pos[link1_idx, i_b]
            pos1 = gu.qd_inv_transform_by_trans_quat(
                shared_pos, dyn_state.links.pos[link1_idx, i_b], dyn_state.links.quat[link1_idx, i_b]
            )
            pos2 = gu.qd_inv_transform_by_trans_quat(
                shared_pos, dyn_state.links.pos[link2_idx, i_b], dyn_state.links.quat[link2_idx, i_b]
            )

            dyn_info.equalities.eq_type[i_e, i_b] = gs.qd_int(gs.EQUALITY_TYPE.WELD)
            dyn_info.equalities.eq_obj1id[i_e, i_b] = link1_idx
            dyn_info.equalities.eq_obj2id[i_e, i_b] = link2_idx

            for i_3 in qd.static(range(3)):
                dyn_info.equalities.eq_data[i_e, i_b][i_3 + 3] = pos1[i_3]
                dyn_info.equalities.eq_data[i_e, i_b][i_3] = pos2[i_3]

            relpose = gu.qd_quat_mul(
                gu.qd_inv_quat(dyn_state.links.quat[link1_idx, i_b]), dyn_state.links.quat[link2_idx, i_b]
            )

            for i_4 in qd.static(range(4)):
                dyn_info.equalities.eq_data[i_e, i_b][i_4 + 6] = relpose[i_4]

            dyn_info.equalities.eq_data[i_e, i_b][10] = 1.0

            dyn_info.equalities.sol_params[i_e, i_b] = qd.Vector(
                [2 * rigid_info.substep_dt[None], 1.0, 0.9, 0.95, 0.001, 0.5, 2.0]
            )

            constraint_state.qd_n_equalities[i_b] = constraint_state.qd_n_equalities[i_b] + 1
    return overflow


@qd.kernel(fastcache=True)
def kernel_delete_weld_constraint(
    link1_idx: qd.i32,
    link2_idx: qd.i32,
    envs_idx: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_e in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
            if (
                dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.WELD
                and dyn_info.equalities.eq_obj1id[i_e, i_b] == link1_idx
                and dyn_info.equalities.eq_obj2id[i_e, i_b] == link2_idx
            ):
                if i_e < constraint_state.qd_n_equalities[i_b] - 1:
                    # Swap-remove must move the whole constraint record, not just its type,
                    # otherwise the surviving slot keeps the deleted constraint's links/data.
                    i_last = constraint_state.qd_n_equalities[i_b] - 1
                    dyn_info.equalities.eq_type[i_e, i_b] = dyn_info.equalities.eq_type[i_last, i_b]
                    dyn_info.equalities.eq_obj1id[i_e, i_b] = dyn_info.equalities.eq_obj1id[i_last, i_b]
                    dyn_info.equalities.eq_obj2id[i_e, i_b] = dyn_info.equalities.eq_obj2id[i_last, i_b]
                    dyn_info.equalities.eq_data[i_e, i_b] = dyn_info.equalities.eq_data[i_last, i_b]
                    dyn_info.equalities.sol_params[i_e, i_b] = dyn_info.equalities.sol_params[i_last, i_b]
                constraint_state.qd_n_equalities[i_b] = constraint_state.qd_n_equalities[i_b] - 1


# =====================================================================================================================
# ================================================= Solving Iteration =================================================
# =====================================================================================================================

# ====================================== Hessian Matrix & Cholesky Factorization ======================================


@qd.func
def func_compute_island_envelope(
    i_b,
    i_island,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Compute one island's skyline envelope: the smallest island-local column that can be structurally nonzero in
    each local row of its Hessian block H = M + J.T @ D @ J. The two coupling sources are known a priori:
    - constraint supports: a constraint couples all DOFs in its support, so its smallest local DOF bounds the others;
    - mass: the kinematic tree (mass_parent_mask) couples a DOF with its ancestors and same-link DOFs.

    The island's DOFs are gathered in ascending global order (dof_id), so local order matches global order and the
    envelope is a valid band. Computed once per step (structural) and reused across Newton iterations.
    """
    EPS = rigid_info.EPS[None]
    n = constraint_state.island.dof_slices.n[i_island, i_b]
    dof_base = constraint_state.island.dof_slices.start[i_island, i_b]
    con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
    con_n = constraint_state.island.constraint_slices.n[i_island, i_b]

    for ld in range(n):
        constraint_state.island.dof_env_start_local[dof_base + ld, i_b] = ld

    # Constraint coupling: a constraint's smallest live (|jac| > EPS) local DOF bounds the envelope of every other live
    # DOF it touches. Iterate the constraint's own support (jac_dofs_idx, mapped to island-local positions via
    # dof_local_pos) rather than scanning the whole island - O(support) instead of O(island size). The |jac| > EPS test
    # matches the whole-island scan it replaces and the assembly's outer guard, so DOFs that are structurally in the
    # support but currently have a zero Jacobian column are excluded identically, keeping the envelope deterministic.
    for i_lcon in range(con_n):
        i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
        col_min = n
        for k_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
            ld = constraint_state.island.dof_local_pos[constraint_state.jac_dofs_idx[i_c, k_, i_b], i_b]
            if ld < col_min:
                col_min = ld
        for k_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
            ld = constraint_state.island.dof_local_pos[constraint_state.jac_dofs_idx[i_c, k_, i_b], i_b]
            if col_min < constraint_state.island.dof_env_start_local[dof_base + ld, i_b]:
                constraint_state.island.dof_env_start_local[dof_base + ld, i_b] = col_min

    # Mass coupling: the kinematic-tree mask is directional (descendant -> ancestor) plus full intra-link, so check
    # both orientations. DOFs are ascending, so the first coupled lower column is the smallest.
    for ld in range(n):
        i_dg = constraint_state.island.dof_id[dof_base + ld, i_b]
        for ld2 in range(ld):
            j_dg = constraint_state.island.dof_id[dof_base + ld2, i_b]
            if rigid_info.mass_parent_mask[i_dg, j_dg] > 0.5 or rigid_info.mass_parent_mask[j_dg, i_dg] > 0.5:
                if ld2 < constraint_state.island.dof_env_start_local[dof_base + ld, i_b]:
                    constraint_state.island.dof_env_start_local[dof_base + ld, i_b] = ld2
                break

    # Transpose the envelope into per-column heights: col_end[c] = max row whose envelope reaches column c. The
    # column-oriented sweeps (rank-1 update, direct factor, backward substitution) iterate rows (c, col_end[c]]
    # instead of testing every row below c against its envelope. O(sum_span), like the envelope itself.
    for ld in range(n):
        constraint_state.island.dof_env_col_end[dof_base + ld, i_b] = ld
    for ld in range(n):
        env_i = constraint_state.island.dof_env_start_local[dof_base + ld, i_b]
        for c in range(env_i, ld):
            if ld > constraint_state.island.dof_env_col_end[dof_base + c, i_b]:
                constraint_state.island.dof_env_col_end[dof_base + c, i_b] = ld


@qd.func
def func_add_cone_hessian_block(
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
    is_removal: qd.template() = False,
    scale_by_jacobi: qd.template() = False,
):
    """Accumulate J_c^T H_c J_c (the coupled elliptic-cone contribution) into the Hessian nt_H[i_b].

    A middle-zone contact adds the packed symmetric local block H_c over the cone rows' shared DOF support (top
    zone adds nothing; bottom zone is the plain per-row J^T D J the caller already accumulated with active=True). H_c
    is positive semi-definite (PSD), so nt_H stays symmetric positive definite (SPD) and the factor kernels are
    unchanged. is_removal negates the block, so a caller bracketing a non-destructive factor can carry the cone through
    an incrementally-maintained nt_H without a rebuild: add before the factor, remove after.

    On the CPU backend every cone's cone_prev_jaref is seeded from its current residuals, so the incremental factor's
    downdate targets exactly the block baked here (an adding caller always precedes the factor there).
    """
    n_rows = qd.static(rigid_config.rows_per_contact)
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_cone = constraint_state.n_constraints_cone[i_b]
    for i_cone in range(n_cone // n_rows):
        i_head = nef + i_cone * n_rows
        rows_efc_D, rows_friction, con_mu, rows_jaref = _func_cone_head_load(
            i_head, i_b, constraint_state, rigid_config
        )
        if qd.static(rigid_config.backend == gs.cpu):
            for i_r in qd.static(range(n_rows)):
                constraint_state.cone_prev_jaref[i_cone * n_rows + i_r, i_b] = rows_jaref[i_r]
        zone, N, T = _func_cone_zone(rows_jaref, rows_efc_D, con_mu, rows_friction, rigid_config)
        if zone == 2:
            _rows_force, _cost, cone_H = _func_cone_middle(
                rows_jaref, rows_efc_D, con_mu, rows_friction, N, T, rigid_config
            )
            jac_n = constraint_state.jac_n_dofs[i_head, i_b]
            for i_d1_ in range(jac_n):
                i_d1 = constraint_state.jac_dofs_idx[i_head, i_d1_, i_b]
                for i_d2_ in range(i_d1_ + 1):
                    i_d2 = constraint_state.jac_dofs_idx[i_head, i_d2_, i_b]
                    row = qd.max(i_d1, i_d2)
                    col = qd.min(i_d1, i_d2)
                    rows_jac_row = qd.Vector.zero(gs.qd_float, n_rows)
                    rows_jac_col = qd.Vector.zero(gs.qd_float, n_rows)
                    for i_r in qd.static(range(n_rows)):
                        rows_jac_row[i_r] = constraint_state.jac[i_head + i_r, row, i_b]
                        rows_jac_col[i_r] = constraint_state.jac[i_head + i_r, col, i_b]
                    block = _func_cone_block_product(cone_H, rows_jac_row, rows_jac_col)
                    w_row = row
                    w_col = col
                    # A caller bracketing a non-destructive factor maintains nt_H in scaled coordinates; the block
                    # rides the stored scale (see nt_jacobi in array_class.py).
                    if qd.static(scale_by_jacobi):
                        block = block * constraint_state.nt_jacobi[row, i_b] * constraint_state.nt_jacobi[col, i_b]
                    if qd.static(is_removal):
                        block = -block
                    constraint_state.nt_H[i_b, w_row, w_col] = constraint_state.nt_H[i_b, w_row, w_col] + block


@qd.func
def func_wrap_cone_hessian(
    constraint_state: array_class.ConstraintState, rigid_config: qd.template(), is_removal: qd.template()
):
    """Add (is_removal=False) or remove (is_removal=True) the coupled elliptic-cone Hessian block of every improved
    env, a no-op unless the elliptic cone is active.

    Bracketing the per-island tiled factor+solve, which reads nt_H without consuming it, with add then remove lets the
    cone ride the incrementally maintained nt_H: the current cone block is present while the factor reads nt_H, then
    removed so the next patch lands on a cone-free Hessian.
    """
    if qd.static(rigid_config.enable_elliptic_friction):
        _B = constraint_state.jac.shape[2]
        qd.loop_config(name="wrap_cone_hessian", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                func_add_cone_hessian_block(
                    i_b,
                    constraint_state,
                    rigid_config,
                    is_removal=is_removal,
                    scale_by_jacobi=rigid_config.enable_jacobi_equilibration,
                )


@qd.func
def func_add_cone_hessian_block_island(
    i_b,
    i_island,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
    scale_by_jacobi: qd.template() = False,
):
    """Accumulate the coupled elliptic-cone contribution J_c^T H_c J_c of one island's middle-zone cones into nt_H.

    Island analogue of func_add_cone_hessian_block: a cone's rows share one DOF support and land in one island,
    so its middle-zone coupling is scattered over that support in the same island-local orientation as the
    assembly's J^T D J loop; the per-row J^T D J of the active rows is the caller's job. On the CPU backend every
    cone's cone_prev_jaref is seeded from its current residuals, so the incremental factor's downdate targets
    exactly the block baked here.
    """
    n_rows = qd.static(rigid_config.rows_per_contact)
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_cone = constraint_state.n_constraints_cone[i_b]
    con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
    con_n = constraint_state.island.constraint_slices.n[i_island, i_b]
    for i_lcon in range(con_n):
        i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
        if i_c >= nef and i_c < nef + n_cone and (i_c - nef) % n_rows == 0:
            rows_efc_D, rows_friction, con_mu, rows_jaref = _func_cone_head_load(
                i_c, i_b, constraint_state, rigid_config
            )
            # cone_prev_jaref backs the CPU incremental downdate, so seed it on the CPU backend.
            if qd.static(rigid_config.backend == gs.cpu):
                i_cone_row = i_c - nef
                for i_r in qd.static(range(n_rows)):
                    constraint_state.cone_prev_jaref[i_cone_row + i_r, i_b] = rows_jaref[i_r]
            zone, N, T = _func_cone_zone(rows_jaref, rows_efc_D, con_mu, rows_friction, rigid_config)
            if zone == 2:
                _rows_force, _cost, cone_H = _func_cone_middle(
                    rows_jaref, rows_efc_D, con_mu, rows_friction, N, T, rigid_config
                )
                jac_n = constraint_state.jac_n_dofs[i_c, i_b]
                for i_d1_ in range(jac_n):
                    i_d1 = constraint_state.jac_dofs_idx[i_c, i_d1_, i_b]
                    for i_d2_ in range(i_d1_, jac_n):
                        i_d2 = constraint_state.jac_dofs_idx[i_c, i_d2_, i_b]
                        row = qd.max(i_d1, i_d2)
                        col = qd.min(i_d1, i_d2)
                        if qd.static(rigid_config.sparse_solve):
                            if (
                                constraint_state.island.dof_local_pos[i_d1, i_b]
                                >= constraint_state.island.dof_local_pos[i_d2, i_b]
                            ) != (i_d1 >= i_d2):
                                row, col = col, row
                        rows_jac_row = qd.Vector.zero(gs.qd_float, n_rows)
                        rows_jac_col = qd.Vector.zero(gs.qd_float, n_rows)
                        for i_r in qd.static(range(n_rows)):
                            rows_jac_row[i_r] = constraint_state.jac[i_c + i_r, row, i_b]
                            rows_jac_col[i_r] = constraint_state.jac[i_c + i_r, col, i_b]
                        block = _func_cone_block_product(cone_H, rows_jac_row, rows_jac_col)
                        # H is maintained in scaled coordinates; see nt_jacobi in array_class.py
                        if qd.static(scale_by_jacobi):
                            block = block * constraint_state.nt_jacobi[row, i_b] * constraint_state.nt_jacobi[col, i_b]
                        constraint_state.nt_H[i_b, row, col] = constraint_state.nt_H[i_b, row, col] + block


@qd.func
def func_copy_cone_free_hessian_island(
    i_b, i_island, constraint_state: array_class.ConstraintState, save: qd.template()
):
    """Copy one island's skyline-envelope block between nt_H's factor slots and its packed cone-free mirror.

    save=True snapshots the freshly assembled cone-free block (before the cone blocks land) into each slot's mirror
    (diagonal into nt_H_cone_free_diag, whose lower slot belongs to the factor); save=False restores it for a
    rebuild. The islands partition the DOFs, so a block's mirror slots are never touched by another island or by any
    factor-side consumer. The envelope footprint covers every entry the assembly, cone bake and factor touch, so the
    copy is a complete block transfer.
    """
    n = constraint_state.island.dof_slices.n[i_island, i_b]
    dof_base = constraint_state.island.dof_slices.start[i_island, i_b]
    for i_d in range(n):
        i_dg = constraint_state.island.dof_id[dof_base + i_d, i_b]
        env_i = constraint_state.island.dof_env_start_local[dof_base + i_d, i_b]
        if qd.static(save):
            constraint_state.nt_H_cone_free_diag[i_b, i_dg] = constraint_state.nt_H[i_b, i_dg, i_dg]
        else:
            constraint_state.nt_H[i_b, i_dg, i_dg] = constraint_state.nt_H_cone_free_diag[i_b, i_dg]
        for j_d in range(env_i, i_d):
            j_dg = constraint_state.island.dof_id[dof_base + j_d, i_b]
            if qd.static(save):
                constraint_state.nt_H[i_b, j_dg, i_dg] = constraint_state.nt_H[i_b, i_dg, j_dg]
            else:
                constraint_state.nt_H[i_b, i_dg, j_dg] = constraint_state.nt_H[i_b, j_dg, i_dg]


@qd.func
def func_update_cone_free_hessian_flip(
    i_b,
    i_c,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Accumulate a flipped constraint's signed J^T D J rank-1 block into the packed cone-free Hessian.

    The sign follows the constraint's current active state (+D turned active, -D turned inactive), keeping the
    cone-free mirror of nt_H equal to the assembly of the current active set. The scatter mirrors the assembly's
    island-local storage orientation, transposed into each slot's mirror, with the diagonal in nt_H_cone_free_diag.
    """
    efc_D = constraint_state.efc_D[i_c, i_b]
    if not constraint_state.active[i_c, i_b]:
        efc_D = -efc_D
    jac_n = constraint_state.jac_n_dofs[i_c, i_b]
    for i_d1_ in range(jac_n):
        i_d1 = constraint_state.jac_dofs_idx[i_c, i_d1_, i_b]
        for i_d2_ in range(i_d1_, jac_n):
            i_d2 = constraint_state.jac_dofs_idx[i_c, i_d2_, i_b]
            v = constraint_state.jac[i_c, i_d1, i_b] * constraint_state.jac[i_c, i_d2, i_b] * efc_D
            # The mirror is maintained in scaled coordinates; see nt_jacobi in array_class.py
            if qd.static(rigid_config.enable_jacobi_equilibration):
                v = v * constraint_state.nt_jacobi[i_d1, i_b] * constraint_state.nt_jacobi[i_d2, i_b]
            if i_d1 == i_d2:
                constraint_state.nt_H_cone_free_diag[i_b, i_d1] = constraint_state.nt_H_cone_free_diag[i_b, i_d1] + v
            else:
                row = qd.max(i_d1, i_d2)
                col = qd.min(i_d1, i_d2)
                if qd.static(rigid_config.sparse_solve):
                    if (
                        constraint_state.island.dof_local_pos[i_d1, i_b]
                        >= constraint_state.island.dof_local_pos[i_d2, i_b]
                    ) != (i_d1 >= i_d2):
                        row, col = col, row
                constraint_state.nt_H[i_b, col, row] = constraint_state.nt_H[i_b, col, row] + v


@qd.func
def func_hessian_direct_batch(
    i_b,
    i_island,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Compute the Hessian block H = M + J.T @ D @ J of island i_island.

    Only the lower triangle is written (H is symmetric); the solver always reads from the lower triangle.

    The island's n x n block is assembled in place into the lower triangle of nt_H[i_b] at the island's global DOF
    rows/cols (dof_id maps the island's local index to its global dof, in ascending order). The global Hessian is block-
    diagonal by island, so the off-block entries are left untouched. The island's constraints are listed in
    constraint_id[constraint.start : +constraint.n].
    """
    EPS = rigid_info.EPS[None]

    n = constraint_state.island.dof_slices.n[i_island, i_b]
    dof_base = constraint_state.island.dof_slices.start[i_island, i_b]
    con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
    # Self-contained scale refresh over the island's own DOFs and constraints: assembly consumes it below, and a
    # lone island's rebuild must not depend on any other island assembling (see nt_jacobi in array_class.py).
    if qd.static(rigid_config.enable_jacobi_equilibration):
        con_n_scale = constraint_state.island.constraint_slices.n[i_island, i_b]
        for i_d in range(n):
            i_dg = constraint_state.island.dof_id[dof_base + i_d, i_b]
            constraint_state.nt_jacobi[i_dg, i_b] = rigid_info.mass_mat[i_dg, i_dg, i_b]
        # Each active row adds D * jac^2 to the diagonal of every dof of its support, the rows in list order so every dof
        # sums its rows in that order. Walking the rows' supports costs their total size; a sweep of every dof of the
        # island over every row of the island would read n_dofs * n_rows Jacobian entries, most of them structural zeros.
        for i_lcon in range(con_n_scale):
            i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
            if constraint_state.active[i_c, i_b]:
                efc_D = constraint_state.efc_D[i_c, i_b]
                for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                    i_dg = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                    constraint_state.nt_jacobi[i_dg, i_b] = (
                        constraint_state.nt_jacobi[i_dg, i_b] + efc_D * constraint_state.jac[i_c, i_dg, i_b] ** 2
                    )
        if qd.static(rigid_config.enable_elliptic_friction):
            n_rows_scale = qd.static(rigid_config.rows_per_contact)
            nef_scale = constraint_state.n_constraints_equality[i_b] + constraint_state.n_constraints_frictionloss[i_b]
            n_cone_scale = constraint_state.n_constraints_cone[i_b]
            for i_lcon in range(con_n_scale):
                i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
                if i_c >= nef_scale and i_c < nef_scale + n_cone_scale and (i_c - nef_scale) % n_rows_scale == 0:
                    rows_efc_D, rows_friction, con_mu, rows_jaref = _func_cone_head_load(
                        i_c, i_b, constraint_state, rigid_config
                    )
                    zone, N, T = _func_cone_zone(rows_jaref, rows_efc_D, con_mu, rows_friction, rigid_config)
                    if zone == 2:
                        _rows_force, _cost, cone_H = _func_cone_middle(
                            rows_jaref, rows_efc_D, con_mu, rows_friction, N, T, rigid_config
                        )
                        jac_n = constraint_state.jac_n_dofs[i_c, i_b]
                        for i_d1_ in range(jac_n):
                            i_d1 = constraint_state.jac_dofs_idx[i_c, i_d1_, i_b]
                            rows_jac = qd.Vector.zero(gs.qd_float, n_rows_scale)
                            for i_r in qd.static(range(n_rows_scale)):
                                rows_jac[i_r] = constraint_state.jac[i_c + i_r, i_d1, i_b]
                            constraint_state.nt_jacobi[i_d1, i_b] = constraint_state.nt_jacobi[
                                i_d1, i_b
                            ] + _func_cone_block_product(cone_H, rows_jac, rows_jac)
        for i_d in range(n):
            i_dg = constraint_state.island.dof_id[dof_base + i_d, i_b]
            hess_diag = constraint_state.nt_jacobi[i_dg, i_b]
            s = gs.qd_float(1.0)
            if hess_diag > 0.0:
                s = 1.0 / qd.sqrt(hess_diag)
            constraint_state.nt_jacobi[i_dg, i_b] = s
    con_n = constraint_state.island.constraint_slices.n[i_island, i_b]
    # The assembly and the factor visit only the row's skyline envelope [env_start, i_d]: entries below env_start are
    # structurally zero (no constraint or mass coupling reaches them). The skyline solve reads within the same band; the
    # dense block solve (see func_cholesky_solve_batch) reads the whole lower triangle, so off the skyline path the
    # entries below the envelope are zeroed here rather than left as an earlier step's factor wrote them. The extra
    # stores run once per seed, which the solve's per-iteration dense reads amortize to nothing measurable.
    for i_d in range(n):
        i_dg = constraint_state.island.dof_id[dof_base + i_d, i_b]
        env_i = constraint_state.island.dof_env_start_local[dof_base + i_d, i_b]
        j_lo = env_i if qd.static(rigid_config.sparse_solve) else 0
        for j_d in range(j_lo, i_d + 1):
            j_dg = constraint_state.island.dof_id[dof_base + j_d, i_b]
            constraint_state.nt_H[i_b, i_dg, j_dg] = gs.qd_float(0.0)
    # H += J.T @ D @ J by scattering each island constraint's rank update over the DOF pairs in its support
    # (jac_dofs_idx), the triangle oriented by island-local position: the fill-reducing dof_id of the CPU skyline path
    # is not globally monotonic, and every per-island factor/solve reads the block through the same orientation.
    for i_lcon in range(con_n):
        i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
        # An inactive constraint contributes nothing to H; skip its whole scatter instead of multiplying by 0
        if constraint_state.active[i_c, i_b]:
            efc_D = constraint_state.efc_D[i_c, i_b]
            jac_n = constraint_state.jac_n_dofs[i_c, i_b]
            for i_d1_ in range(jac_n):
                i_d1 = constraint_state.jac_dofs_idx[i_c, i_d1_, i_b]
                for i_d2_ in range(i_d1_, jac_n):
                    i_d2 = constraint_state.jac_dofs_idx[i_c, i_d2_, i_b]
                    row = qd.max(i_d1, i_d2)
                    col = qd.min(i_d1, i_d2)
                    if qd.static(rigid_config.sparse_solve):
                        if (
                            constraint_state.island.dof_local_pos[i_d1, i_b]
                            >= constraint_state.island.dof_local_pos[i_d2, i_b]
                        ) != (i_d1 >= i_d2):
                            row, col = col, row
                    contrib = constraint_state.jac[i_c, i_d1, i_b] * constraint_state.jac[i_c, i_d2, i_b] * efc_D
                    # Each contribution carries its rows' scales so H assembles equilibrated; see nt_jacobi in
                    # array_class.py.
                    if qd.static(rigid_config.enable_jacobi_equilibration):
                        contrib = (
                            contrib * constraint_state.nt_jacobi[i_d1, i_b] * constraint_state.nt_jacobi[i_d2, i_b]
                        )
                    constraint_state.nt_H[i_b, row, col] = constraint_state.nt_H[i_b, row, col] + contrib
    # H += M over the island's dofs, bounded by each dof's mass block (dofs_mass_block_start, mapped to local through
    # dof_local_pos): the mass couples no dofs across blocks, and a block lies within the envelope.
    for i_d in range(n):
        i_dg = constraint_state.island.dof_id[dof_base + i_d, i_b]
        mass_block_start = rigid_info.dofs_mass_block_start[i_dg]
        mass_lo = constraint_state.island.dof_local_pos[mass_block_start, i_b]
        for j_d in range(mass_lo, i_d + 1):
            j_dg = constraint_state.island.dof_id[dof_base + j_d, i_b]
            mass = rigid_info.mass_mat[i_dg, j_dg, i_b]
            # Scaled like the island scatter above
            if qd.static(rigid_config.enable_jacobi_equilibration):
                mass = mass * constraint_state.nt_jacobi[i_dg, i_b] * constraint_state.nt_jacobi[j_dg, i_b]
            constraint_state.nt_H[i_b, i_dg, j_dg] = constraint_state.nt_H[i_b, i_dg, j_dg] + mass
    # Persist the cone-free block before the cone blocks land, so a rebuild restores it by an envelope copy and
    # bakes the current cone blocks on top instead of reassembling J^T D J (func_factor_island_incremental_or_direct).
    if qd.static(rigid_config.enable_cone_free_hessian_reuse):
        func_copy_cone_free_hessian_island(i_b, i_island, constraint_state, save=True)
    # Coupled elliptic-cone Hessian block for this island's middle-zone cones; the per-row J^T D J of the active
    # rows was already added above.
    if qd.static(rigid_config.enable_elliptic_friction):
        func_add_cone_hessian_block_island(
            i_b, i_island, constraint_state, rigid_config, scale_by_jacobi=rigid_config.enable_jacobi_equilibration
        )


@qd.func
def func_island_assemble_factor_solve_tiled(
    i_b,
    i_island,
    tid,
    sh_L,
    sh_v,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    TileCls: qd.template(),
    tile_size: qd.template(),
    max_dofs: qd.template(),
    is_last_class: qd.template(),
    write_L: qd.template() = False,
):
    """Barrier-free tiled Cholesky factor + triangular solve of one island's Newton system.

    Stages the island's Hessian block, held at its global dof rows and columns of nt_H, into the shared tile sh_L and
    factors it there with register-streaming TileTxT primitives, a block whose dofs are one ascending run copied tile
    by tile and a scattered one gathered through dof_id.

    The call serves one size class of islands (see island_tile_caps), its tile of TileCls with tile_size lanes: an
    island of up to max_dofs dofs factors in the shared tile sh_L of max_dofs rows, and the last class also takes what
    no tile holds, the contiguous islands above max_dofs (factor in global memory) and the scattered ones above it
    (scalar per-island solve on lane 0).

    nt_H holds the island's Hessian block on entry, assembled or maintained by func_island_hessian_assemble_all. The
    two fallbacks above the last cap factor in place, so they consume the block: the graph assembles a contiguous one
    anew every iteration, and the scalar fallback assembles its own.

    L stays in the shared tile sh_L (local island indices); grad/Mgrad are global, reached through dof_id. block.sync
    fences the staging before the cooperative factor and the result before the caller's termination test.

    write_L persists L from sh_L back into the island's nt_H block, needed when a later step reads L from nt_H rather
    than re-factoring (the monolith's incremental rank-1 iterations); the decomposed graph re-factors every iteration
    so it leaves write_L False and keeps nt_H holding the raw Hessian.
    """
    T = qd.static(tile_size)
    LOG2_T = qd.static(T.bit_length() - 1)
    EPS = rigid_info.EPS[None]

    n = constraint_state.island.dof_slices.n[i_island, i_b]
    dof_base = constraint_state.island.dof_slices.start[i_island, i_b]
    con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
    con_n = constraint_state.island.constraint_slices.n[i_island, i_b]
    gbase = constraint_state.island.dof_id[dof_base, i_b]

    # The island's gathered DOFs are ascending, so the block is contiguous iff first and last span exactly n indices.
    # Within the class the block is staged into the shared tile, by tile copies when contiguous and gathered through
    # dof_id otherwise, and factored there in place (fused factor + solve). Above the largest class a contiguous block
    # factors in nt_H global (no shared-memory DOF cap), so even a whole-body-sized island avoids the serial scalar
    # solve, and a scattered one falls back to the scalar per-island solve. Quadrants forbids `return` inside a runtime
    # branch, so these are an if/elif/elif.
    is_contiguous = constraint_state.island.dof_id[dof_base + n - 1, i_b] == gbase + n - 1
    if n <= qd.static(max_dofs):
        # --- Stage the lower triangle of the island block into sh_L (local indices) ---
        N_BLOCKS = (n + T - 1) // T
        if is_contiguous:
            for kb in range(N_BLOCKS):
                lk0 = kb * T
                lk1 = qd.min(lk0 + T, n)
                for ib in range(kb, N_BLOCKS):
                    li0 = ib * T
                    li1 = qd.min(li0 + T, n)
                    H_ik = TileCls.zeros(dtype=gs.qd_float)
                    H_ik[:] = constraint_state.nt_H[i_b, gbase + li0 : gbase + li1, gbase + lk0 : gbase + lk1]
                    sh_L[li0:li1, lk0:lk1] = H_ik
        else:
            n_tri = n * (n + 1) // 2
            i_tri = tid
            while i_tri < n_tri:
                i_d, j_d = linear_to_lower_tri(i_tri)
                gi = constraint_state.island.dof_id[dof_base + i_d, i_b]
                gj = constraint_state.island.dof_id[dof_base + j_d, i_b]
                sh_L[i_d, j_d] = constraint_state.nt_H[i_b, gi, gj]
                i_tri = i_tri + T
        qd.simt.block.sync()

        # --- Blocked left-looking Cholesky in place in sh_L (register tiles, no block sync) ---
        # Column block kb reads its own blocks of H once, before it overwrites them with L, and the prior columns it
        # subtracts already hold L, so the factor needs no second buffer.
        for kb in range(N_BLOCKS):
            lk0 = kb * T
            lk1 = qd.min(lk0 + T, n)
            L_kk = TileCls.eye(dtype=gs.qd_float)
            L_kk[:] = sh_L[lk0:lk1, lk0:lk1]
            for jb in range(kb):
                lj0 = jb * T
                for t in range(T):
                    v = sh_L[lk0:lk1, lj0 + t]
                    L_kk -= qd.outer(v, v)
            # Floored relative to the row's original diagonal; see the tiled factor's floor comment.
            d_row = lk0 + tid
            diag_orig = gs.qd_float(1.0)
            if d_row < lk1:
                diag_orig = sh_L[d_row, d_row]
            L_kk.cholesky_(EPS * qd.max(diag_orig, EPS))
            for ib in range(kb + 1, N_BLOCKS):
                li0 = ib * T
                li1 = qd.min(li0 + T, n)
                L_ik = TileCls.zeros(dtype=gs.qd_float)
                L_ik[:] = sh_L[li0:li1, lk0:lk1]
                for jb in range(kb):
                    lj0 = jb * T
                    for t in range(T):
                        v_own = sh_L[li0:li1, lj0 + t]
                        v_diag = sh_L[lk0:lk1, lj0 + t]
                        L_ik -= qd.outer(v_own, v_diag)
                L_kk.solve_triangular_(L_ik)
                sh_L[li0:li1, lk0:lk1] = L_ik
            sh_L[lk0:lk1, lk0:lk1] = L_kk

        # --- Triangular solve grad -> Mgrad from sh_L (local indices; grad/Mgrad global through dof_id) ---
        k = tid
        while k < n:
            gd = constraint_state.island.dof_id[dof_base + k, i_b]
            sh_v[k] = constraint_state.grad[gd, i_b]
            # L factors the scaled block, so the solve wraps with nt_jacobi (see array_class.py).
            if qd.static(rigid_config.enable_jacobi_equilibration):
                sh_v[k] = sh_v[k] * constraint_state.nt_jacobi[gd, i_b]
            k = k + T
        qd.simt.block.sync()
        for i_r in range(n):
            dot = gs.qd_float(0.0)
            j = tid
            while j < i_r:
                dot = dot + sh_L[i_r, j] * sh_v[j]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                sh_v[i_r] = (sh_v[i_r] - dot) / sh_L[i_r, i_r]
            qd.simt.block.sync()
        for i_r_ in range(n):
            i_r = n - 1 - i_r_
            dot = gs.qd_float(0.0)
            j = i_r + 1 + tid
            while j < n:
                dot = dot + sh_L[j, i_r] * sh_v[j]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                sh_v[i_r] = (sh_v[i_r] - dot) / sh_L[i_r, i_r]
            qd.simt.block.sync()

        # Write the solved Mgrad back to global memory (local sh_v -> global through dof_id)
        k = tid
        while k < n:
            gd = constraint_state.island.dof_id[dof_base + k, i_b]
            constraint_state.Mgrad[gd, i_b] = sh_v[k]
            if qd.static(rigid_config.enable_jacobi_equilibration):
                constraint_state.Mgrad[gd, i_b] = sh_v[k] * constraint_state.nt_jacobi[gd, i_b]
            k = k + T
        qd.simt.block.sync()

        # Persist the factor: store L's lower triangle (local sh_L) at the island's global dof rows and columns of
        # nt_H so a caller that reads L from nt_H instead of re-factoring (the monolith's incremental rank-1
        # iterations, see func_cholesky_solve_batch) finds it there.
        if qd.static(write_L):
            i_r = tid
            while i_r < n:
                gi = constraint_state.island.dof_id[dof_base + i_r, i_b]
                for j in range(i_r + 1):
                    gj = constraint_state.island.dof_id[dof_base + j, i_b]
                    constraint_state.nt_H[i_b, gi, gj] = sh_L[i_r, j]
                i_r = i_r + T
            qd.simt.block.sync()
    elif qd.static(is_last_class):
        if is_contiguous:
            # Contiguous island too large for the shared tile: factor with the same register-streaming tiled
            # algorithm, but keep L in nt_H global so there is no DOF cap. A T-threaded triangular solve then reads
            # L from nt_H using Mgrad as the working vector. This replaces the serial scalar solve, whose O(n^3)
            # factor on a single lane dominates for big islands (e.g. a humanoid body). Left-looking blocked
            # Cholesky with register tiles, prior L columns read back from nt_H (block_dim == T == one subgroup, so
            # the cooperative tile loads/stores are lockstep - no block.sync between column blocks needed).
            N_BLOCKS = (n + T - 1) // T
            for kb in range(N_BLOCKS):
                lk0 = kb * T
                lk1 = qd.min(lk0 + T, n)
                gk0 = gbase + lk0
                gk1 = gbase + lk1
                L_kk = TileCls.eye(dtype=gs.qd_float)
                L_kk[:] = constraint_state.nt_H[i_b, gk0:gk1, gk0:gk1]
                for jb in range(kb):
                    lj0 = jb * T
                    for t in range(T):
                        v = constraint_state.nt_H[i_b, gk0:gk1, gbase + lj0 + t]
                        L_kk -= qd.outer(v, v)
                # Floored relative to the row's original diagonal; see the tiled factor's floor comment.
                d_row = gk0 + tid
                diag_orig = gs.qd_float(1.0)
                if d_row < gk1:
                    diag_orig = constraint_state.nt_H[i_b, d_row, d_row]
                L_kk.cholesky_(EPS * qd.max(diag_orig, EPS))
                for ib in range(kb + 1, N_BLOCKS):
                    li0 = ib * T
                    li1 = qd.min(li0 + T, n)
                    gi0 = gbase + li0
                    gi1 = gbase + li1
                    L_ik = TileCls.zeros(dtype=gs.qd_float)
                    L_ik[:] = constraint_state.nt_H[i_b, gi0:gi1, gk0:gk1]
                    for jb in range(kb):
                        lj0 = jb * T
                        for t in range(T):
                            v_own = constraint_state.nt_H[i_b, gi0:gi1, gbase + lj0 + t]
                            v_diag = constraint_state.nt_H[i_b, gk0:gk1, gbase + lj0 + t]
                            L_ik -= qd.outer(v_own, v_diag)
                    L_kk.solve_triangular_(L_ik)
                    constraint_state.nt_H[i_b, gi0:gi1, gk0:gk1] = L_ik
                constraint_state.nt_H[i_b, gk0:gk1, gk0:gk1] = L_kk
            qd.simt.block.sync()

            # Triangular solve L L^T x = grad -> Mgrad, reading L from nt_H. Mgrad is the working vector (no shared
            # tile, so no DOF cap); the T threads stripe each row's dot product and lane 0 writes the solved entry.
            k = tid
            while k < n:
                constraint_state.Mgrad[gbase + k, i_b] = constraint_state.grad[gbase + k, i_b]
                # L factors the scaled block, so the solve wraps with nt_jacobi (see array_class.py).
                if qd.static(rigid_config.enable_jacobi_equilibration):
                    constraint_state.Mgrad[gbase + k, i_b] = (
                        constraint_state.grad[gbase + k, i_b] * constraint_state.nt_jacobi[gbase + k, i_b]
                    )
                k = k + T
            qd.simt.block.sync()
            for i_r in range(n):
                dot = gs.qd_float(0.0)
                j = tid
                while j < i_r:
                    dot = (
                        dot
                        + constraint_state.nt_H[i_b, gbase + i_r, gbase + j] * constraint_state.Mgrad[gbase + j, i_b]
                    )
                    j = j + T
                dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
                if tid == 0:
                    constraint_state.Mgrad[gbase + i_r, i_b] = (
                        constraint_state.Mgrad[gbase + i_r, i_b] - dot
                    ) / constraint_state.nt_H[i_b, gbase + i_r, gbase + i_r]
                qd.simt.block.sync()
            for i_r_ in range(n):
                i_r = n - 1 - i_r_
                dot = gs.qd_float(0.0)
                j = i_r + 1 + tid
                while j < n:
                    dot = (
                        dot
                        + constraint_state.nt_H[i_b, gbase + j, gbase + i_r] * constraint_state.Mgrad[gbase + j, i_b]
                    )
                    j = j + T
                dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
                if tid == 0:
                    constraint_state.Mgrad[gbase + i_r, i_b] = (
                        constraint_state.Mgrad[gbase + i_r, i_b] - dot
                    ) / constraint_state.nt_H[i_b, gbase + i_r, gbase + i_r]
                qd.simt.block.sync()
            if qd.static(rigid_config.enable_jacobi_equilibration):
                k = tid
                while k < n:
                    constraint_state.Mgrad[gbase + k, i_b] = (
                        constraint_state.Mgrad[gbase + k, i_b] * constraint_state.nt_jacobi[gbase + k, i_b]
                    )
                    k = k + T
                qd.simt.block.sync()
        else:
            # Scattered island above the last cap: scalar per-island solve on lane 0, which writes both L
            # (func_cholesky_factor_direct_batch) and Mgrad (func_cholesky_solve_batch) to global.
            if tid == 0:
                # The factor below overwrites the block with L, so the block is rebuilt from the Jacobian first: a
                # maintained Hessian patched onto last iteration's L would be garbage.
                func_hessian_direct_batch(i_b, i_island, constraint_state, dyn_info, rigid_info, rigid_config)
                func_cholesky_factor_direct_batch(i_b, i_island, constraint_state, rigid_info, rigid_config)
                func_cholesky_solve_batch(
                    i_b,
                    i_island,
                    rhs=constraint_state.grad,
                    out=constraint_state.Mgrad,
                    constraint_state=constraint_state,
                    rigid_config=rigid_config,
                )
            qd.simt.block.sync()


@qd.func
def func_island_hessian_assemble_block(
    i_b,
    i_island,
    tid,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    block_dim: qd.template(),
):
    """Assemble the Hessian block M + J.T @ D @ J of island i_island into nt_H by the block_dim lanes of a block.

    Every lane accumulates one entry of the block's lower triangle over the island's rows, at the island's global dof
    rows and columns, the dofs indexed by offset where the island's list holds consecutive dofs (see dof_range_start in
    IslandState) and read from the list otherwise.

    Under Jacobi equilibration the block is scaled to unit diagonal afterwards, see nt_jacobi in array_class.py.
    """
    n = constraint_state.island.dof_slices.n[i_island, i_b]
    dof_lo = constraint_state.island.dof_slices.start[i_island, i_b]
    dof_base = constraint_state.island.dof_range_start[i_island, i_b]
    con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
    con_n = constraint_state.island.constraint_slices.n[i_island, i_b]
    n_tri = n * (n + 1) // 2
    i_tri = tid
    while i_tri < n_tri:
        i_d, j_d = linear_to_lower_tri(i_tri)
        gi = linesearch.func_list_item(constraint_state.island.dof_id, dof_lo + i_d, dof_lo, dof_base, i_b)
        gj = linesearch.func_list_item(constraint_state.island.dof_id, dof_lo + j_d, dof_lo, dof_base, i_b)
        h = rigid_info.mass_mat[gi, gj, i_b]
        for i_lcon in range(con_n):
            i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
            if constraint_state.active[i_c, i_b]:
                h = (
                    h
                    + constraint_state.jac[i_c, gi, i_b]
                    * constraint_state.jac[i_c, gj, i_b]
                    * constraint_state.efc_D[i_c, i_b]
                )
        constraint_state.nt_H[i_b, gi, gj] = h
        i_tri = i_tri + block_dim
    if qd.static(rigid_config.enable_jacobi_equilibration):
        qd.simt.block.sync()
        i_d = tid
        while i_d < n:
            gi = linesearch.func_list_item(constraint_state.island.dof_id, dof_lo + i_d, dof_lo, dof_base, i_b)
            hess_diag = constraint_state.nt_H[i_b, gi, gi]
            s = gs.qd_float(1.0)
            if hess_diag > 0.0:
                s = 1.0 / qd.sqrt(hess_diag)
            constraint_state.nt_jacobi[gi, i_b] = s
            i_d = i_d + block_dim
        qd.simt.block.sync()
        i_tri = tid
        while i_tri < n_tri:
            i_d, j_d = linear_to_lower_tri(i_tri)
            gi = linesearch.func_list_item(constraint_state.island.dof_id, dof_lo + i_d, dof_lo, dof_base, i_b)
            gj = linesearch.func_list_item(constraint_state.island.dof_id, dof_lo + j_d, dof_lo, dof_base, i_b)
            constraint_state.nt_H[i_b, gi, gj] = (
                constraint_state.nt_H[i_b, gi, gj]
                * constraint_state.nt_jacobi[gi, i_b]
                * constraint_state.nt_jacobi[gj, i_b]
            )
            i_tri = i_tri + block_dim


@qd.func
def func_island_hessian_patch_block(
    i_b,
    i_island,
    tid,
    sh_scan,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
    block_dim: qd.template(),
):
    """Patch the maintained Hessian block of island i_island with its flipped rows, by the block_dim lanes of a block.

    The J^T D J of a row whose active state flipped since the previous iteration is added where the row turned active
    and subtracted where it turned inactive. The flipped rows are listed first, in constraint order, in the island's own
    slice of incr_changed_idx: the lanes stride over the island's rows in chunks, each chunk ranked by a prefix sum over
    the flip flags, per subgroup then across the subgroups through sh_scan (one slot per subgroup). Every lane then
    accumulates the deltas of one lower-triangle entry over that list and adds the sum once, so the accumulation order
    is the row order. The deltas ride the scale the block was assembled with under Jacobi equilibration, see nt_jacobi
    in array_class.py.
    """
    _K = qd.static(32)
    N_SUBGROUPS = qd.static(block_dim // _K)
    con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
    con_n = constraint_state.island.constraint_slices.n[i_island, i_b]
    n_changed = 0
    i_chunk = 0
    while i_chunk < con_n:
        i_lcon = i_chunk + tid
        i_c = -1
        is_flipped = 0
        if i_lcon < con_n:
            i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
            if constraint_state.active[i_c, i_b] ^ constraint_state.prev_active[i_c, i_b]:
                is_flipped = 1
        rank_incl = qd.simt.subgroup.inclusive_add(is_flipped)
        if tid % _K == _K - 1:
            sh_scan[tid // _K] = rank_incl
        qd.simt.block.sync()
        rank = n_changed + rank_incl - 1
        n_chunk = 0
        for i_sub in qd.static(range(N_SUBGROUPS)):
            if i_sub < tid // _K:
                rank = rank + sh_scan[i_sub]
            n_chunk = n_chunk + sh_scan[i_sub]
        if is_flipped == 1:
            constraint_state.incr_changed_idx[con_base + rank, i_b] = i_c
        n_changed = n_changed + n_chunk
        qd.simt.block.sync()
        i_chunk = i_chunk + block_dim
    # One entry per lane, its delta summed over the listed rows and added once: a row-by-row accumulation into nt_H
    # would round each entry differently. The dofs are read through the island's list, by offset where they are one
    # run (see func_list_item).
    if n_changed > 0:
        n = constraint_state.island.dof_slices.n[i_island, i_b]
        dof_lo = constraint_state.island.dof_slices.start[i_island, i_b]
        dof_base = constraint_state.island.dof_range_start[i_island, i_b]
        n_tri = n * (n + 1) // 2
        i_tri = tid
        while i_tri < n_tri:
            i_d, j_d = linear_to_lower_tri(i_tri)
            gi = linesearch.func_list_item(constraint_state.island.dof_id, dof_lo + i_d, dof_lo, dof_base, i_b)
            gj = linesearch.func_list_item(constraint_state.island.dof_id, dof_lo + j_d, dof_lo, dof_base, i_b)
            delta = gs.qd_float(0.0)
            for i_lcon in range(n_changed):
                i_c = constraint_state.incr_changed_idx[con_base + i_lcon, i_b]
                contrib = (
                    constraint_state.efc_D[i_c, i_b]
                    * constraint_state.jac[i_c, gi, i_b]
                    * constraint_state.jac[i_c, gj, i_b]
                )
                if constraint_state.active[i_c, i_b]:
                    delta = delta + contrib
                else:
                    delta = delta - contrib
            if qd.static(rigid_config.enable_jacobi_equilibration):
                delta = delta * constraint_state.nt_jacobi[gi, i_b] * constraint_state.nt_jacobi[gj, i_b]
            constraint_state.nt_H[i_b, gi, gj] = constraint_state.nt_H[i_b, gi, gj] + delta
            i_tri = i_tri + block_dim


@qd.func
def func_island_hessian_assemble_all(
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    patch: qd.template() = False,
):
    """Assemble the Hessian block of every island of the work-list into nt_H ahead of the tiled factor.

    One block of BLOCK_DIM lanes per work item strides over the same work-list as the factor
    (func_island_hessian_assemble_block).

    Under patch the blocks are maintained instead: each one is patched with the rows of the island whose active state
    flipped since the previous iteration (func_island_hessian_patch_block), a patch costing one pass per flipped row
    where the assembly costs one per row. A contiguous island above the last tile cap factors in place (see
    func_island_assemble_factor_solve_tiled), so its block is assembled anew; a scattered one assembles its own.
    """
    N_BLOCKS = qd.static(max(1, rigid_config.island_factor_n_lanes // 32))
    N_CLASSES = qd.static(len(array_class.island_tile_caps(rigid_config)))
    LAST_CAP = qd.static(rigid_config.island_tile_cap_last)
    BLOCK_DIM = qd.static(128)
    region = constraint_state.island.factor_worklist_i_b.shape[0] // N_CLASSES
    qd.loop_config(name="island_hessian_assemble", block_dim=BLOCK_DIM)
    for i in range(N_BLOCKS * BLOCK_DIM):
        blk = i // BLOCK_DIM
        tid = i % BLOCK_DIM
        sh_scan = qd.simt.block.SharedArray((BLOCK_DIM // 32,), gs.qd_int)
        i_work = blk
        n_work = N_CLASSES * region
        while i_work < n_work:
            i_class = i_work // region
            i_b = -1
            i_island = -1
            if i_work - i_class * region < constraint_state.island.factor_worklist_size[i_class]:
                i_b = constraint_state.island.factor_worklist_i_b[i_work]
                i_island = constraint_state.island.factor_worklist_i_island[i_work]
            if i_b >= 0 and constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                if constraint_state.island.improved[i_island, i_b]:
                    if qd.static(patch):
                        n = constraint_state.island.dof_slices.n[i_island, i_b]
                        if n <= LAST_CAP:
                            func_island_hessian_patch_block(
                                i_b, i_island, tid, sh_scan, constraint_state, rigid_config, BLOCK_DIM
                            )
                        elif constraint_state.island.dof_range_start[i_island, i_b] >= 0:
                            func_island_hessian_assemble_block(
                                i_b, i_island, tid, constraint_state, rigid_info, rigid_config, BLOCK_DIM
                            )
                    else:
                        func_island_hessian_assemble_block(
                            i_b, i_island, tid, constraint_state, rigid_info, rigid_config, BLOCK_DIM
                        )
            qd.simt.block.sync()
            i_work = i_work + N_BLOCKS


@qd.func
def func_island_tiled_factor_solve_all(
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    write_L: qd.template() = False,
):
    """Barrier-free per-island factor + solve over the compact (env, island) work-list, one launch per island size
    class (see island_tile_caps).

    Drives func_island_assemble_factor_solve_tiled over the Hessian blocks held in nt_H. grad must already hold
    M*acc - force - qfrc (the no-solve gradient). write_L persists L into nt_H for a caller that reads the factor back
    (the monolith seed); the graph re-factors so it leaves it False.

    Every launch is a static grid of island_factor_n_lanes lanes (see array_class.py) in blocks of the class's tile
    size, grid-striding over the class's own work-list the partition build materialized, so the block count is
    decoupled from the env count and CUDA-graph capture sees a fixed launch. Each launch reserves the shared tile of its
    own class and sweeps the islands of that class alone, so the reservation of a launch follows the islands it factors
    instead of the largest island that could ever form, and the blocks stay resident several to a streaming
    multiprocessor. All T lanes of a block read the same work item, so n_constraints/improved and the hibernation and
    contiguity branches are uniform and the per-island block.sync is well-formed."""
    N_LANES = qd.static(rigid_config.island_factor_n_lanes)
    TILE_CAPS = qd.static(array_class.island_tile_caps(rigid_config))
    N_CLASSES = qd.static(len(TILE_CAPS))
    region = constraint_state.island.factor_worklist_i_b.shape[0] // N_CLASSES
    for i_class in qd.static(range(N_CLASSES)):
        MAX_DOFS = qd.static(TILE_CAPS[i_class])
        IS_LAST_CLASS = qd.static(i_class == N_CLASSES - 1)
        T = qd.static(array_class.cholesky_tile_size_for(MAX_DOFS))
        N_BLOCKS = qd.static(max(1, N_LANES // T))
        n_work = constraint_state.island.factor_worklist_size[i_class]
        qd.loop_config(name="island_tiled_factor_solve", block_dim=T)
        for i in range(N_BLOCKS * T):
            blk = i // T
            tid = i % T
            sh_L = qd.simt.block.SharedArray((MAX_DOFS, MAX_DOFS + 1), gs.qd_float)
            sh_v = qd.simt.block.SharedArray((MAX_DOFS,), gs.qd_float)
            i_work = blk
            while i_work < n_work:
                i_b = constraint_state.island.factor_worklist_i_b[i_class * region + i_work]
                i_island = constraint_state.island.factor_worklist_i_island[i_class * region + i_work]
                if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                    # An island standing still (asleep, or converged in the iterations) keeps its factor and its solve
                    if constraint_state.island.improved[i_island, i_b]:
                        func_island_assemble_factor_solve_tiled(
                            i_b,
                            i_island,
                            tid,
                            sh_L,
                            sh_v,
                            constraint_state,
                            dyn_info,
                            rigid_info,
                            rigid_config,
                            qd.simt.Tile32x32 if qd.static(T == 32) else qd.simt.Tile16x16,
                            T,
                            MAX_DOFS,
                            IS_LAST_CLASS,
                            write_L,
                        )
                    elif qd.static(rigid_config.use_hibernation):
                        if constraint_state.island.is_hibernated[i_island, i_b]:
                            # A hibernated island, whose factor and solve are skipped for the whole step, carries a
                            # zero gradient and search direction so that no stale direction steps its dofs. The
                            # gradient is zeroed where it is computed (func_update_gradient_no_solve,
                            # func_update_gradient_batch).
                            dof_start = constraint_state.island.dof_slices.start[i_island, i_b]
                            i_d_ = tid
                            while i_d_ < constraint_state.island.dof_slices.n[i_island, i_b]:
                                i_d = constraint_state.island.dof_id[dof_start + i_d_, i_b]
                                constraint_state.Mgrad[i_d, i_b] = gs.qd_float(0.0)
                                i_d_ = i_d_ + T
                # Fence the shared tile before this block reuses it for its next work item
                qd.simt.block.sync()
                i_work = i_work + N_BLOCKS


@qd.func
def func_cholesky_factor_direct_batch(
    i_b,
    i_island,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Compute the Cholesky factorization L of the Hessian block H = L @ L.T of island i_island in place, at its
    global DOF rows/cols of nt_H[i_b].

    Beware the Hessian matrix is re-purposed to store its Cholesky factorization to spare memory resources. Only
    the lower triangular part is updated, because the Hessian matrix is symmetric.
    """
    EPS = rigid_info.EPS[None]

    n = constraint_state.island.dof_slices.n[i_island, i_b]
    dof_base = constraint_state.island.dof_slices.start[i_island, i_b]
    # Factor the island's block in place at its global DOF rows/cols (dof_id is ascending, so all accesses below
    # stay in the lower triangle). The factorization is confined to each row's skyline envelope
    # (dof_env_start_local): a row's columns below its envelope start are structurally zero and fill-in stays
    # within the envelope, so a large island factors as a band instead of densely.
    for i_d in range(n):
        i_dg = constraint_state.island.dof_id[dof_base + i_d, i_b]
        i_start = constraint_state.island.dof_env_start_local[dof_base + i_d, i_b]
        hess_diag = constraint_state.nt_H[i_b, i_dg, i_dg]
        tmp = hess_diag
        for j_d in range(i_start, i_d):
            j_dg = constraint_state.island.dof_id[dof_base + j_d, i_b]
            tmp = tmp - constraint_state.nt_H[i_b, i_dg, j_dg] ** 2
        # Floored relative to the row's original diagonal; see the tiled factor's floor comment
        constraint_state.nt_H[i_b, i_dg, i_dg] = qd.sqrt(qd.max(tmp, EPS * qd.max(hess_diag, EPS)))
        inv = 1.0 / constraint_state.nt_H[i_b, i_dg, i_dg]
        # Only rows whose envelope reaches column i_d can be nonzero there
        j_d_end = n
        if qd.static(rigid_config.sparse_solve):
            j_d_end = constraint_state.island.dof_env_col_end[dof_base + i_d, i_b] + 1
        for j_d in range(i_d + 1, j_d_end):
            j_start = constraint_state.island.dof_env_start_local[dof_base + j_d, i_b]
            if j_start <= i_d:
                j_dg = constraint_state.island.dof_id[dof_base + j_d, i_b]
                dot = gs.qd_float(0.0)
                for k_d in range(qd.max(i_start, j_start), i_d):
                    k_dg = constraint_state.island.dof_id[dof_base + k_d, i_b]
                    dot = dot + (constraint_state.nt_H[i_b, j_dg, k_dg] * constraint_state.nt_H[i_b, i_dg, k_dg])
                constraint_state.nt_H[i_b, j_dg, i_dg] = (constraint_state.nt_H[i_b, j_dg, i_dg] - dot) * inv


@qd.func
def func_hessian_and_cholesky_factor_direct_batch(
    i_b,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    compute_envelope: qd.template() = False,
):
    # Combined Hessian build + Cholesky factor of every island of one env. compute_envelope sets each island's
    # structural skyline envelope first (callers do this once per step, then leave it False). An island standing
    # still (asleep, or converged in the iterations) keeps its factor.
    for i_island in range(constraint_state.island.n_islands[i_b]):
        if constraint_state.island.improved[i_island, i_b]:
            if qd.static(compute_envelope):
                func_compute_island_envelope(i_b, i_island, constraint_state, rigid_info, rigid_config)
            func_hessian_direct_batch(i_b, i_island, constraint_state, dyn_info, rigid_info, rigid_config)
            func_cholesky_factor_direct_batch(i_b, i_island, constraint_state, rigid_info, rigid_config)


@qd.func
def func_hessian_and_cholesky_factor_direct(
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    compute_envelope: qd.template() = False,
):
    """Assemble and factor the Hessian block of every island still iterating, over the flat (env, island) grid.

    compute_envelope computes each island's structural skyline envelope before factoring; it is structural, so callers
    set it once per step (func_solve_init) and leave it False for the per-iteration re-factorizations.
    """
    _B = constraint_state.jac.shape[2]

    # The block-diagonal Hessian factors per island; spread the islands across the (env, island) grid so they run
    # concurrently rather than serially within each env. max_islands bounds the per-env island count (at most one
    # island per tree); the guard skips the unused tail.
    max_islands = constraint_state.island.dof_slices.start.shape[0]
    qd.loop_config(
        name="hess_cholesky_factor_direct_island",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL),
        block_dim=32,
    )
    for i_b, i_island in qd.ndrange(_B, max_islands):
        if i_island < constraint_state.island.n_islands[i_b]:
            if constraint_state.island.improved[i_island, i_b]:
                if qd.static(compute_envelope):
                    func_compute_island_envelope(i_b, i_island, constraint_state, rigid_info, rigid_config)
                func_hessian_direct_batch(i_b, i_island, constraint_state, dyn_info, rigid_info, rigid_config)
                func_cholesky_factor_direct_batch(i_b, i_island, constraint_state, rigid_info, rigid_config)


@qd.func
def func_build_changed_constraint_list(i_b, constraint_state: array_class.ConstraintState):
    """Build a compact list of constraint indices whose active state changed.

    This reduces GPU thread divergence in the subsequent incremental Cholesky update by ensuring threads iterate
    only over constraints that need processing, rather than branching over all constraints.
    """
    n_changed = 0
    for i_c in range(constraint_state.n_constraints[i_b]):
        if constraint_state.active[i_c, i_b] ^ constraint_state.prev_active[i_c, i_b]:
            constraint_state.incr_changed_idx[n_changed, i_b] = i_c
            n_changed += 1
    constraint_state.incr_n_changed[i_b] = n_changed


@qd.func
def func_apply_rank1_dense_block(
    i_b, gbase, n, sign, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo
) -> bool:
    """Apply one rank-1 update (sign +1) or downdate (sign -1) to the dense factor L of the dof block
    [gbase, gbase + n) in nt_H: the whole env, or one island whose dofs are one ascending run.

    The working vector is pre-staged over the block's dofs in nt_vec at their global rows. Returns True on a
    non-positive downdate pivot. Shared by the active-set flip update (working vector jac * sqrt(D)) and the coupled
    cone update (block factor of the block staged one column at a time), so both maintain the dense factor through one
    code path.
    """
    EPS = rigid_info.EPS[None]
    is_degenerated = False
    for k_ in range(n):
        k = gbase + k_
        # Both thresholds are relative to the pivot they act on, whose units the working vector shares, so an update
        # counts as reaching a pivot, and a downdate as cancelling it, by the same fraction at any scene scale.
        Lkk = constraint_state.nt_H[i_b, k, k]
        if qd.abs(constraint_state.nt_vec[k, i_b]) > EPS * Lkk:
            tmp = Lkk**2 + sign * constraint_state.nt_vec[k, i_b] ** 2
            if tmp < EPS * Lkk**2:
                is_degenerated = True
                break
            r = qd.sqrt(tmp)
            c = r / Lkk
            cinv = 1 / c
            s = constraint_state.nt_vec[k, i_b] / Lkk
            constraint_state.nt_H[i_b, k, k] = r
            for i in range(k + 1, gbase + n):
                constraint_state.nt_H[i_b, i, k] = (
                    constraint_state.nt_H[i_b, i, k] + s * constraint_state.nt_vec[i, i_b] * sign
                ) * cinv

            for i in range(k + 1, gbase + n):
                constraint_state.nt_vec[i, i_b] = (
                    constraint_state.nt_vec[i, i_b] * c - s * constraint_state.nt_H[i_b, i, k]
                )

    return is_degenerated


@qd.func
def func_rank1_flip_dense_block(
    i_b,
    i_c,
    gbase,
    n,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
) -> bool:
    """Fold the active-set flip of constraint row i_c into the dense factor of the dof block [gbase, gbase + n): the
    row's jac * sqrt(D) over the block's dofs, in scaled coordinates under Jacobi equilibration (see nt_jacobi in
    array_class.py), applied as a rank-1 update where the row turned active and a downdate where it turned inactive.

    Returns True on a non-positive downdate pivot.
    """
    sign = 1.0 if constraint_state.active[i_c, i_b] else -1.0
    efc_D_sqrt = qd.sqrt(constraint_state.efc_D[i_c, i_b])
    for k in range(n):
        i_d = gbase + k
        v = constraint_state.jac[i_c, i_d, i_b] * efc_D_sqrt
        if qd.static(rigid_config.enable_jacobi_equilibration):
            v = v * constraint_state.nt_jacobi[i_d, i_b]
        constraint_state.nt_vec[i_d, i_b] = v
    return func_apply_rank1_dense_block(i_b, gbase, n, sign, constraint_state, rigid_info)


@qd.func
def func_factor_island_incremental_dense(
    i_b,
    i_island,
    gbase,
    n,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
) -> bool:
    """Maintain the dense factor of island i_island, whose dofs are the ascending run [gbase, gbase + n), through one
    rank-1 update or downdate per row of the island that flipped active since the previous iteration (see prev_active).

    Returns True on a non-positive downdate pivot, the caller then refactoring the island directly.
    """
    con_lo = constraint_state.island.constraint_slices.start[i_island, i_b]
    con_hi = con_lo + constraint_state.island.constraint_slices.n[i_island, i_b]
    con_base = linesearch.func_list_range_start(constraint_state.island.constraint_id, con_lo, con_hi, i_b)
    is_degenerated = False
    for i_pos in range(con_lo, con_hi):
        i_c = linesearch.func_list_item(constraint_state.island.constraint_id, i_pos, con_lo, con_base, i_b)
        if constraint_state.active[i_c, i_b] ^ constraint_state.prev_active[i_c, i_b]:
            if func_rank1_flip_dense_block(i_b, i_c, gbase, n, constraint_state, rigid_info, rigid_config):
                is_degenerated = True
    return is_degenerated


@qd.func
def func_hessian_and_cholesky_factor_incremental_dense_batch(
    i_b, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo, rigid_config: qd.template()
) -> bool:
    n_dofs = constraint_state.nt_H.shape[1]
    is_degenerated = False
    for idx in range(constraint_state.incr_n_changed[i_b]):
        i_c = constraint_state.incr_changed_idx[idx, i_b]
        if func_rank1_flip_dense_block(i_b, i_c, 0, n_dofs, constraint_state, rigid_info, rigid_config):
            is_degenerated = True
    return is_degenerated


@qd.func
def func_apply_staged_rank_updates_island(
    i_b,
    i_island,
    ld_start,
    n_u,
    signs,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
) -> bool:
    """Apply n_u staged rank-1 updates/downdates to the island's Cholesky block of L, in place in nt_H, fused into a
    single column sweep.

    The caller stages each update's working vector into nt_vec (slot-minor [i_d * hessian_rank_update_batch + i_u])
    and its sign (+1 update, -1 downdate) into signs, with ld_start the smallest island-local support row across the
    staged vectors. This sweep is agnostic to the source: batched per-constraint J^T D J rows (one rank-1 each) or a
    contact's coupled second-order-cone block (staged as its block factor's columns). A rank-1 rotation at column ld
    only reads state produced by earlier updates at that column and by its own rotations at earlier columns, so
    interleaving the n_u updates per column is bit-identical to running them sequentially while visiting each L
    column once. Returns whether any
    downdate went indefinite, in which case the caller refactors the island directly (discarding the partially
    updated L).
    """
    EPS = rigid_info.EPS[None]
    dof_base = constraint_state.island.dof_slices.start[i_island, i_b]
    n = constraint_state.island.dof_slices.n[i_island, i_b]

    is_degenerated = False
    for ld in range(ld_start, n):
        i_dg = constraint_state.island.dof_id[dof_base + ld, i_b]
        slot_base = i_dg * rigid_config.hessian_rank_update_batch
        # Diagonal phase: chain each update's rotation parameters through Lkk, in batch order (the same value each
        # update would read had the previous ones fully completed - column-local state only).
        cs = qd.Vector.zero(gs.qd_float, rigid_config.hessian_rank_update_batch)
        ss = qd.Vector.zero(gs.qd_float, rigid_config.hessian_rank_update_batch)
        cinvs = qd.Vector.zero(gs.qd_float, rigid_config.hessian_rank_update_batch)
        is_rotated = qd.Vector.zero(gs.qd_int, rigid_config.hessian_rank_update_batch)
        Lkk = constraint_state.nt_H[i_b, i_dg, i_dg]
        for i_u in qd.static(range(rigid_config.hessian_rank_update_batch)):
            if i_u < n_u and not is_degenerated:
                vk = constraint_state.nt_vec[slot_base + i_u, i_b]
                if qd.abs(vk) > EPS * Lkk:
                    tmp = Lkk**2 + signs[i_u] * vk**2
                    if tmp < EPS * Lkk**2:
                        is_degenerated = True
                    else:
                        r = qd.sqrt(tmp)
                        cinvs[i_u] = Lkk / r
                        cs[i_u] = r / Lkk
                        ss[i_u] = vk / Lkk
                        is_rotated[i_u] = 1
                        Lkk = r
        if is_degenerated:
            break
        constraint_state.nt_H[i_b, i_dg, i_dg] = Lkk
        # Row phase: apply the n_u rotations to each coupled row, chaining L through the batch. Only rows whose
        # envelope reaches column ld can couple; col_end bounds them so a banded island sweeps its bandwidth
        # instead of every row below ld.
        jd_end = n
        if qd.static(rigid_config.sparse_solve):
            jd_end = constraint_state.island.dof_env_col_end[dof_base + ld, i_b] + 1
        for jd in range(ld + 1, jd_end):
            if constraint_state.island.dof_env_start_local[dof_base + jd, i_b] <= ld:
                j_dg = constraint_state.island.dof_id[dof_base + jd, i_b]
                j_slot_base = j_dg * rigid_config.hessian_rank_update_batch
                Lj = constraint_state.nt_H[i_b, j_dg, i_dg]
                for i_u in qd.static(range(rigid_config.hessian_rank_update_batch)):
                    if is_rotated[i_u] == 1:
                        vj = constraint_state.nt_vec[j_slot_base + i_u, i_b]
                        Lj = (Lj + signs[i_u] * ss[i_u] * vj) * cinvs[i_u]
                        constraint_state.nt_vec[j_slot_base + i_u, i_b] = cs[i_u] * vj - ss[i_u] * Lj
                constraint_state.nt_H[i_b, j_dg, i_dg] = Lj
        for i_u in qd.static(range(rigid_config.hessian_rank_update_batch)):
            if is_rotated[i_u] == 1:
                constraint_state.nt_vec[slot_base + i_u, i_b] = gs.qd_float(0.0)
    return is_degenerated


@qd.func
def func_rank_batch_update_island(
    i_b,
    i_island,
    batch_ic,
    n_u,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
) -> bool:
    """Stage n_u flipped constraints as per-constraint rank-1 J^T D J updates and apply them to the island's L.

    Each update's sign follows the constraint's active state (+1 turned active, -1 turned inactive), applied through
    func_apply_staged_rank_updates_island. nt_vec holds one working vector per batch slot (slot-minor
    [i_d * hessian_rank_update_batch + i_u]); the caller zeroes the island's entries once per attempt.
    """
    n = constraint_state.island.dof_slices.n[i_island, i_b]
    signs = qd.Vector.zero(gs.qd_float, rigid_config.hessian_rank_update_batch)
    # Rows before the batch's first support DOF hold an exact zero in every working vector, so the sweep starts there.
    ld_start = n
    for i_u in qd.static(range(rigid_config.hessian_rank_update_batch)):
        if i_u < n_u:
            i_c = batch_ic[i_u]
            signs[i_u] = 1.0 if constraint_state.active[i_c, i_b] else -1.0
            efc_D_sqrt = qd.sqrt(constraint_state.efc_D[i_c, i_b])
            for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                slot_base = i_d * rigid_config.hessian_rank_update_batch
                v_stage = constraint_state.jac[i_c, i_d, i_b] * efc_D_sqrt
                # Scaled coordinates, matching the maintained L; see nt_jacobi in array_class.py.
                if qd.static(rigid_config.enable_jacobi_equilibration):
                    v_stage = v_stage * constraint_state.nt_jacobi[i_d, i_b]
                constraint_state.nt_vec[slot_base + i_u, i_b] = v_stage
                ld_support = constraint_state.island.dof_local_pos[i_d, i_b]
                if ld_support < ld_start:
                    ld_start = ld_support
    return func_apply_staged_rank_updates_island(
        i_b, i_island, ld_start, n_u, signs, constraint_state, rigid_info, rigid_config
    )


@qd.func
def func_cone_rank_update_island(
    i_b,
    i_island,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
) -> bool:
    """Maintain this island's coupled elliptic-cone contribution in the Cholesky factor L incrementally.

    A middle-zone contact contributes J_c^T H_c J_c, which varies with the residual each iteration, so this downdates
    the previous block and updates the current one. With H_c = L_c L_c^T, that block equals the sum of rank-1 terms
    w_j w_j^T with w_j = sum over rows i >= j of L_c[i][j] J_i; the previous w_j stage into slots [0, n_rows) (-1) and
    the current into slots [n_rows, 2*n_rows) (+1), applied by the shared rank sweep. Downdating first leaves the
    intermediate factor the well-conditioned cone-free system. cone_prev_jaref holds the previous residuals, indexed
    by the cone row offset past the equality and frictionloss rows. Returns True on a degenerate downdate, so the
    caller refactors the island directly.
    """
    EPS = rigid_info.EPS[None]
    B = qd.static(rigid_config.hessian_rank_update_batch)
    n_rows = qd.static(rigid_config.rows_per_contact)
    qd.static_assert(B >= 2 * n_rows, "the cone downdate + update stages 2 * rows_per_contact nt_vec slots per DOF")
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_cone = constraint_state.n_constraints_cone[i_b]
    con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
    con_n = constraint_state.island.constraint_slices.n[i_island, i_b]
    n = constraint_state.island.dof_slices.n[i_island, i_b]

    signs = qd.Vector.zero(gs.qd_float, B)
    for i_u in qd.static(range(2 * n_rows)):
        signs[i_u] = 2 * (i_u // n_rows) - 1

    is_degenerated = False
    for i_lcon in range(con_n):
        if not is_degenerated:
            i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
            if i_c >= nef and i_c < nef + n_cone and (i_c - nef) % n_rows == 0:
                i_cone_row = i_c - nef
                rows_efc_D, rows_friction, con_mu, rows_jaref_cur = _func_cone_head_load(
                    i_c, i_b, constraint_state, rigid_config
                )
                cur_zone, cN, cT = _func_cone_zone(rows_jaref_cur, rows_efc_D, con_mu, rows_friction, rigid_config)
                rows_jaref_prev = qd.Vector.zero(gs.qd_float, n_rows)
                for i_r in qd.static(range(n_rows)):
                    rows_jaref_prev[i_r] = constraint_state.cone_prev_jaref[i_cone_row + i_r, i_b]
                prev_zone, pN, pT = _func_cone_zone(rows_jaref_prev, rows_efc_D, con_mu, rows_friction, rigid_config)

                if cur_zone == 2 or prev_zone == 2:
                    cone_L_cur = _func_cone_block_chol(
                        rows_jaref_cur, rows_efc_D, con_mu, rows_friction, cur_zone, cN, cT, EPS, rigid_config
                    )
                    cone_L_prev = _func_cone_block_chol(
                        rows_jaref_prev, rows_efc_D, con_mu, rows_friction, prev_zone, pN, pT, EPS, rigid_config
                    )
                    ld_start = n
                    jac_n = constraint_state.jac_n_dofs[i_c, i_b]
                    for i_d_ in range(jac_n):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        slot_base = i_d * B
                        rows_jac = qd.Vector.zero(gs.qd_float, n_rows)
                        for i_r in qd.static(range(n_rows)):
                            rows_jac[i_r] = constraint_state.jac[i_c + i_r, i_d, i_b]
                        for j_r in qd.static(range(n_rows)):
                            w_prev = gs.qd_float(0.0)
                            w_cur = gs.qd_float(0.0)
                            for i_r in qd.static(range(j_r, n_rows)):
                                w_prev = w_prev + cone_L_prev[qd.static(_tri_idx(j_r, i_r, n_rows))] * rows_jac[i_r]
                                w_cur = w_cur + cone_L_cur[qd.static(_tri_idx(j_r, i_r, n_rows))] * rows_jac[i_r]
                            # Scaled coordinates, matching the maintained L; see nt_jacobi in array_class.py.
                            if qd.static(rigid_config.enable_jacobi_equilibration):
                                w_prev = w_prev * constraint_state.nt_jacobi[i_d, i_b]
                                w_cur = w_cur * constraint_state.nt_jacobi[i_d, i_b]
                            constraint_state.nt_vec[slot_base + j_r, i_b] = w_prev
                            constraint_state.nt_vec[slot_base + n_rows + j_r, i_b] = w_cur
                        ld_support = constraint_state.island.dof_local_pos[i_d, i_b]
                        if ld_support < ld_start:
                            ld_start = ld_support

                    if func_apply_staged_rank_updates_island(
                        i_b, i_island, ld_start, 2 * n_rows, signs, constraint_state, rigid_info, rigid_config
                    ):
                        is_degenerated = True

                for i_r in qd.static(range(n_rows)):
                    constraint_state.cone_prev_jaref[i_cone_row + i_r, i_b] = rows_jaref_cur[i_r]
    return is_degenerated


@qd.func
def func_cone_rank_update_whole_env(
    i_b, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo, rigid_config: qd.template()
) -> bool:
    """Maintain the whole-env coupled elliptic-cone contribution in the Cholesky factor L incrementally.

    Whole-env analogue of func_cone_rank_update_island: per middle-zone contact it applies the factor of H_c as one
    column sweep per cone row of the previous block (-1) then of the current block (+1), reusing the same single
    rank-1 sweep as the active-set update. Downdating first keeps the intermediate factor the well-conditioned
    cone-free system. cone_prev_jaref holds the previous residuals. Returns True on a degenerate downdate, so the caller
    refactors directly.
    """
    EPS = rigid_info.EPS[None]
    n_rows = qd.static(rigid_config.rows_per_contact)
    n_dofs = constraint_state.nt_H.shape[1]
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_cone = constraint_state.n_constraints_cone[i_b]

    is_degenerated = False
    if n_cone > 0:
        for i_cone in range(n_cone // n_rows):
            if not is_degenerated:
                i_head = nef + i_cone * n_rows
                i_cone_row = i_head - nef
                rows_efc_D, rows_friction, con_mu, rows_jaref_cur = _func_cone_head_load(
                    i_head, i_b, constraint_state, rigid_config
                )
                cur_zone, cN, cT = _func_cone_zone(rows_jaref_cur, rows_efc_D, con_mu, rows_friction, rigid_config)
                rows_jaref_prev = qd.Vector.zero(gs.qd_float, n_rows)
                for i_r in qd.static(range(n_rows)):
                    rows_jaref_prev[i_r] = constraint_state.cone_prev_jaref[i_cone_row + i_r, i_b]
                prev_zone, pN, pT = _func_cone_zone(rows_jaref_prev, rows_efc_D, con_mu, rows_friction, rigid_config)

                if cur_zone == 2 or prev_zone == 2:
                    cone_L_cur = _func_cone_block_chol(
                        rows_jaref_cur, rows_efc_D, con_mu, rows_friction, cur_zone, cN, cT, EPS, rigid_config
                    )
                    cone_L_prev = _func_cone_block_chol(
                        rows_jaref_prev, rows_efc_D, con_mu, rows_friction, prev_zone, pN, pT, EPS, rigid_config
                    )
                    # Term t stages column t of the previous factor, term n_rows + t of the current one (downdate
                    # first, see the docstring); an all-zero term (side outside the middle zone) stages a zero vector
                    # the sweep skips.
                    updates_coef = qd.Matrix.zero(gs.qd_float, 2 * n_rows, n_rows)
                    for j_r in qd.static(range(n_rows)):
                        for i_r in qd.static(range(j_r, n_rows)):
                            updates_coef[j_r, i_r] = cone_L_prev[qd.static(_tri_idx(j_r, i_r, n_rows))]
                            updates_coef[n_rows + j_r, i_r] = cone_L_cur[qd.static(_tri_idx(j_r, i_r, n_rows))]
                    jac_n = constraint_state.jac_n_dofs[i_head, i_b]
                    for i_u in range(2 * n_rows):
                        if not is_degenerated:
                            sign = 2 * (i_u // n_rows) - 1
                            for p in range(n_dofs):
                                constraint_state.nt_vec[p, i_b] = 0.0
                            for i_d_ in range(jac_n):
                                i_d = constraint_state.jac_dofs_idx[i_head, i_d_, i_b]
                                w = gs.qd_float(0.0)
                                for i_r in qd.static(range(n_rows)):
                                    w = w + updates_coef[i_u, i_r] * constraint_state.jac[i_head + i_r, i_d, i_b]
                                # Scaled coordinates, matching the maintained L; see nt_jacobi in array_class.py
                                if qd.static(rigid_config.enable_jacobi_equilibration):
                                    w = w * constraint_state.nt_jacobi[i_d, i_b]
                                constraint_state.nt_vec[i_d, i_b] = w
                            if func_apply_rank1_dense_block(i_b, 0, n_dofs, sign, constraint_state, rigid_info):
                                is_degenerated = True

                for i_r in qd.static(range(n_rows)):
                    constraint_state.cone_prev_jaref[i_cone_row + i_r, i_b] = rows_jaref_cur[i_r]
    return is_degenerated


@qd.func
def func_factor_island_incremental_or_direct(
    i_b,
    i_island,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Maintain one island's Cholesky factor for the current active set, choosing per island between an incremental
    rank-1 update/downdate and a direct refactor.

    One rank-1 update sweeps the island's skyline envelope at O(sum_span) (sum_span = total row span = envelope
    nonzeros), so n_changed of them cost O(n_changed * sum_span); a direct refactor factors it at O(sum_span_sq)
    (sum_span_sq = sum of squared row spans). The elliptic cone adds one fused multi-rank sweep per middle-zone
    contact to the incremental side, which the rebuild bakes into the Hessian instead. Both costs are read straight
    off the envelope, so the decision compares them directly, with no scene-tuned constant. The choice must be per
    island, not on the env-wide flip count: the rebuild path refactors every island, so a global decision would
    needlessly refactor quiescent islands whenever flips are spread thin across many of them (e.g. several separated
    piles each toggling a single contact).
    """
    c_start = constraint_state.island.constraint_slices.start[i_island, i_b]
    c_n = constraint_state.island.constraint_slices.n[i_island, i_b]

    # Count active-set flips and, for the elliptic cone, the middle-zone cone contacts whose coupled block must be
    # refreshed this iteration (each fires one downdate + update below). A cone whose current AND previous
    # residuals sit outside the middle zone has no block baked in L and leaves the factor valid, so an island with no
    # flips and no middle-zone cones skips entirely; its cone_prev_jaref goes stale, which is safe because a skipped
    # cone's stale residuals stay classified non-middle until the update runs again on its current residuals.
    n_changed = 0
    cone_passes = 0
    if qd.static(rigid_config.enable_elliptic_friction):
        nef = constraint_state.n_constraints_equality[i_b] + constraint_state.n_constraints_frictionloss[i_b]
        ncone = nef + constraint_state.n_constraints_cone[i_b]
        for k in range(c_n):
            i_c = constraint_state.island.constraint_id[c_start + k, i_b]
            if constraint_state.active[i_c, i_b] ^ constraint_state.prev_active[i_c, i_b]:
                n_changed = n_changed + 1
                # Keep the persisted cone-free Hessian synced with the current active set, whichever factor path
                # runs below: the incremental path leaves it the only cone-free image of the flip, and the rebuild
                # path restores it into nt_H right after.
                if qd.static(rigid_config.enable_cone_free_hessian_reuse):
                    func_update_cone_free_hessian_flip(i_b, i_c, constraint_state, rigid_info, rigid_config)
            if i_c >= nef and i_c < ncone and (i_c - nef) % qd.static(rigid_config.rows_per_contact) == 0:
                if _func_cone_head_is_middle(i_c, i_b, nef, constraint_state, rigid_config):
                    cone_passes = cone_passes + 1
    else:
        for k in range(c_n):
            i_c = constraint_state.island.constraint_id[c_start + k, i_b]
            if constraint_state.active[i_c, i_b] ^ constraint_state.prev_active[i_c, i_b]:
                n_changed = n_changed + 1

    if n_changed > 0 or cone_passes > 0:
        dof_base = constraint_state.island.dof_slices.start[i_island, i_b]
        n_isl_dofs = constraint_state.island.dof_slices.n[i_island, i_b]
        # Estimate both costs from the skyline envelope, per envelope entry: a fused pass over the envelope pays the
        # index/envelope work once (n_passes * sum_span) plus one rotation per update (n_changed * sum_span), while a
        # direct refactor pays index work and one FMA on each of the sum_span_sq entries. Each middle-zone cone contact
        # adds a further fused sweep to the incremental side (its downdate + update: 2 rotations per cone row plus one
        # envelope pass), which the rebuild avoids by baking the cone into the Hessian. Incremental wins while
        # (n_changed + n_passes + (2 * rows_per_contact + 1) * cone_passes) * sum_span < 2 * sum_span_sq. No
        # scene-tuned constant.
        sum_span = gs.qd_float(0.0)
        sum_span_sq = gs.qd_float(0.0)
        for ld in range(n_isl_dofs):
            row_span = gs.qd_float(ld - constraint_state.island.dof_env_start_local[dof_base + ld, i_b])
            sum_span = sum_span + row_span
            sum_span_sq = sum_span_sq + row_span**2
        n_passes = (n_changed + rigid_config.hessian_rank_update_batch - 1) // rigid_config.hessian_rank_update_batch
        need_rebuild = (
            gs.qd_float(n_changed + n_passes + qd.static(2 * rigid_config.rows_per_contact + 1) * cone_passes)
            * sum_span
            > 2.0 * sum_span_sq
        )
        if not need_rebuild:
            for ld in range(n_isl_dofs):
                slot_base = constraint_state.island.dof_id[dof_base + ld, i_b] * rigid_config.hessian_rank_update_batch
                for i_u in qd.static(range(rigid_config.hessian_rank_update_batch)):
                    constraint_state.nt_vec[slot_base + i_u, i_b] = gs.qd_float(0.0)
            # Gather the flipped constraints into fixed-size batches; apply each batch as one fused column sweep.
            batch_ic = qd.Vector.zero(gs.qd_int, rigid_config.hessian_rank_update_batch)
            n_u = 0
            for k in range(c_n):
                i_c = constraint_state.island.constraint_id[c_start + k, i_b]
                if constraint_state.active[i_c, i_b] ^ constraint_state.prev_active[i_c, i_b]:
                    batch_ic[n_u] = i_c
                    n_u = n_u + 1
                    if n_u == rigid_config.hessian_rank_update_batch:
                        if func_rank_batch_update_island(
                            i_b, i_island, batch_ic, n_u, constraint_state, rigid_info, rigid_config
                        ):
                            need_rebuild = True
                            break
                        n_u = 0
            if not need_rebuild and n_u > 0:
                if func_rank_batch_update_island(
                    i_b, i_island, batch_ic, n_u, constraint_state, rigid_info, rigid_config
                ):
                    need_rebuild = True
            # The active-set batch above maintains the per-row J^T D J of active rows; the coupled middle-zone cone
            # block (its rows inactive) is disjoint from that and is maintained here by its downdate/update.
            if qd.static(rigid_config.enable_elliptic_friction):
                if not need_rebuild:
                    if func_cone_rank_update_island(i_b, i_island, constraint_state, rigid_info, rigid_config):
                        need_rebuild = True
        if need_rebuild:
            # The persisted cone-free Hessian already reflects the current active set (flip scatters above), so the
            # rebuild restores it by an envelope copy and bakes the current cone blocks on top; without it, the full
            # J^T D J reassembly runs.
            if qd.static(rigid_config.enable_cone_free_hessian_reuse):
                func_copy_cone_free_hessian_island(i_b, i_island, constraint_state, save=False)
                func_add_cone_hessian_block_island(
                    i_b,
                    i_island,
                    constraint_state,
                    rigid_config,
                    scale_by_jacobi=rigid_config.enable_jacobi_equilibration,
                )
            else:
                func_hessian_direct_batch(i_b, i_island, constraint_state, dyn_info, rigid_info, rigid_config)
            func_cholesky_factor_direct_batch(i_b, i_island, constraint_state, rigid_info, rigid_config)


@qd.func
def func_hessian_and_cholesky_factor_incremental_batch(
    i_b,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
) -> bool:
    is_degenerated = False
    # An env holding one island maintains its factor as the whole env, through the changed-row list; the per-island
    # blocks serve the envs holding several, except under the elliptic cone, whose coupled block update
    # (func_cone_rank_update_whole_env) maintains the whole env's dense factor only.
    is_whole_env = constraint_state.island.n_islands[i_b] == 1
    if qd.static(rigid_config.enable_elliptic_friction and rigid_config.backend != gs.cpu):
        # The elliptic cone rides the incremental factor via a per-iteration block update backed by cone_prev_jaref,
        # which only the CPU backend allocates; a GPU scalar factor here instead rebuilds with the cone baked in each
        # iteration.
        func_hessian_and_cholesky_factor_direct_batch(i_b, constraint_state, dyn_info, rigid_info, rigid_config)
    elif is_whole_env:
        func_build_changed_constraint_list(i_b, constraint_state)
        is_degenerated = func_hessian_and_cholesky_factor_incremental_dense_batch(
            i_b, constraint_state, rigid_info, rigid_config
        )
        # The active-set update above maintains the per-row J^T D J of the flipped rows; the coupled middle-zone cone
        # block varies with the residual each iteration, so it rides the same factor via its block update here. Only
        # the CPU backend reaches this incremental path for elliptic (a GPU scalar factor rebuilt above); the static
        # backend guard also keeps func_cone_rank_update_whole_env (which indexes the CPU-only cone_prev_jaref) out of
        # the GPU compilation of this runtime branch.
        if qd.static(rigid_config.enable_elliptic_friction and rigid_config.backend == gs.cpu):
            if not is_degenerated:
                if func_cone_rank_update_whole_env(i_b, constraint_state, rigid_info, rigid_config):
                    is_degenerated = True
    elif qd.static(rigid_config.enable_elliptic_friction):
        # Several islands under the elliptic cone: every island still iterating refactors directly, cone baked in
        for i_island in range(constraint_state.island.n_islands[i_b]):
            if constraint_state.island.improved[i_island, i_b]:
                func_hessian_direct_batch(i_b, i_island, constraint_state, dyn_info, rigid_info, rigid_config)
                func_cholesky_factor_direct_batch(i_b, i_island, constraint_state, rigid_info, rigid_config)
    else:
        # Each island still iterating maintains its own factor: by rank-1 updates on its dense block where its dofs
        # are one ascending run (see dof_range_start in IslandState), refactored directly where they are not or where a
        # downdate went indefinite. An island standing still keeps its factor (see improved in IslandState).
        for i_island in range(constraint_state.island.n_islands[i_b]):
            if constraint_state.island.improved[i_island, i_b]:
                gbase = constraint_state.island.dof_range_start[i_island, i_b]
                n_island_dofs = constraint_state.island.dof_slices.n[i_island, i_b]
                is_island_degenerated = True
                if gbase >= 0:
                    is_island_degenerated = func_factor_island_incremental_dense(
                        i_b, i_island, gbase, n_island_dofs, constraint_state, rigid_info, rigid_config
                    )
                if is_island_degenerated:
                    func_hessian_direct_batch(i_b, i_island, constraint_state, dyn_info, rigid_info, rigid_config)
                    func_cholesky_factor_direct_batch(i_b, i_island, constraint_state, rigid_info, rigid_config)
    return is_degenerated


# ======================================== Cholesky Factorization and Solving =========================================


@qd.func
def func_cholesky_solve_batch(
    i_b,
    i_island,
    rhs: qd.Tensor,
    out: qd.Tensor,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
):
    """Solve L @ L.T @ out = rhs in place for island i_island, the right-hand side pre-scaled and the solution unscaled
    under Jacobi equilibration (see nt_jacobi in array_class.py).

    The island's factor sits in its block of nt_H at its global dof rows and columns, oriented by island-local position;
    the global Hessian is block-diagonal, so the island solves are independent and equal the single dense solve. An
    island whose dofs are one ascending run (see dof_range_start in IslandState) solves its block by offset; the others
    go through the dof list, confined to the skyline envelope of their rows on the CPU skyline path. The dense block
    read relies on the direct assembly zeroing the block below the envelope (see func_hessian_direct_batch). The forward
    solve passes grad and Mgrad, the adjoint solve dL_dqacc and bw_u (see kernel_solve_adjoint_u in backward.py), both
    by keyword (see the quadrants member-expansion note in func_solve_init).
    """
    n = constraint_state.island.dof_slices.n[i_island, i_b]
    gbase = constraint_state.island.dof_range_start[i_island, i_b]
    is_dense_block = False
    if qd.static(not rigid_config.sparse_solve):
        is_dense_block = gbase >= 0
    if is_dense_block:
        for ld in range(n):
            gd = gbase + ld
            curr_out = rhs[gd, i_b]
            if qd.static(rigid_config.enable_jacobi_equilibration):
                curr_out = curr_out * constraint_state.nt_jacobi[gd, i_b]
            for j_d in range(ld):
                curr_out = curr_out - constraint_state.nt_H[i_b, gd, gbase + j_d] * out[gbase + j_d, i_b]
            out[gd, i_b] = curr_out / constraint_state.nt_H[i_b, gd, gd]
        for ld_ in range(n):
            ld = n - 1 - ld_
            gd = gbase + ld
            curr_out = out[gd, i_b]
            for j_d in range(ld + 1, n):
                curr_out = curr_out - constraint_state.nt_H[i_b, gbase + j_d, gd] * out[gbase + j_d, i_b]
            out[gd, i_b] = curr_out / constraint_state.nt_H[i_b, gd, gd]
        if qd.static(rigid_config.enable_jacobi_equilibration):
            for ld in range(n):
                out[gbase + ld, i_b] = out[gbase + ld, i_b] * constraint_state.nt_jacobi[gbase + ld, i_b]
    else:
        dof_lo = constraint_state.island.dof_slices.start[i_island, i_b]
        for ld in range(n):
            gd = constraint_state.island.dof_id[dof_lo + ld, i_b]
            curr_out = rhs[gd, i_b]
            if qd.static(rigid_config.enable_jacobi_equilibration):
                curr_out = curr_out * constraint_state.nt_jacobi[gd, i_b]
            for j_d in range(constraint_state.island.dof_env_start_local[dof_lo + ld, i_b], ld):
                g_jd = constraint_state.island.dof_id[dof_lo + j_d, i_b]
                curr_out = curr_out - constraint_state.nt_H[i_b, gd, g_jd] * out[g_jd, i_b]
            out[gd, i_b] = curr_out / constraint_state.nt_H[i_b, gd, gd]
        for ld_ in range(n):
            ld = n - 1 - ld_
            gd = constraint_state.island.dof_id[dof_lo + ld, i_b]
            curr_out = out[gd, i_b]
            j_d_end = n
            if qd.static(rigid_config.sparse_solve):
                j_d_end = constraint_state.island.dof_env_col_end[dof_lo + ld, i_b] + 1
            for j_d in range(ld + 1, j_d_end):
                if constraint_state.island.dof_env_start_local[dof_lo + j_d, i_b] <= ld:
                    g_jd = constraint_state.island.dof_id[dof_lo + j_d, i_b]
                    curr_out = curr_out - constraint_state.nt_H[i_b, g_jd, gd] * out[g_jd, i_b]
            out[gd, i_b] = curr_out / constraint_state.nt_H[i_b, gd, gd]
        if qd.static(rigid_config.enable_jacobi_equilibration):
            for ld in range(n):
                gd = constraint_state.island.dof_id[dof_lo + ld, i_b]
                out[gd, i_b] = out[gd, i_b] * constraint_state.nt_jacobi[gd, i_b]


# =====================================================================================================================
# ==================================================== Linesearch =====================================================
# =====================================================================================================================


@qd.func
def update_bracket_no_eval_local(p_alpha, p_cost, p_grad, p_hess, alphas, costs, grads, hess):
    """Bracket update using local candidate values. No global memory access or _func_linesearch_eval_at_alpha call.

    Args:
        p_alpha, p_cost, p_grad, p_hess: current bracket point (scalar).
        alphas, costs, grads, hess: qd.Vector(3) of candidate values.
    """
    flag = 0

    for i in qd.static(range(3)):
        if p_grad < 0 and grads[i] < 0 and p_grad < grads[i]:
            p_alpha, p_cost, p_grad, p_hess = alphas[i], costs[i], grads[i], hess[i]
            flag = 1
        elif p_grad > 0 and grads[i] > 0 and p_grad > grads[i]:
            p_alpha, p_cost, p_grad, p_hess = alphas[i], costs[i], grads[i], hess[i]
            flag = 2

    p_next_alpha = p_alpha
    if flag > 0:
        p_next_alpha = p_alpha - p_grad / p_hess

    return flag, p_alpha, p_cost, p_grad, p_hess, p_next_alpha


# =====================================================================================================================
# ================================================= Solving Algorithm =================================================
# =====================================================================================================================


# ====================================================== Helpers ======================================================


def _tri_idx(i, j, dim):
    """Index of entry (i, j), i <= j, in the row-major packing of a dim x dim symmetric matrix's upper triangle.

    Used at trace time only (static loop indices), so the cone block algebra stays dimension-generic across the
    3-row (normal + 2 tangents) and 4-row (torsional friction adds the spin axis) elliptic cones. The same packing
    stores a lower-triangular factor column-major via _tri_idx(col, row).
    """
    return i * dim - i * (i - 1) // 2 + (j - i)


def _friction_blocks(rigid_config):
    """Row offset and width of each independently-bounded friction block of one elliptic contact, at trace time.

    Sliding friction spans the two tangent axes, torsional friction the spin axis and rolling friction the two tangent
    axes again, laid out as described in efc_frictionloss (see array_class.py). Only the 'signorini' resolution bounds
    them separately; 'convex' couples every axis into one ellipsoid.
    """
    blocks = [(1, 2)]
    if rigid_config.enable_torsional_friction:
        blocks.append((3, 1))
    if rigid_config.enable_rolling_friction:
        blocks.append((4, 2))
    return tuple(blocks)


@qd.func
def _func_cone_zone(rows_jaref, rows_efc_D, con_mu, rows_friction, rigid_config: qd.template()):
    """Classify one elliptic contact from its per-row residuals rows_jaref, for the resolution in force.

    Returns (zone, N, T). Under 'convex' these are MuJoCo's three cone zones - 0 = top (dual-cone interior, inactive),
    1 = bottom (polar cone, plain quadratic), 2 = middle (cone boundary) - with N and T the rescaled normal/tangential
    magnitudes the middle-zone force/cost/Hessian reuses.

    Under 'signorini' a contact only ever separates (zone 0) or carries force (zone 2), since friction is bounded
    against the normal force rather than traded against it: N carries that latched normal force, the radius scale of
    every friction disc, and T is unused because each block has its own tangential magnitude. Zone 1 never occurs -
    a sticking block is a plain quadratic within the zone-2 block, not a whole-contact state.

    con_mu, rows_efc_D and rows_friction as returned by _func_cone_head_load.
    """
    zone = 2
    N = gs.qd_float(0.0)
    T = gs.qd_float(0.0)
    if qd.static(rigid_config.enable_signorini_contact):
        N = qd.max(-rows_efc_D[0] * rows_jaref[0], 0.0)
        if N <= 0.0:
            zone = 0
    else:
        N = con_mu * rows_jaref[0]
        T_sq = gs.qd_float(0.0)
        for i_r in qd.static(range(1, rows_jaref.n)):
            u = rows_friction[i_r] * rows_jaref[i_r]
            T_sq = T_sq + u**2
        T = qd.sqrt(T_sq)
        # A residual on the polar boundary leaves the sign of con_mu * N + T to rounding, and the two zones supply
        # different curvature models, so two rigidly rotated copies of one scene take different search directions from
        # the same iterate. Claiming the band for the coupled zone keeps the choice a property of the geometry: force
        # and cost agree across the boundary, only the Hessian differs. The reference behaviour keeps the bare sign.
        POLAR_TIE_RATIO = qd.static(1e-3)
        polar_tie_tol = gs.qd_float(0.0)
        if qd.static(not rigid_config.enable_mujoco_compatibility):
            polar_tie_tol = POLAR_TIE_RATIO * T
        if N >= con_mu * T or (T <= 0.0 and N >= 0.0):
            zone = 0
        elif con_mu * N + T <= -polar_tie_tol or (T <= 0.0 and N < 0.0):
            zone = 1
    return zone, N, T


@qd.func
def _func_cone_head_load(
    i_c,
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
):
    """Load the shared per-contact scalars of the elliptic cone whose head (normal) row is i_c.

    Returns (rows_efc_D, rows_friction, con_mu, rows_jaref): the per-row impedances, the per-row friction
    coefficients (sliding on the head and tangent slots, torsional and rolling on their own slots, see
    efc_frictionloss in array_class.py), the regularized master coefficient con_mu = friction / sqrt(impratio)
    (computed as friction * sqrt(rows_efc_D[0] / rows_efc_D[1]) since the friction rows are impratio times stiffer),
    and the per-row residuals.
    """
    n_rows = qd.static(rigid_config.rows_per_contact)
    rows_efc_D = qd.Vector.zero(gs.qd_float, n_rows)
    rows_jaref = qd.Vector.zero(gs.qd_float, n_rows)
    for i_r in qd.static(range(n_rows)):
        rows_efc_D[i_r] = constraint_state.efc_D[i_c + i_r, i_b]
        rows_jaref[i_r] = constraint_state.Jaref[i_c + i_r, i_b]
    friction = constraint_state.efc_frictionloss[i_c, i_b]
    rows_friction = qd.Vector.zero(gs.qd_float, n_rows)
    for i_r in qd.static(range(n_rows)):
        rows_friction[i_r] = friction
    if qd.static(rigid_config.enable_torsional_friction):
        for i_r in qd.static(range(3, n_rows)):
            rows_friction[i_r] = constraint_state.efc_frictionloss[i_c + i_r, i_b]
    con_mu = friction * qd.sqrt(rows_efc_D[0] / rows_efc_D[1])
    return rows_efc_D, rows_friction, con_mu, rows_jaref


@qd.func
def _func_cone_head_is_middle(
    i_c,
    i_b,
    nef,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
) -> bool:
    """Whether elliptic cone contact head i_c sits in the middle (force-carrying) zone this iteration or last.

    The coupled block only fires its downdate/update when the current or previous residual is in the middle zone;
    the other zones leave the factor untouched. The incremental-vs-rebuild cost model uses this to charge only the
    cone contacts whose update actually runs.
    """
    i_cone_row = i_c - nef
    n_rows = qd.static(rigid_config.rows_per_contact)
    rows_efc_D, rows_friction, con_mu, rows_jaref = _func_cone_head_load(i_c, i_b, constraint_state, rigid_config)
    cur_zone, cur_N, cur_T = _func_cone_zone(rows_jaref, rows_efc_D, con_mu, rows_friction, rigid_config)
    rows_jaref_prev = qd.Vector.zero(gs.qd_float, n_rows)
    for i_r in qd.static(range(n_rows)):
        rows_jaref_prev[i_r] = constraint_state.cone_prev_jaref[i_cone_row + i_r, i_b]
    prev_zone, prev_N, prev_T = _func_cone_zone(rows_jaref_prev, rows_efc_D, con_mu, rows_friction, rigid_config)
    return cur_zone == 2 or prev_zone == 2


@qd.func
def _func_cone_Dm(D0, con_mu):
    """Middle-zone impedance Dm = D0 / (con_mu^2 * (1 + con_mu^2)), folding the normal impedance and the regularized
    coefficient of one elliptic contact."""
    return D0 / (con_mu * con_mu * (1.0 + con_mu * con_mu))


@qd.func
def _func_disc_middle(rows_jaref, rows_efc_D, rows_friction, f_n, rigid_config: qd.template()):
    """Force, cost and packed symmetric local Hessian of one elliptic contact under the 'signorini' resolution.

    The normal row keeps its plain unilateral quadratic. Each friction block (see _friction_blocks) is the Huber of
    the disc of radius coefficient * f_n, with f_n latched from the normal row at the current iterate: quadratic
    while the block sticks, conical once it saturates, the two agreeing in value, force and slope at the transition.
    Freezing the radius is what stops tangential demand from buying normal force, and makes the local Hessian
    block-diagonal - no block sees another's curvature, nor the normal row's.

    Returns (rows_force, cost, cone_H), cone_H packed row-major upper-triangle (see _tri_idx). Both Huber branches
    are positive semi-definite, so the assembled Hessian stays SPD as the coupled cone's does.
    """
    n_rows = qd.static(rigid_config.rows_per_contact)
    rows_force = qd.Vector.zero(gs.qd_float, n_rows)
    cone_H = qd.Vector.zero(gs.qd_float, n_rows * (n_rows + 1) // 2)
    rows_force[0] = f_n
    cone_H[0] = rows_efc_D[0]
    cost = 0.5 * rows_efc_D[0] * rows_jaref[0] ** 2
    for i_0, width in qd.static(_friction_blocks(rigid_config)):
        # The rows of a block share one impedance and one coefficient, both carried on its leading row.
        D = rows_efc_D[i_0]
        radius = rows_friction[i_0] * f_n
        T_sq = gs.qd_float(0.0)
        for i_r in qd.static(range(i_0, i_0 + width)):
            T_sq = T_sq + rows_jaref[i_r] ** 2
        T = qd.sqrt(T_sq)
        if D * T <= radius:
            cost = cost + 0.5 * D * T_sq
            for i_r in qd.static(range(i_0, i_0 + width)):
                rows_force[i_r] = -D * rows_jaref[i_r]
                cone_H[qd.static(_tri_idx(i_r, i_r, n_rows))] = D
        else:
            # T > 0 here: a saturated block has D * T above a radius that is never negative.
            cost = cost + radius * (T - 0.5 * radius / D)
            curvature = radius / T
            for i_r in qd.static(range(i_0, i_0 + width)):
                rows_force[i_r] = -curvature * rows_jaref[i_r]
                cone_H[qd.static(_tri_idx(i_r, i_r, n_rows))] = curvature * (1.0 - (rows_jaref[i_r] / T) ** 2)
                for j_r in qd.static(range(i_r + 1, i_0 + width)):
                    cone_H[qd.static(_tri_idx(i_r, j_r, n_rows))] = (
                        -curvature * rows_jaref[i_r] * rows_jaref[j_r] / T_sq
                    )
    return rows_force, cost, cone_H


@qd.func
def _func_cone_middle(rows_jaref, rows_efc_D, con_mu, rows_friction, N, T, rigid_config: qd.template()):
    """Middle-zone force, cost and packed symmetric local Hessian of one elliptic contact, for the resolution in force:
    MuJoCo's coupled cone under 'convex', the latched-radius friction discs under 'signorini'.

    Returns (rows_force, cost, cone_H), cone_H packed row-major upper-triangle (see _tri_idx). N and T carry what
    _func_cone_zone returned for this contact; only valid in the middle zone.
    """
    if qd.static(rigid_config.enable_signorini_contact):
        return _func_disc_middle(rows_jaref, rows_efc_D, rows_friction, N, rigid_config)
    return _func_cone_middle_convex(rows_jaref, rows_efc_D[0], con_mu, rows_friction, N, T)


@qd.func
def _func_cone_middle_convex(rows_jaref, D0, con_mu, rows_friction, N, T):
    """Middle-zone (cone-boundary) force, cost, and packed symmetric local Hessian for one elliptic contact.

    Returns (rows_force, cost, cone_H): the per-row forces, the coupled cost, and the row-major upper-triangle
    packing (see _tri_idx) of the analytic second-order-cone projection Hessian of MuJoCo's elliptic cone. Only
    valid when the contact is in the middle zone (T > 0).
    """
    n_rows = qd.static(rows_jaref.n)
    rows_u = qd.Vector.zero(gs.qd_float, n_rows)
    for i_r in qd.static(range(1, n_rows)):
        rows_u[i_r] = rows_friction[i_r] * rows_jaref[i_r]
    Dm = _func_cone_Dm(D0, con_mu)
    NmT = N - con_mu * T

    rows_force = qd.Vector.zero(gs.qd_float, n_rows)
    rows_force[0] = -Dm * NmT * con_mu
    for i_r in qd.static(range(1, n_rows)):
        rows_force[i_r] = -rows_force[0] / T * rows_u[i_r] * rows_friction[i_r]
    cost = 0.5 * Dm * NmT**2

    # Curvature in the rescaled U-space, then pre/post-multiplied by G = diag(con_mu, rows_friction[1:]) and
    # scaled by Dm.
    cN_T3 = con_mu * N / T**3
    diag_add = con_mu**2 - con_mu * N / T
    cone_H = qd.Vector.zero(gs.qd_float, n_rows * (n_rows + 1) // 2)
    cone_H[0] = Dm * con_mu**2
    for j_r in qd.static(range(1, n_rows)):
        cone_H[qd.static(_tri_idx(0, j_r, n_rows))] = Dm * con_mu * rows_friction[j_r] * (-(con_mu / T) * rows_u[j_r])
    for i_r in qd.static(range(1, n_rows)):
        cone_H[qd.static(_tri_idx(i_r, i_r, n_rows))] = (
            Dm * rows_friction[i_r] ** 2 * (cN_T3 * rows_u[i_r] ** 2 + diag_add)
        )
        for j_r in qd.static(range(i_r + 1, n_rows)):
            cone_H[qd.static(_tri_idx(i_r, j_r, n_rows))] = (
                Dm * rows_friction[i_r] * rows_friction[j_r] * (cN_T3 * rows_u[i_r] * rows_u[j_r])
            )
    return rows_force, cost, cone_H


@qd.func
def _func_cone_block_product(cone_H, rows_jac_row, rows_jac_col):
    """One entry J_row^T H_c J_col of the scattered coupled-cone Hessian.

    The packed symmetric local block H_c (see _tri_idx) is contracted with the cone rows' jacobian entries at the
    Hessian row DOF and column DOF.
    """
    n_rows = qd.static(rows_jac_row.n)
    product = gs.qd_float(0.0)
    for i_r in qd.static(range(n_rows)):
        product = product + cone_H[qd.static(_tri_idx(i_r, i_r, n_rows))] * rows_jac_row[i_r] * rows_jac_col[i_r]
        for j_r in qd.static(range(i_r + 1, n_rows)):
            product = product + cone_H[qd.static(_tri_idx(i_r, j_r, n_rows))] * (
                rows_jac_row[i_r] * rows_jac_col[j_r] + rows_jac_row[j_r] * rows_jac_col[i_r]
            )
    return product


@qd.func
def _func_cone_block_chol(rows_jaref, rows_efc_D, con_mu, rows_friction, zone, N, T, EPS, rigid_config: qd.template()):
    """Packed lower-triangular Cholesky factor of one contact's middle-zone Hessian H_c, entry L[i][j] stored
    column-major at _tri_idx(j, i).

    All entries are zero outside the middle zone. Diagonals are floored at EPS so a near-degenerate block cannot
    divide by zero (the last one at 0); a truly indefinite downdate built from the factor is caught later by the rank
    sweep's pivot test.
    """
    n_rows = qd.static(rows_jaref.n)
    cone_L = qd.Vector.zero(gs.qd_float, n_rows * (n_rows + 1) // 2)
    if zone == 2:
        _rows_force, _cost, cone_H = _func_cone_middle(
            rows_jaref, rows_efc_D, con_mu, rows_friction, N, T, rigid_config
        )
        for j_r in qd.static(range(n_rows)):
            pivot_sq = cone_H[qd.static(_tri_idx(j_r, j_r, n_rows))]
            for k_r in qd.static(range(j_r)):
                pivot_sq = pivot_sq - cone_L[qd.static(_tri_idx(k_r, j_r, n_rows))] ** 2
            if qd.static(j_r < n_rows - 1):
                cone_L[qd.static(_tri_idx(j_r, j_r, n_rows))] = qd.sqrt(qd.max(pivot_sq, EPS))
            else:
                cone_L[qd.static(_tri_idx(j_r, j_r, n_rows))] = qd.sqrt(qd.max(pivot_sq, 0.0))
            for i_r in qd.static(range(j_r + 1, n_rows)):
                offdiag = cone_H[qd.static(_tri_idx(j_r, i_r, n_rows))]
                for k_r in qd.static(range(j_r)):
                    offdiag = (
                        offdiag
                        - cone_L[qd.static(_tri_idx(k_r, i_r, n_rows))] * cone_L[qd.static(_tri_idx(k_r, j_r, n_rows))]
                    )
                cone_L[qd.static(_tri_idx(j_r, i_r, n_rows))] = offdiag / cone_L[qd.static(_tri_idx(j_r, j_r, n_rows))]
    return cone_L


@qd.func
def func_cone_middle_cost(
    i_c,
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
):
    """Middle-zone cost of the elliptic contact whose head row is i_c; 0 outside.

    Shared by every linesearch/cost path so the coupled term is computed identically across arms, and taken from
    _func_cone_middle so the cost driving warm-start and convergence decisions is the one whose forces and Hessian
    the solve actually applies, under either resolution.
    """
    rows_efc_D, rows_friction, con_mu, rows_jaref = _func_cone_head_load(i_c, i_b, constraint_state, rigid_config)
    zone, N, T = _func_cone_zone(rows_jaref, rows_efc_D, con_mu, rows_friction, rigid_config)
    c = gs.qd_float(0.0)
    if zone == 2:
        _rows_force, c, _cone_H = _func_cone_middle(rows_jaref, rows_efc_D, con_mu, rows_friction, N, T, rigid_config)
    return c


@qd.func
def _func_disc_cost_along_alpha(rows_jaref, rows_jv, alpha, rows_efc_D, rows_friction, rigid_config: qd.template()):
    """Latched-radius contact cost and its first/second derivatives in the linesearch step alpha ('signorini').

    Evaluated at rows_jaref + alpha * rows_jv, returning (cost, dcost/dalpha, d2cost/dalpha2). The disc radii are
    latched at the alpha=0 residuals and stay fixed along the whole search line, so cost, gradient and curvature all
    describe the one frozen objective the Newton step was computed from - the linesearch cannot then accept a step
    against a cost the step did not descend. The normal row's activity and each block's stick/slip regime are
    re-classified at this alpha, exactly as every other unilateral and Huber row is.
    """
    f_n = qd.max(-rows_efc_D[0] * rows_jaref[0], 0.0)
    rows_jaref_alpha = rows_jaref + alpha * rows_jv
    cost = gs.qd_float(0.0)
    grad = gs.qd_float(0.0)
    hess = gs.qd_float(0.0)
    if rows_jaref_alpha[0] < 0.0:
        cost = 0.5 * rows_efc_D[0] * rows_jaref_alpha[0] ** 2
        grad = rows_efc_D[0] * rows_jaref_alpha[0] * rows_jv[0]
        hess = rows_efc_D[0] * rows_jv[0] ** 2
    for i_0, width in qd.static(_friction_blocks(rigid_config)):
        D = rows_efc_D[i_0]
        radius = rows_friction[i_0] * f_n
        T_sq = gs.qd_float(0.0)
        dT_sum = gs.qd_float(0.0)
        d2T_sum = gs.qd_float(0.0)
        for i_r in qd.static(range(i_0, i_0 + width)):
            T_sq = T_sq + rows_jaref_alpha[i_r] ** 2
            dT_sum = dT_sum + rows_jaref_alpha[i_r] * rows_jv[i_r]
            d2T_sum = d2T_sum + rows_jv[i_r] ** 2
        T = qd.sqrt(T_sq)
        if D * T <= radius:
            cost = cost + 0.5 * D * T_sq
            grad = grad + D * dT_sum
            hess = hess + D * d2T_sum
        else:
            dT = dT_sum / T
            cost = cost + radius * (T - 0.5 * radius / D)
            grad = grad + radius * dT
            hess = hess + radius * (d2T_sum - dT**2) / T
    return cost, grad, hess


@qd.func
def _func_cone_cost_along_alpha(
    rows_jaref, rows_jv, alpha, rows_efc_D, con_mu, rows_friction, rigid_config: qd.template()
):
    """Exact elliptic-contact cost and its first/second derivatives in the linesearch step alpha, for the resolution in
    force: MuJoCo's coupled cone under 'convex', the latched-radius friction discs under 'signorini'.

    Evaluated at rows_jaref + alpha * rows_jv, returning (cost, dcost/dalpha, d2cost/dalpha2). Under 'convex' the zone
    is re-classified at this alpha, so the cost is exact along the whole search line (top: 0; bottom: plain quadratic;
    middle: the cone potential 0.5*Dm*(N - con_mu*T)^2 differentiated through T = ||rows_friction[1:] *
    rows_jaref[1:]||).
    """
    if qd.static(rigid_config.enable_signorini_contact):
        return _func_disc_cost_along_alpha(rows_jaref, rows_jv, alpha, rows_efc_D, rows_friction, rigid_config)
    rows_jaref_alpha = rows_jaref + alpha * rows_jv
    zone, N, T = _func_cone_zone(rows_jaref_alpha, rows_efc_D, con_mu, rows_friction, rigid_config)
    cost = gs.qd_float(0.0)
    grad = gs.qd_float(0.0)
    hess = gs.qd_float(0.0)
    if zone == 1:
        for i_r in qd.static(range(rows_jaref_alpha.n)):
            cost = cost + rows_efc_D[i_r] * rows_jaref_alpha[i_r] ** 2
            grad = grad + rows_efc_D[i_r] * rows_jaref_alpha[i_r] * rows_jv[i_r]
            hess = hess + rows_efc_D[i_r] * rows_jv[i_r] ** 2
        cost = 0.5 * cost
    elif zone == 2:
        Dm = _func_cone_Dm(rows_efc_D[0], con_mu)
        dN = con_mu * rows_jv[0]
        dT_sum = gs.qd_float(0.0)
        d2T_sum = gs.qd_float(0.0)
        for i_r in qd.static(range(1, rows_jaref_alpha.n)):
            u = rows_friction[i_r] * rows_jaref_alpha[i_r]
            du = rows_friction[i_r] * rows_jv[i_r]
            dT_sum = dT_sum + u * du
            d2T_sum = d2T_sum + du**2
        dT = dT_sum / T
        d2T = (d2T_sum - dT**2) / T
        g = N - con_mu * T
        dg = dN - con_mu * dT
        d2g = -con_mu * d2T
        cost = 0.5 * Dm * g**2
        grad = Dm * g * dg
        hess = Dm * (dg**2 + g * d2g)
    return cost, grad, hess


@qd.func
def _func_disc_cost_diff_along_alpha(
    rows_jaref, rows_jv, alpha, rows_efc_D, rows_friction, rigid_config: qd.template()
):
    """Shifted latched-radius contact cost cost(alpha) - cost(0) and its derivatives in alpha ('signorini').

    Derivatives match _func_disc_cost_along_alpha exactly. Each term cancels its own shared constant analytically
    whenever its regime is unchanged - the normal row and a sticking block as a pure quadratic in alpha, a saturated
    block as the radius times the change in tangential magnitude - rather than subtracting two large absolute costs,
    which rounds the delta to zero in float32 near convergence. Only a term that changed regime pays the absolute
    subtraction, and there the two costs genuinely differ.
    """
    f_n = qd.max(-rows_efc_D[0] * rows_jaref[0], 0.0)
    rows_jaref_alpha = rows_jaref + alpha * rows_jv
    _cost_alpha, grad, hess = _func_disc_cost_along_alpha(
        rows_jaref, rows_jv, alpha, rows_efc_D, rows_friction, rigid_config
    )
    cost_diff = gs.qd_float(0.0)
    is_normal_active = rows_jaref_alpha[0] < 0.0
    if is_normal_active == (rows_jaref[0] < 0.0):
        if is_normal_active:
            cost_diff = rows_efc_D[0] * (alpha * rows_jv[0] * rows_jaref[0] + 0.5 * alpha**2 * rows_jv[0] ** 2)
    elif is_normal_active:
        cost_diff = 0.5 * rows_efc_D[0] * rows_jaref_alpha[0] ** 2
    else:
        cost_diff = -0.5 * rows_efc_D[0] * rows_jaref[0] ** 2
    for i_0, width in qd.static(_friction_blocks(rigid_config)):
        D = rows_efc_D[i_0]
        radius = rows_friction[i_0] * f_n
        T_sq = gs.qd_float(0.0)
        T0_sq = gs.qd_float(0.0)
        for i_r in qd.static(range(i_0, i_0 + width)):
            T_sq = T_sq + rows_jaref_alpha[i_r] ** 2
            T0_sq = T0_sq + rows_jaref[i_r] ** 2
        T = qd.sqrt(T_sq)
        T0 = qd.sqrt(T0_sq)
        is_stick = D * T <= radius
        if is_stick == (D * T0 <= radius):
            if is_stick:
                cost_diff = cost_diff + 0.5 * D * (T_sq - T0_sq)
            else:
                cost_diff = cost_diff + radius * (T - T0)
        elif is_stick:
            cost_diff = cost_diff + 0.5 * D * T_sq - radius * (T0 - 0.5 * radius / D)
        else:
            cost_diff = cost_diff + radius * (T - 0.5 * radius / D) - 0.5 * D * T0_sq
    return cost_diff, grad, hess


@qd.func
def _func_cone_cost_diff_along_alpha(
    rows_jaref, rows_jv, alpha, rows_efc_D, con_mu, rows_friction, rigid_config: qd.template()
):
    """Shifted elliptic-contact cost cost(alpha) - cost(0) and its first/second derivatives in alpha, for the
    resolution in force.

    Derivatives match _func_cone_cost_along_alpha exactly; the value is the delta from alpha=0, computed zone-aware so
    the same-zone cases cancel the shared constant analytically (both-bottom: pure quadratic in alpha; both-middle:
    difference of squares on the cone gap) instead of subtracting two large absolute costs, which rounds the delta to
    zero in float32 near convergence. Mixed-zone pairs fall back to the absolute subtraction, where the two costs
    genuinely differ.
    """
    if qd.static(rigid_config.enable_signorini_contact):
        return _func_disc_cost_diff_along_alpha(rows_jaref, rows_jv, alpha, rows_efc_D, rows_friction, rigid_config)
    cost_alpha, grad, hess = _func_cone_cost_along_alpha(
        rows_jaref, rows_jv, alpha, rows_efc_D, con_mu, rows_friction, rigid_config
    )
    rows_jaref_alpha = rows_jaref + alpha * rows_jv
    zone, N, T = _func_cone_zone(rows_jaref_alpha, rows_efc_D, con_mu, rows_friction, rigid_config)
    zone0, N0, T0 = _func_cone_zone(rows_jaref, rows_efc_D, con_mu, rows_friction, rigid_config)
    cost_diff = gs.qd_float(0.0)
    if zone == zone0:
        if zone == 1:
            dcost_0 = gs.qd_float(0.0)
            d2cost = gs.qd_float(0.0)
            for i_r in qd.static(range(rows_jaref.n)):
                dcost_0 = dcost_0 + rows_efc_D[i_r] * rows_jv[i_r] * rows_jaref[i_r]
                d2cost = d2cost + rows_efc_D[i_r] * rows_jv[i_r] ** 2
            cost_diff = alpha * dcost_0 + alpha**2 * (0.5 * d2cost)
        elif zone == 2:
            Dm = _func_cone_Dm(rows_efc_D[0], con_mu)
            g = N - con_mu * T
            g0 = N0 - con_mu * T0
            cost_diff = 0.5 * Dm * (g - g0) * (g + g0)
    else:
        cost_0, _grad_0, _hess_0 = _func_cone_cost_along_alpha(
            rows_jaref, rows_jv, 0.0, rows_efc_D, con_mu, rows_friction, rigid_config
        )
        cost_diff = cost_alpha - cost_0
    return cost_diff, grad, hess


@qd.func
def func_cone_update_rows(
    i_c,
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
):
    """Recompute active and efc_force for the rows of the elliptic cone whose head (normal) row is i_c, and return
    the coupled middle-zone cost contribution (0 outside the middle zone).

    The cone rows are one second-order cone (SOC) processed together at the head: bottom-zone rows keep active=True
    so the shared per-row cost/Hessian passes handle them, while middle-zone rows are excluded (active=False) with
    their coupled cost returned here and their coupled Hessian block added in assembly. Shared by every force-update
    path so the cone resolves identically across arms; per-row callers invoke it from the head thread only, keeping
    each row written exactly once (race-free).
    """
    n_rows = qd.static(rigid_config.rows_per_contact)
    rows_efc_D, rows_friction, con_mu, rows_jaref = _func_cone_head_load(i_c, i_b, constraint_state, rigid_config)
    zone, N, T = _func_cone_zone(rows_jaref, rows_efc_D, con_mu, rows_friction, rigid_config)
    cost = gs.qd_float(0.0)
    if zone == 0:  # top: inactive
        for i_r in qd.static(range(n_rows)):
            constraint_state.active[i_c + i_r, i_b] = False
            constraint_state.efc_force[i_c + i_r, i_b] = 0.0
    elif zone == 1:  # bottom: plain quadratic on all rows
        for i_r in qd.static(range(n_rows)):
            constraint_state.active[i_c + i_r, i_b] = True
            constraint_state.efc_force[i_c + i_r, i_b] = -rows_jaref[i_r] * rows_efc_D[i_r]
    else:  # middle: cone boundary. Excluded from the per-row cost/Hessian; handled coupled.
        rows_force, cost_middle, _cone_H = _func_cone_middle(
            rows_jaref, rows_efc_D, con_mu, rows_friction, N, T, rigid_config
        )
        for i_r in qd.static(range(n_rows)):
            constraint_state.active[i_c + i_r, i_b] = False
            constraint_state.efc_force[i_c + i_r, i_b] = rows_force[i_r]
        cost = cost_middle
    return cost


@qd.func
def func_is_row_moving(i_c, i_b, constraint_state: array_class.ConstraintState, skip_settled_islands: qd.template()):
    """Whether constraint row i_c belongs to an island still iterating (see improved in IslandState).

    A row coupling no dof belongs to no island and is taken as moving, its update costing nothing. Always True unless
    skip_settled_islands, the gate of the per-iteration passes, which run for an env still iterating: an env holding one
    island then moves as a whole, and the island lookup is spared.
    """
    is_moving = True
    if qd.static(skip_settled_islands):
        if constraint_state.island.n_islands[i_b] > 1:
            i_island = constraint_state.island.constraint_island_idx[i_c, i_b]
            if i_island >= 0:
                is_moving = constraint_state.island.improved[i_island, i_b] != 0
    return is_moving


@qd.func
def func_is_dof_moving(i_d, i_b, constraint_state: array_class.ConstraintState, skip_settled_islands: qd.template()):
    """Whether dof i_d belongs to an island still iterating, see func_is_row_moving."""
    is_moving = True
    if qd.static(skip_settled_islands):
        if constraint_state.island.n_islands[i_b] > 1:
            i_island = constraint_state.island.dofs_island_idx[i_d, i_b]
            is_moving = constraint_state.island.improved[i_island, i_b] != 0
    return is_moving


@qd.func
def func_qfrc_scatter_sparse(i_b, constraint_state: array_class.ConstraintState, skip_settled_islands: qd.template()):
    """qfrc_constraint = J^T @ efc_force of one env by scattering each moving row over its sparse support
    (jac_dofs_idx), the dofs of the moving islands cleared first (see func_is_row_moving)."""
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    for i_d in range(n_dofs):
        if func_is_dof_moving(i_d, i_b, constraint_state, skip_settled_islands):
            constraint_state.qfrc_constraint[i_d, i_b] = gs.qd_float(0.0)
    for i_c in range(constraint_state.n_constraints[i_b]):
        if func_is_row_moving(i_c, i_b, constraint_state, skip_settled_islands):
            for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                constraint_state.qfrc_constraint[i_d, i_b] = (
                    constraint_state.qfrc_constraint[i_d, i_b]
                    + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                )


@qd.func
def func_qfrc_gather_dense(i_b, constraint_state: array_class.ConstraintState):
    """qfrc_constraint = J^T @ efc_force of one env by gathering every dof over every row."""
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    for i_d in range(n_dofs):
        qfrc_constraint = gs.qd_float(0.0)
        for i_c in range(constraint_state.n_constraints[i_b]):
            qfrc_constraint = (
                qfrc_constraint + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            )
        constraint_state.qfrc_constraint[i_d, i_b] = qfrc_constraint


@qd.func
def func_update_constraint_batch(
    i_b,
    qacc: qd.Tensor,
    Ma: qd.Tensor,
    cost: qd.Tensor,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
    skip_settled_islands: qd.template() = False,
):
    """Active flags, constraint forces, qfrc_constraint and cost of one env from its current Jaref.

    Under skip_settled_islands the rows and dofs of an island that stands still (see improved in IslandState) keep
    their values: its Jaref is frozen, so recomputing them gives the same result, and the iterations charge an env only
    for the islands still moving. The seed leaves it False, the island labels being resolved after this pass there.
    The cost then sums the visited rows and dofs, which the iterations read nowhere; only the seed compares costs."""
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    ncone = nef
    if qd.static(rigid_config.enable_elliptic_friction):
        ncone = ncone + constraint_state.n_constraints_cone[i_b]

    cost_i = gs.qd_float(0.0)

    # Snapshot the previous active set in a separate pass BEFORE any active is recomputed: a coupled elliptic-cone
    # head writes active for its two tangent rows, so capturing prev_active inline (per row, in the recompute loop)
    # would read a tangent row's already-updated value once the head ran first, hiding its flip from the incremental
    # factor's changed-constraint list. Pyramidal rows only write their own active, so they keep the fused inline
    # snapshot below.
    if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton and rigid_config.enable_elliptic_friction):
        for i_c in range(constraint_state.n_constraints[i_b]):
            constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

    # Beware 'active' does not refer to whether a constraint is active, but rather whether its quadratic cost is active
    for i_c in range(constraint_state.n_constraints[i_b]):
        is_cone_row = False
        if qd.static(rigid_config.enable_elliptic_friction):
            is_cone_row = nef <= i_c and i_c < ncone
        if not func_is_row_moving(i_c, i_b, constraint_state, skip_settled_islands):
            # A settled row keeps its values and shows no flip to the incremental factor, whose changed-row scan runs
            # over the whole env.
            if qd.static(
                rigid_config.solver_type == gs.constraint_solver.Newton and not rigid_config.enable_elliptic_friction
            ):
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
        elif is_cone_row:
            # Elliptic cone contact: the coupled rows are resolved at the head row, which also writes the friction
            # rows.
            if (i_c - nef) % qd.static(rigid_config.rows_per_contact) == 0:
                cost_i = cost_i + func_cone_update_rows(i_c, i_b, constraint_state, rigid_config)
        else:
            if qd.static(
                rigid_config.solver_type == gs.constraint_solver.Newton and not rigid_config.enable_elliptic_friction
            ):
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
            constraint_state.active[i_c, i_b] = True
            floss_force = gs.qd_float(0.0)
            if ne <= i_c and i_c < nef:  # Friction constraints
                f = constraint_state.efc_frictionloss[i_c, i_b]
                r = constraint_state.diag[i_c, i_b]
                rf = r * f
                linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
                linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
                constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
                floss_force = linear_neg * f + linear_pos * -f
                floss_cost_local = linear_neg * f * (-0.5 * rf - constraint_state.Jaref[i_c, i_b])
                floss_cost_local = floss_cost_local + linear_pos * f * (-0.5 * rf + constraint_state.Jaref[i_c, i_b])
                cost_i = cost_i + floss_cost_local
            elif nef <= i_c:  # Contact / joint-limit constraints (unilateral)
                constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0

            constraint_state.efc_force[i_c, i_b] = floss_force + (
                -constraint_state.Jaref[i_c, i_b] * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
            )

    # qfrc_constraint = J^T @ efc_force. The CPU skyline solve scatters each row over its sparse support, and so does an
    # env holding several islands, its cost following the islands' sizes, while one island spanning the env gathers
    # every dof over every row, whose loads carry no dependent index.
    is_gathered = constraint_state.island.n_islands[i_b] == 1
    if qd.static(rigid_config.sparse_solve):
        is_gathered = False
    if is_gathered:
        func_qfrc_gather_dense(i_b, constraint_state)
    else:
        func_qfrc_scatter_sparse(i_b, constraint_state, skip_settled_islands)

    # (Mx - Mx') * (x - x')
    for i_d in range(n_dofs):
        if func_is_dof_moving(i_d, i_b, constraint_state, skip_settled_islands):
            v = (
                0.5
                * (Ma[i_d, i_b] - dyn_state.dofs.force[i_d, i_b])
                * (qacc[i_d, i_b] - dyn_state.dofs.acc_smooth[i_d, i_b])
            )
            cost_i = cost_i + v

    # D * (Jx - aref) ** 2
    for i_c in range(constraint_state.n_constraints[i_b]):
        if func_is_row_moving(i_c, i_b, constraint_state, skip_settled_islands):
            cost_i = cost_i + 0.5 * (
                constraint_state.Jaref[i_c, i_b] ** 2
                * constraint_state.efc_D[i_c, i_b]
                * constraint_state.active[i_c, i_b]
            )

    cost[i_b] = cost_i


@qd.func
def _func_update_efc_force_body(i_c, i_b, constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Compute active and write efc_force for one (constraint, env) pair.

    Same semantics as the per-constraint loop in ``func_update_constraint_batch`` (lines computing ``active``,
    ``floss_force``, ``efc_force``). Friction cost contribution is *not* accumulated here; it's recomputed in
    ``_func_update_cost_coop`` together with the quadratic term to avoid an extra atomic or shared-memory exchange
    between kernels.
    """
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    ncone = nef
    if qd.static(rigid_config.enable_elliptic_friction):
        ncone = ncone + constraint_state.n_constraints_cone[i_b]

    if qd.static(rigid_config.enable_elliptic_friction) and (nef <= i_c and i_c < ncone):
        # Elliptic cone contact (cooperative one-thread-per-row): only the head thread resolves the coupled rows
        # and writes all of them; the friction-row threads are no-ops so each row is written exactly once
        # (race-free). The coupled middle-zone cost is discarded here; _func_update_cost_coop recomputes it.
        if (i_c - nef) % qd.static(rigid_config.rows_per_contact) == 0:
            func_cone_update_rows(i_c, i_b, constraint_state, rigid_config)
    else:
        if qd.static(
            rigid_config.solver_type == gs.constraint_solver.Newton and not rigid_config.enable_elliptic_friction
        ):
            constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
        constraint_state.active[i_c, i_b] = True
        floss_force = gs.qd_float(0.0)
        if ne <= i_c and i_c < nef:
            f = constraint_state.efc_frictionloss[i_c, i_b]
            r = constraint_state.diag[i_c, i_b]
            rf = r * f
            linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
            linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
            constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
            floss_force = linear_neg * f + linear_pos * -f
        elif nef <= i_c:
            constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0

        constraint_state.efc_force[i_c, i_b] = floss_force + (
            -constraint_state.Jaref[i_c, i_b] * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
        )


@qd.func
def _func_update_efc_force(constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Compute active and efc_force for every (constraint, env) with one thread per pair (qd.ndrange-parallel).

    Iteration order picks the coalesced ndrange under each layout: under transposed jac/Jaref/efc_force, lanes vary
    i_c so adjacent reads of the flipped per-constraint tensors stride 1; under canonical, lanes vary i_b.
    """
    len_constraints = constraint_state.active.shape[0]
    _B = constraint_state.grad.shape[1]

    # Snapshot prev_active in its own parallel pass so every row is captured before any active recompute: the cone head
    # thread rewrites its two tangent rows' active, which would otherwise race the tangent threads capturing
    # prev_active. Pyramidal threads only write their own row, so they snapshot inline in the body (no extra pass).
    if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton and rigid_config.enable_elliptic_friction):
        qd.loop_config(name="snapshot_prev_active", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_c, i_b in qd.ndrange(
            len_constraints, _B, axes=qd.static((1, 0) if rigid_config.constraint_layout_batch_first else None)
        ):
            if i_c < constraint_state.n_constraints[i_b]:
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

    qd.loop_config(name="update_constraint_forces", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_c, i_b in qd.ndrange(
        len_constraints, _B, axes=qd.static((1, 0) if rigid_config.constraint_layout_batch_first else None)
    ):
        if i_c < constraint_state.n_constraints[i_b]:
            _func_update_efc_force_body(i_c, i_b, constraint_state, rigid_config)


@qd.func
def _func_update_qfrc_constraint_coop(constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Compute qfrc_constraint = J^T @ efc_force with one cooperating warp per (env, dof).

    32 lanes stride i_c so adjacent reads of jac[i_c, i_d, i_b] and efc_force[i_c, i_b] are stride-1 under the flipped
    jac and flipped efc_force layouts; each (env, dof) warp reduces over its constraints with one warp-reduce.

    Gridding over (env, dof) - one warp per dof rather than one warp per env looping all dofs - keeps the GPU busy when
    the env count alone does not fill it (a single env with many dofs leaves all but one warp idle in the per-env
    layout). The per-lane summation order is unchanged, so the result is bit-identical to the per-env loop.
    """
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]
    _K = qd.static(32)

    qd.loop_config(name="update_constraint_qfrc", block_dim=_K)
    for i_flat in range(_B * n_dofs * _K):
        tid = i_flat % _K
        work = i_flat // _K
        i_d = work % n_dofs
        i_b = work // n_dofs
        n_con = constraint_state.n_constraints[i_b]

        qfrc_lane = gs.qd_float(0.0)
        i_c = tid
        while i_c < n_con:
            qfrc_lane = qfrc_lane + constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            i_c = i_c + _K
        qfrc_total = qd.simt.subgroup.reduce_all_add_tiled(qfrc_lane, 5)
        if tid == 0:
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc_total


@qd.func
def _func_update_cost_coop(
    qacc: qd.template(),
    Ma: qd.template(),
    cost: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
):
    """Compute the linesearch cost (M-norm Gauss + quadratic constraint terms) using one cooperating warp per env.

    Inner loop over dofs (lanes stride i_d): DOF-vec family is canonical (n_dofs, _B) so reads here are *not*
    coalesced under the flipped layout, but the working set is small enough to live in cache. Inner loop over
    constraints (lanes stride i_c): coalesced under flipped Jaref/efc_D/active. One reduce_all_add_tiled per scalar at
    the end.
    """
    _B = constraint_state.grad.shape[1]
    _K = qd.static(32)

    qd.loop_config(name="update_constraint_cost", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K
        n_dofs = constraint_state.qfrc_constraint.shape[0]
        ne = constraint_state.n_constraints_equality[i_b]
        nef = ne + constraint_state.n_constraints_frictionloss[i_b]
        ncone = nef
        if qd.static(rigid_config.enable_elliptic_friction):
            ncone = ncone + constraint_state.n_constraints_cone[i_b]
        n_con = constraint_state.n_constraints[i_b]

        cost_i = gs.qd_float(0.0)

        i_d = tid
        while i_d < n_dofs:
            v = (
                0.5
                * (Ma[i_d, i_b] - dyn_state.dofs.force[i_d, i_b])
                * (qacc[i_d, i_b] - dyn_state.dofs.acc_smooth[i_d, i_b])
            )
            cost_i = cost_i + v
            i_d = i_d + _K

        i_c = tid
        while i_c < n_con:
            Jaref_c = constraint_state.Jaref[i_c, i_b]
            cost_i = cost_i + 0.5 * (
                Jaref_c * Jaref_c * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
            )
            if ne <= i_c and i_c < nef:
                f = constraint_state.efc_frictionloss[i_c, i_b]
                r = constraint_state.diag[i_c, i_b]
                rf = r * f
                linear_neg = Jaref_c <= -rf
                linear_pos = Jaref_c >= rf
                cost_i = cost_i + linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
            if qd.static(rigid_config.enable_elliptic_friction) and (
                nef <= i_c and i_c < ncone and (i_c - nef) % qd.static(rigid_config.rows_per_contact) == 0
            ):
                # Middle-zone cone: add the coupled cost at the head only (per-row quadratics of bottom-zone rows are
                # covered by the active-masked term above; top/middle rows are inactive there).
                cost_i = cost_i + func_cone_middle_cost(i_c, i_b, constraint_state, rigid_config)
            i_c = i_c + _K

        cost_i = qd.simt.subgroup.reduce_all_add_tiled(cost_i, 5)

        if tid == 0:
            cost[i_b] = cost_i


@qd.func
def func_update_constraint(
    qacc: qd.Tensor,
    Ma: qd.Tensor,
    cost: qd.Tensor,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
):
    """Compute active / efc_force / qfrc_constraint / cost.

    Under ``enable_cooperative_constraint_kernels=True`` we run three sub-kernels (``_func_update_efc_force``,
    ``_func_update_qfrc_constraint_coop``, ``_func_update_cost_coop``) so per-constraint reads/writes coalesce against
    the flipped jac and Tier-1 constraint-state tensors. Under canonical we keep the original 1-thread-per-env loop
    (bit-identical to the previous code path). The transpose heuristic disables the flip entirely under sparse_solve,
    so sparse runs always take the canonical path here.
    """
    if qd.static(rigid_config.enable_cooperative_constraint_kernels):
        _func_update_efc_force(constraint_state, rigid_config)
        _func_update_qfrc_constraint_coop(constraint_state, rigid_config)
        _func_update_cost_coop(qacc, Ma, cost, dyn_state, constraint_state, rigid_config)
    else:
        _B = constraint_state.jac.shape[2]
        qd.loop_config(name="update_constraint", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            func_update_constraint_batch(i_b, qacc, Ma, cost, dyn_state, constraint_state, rigid_config)


@qd.func
def func_update_gradient_batch(
    i_b,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_dofs = constraint_state.grad.shape[0]

    # The gradient of an island standing still is kept (see improved in IslandState), and a hibernated island carries
    # a zero gradient and search direction, see func_island_tiled_factor_solve_all. An env holding one island that
    # iterates moves as a whole and needs no lookup per dof.
    if constraint_state.island.n_islands[i_b] == 1 and constraint_state.island.improved[0, i_b]:
        for i_d in range(n_dofs):
            constraint_state.grad[i_d, i_b] = (
                constraint_state.Ma[i_d, i_b]
                - dyn_state.dofs.force[i_d, i_b]
                - constraint_state.qfrc_constraint[i_d, i_b]
            )
    else:
        for i_d in range(n_dofs):
            i_island = constraint_state.island.dofs_island_idx[i_d, i_b]
            if constraint_state.island.improved[i_island, i_b]:
                constraint_state.grad[i_d, i_b] = (
                    constraint_state.Ma[i_d, i_b]
                    - dyn_state.dofs.force[i_d, i_b]
                    - constraint_state.qfrc_constraint[i_d, i_b]
                )
            elif qd.static(rigid_config.use_hibernation):
                if constraint_state.island.is_hibernated[i_island, i_b]:
                    constraint_state.grad[i_d, i_b] = gs.qd_float(0.0)
                    constraint_state.Mgrad[i_d, i_b] = gs.qd_float(0.0)

    if qd.static(rigid_config.solver_type == gs.constraint_solver.CG):
        func_solve_mass_batch(
            i_b, constraint_state.grad, constraint_state.Mgrad, dyn_state, dyn_info, rigid_info, rigid_config
        )

    if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton):
        # Mgrad = H^{-1} @ grad solved per island on each island's block (factored above)
        for i_island in range(constraint_state.island.n_islands[i_b]):
            if constraint_state.island.improved[i_island, i_b]:
                func_cholesky_solve_batch(
                    i_b,
                    i_island,
                    rhs=constraint_state.grad,
                    out=constraint_state.Mgrad,
                    constraint_state=constraint_state,
                    rigid_config=rigid_config,
                )


@qd.func
def func_update_gradient_no_solve(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Compute the gradient only (no Cholesky solve), used with a fused factor+solve that consumes grad directly.

    Under enable_cooperative_constraint_kernels the ndrange is swapped so adjacent lanes vary i_d - 3 of 4 in-loop
    accesses (grad, Ma, qfrc_constraint) are DOF-vec flipped; only dyn_state.dofs.force stays canonical.
    """
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.grad.shape[0]
    qd.loop_config(name="update_gradient_no_solve", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if rigid_config.enable_cooperative_constraint_kernels else None)
    ):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            # The gradient of an island standing still is kept (see improved in IslandState), and a hibernated island
            # carries a zero gradient, see func_island_tiled_factor_solve_all.
            i_island = constraint_state.island.dofs_island_idx[i_d, i_b]
            if constraint_state.island.improved[i_island, i_b]:
                constraint_state.grad[i_d, i_b] = (
                    constraint_state.Ma[i_d, i_b]
                    - dyn_state.dofs.force[i_d, i_b]
                    - constraint_state.qfrc_constraint[i_d, i_b]
                )
            elif qd.static(rigid_config.use_hibernation):
                if constraint_state.island.is_hibernated[i_island, i_b]:
                    constraint_state.grad[i_d, i_b] = gs.qd_float(0.0)


@qd.func
def func_update_gradient(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Compute the gradient of every env and its solve (Mgrad = H^-1 grad for Newton, M^-1 grad for CG), one
    thread per env."""
    _B = constraint_state.jac.shape[2]

    qd.loop_config(name="update_gradient", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        func_update_gradient_batch(i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)


@qd.func
def initialize_Jaref(qacc: qd.Tensor, constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    if qd.static(rigid_config.parallel_init):
        _initialize_Jaref_parallel(qacc, constraint_state, rigid_config)
    else:
        _initialize_Jaref_per_env(qacc, constraint_state, rigid_config)


@qd.func
def _initialize_Jaref_body(
    i_c, i_b, n_dofs, qacc: qd.template(), constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    Jaref = -constraint_state.aref[i_c, i_b]
    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
        Jaref = Jaref + constraint_state.jac[i_c, i_d, i_b] * qacc[i_d, i_b]
    constraint_state.Jaref[i_c, i_b] = Jaref


@qd.func
def _initialize_Jaref_per_env(
    qacc: qd.template(), constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]

    qd.loop_config(name="init_jaref", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_c in range(constraint_state.n_constraints[i_b]):
            _initialize_Jaref_body(i_c, i_b, n_dofs, qacc, constraint_state, rigid_config)


@qd.func
def _initialize_Jaref_parallel(
    qacc: qd.template(), constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    """Initialize Jaref = J @ qacc, parallelised over (constraint, env)."""
    _B = constraint_state.jac.shape[2]
    n_dofs = constraint_state.jac.shape[1]
    len_constraints = constraint_state.Jaref.shape[0]

    # Innermost ndrange axis matches the stride-1 axis of jac so jac loads coalesce: i_c-innermost under the flipped
    # layout, i_b-innermost under canonical.
    qd.loop_config(name="init_jaref_parallel", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_c, i_b in qd.ndrange(
        len_constraints, _B, axes=qd.static((1, 0) if rigid_config.constraint_layout_batch_first else None)
    ):
        if i_c < constraint_state.n_constraints[i_b]:
            _initialize_Jaref_body(i_c, i_b, n_dofs, qacc, constraint_state, rigid_config)


@qd.func
def initialize_Ma(
    Ma: qd.Tensor,
    qacc: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    _B = rigid_info.mass_mat.shape[2]
    n_dofs = qacc.shape[0]

    # Flipped mass_mat layout=(2,1,0): physical (_B, n_dofs, n_dofs) with i_d1 stride-1. Make i_d1 the innermost
    # ndrange axis so adjacent lanes vary i_d1 -> coalesced reads of mass_mat[i_d1, i_d2, i_b]. qacc[i_d2, i_b] is
    # constant within the warp -> broadcast load.
    qd.loop_config(name="init_ma", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d1, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if rigid_config.constraint_layout_batch_first else None)
    ):
        Ma_ = gs.qd_float(0.0)
        # Mass couples only DOFs within the same kinematic-tree block, so restrict to i_d1's block (cross-block is
        # zero).
        for i_d2 in range(rigid_info.dofs_mass_block_start[i_d1], rigid_info.dofs_mass_block_end[i_d1]):
            Ma_ = Ma_ + rigid_info.mass_mat[i_d1, i_d2, i_b] * qacc[i_d2, i_b]
        Ma[i_d1, i_b] = Ma_


# ======================================================= Core ========================================================


@qd.kernel(fastcache=True)
def func_solve_init(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_decomposed: qd.template(),
):
    # is_decomposed is a hardcoded constant forwarded by the dispatch entrypoint that calls this (the decomposed arm
    # passes True, the monolith passes False). func_solve_init runs as a separate kernel before the perf-dispatcher
    # picks an arm, so it CANNOT detect the arm itself - the entrypoint must declare it. The decomposed arm rebuilds
    # the Hessian on its first graph iteration regardless, so it skips the init factor/gradient here entirely.
    _B = dyn_state.dofs.acc_smooth.shape[1]
    n_dofs = dyn_state.dofs.acc_smooth.shape[0]

    # The one arm whose factor, gradient and convergence certificate are all seeded inside its own body (see
    # _kernel_solve_monolith) rather than here: the GPU per-island monolith with the cooperative kernels off
    # self-inits per env, so every seed in this kernel routes around it. The perf dispatcher may serve the same
    # simulation with either arm from one step to the next, so this predicate is a property of the entrypoint that
    # launched this init (via is_decomposed), evaluated per instantiation.
    is_self_seeding = qd.static(
        rigid_config.solver_type == gs.constraint_solver.Newton
        and not is_decomposed
        and rigid_config.backend != gs.cpu
        and not rigid_config.enable_cooperative_constraint_kernels
    )

    if qd.static(rigid_config.enable_mujoco_compatibility):
        # Compute cost for warmstart state (i.e. acceleration at previous timestep)
        initialize_Ma(constraint_state.Ma_ws, constraint_state.qacc_ws, dyn_info, rigid_info, rigid_config)

        # Keyword calls: passing a struct member positionally alongside its parent struct breaks quadrants'
        # func-argument expansion (the member is duplicated in the flattened call).
        initialize_Jaref(qacc=constraint_state.qacc_ws, constraint_state=constraint_state, rigid_config=rigid_config)
        func_update_constraint(
            qacc=constraint_state.qacc_ws,
            Ma=constraint_state.Ma_ws,
            cost=constraint_state.cost_ws,
            dyn_state=dyn_state,
            constraint_state=constraint_state,
            rigid_config=rigid_config,
        )

        # Compute cost for current state (assuming constraint-free acceleration)
        initialize_Ma(constraint_state.Ma, dyn_state.dofs.acc_smooth, dyn_info, rigid_info, rigid_config)

        initialize_Jaref(dyn_state.dofs.acc_smooth, constraint_state, rigid_config)
        # Keyword call: see the quadrants member-expansion note above.
        func_update_constraint(
            qacc=dyn_state.dofs.acc_smooth,
            Ma=constraint_state.Ma,
            cost=constraint_state.cost,
            dyn_state=dyn_state,
            constraint_state=constraint_state,
            rigid_config=rigid_config,
        )

        # Pick the best starting point between current state and warmstart
        qd.loop_config(name="solve_init_pick_warmstart", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d, i_b in qd.ndrange(n_dofs, _B):
            if constraint_state.cost_ws[i_b] < constraint_state.cost[i_b]:
                constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
                constraint_state.Ma[i_d, i_b] = constraint_state.Ma_ws[i_d, i_b]
            else:
                constraint_state.qacc[i_d, i_b] = dyn_state.dofs.acc_smooth[i_d, i_b]
    else:
        # Always initialize from warmstart.
        # Under the DOF-vec flip, both qacc and qacc_ws are env-leading; swap the ndrange so adjacent lanes vary i_d
        # to coalesce those writes/reads. The dofs_state.acc_smooth read remains canonical (small per-env working
        # set, dominated by the qacc write).
        qd.loop_config(name="from_warmstart", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d, i_b in qd.ndrange(
            n_dofs, _B, axes=qd.static((1, 0) if rigid_config.constraint_layout_batch_first else None)
        ):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.is_warmstart[i_b]:
                constraint_state.qacc[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
            else:
                constraint_state.qacc[i_d, i_b] = dyn_state.dofs.acc_smooth[i_d, i_b]

        initialize_Ma(constraint_state.Ma, constraint_state.qacc, dyn_info, rigid_info, rigid_config)

    # Initialize solver accordingly Keyword calls: see the quadrants member-expansion note in func_solve_init.
    initialize_Jaref(qacc=constraint_state.qacc, constraint_state=constraint_state, rigid_config=rigid_config)
    func_update_constraint(
        qacc=constraint_state.qacc,
        Ma=constraint_state.Ma,
        cost=constraint_state.cost,
        dyn_state=dyn_state,
        constraint_state=constraint_state,
        rigid_config=rigid_config,
    )

    # The island partition itself (links_island_idx / dof_id) is built earlier, in add_inequality_constraints, before
    # the contact constraints are assembled; grouping the constraints by island needs the assembled jac, so it rides
    # this per-env loop, ahead of every per-island consumer below: by the lanes of the env's block where the
    # cooperative kernels run, serially otherwise.
    if qd.static(rigid_config.enable_cooperative_constraint_kernels):
        _K = qd.static(32)
        qd.loop_config(name="init_improved", block_dim=_K)
        for i_flat in range(_B * _K):
            tid = i_flat % _K
            i_b = i_flat // _K
            sh_chunk = qd.simt.block.SharedArray((_K,), gs.qd_int)
            if tid == 0:
                constraint_state.improved[i_b] = constraint_state.n_constraints[i_b] > 0
            func_group_constraints_by_island_coop(i_b, tid, sh_chunk, constraint_state, rigid_config)
    else:
        qd.loop_config(name="init_improved", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in qd.ndrange(_B):
            constraint_state.improved[i_b] = constraint_state.n_constraints[i_b] > 0
            func_group_constraints_by_island(i_b, constraint_state, rigid_config)
    constraint_state.solver_iter_counter[()] = 0

    if qd.static(
        rigid_config.solver_type == gs.constraint_solver.Newton and rigid_config.enable_cooperative_constraint_kernels
    ):
        # The cooperative seed: every island's Hessian block assembled into nt_H, then factored and solved in its shared
        # tile, the same barrier-free factor the decomposed graph runs every iteration. The factor reads nt_H without
        # consuming it, so the graph starts from the assembled Hessian and maintains it from its first iteration on
        # (see _kernel_solve_graph), the coupled elliptic-cone block bracketed around the factor as the graph does. The
        # monolith reads L back from nt_H in its incremental iterations, so it persists L instead (write_L=True).
        func_island_hessian_assemble_all(constraint_state, rigid_info, rigid_config)
        func_update_gradient_no_solve(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
        func_wrap_cone_hessian(constraint_state, rigid_config, is_removal=False)
        func_island_tiled_factor_solve_all(
            constraint_state, dyn_info, rigid_info, rigid_config, write_L=qd.static(not is_decomposed)
        )
        if qd.static(is_decomposed):
            func_wrap_cone_hessian(constraint_state, rigid_config, is_removal=True)
    else:
        if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton and not is_self_seeding):
            # Seed the initial Hessian factor. The decomposed arm has no self-init: its graph is linesearch-first, so
            # its first linesearch consumes the search direction computed here (this kernel is its "iteration 0"; the
            # graph then computes each subsequent direction at the end of an iteration). The monolith seeds it here
            # except where its body self-inits the factor per env (is_self_seeding).
            # compute_envelope=True computes each island's structural skyline envelope once, reused per iteration.
            func_hessian_and_cholesky_factor_direct(
                constraint_state, dyn_info, rigid_info, rigid_config, compute_envelope=True
            )

        if qd.static(not is_self_seeding):
            # Initial gradient (Mgrad = H^-1 grad for Newton, grad for CG), the decomposed arm's first search direction
            func_update_gradient(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)

    qd.loop_config(name="assign_search", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if rigid_config.constraint_layout_batch_first else None)
    ):
        constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]

    if qd.static(
        not rigid_config.requires_grad and not rigid_config.enable_mujoco_compatibility and not is_self_seeding
    ):
        # The certificate consumes the seed gradient, so the one arm that self-seeds it in its own body certifies
        # there instead (see _kernel_solve_monolith). The differentiable path keeps its full iteration trace.
        # TODO: Remove the MuJoCo compatibility gate once on MuJoCo 3.11, which carries this criterion natively.
        if qd.static(rigid_config.enable_cooperative_constraint_kernels):
            _K = qd.static(32)
            qd.loop_config(name="certify_converged", block_dim=_K)
            for i_flat in range(_B * _K):
                tid = i_flat % _K
                i_b = i_flat // _K
                sh_acc = qd.simt.block.SharedArray((7 * _K,), gs.qd_float)
                sh_pending = qd.simt.block.SharedArray((_K,), gs.qd_int)
                sh_alpha = qd.simt.block.SharedArray((_K,), gs.qd_float)
                if constraint_state.n_constraints[i_b] > 0:
                    improved = linesearch.func_exit_islands_coop(
                        i_b, tid, sh_acc, sh_pending, sh_alpha, constraint_state, rigid_info, rigid_config, certify=True
                    )
                    if tid == 0:
                        constraint_state.improved[i_b] = improved
        else:
            qd.loop_config(name="certify_converged", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
            for i_b in qd.ndrange(_B):
                if constraint_state.n_constraints[i_b] > 0:
                    constraint_state.improved[i_b] = linesearch.func_exit_islands_serial(
                        i_b, constraint_state, rigid_info, rigid_config, certify=True
                    )


@qd.func
def func_solve_iter(
    i_b,
    it,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """One Newton / CG iteration of one env, every pass confined to the islands still iterating.

    The iteration runs the line search and step, the constraint update, the Hessian factor, the gradient, the exit test
    and the next search direction. A converged island stands still and costs nothing while the others iterate on (see
    improved in IslandState). The line search and the exit test run per island in lockstep (see linesearch.py), each
    island holding its own direction, step and convergence state.
    """
    constraint_state.improved[i_b] = linesearch.func_linesearch_islands_serial(
        i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config
    )

    if constraint_state.improved[i_b]:
        # Keyword call: see the quadrants member-expansion note in func_solve_init.
        func_update_constraint_batch(
            i_b=i_b,
            qacc=constraint_state.qacc,
            Ma=constraint_state.Ma,
            cost=constraint_state.cost,
            dyn_state=dyn_state,
            constraint_state=constraint_state,
            rigid_config=rigid_config,
            skip_settled_islands=True,
        )

        if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton):
            # Within a step jac, M and efc_D are fixed, so H = M + J.T diag(D active) J depends only on the active mask;
            # the linesearch only moves qacc, never H. func_solve_init already seeded the factor (nt_H holds L for the
            # seed's active set, and update_constraint above set prev_active to it), so every iteration maintains it
            # by a rank-1 update/downdate per row that flipped active, a degenerate downdate falling back to a direct
            # refactor of the island. The CPU skyline path also refactors an island directly where the flips outnumber
            # what its envelope makes cheaper to update (see func_factor_island_incremental_or_direct).
            if qd.static(rigid_config.sparse_solve):
                for i_island in range(constraint_state.island.n_islands[i_b]):
                    if constraint_state.island.improved[i_island, i_b]:
                        func_factor_island_incremental_or_direct(
                            i_b, i_island, constraint_state, dyn_info, rigid_info, rigid_config
                        )
            else:
                is_degenerated = func_hessian_and_cholesky_factor_incremental_batch(
                    i_b, constraint_state, dyn_info, rigid_info, rigid_config
                )
                if is_degenerated:
                    func_hessian_and_cholesky_factor_direct_batch(
                        i_b, constraint_state, dyn_info, rigid_info, rigid_config
                    )

        func_update_gradient_batch(i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)

        constraint_state.improved[i_b] = linesearch.func_exit_islands_serial(
            i_b, constraint_state, rigid_info, rigid_config, certify=False
        )


def _get_static_config(*args, **kwargs):
    # Positional index of rigid_config in func_solve_body's signature.
    return args[4] if len(args) > 4 else kwargs["rigid_config"]


@qd.perf_dispatch(
    get_geometry_hash=lambda *args, **kwargs: (*args, frozendict(kwargs)),
    first_warmup=1,
    warmup=0,
    active=2,
    repeat_after_count=300,
    repeat_after_seconds=0,
)
def func_solve_body(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    _n_iterations: int,
) -> None: ...


@qd.kernel(fastcache=True)
def _kernel_solve_monolith(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    _n_iterations: int,
):
    _B = constraint_state.grad.shape[1]
    n_dofs = constraint_state.qacc.shape[0]

    # The monolith arm solves each env whole (32 envs packed per warp); islands change only the per-env factor's block
    # structure, handled inside func_solve_iter, not the iteration scheme. Per-island parallelism is the decomposed
    # arm's job, so there is no separate island body here - ON and OFF run the identical packed-env solve.
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        # A fully-asleep env has no awake DOF to move, so its Newton solve is a no-op. Skip the whole iteration loop
        # so step time tracks the awake set, not the total body count.
        has_awake_work = constraint_state.n_constraints[i_b] > 0
        if qd.static(rigid_config.use_hibernation):
            has_awake_work = has_awake_work and rigid_info.n_awake_dofs[i_b] > 0
        if has_awake_work:
            if qd.static(
                rigid_config.backend != gs.cpu
                and rigid_config.solver_type == gs.constraint_solver.Newton
                and not rigid_config.enable_cooperative_constraint_kernels
            ):
                # A GPU with the cooperative kernels off: func_solve_init skips its seed, so the monolith self-seeds
                # each island's scalar factor + gradient + search here (once per step). With the cooperative kernels
                # on, func_solve_init already seeded the factor (L in nt_H).
                func_hessian_and_cholesky_factor_direct_batch(
                    i_b, constraint_state, dyn_info, rigid_info, rigid_config, compute_envelope=True
                )
                func_update_gradient_batch(i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
                for i_d in range(n_dofs):
                    constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]
                if qd.static(not rigid_config.requires_grad and not rigid_config.enable_mujoco_compatibility):
                    # Certificate gating: see func_solve_init.
                    constraint_state.improved[i_b] = linesearch.func_exit_islands_serial(
                        i_b, constraint_state, rigid_info, rigid_config, certify=True
                    )
            for it in range(rigid_info.iterations[None]):
                # Checked at the top so a solve certified converged on its seed runs zero iterations.
                if not constraint_state.improved[i_b]:
                    break
                func_solve_iter(i_b, it, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
        else:
            constraint_state.improved[i_b] = False


@func_solve_body.register(
    # Runs whenever the decomposed arm is not specifically preferred. Solves each env whole with 32 envs packed per
    # warp; islands only reshape the per-env factor into block-diagonal blocks, they do not change the solve scheme.
    is_compatible=lambda *args, **kwargs: (
        (rigid_config := _get_static_config(*args, **kwargs)).prefer_decomposed_solver != 1
    )
)
def func_solve_body_monolith(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, _n_iterations):
    # This entrypoint statically IS the monolith arm, so it owns its init: it forwards is_decomposed=False to
    # func_solve_init (which groups the constraints by island, factors, and seeds the gradient the packed-env body
    # consumes), then runs the solve kernel. Keeping the init inside the entrypoint (rather than in resolve, before the
    # dispatch) is what lets each arm declare its own init behavior - the dispatcher may run a different arm on the next
    # step during autotuning.
    func_solve_init(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_decomposed=False)
    _kernel_solve_monolith(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, _n_iterations)


# =====================================================================================================================
# ==================================================== Finalization ===================================================
# =====================================================================================================================


@qd.kernel(fastcache=True)
def func_update_contact_force(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    n_links = dyn_state.links.contact_force.shape[0]
    _B = dyn_state.links.contact_force.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_l, i_b in qd.ndrange(n_links, _B):
        dyn_state.links.contact_force[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        const_start = constraint_state.n_constraints_equality[i_b] + constraint_state.n_constraints_frictionloss[i_b]

        # contact constraints should be after equality and frictionloss constraints and before joint limit constraints
        for i_c in range(collider_state.n_contacts[i_b]):
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            contact_data_normal = collider_state.contact_data.normal[i_col, i_b]
            contact_data_friction = collider_state.contact_data.friction[i_col, i_b]
            contact_data_link_a = collider_state.contact_data.link_a[i_col, i_b]
            contact_data_link_b = collider_state.contact_data.link_b[i_col, i_b]

            rows_per_contact = qd.static(rigid_config.rows_per_contact)
            force = qd.Vector.zero(gs.qd_float, 3)
            d1, d2 = gu.qd_orthogonals(contact_data_normal)
            if qd.static(rigid_config.enable_elliptic_friction):
                # Cone rows [normal, t1, t2(, spin)(, roll1, roll2)] contiguous in the collision segment; the spin
                # and rolling rows carry torque only, so the linear contact force sums the three translational
                # directions.
                base = i_c * rows_per_contact + const_start
                force = -contact_data_normal * constraint_state.efc_force[base, i_b]
                force = force + d1 * constraint_state.efc_force[base + 1, i_b]
                force = force + d2 * constraint_state.efc_force[base + 2, i_b]
            else:
                for i_dir in qd.static(range(4)):
                    d = (2 * (i_dir % 2) - 1) * (d1 if i_dir < 2 else d2)
                    n = d * contact_data_friction - contact_data_normal
                    force = force + n * constraint_state.efc_force[i_c * rows_per_contact + i_dir + const_start, i_b]
                # The torsional and rolling pyramid pairs mix the spin and tangent axes through the angular
                # jacobian; their linear part is the shared normal opposition.
                if qd.static(rigid_config.enable_torsional_friction):
                    for i_dir in qd.static(range(4, rows_per_contact)):
                        force = (
                            force
                            - contact_data_normal
                            * constraint_state.efc_force[i_c * rows_per_contact + i_dir + const_start, i_b]
                        )

            # An inert contact keeps the force of the last solve it took part in, so a resting sleeper keeps reporting
            # the support force it is at rest under.
            if qd.static(rigid_config.use_hibernation):
                if _is_contact_inert(contact_data_link_a, contact_data_link_b, i_b, dyn_state, dyn_info, rigid_config):
                    force = collider_state.contact_data.force[i_col, i_b]
            collider_state.contact_data.force[i_col, i_b] = force

            dyn_state.links.contact_force[contact_data_link_a, i_b] = (
                dyn_state.links.contact_force[contact_data_link_a, i_b] - force
            )
            dyn_state.links.contact_force[contact_data_link_b, i_b] = (
                dyn_state.links.contact_force[contact_data_link_b, i_b] + force
            )


@qd.kernel(fastcache=True)
def func_update_qacc(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    n_dofs = dyn_state.dofs.acc.shape[0]
    _B = dyn_state.dofs.acc.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        # A hibernated dof is left out of the solve, so its zero acceleration and last awake forces stand.
        if qd.static(rigid_config.use_hibernation):
            if dyn_state.dofs.is_hibernated[i_d, i_b]:
                continue
        dyn_state.dofs.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        dyn_state.dofs.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
        dyn_state.dofs.force[i_d, i_b] = dyn_state.dofs.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]
        constraint_state.qacc_ws[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        if qd.math.isnan(constraint_state.qacc[i_d, i_b]):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_FORCE_NAN

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        constraint_state.is_warmstart[i_b] = True


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.constraint_solver_decomp")
