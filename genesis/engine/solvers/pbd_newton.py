"""Position based dynamics (PBD) as a participant of the NewtonSolver.

The PBD particles enter the joint solve with their inertia alone. PBD projects its own constraints for the substep before
the solve, so its particles hold their start position `ipos` and the end position `pos` of the projection. The joint
solve takes the end positions as unknowns, starting at the projection x~, with the inertia energy 1/2 m |x - x~|^2 around
it in the dt^2-scaled units of the material point method (MPM) energy. PBD's constraints stay with PBD, which projects
them again the next substep, so the coupling is strong in the contact and staggered in the PBD constraints. After the
solve, a PBD particle takes the velocity of its displacement over the substep.
"""

import numpy as np

import quadrants as qd

import genesis as gs

from .newton_solver import NewtonHost, kernel_assemble_participant


@qd.data_oriented
class PbdNewtonData:
    """The PBD rows of the joint solve, per reordered PBD particle and environment, indexed [i_q, i_b].

    The vectors are the projection x~ of PBD's own substep, the gradient, the preconditioned conjugate gradient (PCG)
    residual, preconditioned residual, direction, operator image and solution, and the line-search start. `diag_inv`
    holds the 3 x 3 diagonal block of every particle, then its inverse.

    `pos_begin` and `vel_begin` hold the position and velocity of every PBD particle at the start of the substep,
    indexed [i_p, i_b] like PBD's own particles, which the particles of a failed environment return to."""

    def __init__(self, pbd, n_envs):
        self.pbd = pbd
        self.substep_dt = qd.ndarray(gs.qd_float, shape=())
        self.substep_dt.from_numpy(np.array(pbd.substep_dt, dtype=gs.np_float))
        shape = (max(pbd.n_particles, 1), n_envs)
        self.pos_begin = qd.ndarray(gs.qd_vec3, shape=shape)
        self.vel_begin = qd.ndarray(gs.qd_vec3, shape=shape)
        self.target = qd.ndarray(gs.qd_vec3, shape=shape)
        self.grad = qd.ndarray(gs.qd_vec3, shape=shape)
        self.r = qd.ndarray(gs.qd_vec3, shape=shape)
        self.z = qd.ndarray(gs.qd_vec3, shape=shape)
        self.p = qd.ndarray(gs.qd_vec3, shape=shape)
        self.ap = qd.ndarray(gs.qd_vec3, shape=shape)
        self.sol = qd.ndarray(gs.qd_vec3, shape=shape)
        self.x_prev = qd.ndarray(gs.qd_vec3, shape=shape)
        self.diag_inv = qd.ndarray(gs.qd_mat3, shape=shape)


# ================================================ Layer 0: access ================================================


@qd.func
def is_moving(i_q: int, i_b: int, pbd: qd.template()):
    """Whether a PBD particle is an unknown of the solve: active and free, an attached particle keeping its position."""
    return pbd.particles_ng_reordered[i_q, i_b].active and pbd.particles_reordered[i_q, i_b].free


# ============================================== Layer 1: the PBD block ==============================================


@qd.func(requires_top_level=True)
def func_pbd_begin(pbd_data: qd.template()):
    """The projection of PBD's substep as the target of the inertia, the end positions starting at it."""
    qd.loop_config(name="mpm_pbd_begin")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        pbd_data.target[i_q, i_b] = pbd_data.pbd.particles_reordered[i_q, i_b].pos


@qd.func(requires_top_level=True)
def func_pbd_gradient(pbd_data: qd.template()):
    """The gradient m (x - x~) of the inertia and its diagonal block m I, before the contact rows add theirs."""
    qd.loop_config(name="mpm_pbd_gradient")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        grad = qd.Vector.zero(gs.qd_float, 3)
        m = gs.qd_float(1.0)
        if is_moving(i_q, i_b, pbd_data.pbd):
            m = pbd_data.pbd.particles_info_reordered[i_q, i_b].mass
            grad = m * (pbd_data.pbd.particles_reordered[i_q, i_b].pos - pbd_data.target[i_q, i_b])
        pbd_data.grad[i_q, i_b] = grad
        pbd_data.diag_inv[i_q, i_b] = m * qd.Matrix.identity(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_pbd_factor(pbd_data: qd.template()):
    """The inverse of every particle's diagonal block, once the contact rows added theirs."""
    qd.loop_config(name="mpm_pbd_factor")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        pbd_data.diag_inv[i_q, i_b] = pbd_data.diag_inv[i_q, i_b].inverse()


@qd.func(requires_top_level=True)
def func_pbd_pcg_init(pbd_data: qd.template(), pcg_scalars: qd.template(), is_adjoint: qd.template()):
    """The PBD part of the preconditioned conjugate gradient (PCG) start from 0: r = grad, solution and Ap cleared.

    With `is_adjoint`, an environment left out of the solve keeps its solution (see func_init_linear_pcg_residual)."""
    qd.loop_config(name="mpm_pbd_pcg_init")
    for i_q, i_b in qd.ndrange(pbd_data.r.shape[0], pbd_data.r.shape[1]):
        pbd_data.r[i_q, i_b] = pbd_data.grad[i_q, i_b]
        is_kept = False
        if qd.static(is_adjoint):
            is_kept = pcg_scalars.is_converged[i_b] != 0
        if not is_kept:
            pbd_data.sol[i_q, i_b] = qd.Vector.zero(gs.qd_float, 3)
        pbd_data.ap[i_q, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_pbd_precondition(pbd_data: qd.template()):
    """z = M^-1 r by the inverse diagonal blocks, with z = 0 for an attached particle so that it keeps its position."""
    qd.loop_config(name="mpm_pbd_precondition")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        z = qd.Vector.zero(gs.qd_float, 3)
        if is_moving(i_q, i_b, pbd_data.pbd):
            z = pbd_data.diag_inv[i_q, i_b] @ pbd_data.r[i_q, i_b]
        pbd_data.z[i_q, i_b] = z


@qd.func(requires_top_level=True)
def func_pbd_pcg_init_direction(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Set the first search direction p = z over the PBD rows, and add r . z to rz."""
    qd.loop_config(name="mpm_pbd_pcg_init_direction")
    for i_q, i_b in qd.ndrange(pbd_data.r.shape[0], pbd_data.r.shape[1]):
        pbd_data.p[i_q, i_b] = pbd_data.z[i_q, i_b]
        qd.atomic_add(pcg_scalars.rz[i_b], pbd_data.r[i_q, i_b].dot(pbd_data.z[i_q, i_b]))


@qd.func(requires_top_level=True)
def func_pbd_matvec(pbd_data: qd.template()):
    """Ap = m p over the PBD rows, before the contact rows add their term."""
    qd.loop_config(name="mpm_pbd_matvec")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        m = gs.qd_float(0.0)
        if is_moving(i_q, i_b, pbd_data.pbd):
            m = pbd_data.pbd.particles_info_reordered[i_q, i_b].mass
        pbd_data.ap[i_q, i_b] = m * pbd_data.p[i_q, i_b]


@qd.func(requires_top_level=True)
def func_pbd_matvec_dot(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Add p . Ap over the PBD rows to p . Ap, once Ap holds the whole joint operator."""
    qd.loop_config(name="mpm_pbd_matvec_dot")
    for i_q, i_b in qd.ndrange(pbd_data.p.shape[0], pbd_data.p.shape[1]):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.p_dot_ap[i_b], pbd_data.p[i_q, i_b].dot(pbd_data.ap[i_q, i_b]))


@qd.func(requires_top_level=True)
def func_pbd_pcg_update_xr(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Update the solution and the residual over the PBD rows, x += alpha p and r -= alpha Ap."""
    qd.loop_config(name="mpm_pbd_pcg_update_xr")
    for i_q, i_b in qd.ndrange(pbd_data.p.shape[0], pbd_data.p.shape[1]):
        if pcg_scalars.is_converged[i_b] == 0:
            alpha = pcg_scalars.alpha[i_b]
            pbd_data.sol[i_q, i_b] += alpha * pbd_data.p[i_q, i_b]
            pbd_data.r[i_q, i_b] -= alpha * pbd_data.ap[i_q, i_b]


@qd.func(requires_top_level=True)
def func_pbd_pcg_dot_rz(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Add r . z over the PBD rows to rz_new."""
    qd.loop_config(name="mpm_pbd_pcg_dot_rz")
    for i_q, i_b in qd.ndrange(pbd_data.r.shape[0], pbd_data.r.shape[1]):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.rz_new[i_b], pbd_data.r[i_q, i_b].dot(pbd_data.z[i_q, i_b]))


@qd.func(requires_top_level=True)
def func_pbd_pcg_direction(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Update the search direction p = z + beta p over the PBD rows, and clear Ap."""
    qd.loop_config(name="mpm_pbd_pcg_direction")
    for i_q, i_b in qd.ndrange(pbd_data.p.shape[0], pbd_data.p.shape[1]):
        pbd_data.ap[i_q, i_b] = qd.Vector.zero(gs.qd_float, 3)
        if pcg_scalars.is_converged[i_b] == 0:
            pbd_data.p[i_q, i_b] = pbd_data.z[i_q, i_b] + pcg_scalars.beta[i_b] * pbd_data.p[i_q, i_b]


@qd.func(requires_top_level=True)
def func_pbd_start_line_search(pbd_data: qd.template(), control: qd.template()):
    """The line-search start point, and the largest component of the step -x into the Newton measure."""
    qd.loop_config(name="mpm_pbd_start_line_search")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        pbd_data.x_prev[i_q, i_b] = pbd_data.pbd.particles_reordered[i_q, i_b].pos
        qd.atomic_max(control.max_disp[i_b], qd.abs(pbd_data.sol[i_q, i_b]).max())


@qd.func(requires_top_level=True)
def func_pbd_step_forward(pbd_data: qd.template(), control: qd.template()):
    qd.loop_config(name="mpm_pbd_step_forward")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        if is_moving(i_q, i_b, pbd_data.pbd):
            step = control.alpha[i_b] * pbd_data.sol[i_q, i_b]
            pbd_data.pbd.particles_reordered[i_q, i_b].pos = pbd_data.x_prev[i_q, i_b] - step


@qd.func(requires_top_level=True)
def func_pbd_energy(pbd_data: qd.template(), control: qd.template(), is_trial: qd.template()):
    """Add the inertia 1/2 m |x - x~|^2 of the PBD particles to the start or trial energy."""
    qd.loop_config(name="mpm_pbd_energy")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        if is_moving(i_q, i_b, pbd_data.pbd):
            diff = pbd_data.pbd.particles_reordered[i_q, i_b].pos - pbd_data.target[i_q, i_b]
            energy = 0.5 * pbd_data.pbd.particles_info_reordered[i_q, i_b].mass * diff.dot(diff)
            if is_trial:
                qd.atomic_add(control.energy_trial[i_b], energy)
            else:
                qd.atomic_add(control.energy_start[i_b], energy)


@qd.func(requires_top_level=True)
def func_pbd_finish(dt: float, pbd_data: qd.template(), control: qd.template()):
    """The velocity of every moving PBD particle from its displacement over the substep, then the position and velocity
    of the substep start for every particle of an environment whose solve failed."""
    qd.loop_config(name="mpm_pbd_finish")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        if is_moving(i_q, i_b, pbd_data.pbd):
            pbd_data.pbd.particles_reordered[i_q, i_b].vel = (
                pbd_data.pbd.particles_reordered[i_q, i_b].pos - pbd_data.pbd.particles_reordered[i_q, i_b].ipos
            ) / dt

    qd.loop_config(name="mpm_pbd_freeze")
    for i_p, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        if control.has_failed_non_finite[i_b] != 0 and pbd_data.pbd.particles_ng[i_p, i_b].active:
            i_q = pbd_data.pbd.particles_ng[i_p, i_b].reordered_idx
            pbd_data.pbd.particles_reordered[i_q, i_b].pos = pbd_data.pos_begin[i_p, i_b]
            pbd_data.pbd.particles_reordered[i_q, i_b].vel = pbd_data.vel_begin[i_p, i_b]


# =============================================== Layer 2: participant ===============================================


@qd.data_oriented
class PbdNewtonParticipant:
    """Device stages of the PBD end positions in the joint solve, which own the PBD data of the solve.

    It is the one place that selects the PBD arrays the couplings read, the particles in PBD's reordered layout. They
    are properties of the solver in `pbd_data`, so that the walk over the data of the NewtonSolver (see
    iter_data_oriented) stops at the solver, which owns them."""

    def __init__(self, pbd, n_envs: int):
        self.pbd_data = PbdNewtonData(pbd, n_envs)

    @property
    def n_particles(self):
        return self.pbd_data.pbd.n_particles

    @property
    def particle_radius(self):
        return self.pbd_data.pbd.particle_radius

    @property
    def particles_reordered(self):
        return self.pbd_data.pbd.particles_reordered

    @property
    def particles_ng_reordered(self):
        return self.pbd_data.pbd.particles_ng_reordered

    @property
    def particles_info_reordered(self):
        return self.pbd_data.pbd.particles_info_reordered

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int):
        func_pbd_gradient(self.pbd_data)

    @qd.func(requires_top_level=True)
    def func_factor(self):
        func_pbd_factor(self.pbd_data)

    @qd.func(requires_top_level=True)
    def func_pcg_init(self, pcg_scalars: qd.template(), is_adjoint: qd.template()):
        func_pbd_pcg_init(self.pbd_data, pcg_scalars, is_adjoint)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        func_pbd_precondition(self.pbd_data)

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, pcg_scalars: qd.template()):
        func_pbd_pcg_init_direction(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_matvec(self, pcg_scalars: qd.template()):
        func_pbd_matvec(self.pbd_data)

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, pcg_scalars: qd.template()):
        func_pbd_matvec_dot(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, pcg_scalars: qd.template()):
        func_pbd_pcg_update_xr(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, pcg_scalars: qd.template()):
        func_pbd_pcg_dot_rz(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, pcg_scalars: qd.template()):
        func_pbd_pcg_direction(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f: int):
        func_pbd_start_line_search(self.pbd_data, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f: int):
        func_pbd_step_forward(self.pbd_data, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f: int, is_trial: qd.template()):
        func_pbd_energy(self.pbd_data, control, is_trial)


@qd.kernel
def kernel_pbd_save(pbd_data: qd.template()):
    for i_p, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        pbd_data.pos_begin[i_p, i_b] = pbd_data.pbd.particles[i_p, i_b].pos
        pbd_data.vel_begin[i_p, i_b] = pbd_data.pbd.particles[i_p, i_b].vel


@qd.kernel
def kernel_pbd_begin(pbd_data: qd.template()):
    func_pbd_begin(pbd_data)


@qd.kernel
def kernel_pbd_finish(pbd_data: qd.template(), control: qd.template()):
    func_pbd_finish(pbd_data.substep_dt[()], pbd_data, control)


class PbdNewtonHost(NewtonHost):
    """Host stages of PBD around the joint solve: the projection of its substep as the inertia target before, and its
    velocities from the solved positions after.

    PBD runs its own substep before the solve, so the host saves the start of the substep first."""

    def __init__(self, device: PbdNewtonParticipant):
        self._device = device

    def newton_begin(self, f):
        kernel_pbd_save(self._device.pbd_data)

    def newton_prepare(self, f):
        kernel_pbd_begin(self._device.pbd_data)

    def newton_assemble(self, f, is_exact=False):
        kernel_assemble_participant(f, self._device)

    def newton_finish(self, f, control):
        kernel_pbd_finish(self._device.pbd_data, control)
