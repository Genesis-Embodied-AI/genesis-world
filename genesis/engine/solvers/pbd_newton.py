"""Position based dynamics (PBD) as a participant of the NewtonSolver.

The PBD particles enter the joint solve with their inertia alone. PBD projects its own constraints for the substep before
the solve, so its particles hold their start position `ipos` and the end position `pos` of the projection. The joint
solve takes the end positions as unknowns, starting at the projection x~, with the inertia energy 1/2 m |x - x~|^2 around
it in the dt^2-scaled units of the material point method (MPM) energy. PBD's constraints stay with PBD, which projects
them again the next substep, so the coupling is strong in the contact and staggered in the PBD constraints. The
particles own their rows of the linear system of the solve (see newton_linear.py), preconditioned by their diagonal
blocks. An attached particle is an eliminated row (see newton_linear.py and is_moving). After the solve, a PBD particle takes the velocity of its displacement over the substep.
"""

import numpy as np

import quadrants as qd

import genesis as gs

from genesis.utils.data import iter_data

from .newton_linear import (
    BlockJacobiScratch,
    LinearBlock,
    NewtonLinearScratch,
    block_jacobi_visit,
    build_block_jacobi,
    build_linear_block,
    func_block_jacobi_apply,
    func_block_jacobi_invert,
    func_block_jacobi_reset,
    write_block,
)
from .newton_solver import NewtonDevice, NewtonHost


@qd.data_oriented
class PbdNewtonData:
    """The PBD state of the joint solve, per reordered PBD particle and environment, indexed [i_q, i_b]: the
    projection x~ of PBD's own substep and the line-search start.

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
        self.x_prev = qd.ndarray(gs.qd_vec3, shape=shape)


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


@qd.func
def pbd_row(i_q: int, i_b: int, pbd: qd.template()):
    """Row of the linear system of a PBD particle."""
    return pbd.linear_block.row_start[()] + i_b * pbd.n_vertex_particles + i_q


@qd.func(requires_top_level=True)
def func_pbd_assemble(pbd: qd.template(), linear: NewtonLinearScratch, rows: LinearBlock):
    """Write the gradient m (x - x~) of the inertia of every particle and its block m I, I on an attached particle."""
    qd.loop_config(name="mpm_pbd_assemble")
    for i_q, i_b in qd.ndrange(pbd.pbd_data.pbd._n_particles, pbd.pbd_data.pbd._B):
        grad = qd.Vector.zero(gs.qd_float, 3)
        m = gs.qd_float(1.0)
        if is_moving(i_q, i_b, pbd):
            m = pbd.particles_info_reordered[i_q, i_b].mass
            grad = m * (pbd.particles_reordered[i_q, i_b].pos - pbd.pbd_data.target[i_q, i_b])
        row = pbd_row(i_q, i_b, pbd)
        i_t = rows.triplet_start[()] + row - rows.row_start[()]
        write_block(i_t, row, row, m * qd.Matrix.identity(gs.qd_float, 3), linear)
        linear.b_rhs[row] = grad
        linear.pcg.row_env[row] = i_b


@qd.func(requires_top_level=True)
def func_pbd_start_line_search(pbd: qd.template(), control: qd.template()):
    """The line-search start point, and the largest component of the step into the Newton measure."""
    qd.loop_config(name="mpm_pbd_start_line_search")
    for i_q, i_b in qd.ndrange(pbd.pbd_data.pbd._n_particles, pbd.pbd_data.pbd._B):
        pbd.pbd_data.x_prev[i_q, i_b] = pbd.particles_reordered[i_q, i_b].pos
        qd.atomic_max(control.max_disp[i_b], qd.abs(pbd.linear.solution[pbd_row(i_q, i_b, pbd)]).max())


@qd.func(requires_top_level=True)
def func_pbd_step_forward(pbd: qd.template(), control: qd.template()):
    qd.loop_config(name="mpm_pbd_step_forward")
    for i_q, i_b in qd.ndrange(pbd.pbd_data.pbd._n_particles, pbd.pbd_data.pbd._B):
        step = control.alpha[i_b] * pbd.linear.solution[pbd_row(i_q, i_b, pbd)]
        pbd.particles_reordered[i_q, i_b].pos = pbd.pbd_data.x_prev[i_q, i_b] - step


@qd.func(requires_top_level=True)
def func_pbd_energy(pbd_data: qd.template(), energies: qd.template()):
    """Add the inertia 1/2 m |x - x~|^2 of the PBD particles into the entry of every environment in `energies`."""
    qd.loop_config(name="mpm_pbd_energy")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        if is_moving(i_q, i_b, pbd_data.pbd):
            diff = pbd_data.pbd.particles_reordered[i_q, i_b].pos - pbd_data.target[i_q, i_b]
            energy = 0.5 * pbd_data.pbd.particles_info_reordered[i_q, i_b].mass * diff.dot(diff)
            qd.atomic_add(energies[i_b], energy)


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
class PbdNewtonParticipant(NewtonDevice):
    """Device stages of the PBD end positions in the joint solve, which own the PBD data of the solve.

    It is the one place that selects the PBD arrays the couplings read, the particles in PBD's reordered layout. They
    are properties of the solver in `pbd_data`, so that the walk over the data of the NewtonSolver (see
    iter_data_oriented) stops at the solver, which owns them."""

    # The rows of a particle are its own
    n_vertex_rows = 1

    def __init__(self, pbd, n_envs: int):
        self.pbd_data = PbdNewtonData(pbd, n_envs)
        # The particles of every environment are contact vertices (see newton_contact.py)
        self.vertex_offset = qd.ndarray(gs.qd_int, shape=())
        self.vertex_offset.fill(0)
        self.vertex_count = qd.ndarray(gs.qd_int, shape=())
        self.vertex_count.fill(n_envs * pbd.n_particles)
        self.n_vertex_particles = max(pbd.n_particles, 1)
        # The linear system of the solve, bound by the NewtonSolver, and the rows of the particles in it
        self.linear: NewtonLinearScratch | None = None
        self.max_linear_rows = n_envs * self.n_vertex_particles
        self.linear_block: LinearBlock = build_linear_block(self.max_linear_rows, self.max_linear_rows)
        self.jacobi: BlockJacobiScratch = build_block_jacobi(self.max_linear_rows)

    @property
    def data(self):
        yield from iter_data(self, "linear_block")
        yield from iter_data(self, "jacobi")

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
    def func_assemble(self, f: int, is_exact: qd.template()):
        func_pbd_assemble(self, self.linear, self.linear_block)

    @qd.func(requires_top_level=True)
    def func_factor_begin(self):
        func_block_jacobi_reset(self.linear_block.n_rows[()], self.jacobi)

    @qd.func
    def func_linear_visit_block(self, i_r: int, i_c: int, H: qd.types.matrix(3, 3)):
        block_jacobi_visit(i_r, i_c, H, self.jacobi)

    @qd.func(requires_top_level=True)
    def func_factor(self):
        func_block_jacobi_invert(self.linear_block.n_rows[()], self.jacobi)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        func_block_jacobi_apply(self.linear_block, self.linear, self.jacobi)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f: int):
        func_pbd_start_line_search(self, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f: int):
        func_pbd_step_forward(self, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, f: int, energies: qd.template()):
        func_pbd_energy(self.pbd_data, energies)

    # ------------------------------------------- vertex map -------------------------------------------

    @qd.func
    def func_vertex_value(self, f: int, i_v: int):
        i_b = i_v // self.n_vertex_particles
        i_q = i_v - i_b * self.n_vertex_particles
        return self.particles_reordered[i_q, i_b].pos - self.particles_reordered[i_q, i_b].ipos

    @qd.func
    def func_vertex_rows(self, i_v: int):
        i_b = i_v // self.n_vertex_particles
        rows = qd.Vector([self.linear_block.row_start[()] + i_v], dt=gs.qd_int)
        weights = qd.Vector([gs.qd_float(is_moving(i_v - i_b * self.n_vertex_particles, i_b, self))], dt=gs.qd_float)
        return 1, rows, weights


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

    def newton_finish(self, f, control):
        kernel_pbd_finish(self._device.pbd_data, control)
