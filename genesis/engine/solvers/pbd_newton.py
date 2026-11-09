"""Position based dynamics (PBD) as a participant of the NewtonSolver."""

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.newton_control import NewtonControl
from genesis.utils.data import ENVS, Data, DataKind, Size, field, iter_data

from .newton_contact import VertexRange
from .newton_linear import (
    BlockJacobiOwner,
    BlockJacobiScratch,
    LinearBlock,
    NewtonLinearScratch,
    build_block_jacobi,
    build_linear_block,
    write_block,
)
from .newton_solver import NewtonDevice, NewtonHost

PBD_PARTICLES = Size("pbd_particles")


class PbdNewtonScratch(Data, kind=DataKind.SCRATCH, shape=(PBD_PARTICLES, ENVS)):
    """The positions and velocities of the PBD particles at the substep start, in their own order, then the target of
    their inertia and their positions at the start of the line search, in the reordered layout."""

    pos_begin: qd.Tensor = field((float, 3), ndarray=True)
    vel_begin: qd.Tensor = field((float, 3), ndarray=True)
    target: qd.Tensor = field((float, 3), ndarray=True)
    x_prev: qd.Tensor = field((float, 3), ndarray=True)


@qd.func
def is_moving(i_q: int, i_b: int, pbd: qd.template()):
    """Whether a PBD particle is an unknown of the solve: active and free, an attached particle keeping its position."""
    return pbd.particles_ng_reordered[i_q, i_b].active and pbd.particles_reordered[i_q, i_b].free


@qd.data_oriented
class PbdNewtonParticipant(BlockJacobiOwner, NewtonDevice):
    """Device stages of the PBD end positions in the joint solve."""

    # The rows of a particle are its own
    n_vertex_rows = 1

    def __init__(self, pbd, n_envs: int):
        self.solver = pbd
        self.substep_dt = pbd.substep_dt
        self.scratch = PbdNewtonScratch(sizes={PBD_PARTICLES: pbd.n_particles, ENVS: n_envs})
        # The particles of every environment are contact vertices (see newton_contact.py)
        self.vertex_range = VertexRange(values={"offset": 0, "count": n_envs * pbd.n_particles})
        self.n_vertex_particles = max(pbd.n_particles, 1)
        # The linear system of the solve, bound by the NewtonSolver, and the rows of the particles in it
        self.linear: NewtonLinearScratch | None = None
        self.max_linear_rows = n_envs * self.n_vertex_particles
        self.linear_block: LinearBlock = build_linear_block(self.max_linear_rows, self.max_linear_rows)
        self.jacobi: BlockJacobiScratch = build_block_jacobi(self.max_linear_rows)

    @property
    def data(self):
        yield from iter_data(self, "scratch")
        yield from iter_data(self, "vertex_range")
        yield from iter_data(self, "linear_block")
        yield from iter_data(self, "jacobi")

    @qd.func
    def row(self, i_q: int, i_b: int):
        """Row of the linear system of a PBD particle."""
        return self.linear_block.row_start[()] + i_b * self.n_vertex_particles + i_q

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int, is_exact: qd.template()):
        """Write the gradient m (x - x~) of the inertia of every particle and its block m I, I on an attached
        particle."""
        qd.loop_config(name="mpm_pbd_assemble")
        for i_q, i_b in qd.ndrange(self.solver._n_particles, self.solver._B):
            grad = qd.Vector.zero(gs.qd_float, 3)
            m = gs.qd_float(1.0)
            if is_moving(i_q, i_b, self.solver):
                m = self.solver.particles_info_reordered[i_q, i_b].mass
                grad = m * (self.solver.particles_reordered[i_q, i_b].pos - self.scratch.target[i_q, i_b])
            row = self.row(i_q, i_b)
            i_t = self.linear_block.triplet_start[()] + row - self.linear_block.row_start[()]
            write_block(i_t, row, row, m * qd.Matrix.identity(gs.qd_float, 3), self.linear)
            self.linear.b_rhs[row] = grad
            self.linear.pcg.rows_env[row] = i_b

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: NewtonControl, f: int):
        """Record the line-search start point, and add the largest component of the step into the Newton measure."""
        qd.loop_config(name="mpm_pbd_start_line_search")
        for i_q, i_b in qd.ndrange(self.solver._n_particles, self.solver._B):
            self.scratch.x_prev[i_q, i_b] = self.solver.particles_reordered[i_q, i_b].pos
            qd.atomic_max(control.max_displacement[i_b], qd.abs(self.linear.solution[self.row(i_q, i_b)]).max())

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: NewtonControl, f: int):
        qd.loop_config(name="mpm_pbd_step_forward")
        for i_q, i_b in qd.ndrange(self.solver._n_particles, self.solver._B):
            step = control.alpha[i_b] * self.linear.solution[self.row(i_q, i_b)]
            self.solver.particles_reordered[i_q, i_b].pos = self.scratch.x_prev[i_q, i_b] - step

    @qd.func(requires_top_level=True)
    def func_energy(self, f: int, energies: qd.template()):
        """Add the inertia 1/2 m |x - x~|^2 of the PBD particles into the entry of every environment in `energies`."""
        qd.loop_config(name="mpm_pbd_energy")
        for i_q, i_b in qd.ndrange(self.solver._n_particles, self.solver._B):
            if is_moving(i_q, i_b, self.solver):
                diff = self.solver.particles_reordered[i_q, i_b].pos - self.scratch.target[i_q, i_b]
                energy = 0.5 * self.solver.particles_info_reordered[i_q, i_b].mass * diff.dot(diff)
                qd.atomic_add(energies[i_b], energy)

    # ------------------------------------------- vertex map -------------------------------------------

    @qd.func
    def func_vertex_value(self, f: int, i_v: int):
        i_b = i_v // self.n_vertex_particles
        i_q = i_v - i_b * self.n_vertex_particles
        return self.solver.particles_reordered[i_q, i_b].pos - self.solver.particles_reordered[i_q, i_b].ipos

    @qd.func
    def func_vertex_rows(self, i_v: int):
        i_b = i_v // self.n_vertex_particles
        rows = qd.Vector([self.linear_block.row_start[()] + i_v], dt=gs.qd_int)
        is_row_moving = is_moving(i_v - i_b * self.n_vertex_particles, i_b, self.solver)
        return 1, rows, qd.Vector([gs.qd_float(is_row_moving)], dt=gs.qd_float)


@qd.kernel
def kernel_pbd_save(pbd: qd.template()):
    for i_p, i_b in qd.ndrange(pbd.solver._n_particles, pbd.solver._B):
        pbd.scratch.pos_begin[i_p, i_b] = pbd.solver.particles[i_p, i_b].pos
        pbd.scratch.vel_begin[i_p, i_b] = pbd.solver.particles[i_p, i_b].vel


@qd.kernel
def kernel_pbd_begin(pbd: qd.template()):
    """Take the projection of PBD's substep as the target of the inertia, the end positions starting at it."""
    for i_q, i_b in qd.ndrange(pbd.solver._n_particles, pbd.solver._B):
        pbd.scratch.target[i_q, i_b] = pbd.solver.particles_reordered[i_q, i_b].pos


@qd.kernel
def kernel_pbd_finish(dt: float, pbd: qd.template(), control: NewtonControl):
    """Set the velocity of every moving PBD particle from its displacement over the substep, then the position and
    velocity of the substep start of every particle of an environment whose solve failed."""
    for i_q, i_b in qd.ndrange(pbd.solver._n_particles, pbd.solver._B):
        if is_moving(i_q, i_b, pbd.solver):
            particle = pbd.solver.particles_reordered[i_q, i_b]
            pbd.solver.particles_reordered[i_q, i_b].vel = (particle.pos - particle.ipos) / dt
    for i_p, i_b in qd.ndrange(pbd.solver._n_particles, pbd.solver._B):
        if control.has_failed_non_finite[i_b] != 0 and pbd.solver.particles_ng[i_p, i_b].active:
            i_q = pbd.solver.particles_ng[i_p, i_b].reordered_idx
            pbd.solver.particles_reordered[i_q, i_b].pos = pbd.scratch.pos_begin[i_p, i_b]
            pbd.solver.particles_reordered[i_q, i_b].vel = pbd.scratch.vel_begin[i_p, i_b]


class PbdNewtonHost(NewtonHost):
    """Host stages of PBD around the joint solve: the projection of its substep as the inertia target before, and its
    velocities from the solved positions after."""

    def __init__(self, device: PbdNewtonParticipant):
        self._device = device

    def newton_begin(self, f):
        kernel_pbd_save(self._device)

    def newton_prepare(self, f):
        kernel_pbd_begin(self._device)

    def newton_finish(self, f, control):
        kernel_pbd_finish(self._device.substep_dt, self._device, control)
