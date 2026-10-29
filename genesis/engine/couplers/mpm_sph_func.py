"""Coupling of the implicit MPM solver with SPH: the MPM particles as boundary particles of the SPH fluid (Akinci et al.
2012, "Versatile Rigid-Fluid Coupling for Incompressible SPH"), with the DFSPH boundary terms of Bender and Koschier
2017.

An MPM particle b of volume V_b counts in the density of a fluid particle i as rho0 V_b W_ib, and a pressure that
mirrors the fluid's pushes the fluid off it. Over one SPH integration step the fluid positions are fixed and the MPM
particles hold their substep start, so the boundary enters every SPH stage through three sums per fluid particle,
gathered once per step: sum_b V_b W_ib, sum_b V_b grad W_ib and sum_b V_b v_b . grad W_ib. The velocity change the
boundary gives fluid particle i is then s_i sum_b V_b grad W_ib for a scalar s_i, which the WCSPH equation of state or
the DFSPH stiffness of every solver iteration sets. At the end of the step, every MPM particle takes the opposite
impulse -m_i s_i V_b grad W_ib of each fluid particle, and the MPM substep adds the impulses to the particle
velocities before its P2G, so that the momentum the fluid loses to the boundary goes to the MPM material.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.utils import array_class

from .mpm_pbd_func import kernel_fit_mpm_obstacle_leaves


@qd.data_oriented
class MpmSphBoundary:
    """The MPM particles at the start of the substep as boundary particles of SPH (see SPHSolver.set_boundary): an
    LBVH over them for every environment (see genesis.engine.bvh), the boundary sums of every reordered SPH particle
    and environment at i_b n + i_s, its boundary velocity coefficient `s`, and the impulse every MPM particle takes,
    three scalars at 3 (i_b n_mpm + i_p)."""

    def __init__(self, mpm_solver, sph, params, n_envs):
        self.mpm_particles = mpm_solver.particles
        self.mpm_particles_ng = mpm_solver.particles_ng
        self.mpm_particles_info = mpm_solver.particles_info
        self.sph = sph
        self.params = params
        self.n = max(sph.n_particles, 1)
        self.n_mpm = max(mpm_solver.n_particles, 1)
        self.n_envs = n_envs
        self.frame = qd.ndarray(qd.i32, shape=())
        self.b_rho = qd.ndarray(gs.qd_float, shape=(self.n * n_envs,))
        self.b_grad = qd.ndarray(gs.qd_float, shape=(3 * self.n * n_envs,))
        self.b_vel_grad = qd.ndarray(gs.qd_float, shape=(self.n * n_envs,))
        self.b_count = qd.ndarray(qd.i32, shape=(self.n * n_envs,))
        self.s = qd.ndarray(gs.qd_float, shape=(self.n * n_envs,))
        self.impulse = qd.ndarray(gs.qd_float, shape=(3 * self.n_mpm * n_envs,))
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, self.n_mpm)
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        self.results = array_class.get_bvh_query_results(1)

    def build(self, f):
        """Fit the leaves to the MPM particles at frame f and build the trees, at the start of SPH's substep."""
        kernel_fit_mpm_obstacle_leaves(self, f)
        build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)

    def gather(self):
        """The boundary sums of every SPH particle, once SPH reordered its particles for an integration step."""
        kernel_gather_boundary(self)

    def scatter(self):
        """The impulses of the SPH particles on the MPM particles over an integration step."""
        kernel_scatter_boundary(self)

    @qd.func
    def b_grad_at(self, k):
        return qd.Vector([self.b_grad[3 * k + d] for d in qd.static(range(3))], dt=gs.qd_float)


@qd.func
def func_boundary_gather_filter(i_t: int, i_a: int, i_q: int, boundary: qd.template()) -> bool:
    """Add MPM particle i_a to the boundary sums of SPH particle i_q of environment i_t if it lies within the kernel
    support. Records no pair."""
    f = boundary.frame[()]
    sph = boundary.sph
    r = sph.particles_reordered[i_q, i_t].pos - boundary.mpm_particles.pos[f, i_a, i_t]
    dist = r.norm()
    if dist < qd.static(sph.support_radius):
        k = i_t * boundary.n + i_q
        volume = boundary.params.particle_volume[()]
        grad = volume * sph.cubic_kernel_derivative(r)
        boundary.b_rho[k] += volume * sph.cubic_kernel(dist)
        for d in qd.static(range(3)):
            boundary.b_grad[3 * k + d] += grad[d]
        boundary.b_vel_grad[k] += boundary.mpm_particles.vel[f, i_a, i_t].dot(grad)
        boundary.b_count[k] += 1
    return True


@qd.func
def func_boundary_scatter_filter(i_t: int, i_a: int, i_q: int, boundary: qd.template()) -> bool:
    """Give MPM particle i_a the impulse -m_i s_i V_b grad W_ib of SPH particle i_q of environment i_t. Records no
    pair."""
    f = boundary.frame[()]
    sph = boundary.sph
    r = sph.particles_reordered[i_q, i_t].pos - boundary.mpm_particles.pos[f, i_a, i_t]
    if r.norm() < qd.static(sph.support_radius):
        k = i_t * boundary.n + i_q
        coef = -sph.particles_info_reordered[i_q, i_t].mass * boundary.s[k] * boundary.params.particle_volume[()]
        impulse = coef * sph.cubic_kernel_derivative(r)
        base = 3 * (i_t * boundary.n_mpm + i_a)
        for d in qd.static(range(3)):
            qd.atomic_add(boundary.impulse[base + d], impulse[d])
    return True


@qd.kernel
def kernel_gather_boundary(boundary: qd.template()):
    """The boundary sums of every active SPH particle by a box query of the MPM trees over its kernel support, and its
    boundary velocity coefficient cleared for the step."""
    sph = boundary.sph
    h = qd.static(sph.support_radius)
    qd.loop_config(name="sph_mpm_boundary_gather")
    for i_s, i_b in qd.ndrange(sph._n_particles, sph._B):
        k = i_b * boundary.n + i_s
        boundary.b_rho[k] = 0.0
        for d in qd.static(range(3)):
            boundary.b_grad[3 * k + d] = 0.0
        boundary.b_vel_grad[k] = 0.0
        boundary.b_count[k] = 0
        boundary.s[k] = 0.0
        if sph.particles_ng_reordered[i_s, i_b].active:
            x = sph.particles_reordered[i_s, i_b].pos
            func_bvh_query_aabb(
                i_b, i_s, x - h, x + h, boundary.tree, boundary.results, func_boundary_gather_filter, boundary
            )


@qd.kernel
def kernel_scatter_boundary(boundary: qd.template()):
    """The impulse of every SPH particle with a boundary velocity coefficient on the MPM particles of its support."""
    sph = boundary.sph
    h = qd.static(sph.support_radius)
    qd.loop_config(name="sph_mpm_boundary_scatter")
    for i_s, i_b in qd.ndrange(sph._n_particles, sph._B):
        if sph.particles_ng_reordered[i_s, i_b].active and boundary.s[i_b * boundary.n + i_s] != 0.0:
            x = sph.particles_reordered[i_s, i_b].pos
            func_bvh_query_aabb(
                i_b, i_s, x - h, x + h, boundary.tree, boundary.results, func_boundary_scatter_filter, boundary
            )


@qd.kernel
def kernel_apply_boundary_impulse(boundary: qd.template(), f: qd.i32):
    """Add the impulses of the SPH particles to the velocity of every active MPM particle at frame f, before the MPM
    P2G, and clear them."""
    for i_p, i_b in qd.ndrange(boundary.n_mpm, boundary.n_envs):
        base = 3 * (i_b * boundary.n_mpm + i_p)
        if boundary.mpm_particles_ng.active[f, i_p, i_b]:
            mass = boundary.mpm_particles_info.mass[i_p] / boundary.params.mass_scale[()]
            impulse = qd.Vector([boundary.impulse[base + d] for d in qd.static(range(3))], dt=gs.qd_float)
            boundary.mpm_particles.vel[f, i_p, i_b] = boundary.mpm_particles.vel[f, i_p, i_b] + impulse / mass
        for d in qd.static(range(3)):
            boundary.impulse[base + d] = 0.0
