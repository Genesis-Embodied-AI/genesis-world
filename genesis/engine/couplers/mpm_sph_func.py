"""Coupling of the implicit material point method (MPM) solver with smoothed particle hydrodynamics (SPH), the MPM
particles acting as boundary particles of the fluid.

The boundary follows Akinci et al. 2012 ("Versatile Rigid-Fluid Coupling for Incompressible SPH"), with the
divergence-free SPH (DFSPH) boundary terms of Bender and Koschier 2017.

An MPM particle b of volume V_b counts in the density of a fluid particle i as rho0 V_b W_ib, and a pressure that
mirrors the fluid's pushes the fluid off it. Over one SPH integration step the fluid positions are fixed and the MPM
particles hold their substep start, so the boundary enters every SPH stage through three sums per fluid particle,
gathered once per step: sum_b V_b W_ib, sum_b V_b grad W_ib and sum_b V_b v_b . grad W_ib. The velocity change the
boundary gives fluid particle i is then s_i sum_b V_b grad W_ib for a scalar s_i, which the weakly compressible SPH
(WCSPH) equation of state or the DFSPH stiffness of every solver iteration sets. At the end of the step, every MPM
particle takes the opposite impulse -m_i s_i V_b grad W_ib of each fluid particle, and the MPM substep adds the impulses
to the particle velocities before its particle-to-grid transfer, so that the momentum the fluid loses to the boundary
goes to the MPM material.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.utils import array_class

from .mpm_pbd_func import kernel_fit_mpm_obstacle_leaves


@qd.data_oriented
class MpmSphBoundary:
    """The MPM particles at the start of the substep as boundary particles of SPH (see SPHSolver.set_boundary).

    It holds a linear bounding volume hierarchy (LBVH) over them for every environment (see genesis.engine.bvh), the
    impulse every MPM particle takes at [i_p, i_b], and the boundary sums and the boundary velocity coefficient `s` of
    every reordered SPH particle at [i_s, i_b]."""

    def __init__(self, mpm_solver, sph, params, n_envs):
        self.mpm_particles = mpm_solver.particles
        self.mpm_particles_ng = mpm_solver.particles_ng
        self.mpm_particles_info = mpm_solver.particles_info
        self.sph = sph
        self.params = params
        n_mpm = max(mpm_solver.n_particles, 1)
        sph_shape = (max(sph.n_particles, 1), n_envs)
        self.frame = qd.ndarray(gs.qd_int, shape=())
        self.rho_sum = qd.ndarray(gs.qd_float, shape=sph_shape)
        self.grad_sum = qd.ndarray(gs.qd_vec3, shape=sph_shape)
        self.vel_grad_sum = qd.ndarray(gs.qd_float, shape=sph_shape)
        self.n_neighbors = qd.ndarray(gs.qd_int, shape=sph_shape)
        self.s = qd.ndarray(gs.qd_float, shape=sph_shape)
        self.impulse = qd.ndarray(gs.qd_vec3, shape=(n_mpm, n_envs))
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, n_mpm)
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        self.results = array_class.get_bvh_query_results(1)

    def build(self, f):
        """Fit the leaves to the MPM particles and build the trees, at the start of SPH's substep.

        The leaves sit at the MPM particles of frame f."""
        kernel_fit_mpm_obstacle_leaves(f, self)
        build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)

    def gather(self):
        """The boundary sums of every SPH particle, once SPH reordered its particles for an integration step."""
        kernel_gather_boundary(self)

    def scatter(self):
        """The impulses of the SPH particles on the MPM particles over an integration step."""
        kernel_scatter_boundary(self)

    def apply_impulse(self, f):
        """Hand the MPM particles of frame f the impulses SPH's substep accumulated, before the MPM particle-to-grid
        transfer."""
        kernel_apply_boundary_impulse(f, self)


@qd.func
def func_boundary_gather_filter(i_t: int, i_a: int, i_q: int, boundary: qd.template()) -> bool:
    """Add an MPM particle to the boundary sums of an SPH particle if it lies within the kernel support.

    The MPM particle is i_a and the SPH particle i_q of environment i_t. The filter records no pair."""
    f = boundary.frame[()]
    sph = boundary.sph
    r = sph.particles_reordered[i_q, i_t].pos - boundary.mpm_particles.pos[f, i_a, i_t]
    dist = r.norm()
    if dist < qd.static(sph.support_radius):
        volume = boundary.params.particle_volume[()]
        grad = volume * sph.cubic_kernel_derivative(r)
        boundary.rho_sum[i_q, i_t] += volume * sph.cubic_kernel(dist)
        boundary.grad_sum[i_q, i_t] += grad
        boundary.vel_grad_sum[i_q, i_t] += boundary.mpm_particles.vel[f, i_a, i_t].dot(grad)
        boundary.n_neighbors[i_q, i_t] += 1
    return True


@qd.func
def func_boundary_scatter_filter(i_t: int, i_a: int, i_q: int, boundary: qd.template()) -> bool:
    """Give an MPM particle within the kernel support of an SPH particle its impulse -m_i s_i V_b grad W_ib.

    The MPM particle is i_a and the SPH particle i_q of environment i_t. The filter records no pair."""
    f = boundary.frame[()]
    sph = boundary.sph
    r = sph.particles_reordered[i_q, i_t].pos - boundary.mpm_particles.pos[f, i_a, i_t]
    if r.norm() < qd.static(sph.support_radius):
        coef = -sph.particles_info_reordered[i_q, i_t].mass * boundary.s[i_q, i_t] * boundary.params.particle_volume[()]
        qd.atomic_add(boundary.impulse[i_a, i_t], coef * sph.cubic_kernel_derivative(r))
    return True


@qd.kernel
def kernel_gather_boundary(boundary: qd.template()):
    """The boundary sums of every active SPH particle by a box query of the MPM trees over its kernel support, and its
    boundary velocity coefficient cleared for the step."""
    sph = boundary.sph
    h = qd.static(sph.support_radius)
    qd.loop_config(name="sph_mpm_boundary_gather")
    for i_s, i_b in qd.ndrange(sph._n_particles, sph._B):
        boundary.rho_sum[i_s, i_b] = 0.0
        boundary.grad_sum[i_s, i_b] = qd.Vector.zero(gs.qd_float, 3)
        boundary.vel_grad_sum[i_s, i_b] = 0.0
        boundary.n_neighbors[i_s, i_b] = 0
        boundary.s[i_s, i_b] = 0.0
        if sph.particles_ng_reordered[i_s, i_b].active:
            x = sph.particles_reordered[i_s, i_b].pos
            func_bvh_query_aabb(
                i_b, i_s, x - h, x + h, boundary.tree, boundary.results, func_boundary_gather_filter, boundary
            )


@qd.kernel
def kernel_scatter_boundary(boundary: qd.template()):
    """The impulse of every SPH particle on the MPM particles of its support, which the gather counted."""
    sph = boundary.sph
    h = qd.static(sph.support_radius)
    qd.loop_config(name="sph_mpm_boundary_scatter")
    for i_s, i_b in qd.ndrange(sph._n_particles, sph._B):
        if sph.particles_ng_reordered[i_s, i_b].active and boundary.n_neighbors[i_s, i_b] > 0:
            x = sph.particles_reordered[i_s, i_b].pos
            func_bvh_query_aabb(
                i_b, i_s, x - h, x + h, boundary.tree, boundary.results, func_boundary_scatter_filter, boundary
            )


@qd.kernel
def kernel_apply_boundary_impulse(f: int, boundary: qd.template()):
    """Add the impulses of the SPH particles to the velocity of every active MPM particle before the MPM
    particle-to-grid transfer, and clear them.

    The velocities are those of frame f."""
    for i_p, i_b in qd.ndrange(boundary.impulse.shape[0], boundary.impulse.shape[1]):
        if boundary.mpm_particles_ng.active[f, i_p, i_b]:
            mass = boundary.mpm_particles_info.mass[i_p] / boundary.params.mass_scale[()]
            impulse = boundary.impulse[i_p, i_b]
            boundary.mpm_particles.vel[f, i_p, i_b] = boundary.mpm_particles.vel[f, i_p, i_b] + impulse / mass
        boundary.impulse[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
