"""Coupling of the implicit material point method (MPM) solver with smoothed particle hydrodynamics (SPH).

The MPM particles act as boundary particles of the fluid (Akinci et al. 2012, with the divergence-free SPH (DFSPH)
boundary terms of Bender and Koschier 2017)."""

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_func import EMPTY_BOX_HALF
from genesis.utils.array_class import BVHQueryResults
from genesis.utils.data import ENVS, Data, DataKind, Size, field

SPH_PARTICLES = Size("sph_particles")
MPM_PARTICLES = Size("mpm_particles")


class MpmSphBoundaryScratch(Data, kind=DataKind.SCRATCH):
    """The frame of the MPM particles in the tree of MpmSphBoundary, the boundary sums of every SPH particle, and the
    impulses of the SPH particles on every MPM particle over the substep of SPH."""

    frame: qd.Tensor = field(int, ndarray=True)
    rho_sum: qd.Tensor = field(float, shape=(SPH_PARTICLES, ENVS), ndarray=True)
    grad_sum: qd.Tensor = field((float, 3), shape=(SPH_PARTICLES, ENVS), ndarray=True)
    vel_grad_sum: qd.Tensor = field(float, shape=(SPH_PARTICLES, ENVS), ndarray=True)
    n_neighbors: qd.Tensor = field(int, shape=(SPH_PARTICLES, ENVS), ndarray=True)
    impulse_coef: qd.Tensor = field(float, shape=(SPH_PARTICLES, ENVS), ndarray=True)
    impulse: qd.Tensor = field((float, 3), shape=(MPM_PARTICLES, ENVS), ndarray=True)


@qd.data_oriented
class MpmSphBoundary:
    """The MPM particles at the start of the substep as boundary particles of SPH (see SPHSolver.set_boundary)."""

    def __init__(self, mpm_solver, sph, mpm_info, n_envs):
        self.mpm_particles = mpm_solver.particles
        self.mpm_particles_ng = mpm_solver.particles_ng
        self.mpm_particles_info = mpm_solver.particles_info
        self.sph = sph
        self.mpm_info = mpm_info
        self.scratch = MpmSphBoundaryScratch(
            sizes={SPH_PARTICLES: sph.n_particles, MPM_PARTICLES: mpm_solver.n_particles, ENVS: n_envs}
        )
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(mpm_solver.n_particles, 1))
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        self.results = BVHQueryResults(sizes=dict(query_results=1))

    def build(self, f):
        """Fit the leaves to the MPM particles and build the trees, at the start of SPH's substep."""
        kernel_fit_mpm_particle_leaves(f, self)
        build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)

    def gather(self):
        """The boundary sums of every SPH particle, once SPH reordered its particles for an integration step."""
        kernel_gather_boundary(self)

    def scatter(self):
        """The impulses of the SPH particles on the MPM particles over an integration step."""
        kernel_scatter_boundary(self)

    def apply_impulse(self, f):
        """Hand the MPM particles the impulses SPH's substep accumulated, before the MPM particle-to-grid transfer.

        The impulses go to the velocities of frame `f`."""
        kernel_apply_boundary_impulse(f, self)


@qd.kernel
def kernel_fit_mpm_particle_leaves(f: int, query: qd.template()):
    """Fit a point leaf at every active MPM particle, and an empty box with a finite center at the others.

    The leaves are those of the tree of `query` (MpmSphBoundary, MpmParticleObstacles), at the positions of frame `f`,
    which the query records."""
    for _ in range(1):
        query.scratch.frame[()] = f
    for i_b, i_p in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        if query.mpm_particles_ng.active[f, i_p, i_b]:
            x = query.mpm_particles.pos[f, i_p, i_b]
            query.leaves.aabbs_min[i_b, i_p] = x
            query.leaves.aabbs_max[i_b, i_p] = x
        else:
            query.leaves.aabbs_min[i_b, i_p] = qd.Vector([EMPTY_BOX_HALF] * 3, dt=gs.qd_float)
            query.leaves.aabbs_max[i_b, i_p] = qd.Vector([-EMPTY_BOX_HALF] * 3, dt=gs.qd_float)


@qd.func
def func_boundary_gather_filter(i_t: int, i_a: int, i_q: int, boundary: qd.template()) -> bool:
    """Add an MPM particle to the boundary sums of an SPH particle if it lies within the kernel support."""
    f = boundary.scratch.frame[()]
    r = boundary.sph.particles_reordered[i_q, i_t].pos - boundary.mpm_particles.pos[f, i_a, i_t]
    dist = r.norm()
    if dist < qd.static(boundary.sph.support_radius):
        volume = boundary.mpm_info.particle_volume[()]
        grad = volume * boundary.sph.cubic_kernel_derivative(r)
        # The thread of the SPH particle alone accumulates its sums
        boundary.scratch.rho_sum[i_q, i_t] = boundary.scratch.rho_sum[i_q, i_t] + volume * boundary.sph.cubic_kernel(
            dist
        )
        boundary.scratch.grad_sum[i_q, i_t] = boundary.scratch.grad_sum[i_q, i_t] + grad
        vel_grad = boundary.mpm_particles.vel[f, i_a, i_t].dot(grad)
        boundary.scratch.vel_grad_sum[i_q, i_t] = boundary.scratch.vel_grad_sum[i_q, i_t] + vel_grad
        boundary.scratch.n_neighbors[i_q, i_t] = boundary.scratch.n_neighbors[i_q, i_t] + 1
    return True


@qd.func
def func_boundary_scatter_filter(i_t: int, i_a: int, i_q: int, boundary: qd.template()) -> bool:
    """Give an MPM particle within the kernel support of an SPH particle its impulse -m_i s_i V_b grad W_ib."""
    f = boundary.scratch.frame[()]
    r = boundary.sph.particles_reordered[i_q, i_t].pos - boundary.mpm_particles.pos[f, i_a, i_t]
    if r.norm() < qd.static(boundary.sph.support_radius):
        coef = (
            -boundary.sph.particles_info_reordered[i_q, i_t].mass
            * boundary.scratch.impulse_coef[i_q, i_t]
            * boundary.mpm_info.particle_volume[()]
        )
        qd.atomic_add(boundary.scratch.impulse[i_a, i_t], coef * boundary.sph.cubic_kernel_derivative(r))
    return True


@qd.kernel
def kernel_gather_boundary(boundary: qd.template()):
    """The boundary sums of every active SPH particle by a box query of the MPM trees over its kernel support, and its
    boundary velocity coefficient cleared for the step."""
    h = qd.static(boundary.sph.support_radius)
    qd.loop_config(name="sph_mpm_boundary_gather")
    for i_p, i_b in qd.ndrange(boundary.sph._n_particles, boundary.sph._B):
        boundary.scratch.rho_sum[i_p, i_b] = 0.0
        boundary.scratch.grad_sum[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        boundary.scratch.vel_grad_sum[i_p, i_b] = 0.0
        boundary.scratch.n_neighbors[i_p, i_b] = 0
        boundary.scratch.impulse_coef[i_p, i_b] = 0.0
        if boundary.sph.particles_ng_reordered[i_p, i_b].active:
            x = boundary.sph.particles_reordered[i_p, i_b].pos
            func_bvh_query_aabb(
                i_b, i_p, x - h, x + h, boundary.tree, boundary.results, func_boundary_gather_filter, boundary
            )


@qd.kernel
def kernel_scatter_boundary(boundary: qd.template()):
    """The impulse of every SPH particle on the MPM particles of its support, which the gather counted."""
    h = qd.static(boundary.sph.support_radius)
    qd.loop_config(name="sph_mpm_boundary_scatter")
    for i_p, i_b in qd.ndrange(boundary.sph._n_particles, boundary.sph._B):
        if boundary.sph.particles_ng_reordered[i_p, i_b].active and boundary.scratch.n_neighbors[i_p, i_b] > 0:
            x = boundary.sph.particles_reordered[i_p, i_b].pos
            func_bvh_query_aabb(
                i_b, i_p, x - h, x + h, boundary.tree, boundary.results, func_boundary_scatter_filter, boundary
            )


@qd.kernel
def kernel_apply_boundary_impulse(f: int, boundary: qd.template()):
    """Add the impulses of the SPH particles to the velocity of every active MPM particle, and clear them.

    The velocities are those of frame `f`."""
    for i_p, i_b in qd.ndrange(boundary.scratch.impulse.shape[0], boundary.scratch.impulse.shape[1]):
        if boundary.mpm_particles_ng.active[f, i_p, i_b]:
            mass = boundary.mpm_particles_info.mass[i_p] / boundary.mpm_info.mass_scale[()]
            impulse = boundary.scratch.impulse[i_p, i_b]
            boundary.mpm_particles.vel[f, i_p, i_b] = boundary.mpm_particles.vel[f, i_p, i_b] + impulse / mass
        boundary.scratch.impulse[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
