import math
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import (
    PBD2DEntity,
    PBD3DEntity,
    PBDFreeParticleEntity,
    PBDParticleEntity,
)
from genesis.engine.entities.pbd_entity import PBDTetEntity
from genesis.engine.materials import PBD
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import PBDSolverState
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.geom import SpatialHasher

from genesis.engine.numerics.newton_control import has_converged

from .newton_solver import NewtonHost
from .particle_solver import ParticleSolver
from .pbd_newton import PbdNewtonHost, PbdNewtonParticipant

if TYPE_CHECKING:
    from genesis.engine.entities import PBD2DEntity, PBD3DEntity, PBDFreeParticleEntity, PBDParticleEntity

# Collision candidates of a particle over a substep: the nearest COLLISION_NEIGHBORS particles within its collision
# reach (see PbdConstraintLoop)
COLLISION_NEIGHBORS = 16

PARTICLES = Size("particles")
EDGES = Size("edges")
INNER_EDGES = Size("inner_edges")
ELEMS = Size("elems")


def collision_reach(particle_size: float) -> float:
    """Distance of the start or predicted position of a particle within which another one is a collision candidate, a
    reach the 27 hash cells around the particle cover."""
    return 1.25 * particle_size


class PbdConstraintLoop(Data, kind=DataKind.SCRATCH):
    """The data of the unified constraint loop of a substep."""

    is_running: qd.Tensor = field(int, ndarray=True)
    n_iterations: qd.Tensor = field(int, ndarray=True)
    max_correction: qd.Tensor = field(float, ndarray=True)
    edges_lambda: qd.Tensor = field(float, shape=(EDGES, ENVS))
    inner_edges_lambda: qd.Tensor = field(float, shape=(INNER_EDGES, ENVS))
    elems_lambda: qd.Tensor = field(float, shape=(ELEMS, ENVS))
    n_collision_neighbors: qd.Tensor = field(int, shape=(PARTICLES, ENVS))
    collision_neighbors_idx: qd.Tensor = field(int, shape=(PARTICLES, COLLISION_NEIGHBORS, ENVS))
    collision_neighbors_dist: qd.Tensor = field(float, shape=(PARTICLES, COLLISION_NEIGHBORS, ENVS))
    # See collision_reach
    collision_reach: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    constraint_counts: qd.Tensor = field((float, 3), shape=(PARTICLES,), kind=DataKind.CONSTANT)


@qd.data_oriented
class PBDSolver(ParticleSolver):
    material_cls = PBD.Base
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    class MATERIAL(gs.IntEnum):
        CLOTH = 0
        ELASTIC = 1
        LIQUID = 2
        PARTICLE = 3  # non-physics particles

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._particle_size = options.particle_size
        self._max_stretch_solver_iterations = options.max_stretch_solver_iterations
        self._max_bending_solver_iterations = options.max_bending_solver_iterations
        self._max_volume_solver_iterations = options.max_volume_solver_iterations
        self._max_density_solver_iterations = options.max_density_solver_iterations
        self._max_viscosity_solver_iterations = options.max_viscosity_solver_iterations
        self._solver_iterations = options.solver_iterations
        self._sor_relaxation = options.sor_relaxation
        self._stabilization_iterations = options.stabilization_iterations
        self._sleep_threshold = options.sleep_threshold
        self._use_xpbd = options.use_xpbd
        self._position_tolerance = options.position_tolerance
        # Particles of another solver the PBD particles collide with as immovable obstacles, set by that solver before
        # the first substep (see set_particle_obstacles)
        self._particle_obstacles = None
        self._has_particle_obstacles = False

        self._n_vvert_supports = self.scene.options.vis.n_support_neighbors

        # -Neighbours_Setting-
        self.dist_scale = self.particle_radius / 0.4  # @Zhenjia: double check this
        self.h = 1.0
        self.h_2 = self.h**2
        self.h_6 = self.h**6
        self.h_9 = self.h**9

        # -POLY6_KERNEL-
        self.poly6_Coe = 315.0 / (64 * math.pi)

        # -SPIKY_KERNEL-
        self.spiky_Coe = -45.0 / math.pi

        # -LAMBDAS-
        self.lambda_epsilon = 100.0

        # -S_CORR-
        self.S_Corr_delta_q = 0.3
        self.S_Corr_k = 0.0001

        # -Gradient Approx. delta difference-
        self.g_del = 0.01

        self.vorqd_epsilon = 0.01

        # spatial hasher
        self.sh = SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
        )

    def init_vvert_fields(self):
        struct_vvert_info = qd.types.struct(
            support_idxs=qd.types.vector(self._n_vvert_supports, gs.qd_int),
            support_weights=qd.types.vector(self._n_vvert_supports, gs.qd_float),
        )
        self.vverts_info = struct_vvert_info.field(shape=(max(self._n_vverts, 1),), layout=qd.Layout.SOA)

        struct_vvert_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            active=gs.qd_bool,
        )
        self.vverts_render = struct_vvert_state_render.field(
            shape=(max(self._n_vverts, 1), self._B), layout=qd.Layout.SOA
        )

        # UV coordinates for visual vertices (static, same across all batch envs)
        self.vverts_uvs = qd.field(dtype=gs.qd_vec2, shape=(max(self._n_vverts, 1),))

        # Triangle face indices for visual mesh (static)
        self.vfaces_indices = qd.field(dtype=gs.qd_ivec3, shape=(max(self._n_vfaces, 1),))

    def init_particle_fields(self):
        # particles information (static)
        struct_particle_info = qd.types.struct(
            mass=gs.qd_float,
            pos_rest=gs.qd_vec3,
            rho_rest=gs.qd_float,
            material_type=gs.qd_int,
            mu_s=gs.qd_float,
            mu_k=gs.qd_float,
            air_resistance=gs.qd_float,
            density_relaxation=gs.qd_float,
            viscosity_relaxation=gs.qd_float,
        )
        # particles state (dynamic)
        struct_particle_state = qd.types.struct(
            free=gs.qd_bool,  # if not free, the particle is not affected by internal forces and solely controlled by external user until released
            pos=gs.qd_vec3,  # position
            ipos=gs.qd_vec3,  # initial position
            dpos=gs.qd_vec3,  # delta position
            # number of constraints whose corrections dpos accumulates, for their average
            dcount=gs.qd_float,
            vel=gs.qd_vec3,  # velocity
            lam=gs.qd_float,
            rho=gs.qd_float,
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            active=gs.qd_bool,
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            active=gs.qd_bool,
        )

        self.particles_info = struct_particle_info.field(shape=(self._n_particles,), layout=qd.Layout.SOA)
        self.particles_info_reordered = struct_particle_info.field(
            shape=(self._n_particles, self._B), layout=qd.Layout.SOA
        )
        self.particles = struct_particle_state.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_reordered = struct_particle_state.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_ng = struct_particle_state_ng.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_ng_reordered = struct_particle_state_ng.field(
            shape=(self._n_particles, self._B), layout=qd.Layout.SOA
        )
        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), layout=qd.Layout.SOA
        )

    def init_edge_fields(self):
        # edges information for stretch. edge: (v1, v2)
        struct_edge_info = qd.types.struct(
            len_rest=gs.qd_float,
            stretch_compliance=gs.qd_float,
            stretch_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
        )
        self.edges_info = struct_edge_info.field(shape=(max(1, self._n_edges),), layout=qd.Layout.SOA)

        # inner edges information for bending. edge: (v1, v2), adjacent faces: (v1, v2, v3) and (v1, v2, v4)
        struct_inner_edge_info = qd.types.struct(
            len_rest=gs.qd_float,
            bending_compliance=gs.qd_float,
            bending_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
            v3=gs.qd_int,
            v4=gs.qd_int,
        )
        self.inner_edges_info = struct_inner_edge_info.field(shape=(max(self._n_inner_edges, 1),), layout=qd.Layout.SOA)

    def init_elem_fields(self):
        struct_elem_info = qd.types.struct(
            vol_rest=gs.qd_float,
            volume_compliance=gs.qd_float,
            volume_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
            v3=gs.qd_int,
            v4=gs.qd_int,
        )
        self.elems_info = struct_elem_info.field(shape=(max(self._n_elems, 1),), layout=qd.Layout.SOA)

    def build_newton_participant(self) -> tuple[NewtonHost, PbdNewtonParticipant]:
        """Build the participant of the PBD particle positions in the joint Newton solve, whose inertia targets the end
        positions of the projection of PBD's own substep (see PbdNewtonHost)."""
        participant = PbdNewtonParticipant(self, self._B)
        return PbdNewtonHost(participant), participant

    def set_particle_obstacles(self, obstacles):
        """Make the PBD particles collide with the particles of another solver as immovable obstacles.

        `obstacles` provides `build(f)`, called at the start of every substep, and the top-level funcs
        `func_project(f, pbd)` and `func_stabilize(f, pbd)`, which accumulate the corrections of the PBD particles in
        `dpos` and `dcount` of the reordered particles."""
        self._particle_obstacles = obstacles
        self._has_particle_obstacles = True

    def build(self):
        super().build()

        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_fluid_particles = self.n_fluid_particles
        self._n_edges = self.n_edges
        self._n_inner_edges = self.n_inner_edges
        self._n_elems = self.n_elems
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        if self.is_active:
            self.sh.build(self._n_particles, self._B)

            self.init_particle_fields()
            self.init_edge_fields()
            self.init_elem_fields()
            self.init_vvert_fields()
            self.constraint_loop = PbdConstraintLoop(
                sizes={
                    PARTICLES: self._n_particles,
                    EDGES: self._n_edges,
                    INNER_EDGES: self._n_inner_edges,
                    ELEMS: self._n_elems,
                    ENVS: self._B,
                },
                values={"collision_reach": collision_reach(self._particle_size)},
            )

            for entity in self._entities:
                entity._add_to_solver()
            kernel_count_constraints(self)

        # Kernels of this solver take the solver itself, so gravity has to be a field for them.
        self._build_gravity()

    @qd.kernel
    def add_uvs_and_faces(
        self,
        vvert_start: int,
        vface_start: int,
        uvs: qd.types.ndarray(element_dim=1),
        faces: qd.types.ndarray(element_dim=1),
    ):
        """Add the render UVs and faces of an entity, an empty `uvs` leaving its slice of the UV buffer zeroed."""
        for i_vv_ in range(uvs.shape[0]):
            self.vverts_uvs[i_vv_ + vvert_start] = uvs[i_vv_]
        for i_vf_ in range(faces.shape[0]):
            self.vfaces_indices[i_vf_ + vface_start] = faces[i_vf_] + vvert_start

    @qd.kernel
    def add_particles(
        self,
        f: int,
        particle_start: int,
        active: int,
        material_type: int,
        mass: float,
        rho_rest: float,
        mu_s: float,
        mu_k: float,
        air_resistance: float,
        density_relaxation: float,
        viscosity_relaxation: float,
        particles: qd.types.ndarray(),
    ):
        """Add the particles of an entity at rest, each property a material leaves out being given zero."""
        for i_p_ in range(particles.shape[0]):
            i_p = i_p_ + particle_start
            for j in qd.static(range(3)):
                self.particles_info[i_p].pos_rest[j] = particles[i_p_, j]
            self.particles_info[i_p].material_type = material_type
            self.particles_info[i_p].mass = mass
            self.particles_info[i_p].rho_rest = rho_rest
            self.particles_info[i_p].mu_s = mu_s
            self.particles_info[i_p].mu_k = mu_k
            self.particles_info[i_p].air_resistance = air_resistance
            self.particles_info[i_p].density_relaxation = density_relaxation
            self.particles_info[i_p].viscosity_relaxation = viscosity_relaxation

        for i_p_, i_b in qd.ndrange(particles.shape[0], self._B):
            i_p = i_p_ + particle_start
            for j in qd.static(range(3)):
                self.particles[i_p, i_b].pos[j] = particles[i_p_, j]
            self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].dpos = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].free = True
            self.particles_ng[i_p, i_b].active = qd.cast(active, gs.qd_bool)

    @qd.kernel
    def add_edges(
        self,
        edge_start: int,
        particle_start: int,
        stretch_compliance: float,
        stretch_relaxation: float,
        edges: qd.types.ndarray(),
        edges_len_rest: qd.types.ndarray(),
    ):
        """Add the stretch constraints of the edges of an entity, given by the local indices of their particles."""
        for i_e_ in range(edges.shape[0]):
            i_e = i_e_ + edge_start
            self.edges_info[i_e].stretch_compliance = stretch_compliance
            self.edges_info[i_e].stretch_relaxation = stretch_relaxation
            self.edges_info[i_e].len_rest = edges_len_rest[i_e_]
            self.edges_info[i_e].v1 = particle_start + edges[i_e_, 0]
            self.edges_info[i_e].v2 = particle_start + edges[i_e_, 1]

    @qd.kernel
    def add_inner_edges(
        self,
        inner_edge_start: int,
        particle_start: int,
        bending_compliance: float,
        bending_relaxation: float,
        inner_edges: qd.types.ndarray(),
        inner_edges_len_rest: qd.types.ndarray(),
    ):
        """Add the bending constraints of the inner edges of an entity, given by the local indices of their particles."""
        for i_ie_ in range(inner_edges.shape[0]):
            i_ie = i_ie_ + inner_edge_start
            self.inner_edges_info[i_ie].bending_compliance = bending_compliance
            self.inner_edges_info[i_ie].bending_relaxation = bending_relaxation
            self.inner_edges_info[i_ie].len_rest = inner_edges_len_rest[i_ie_]
            self.inner_edges_info[i_ie].v1 = particle_start + inner_edges[i_ie_, 0]
            self.inner_edges_info[i_ie].v2 = particle_start + inner_edges[i_ie_, 1]
            self.inner_edges_info[i_ie].v3 = particle_start + inner_edges[i_ie_, 2]
            self.inner_edges_info[i_ie].v4 = particle_start + inner_edges[i_ie_, 3]

    @qd.kernel
    def add_elems(
        self,
        elem_start: int,
        particle_start: int,
        volume_compliance: float,
        volume_relaxation: float,
        elems: qd.types.ndarray(),
        elems_vol_rest: qd.types.ndarray(),
    ):
        """Add the volume constraints of the tetrahedra of an entity, given by the local indices of their particles."""
        for i_el_ in range(elems.shape[0]):
            i_el = i_el_ + elem_start
            self.elems_info[i_el].volume_compliance = volume_compliance
            self.elems_info[i_el].volume_relaxation = volume_relaxation
            self.elems_info[i_el].vol_rest = elems_vol_rest[i_el_]
            self.elems_info[i_el].v1 = particle_start + elems[i_el_, 0]
            self.elems_info[i_el].v2 = particle_start + elems[i_el_, 1]
            self.elems_info[i_el].v3 = particle_start + elems[i_el_, 2]
            self.elems_info[i_el].v4 = particle_start + elems[i_el_, 3]

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "PBD2DEntity | PBD3DEntity | PBDParticleEntity | PBDFreeParticleEntity":
        if isinstance(material, gs.materials.PBD.Cloth):
            entity = PBD2DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                inner_edge_start=self.n_inner_edges,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
                name=name,
            )

        elif isinstance(material, gs.materials.PBD.Elastic):
            entity = PBD3DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                elem_start=self.n_elems,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
                name=name,
            )

        elif isinstance(material, gs.materials.PBD.Liquid):
            entity = PBDParticleEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                name=name,
            )

        elif isinstance(material, gs.materials.PBD.Particle):
            entity = PBDFreeParticleEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                name=name,
            )

        else:
            raise NotImplementedError()

        self._entities.append(entity)

        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def poly6(self, dist):
        # dist is a VECTOR
        result = gs.qd_float(0.0)
        d = dist.norm() / self.dist_scale
        if 0 < d < self.h:
            rhs = (self.h_2 - d * d) * (self.h_2 - d * d) * (self.h_2 - d * d)
            result = self.poly6_Coe * rhs / self.h_9
        return result

    @qd.func
    def poly6_scalar(self, dist):
        # dist is a SCALAR
        result = gs.qd_float(0.0)
        d = dist
        if 0 < d < self.h:
            rhs = (self.h_2 - d * d) * (self.h_2 - d * d) * (self.h_2 - d * d)
            result = self.poly6_Coe * rhs / self.h_9
        return result

    @qd.func
    def spiky(self, dist):
        # dist is a VECTOR
        result = qd.Vector.zero(gs.qd_float, 3)
        d = dist.norm() / self.dist_scale
        if 0 < d < self.h:
            m = (self.h - d) * (self.h - d)
            result = (self.spiky_Coe * m / (self.h_6 * d)) * dist / self.dist_scale
        return result

    @qd.func
    def S_Corr(self, dist):
        upper = self.poly6(dist)
        lower = self.poly6_scalar(self.S_Corr_delta_q)
        m = upper / lower
        return -1.0 * self.S_Corr_k * m * m * m * m

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------
    @qd.kernel
    def _kernel_store_initial_pos(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles[i_p, i_b].ipos = self.particles[i_p, i_b].pos

    @qd.kernel
    def _kernel_reorder_particles(self, f: qd.i32):
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )

        # copy to reordered
        self.particles_ng_reordered.active.fill(False)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx

                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]
                self.particles_info_reordered[reordered_idx, i_b] = self.particles_info[i_p]
                self.particles_ng_reordered[reordered_idx, i_b].active = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_apply_external_force(self, f: qd.i32, t: float):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].free:
                # gravity
                self.particles[i_p, i_b].vel = self.particles[i_p, i_b].vel + self._gravity[i_b] * self._substep_dt

                # external force fields
                acc = qd.Vector.zero(gs.qd_float, 3)
                for i_ff in qd.static(range(len(self._ffs))):
                    acc += self._ffs[i_ff].get_acc(self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel, t, i_p)
                self.particles[i_p, i_b].vel = self.particles[i_p, i_b].vel + acc * self._substep_dt

                if self.particles_info[i_p].material_type == self.MATERIAL.CLOTH:
                    f_air_resistance = (
                        self.particles_info[i_p].air_resistance
                        * self.particles[i_p, i_b].vel.norm()
                        * self.particles[i_p, i_b].vel
                    )
                    self.particles[i_p, i_b].vel = (
                        self.particles[i_p, i_b].vel
                        - f_air_resistance / self.particles_info[i_p].mass * self._substep_dt
                    )

            # attached particles are not free but still need to update position to follow the link
            self.particles[i_p, i_b].pos = (
                self.particles[i_p, i_b].pos + self.particles[i_p, i_b].vel * self._substep_dt
            )

    @qd.kernel
    def _kernel_solve_boundary_collision(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            # boundary is enforced regardless of whether free
            pos_new, vel_new = self.boundary.impose_pos_vel(self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel)
            self.particles[i_p, i_b].pos = pos_new
            self.particles[i_p, i_b].vel = vel_new

    @qd.kernel
    def _kernel_solve_viscosity(self, f: qd.i32):
        for _ in qd.static(range(self._max_viscosity_solver_iterations)):
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID:
                    pos_i = self.particles_reordered[i_p, i_b].pos
                    base = self.sh.pos_to_grid(pos_i)
                    xsph_sum = qd.Vector.zero(gs.qd_float, 3)
                    omega_sum = qd.Vector.zero(gs.qd_float, 3)
                    # -For Gradient Approx.-
                    dx_sum = qd.Vector.zero(gs.qd_float, 3)
                    dy_sum = qd.Vector.zero(gs.qd_float, 3)
                    dz_sum = qd.Vector.zero(gs.qd_float, 3)
                    n_dx_sum = qd.Vector.zero(gs.qd_float, 3)
                    n_dy_sum = qd.Vector.zero(gs.qd_float, 3)
                    n_dz_sum = qd.Vector.zero(gs.qd_float, 3)
                    dx = qd.Vector([self.g_del, 0.0, 0.0], dt=gs.qd_float)
                    dy = qd.Vector([0.0, self.g_del, 0.0], dt=gs.qd_float)
                    dz = qd.Vector([0.0, 0.0, self.g_del], dt=gs.qd_float)

                    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            pos_j = self.particles_reordered[j, i_b].pos
                            v_ij = (self.particles_reordered[j, i_b].pos - self.particles_reordered[j, i_b].ipos) - (
                                self.particles_reordered[i_p, i_b].pos - self.particles_reordered[i_p, i_b].ipos
                            )

                            dist = pos_i - pos_j
                            # ---Vorticity---
                            omega_sum += v_ij.cross(self.spiky(dist))
                            # -Gradient Approx.-
                            dx_sum += v_ij.cross(self.spiky(dist + dx))
                            dy_sum += v_ij.cross(self.spiky(dist + dy))
                            dz_sum += v_ij.cross(self.spiky(dist + dz))
                            n_dx_sum += v_ij.cross(self.spiky(dist - dx))
                            n_dy_sum += v_ij.cross(self.spiky(dist - dy))
                            n_dz_sum += v_ij.cross(self.spiky(dist - dz))
                            # ---Viscosity---
                            poly = self.poly6(dist)
                            xsph_sum += poly * v_ij

                    # # ---Vorticity---
                    # n_x = (dx_sum.norm() - n_dx_sum.norm()) / (2 * self.g_del)
                    # n_y = (dy_sum.norm() - n_dy_sum.norm()) / (2 * self.g_del)
                    # n_z = (dz_sum.norm() - n_dz_sum.norm()) / (2 * self.g_del)
                    # n = qd.Vector([n_x, n_y, n_z])
                    # big_n = n.normalized()
                    # if not omega_sum.norm() == 0.0:
                    #     vorticity[p] = vorqd_epsilon * big_n.cross(omega_sum)

                    # ---Viscosity---
                    self.particles_reordered[i_p, i_b].dpos = (
                        self.particles_reordered[i_p, i_b].dpos
                        + xsph_sum * self.particles_info_reordered[i_p, i_b].viscosity_relaxation
                    )

            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if (
                    self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                    and self.particles_reordered[i_p, i_b].free
                ):
                    self.particles_reordered[i_p, i_b].pos = (
                        self.particles_reordered[i_p, i_b].pos + self.particles_reordered[i_p, i_b].dpos
                    )
                    self.particles_reordered[i_p, i_b].dpos.fill(0)

    @qd.kernel
    def _kernel_compute_velocity(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles_reordered[i_p, i_b].vel = (
                self.particles_reordered[i_p, i_b].pos - self.particles_reordered[i_p, i_b].ipos
            ) / self._substep_dt

    @qd.kernel
    def _kernel_copy_from_reordered(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx
                self.particles[i_p, i_b] = self.particles_reordered[reordered_idx, i_b]

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        """One substep of the unified particle solver (Macklin et al. 2014, Algorithm 1)."""
        if self.is_active:
            self._kernel_store_initial_pos(f)
            if self._use_xpbd:
                kernel_reset_lambda(self)
            self._kernel_apply_external_force(f, self._sim.cur_t)
            self._kernel_reorder_particles(f)
            kernel_gather_collision_neighbors(f, self)
            if self._has_particle_obstacles:
                self._particle_obstacles.build(f)

            for _ in range(self._stabilization_iterations):
                kernel_solve_collision(f, self, is_stabilization=True)
                if self._has_particle_obstacles:
                    kernel_solve_obstacles(f, self, is_stabilization=True)
                kernel_apply_corrections(f, self, is_average=True, is_stabilization=True)

            kernel_constraint_loop(f, self)

            if self._n_fluid_particles > 0:
                self._kernel_solve_viscosity(f)

            # compute effective velocity
            self._kernel_compute_velocity(f)
            if self._sleep_threshold > 0.0:
                kernel_sleep(f, self)

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_copy_from_reordered(f)

            # boundary collision
            self._kernel_solve_boundary_collision(f)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.free)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, _n_particles, 3]
        vel: qd.types.ndarray(),  # shape [B, _n_particles, 3]
        free: qd.types.ndarray(),  # shape [B, _n_particles]
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
            self.particles[i_p, i_b].free = free[i_b, i_p]

    def get_state(self, f):
        if self.is_active:
            state = PBDSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.free)
        else:
            state = None
        return state

    def get_state_render(self):
        """
        Get visual vertex positions, UVs, and face indices for rendering.

        Returns
        -------
        tuple
            (vverts_pos, vverts_uvs, vfaces_indices) - vertex positions, UV coords, and triangle indices
        """
        if not self.is_active or self._n_vverts == 0:
            return None, None, None

        # Make sure render fields are up to date
        self.update_render_fields()

        # Return the Quadrants fields directly for GPU access
        return self.vverts_render.pos, self.vverts_uvs, self.vfaces_indices

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, _n_particles, 3]
        vel: qd.types.ndarray(),  # shape [B, _n_particles, 3]
        free: qd.types.ndarray(),  # shape [B, _n_particles]
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
            free[i_b, i_p] = qd.cast(self.particles[i_p, i_b].free, gs.qd_bool)

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng[i_p, i_b].active

        for i_v, i_b in qd.ndrange(self._n_vverts, self._B):
            vvert_pos = qd.Vector.zero(gs.qd_float, 3)
            for j in range(self._n_vvert_supports):
                vvert_pos += (
                    self.particles[self.vverts_info.support_idxs[i_v][j], i_b].pos
                    * self.vverts_info.support_weights[i_v][j]
                )
            self.vverts_render[i_v, i_b].pos = vvert_pos
            self.vverts_render[i_v, i_b].active = self.particles_render[
                self.vverts_info.support_idxs[i_v][0], i_b
            ].active

    @gs.assert_built
    @mutates(StateChange.DYNAMICS, differentiable=False)
    def fix_particles(self, particles_idx=None, envs_idx=None):
        """
        Fix some particles where they are, until they are released (see 'release_particles').

        Parameters
        ----------
        particles_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the particles (see 'set_particles_pos'). If None, all particles will be considered.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        _, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        self._kernel_fix_particles(particles_idx, envs_idx)

    @gs.assert_built
    @mutates(StateChange.DYNAMICS, differentiable=False)
    def fix_particles_to_link(self, link_idx, particles_idx=None, envs_idx=None):
        """
        Attach some particles to a rigid link, which carries them along until they are released (see
        'release_particles').

        Parameters
        ----------
        link_idx : int
            Global index of the rigid link.
        particles_idx, envs_idx : see 'fix_particles'.
        """
        _, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        self._sim._coupler.kernel_attach_pbd_to_rigid_link(
            particles_idx, envs_idx, link_idx, self._sim.rigid_solver.dyn_state.links
        )

    @gs.assert_built
    @mutates(StateChange.DYNAMICS, differentiable=False)
    def release_particles(self, particles_idx=None, envs_idx=None):
        """Release some particles fixed in place or to a rigid link, which then move freely again (see 'fix_particles')."""
        _, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        self._kernel_release_particles(particles_idx, envs_idx)
        self._sim._coupler.kernel_pbd_rigid_clear_animate_particles_by_link(particles_idx, envs_idx)

    @qd.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # The earlier occurrences of a repeated particle are -1 (see 'drop_repeated_index')
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
                self.particles[i_p, i_b].vel.fill(0.0)

    @qd.kernel
    def _kernel_get_particles_pos(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[i_p, i_b].pos[i]

    @qd.kernel
    def _kernel_set_particles_vel(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]

    @qd.kernel
    def _kernel_get_particles_vel(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[i_p, i_b].vel[i]

    @qd.kernel
    def _kernel_set_particles_active(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_ng[i_p, i_b].active = qd.cast(actives[i_b_, i_p_], gs.qd_bool)

    @qd.kernel
    def _kernel_get_particles_active(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_fix_particles(self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles[i_p, i_b].free = False

    @qd.kernel
    def _kernel_release_particles(self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles[i_p, i_b].free = True

    @qd.kernel
    def _kernel_get_particles_mass(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), masses: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            masses[i_b_, i_p_] = self.particles_info[i_p].mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum([entity.n_particles for entity in self._entities])

    @property
    def n_fluid_particles(self):
        if self.is_built:
            return self._n_fluid_particles
        return sum(entity.n_fluid_particles for entity in self._entities if isinstance(entity, PBDParticleEntity))

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        return sum(entity.n_edges for entity in self._entities if isinstance(entity, PBDTetEntity))

    @property
    def n_inner_edges(self):
        if self.is_built:
            return self._n_inner_edges
        return sum(entity.n_inner_edges for entity in self._entities if isinstance(entity, PBD2DEntity))

    @property
    def n_elems(self):
        if self.is_built:
            return self._n_elems
        return sum(entity.n_elems for entity in self._entities if isinstance(entity, PBD3DEntity))

    @property
    def n_vvert_supports(self):
        """
        Number of particles supporting each visual vertex.
        """
        return self._n_vvert_supports

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def hash_grid_res(self):
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        return self.sh.cell_size

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound


# ============================================== unified constraint loop ===============================================
# The constraints of the unified loop (Macklin et al. 2014) act on the reordered particles, the topology constraints
# reaching their particles through the reorder index. A constraint func accumulates its corrections and their count in
# `dpos` and `dcount`, which `func_apply_corrections` applies.


# ========================================== Layer 0: contact corrections ==========================================


@qd.func
def inverse_mass(is_free, mass: float):
    """Inverse mass of a particle, zero for a fixed one."""
    return is_free / mass


@qd.func
def neighbor_separation(x_i: qd.types.vector(3), x_j: qd.types.vector(3)):
    """Separation of a particle from a neighbour, the argument of the smoothing kernels."""
    return x_i - x_j


@qd.func
def mass_share(w_i: float, w_j: float):
    """Share of a particle of inverse mass w_i in the correction of its contact with a particle of inverse mass w_j."""
    return w_i / (w_i + w_j)


@qd.func
def particle_overlap(
    x_i: qd.types.vector(3),
    x_j: qd.types.vector(3),
    rest_i: qd.types.vector(3),
    rest_j: qd.types.vector(3),
    target: float,
):
    """Whether two particles, apart at rest, overlap within the target distance, with their separation and its
    length."""
    separation = x_i - x_j
    distance = separation.norm(gs.EPS)
    return distance < target and (rest_i - rest_j).norm(gs.EPS) > target, separation, distance


@qd.func
def separation_correction(separation: qd.types.vector(3), distance: float, target: float, share: float):
    """Correction of a particle that moves its share of the way that brings it to the target distance from its partner,
    along their separation."""
    return share * (target - distance) * (separation / distance)


@qd.func
def friction_correction(
    x_i: qd.types.vector(3),
    x0_i: qd.types.vector(3),
    x_j: qd.types.vector(3),
    x0_j: qd.types.vector(3),
    separation: qd.types.vector(3),
    distance: float,
    target: float,
    mu_s: float,
    mu_k: float,
    share: float,
):
    """Friction correction of a particle in contact (Macklin et al. 2014, Eq. 23-24): its share of its tangential
    displacement relative to its partner over the substep, undone within the static cone and limited by the kinetic one
    outside it."""
    normal = separation / distance
    dx = (x_i - x0_i) - (x_j - x0_j)
    tangential = -(dx - normal * normal.dot(dx))
    depth = target - distance
    correction = share * tangential
    if tangential.norm() >= mu_s * depth:
        correction = correction * qd.min(1.0, mu_k * depth / tangential.norm(gs.EPS))
    return correction


@qd.func
def is_collision_candidate(
    x_i: qd.types.vector(3),
    x_j: qd.types.vector(3),
    x0_i: qd.types.vector(3),
    x0_j: qd.types.vector(3),
    rest_i: qd.types.vector(3),
    rest_j: qd.types.vector(3),
    particle_size: float,
    reach: float,
):
    """Whether a particle, apart from another at rest, comes within the collision reach of it at its start or
    predicted position, with the nearer of those distances."""
    distance = qd.min((x_i - x_j).norm(), (x0_i - x0_j).norm())
    return (rest_i - rest_j).norm() > particle_size and distance < reach, distance


# ======================================== Layer 0: constraint projections ========================================


@qd.func
def compliance_alpha(compliance: float, dt: float):
    """Compliance of a constraint scaled by the substep, alpha = compliance / dt^2 (Macklin et al. 2016)."""
    return compliance / (dt**2)


@qd.func
def xpbd_delta_lambda(constraint: float, alpha: float, lam: float, denominator: float):
    """Increment of the Lagrange multiplier of a constraint, Delta lambda = (-C - alpha lambda) / (sum_i n_i w_i
    |grad C_i|^2 + alpha), the denominator holding the sum."""
    return (-constraint - alpha * lam) / denominator


@qd.func
def stretch_projection(
    x1: qd.types.vector(3),
    x2: qd.types.vector(3),
    w1: float,
    w2: float,
    count1: float,
    count2: float,
    rest_length: float,
    compliance: float,
    relaxation: float,
    lam: float,
    dt: float,
    use_xpbd: qd.template(),
):
    """Whether a stretch constraint projects, its new Lagrange multiplier, and the corrections of its two particles."""
    n = x1 - x2
    constraint = n.norm() - rest_length
    alpha = compliance_alpha(compliance, dt)
    is_projecting = False
    lam_new = lam
    dp = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(use_xpbd):
        # |grad C_i| = 1
        denominator = count1 * w1 + count2 * w2 + alpha
        if denominator > 0.0:
            is_projecting = True
            delta_lambda = xpbd_delta_lambda(constraint, alpha, lam, denominator)
            lam_new = lam + delta_lambda
            dp = delta_lambda * n / n.norm(gs.EPS)
    elif w1 + w2 + alpha > 0.0:
        is_projecting = True
        dp = -constraint / (w1 + w2 + alpha) * n / n.norm(gs.EPS) * relaxation
    return is_projecting, lam_new, dp * w1, -dp * w2


@qd.func
def bending_projection(
    positions: qd.types.matrix(4, 3),
    w: qd.types.vector(4),
    counts: qd.types.vector(4),
    compliance: float,
    relaxation: float,
    lam: float,
    dt: float,
    use_xpbd: qd.template(),
):
    """Whether a bending constraint projects, its new Lagrange multiplier, and the corrections of its four particles,
    row k of a 4 x 3 matrix (Mueller et al. 2007, Appendix A)."""
    is_projecting = w.sum() > 0.0
    lam_new = lam
    corrections = qd.Matrix.zero(gs.qd_float, 4, 3)
    if is_projecting:
        # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
        p2 = positions[1, :] - positions[0, :]
        p3 = positions[2, :] - positions[0, :]
        p4 = positions[3, :] - positions[0, :]
        l23 = p2.cross(p3).norm()
        l24 = p2.cross(p4).norm()
        n1 = p2.cross(p3) / l23
        n2 = p2.cross(p4) / l24
        d = qd.math.clamp(n1.dot(n2), -1.0, 1.0)
        # The angle between the normals by the arc tangent of its sine and cosine: at the flat state, where they are
        # opposite, the arc cosine of d turns the rounding of their unit length into a bend of the square root of the
        # precision
        sin_angle = n1.cross(n2).norm()

        # eq. (25), (26) and (27)
        q3 = (p2.cross(n2) + n1.cross(p2) * d) / l23
        q4 = (p2.cross(n1) + n2.cross(p2) * d) / l24
        q2 = -(p3.cross(n2) + n1.cross(p3) * d) / l23 - (p4.cross(n1) + n2.cross(p4) * d) / l24
        q1 = -q2 - q3 - q4
        q = qd.Matrix.rows([q1, q2, q3, q4])
        q_norm_sqr = qd.Vector([q1.norm_sqr(), q2.norm_sqr(), q3.norm_sqr(), q4.norm_sqr()], dt=gs.qd_float)
        constraint = qd.atan2(sin_angle, d) - qd.acos(-1.0)

        alpha = compliance_alpha(compliance, dt)
        if qd.static(use_xpbd):
            # grad C_i = q_i / sin, so with g = sin^2 and S = sum n_i w_i |q_i|^2:
            # Delta lambda = g (-C - alpha lambda) / (S + alpha g)
            # Delta x_i = w_i q_i Delta lambda / sin
            g = sin_angle**2
            sum_nwq = (counts * w).dot(q_norm_sqr)
            scale = gs.qd_float(0.0)
            if sum_nwq + alpha * g > 0.0:
                scale = xpbd_delta_lambda(constraint, alpha, lam, sum_nwq + alpha * g)
            lam_new = lam + g * scale
            constraint = sin_angle * scale
        else:
            # eq. (29)
            constraint = -sin_angle * constraint / (w.dot(q_norm_sqr) + alpha) * relaxation
        for k in qd.static(range(4)):
            correction = w[k] * constraint * q[k, :]
            for j in qd.static(range(3)):
                corrections[k, j] = correction[j]
    return is_projecting, lam_new, corrections


@qd.func
def volume_projection(
    positions: qd.types.matrix(4, 3),
    inv_masses: qd.types.vector(4),
    counts: qd.types.vector(4),
    rest_volume: float,
    compliance: float,
    relaxation: float,
    lam: float,
    dt: float,
    use_xpbd: qd.template(),
):
    """Whether a volume constraint projects, its new Lagrange multiplier, and the corrections of its four particles,
    row k of a 4 x 3 matrix, each along the gradient of the volume in its position."""
    p1 = positions[0, :]
    p2 = positions[1, :]
    p3 = positions[2, :]
    p4 = positions[3, :]
    grads = qd.Matrix.rows(
        [
            (p4 - p2).cross(p3 - p2) / 6.0,
            (p3 - p1).cross(p4 - p1) / 6.0,
            (p4 - p1).cross(p2 - p1) / 6.0,
            (p2 - p1).cross(p3 - p1) / 6.0,
        ]
    )
    # The inverse masses weighted by the squared norm of their gradient
    w = inv_masses * qd.Vector([grads[k, :].norm_sqr() for k in qd.static(range(4))], dt=gs.qd_float)
    is_projecting = w.sum() > 0.0
    lam_new = lam
    corrections = qd.Matrix.zero(gs.qd_float, 4, 3)
    if is_projecting:
        constraint = gu.qd_tet_vol(p1, p2, p3, p4) - rest_volume
        alpha = compliance_alpha(compliance, dt)
        # The correction of every vertex along its gradient
        scales = qd.Vector.zero(gs.qd_float, 4)
        if qd.static(use_xpbd):
            delta_lambda = xpbd_delta_lambda(constraint, alpha, lam, counts.dot(w) + alpha)
            lam_new = lam + delta_lambda
            # Delta x_i = w_i grad C_i Delta lambda, with the inverse mass alone
            scales = delta_lambda * inv_masses
        else:
            scales = (-constraint / (w.sum() + alpha) * relaxation) * w
        for k in qd.static(range(4)):
            correction = scales[k] * grads[k, :]
            for j in qd.static(range(3)):
                corrections[k, j] = correction[j]
    return is_projecting, lam_new, corrections


@qd.func
def density_neighbor_terms(poly6: float, mass_j: float, spiky: qd.types.vector(3), rho_rest: float):
    """Terms of a neighbour in the density constraint of a liquid particle (Macklin and Mueller 2013): its share of
    the density, its gradient of the constraint, and the square of that gradient."""
    s = spiky / rho_rest
    return poly6 * mass_j, s, s.dot(s)


@qd.func
def density_multiplier(rho: float, rho_rest: float, lower_sum: float, spiky_i: qd.types.vector(3), epsilon: float):
    """Lagrange multiplier of the density constraint rho / rho_rest - 1 of a liquid particle, relaxed by epsilon."""
    return -1.0 * ((rho / rho_rest - 1.0) / (lower_sum + spiky_i.dot(spiky_i) + epsilon))


@qd.func
def density_correction(
    lam_i: float,
    lam_j: float,
    scorr: float,
    spiky: qd.types.vector(3),
    rho_rest: float,
    dist_scale: float,
    relaxation: float,
):
    """Correction of a liquid particle by a neighbour of its density constraint, with the tensile correction scorr."""
    return (lam_i + lam_j + scorr) * spiky / rho_rest * dist_scale * relaxation


@qd.func
def averaged_correction(dp: qd.types.vector(3), n_corrections: float, sor_relaxation: float):
    """Over-relaxed mean of the corrections of a particle (Macklin et al. 2014, Eq. 13)."""
    return sor_relaxation / n_corrections * dp


@qd.func
def corrected_position(x: qd.types.vector(3), dp: qd.types.vector(3)):
    return x + dp


@qd.func
def correction_size(dp: qd.types.vector(3)):
    """Length of the correction of a particle, the measure of the convergence of the constraint loop."""
    return dp.norm()


@qd.func
def is_asleep(x: qd.types.vector(3), x0: qd.types.vector(3), sleep_threshold: float):
    """Whether a particle moved less than the sleep threshold over the substep (Macklin et al. 2014, Eq. 14)."""
    return (x - x0).norm() < sleep_threshold


# ================================================= Layer 1: stages =================================================


@qd.kernel
def kernel_count_constraints(pbd_solver: qd.template()):
    for i_p in range(pbd_solver._n_particles):
        pbd_solver.constraint_loop.constraint_counts[i_p] = qd.Vector.zero(gs.qd_float, 3)
    for i_e in range(pbd_solver._n_edges):
        edge = pbd_solver.edges_info[i_e]
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[edge.v1][0], 1.0)
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[edge.v2][0], 1.0)
    for i_ie in range(pbd_solver._n_inner_edges):
        info = pbd_solver.inner_edges_info[i_ie]
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[info.v1][1], 1.0)
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[info.v2][1], 1.0)
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[info.v3][1], 1.0)
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[info.v4][1], 1.0)
    for i_el in range(pbd_solver._n_elems):
        elem = pbd_solver.elems_info[i_el]
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[elem.v1][2], 1.0)
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[elem.v2][2], 1.0)
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[elem.v3][2], 1.0)
        qd.atomic_add(pbd_solver.constraint_loop.constraint_counts[elem.v4][2], 1.0)


@qd.kernel
def kernel_reset_lambda(pbd_solver: qd.template()):
    for i_e, i_b in qd.ndrange(pbd_solver._n_edges, pbd_solver._B):
        pbd_solver.constraint_loop.edges_lambda[i_e, i_b] = 0.0
    for i_ie, i_b in qd.ndrange(pbd_solver._n_inner_edges, pbd_solver._B):
        pbd_solver.constraint_loop.inner_edges_lambda[i_ie, i_b] = 0.0
    for i_el, i_b in qd.ndrange(pbd_solver._n_elems, pbd_solver._B):
        pbd_solver.constraint_loop.elems_lambda[i_el, i_b] = 0.0


@qd.func
def func_add_correction(i_r: int, i_b: int, dp: qd.types.vector(3), pbd_solver: qd.template()):
    qd.atomic_add(pbd_solver.particles_reordered[i_r, i_b].dpos, dp)
    qd.atomic_add(pbd_solver.particles_reordered[i_r, i_b].dcount, 1.0)


@qd.func
def func_inv_mass(i_r: int, i_b: int, pbd_solver: qd.template()):
    return inverse_mass(
        pbd_solver.particles_reordered[i_r, i_b].free, pbd_solver.particles_info_reordered[i_r, i_b].mass
    )


@qd.func
def func_within_budget(pbd_solver: qd.template(), budget: qd.template()):
    return pbd_solver.constraint_loop.n_iterations[()] < budget


@qd.func(requires_top_level=True)
def func_solve_stretch(f: int, pbd_solver: qd.template()):
    for i_e, i_b in qd.ndrange(pbd_solver._n_edges, pbd_solver._B):
        edge = pbd_solver.edges_info[i_e]
        is_constraint_active = (
            func_within_budget(pbd_solver, pbd_solver._max_stretch_solver_iterations)
            and pbd_solver.particles_ng[edge.v1, i_b].active
            and pbd_solver.particles_ng[edge.v2, i_b].active
        )
        if is_constraint_active:
            i_r1 = pbd_solver.particles_ng[edge.v1, i_b].reordered_idx
            i_r2 = pbd_solver.particles_ng[edge.v2, i_b].reordered_idx
            lam = gs.qd_float(0.0)
            if qd.static(pbd_solver._use_xpbd):
                lam = pbd_solver.constraint_loop.edges_lambda[i_e, i_b]
            is_projecting, lam_new, dp1, dp2 = stretch_projection(
                pbd_solver.particles_reordered[i_r1, i_b].pos,
                pbd_solver.particles_reordered[i_r2, i_b].pos,
                func_inv_mass(i_r1, i_b, pbd_solver),
                func_inv_mass(i_r2, i_b, pbd_solver),
                pbd_solver.constraint_loop.constraint_counts[edge.v1][0],
                pbd_solver.constraint_loop.constraint_counts[edge.v2][0],
                edge.len_rest,
                edge.stretch_compliance,
                edge.stretch_relaxation,
                lam,
                pbd_solver._substep_dt,
                pbd_solver._use_xpbd,
            )
            if is_projecting:
                if qd.static(pbd_solver._use_xpbd):
                    pbd_solver.constraint_loop.edges_lambda[i_e, i_b] = lam_new
                func_add_correction(i_r1, i_b, dp1, pbd_solver)
                func_add_correction(i_r2, i_b, dp2, pbd_solver)


@qd.func(requires_top_level=True)
def func_solve_bending(f: int, pbd_solver: qd.template()):
    for i_ie, i_b in qd.ndrange(pbd_solver._n_inner_edges, pbd_solver._B):
        info = pbd_solver.inner_edges_info[i_ie]
        vertices = qd.Vector([info.v1, info.v2, info.v3, info.v4], dt=gs.qd_int)
        is_constraint_active = func_within_budget(pbd_solver, pbd_solver._max_bending_solver_iterations)
        for k in qd.static(range(4)):
            is_constraint_active = is_constraint_active and pbd_solver.particles_ng[vertices[k], i_b].active
        if is_constraint_active:
            i_rs = qd.Vector(
                [pbd_solver.particles_ng[vertices[k], i_b].reordered_idx for k in qd.static(range(4))], dt=gs.qd_int
            )
            lam = gs.qd_float(0.0)
            if qd.static(pbd_solver._use_xpbd):
                lam = pbd_solver.constraint_loop.inner_edges_lambda[i_ie, i_b]
            is_projecting, lam_new, corrections = bending_projection(
                qd.Matrix.rows([pbd_solver.particles_reordered[i_rs[k], i_b].pos for k in qd.static(range(4))]),
                qd.Vector([func_inv_mass(i_rs[k], i_b, pbd_solver) for k in qd.static(range(4))], dt=gs.qd_float),
                qd.Vector(
                    [pbd_solver.constraint_loop.constraint_counts[vertices[k]][1] for k in qd.static(range(4))],
                    dt=gs.qd_float,
                ),
                info.bending_compliance,
                info.bending_relaxation,
                lam,
                pbd_solver._substep_dt,
                pbd_solver._use_xpbd,
            )
            if is_projecting:
                if qd.static(pbd_solver._use_xpbd):
                    pbd_solver.constraint_loop.inner_edges_lambda[i_ie, i_b] = lam_new
                for k in qd.static(range(4)):
                    func_add_correction(i_rs[k], i_b, corrections[k, :], pbd_solver)


@qd.func(requires_top_level=True)
def func_solve_volume(f: int, pbd_solver: qd.template()):
    for i_el, i_b in qd.ndrange(pbd_solver._n_elems, pbd_solver._B):
        elem = pbd_solver.elems_info[i_el]
        vertices = qd.Vector([elem.v1, elem.v2, elem.v3, elem.v4], dt=gs.qd_int)
        is_constraint_active = func_within_budget(pbd_solver, pbd_solver._max_volume_solver_iterations)
        for k in qd.static(range(4)):
            is_constraint_active = is_constraint_active and pbd_solver.particles_ng[vertices[k], i_b].active
        if is_constraint_active:
            i_rs = qd.Vector(
                [pbd_solver.particles_ng[vertices[k], i_b].reordered_idx for k in qd.static(range(4))], dt=gs.qd_int
            )
            lam = gs.qd_float(0.0)
            if qd.static(pbd_solver._use_xpbd):
                lam = pbd_solver.constraint_loop.elems_lambda[i_el, i_b]
            is_projecting, lam_new, corrections = volume_projection(
                qd.Matrix.rows([pbd_solver.particles_reordered[i_rs[k], i_b].pos for k in qd.static(range(4))]),
                qd.Vector([func_inv_mass(i_rs[k], i_b, pbd_solver) for k in qd.static(range(4))], dt=gs.qd_float),
                qd.Vector(
                    [pbd_solver.constraint_loop.constraint_counts[vertices[k]][2] for k in qd.static(range(4))],
                    dt=gs.qd_float,
                ),
                elem.vol_rest,
                elem.volume_compliance,
                elem.volume_relaxation,
                lam,
                pbd_solver._substep_dt,
                pbd_solver._use_xpbd,
            )
            if is_projecting:
                if qd.static(pbd_solver._use_xpbd):
                    pbd_solver.constraint_loop.elems_lambda[i_el, i_b] = lam_new
                for k in qd.static(range(4)):
                    func_add_correction(i_rs[k], i_b, corrections[k, :], pbd_solver)


@qd.func(requires_top_level=True)
def func_solve_density(f: int, pbd_solver: qd.template()):
    # ---Calculate lambdas---
    for i_p, i_b in qd.ndrange(pbd_solver._n_particles, pbd_solver._B):
        if (
            func_within_budget(pbd_solver, pbd_solver._max_density_solver_iterations)
            and pbd_solver.particles_info_reordered[i_p, i_b].material_type == pbd_solver.MATERIAL.LIQUID
        ):
            pos_i = pbd_solver.particles_reordered[i_p, i_b].pos
            rho_rest = pbd_solver.particles_info_reordered[i_p, i_b].rho_rest
            base = pbd_solver.sh.pos_to_grid(pos_i)
            lower_sum = gs.qd_float(0.0)
            rho = gs.qd_float(0.0)
            spiky_i = qd.Vector.zero(gs.qd_float, 3)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = pbd_solver.sh.grid_to_slot(base + offset)
                for j in range(
                    pbd_solver.sh.slot_start[slot_idx, i_b],
                    pbd_solver.sh.slot_size[slot_idx, i_b] + pbd_solver.sh.slot_start[slot_idx, i_b],
                ):
                    r = neighbor_separation(pos_i, pbd_solver.particles_reordered[j, i_b].pos)
                    rho_j, s, s_sqr = density_neighbor_terms(
                        pbd_solver.poly6(r),
                        pbd_solver.particles_info_reordered[j, i_b].mass,
                        pbd_solver.spiky(r),
                        rho_rest,
                    )
                    rho += rho_j
                    spiky_i += s
                    lower_sum += s_sqr
            pbd_solver.particles_reordered[i_p, i_b].lam = density_multiplier(
                rho, rho_rest, lower_sum, spiky_i, pbd_solver.lambda_epsilon
            )

    # ---Calculate delta pos---
    for i_p, i_b in qd.ndrange(pbd_solver._n_particles, pbd_solver._B):
        if (
            func_within_budget(pbd_solver, pbd_solver._max_density_solver_iterations)
            and pbd_solver.particles_info_reordered[i_p, i_b].material_type == pbd_solver.MATERIAL.LIQUID
        ):
            pos_i = pbd_solver.particles_reordered[i_p, i_b].pos
            base = pbd_solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = pbd_solver.sh.grid_to_slot(base + offset)
                for j in range(
                    pbd_solver.sh.slot_start[slot_idx, i_b],
                    pbd_solver.sh.slot_size[slot_idx, i_b] + pbd_solver.sh.slot_start[slot_idx, i_b],
                ):
                    if i_p != j:
                        r = neighbor_separation(pos_i, pbd_solver.particles_reordered[j, i_b].pos)
                        pbd_solver.particles_reordered[i_p, i_b].dpos = pbd_solver.particles_reordered[
                            i_p, i_b
                        ].dpos + density_correction(
                            pbd_solver.particles_reordered[i_p, i_b].lam,
                            pbd_solver.particles_reordered[j, i_b].lam,
                            pbd_solver.S_Corr(r),
                            pbd_solver.spiky(r),
                            pbd_solver.particles_info_reordered[i_p, i_b].rho_rest,
                            pbd_solver.dist_scale,
                            pbd_solver.particles_info_reordered[i_p, i_b].density_relaxation,
                        )

    for i_p, i_b in qd.ndrange(pbd_solver._n_particles, pbd_solver._B):
        if (
            func_within_budget(pbd_solver, pbd_solver._max_density_solver_iterations)
            and pbd_solver.particles_info_reordered[i_p, i_b].material_type == pbd_solver.MATERIAL.LIQUID
            and pbd_solver.particles_reordered[i_p, i_b].free
        ):
            pbd_solver.particles_reordered[i_p, i_b].pos = corrected_position(
                pbd_solver.particles_reordered[i_p, i_b].pos, pbd_solver.particles_reordered[i_p, i_b].dpos
            )
            pbd_solver.particles_reordered[i_p, i_b].dpos = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_gather_collision_neighbors(f: int, pbd_solver: qd.template()):
    """Gather the collision candidates of every particle for the substep, once the spatial hash holds the predicted
    positions."""
    for i_p, i_b in qd.ndrange(pbd_solver._n_particles, pbd_solver._B):
        pbd_solver.constraint_loop.n_collision_neighbors[i_p, i_b] = 0
        if pbd_solver.particles_info_reordered[i_p, i_b].material_type != pbd_solver.MATERIAL.PARTICLE:
            base = pbd_solver.sh.pos_to_grid(pbd_solver.particles_reordered[i_p, i_b].pos)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                i_slot = pbd_solver.sh.grid_to_slot(base + offset)
                for j_p in range(
                    pbd_solver.sh.slot_start[i_slot, i_b],
                    pbd_solver.sh.slot_size[i_slot, i_b] + pbd_solver.sh.slot_start[i_slot, i_b],
                ):
                    if (
                        i_p != j_p
                        and (
                            pbd_solver.particles_reordered[i_p, i_b].free
                            or pbd_solver.particles_reordered[j_p, i_b].free
                        )
                        and not (
                            pbd_solver.particles_info_reordered[i_p, i_b].material_type == pbd_solver.MATERIAL.LIQUID
                            and pbd_solver.particles_info_reordered[j_p, i_b].material_type
                            == pbd_solver.MATERIAL.LIQUID
                        )
                    ):
                        is_candidate, distance = is_collision_candidate(
                            pbd_solver.particles_reordered[i_p, i_b].pos,
                            pbd_solver.particles_reordered[j_p, i_b].pos,
                            pbd_solver.particles_reordered[i_p, i_b].ipos,
                            pbd_solver.particles_reordered[j_p, i_b].ipos,
                            pbd_solver.particles_info_reordered[i_p, i_b].pos_rest,
                            pbd_solver.particles_info_reordered[j_p, i_b].pos_rest,
                            pbd_solver._particle_size,
                            pbd_solver.constraint_loop.collision_reach[()],
                        )
                        if is_candidate:
                            func_insert_nearest(
                                i_p,
                                i_b,
                                j_p,
                                distance,
                                pbd_solver.constraint_loop.n_collision_neighbors,
                                pbd_solver.constraint_loop.collision_neighbors_idx,
                                pbd_solver.constraint_loop.collision_neighbors_dist,
                                COLLISION_NEIGHBORS,
                            )


@qd.func
def func_insert_nearest(
    i_p: int,
    i_b: int,
    j: int,
    distance: float,
    n_neighbors: qd.template(),
    neighbors_idx: qd.template(),
    neighbors_dist: qd.template(),
    capacity: qd.template(),
):
    """Keep a candidate among the `capacity` nearest neighbours of a particle, replacing the farthest one once they are
    full."""
    n_kept = n_neighbors[i_p, i_b]
    if n_kept < capacity:
        neighbors_idx[i_p, n_kept, i_b] = j
        neighbors_dist[i_p, n_kept, i_b] = distance
        n_neighbors[i_p, i_b] = n_kept + 1
    else:
        i_nb_max = 0
        for i_nb in range(1, capacity):
            if neighbors_dist[i_p, i_nb, i_b] > neighbors_dist[i_p, i_nb_max, i_b]:
                i_nb_max = i_nb
        if distance < neighbors_dist[i_p, i_nb_max, i_b]:
            neighbors_idx[i_p, i_nb_max, i_b] = j
            neighbors_dist[i_p, i_nb_max, i_b] = distance


@qd.func
def func_solve_collision_start(i: int, j: int, i_b: int, pbd_solver: qd.template()):
    """The non-penetration of two particles at their start positions, as a correction of the first, for the
    pre-stabilization."""
    # The target particle distance is 2 * particle radius, i.e. particle_size
    is_overlapping, separation, distance = particle_overlap(
        pbd_solver.particles_reordered[i, i_b].ipos,
        pbd_solver.particles_reordered[j, i_b].ipos,
        pbd_solver.particles_info_reordered[i, i_b].pos_rest,
        pbd_solver.particles_info_reordered[j, i_b].pos_rest,
        pbd_solver._particle_size,
    )
    if is_overlapping:
        share = mass_share(func_inv_mass(i, i_b, pbd_solver), func_inv_mass(j, i_b, pbd_solver))
        func_add_correction(
            i, i_b, separation_correction(separation, distance, pbd_solver._particle_size, share), pbd_solver
        )


@qd.func
def func_solve_collision_pair(i: int, j: int, i_b: int, pbd_solver: qd.template()):
    """The contact of two particles at their current positions, with friction, as a correction of the first."""
    particle_i = pbd_solver.particles_reordered[i, i_b]
    particle_j = pbd_solver.particles_reordered[j, i_b]
    info_i = pbd_solver.particles_info_reordered[i, i_b]
    info_j = pbd_solver.particles_info_reordered[j, i_b]
    # The target particle distance is 2 * particle radius, i.e. particle_size
    is_overlapping, separation, distance = particle_overlap(
        particle_i.pos, particle_j.pos, info_i.pos_rest, info_j.pos_rest, pbd_solver._particle_size
    )
    if is_overlapping:
        share = mass_share(func_inv_mass(i, i_b, pbd_solver), func_inv_mass(j, i_b, pbd_solver))
        func_add_correction(
            i, i_b, separation_correction(separation, distance, pbd_solver._particle_size, share), pbd_solver
        )
        friction = friction_correction(
            particle_i.pos,
            particle_i.ipos,
            particle_j.pos,
            particle_j.ipos,
            separation,
            distance,
            pbd_solver._particle_size,
            qd.max(info_i.mu_s, info_j.mu_s),
            qd.max(info_i.mu_k, info_j.mu_k),
            share,
        )
        # The thread of particle i alone corrects it in this loop
        pbd_solver.particles_reordered[i, i_b].dpos = pbd_solver.particles_reordered[i, i_b].dpos + friction


@qd.kernel
def kernel_solve_collision(f: int, pbd_solver: qd.template(), is_stabilization: qd.template()):
    func_solve_collision(f, pbd_solver, is_stabilization)


@qd.func(requires_top_level=True)
def func_solve_collision(f: int, pbd_solver: qd.template(), is_stabilization: qd.template()):
    """The particle contacts of every particle against its collision candidates of the substep, with friction (Macklin
    et al. 2014, Sec. 6.1), as a correction of the particle."""
    for i_p, i_b in qd.ndrange(pbd_solver._n_particles, pbd_solver._B):
        for i_nb in range(pbd_solver.constraint_loop.n_collision_neighbors[i_p, i_b]):
            j_p = pbd_solver.constraint_loop.collision_neighbors_idx[i_p, i_nb, i_b]
            if qd.static(is_stabilization):
                func_solve_collision_start(i_p, j_p, i_b, pbd_solver)
            else:
                func_solve_collision_pair(i_p, j_p, i_b, pbd_solver)


@qd.kernel
def kernel_solve_obstacles(f: int, pbd_solver: qd.template(), is_stabilization: qd.template()):
    if qd.static(is_stabilization):
        pbd_solver._particle_obstacles.func_stabilize(f, pbd_solver)
    else:
        pbd_solver._particle_obstacles.func_project(f, pbd_solver)


@qd.kernel
def kernel_apply_corrections(
    f: int, pbd_solver: qd.template(), is_average: qd.template(), is_stabilization: qd.template()
):
    func_apply_corrections(f, pbd_solver, is_average, is_stabilization)


@qd.func(requires_top_level=True)
def func_apply_corrections(
    f: int, pbd_solver: qd.template(), is_average: qd.template(), is_stabilization: qd.template()
):
    """Move every free particle by the corrections its constraints accumulated, and clear them.

    With `is_average`, the particle moves by the over-relaxed mean of its corrections (Macklin et al. 2014, Eq. 13),
    which the contacts take since they conflict under the parallel Jacobi update. Otherwise it moves by their sum, which
    the topology constraints take since their own relaxation factors keep them stable and their mean would soften a
    cloth by its number of edges per particle. With `is_stabilization`, the start position moves too, so that resolving
    a contact violated at the start of the substep adds no velocity (Sec. 4.4)."""
    for i_p, i_b in qd.ndrange(pbd_solver._n_particles, pbd_solver._B):
        n_c = pbd_solver.particles_reordered[i_p, i_b].dcount
        if (
            n_c > 0.0
            and pbd_solver.particles_reordered[i_p, i_b].free
            and pbd_solver.particles_info_reordered[i_p, i_b].material_type != pbd_solver.MATERIAL.PARTICLE
        ):
            dp = pbd_solver.particles_reordered[i_p, i_b].dpos
            if qd.static(is_average):
                dp = averaged_correction(dp, n_c, pbd_solver._sor_relaxation)
            pbd_solver.particles_reordered[i_p, i_b].pos = corrected_position(
                pbd_solver.particles_reordered[i_p, i_b].pos, dp
            )
            if qd.static(is_stabilization):
                pbd_solver.particles_reordered[i_p, i_b].ipos = corrected_position(
                    pbd_solver.particles_reordered[i_p, i_b].ipos, dp
                )
            elif qd.static(pbd_solver._position_tolerance > 0.0):
                qd.atomic_max(pbd_solver.constraint_loop.max_correction[()], correction_size(dp))
        pbd_solver.particles_reordered[i_p, i_b].dpos = qd.Vector.zero(gs.qd_float, 3)
        pbd_solver.particles_reordered[i_p, i_b].dcount = 0.0


@qd.kernel(graph=True)
def kernel_constraint_loop(f: int, pbd_solver: qd.template()):
    for _ in range(1):
        pbd_solver.constraint_loop.n_iterations[()] = 0
        pbd_solver.constraint_loop.is_running[()] = 1
    while qd.graph.do_while(pbd_solver.constraint_loop.is_running):
        if qd.static(pbd_solver._position_tolerance > 0.0):
            for _ in range(1):
                pbd_solver.constraint_loop.max_correction[()] = 0.0
        if qd.static(pbd_solver._n_edges > 0):
            func_solve_stretch(f, pbd_solver)
            func_apply_corrections(f, pbd_solver, is_average=False, is_stabilization=False)
        if qd.static(pbd_solver._n_inner_edges > 0):
            func_solve_bending(f, pbd_solver)
            func_apply_corrections(f, pbd_solver, is_average=False, is_stabilization=False)
        if qd.static(pbd_solver._n_elems > 0):
            func_solve_volume(f, pbd_solver)
            func_apply_corrections(f, pbd_solver, is_average=False, is_stabilization=False)
        if qd.static(pbd_solver._n_fluid_particles > 0):
            func_solve_density(f, pbd_solver)
        func_solve_collision(f, pbd_solver, is_stabilization=False)
        if qd.static(pbd_solver._has_particle_obstacles):
            pbd_solver._particle_obstacles.func_project(f, pbd_solver)
        func_apply_corrections(f, pbd_solver, is_average=True, is_stabilization=False)
        func_end_iteration(pbd_solver)


@qd.func(requires_top_level=True)
def func_end_iteration(pbd_solver: qd.template()):
    for _ in range(1):
        n_iterations = pbd_solver.constraint_loop.n_iterations[()] + 1
        pbd_solver.constraint_loop.n_iterations[()] = n_iterations
        is_done = n_iterations >= pbd_solver._solver_iterations
        if qd.static(pbd_solver._position_tolerance > 0.0):
            if has_converged(pbd_solver.constraint_loop.max_correction[()], pbd_solver._position_tolerance):
                is_done = True
        pbd_solver.constraint_loop.is_running[()] = 0 if is_done else 1


@qd.kernel
def kernel_sleep(f: int, pbd_solver: qd.template()):
    """Keep at its start position every particle that moved less than the sleep threshold over the substep, its
    velocity kept (Macklin et al. 2014, Eq. 14)."""
    for i_p, i_b in qd.ndrange(pbd_solver._n_particles, pbd_solver._B):
        ipos = pbd_solver.particles_reordered[i_p, i_b].ipos
        if is_asleep(pbd_solver.particles_reordered[i_p, i_b].pos, ipos, pbd_solver._sleep_threshold):
            pbd_solver.particles_reordered[i_p, i_b].pos = ipos
