import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary, func_impose_cube_boundary
from genesis.engine.entities import PBD2DEntity, PBD3DEntity, PBDFreeParticleEntity, PBDParticleEntity
from genesis.engine.entities.pbd_entity import PBDTetEntity
from genesis.engine.materials import PBD
from genesis.engine.state_change import StateChange, mutates
from genesis.options.solvers import PBDOptions
from genesis.utils.data import ENVS, Config, ConfigItem, Data, DataItem, DataKind, Size, eoa, field, iter_data
from genesis.utils.geom import (
    PARTICLES,
    SpatialHashGrid,
    func_hash_grid_to_slot,
    func_hash_pos_to_grid,
    func_hash_sort_particles,
)

from .base_solver import VFACES, VVERTS, SimulatorState
from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


EDGES = Size("edges")
INNER_EDGES = Size("inner_edges")
ELEMS = Size("elems")


class PBDParticles(Data, kind=DataKind.STATE, shape=(PARTICLES, ENVS)):
    """
    Runtime state of the position-based dynamics (PBD) particles, one value per particle and environment.

    `ipos` is the position at the start of the substep, `dpos` the correction of the position a constraint projection
    accumulates, and `lam` the Lagrange multiplier of the density constraint. A particle that is not free takes no
    correction, and follows what the user or a rigid link imposes.
    """

    pos: qd.Tensor = field((float, 3))
    vel: qd.Tensor = field((float, 3))
    ipos: qd.Tensor = field((float, 3))
    dpos: qd.Tensor = field((float, 3))
    lam: qd.Tensor = field(float)
    is_free: qd.Tensor = field(bool)
    is_active: qd.Tensor = field(bool)


class PBDParticlesInfo(Data, kind=DataKind.CONSTANT, shape=(PARTICLES,)):
    """
    Model parameters of the position-based dynamics (PBD) particles, one value per particle.

    `pos_rest` is the position at rest, `rho_rest` the rest density of a liquid, `mu_s` and `mu_k` the static and
    kinetic friction coefficients, and the relaxations scale the corrections of the density and viscosity constraints.
    """

    mass: qd.Tensor = field(float)
    pos_rest: qd.Tensor = field((float, 3))
    rho_rest: qd.Tensor = field(float)
    material_type: qd.Tensor = field(int)
    mu_s: qd.Tensor = field(float)
    mu_k: qd.Tensor = field(float)
    air_resistance: qd.Tensor = field(float)
    density_relaxation: qd.Tensor = field(float)
    viscosity_relaxation: qd.Tensor = field(float)


class PBDEdgesInfo(Data, kind=DataKind.CONSTANT, shape=(EDGES,)):
    """The stretch constraints, one per edge between the two particles `particles_idx`."""

    len_rest: qd.Tensor = field(float)
    stretch_compliance: qd.Tensor = field(float)
    stretch_relaxation: qd.Tensor = field(float)
    particles_idx: qd.Tensor = field((int, 2))


class PBDInnerEdgesInfo(Data, kind=DataKind.CONSTANT, shape=(INNER_EDGES,)):
    """
    The bending constraints, one per inner edge.

    The edge joins the first two particles of `particles_idx`, its adjacent faces closing on the last two.
    """

    len_rest: qd.Tensor = field(float)
    bending_compliance: qd.Tensor = field(float)
    bending_relaxation: qd.Tensor = field(float)
    particles_idx: qd.Tensor = field((int, 4))


class PBDElemsInfo(Data, kind=DataKind.CONSTANT, shape=(ELEMS,)):
    """The volume constraints, one per tetrahedron of the four particles `particles_idx`."""

    vol_rest: qd.Tensor = field(float)
    volume_compliance: qd.Tensor = field(float)
    volume_relaxation: qd.Tensor = field(float)
    particles_idx: qd.Tensor = field((int, 4))


class PBDState(Data, kind=DataKind.STATE):
    """
    Runtime state of the position-based dynamics (PBD) solver.

    The constraints of the topology run on the particles, those of their neighborhood on the copy of the particles
    sorted by hash grid cell, `particles_reordered`, which the particles take back at the end of the substep.
    """

    particles: PBDParticles
    particles_reordered: PBDParticles = field(kind=DataKind.SCRATCH)
    hash_grid: SpatialHashGrid


class PBDInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the position-based dynamics (PBD) solver.

    The smoothing kernels of the density and viscosity constraints take the distances between particles divided by
    `dist_scale`, which gives them a support radius of one, and `poly6_coef` and `spiky_coef` normalize them.
    `lambda_eps` regularizes the Lagrange multiplier of the density constraint, and `s_corr_delta_q` and `s_corr_k`
    shape its artificial pressure. `vvert_uvs` and `vfaces_vverts_idx` are the texture coordinates (UVs) of the visual
    vertices and the visual vertices of the faces.
    """

    particles: PBDParticlesInfo
    edges: PBDEdgesInfo
    inner_edges: PBDInnerEdgesInfo
    elems: PBDElemsInfo
    boundary: CubeBoundary
    vverts_uvs: qd.Tensor = field((float, 2), shape=(VVERTS,))
    vfaces_vverts_idx: qd.Tensor = field((int, 3), shape=(VFACES,))
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO)
    substep_dt: qd.Tensor = field(float, shape=())
    substep_dt_sq: qd.Tensor = field(float, shape=())
    particle_size: qd.Tensor = field(float, shape=())
    dist_scale: qd.Tensor = field(float, shape=())
    poly6_coef: qd.Tensor = field(float, shape=())
    spiky_coef: qd.Tensor = field(float, shape=())
    lambda_eps: qd.Tensor = field(float, shape=())
    s_corr_delta_q: qd.Tensor = field(float, shape=())
    s_corr_k: qd.Tensor = field(float, shape=())
    EPS: qd.Tensor = field(float, shape=())


class PBDConfig(Config):
    """
    The iterations of the constraint projections of the position-based dynamics (PBD) solver.

    Its kernels unroll them.
    """

    n_stretch_iterations: int
    n_bending_iterations: int
    n_volume_iterations: int
    n_density_iterations: int
    n_viscosity_iterations: int


class PBDSolver(ParticleSolver):
    material_cls = PBD.Base

    class MATERIAL(gs.IntEnum):
        CLOTH = 0
        ELASTIC = 1
        LIQUID = 2
        # Particles following no physics
        PARTICLE = 3

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: PBDOptions) -> None:
        super().__init__(scene, sim, options)

        self._max_stretch_solver_iterations = options.max_stretch_solver_iterations
        self._max_bending_solver_iterations = options.max_bending_solver_iterations
        self._max_volume_solver_iterations = options.max_volume_solver_iterations
        self._max_density_solver_iterations = options.max_density_solver_iterations
        self._max_viscosity_solver_iterations = options.max_viscosity_solver_iterations

        self._hash_grid_cell_size = options.hash_grid_cell_size
        self._hash_grid_res = options._hash_grid_res

        self._n_edges: int | None = None
        self._n_inner_edges: int | None = None
        self._n_elems: int | None = None

        self.pbd_state: PBDState | None = None
        self.pbd_info: PBDInfo | None = None
        self.pbd_config: PBDConfig | None = None

    def build(self) -> None:
        super().build()

        self._n_edges = self.n_edges
        self._n_inner_edges = self.n_inner_edges
        self._n_elems = self.n_elems

        is_unused = not self.is_active
        sizes = dict(envs=self._B, particles=self._n_particles)
        self.pbd_state = PBDState(
            sizes=dict(sizes, hash_slots=np.prod(self._hash_grid_res)),
            values=dict(hash_grid=dict(cell_size=self._hash_grid_cell_size, grid_res=self._hash_grid_res)),
            env_first=False,
            unused=is_unused,
        )
        info_values = {}
        if self.is_active:
            # The kernels of the density and viscosity constraints have a support radius of one once the distances are
            # divided by the scale.
            info_values = dict(
                boundary=dict(lower=self.boundary_lower, upper=self.boundary_upper, restitution=0.0),
                substep_dt=self.substep_dt,
                substep_dt_sq=self.substep_dt**2,
                particle_size=self._particle_size,
                dist_scale=self.particle_radius / 0.4,
                poly6_coef=315.0 / (64 * math.pi),
                spiky_coef=-45.0 / math.pi,
                lambda_eps=100.0,
                s_corr_delta_q=0.3,
                s_corr_k=0.0001,
                EPS=gs.EPS,
            )
        self.pbd_info = PBDInfo(
            sizes=dict(
                sizes,
                edges=self._n_edges,
                inner_edges=self._n_inner_edges,
                elems=self._n_elems,
                vverts=self._n_vverts,
                vfaces=self._n_vfaces,
            ),
            values=info_values,
            unused=is_unused,
        )
        self.pbd_config = PBDConfig(
            n_stretch_iterations=self._max_stretch_solver_iterations,
            n_bending_iterations=self._max_bending_solver_iterations,
            n_volume_iterations=self._max_volume_solver_iterations,
            n_density_iterations=self._max_density_solver_iterations,
            n_viscosity_iterations=self._max_viscosity_solver_iterations,
        )
        self._particles_pos = self.pbd_state.particles.pos
        self._particles_vel = self.pbd_state.particles.vel
        self._particles_is_active = self.pbd_state.particles.is_active
        self._particles_mass = self.pbd_info.particles.mass
        self._build_render(has_vverts=True)

        if self.is_active:
            for entity in self._entities:
                entity._add_to_solver()

        self._build_gravity(self.pbd_info.gravity)

    def add_uvs_and_faces(self, vvert_start: int, vface_start: int, uvs: np.ndarray, faces: np.ndarray) -> None:
        """
        Add the render texture coordinates (UVs) and faces of an entity.

        An empty `uvs` leaves the slice of the entity in the UV buffer zeroed.
        """
        kernel_add_uvs_and_faces(vvert_start, vface_start, uvs, faces, self.pbd_info)

    def add_particles(
        self,
        particle_start: int,
        is_active: bool,
        material_type: int,
        mass: float,
        rho_rest: float,
        mu_s: float,
        mu_k: float,
        air_resistance: float,
        density_relaxation: float,
        viscosity_relaxation: float,
        particles: np.ndarray,
    ) -> None:
        """Add the particles of an entity at rest, each property a material leaves out being given zero."""
        kernel_add_particles(
            particle_start,
            is_active,
            material_type,
            mass,
            rho_rest,
            mu_s,
            mu_k,
            air_resistance,
            density_relaxation,
            viscosity_relaxation,
            particles,
            self.pbd_state,
            self.pbd_info,
        )

    def add_edges(
        self,
        edge_start: int,
        particle_start: int,
        stretch_compliance: float,
        stretch_relaxation: float,
        edges: np.ndarray,
        edges_len_rest: np.ndarray,
    ) -> None:
        """Add the stretch constraints of the edges of an entity, given by the local indices of their particles."""
        kernel_add_edges(
            edge_start, particle_start, stretch_compliance, stretch_relaxation, edges, edges_len_rest, self.pbd_info
        )

    def add_inner_edges(
        self,
        inner_edge_start: int,
        particle_start: int,
        bending_compliance: float,
        bending_relaxation: float,
        inner_edges: np.ndarray,
        inner_edges_len_rest: np.ndarray,
    ) -> None:
        """Add the bending constraints of the inner edges of an entity, given by the local indices of its particles."""
        kernel_add_inner_edges(
            inner_edge_start,
            particle_start,
            bending_compliance,
            bending_relaxation,
            inner_edges,
            inner_edges_len_rest,
            self.pbd_info,
        )

    def add_elems(
        self,
        elem_start: int,
        particle_start: int,
        volume_compliance: float,
        volume_relaxation: float,
        elems: np.ndarray,
        elems_vol_rest: np.ndarray,
    ) -> None:
        """Add the volume constraints of the tetrahedra of an entity, given by the local indices of their particles."""
        kernel_add_elems(
            elem_start, particle_start, volume_compliance, volume_relaxation, elems, elems_vol_rest, self.pbd_info
        )

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

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
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        kernel_apply_external_force(self._sim.sim_state, self.pbd_state, self.pbd_info, self._force_fields)

        # The constraints of the topology need no spatial hashing
        if self._n_edges > 0:
            kernel_solve_stretch(self.pbd_state, self.pbd_info, self.pbd_config)
        if self._n_inner_edges > 0:
            kernel_solve_bending(self.pbd_state, self.pbd_info, self.pbd_config)
        if self._n_elems > 0:
            kernel_solve_volume(self.pbd_state, self.pbd_info, self.pbd_config)

        kernel_reorder_particles(self.pbd_state)
        kernel_solve_density(self.pbd_state, self.pbd_info, self.pbd_config)
        kernel_solve_viscosity(self.pbd_state, self.pbd_info, self.pbd_config)
        kernel_solve_collision(self.pbd_state, self.pbd_info)
        kernel_compute_velocity(self.pbd_state, self.pbd_info)

    def substep_post_coupling(self, f):
        if self.is_active:
            kernel_copy_from_reordered(self.pbd_state)
            kernel_solve_boundary_collision(self.pbd_state, self.pbd_info)

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        yield from iter_data(self, "pbd_info")
        yield from iter_data(self, "pbd_state")
        yield from iter_data(self, "pbd_config")

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state_render(self):
        """
        Get visual vertex positions, texture coordinates (UVs), and face indices for rendering.

        Returns
        -------
        tuple
            (vverts_pos, vverts_uvs, vfaces_indices) - vertex positions, UV coords, and triangle indices
        """
        if not self.is_active or self._n_vverts == 0:
            return None, None, None

        # Make sure render fields are up to date
        self.update_render_fields()

        return self.vverts_render.pos, self.pbd_info.vverts_uvs, self.pbd_info.vfaces_vverts_idx

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
        kernel_set_particles_free(particles_idx, envs_idx, is_free=False, pbd_state=self.pbd_state)

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
        self._sim.coupler.attach_pbd_particles_to_link(particles_idx, envs_idx, link_idx)

    @gs.assert_built
    @mutates(StateChange.DYNAMICS, differentiable=False)
    def release_particles(self, particles_idx=None, envs_idx=None):
        """Release some particles fixed in place or to a rigid link, which move freely again (see 'fix_particles')."""
        _, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        kernel_set_particles_free(particles_idx, envs_idx, is_free=True, pbd_state=self.pbd_state)
        self._sim.coupler.detach_pbd_particles(particles_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_fluid_particles(self) -> int:
        return sum([entity.n_fluid_particles for entity in self._entities if isinstance(entity, PBDParticleEntity)])

    @property
    def n_edges(self) -> int:
        if self.is_built:
            return self._n_edges
        return sum([entity.n_edges for entity in self._entities if isinstance(entity, PBDTetEntity)])

    @property
    def n_inner_edges(self) -> int:
        if self.is_built:
            return self._n_inner_edges
        return sum([entity.n_inner_edges for entity in self._entities if isinstance(entity, PBD2DEntity)])

    @property
    def n_elems(self) -> int:
        if self.is_built:
            return self._n_elems
        return sum([entity.n_elems for entity in self._entities if isinstance(entity, PBD3DEntity)])

    @property
    def hash_grid_res(self) -> tuple[int, int, int]:
        return self._hash_grid_res

    @property
    def hash_grid_cell_size(self) -> float:
        return self._hash_grid_cell_size


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_poly6(dist: float, pbd_info: PBDInfo):
    """Return the poly6 smoothing kernel at a distance divided by the scale of the distances (see PBDInfo)."""
    result = gs.qd_float(0.0)
    if 0 < dist < 1.0:
        rhs = (1.0 - dist * dist) * (1.0 - dist * dist) * (1.0 - dist * dist)
        result = pbd_info.poly6_coef[None] * rhs
    return result


@qd.func
def func_spiky(r: qd.types.vector(3), pbd_info: PBDInfo):
    """Return the gradient of the spiky smoothing kernel at a vector (see PBDInfo)."""
    result = qd.Vector.zero(gs.qd_float, 3)
    dist_scale = pbd_info.dist_scale[None]
    dist = r.norm() / dist_scale
    if 0 < dist < 1.0:
        m = (1.0 - dist) * (1.0 - dist)
        result = (pbd_info.spiky_coef[None] * m / dist) * r / dist_scale
    return result


@qd.func
def func_s_corr(r: qd.types.vector(3), poly6_ref: float, pbd_info: PBDInfo):
    """
    Return the artificial pressure of the density constraint between two particles at a vector.

    `poly6_ref` is the poly6 smoothing kernel at the reference distance of the artificial pressure (see PBDInfo).
    """
    m = func_poly6(r.norm() / pbd_info.dist_scale[None], pbd_info) / poly6_ref
    return -1.0 * pbd_info.s_corr_k[None] * m * m * m * m


@qd.func
def func_apply_dpos(i_p: int, i_b: int, particles: PBDParticles):
    """Apply the correction a constraint projection accumulated on a particle, then clear it."""
    particles.pos[i_p, i_b] = particles.pos[i_p, i_b] + particles.dpos[i_p, i_b]
    particles.dpos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_apply_external_force(
    sim_state: SimulatorState, pbd_state: PBDState, pbd_info: PBDInfo, force_fields: qd.template()
):
    """
    Advance the free particles by gravity, the force fields and the air resistance of a cloth, from where they start.

    The particles that are not free still move by their velocity, which carries those attached to a rigid link.
    """
    dt = pbd_info.substep_dt[None]
    for i_p, i_b in qd.ndrange(*pbd_state.particles.pos.shape, axes=qd.static(eoa(pbd_state.particles.ipos))):
        pbd_state.particles.ipos[i_p, i_b] = pbd_state.particles.pos[i_p, i_b]
        if pbd_state.particles.is_free[i_p, i_b]:
            vel = pbd_state.particles.vel[i_p, i_b] + pbd_info.gravity[i_b] * dt
            pos = pbd_state.particles.pos[i_p, i_b]
            acc = force_fields.add_acc(
                acc=qd.Vector.zero(gs.qd_float, 3), pos=pos, vel=vel, t=sim_state.t[i_b], i_p=i_p
            )
            vel = vel + acc * dt
            if pbd_info.particles.material_type[i_p] == PBDSolver.MATERIAL.CLOTH:
                f_air_resistance = pbd_info.particles.air_resistance[i_p] * vel.norm() * vel
                vel = vel - f_air_resistance / pbd_info.particles.mass[i_p] * dt
            pbd_state.particles.vel[i_p, i_b] = vel
        pbd_state.particles.pos[i_p, i_b] = pbd_state.particles.pos[i_p, i_b] + pbd_state.particles.vel[i_p, i_b] * dt


@qd.kernel(fastcache=True)
def kernel_solve_stretch(pbd_state: PBDState, pbd_info: PBDInfo, pbd_config: qd.template()):
    """Project the stretch constraints of the edges on the particles, the given number of times."""
    for _ in qd.static(range(pbd_config.n_stretch_iterations)):
        for i_e, i_b in qd.ndrange(
            pbd_info.edges.len_rest.shape[0],
            pbd_state.particles.pos.shape[1],
            axes=qd.static(eoa(pbd_state.particles.pos)),
        ):
            i_v1, i_v2 = pbd_info.edges.particles_idx[i_e]
            w1 = pbd_state.particles.is_free[i_v1, i_b] / pbd_info.particles.mass[i_v1]
            w2 = pbd_state.particles.is_free[i_v2, i_b] / pbd_info.particles.mass[i_v2]
            n = pbd_state.particles.pos[i_v1, i_b] - pbd_state.particles.pos[i_v2, i_b]
            C = n.norm() - pbd_info.edges.len_rest[i_e]
            alpha = pbd_info.edges.stretch_compliance[i_e] / pbd_info.substep_dt_sq[None]
            dp = -C / (w1 + w2 + alpha) * n / n.norm(pbd_info.EPS[None]) * pbd_info.edges.stretch_relaxation[i_e]
            qd.atomic_add(pbd_state.particles.dpos[i_v1, i_b], dp * w1)
            qd.atomic_add(pbd_state.particles.dpos[i_v2, i_b], -dp * w2)
        func_apply_topology_dpos(pbd_state, pbd_info)


@qd.kernel(fastcache=True)
def kernel_solve_bending(pbd_state: PBDState, pbd_info: PBDInfo, pbd_config: qd.template()):
    """
    Project the bending constraints of the inner edges on the particles, the given number of times.

    It follows the bending constraint projection of position based dynamics, extended by its compliance (XPBD).
    """
    for _ in qd.static(range(pbd_config.n_bending_iterations)):
        for i_ie, i_b in qd.ndrange(
            pbd_info.inner_edges.len_rest.shape[0],
            pbd_state.particles.pos.shape[1],
            axes=qd.static(eoa(pbd_state.particles.pos)),
        ):
            i_vs = pbd_info.inner_edges.particles_idx[i_ie]
            w = qd.Vector(
                [
                    pbd_state.particles.is_free[i_vs[k], i_b] / pbd_info.particles.mass[i_vs[k]]
                    for k in qd.static(range(4))
                ]
            )
            if w.sum() > 0.0:
                p1 = pbd_state.particles.pos[i_vs[0], i_b]
                p2 = pbd_state.particles.pos[i_vs[1], i_b] - p1
                p3 = pbd_state.particles.pos[i_vs[2], i_b] - p1
                p4 = pbd_state.particles.pos[i_vs[3], i_b] - p1
                l23 = p2.cross(p3).norm()
                l24 = p2.cross(p4).norm()
                n1 = p2.cross(p3) / l23
                n2 = p2.cross(p4) / l24
                d = qd.math.clamp(n1.dot(n2), -1.0, 1.0)

                q3 = (p2.cross(n2) + n1.cross(p2) * d) / l23
                q4 = (p2.cross(n1) + n2.cross(p2) * d) / l24
                q2 = -(p3.cross(n2) + n1.cross(p3) * d) / l23 - (p4.cross(n1) + n2.cross(p4) * d) / l24
                q1 = -q2 - q3 - q4
                sum_wq = w[0] * q1.norm_sqr() + w[1] * q2.norm_sqr() + w[2] * q3.norm_sqr() + w[3] * q4.norm_sqr()
                constraint = qd.acos(d) - qd.acos(-1.0)

                alpha = pbd_info.inner_edges.bending_compliance[i_ie] / pbd_info.substep_dt_sq[None]
                constraint = (
                    -qd.sqrt(1 - d**2) * constraint / (sum_wq + alpha) * pbd_info.inner_edges.bending_relaxation[i_ie]
                )

                qs = qd.Matrix.cols([q1, q2, q3, q4])
                for k in qd.static(range(4)):
                    qd.atomic_add(pbd_state.particles.dpos[i_vs[k], i_b], w[k] * constraint * qs[:, k])
        func_apply_topology_dpos(pbd_state, pbd_info)


@qd.kernel(fastcache=True)
def kernel_solve_volume(pbd_state: PBDState, pbd_info: PBDInfo, pbd_config: qd.template()):
    """Project the volume constraints of the tetrahedra on the particles, the given number of times."""
    for _ in qd.static(range(pbd_config.n_volume_iterations)):
        for i_el, i_b in qd.ndrange(
            pbd_info.elems.vol_rest.shape[0],
            pbd_state.particles.pos.shape[1],
            axes=qd.static(eoa(pbd_state.particles.pos)),
        ):
            i_vs = pbd_info.elems.particles_idx[i_el]
            p1 = pbd_state.particles.pos[i_vs[0], i_b]
            p2 = pbd_state.particles.pos[i_vs[1], i_b]
            p3 = pbd_state.particles.pos[i_vs[2], i_b]
            p4 = pbd_state.particles.pos[i_vs[3], i_b]

            grads = qd.Matrix.cols(
                [
                    (p4 - p2).cross(p3 - p2) / 6.0,
                    (p3 - p1).cross(p4 - p1) / 6.0,
                    (p4 - p1).cross(p2 - p1) / 6.0,
                    (p2 - p1).cross(p3 - p1) / 6.0,
                ]
            )
            w = qd.Vector(
                [
                    pbd_state.particles.is_free[i_vs[k], i_b]
                    / pbd_info.particles.mass[i_vs[k]]
                    * grads[:, k].norm_sqr()
                    for k in qd.static(range(4))
                ]
            )
            if w.sum() > 0.0:
                C = gu.qd_tet_vol(p1, p2, p3, p4) - pbd_info.elems.vol_rest[i_el]
                alpha = pbd_info.elems.volume_compliance[i_el] / pbd_info.substep_dt_sq[None]
                s = -C / (w.sum() + alpha) * pbd_info.elems.volume_relaxation[i_el]
                for k in qd.static(range(4)):
                    qd.atomic_add(pbd_state.particles.dpos[i_vs[k], i_b], s * w[k] * grads[:, k])
        func_apply_topology_dpos(pbd_state, pbd_info)


@qd.func
def func_apply_topology_dpos(pbd_state: PBDState, pbd_info: PBDInfo):
    """Apply the corrections of a projection of the constraints of the topology to the free physical particles."""
    for i_p, i_b in qd.ndrange(*pbd_state.particles.pos.shape, axes=qd.static(eoa(pbd_state.particles.is_free))):
        if (
            pbd_state.particles.is_free[i_p, i_b]
            and pbd_info.particles.material_type[i_p] != PBDSolver.MATERIAL.PARTICLE
        ):
            func_apply_dpos(i_p, i_b, pbd_state.particles)


@qd.func
def func_copy_particle(i_p_src: int, i_p_dst: int, i_b: int, particles_src: PBDParticles, particles_dst: PBDParticles):
    """Copy the state of a particle onto a slot of the same environment, its activity aside."""
    particles_dst.pos[i_p_dst, i_b] = particles_src.pos[i_p_src, i_b]
    particles_dst.vel[i_p_dst, i_b] = particles_src.vel[i_p_src, i_b]
    particles_dst.ipos[i_p_dst, i_b] = particles_src.ipos[i_p_src, i_b]
    particles_dst.dpos[i_p_dst, i_b] = particles_src.dpos[i_p_src, i_b]
    particles_dst.lam[i_p_dst, i_b] = particles_src.lam[i_p_src, i_b]
    particles_dst.is_free[i_p_dst, i_b] = particles_src.is_free[i_p_src, i_b]


@qd.kernel(fastcache=True)
def kernel_reorder_particles(pbd_state: PBDState):
    """Sort the particles by hash grid cell, into the copies that the neighborhood constraints run on."""
    func_hash_sort_particles(pbd_state.particles.pos, pbd_state.particles.is_active, pbd_state.hash_grid)

    # The active particles take the first sorted indices (see func_hash_sort_particles)
    for i_r, i_b in qd.ndrange(
        *pbd_state.particles.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.is_active))
    ):
        is_active = i_r < pbd_state.hash_grid.n_particles[i_b]
        pbd_state.particles_reordered.is_active[i_r, i_b] = is_active
        if is_active:
            i_p = pbd_state.hash_grid.particles_idx[i_r, i_b]
            func_copy_particle(i_p, i_r, i_b, pbd_state.particles, pbd_state.particles_reordered)


@qd.kernel(fastcache=True)
def kernel_copy_from_reordered(pbd_state: PBDState):
    """Give the active particles back the state of their sorted copy."""
    for i_r, i_b in qd.ndrange(
        *pbd_state.particles.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.is_active))
    ):
        if pbd_state.particles_reordered.is_active[i_r, i_b]:
            i_p = pbd_state.hash_grid.particles_idx[i_r, i_b]
            func_copy_particle(i_r, i_p, i_b, pbd_state.particles_reordered, pbd_state.particles)


@qd.func
def func_solve_collision(i_p: int, j_p: int, i_b: int, pbd_state: PBDState, pbd_info: PBDInfo):
    """Accumulate the correction of a sorted particle colliding with another one, friction included."""
    i_o = pbd_state.hash_grid.particles_idx[i_p, i_b]
    j_o = pbd_state.hash_grid.particles_idx[j_p, i_b]
    eps = pbd_info.EPS[None]
    pos_ij = pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.pos[j_p, i_b]
    cur_dist = pos_ij.norm(eps)
    rest_dist = (pbd_info.particles.pos_rest[i_o] - pbd_info.particles.pos_rest[j_o]).norm(eps)
    # The particles touch at twice their radius
    target_dist = pbd_info.particle_size[None]
    if cur_dist < target_dist and rest_dist > target_dist:
        wi = pbd_state.particles_reordered.is_free[i_p, i_b] / pbd_info.particles.mass[i_o]
        wj = pbd_state.particles_reordered.is_free[j_p, i_b] / pbd_info.particles.mass[j_o]
        n = pos_ij / cur_dist
        dpos = pbd_state.particles_reordered.dpos[i_p, i_b] + wi / (wi + wj) * (target_dist - cur_dist) * n

        # Friction of unified particle physics, from the tangential displacements of the particles over the substep
        dv = (pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.ipos[i_p, i_b]) - (
            pbd_state.particles_reordered.pos[j_p, i_b] - pbd_state.particles_reordered.ipos[j_p, i_b]
        )
        dpos_tan = -(dv - n * n.dot(dv))
        d = target_dist - cur_dist
        mu_s = qd.max(pbd_info.particles.mu_s[i_o], pbd_info.particles.mu_s[j_o])
        mu_k = qd.max(pbd_info.particles.mu_k[i_o], pbd_info.particles.mu_k[j_o])
        if dpos_tan.norm() < mu_s * d:
            dpos = dpos + wi / (wi + wj) * dpos_tan
        else:
            dpos = dpos + wi / (wi + wj) * dpos_tan * qd.min(1.0, mu_k * d / dpos_tan.norm(eps))
        pbd_state.particles_reordered.dpos[i_p, i_b] = dpos


@qd.kernel(fastcache=True)
def kernel_solve_collision(pbd_state: PBDState, pbd_info: PBDInfo):
    """Project the collisions of the sorted particles with their neighbors, two liquid particles never colliding."""
    for i_p, i_b in qd.ndrange(
        *pbd_state.particles_reordered.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.is_active))
    ):
        if pbd_state.particles_reordered.is_active[i_p, i_b]:
            i_o = pbd_state.hash_grid.particles_idx[i_p, i_b]
            material_type_i = pbd_info.particles.material_type[i_o]
            if material_type_i != PBDSolver.MATERIAL.PARTICLE:
                base = func_hash_pos_to_grid(pbd_state.particles_reordered.pos[i_p, i_b], pbd_state.hash_grid)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    i_s = func_hash_grid_to_slot(base + offset, pbd_state.hash_grid)
                    slot_start = pbd_state.hash_grid.slot_start[i_s, i_b]
                    for j_p in range(slot_start, pbd_state.hash_grid.slot_size[i_s, i_b] + slot_start):
                        j_o = pbd_state.hash_grid.particles_idx[j_p, i_b]
                        is_liquid_pair = (
                            material_type_i == PBDSolver.MATERIAL.LIQUID
                            and pbd_info.particles.material_type[j_o] == PBDSolver.MATERIAL.LIQUID
                        )
                        is_any_free = (
                            pbd_state.particles_reordered.is_free[i_p, i_b]
                            or pbd_state.particles_reordered.is_free[j_p, i_b]
                        )
                        if i_p != j_p and is_any_free and not is_liquid_pair:
                            func_solve_collision(i_p, j_p, i_b, pbd_state, pbd_info)

    for i_p, i_b in qd.ndrange(
        *pbd_state.particles_reordered.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.is_active))
    ):
        if pbd_state.particles_reordered.is_active[i_p, i_b]:
            i_o = pbd_state.hash_grid.particles_idx[i_p, i_b]
            if (
                pbd_state.particles_reordered.is_free[i_p, i_b]
                and pbd_info.particles.material_type[i_o] != PBDSolver.MATERIAL.PARTICLE
            ):
                func_apply_dpos(i_p, i_b, pbd_state.particles_reordered)


@qd.kernel(fastcache=True)
def kernel_solve_boundary_collision(pbd_state: PBDState, pbd_info: PBDInfo):
    """Keep every particle inside the boundary, whether free or not."""
    for i_p, i_b in qd.ndrange(*pbd_state.particles.pos.shape, axes=qd.static(eoa(pbd_state.particles.pos))):
        pos, vel = func_impose_cube_boundary(
            pbd_state.particles.pos[i_p, i_b], pbd_state.particles.vel[i_p, i_b], pbd_info.boundary
        )
        pbd_state.particles.pos[i_p, i_b] = pos
        pbd_state.particles.vel[i_p, i_b] = vel


@qd.kernel(fastcache=True)
def kernel_solve_density(pbd_state: PBDState, pbd_info: PBDInfo, pbd_config: qd.template()):
    """
    Project the density constraints of the sorted liquid particles, the given number of times.

    It follows position based fluids: a Lagrange multiplier per particle, then the correction of its position, with an
    artificial pressure that keeps the particles from clustering.
    """
    for _ in qd.static(range(pbd_config.n_density_iterations)):
        for i_p, i_b in qd.ndrange(
            *pbd_state.particles_reordered.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.pos))
        ):
            i_o = pbd_state.hash_grid.particles_idx[i_p, i_b]
            if (
                pbd_state.particles_reordered.is_active[i_p, i_b]
                and pbd_info.particles.material_type[i_o] == PBDSolver.MATERIAL.LIQUID
            ):
                pos_i = pbd_state.particles_reordered.pos[i_p, i_b]
                rho_rest = pbd_info.particles.rho_rest[i_o]
                dist_scale = pbd_info.dist_scale[None]
                base = func_hash_pos_to_grid(pos_i, pbd_state.hash_grid)
                lower_sum = gs.qd_float(0.0)
                rho = gs.qd_float(0.0)
                spiky_i = qd.Vector.zero(gs.qd_float, 3)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    i_s = func_hash_grid_to_slot(base + offset, pbd_state.hash_grid)
                    slot_start = pbd_state.hash_grid.slot_start[i_s, i_b]
                    for j_p in range(slot_start, pbd_state.hash_grid.slot_size[i_s, i_b] + slot_start):
                        j_o = pbd_state.hash_grid.particles_idx[j_p, i_b]
                        r = pos_i - pbd_state.particles_reordered.pos[j_p, i_b]
                        rho += func_poly6(r.norm() / dist_scale, pbd_info) * pbd_info.particles.mass[j_o]
                        s = func_spiky(r, pbd_info) / rho_rest
                        spiky_i += s
                        lower_sum += s.dot(s)
                constraint = (rho / rho_rest) - 1.0
                lower_sum += spiky_i.dot(spiky_i)
                pbd_state.particles_reordered.lam[i_p, i_b] = -1.0 * (
                    constraint / (lower_sum + pbd_info.lambda_eps[None])
                )

        for i_p, i_b in qd.ndrange(
            *pbd_state.particles_reordered.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.pos))
        ):
            i_o = pbd_state.hash_grid.particles_idx[i_p, i_b]
            if (
                pbd_state.particles_reordered.is_active[i_p, i_b]
                and pbd_info.particles.material_type[i_o] == PBDSolver.MATERIAL.LIQUID
            ):
                # What the particle contributes to the correction it takes from each neighbor, read once for all of them
                pos_i = pbd_state.particles_reordered.pos[i_p, i_b]
                lam_i = pbd_state.particles_reordered.lam[i_p, i_b]
                rho_rest = pbd_info.particles.rho_rest[i_o]
                dist_scale = pbd_info.dist_scale[None]
                density_relaxation = pbd_info.particles.density_relaxation[i_o]
                poly6_ref = func_poly6(pbd_info.s_corr_delta_q[None], pbd_info)
                base = func_hash_pos_to_grid(pos_i, pbd_state.hash_grid)
                dpos = pbd_state.particles_reordered.dpos[i_p, i_b]
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    i_s = func_hash_grid_to_slot(base + offset, pbd_state.hash_grid)
                    slot_start = pbd_state.hash_grid.slot_start[i_s, i_b]
                    for j_p in range(slot_start, pbd_state.hash_grid.slot_size[i_s, i_b] + slot_start):
                        if i_p != j_p:
                            r = pos_i - pbd_state.particles_reordered.pos[j_p, i_b]
                            lam_sum = (
                                lam_i
                                + pbd_state.particles_reordered.lam[j_p, i_b]
                                + func_s_corr(r, poly6_ref, pbd_info)
                            )
                            dpos = dpos + lam_sum * func_spiky(r, pbd_info) / rho_rest * dist_scale * density_relaxation
                pbd_state.particles_reordered.dpos[i_p, i_b] = dpos

        func_apply_liquid_dpos(pbd_state, pbd_info)


@qd.kernel(fastcache=True)
def kernel_solve_viscosity(pbd_state: PBDState, pbd_info: PBDInfo, pbd_config: qd.template()):
    """Project the viscosity of the sorted liquid particles, the given number of times, smoothing their velocity."""
    for _ in qd.static(range(pbd_config.n_viscosity_iterations)):
        for i_p, i_b in qd.ndrange(
            *pbd_state.particles_reordered.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.pos))
        ):
            i_o = pbd_state.hash_grid.particles_idx[i_p, i_b]
            if (
                pbd_state.particles_reordered.is_active[i_p, i_b]
                and pbd_info.particles.material_type[i_o] == PBDSolver.MATERIAL.LIQUID
            ):
                # The displacement of the particle over the substep, read once for all its neighbors
                pos_i = pbd_state.particles_reordered.pos[i_p, i_b]
                dpos_i = pos_i - pbd_state.particles_reordered.ipos[i_p, i_b]
                dist_scale = pbd_info.dist_scale[None]
                base = func_hash_pos_to_grid(pos_i, pbd_state.hash_grid)
                xsph_sum = qd.Vector.zero(gs.qd_float, 3)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    i_s = func_hash_grid_to_slot(base + offset, pbd_state.hash_grid)
                    slot_start = pbd_state.hash_grid.slot_start[i_s, i_b]
                    for j_p in range(slot_start, pbd_state.hash_grid.slot_size[i_s, i_b] + slot_start):
                        pos_j = pbd_state.particles_reordered.pos[j_p, i_b]
                        v_ij = (pos_j - pbd_state.particles_reordered.ipos[j_p, i_b]) - dpos_i
                        xsph_sum += func_poly6((pos_i - pos_j).norm() / dist_scale, pbd_info) * v_ij
                pbd_state.particles_reordered.dpos[i_p, i_b] = (
                    pbd_state.particles_reordered.dpos[i_p, i_b]
                    + xsph_sum * pbd_info.particles.viscosity_relaxation[i_o]
                )

        func_apply_liquid_dpos(pbd_state, pbd_info)


@qd.func
def func_apply_liquid_dpos(pbd_state: PBDState, pbd_info: PBDInfo):
    """Apply the corrections of a projection of the constraints of the liquids to the free sorted liquid particles."""
    for i_p, i_b in qd.ndrange(
        *pbd_state.particles_reordered.pos.shape, axes=qd.static(eoa(pbd_state.hash_grid.particles_idx))
    ):
        i_o = pbd_state.hash_grid.particles_idx[i_p, i_b]
        if (
            pbd_state.particles_reordered.is_active[i_p, i_b]
            and pbd_state.particles_reordered.is_free[i_p, i_b]
            and pbd_info.particles.material_type[i_o] == PBDSolver.MATERIAL.LIQUID
        ):
            func_apply_dpos(i_p, i_b, pbd_state.particles_reordered)


@qd.kernel(fastcache=True)
def kernel_compute_velocity(pbd_state: PBDState, pbd_info: PBDInfo):
    """Set the velocity of the sorted particles to their displacement over the substep."""
    for i_p, i_b in qd.ndrange(
        *pbd_state.particles_reordered.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.vel))
    ):
        if pbd_state.particles_reordered.is_active[i_p, i_b]:
            pbd_state.particles_reordered.vel[i_p, i_b] = (
                pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.ipos[i_p, i_b]
            ) / pbd_info.substep_dt[None]


# ------------------------------------------------------------------------------------
# --------------------------------------- io -----------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_add_uvs_and_faces(
    vvert_start: int,
    vface_start: int,
    uvs: qd.types.ndarray(element_dim=1),
    faces: qd.types.ndarray(element_dim=1),
    pbd_info: PBDInfo,
):
    for i_vv_ in range(uvs.shape[0]):
        pbd_info.vverts_uvs[i_vv_ + vvert_start] = uvs[i_vv_]
    for i_vf_ in range(faces.shape[0]):
        pbd_info.vfaces_vverts_idx[i_vf_ + vface_start] = faces[i_vf_] + vvert_start


@qd.kernel(fastcache=True)
def kernel_add_particles(
    particle_start: int,
    is_active: bool,
    material_type: int,
    mass: float,
    rho_rest: float,
    mu_s: float,
    mu_k: float,
    air_resistance: float,
    density_relaxation: float,
    viscosity_relaxation: float,
    particles: qd.types.ndarray(),
    pbd_state: PBDState,
    pbd_info: PBDInfo,
):
    for i_p_ in range(particles.shape[0]):
        i_p = i_p_ + particle_start
        pbd_info.particles.pos_rest[i_p] = qd.Vector([particles[i_p_, 0], particles[i_p_, 1], particles[i_p_, 2]])
        pbd_info.particles.material_type[i_p] = material_type
        pbd_info.particles.mass[i_p] = mass
        pbd_info.particles.rho_rest[i_p] = rho_rest
        pbd_info.particles.mu_s[i_p] = mu_s
        pbd_info.particles.mu_k[i_p] = mu_k
        pbd_info.particles.air_resistance[i_p] = air_resistance
        pbd_info.particles.density_relaxation[i_p] = density_relaxation
        pbd_info.particles.viscosity_relaxation[i_p] = viscosity_relaxation

    for i_p_, i_b in qd.ndrange(
        particles.shape[0], pbd_state.particles.pos.shape[1], axes=qd.static(eoa(pbd_state.particles.pos))
    ):
        i_p = i_p_ + particle_start
        pbd_state.particles.pos[i_p, i_b] = qd.Vector([particles[i_p_, 0], particles[i_p_, 1], particles[i_p_, 2]])
        pbd_state.particles.vel[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        pbd_state.particles.dpos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        pbd_state.particles.is_free[i_p, i_b] = True
        pbd_state.particles.is_active[i_p, i_b] = is_active


@qd.kernel(fastcache=True)
def kernel_add_edges(
    edge_start: int,
    particle_start: int,
    stretch_compliance: float,
    stretch_relaxation: float,
    edges: qd.types.ndarray(),
    edges_len_rest: qd.types.ndarray(),
    pbd_info: PBDInfo,
):
    for i_e_ in range(edges.shape[0]):
        i_e = i_e_ + edge_start
        pbd_info.edges.stretch_compliance[i_e] = stretch_compliance
        pbd_info.edges.stretch_relaxation[i_e] = stretch_relaxation
        pbd_info.edges.len_rest[i_e] = edges_len_rest[i_e_]
        pbd_info.edges.particles_idx[i_e] = qd.Vector([edges[i_e_, 0], edges[i_e_, 1]]) + particle_start


@qd.kernel(fastcache=True)
def kernel_add_inner_edges(
    inner_edge_start: int,
    particle_start: int,
    bending_compliance: float,
    bending_relaxation: float,
    inner_edges: qd.types.ndarray(),
    inner_edges_len_rest: qd.types.ndarray(),
    pbd_info: PBDInfo,
):
    for i_ie_ in range(inner_edges.shape[0]):
        i_ie = i_ie_ + inner_edge_start
        pbd_info.inner_edges.bending_compliance[i_ie] = bending_compliance
        pbd_info.inner_edges.bending_relaxation[i_ie] = bending_relaxation
        pbd_info.inner_edges.len_rest[i_ie] = inner_edges_len_rest[i_ie_]
        pbd_info.inner_edges.particles_idx[i_ie] = (
            qd.Vector([inner_edges[i_ie_, k] for k in qd.static(range(4))]) + particle_start
        )


@qd.kernel(fastcache=True)
def kernel_add_elems(
    elem_start: int,
    particle_start: int,
    volume_compliance: float,
    volume_relaxation: float,
    elems: qd.types.ndarray(),
    elems_vol_rest: qd.types.ndarray(),
    pbd_info: PBDInfo,
):
    for i_el_ in range(elems.shape[0]):
        i_el = i_el_ + elem_start
        pbd_info.elems.volume_compliance[i_el] = volume_compliance
        pbd_info.elems.volume_relaxation[i_el] = volume_relaxation
        pbd_info.elems.vol_rest[i_el] = elems_vol_rest[i_el_]
        pbd_info.elems.particles_idx[i_el] = qd.Vector([elems[i_el_, k] for k in qd.static(range(4))]) + particle_start


@qd.kernel(fastcache=True)
def kernel_set_particles_free(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), is_free: bool, pbd_state: PBDState
):
    for i_p_, i_b_ in qd.ndrange(
        particles_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(pbd_state.particles.is_free))
    ):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        pbd_state.particles.is_free[i_p, i_b] = is_free
