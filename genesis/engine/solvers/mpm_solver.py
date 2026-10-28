from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import MPMEntity
from genesis.engine.materials import MPM
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import MPMSolverState
from genesis.grad.bridge import no_grad, register_custom_backward
from genesis.options.solvers import MPMOptions
from genesis.utils.misc import DeprecationError, broadcast_tensor, qd_isfinite, qd_to_torch

from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


@qd.data_oriented
class MPMSolver(ParticleSolver):
    material_cls = MPM.Base
    has_frames = True

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: "MPMOptions"):
        super().__init__(scene, sim, options)

        # options
        self._grid_density = options.grid_density
        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._enable_CPIC = options.enable_CPIC
        self._constraints_initialized = False

        self._n_vvert_supports = self.scene.options.vis.n_support_neighbors

        # `_particle_volume_scale` is used to avoid potential numerical instability, as the actual `_particle_volume` may be very small.
        # Note that the magnitude of `_particle_volume` doesn't affect MPM simulation itself, but it is used to compute particle
        # mass. We need to account for this scale when handling coupling.
        self._particle_volume_real = float(self._particle_size**3)
        self._particle_volume_scale = 1e3
        self._particle_volume = self._particle_volume_real * self._particle_volume_scale

        # Other derived parameters
        self._dx = float(1.0 / self._grid_density)
        self._inv_dx = float(self._grid_density)
        self._lower_bound_cell = np.round(self._grid_density * self._lower_bound).astype(gs.np_int)
        self._upper_bound_cell = np.round(self._grid_density * self._upper_bound).astype(gs.np_int)
        self._grid_res = self._upper_bound_cell - self._lower_bound_cell + 1  # +1 to include both corner
        self._grid_offset = qd.Vector(self._lower_bound_cell)
        if np.prod(self._grid_res) > 1e9:
            gs.raise_exception(
                "Grid size larger than 1e9 not supported by MPM solver. Please reduce 'grid_density', or set tighter "
                "boundaries via 'lower_bound' / 'upper_bound'."
            )

        # materials
        self._materials = list()
        self._materials_idx = list()
        self._materials_update_F_S_Jp = list()
        self._materials_update_stress = list()

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        # safety padding
        self.boundary_padding = 3 * self._dx
        self.boundary = CubeBoundary(
            lower=self._lower_bound + self.boundary_padding, upper=self._upper_bound - self.boundary_padding
        )

    def init_particle_fields(self):
        # dynamic particle state
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            C=gs.qd_mat3,  # affine velocity field
            F=gs.qd_mat3,  # deformation gradient
            F_tmp=gs.qd_mat3,  # temp deformation gradient
            U=gs.qd_mat3,  # SVD
            V=gs.qd_mat3,  # SVD
            S=gs.qd_mat3,  # SVD
            actu=gs.qd_float,  # actuation
            Jp=gs.qd_float,  # volume ratio
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = qd.types.struct(active=gs.qd_bool)

        # static particle info
        struct_particle_info = qd.types.struct(
            material_idx=gs.qd_int,
            mass=gs.qd_float,
            default_Jp=gs.qd_float,
            free=gs.qd_bool,
            # for muscle
            muscle_group=gs.qd_int,
            muscle_direction=gs.qd_vec3,
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(pos=gs.qd_vec3, vel=gs.qd_vec3, active=gs.qd_bool)

        # construct fields
        self.particles = struct_particle_state.field(
            shape=(self.n_frames, self._n_particles, self._B), needs_grad=True, layout=qd.Layout.SOA
        )
        self.particles_ng = struct_particle_state_ng.field(
            shape=(self.n_frames, self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_info = struct_particle_info.field(
            shape=self._n_particles, needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

    def init_grid_fields(self):
        grid_cell_state = qd.types.struct(
            mass=gs.qd_float,  # mass
            vel_in=gs.qd_vec3,  # input momentum/velocity
            vel_out=gs.qd_vec3,  # output momentum/velocity
        )
        # Grid is only ever indexed at [f] (never [f+1]) in p2g/g2p/reset/coupler, so one frame per substep is enough.
        # Particles still need one more, because g2p writes the next-frame state at [f+1].
        # The errors of each env, raised at the beginning of the next step (see 'Solver.register_errors')
        self.errno = qd.field(gs.qd_int, shape=(self._B,))
        self.register_errors(
            self.errno,
            {
                array_class.ErrorCode.INVALID_PARTICLE_NAN: (
                    "NaN detected in MPM states. Try reducing the time step size or adjusting simulation parameters."
                ),
            },
        )

        self.grid = grid_cell_state.field(
            shape=(self.n_frames - 1, *self._grid_res, self._B), needs_grad=True, layout=qd.Layout.SOA
        )

        # Sparse-reset bookkeeping for forward-only mode. A single global dirty list captures unique cells touched by
        # p2g across all envs in the current substep; reset_dirty_cells zeroes those cells across all envs immediately
        # after g2p, so the grid is always zero at the start of each substep and no per-substep state has to outlive a
        # substep. grid_dirty_flag is a per-cell tristate (env-shared) that deduplicates appends: only the first env to
        # touch a cell records it. List size is bounded by the total grid cell count because dedup ensures no more than
        # that many unique entries.
        # TODO: support sparse reset under requires_grad. Quadrants' differentiable framework needs the grid state at
        # every intermediate substep for the backward pass, so we cannot eagerly wipe cells; revisit if checkpointing
        # or selective grad masking becomes available.
        if not self._sim.requires_grad:
            self._grid_total = int(np.prod(self._grid_res))
            self.grid_dirty_flag = qd.field(gs.qd_int, shape=self._grid_res)
            self.grid_dirty_list = qd.field(gs.qd_int, shape=(self._grid_total,))
            self.grid_dirty_count = qd.field(gs.qd_int, shape=(1,))

    def init_vvert_fields(self):
        struct_vvert_info = qd.types.struct(
            support_idxs=qd.types.vector(self._n_vvert_supports, gs.qd_int),
            support_weights=qd.types.vector(self._n_vvert_supports, gs.qd_float),
        )
        self.vverts_info = struct_vvert_info.field(shape=(max(1, self._n_vverts),), layout=qd.Layout.SOA)

        struct_vvert_state_render = qd.types.struct(pos=gs.qd_vec3, active=gs.qd_bool)
        self.vverts_render = struct_vvert_state_render.field(
            shape=(max(1, self._n_vverts), self._B), layout=qd.Layout.SOA
        )

    def init_constraints(self):
        """Lazy initialization of particle constraint fields."""
        # Memory check: ensure index fits in int32
        if self._n_particles * self._B * 3 > np.iinfo(np.int32).max:
            gs.raise_exception(
                f"Particle constraint shape (n_envs={self._B}, n_particles={self._n_particles}, 3) is too large. "
                "Consider reducing n_envs or n_particles."
            )

        self._constraints_initialized = True

        particle_constraint_info = qd.types.struct(
            is_constrained=gs.qd_bool,  # whether particle is constrained
            target_pos=gs.qd_vec3,  # target position for the constraint
            stiffness=gs.qd_float,  # spring stiffness
            link_idx=gs.qd_int,  # index of the rigid link (-1 if not linked)
            link_local_pos=gs.qd_vec3,  # offset from link origin in link's local frame
        )

        # FIXME: AOS, which does not match other Genesis structs. Old, untested code. We prefer not to touch for now.
        self.particle_constraints = particle_constraint_info.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.AOS
        )

        self.particle_constraints.is_constrained.fill(False)
        self.particle_constraints.link_idx.fill(-1)

    def reset_grad(self):
        self.particles.grad.fill(0.0)
        self.grid.grad.fill(0.0)

    def build(self):
        super().build()

        # particles and entities
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        self._coupler = self.sim._coupler

        # Aggregate SVD requirement across registered materials. If no material reads U/V/S, the solver can skip the SVD
        # kernel entirely on the forward pass and derive J from det(F_tmp) directly. Compile-time constant consumed by
        # p2g via qd.static. Forced True under requires_grad so the backward path stays exactly as upstream.
        # TODO: validate the J = det(F_tmp) shortcut composes with quadrants autograd before enabling it there.
        self.needs_svd = self._sim.requires_grad or any(m.needs_svd for m in self._materials)

        if self.is_active:
            if self._enable_CPIC and self._sim.requires_grad:
                gs.raise_exception(
                    "CPIC is not supported in differentiable mode yet. Submit a feature request if you need it."
                )

            self.init_particle_fields()
            self.init_grid_fields()
            self.init_vvert_fields()

            for entity in self._entities:
                entity._add_to_solver()

            # See: https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L84
            suggested_dt = 2e-2 * self._dx
            if self.substep_dt > suggested_dt:
                gs.logger.warning(
                    f"Current `substep_dt` ({self.substep_dt:.6g}) is greater than suggested_dt ({suggested_dt:.6g}, "
                    "calculated based on `grid_density`). Simulation might be unstable."
                )

        # Kernels of this solver take the solver itself, so gravity has to be a field for them.
        self._build_gravity(as_field=True)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "MPMEntity":
        self.add_material(material)

        # create entity
        entity = MPMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )
        self._entities.append(entity)

        return entity

    def add_material(self, material):
        # Register material update methods if and only if the provided material is not already registered
        for material_i in self._materials:
            if material == material_i:
                material.idx = material_i.idx
                break
        else:
            material.idx = len(self._materials_idx)
            self._materials_idx.append(material.idx)
            self._materials_update_F_S_Jp.append(material.update_F_S_Jp)
            self._materials_update_stress.append(material.update_stress)
        self._materials.append(material)

    @qd.func
    def stencil_range(self):
        return qd.ndrange(3, 3, 3)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def compute_F_tmp(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles[f, i_p, i_b].F_tmp = (
                    qd.Matrix.identity(gs.qd_float, 3) + self.substep_dt * self.particles[f, i_p, i_b].C
                ) @ self.particles[f, i_p, i_b].F

    @qd.kernel
    def svd_grad(self, f: int):
        """Reverse the singular value decomposition of a frame, consuming the adjoint of its factors like autodiff."""
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles.grad[f, i_p, i_b].F_tmp += backward_svd(
                    self.particles.grad[f, i_p, i_b].U,
                    self.particles.grad[f, i_p, i_b].S,
                    self.particles.grad[f, i_p, i_b].V,
                    self.particles[f, i_p, i_b].U,
                    self.particles[f, i_p, i_b].S,
                    self.particles[f, i_p, i_b].V,
                )
                self.particles.grad[f, i_p, i_b].U = qd.Matrix.zero(gs.qd_float, 3, 3)
                self.particles.grad[f, i_p, i_b].S = qd.Matrix.zero(gs.qd_float, 3, 3)
                self.particles.grad[f, i_p, i_b].V = qd.Matrix.zero(gs.qd_float, 3, 3)

    # The decomposition is reversed in closed form, which autodiff cannot derive
    @register_custom_backward(svd_grad)
    @qd.kernel
    def svd(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles[f, i_p, i_b].U, self.particles[f, i_p, i_b].S, self.particles[f, i_p, i_b].V = qd.svd(
                    self.particles[f, i_p, i_b].F_tmp, gs.qd_float
                )

    @qd.kernel
    def compute_F_tmp_and_svd(self, f: qd.i32):
        # Fused F_tmp + SVD: keeps F_tmp in register/local-memory for the SVD instead of round-tripping through global
        # memory. Only used on the forward pass; the backward path continues to call compute_F_tmp / svd separately so
        # their autodiff remains composed unchanged.
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                F_tmp = (
                    qd.Matrix.identity(gs.qd_float, 3) + self.substep_dt * self.particles[f, i_p, i_b].C
                ) @ self.particles[f, i_p, i_b].F
                self.particles[f, i_p, i_b].F_tmp = F_tmp
                self.particles[f, i_p, i_b].U, self.particles[f, i_p, i_b].S, self.particles[f, i_p, i_b].V = qd.svd(
                    F_tmp, gs.qd_float
                )

    @qd.kernel
    def compute_F_tmp_only(self, f: qd.i32):
        # Fast path when no registered material needs U/V/S: compute F_tmp without SVD. p2g then derives J from
        # det(F_tmp) directly. Only used on the forward pass.
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles[f, i_p, i_b].F_tmp = (
                    qd.Matrix.identity(gs.qd_float, 3) + self.substep_dt * self.particles[f, i_p, i_b].C
                ) @ self.particles[f, i_p, i_b].F

    @qd.kernel
    def p2g(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                # A. update F (deformation gradient), S (Sigma from SVD(F), essentially represents volume) and Jp
                # (volume compression ratio) based on material type.
                # det(F_tmp) == det(U S V^T) == det(S) for qd.svd (U, V are proper rotations), so when no material
                # needs U/V/S we can read J directly from F_tmp and skip the SVD kernel. Predeclared outside the
                # qd.static branch because quadrants scopes qd.static branches locally.
                J = gs.qd_float(0.0)
                if qd.static(self.needs_svd):
                    J = self.particles[f, i_p, i_b].S.determinant()
                else:
                    J = self.particles[f, i_p, i_b].F_tmp.determinant()
                F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
                S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
                Jp_new = gs.qd_float(1.0)
                for material_idx in qd.static(self._materials_idx):
                    if self.particles_info[i_p].material_idx == material_idx:
                        F_new, S_new, Jp_new = self._materials_update_F_S_Jp[material_idx](
                            J=J,
                            F_tmp=self.particles[f, i_p, i_b].F_tmp,
                            U=self.particles[f, i_p, i_b].U,
                            S=self.particles[f, i_p, i_b].S,
                            V=self.particles[f, i_p, i_b].V,
                            Jp=self.particles[f, i_p, i_b].Jp,
                        )
                self.particles[f + 1, i_p, i_b].F = F_new
                self.particles[f + 1, i_p, i_b].Jp = Jp_new
                # The actuation holds over every substep of the step it is set at, and until it is set again
                self.particles[f + 1, i_p, i_b].actu = self.particles[f, i_p, i_b].actu

                # B. compute stress
                # NOTE:
                # 1. Here we pass in both F_tmp and the updated F_new because in the official Quadrants example, F_new is
                # used for stress computation. However, although this works for both elastic and elasto-plastic
                # materials, it is mathematically incorrect for liquid material with non-zero viscosity (mu). In the
                # latter case, stress computation needs to be based on the F_tmp (deformation gradient before resetting
                # to identity).
                # 2. Jp is only used by Snow material, and it uses Jp from the previous frame, not the updated one.
                stress = qd.Matrix.zero(gs.qd_float, 3, 3)
                for material_idx in qd.static(self._materials_idx):
                    if self.particles_info[i_p].material_idx == material_idx:
                        stress = self._materials_update_stress[material_idx](
                            U=self.particles[f, i_p, i_b].U,
                            S=S_new,
                            V=self.particles[f, i_p, i_b].V,
                            F_tmp=self.particles[f, i_p, i_b].F_tmp,
                            F_new=F_new,
                            J=J,
                            Jp=self.particles[f, i_p, i_b].Jp,
                            actu=self.particles[f, i_p, i_b].actu,
                            m_dir=self.particles_info[i_p].muscle_direction,
                        )
                stress = (-self.substep_dt * self._particle_volume * 4 * self._inv_dx * self._inv_dx) * stress
                affine = stress + self.particles_info[i_p].mass * self.particles[f, i_p, i_b].C

                # C. project onto grid
                base = qd.floor(self.particles[f, i_p, i_b].pos * self._inv_dx - 0.5).cast(gs.qd_int)
                fx = self.particles[f, i_p, i_b].pos * self._inv_dx - base.cast(gs.qd_float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                for offset in qd.static(qd.grouped(self.stencil_range())):
                    dpos = (offset.cast(gs.qd_float) - fx) * self._dx
                    weight = gs.qd_float(1.0)
                    for d in qd.static(range(3)):
                        weight *= w[offset[d]][d]

                    sep_geom_idx = -1
                    if qd.static(self._enable_CPIC and self.sim.rigid_solver.is_active):
                        # check if particle and cell center are at different side of any thin object
                        cell_pos = (base + offset) * self._dx

                        for i_g in range(self.sim.rigid_solver.n_geoms):
                            if geoms_info.needs_coup[i_g]:
                                sdf_normal_particle = self._coupler.mpm_rigid_normal[i_p, i_g, i_b]
                                sdf_normal_cell = sdf.sdf_func_normal_world(
                                    i_g,
                                    i_b,
                                    cell_pos,
                                    geoms_state,
                                    geoms_info,
                                    rigid_info,
                                    sdf_info,
                                    collider_static_config,
                                )

                                if sdf_normal_particle.dot(sdf_normal_cell) < 0:  # separated by geom i_g
                                    sep_geom_idx = i_g
                                    break
                        self._coupler.cpic_flag[i_p, offset[0], offset[1], offset[2], i_b] = sep_geom_idx
                    if sep_geom_idx == -1:
                        cell_ijk = base - self._grid_offset + offset
                        self.grid[f, cell_ijk, i_b].vel_in += weight * (
                            self.particles_info[i_p].mass * self.particles[f, i_p, i_b].vel + affine @ dpos
                        )
                        mass_contrib = weight * self.particles_info[i_p].mass
                        prev_mass = qd.atomic_add(self.grid[f, cell_ijk, i_b].mass, mass_contrib)
                        # Sparse-reset bookkeeping runs forward-only: backward mode composes p2g through autodiff where
                        # these atomics are meaningless. Per-env first-writer (prev_mass == 0) tries to claim the cell
                        # in the env-shared dirty flag via atomic_or; only the very first env to touch this cell across
                        # the whole batch then appends to the global list. List size is bounded by grid_total because
                        # dedup ensures uniqueness.
                        if qd.static(not self._sim.requires_grad):
                            if prev_mass == gs.qd_float(0.0) and mass_contrib > gs.qd_float(0.0):
                                was_dirty = qd.atomic_or(self.grid_dirty_flag[cell_ijk], gs.qd_int(1))
                                if was_dirty == gs.qd_int(0):
                                    slot_idx = qd.atomic_add(self.grid_dirty_count[0], 1)
                                    flat = (cell_ijk[0] * self._grid_res[1] + cell_ijk[1]) * self._grid_res[2] + (
                                        cell_ijk[2]
                                    )
                                    self.grid_dirty_list[slot_idx] = flat

                    if not self.particles_info[i_p].free:  # non-free particles behave as boundary conditions
                        self.grid[f, base - self._grid_offset + offset, i_b].vel_in = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def g2p(
        self,
        f: qd.i32,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_info: array_class.RigidInfo,
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                base = qd.floor(self.particles[f, i_p, i_b].pos * self._inv_dx - 0.5).cast(gs.qd_int)
                fx = self.particles[f, i_p, i_b].pos * self._inv_dx - base.cast(gs.qd_float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_vel = qd.Vector.zero(gs.qd_float, 3)
                new_C = qd.Matrix.zero(gs.qd_float, 3, 3)
                for offset in qd.static(qd.grouped(self.stencil_range())):
                    dpos = offset.cast(gs.qd_float) - fx
                    grid_vel = self.grid[f, base - self._grid_offset + offset, i_b].vel_out
                    weight = gs.qd_float(1.0)
                    for d in qd.static(range(3)):
                        weight *= w[offset[d]][d]

                    if qd.static(self._enable_CPIC and self.sim.rigid_solver.is_active):
                        sep_geom_idx = self._coupler.cpic_flag[i_p, offset[0], offset[1], offset[2], i_b]
                        if sep_geom_idx != -1:
                            grid_vel = self.sim.coupler._func_collide_in_rigid_geom(
                                self.particles[f, i_p, i_b].pos,
                                self.particles[f, i_p, i_b].vel,
                                self.particles_info[i_p].mass * weight / self._particle_volume_scale,
                                self._coupler.mpm_rigid_normal[i_p, sep_geom_idx, i_b],
                                1.0,
                                sep_geom_idx,
                                i_b,
                                geoms_info=geoms_info,
                                links_state=links_state,
                                rigid_info=rigid_info,
                            )

                    new_vel += weight * grid_vel
                    new_C += 4 * self._inv_dx * weight * grid_vel.outer_product(dpos)

                # compute actual new_pos with new_vel
                new_pos = self.particles[f, i_p, i_b].pos + self.substep_dt * new_vel

                # impose boundary for safety, in case simulation explodes and tries to access illegal cell address
                new_pos, new_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

                # advect to next frame
                self.particles[f + 1, i_p, i_b].vel = new_vel
                self.particles[f + 1, i_p, i_b].C = new_C
                self.particles[f + 1, i_p, i_b].pos = new_pos
            else:
                self.particles[f + 1, i_p, i_b].vel = self.particles[f, i_p, i_b].vel
                self.particles[f + 1, i_p, i_b].pos = self.particles[f, i_p, i_b].pos
                self.particles[f + 1, i_p, i_b].C = self.particles[f, i_p, i_b].C
                self.particles[f + 1, i_p, i_b].F = self.particles[f, i_p, i_b].F
                self.particles[f + 1, i_p, i_b].Jp = self.particles[f, i_p, i_b].Jp

            self.particles_ng[f + 1, i_p, i_b].active = self.particles_ng[f, i_p, i_b].active

    @no_grad
    @qd.kernel
    def _kernel_flag_non_finite_pos(self, f: int):
        """Flag INVALID_PARTICLE_NAN for every env with a particle whose position in the given frame is not finite."""
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            pos = self.particles[f, i_p, i_b].pos
            if not qd_isfinite(pos):
                self.errno[i_b] = self.errno[i_b] | array_class.ErrorCode.INVALID_PARTICLE_NAN

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        if self._sim.requires_grad:
            self.reset_grid_and_grad(f)
            self.compute_F_tmp(f)
            self.svd(f)
        else:
            # Forward-only path: the grid is left zeroed by reset_dirty_cells at the end of the previous substep (and
            # is zero-init on the first call), so no reset kernel is needed here.
            if self.needs_svd:
                self.compute_F_tmp_and_svd(f)
            else:
                # All registered materials ignore U/V/S; skip SVD entirely (e.g. scenes of only non-viscous Liquid
                # and/or neohooken Elastic).
                self.compute_F_tmp_only(f)
        self.p2g(
            f,
            self.sim.coupler.rigid_solver.dyn_state.geoms,
            self.sim.coupler.rigid_solver.dyn_info.geoms,
            self.sim.coupler.rigid_solver.dyn_state.links,
            self.sim.coupler.rigid_solver.rigid_info,
            self.sim.coupler.rigid_solver.collider.sdf.sdf_info,
            self.sim.coupler.rigid_solver.collider.collider_config,
        )

    def substep_post_coupling(self, f):
        self.g2p(
            f,
            self.sim.coupler.rigid_solver.dyn_info.geoms,
            self.sim.coupler.rigid_solver.dyn_state.links,
            self.sim.coupler.rigid_solver.rigid_info,
        )

        # The particle constraints apply after g2p
        if self._constraints_initialized:
            self.apply_particle_constraints(f, self.sim.coupler.rigid_solver.dyn_state.links)

        # Eager sparse reset: zero only the cells p2g touched this substep, across all envs, then clear the global
        # dirty count so the next substep starts fresh. The grid is no longer read after g2p / constraints, so it is
        # safe to wipe here. Forward-only; backward composes p2g/g2p through autodiff and uses reset_grid_and_grad.
        if not self._sim.requires_grad:
            self.reset_dirty_cells(f)
            if gs.use_zerocopy:
                grid_dirty_count = qd_to_torch(self.grid_dirty_count, copy=False)
                grid_dirty_count.zero_()
                # The next kernel reads the count (see qd_zero_grad in misc.py)
                if gs.backend == gs.metal:
                    torch.mps.synchronize()
            else:
                self.grid_dirty_count[0] = 0

        # The positions a step ends with are checked once, an error being raised at the beginning of the next step
        if (self._sim._cur_substep_global + 1) % self._substeps == 0:
            self._kernel_flag_non_finite_pos(f + 1)

    @qd.kernel
    def copy_frame(self, source: qd.i32, target: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles[target, i_p, i_b].pos = self.particles[source, i_p, i_b].pos
            self.particles[target, i_p, i_b].vel = self.particles[source, i_p, i_b].vel
            self.particles[target, i_p, i_b].F = self.particles[source, i_p, i_b].F
            self.particles[target, i_p, i_b].C = self.particles[source, i_p, i_b].C
            self.particles[target, i_p, i_b].Jp = self.particles[source, i_p, i_b].Jp
            self.particles[target, i_p, i_b].actu = self.particles[source, i_p, i_b].actu

            self.particles_ng[target, i_p, i_b].active = self.particles_ng[source, i_p, i_b].active

    # The grid of the substep starts from zero, its adjoint included
    @no_grad
    @qd.kernel
    def reset_grid_and_grad(self, f: qd.i32):
        # Zero out the grid at frame f for *all* grid cells and *all* batch indices
        for i, j, k, i_b in qd.ndrange(*self._grid_res, self._B):
            self.grid[f, i, j, k, i_b].vel_in = qd.Vector.zero(gs.qd_float, 3)
            self.grid[f, i, j, k, i_b].mass = gs.qd_float(0.0)
            self.grid[f, i, j, k, i_b].vel_out = qd.Vector.zero(gs.qd_float, 3)

            self.grid.grad[f, i, j, k, i_b].vel_in = qd.Vector.zero(gs.qd_float, 3)
            self.grid.grad[f, i, j, k, i_b].mass = gs.qd_float(0.0)
            self.grid.grad[f, i, j, k, i_b].vel_out = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def reset_dirty_cells(self, f: qd.i32):
        # Zero the cells p2g touched this substep, across all envs, plus their dirty flags. The dirty list is shared
        # across envs (deduplicated via grid_dirty_flag), so threads with slot >= grid_dirty_count idle. Only one thread
        # per slot (i_b == 0) writes the flag back to zero so it can be reused next substep.
        for slot, i_b in qd.ndrange(self._grid_total, self._B):
            if slot < self.grid_dirty_count[0]:
                flat = self.grid_dirty_list[slot]
                k = flat % self._grid_res[2]
                rem = flat // self._grid_res[2]
                j = rem % self._grid_res[1]
                i = rem // self._grid_res[1]
                self.grid[f, i, j, k, i_b].mass = gs.qd_float(0.0)
                self.grid[f, i, j, k, i_b].vel_in = qd.Vector.zero(gs.qd_float, 3)
                self.grid[f, i, j, k, i_b].vel_out = qd.Vector.zero(gs.qd_float, 3)
                if i_b == 0:
                    self.grid_dirty_flag[i, j, k] = gs.qd_int(0)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_differentiable(self) -> bool:
        return not self._enable_CPIC

    @property
    def data(self):
        """Yield the state of the particles, the frame 0 of which a boundary checkpoint holds (see 'Solver.data')."""
        for name, field in (
            ("particles.pos", self.particles.pos),
            ("particles.vel", self.particles.vel),
            ("particles.C", self.particles.C),
            ("particles.F", self.particles.F),
            ("particles.Jp", self.particles.Jp),
            ("particles.actu", self.particles.actu),
            ("particles_ng.active", self.particles_ng.active),
        ):
            yield array_class.ArrayItem(name, array_class.DataKind.STATE, field)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @mutates()
    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.actu, state.active)
            self.clear_errno(envs_idx)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, n_particles, 3]
        vel: qd.types.ndarray(),  # shape [B, n_particles, 3]
        C: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
        F: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
        Jp: qd.types.ndarray(),  # shape [B, n_particles]
        actu: qd.types.ndarray(),  # shape [B, n_particles]
        active: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            # Write pos, vel
            for j in qd.static(range(3)):
                self.particles[f, i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[f, i_p, i_b].vel[j] = vel[i_b, i_p, j]
                # Write C, F
                for k in qd.static(range(3)):
                    self.particles[f, i_p, i_b].C[j, k] = C[i_b, i_p, j, k]
                    self.particles[f, i_p, i_b].F[j, k] = F[i_b, i_p, j, k]
            # Write Jp, actu, active
            self.particles[f, i_p, i_b].Jp = Jp[i_b, i_p]
            self.particles[f, i_p, i_b].actu = actu[i_b, i_p]
            self.particles_ng[f, i_p, i_b].active = active[i_b, i_p]

    def get_state(self, f):
        if not self.is_active:
            return None

        pos, vel, C, F, Jp = self._read_data(
            (self.particles.pos, self.particles.vel, self.particles.C, self.particles.F, self.particles.Jp),
            None,
            None,
            copy=True,
        )
        # The actuation and the activity are inputs of the step, which carry no gradient back
        actu, active = [
            qd_to_torch(field, row_mask=0, keepdim=False).movedim(1, 0).clone(memory_format=torch.contiguous_format)
            for field in (self.particles.actu, self.particles_ng.active)
        ]
        return MPMSolverState(self._scene, pos, vel, C, F, Jp, actu, active)

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[f, i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[f, i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng[f, i_p, i_b].active

        for i_v, i_b in qd.ndrange(self._n_vverts, self._B):
            vvert_pos = qd.Vector.zero(gs.qd_float, 3)
            for j in range(self._n_vvert_supports):
                vvert_pos += (
                    self.particles[f, self.vverts_info.support_idxs[i_v][j], i_b].pos
                    * self.vverts_info.support_weights[i_v][j]
                )
            self.vverts_render[i_v, i_b].pos = vvert_pos
            self.vverts_render[i_v, i_b].active = self.particles_render[
                self.vverts_info.support_idxs[i_v][0], i_b
            ].active

    @qd.kernel
    def add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        material_idx: qd.i32,
        mat_default_Jp: qd.f32,
        mat_rho: qd.f32,
        pos: qd.types.ndarray(),  # shape [n_particles, 3]
    ):
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start

            self.particles_info[i_p].material_idx = material_idx
            self.particles_info[i_p].default_Jp = mat_default_Jp
            self.particles_info[i_p].mass = self._particle_volume * mat_rho
            self.particles_info[i_p].free = True
            self.particles_info[i_p].muscle_group = 0
            self.particles_info[i_p].muscle_direction = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start

            self.particles_ng[f, i_p, i_b].active = qd.cast(active, gs.qd_bool)
            for i in qd.static(range(3)):
                self.particles[f, i_p, i_b].pos[i] = pos[i_p_, i]

            self.particles[f, i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            self.particles[f, i_p, i_b].F = qd.Matrix.identity(gs.qd_float, 3)
            self.particles[f, i_p, i_b].C = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles[f, i_p, i_b].Jp = mat_default_Jp
            self.particles[f, i_p, i_b].actu = gs.qd_float(0.0)

    @gs.assert_built
    @mutates()
    def set_particles_actu(self, actus, particles_idx=None, envs_idx=None, *, n_groups=1):
        """
        Set the actuation of some particles, each particle taking the actuation of its muscle group.

        Parameters
        ----------
        actus : array_like, shape ([M,] [N,] [n_groups,])
            Actuation of each muscle group, for each particle.
        particles_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the particles (see 'set_particles_pos'). If None, all particles will be considered.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        n_groups : int, optional
            The number of muscle groups. Defaults to 1.
        """
        actus, particles_idx, envs_idx = self._sanitize_io_variables(
            actus, particles_idx, self.n_particles, "particles_idx", envs_idx, (n_groups,), batched_idx=True
        )
        self._kernel_set_particles_actu(particles_idx, envs_idx, actus, n_groups)

    def get_particles_actu(self, particles_idx=None, envs_idx=None):
        """Get the actuation of some particles (see 'set_particles_actu')."""
        actus, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True
        )
        self._kernel_get_particles_actu(particles_idx, envs_idx, actus)
        return actus[0] if self._scene.n_envs == 0 else actus

    @mutates(StateChange.MODEL)
    def set_particles_muscle_group(self, muscle_group, particles_idx=None):
        """Set the muscle group of some particles, shared by every environment."""
        muscle_group, particles_idx, _ = self._sanitize_io_variables(
            muscle_group, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_int, batched=False
        )
        self._kernel_set_particles_muscle_group(particles_idx, muscle_group)

    def get_particles_muscle_group(self, particles_idx=None):
        """Get the muscle group of some particles (see 'set_particles_muscle_group')."""
        muscle_group, particles_idx, _ = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_int, batched=False
        )
        self._kernel_get_particles_muscle_group(particles_idx, muscle_group)
        return muscle_group

    @mutates(StateChange.MODEL)
    def set_particles_muscle_direction(self, muscle_direction, particles_idx=None):
        """Set the unit direction of the muscle fibers of some particles, shared by every environment."""
        muscle_direction, particles_idx, _ = self._sanitize_io_variables(
            muscle_direction, particles_idx, self.n_particles, "particles_idx", element_shape=(3,), batched=False
        )
        self._kernel_set_particles_muscle_direction(particles_idx, muscle_direction)

    @mutates(StateChange.MODEL)
    def set_particles_free(self, free, particles_idx=None):
        """Set whether some particles move freely or stay fixed, shared by every environment."""
        free, particles_idx, _ = self._sanitize_io_variables(
            free, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_bool, batched=False
        )
        self._kernel_set_particles_free(particles_idx, free)

    def get_particles_free(self, particles_idx=None):
        """Get whether some particles move freely (see 'set_particles_free')."""
        free, particles_idx, _ = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_bool, batched=False
        )
        self._kernel_get_particles_free(particles_idx, free)
        return free

    @qd.kernel
    def _kernel_set_particles_pos(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # The earlier occurrences of a repeated particle are -1 (see 'drop_repeated_index'). Between two steps, the
            # state sits in frame 0 (see 'Solver.has_frames').
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles[0, i_p, i_b].pos[i] = poss[i_b_, i_p_, i]

                # Reset these attributes whenever overwriting particle positions manually
                self.particles[0, i_p, i_b].vel.fill(0.0)
                self.particles[0, i_p, i_b].F = qd.Matrix.identity(gs.qd_float, 3)
                self.particles[0, i_p, i_b].C.fill(0.0)
                self.particles[0, i_p, i_b].Jp = self.particles_info[i_p].default_Jp

    @qd.kernel
    def _kernel_get_particles_pos(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[0, i_p, i_b].pos[i]

    @qd.kernel
    def _kernel_set_particles_vel(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles[0, i_p, i_b].vel[i] = vels[i_b_, i_p_, i]

    @qd.kernel
    def _kernel_get_particles_vel(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[0, i_p, i_b].vel[i]

    @qd.kernel
    def _kernel_set_particles_active(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_ng[0, i_p, i_b].active = qd.cast(actives[i_b_, i_p_], gs.qd_bool)

    @qd.kernel
    def _kernel_get_particles_active(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[0, i_p, i_b].active

    @qd.kernel
    def _kernel_set_particles_actu(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actus: qd.types.ndarray(),  # shape [B, n_particles, n_groups]
        n_groups: int,
    ):
        for i_p_, i_g, i_b_ in qd.ndrange(particles_idx.shape[1], n_groups, envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                if self.particles_info[i_p].muscle_group == i_g:
                    self.particles[0, i_p, i_b].actu = actus[i_b_, i_p_, i_g]

    @qd.kernel
    def _kernel_get_particles_actu(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actus: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            actus[i_b_, i_p_] = self.particles[0, i_p, i_b].actu

    @qd.kernel
    def _kernel_set_particles_muscle_group(self, particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_info[i_p].muscle_group = muscle_group[i_p_]

    @qd.kernel
    def _kernel_get_particles_muscle_group(self, particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            muscle_group[i_p_] = self.particles_info[i_p].muscle_group

    @qd.kernel
    def _kernel_set_particles_muscle_direction(
        self, particles_idx: qd.types.ndarray(), muscle_direction: qd.types.ndarray()
    ):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles_info[i_p].muscle_direction[i] = muscle_direction[i_p_, i]

    @qd.kernel
    def _kernel_set_particles_free(self, particles_idx: qd.types.ndarray(), free: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_info[i_p].free = free[i_p_]

    @qd.kernel
    def _kernel_get_particles_free(self, particles_idx: qd.types.ndarray(), free: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            free[i_p_] = self.particles_info[i_p].free

    @qd.kernel
    def _kernel_get_particles_mass(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), masses: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            masses[i_b_, i_p_] = self.particles_info[i_p].mass / self._particle_volume_scale

    # ------------------------------------------------------------------------------------
    # -------------------------------- particle constraints ------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_particle_constraints(self, particles_mask, link_idx, stiffness, *, particles_start=0):
        """
        Attach some particles to a rigid link with critically damped springs.

        Each particle is pulled toward the position it holds now, expressed in the frame of the link.

        Parameters
        ----------
        particles_mask : array_like, shape ([n_envs,] n_particles)
            Boolean mask over consecutive particles of the solver, selecting those to attach.
        link_idx : int
            Global index of the rigid link.
        stiffness : float
            Stiffness of the springs.
        particles_start : int, optional
            Global index of the particle the mask starts at. Defaults to 0.
        """
        if not isinstance(link_idx, int):
            gs.raise_exception("link_idx must be an integer.")
        if not self._constraints_initialized:
            self.init_constraints()
        particles_mask = broadcast_tensor(
            particles_mask, gs.tc_bool, (self._B, -1), ("envs_idx", "particles_idx")
        ).contiguous()
        if particles_start < 0 or particles_start + particles_mask.shape[1] > self.n_particles:
            gs.raise_exception("The particle mask overruns the particles of the solver.")
        rigid_solver = self._sim.rigid_solver
        # The attachment is a configuration of the constraint, which carries no gradient
        with torch.no_grad():
            link_pos = rigid_solver.get_links_pos(links_idx=link_idx)[..., 0, :]
            link_quat = rigid_solver.get_links_quat(links_idx=link_idx)[..., 0, :]
        link_pos = broadcast_tensor(link_pos, gs.tc_float, (self._B, 3)).contiguous()
        link_quat = broadcast_tensor(link_quat, gs.tc_float, (self._B, 4)).contiguous()
        self._kernel_set_particle_constraints(particles_start, link_idx, stiffness, particles_mask, link_pos, link_quat)

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def remove_particle_constraints(self, particles_mask=None, *, particles_start=0):
        """
        Detach some particles from the rigid link they are attached to (see 'set_particle_constraints').

        Parameters
        ----------
        particles_mask : None | array_like, shape ([n_envs,] n_particles), optional
            Boolean mask over consecutive particles of the solver, selecting those to detach. If None, all particles
            from `particles_start` on will be detached. Defaults to None.
        particles_start : int, optional
            Global index of the particle the mask starts at. Defaults to 0.
        """
        if not self._constraints_initialized:
            return
        n_particles = -1
        if particles_mask is None:
            particles_mask = True
            n_particles = self.n_particles - particles_start
        particles_mask = broadcast_tensor(
            particles_mask, gs.tc_bool, (self._B, n_particles), ("envs_idx", "particles_idx")
        ).contiguous()
        if particles_start < 0 or particles_start + particles_mask.shape[1] > self.n_particles:
            gs.raise_exception("The particle mask overruns the particles of the solver.")
        self._kernel_remove_particle_constraints(particles_start, particles_mask)

    @qd.kernel
    def _kernel_set_particle_constraints(
        self,
        particles_start: int,
        link_idx: int,
        stiffness: float,
        particles_mask: qd.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
        link_pos: qd.types.ndarray(),  # shape [n_envs, 3]
        link_quat: qd.types.ndarray(),  # shape [n_envs, 4]
    ):
        for i_p_, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_]:
                i_p = i_p_ + particles_start
                # Get current particle position. Between two steps, the state sits in frame 0 (see 'Solver.has_frames')
                pos = self.particles[0, i_p, i_b].pos

                # Get link transform
                l_pos = qd.Vector([link_pos[i_b, 0], link_pos[i_b, 1], link_pos[i_b, 2]], dt=gs.qd_float)
                l_quat = qd.Vector(
                    [link_quat[i_b, 0], link_quat[i_b, 1], link_quat[i_b, 2], link_quat[i_b, 3]], dt=gs.qd_float
                )

                # Compute offset in link's local frame
                local_pos = gu.qd_inv_transform_by_trans_quat(pos, l_pos, l_quat)

                # Store constraint info
                self.particle_constraints[i_p, i_b].is_constrained = True
                self.particle_constraints[i_p, i_b].target_pos = pos  # initial target is current position
                self.particle_constraints[i_p, i_b].stiffness = stiffness
                self.particle_constraints[i_p, i_b].link_idx = link_idx
                self.particle_constraints[i_p, i_b].link_local_pos = local_pos

    @qd.kernel
    def _kernel_remove_particle_constraints(
        self,
        particles_start: int,
        particles_mask: qd.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
    ):
        for i_p_, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_]:
                i_p = i_p_ + particles_start
                self.particle_constraints[i_p, i_b].is_constrained = False
                self.particle_constraints[i_p, i_b].link_idx = -1

    @qd.kernel
    def apply_particle_constraints(self, f: qd.i32, links_state: array_class.LinksState):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particle_constraints[i_p, i_b].is_constrained:
                # The target of a particle attached to a link follows the link. It stays a local, since autodiff cannot
                # reverse a field read back after it is overwritten in the same kernel.
                target_pos = self.particle_constraints[i_p, i_b].target_pos
                i_l = self.particle_constraints[i_p, i_b].link_idx
                if i_l >= 0:
                    link_pos = links_state.pos[i_l, i_b]
                    link_quat = links_state.quat[i_l, i_b]
                    local_pos = self.particle_constraints[i_p, i_b].link_local_pos
                    target_pos = gu.qd_transform_by_trans_quat(local_pos, link_pos, link_quat)

                # Apply spring force to velocity
                stiffness = self.particle_constraints[i_p, i_b].stiffness
                mass = self.particles_info[i_p].mass / self._particle_volume_scale

                pos = self.particles[f + 1, i_p, i_b].pos
                vel = self.particles[f + 1, i_p, i_b].vel

                pos_error = pos - target_pos
                spring_force = -stiffness * pos_error
                damping_force = -2.0 * qd.math.sqrt(stiffness * mass) * vel

                # An accumulation, which autodiff reverses, where an overwrite of the velocity it reads is not
                self.particles[f + 1, i_p, i_b].vel += self.substep_dt * (spring_force + damping_force) / mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum(entity.n_particles for entity in self._entities)

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
    def grid_density(self):
        return self._grid_density

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def leaf_block_size(self):
        raise DeprecationError("This property has been removed.")

    @property
    def use_sparse_grid(self):
        return DeprecationError("This property has been removed.")

    @property
    def dx(self):
        return self._dx

    @property
    def inv_dx(self):
        return self._inv_dx

    @property
    def particle_volume_real(self):
        return self._particle_volume_real

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_volume_scale(self):
        return self._particle_volume_scale

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def lower_bound_cell(self):
        return self._lower_bound_cell

    @property
    def upper_bound_cell(self):
        return self._upper_bound_cell

    @property
    def grid_res(self):
        return self._grid_res

    @property
    def grid_offset(self):
        return self._grid_offset

    @property
    def enable_CPIC(self):
        return self._enable_CPIC

    @property
    def enable_particle_constraints(self):
        return self._enable_particle_constraints


@qd.func
def signmax(a, eps):
    sign = qd.select(a >= 0, 1.0, -1.0)
    return sign * qd.max(qd.abs(a), eps)


@qd.func
def backward_svd(grad_U, grad_S, grad_V, U, S, V):
    # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
    vt = V.transpose()
    ut = U.transpose()
    S_term = U @ grad_S @ vt

    s = qd.Vector.zero(gs.qd_float, 3)
    s = qd.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
    F = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        if i == j:
            F[i, j] = 0.0
        else:
            F[i, j] = 1.0 / signmax(s[j] - s[i], 1e-6)
    u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
    v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
    return u_term + v_term + S_term
