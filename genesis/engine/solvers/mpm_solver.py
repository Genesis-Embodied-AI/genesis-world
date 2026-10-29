from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.couplers import mpm_exchange_func, mpm_rigid_func
from genesis.engine.couplers.mpm_rigid_func import func_build_rigid_rows
from genesis.engine.entities import MPMEntity
from genesis.engine.materials import MPM
from genesis.engine.numerics.bcoo_matrix import func_sort_reduce_bcoo, func_sort_reduce_bcoo_graph, sort_reduce_bcoo
from genesis.engine.numerics.dynamic_exclusive_sum import kernel_dynamic_exclusive_sum
from genesis.engine.numerics.dynamic_radix_sort import dynamic_radix_sort
from genesis.engine.numerics.linear_pcg import (
    func_init_linear_pcg_direction,
    func_init_linear_pcg_residual,
    func_linear_pcg_direction_update,
    func_linear_pcg_spmv,
    func_linear_pcg_update_xr,
)
from genesis.engine.numerics.newton_control import (
    func_accept_or_backtrack,
    func_check_convergence,
    func_init_line_search,
    func_reset_newton_control,
    func_reset_newton_measures,
    func_update_convergence,
)
from genesis.engine.states.solvers import MPMSolverState
from genesis.options.solvers import LegacyCouplerOptions, MPMOptions
from genesis.utils.misc import DeprecationError, qd_to_torch

from . import mpm_func
from .base_solver import GravityMixin, Solver, TimeBasedMixin
from .mpm_data import (
    BLOCK_CELLS,
    MpmForceFields,
    MpmGrid,
    MpmLinearSystem,
    MpmMaterialTable,
    MpmNewton,
    MpmParams,
    MpmParticles,
    MpmParticlesInfo,
    MpmParticlesNG,
    MpmRigidCoupling,
    MpmStaticConfig,
    get_mpm_material_table,
    get_mpm_newton_control,
    get_mpm_particles,
    get_mpm_particles_info,
    get_mpm_particles_ng,
    initial_capacities,
    triplet_capacity,
    CAPACITY_GROW_FACTOR,
    MIN_BLOCKS,
)

if TYPE_CHECKING:
    from genesis.engine.entities import MPMEntity
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# Largest grid node velocity update, in m/s, under which the Newton solve of a substep is converged
NEWTON_VELOCITY_TOL = 5e-2

# Run the substep as the single checkpointed graph `kernel_substep`, or as separate launches driven from the host with
# the PCG iterations and the radix sorts as small graphs of their own.
# FIXME: The single graph takes several minutes to compile cold, against a fraction of that for the separate launches,
# whose kernels each compile once and are shared by every call site; the host-driven substep is the default meanwhile.
USE_SUBSTEP_GRAPH = False

# Run the Newton iterations of the host-driven substep as one graph (`kernel_newton_loop`), or drive them from the host
# with only the PCG iterations as a graph. The graph runs faster, the host loop compiles faster.
# FIXME: The Newton graph compiles cold in about 47 s against about 15 s for its stages as separate kernels (the
# compile time of a graph grows faster than its size), and a change of any stage in it recompiles it whole.
USE_NEWTON_GRAPH = False


@qd.data_oriented
class MPMSolver(GravityMixin, TimeBasedMixin, Solver):
    """Implicit material point method.

    Every substep solves for the grid node displacement that minimizes the incremental potential of the grid inertia
    and the particle elastic energy, on a block-sparse grid rebuilt from the particles, by a Newton method whose linear
    systems a preconditioned conjugate gradient solves. The substep runs as separate launches driven from the host, or
    as one graph (see `USE_SUBSTEP_GRAPH`); this class owns its data and composes the stages of `mpm_func`.
    """

    material_cls = MPM.Base
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
        # Particle masses are stored scaled by it; the implicit step divides it back out to work in kilograms.
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

        # materials
        self._materials = list()
        self._materials_idx = list()

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        # safety padding
        self.boundary_padding = 3 * self._dx
        self.boundary = CubeBoundary(
            lower=self._lower_bound + self.boundary_padding, upper=self._upper_bound - self.boundary_padding
        )

    def init_particle_fields(self):
        # Particle state of the substep window, its gradient-free part and the static info, as dynamic tensors in the
        # default array mode, so that a scene with another particle count reuses the compiled kernels
        n_frames = self._sim.substeps_local + 1
        self.particles = get_mpm_particles(n_frames, self._n_particles, self._B)
        self.particles_ng = get_mpm_particles_ng(n_frames, self._n_particles, self._B)
        self.particles_info = get_mpm_particles_info(self._n_particles)

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(pos=gs.qd_vec3, vel=gs.qd_vec3, active=gs.qd_bool)
        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

    def init_implicit_data(self):
        """Allocate the data of the implicit step at the smallest capacity; the first substep sizes it to the
        particles, and later substeps grow it when the grid or the matrix overflows."""
        n_sort = self._B * self._n_particles
        max_blocks = MIN_BLOCKS
        max_triplets = triplet_capacity(max_blocks, MIN_BLOCKS)
        self._mpm_sized = False
        materials = {material.idx: material for material in self._materials}
        n_materials = len(materials)
        table = np.zeros((mpm_func.N_MATERIAL_PARAMS, n_materials), dtype=gs.np_float)
        family = np.zeros(n_materials, dtype=gs.np_int)
        for idx, material in materials.items():
            table[mpm_func.PARAM_MU, idx] = material.mu
            table[mpm_func.PARAM_LAM, idx] = material.lam
            family[idx] = mpm_func.FAMILY_FIXED_COROTATED

        rigid = self._sim.rigid_solver
        self._mpm_config = MpmStaticConfig(
            has_rigid=self._has_rigid,
            rigid_batch_links_info=rigid._options.batch_links_info,
            has_rigid_rows=self._has_rigid and not rigid._disable_constraint,
            has_terrain=self._has_rigid and rigid.collider.collider_config.has_terrain,
        )
        # Rigid geoms give contact rows whenever Rigid is coupled, but its dofs join the Newton solve only if it has any:
        # rows on static geoms (a plane, a terrain) act on the particle displacement alone, as wall rows do
        self._rigid_coupled = self._has_rigid and rigid.n_dofs > 0
        self._mpm_coupling = MpmRigidCoupling(rigid.n_dofs_ if self._rigid_coupled else 1, rigid.n_links, self._B)
        # Rigid stops its substep after assembling its constraint rows; the MPM substep solves them with its own
        # unknowns and finishes Rigid's substep
        rigid._defer_constraint_solve = self._rigid_coupled
        self._mpm_params = MpmParams(
            n_envs=self._B,
            n_particles=self._n_particles,
            substep_dt=self.substep_dt,
            dx=self._dx,
            particle_volume=self._particle_volume_real,
            mass_scale=self._particle_volume_scale,
            n_materials=n_materials,
            wall_sol_params=self._wall_sol_params(),
            wall_lower=self.boundary.lower,
            wall_upper=self.boundary.upper,
        )
        self._mpm_materials = get_mpm_material_table(table.reshape(-1), family)
        self._mpm_grid = MpmGrid(n_sort, max_blocks, max_blocks)
        self._mpm_newton = MpmNewton(n_sort, max_blocks * BLOCK_CELLS)
        self._mpm_linear = MpmLinearSystem(max_blocks * BLOCK_CELLS, max_triplets)
        self._mpm_control = get_mpm_newton_control()
        self._mpm_control.disp_tol.from_numpy(np.array(NEWTON_VELOCITY_TOL * self.substep_dt, dtype=gs.np_float))
        self._mpm_force_fields = MpmForceFields(self._ffs)

    def _wall_sol_params(self):
        """Rigid's default solref / solimp parameters, with the Rigid constraint timeconst floored at twice the
        substep as Rigid floors its contacts."""
        params = gu.default_solver_params()
        timeconst = self._sim.rigid_solver._options.constraint_timeconst
        params[0] = max(timeconst if timeconst is not None else 0.0, 2.0 * self.substep_dt)
        return tuple(float(p) for p in params)

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

    def init_ckpt(self):
        self._ckpt = dict()

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
        # The implicit solver raises for differentiable simulation, so its particle data carries no gradient yet
        self._queried_states.clear()

        for entity in self._entities:
            entity.reset_grad()

    def _check_supported(self):
        """Raise for the configurations the implicit solver does not handle yet."""
        if self._sim.requires_grad:
            gs.raise_exception("Differentiable simulation is not supported by the implicit MPM solver yet.")
        if self._enable_CPIC:
            gs.raise_exception("CPIC is not supported by the implicit MPM solver.")
        for material in self._materials:
            if type(material) is not MPM.Elastic or material.model != "corotation":
                gs.raise_exception(
                    f"The implicit MPM solver supports 'MPM.Elastic(model=\"corotation\")' only for now, got "
                    f"{type(material).__name__}."
                )
        # Only the legacy coupler couples MPM with other solvers, each pair behind its own option
        coupler_options = self._sim.coupler.options
        is_legacy = isinstance(coupler_options, LegacyCouplerOptions)
        self._has_rigid = self._sim.rigid_solver.is_active and is_legacy and coupler_options.rigid_mpm
        if self._has_rigid:
            rigid_options = self._sim.rigid_solver._options
            if rigid_options.friction_cone == gs.friction_cone.elliptic:
                gs.raise_exception(
                    "Coupling the implicit MPM solver with an elliptic Rigid friction cone is not supported."
                )
            # The joint solve uses Rigid's Newton Hessian factor as its preconditioner and integrates every Rigid dof
            if rigid_options.constraint_solver != gs.constraint_solver.Newton:
                gs.raise_exception("Coupling the implicit MPM solver with Rigid requires Rigid's Newton solver.")
            if self._sim.rigid_solver._use_hibernation or rigid_options.noslip_iterations > 0:
                gs.raise_exception(
                    "Coupling the implicit MPM solver with Rigid does not support hibernation or noslip iterations."
                )
        if self._sim.tool_solver.is_active and is_legacy:
            gs.raise_exception("Coupling the implicit MPM solver with ToolSolver is not supported yet.")
        # Momentum exchanges with the particle and FEM solvers, at the free node velocity before the Newton solve
        self._sph_coupled = self._sim.sph_solver.is_active and is_legacy and coupler_options.mpm_sph
        self._pbd_coupled = self._sim.pbd_solver.is_active and is_legacy and coupler_options.mpm_pbd
        self._fem_coupled = self._sim.fem_solver.is_active and is_legacy and coupler_options.fem_mpm
        if self._fem_coupled and not self._sim.fem_solver._use_implicit_solver:
            gs.raise_exception(
                "Coupling the implicit MPM solver with FEM requires FEM's implicit solver "
                "(FEMOptions(use_implicit_solver=True)), or LegacyCouplerOptions(fem_mpm=False)."
            )

    def build(self):
        super().build()

        # particles and entities
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        self._coupler = self.sim._coupler

        if self.is_active:
            self._check_supported()

            self.init_particle_fields()
            self.init_vvert_fields()
            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

            self.init_implicit_data()

        self._build_gravity()

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
        # Register the material if and only if it is not already registered
        for material_i in self._materials:
            if material == material_i:
                material.idx = material_i.idx
                break
        else:
            material.idx = len(self._materials_idx)
            self._materials_idx.append(material.idx)
        self._materials.append(material)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _is_state_valid(self, f: qd.i32) -> qd.i32:
        is_success = True
        for i_p, i_b, i_3 in qd.ndrange(self._n_particles, self._B, 3):
            if qd.math.isnan(self.particles.pos[f, i_p, i_b][i_3]):
                is_success = False
        return is_success

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        pass

    def substep_pre_coupling_grad(self, f):
        gs.raise_exception("Differentiable simulation is not supported by the implicit MPM solver yet.")

    def _resize_blocks(self, max_blocks):
        """Reallocate the grid, the Newton state and the matrix rows for max_blocks node blocks. The grid carries no
        state across substeps and the growth happens before any consumer reads it, so nothing is copied."""
        n_sort = self._B * self._n_particles
        self._mpm_grid.allocate(n_sort, max_blocks, max_blocks)
        self._mpm_newton.allocate(n_sort, max_blocks * BLOCK_CELLS)
        self._mpm_linear.allocate_rows(max_blocks * BLOCK_CELLS)

    def _size_to_particles(self, f):
        positions = self.particles.pos.to_numpy()[f].transpose(1, 0, 2)
        max_blocks, max_triplets = initial_capacities(positions, self._inv_dx)
        self._resize_blocks(max_blocks)
        self._mpm_linear.matrix.grow(max_triplets)
        self._mpm_sized = True

    def _grow_grid(self):
        grid = self._mpm_grid
        demand = max(int(grid.grid_demand.to_numpy()), int(grid.n_particle_blocks.to_numpy()))
        self._resize_blocks(max(int(np.ceil(demand * CAPACITY_GROW_FACTOR)), grid.max_blocks + 1))
        grid.grid_overflow.from_numpy(np.array(0, dtype=np.int32))

    def _grow_triplets(self):
        matrix = self._mpm_linear.matrix
        demand = int(matrix.triplet_demand.to_numpy())
        matrix.grow(max(int(np.ceil(demand * CAPACITY_GROW_FACTOR)), int(matrix.max_triplets.to_numpy()) + 1))
        matrix.triplet_overflow.from_numpy(np.array(0, dtype=np.int32))

    def substep_post_coupling(self, f):
        if not self._mpm_sized:
            self._size_to_particles(f)
        if USE_SUBSTEP_GRAPH:
            self._substep_graph(f)
        else:
            self._substep_launches(f)

        # Apply particle constraints after g2p
        if self._constraints_initialized:
            self.apply_particle_constraints(f, self.sim.coupler.rigid_solver.dyn_state.links)

        # FIXME: Use existing errno mechanism for this.
        # Rate-limit the NaN check. _is_state_valid triggers a GPU->CPU sync on its return value, so calling it every
        # substep forces a sync every substep. Matching the cadence used by the rigid solver keeps the safety net while
        # letting the GPU stay ahead of the host queue.
        if self._sim._cur_substep_global % gs.engine.simulator.RATE_CHECK_ERRNO == 0:
            if not self._is_state_valid(f):
                gs.raise_exception(
                    "NaN detected in MPM states. Try reducing the time step size or adjusting simulation parameters."
                )
            if int(self._mpm_control.failed.to_numpy()):
                gs.logger.warning("The implicit MPM Newton solve stopped before converging.")

    def _assemble(self, f):
        """Assemble the Newton system, growing the triplet storage until it holds the assembly."""
        args = (
            self.particles,
            self.particles_info,
            self._mpm_materials,
            self._mpm_grid,
            self._mpm_newton,
            self._mpm_linear,
            self._mpm_control,
            self._mpm_params,
            f,
        )
        kernel_assemble(*args)
        while int(self._mpm_linear.matrix.triplet_overflow.to_numpy()):
            self._grow_triplets()
            kernel_assemble(*args)
        sort_reduce_bcoo(self._mpm_linear.matrix)

    def _rigid_args(self):
        rigid = self._sim.rigid_solver
        return (
            rigid.dyn_state,
            rigid.constraint_solver.constraint_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.rigid_config,
            self._mpm_config,
            self._mpm_params,
        )

    def _rigid_point_disp(self, f):
        kernel_rigid_point_disp(
            self.particles, self._mpm_grid, self._mpm_newton, self._mpm_coupling, *self._rigid_args(), f
        )

    def _rigid_energy(self, is_trial):
        kernel_rigid_energy(self._mpm_coupling, self._mpm_control, *self._rigid_args(), is_trial)

    def _newton_loop_launches(self, f):
        """The Newton iterations driven from the host, the PCG iterations of each as one graph."""
        grid, newton, linear, control = self._mpm_grid, self._mpm_newton, self._mpm_linear, self._mpm_control
        coupling, has_rigid = self._mpm_coupling, self._rigid_coupled
        params = self._mpm_params
        energy_args = (self.particles, self.particles_info, self._mpm_materials, grid, newton, control, params, f)
        while True:
            self._assemble(f)
            if has_rigid:
                rigid_args = self._rigid_args()
                kernel_rigid_newton_prepare(self.particles, grid, newton, coupling, *rigid_args, f)
                kernel_coupled_pcg_start(grid, linear, coupling, *rigid_args)
                kernel_coupled_pcg_loop(self.particles, grid, newton, linear, coupling, *rigid_args, f)
                kernel_rigid_start_line_search(coupling, control)
            else:
                kernel_pcg_start(grid, linear)
                kernel_pcg_loop(grid, linear)
            kernel_start_line_search(self.particles, grid, newton, linear, control, params, f)
            while True:
                kernel_step_forward(grid, newton, control)
                if has_rigid:
                    kernel_rigid_step_forward(coupling, control)
                    self._rigid_point_disp(f)
                kernel_energy(*energy_args, 1)
                if has_rigid:
                    self._rigid_energy(1)
                kernel_accept_or_backtrack(control)
                if not int(control.ls_cond.to_numpy()):
                    break
            kernel_update_convergence(control)
            if not int(control.newton_cond.to_numpy()):
                break

    def _exchange_momentum(self, f):
        """The momentum exchanges with SPH, PBD and FEM, on the free node velocity of the substep."""
        grid, params = self._mpm_grid, self._mpm_params
        for solver, is_coupled, only_free in (
            (self._sim.sph_solver, self._sph_coupled, False),
            (self._sim.pbd_solver, self._pbd_coupled, True),
        ):
            if is_coupled:
                stencil = int(np.floor(self._dx / solver.hash_grid_cell_size)) + 2
                kernel_exchange_particles(grid, params, solver, stencil, only_free)
        if self._fem_coupled:
            kernel_exchange_fem(grid, params, self._sim.fem_solver, f)

    def _substep_launches(self, f):
        """The substep of `kernel_substep` as separate launches, with the Newton, line-search and PCG loops and the
        capacity growth driven from the host. When coupled with Rigid, the Newton solve is over the grid displacement
        and Rigid's dofs together, and finishes Rigid's substep."""
        rigid = self._sim.rigid_solver
        grid, newton, linear, control = self._mpm_grid, self._mpm_newton, self._mpm_linear, self._mpm_control
        params, config, coupling = self._mpm_params, self._mpm_config, self._mpm_coupling
        has_rigid = self._rigid_coupled
        energy_args = (self.particles, self.particles_info, self._mpm_materials, grid, newton, control, params, f)

        build_grid(self.particles, self.particles_ng, grid, params, f)
        while int(grid.grid_overflow.to_numpy()):
            self._grow_grid()
            build_grid(self.particles, self.particles_ng, grid, params, f)
        kernel_predict(
            self.particles,
            self.particles_info,
            grid,
            newton,
            self._mpm_force_fields,
            params,
            self._gravity,
            f,
            self._sim.cur_t,
        )
        self._exchange_momentum(f)
        # The physical Hessian at u = 0, whose node diagonal blocks give the contact rows their inverse weights
        self._assemble(f)
        kernel_contact_rows(
            self.particles,
            grid,
            newton,
            linear,
            control,
            config,
            params,
            rigid.dyn_state.geoms,
            rigid.dyn_info.geoms,
            rigid.dyn_state.links,
            rigid.dyn_info.links,
            rigid.rigid_info,
            rigid.collider.sdf.sdf_info,
            f,
        )
        if has_rigid:
            kernel_rigid_begin(coupling, *self._rigid_args())
            self._rigid_point_disp(f)
        kernel_energy(*energy_args, 0)
        if has_rigid:
            self._rigid_energy(0)

        if USE_NEWTON_GRAPH:
            # The assembly above sized the triplet storage for the substep's fixed grid topology, which every assembly
            # of the Newton loop shares
            newton_args = (self.particles, self.particles_info, self._mpm_materials, grid, newton, linear, control)
            if has_rigid:
                kernel_coupled_newton_loop(*newton_args, coupling, *self._rigid_args(), f)
            else:
                kernel_newton_loop(*newton_args, params, f)
        else:
            self._newton_loop_launches(f)

        kernel_finalize(self.particles, self.particles_ng, grid, newton, params, f)
        if has_rigid:
            kernel_rigid_write_back(self.particles, grid, newton, coupling, *self._rigid_args(), f)
            rigid.finish_deferred_substep()
            kernel_add_mpm_contact_force(coupling, rigid.dyn_state.links, params)

    def _substep_graph(self, f):
        if self._rigid_coupled:
            gs.raise_exception(
                "The implicit MPM substep graph does not support the coupling with Rigid dofs; use USE_NEWTON_GRAPH."
            )
        if self._sph_coupled or self._pbd_coupled or self._fem_coupled:
            gs.raise_exception(
                "The implicit MPM substep graph does not support the coupling with SPH, PBD or FEM; use "
                "USE_NEWTON_GRAPH."
            )
        rigid = self._sim.rigid_solver
        args = (
            self.particles,
            self.particles_ng,
            self.particles_info,
            self._mpm_materials,
            self._mpm_grid,
            self._mpm_newton,
            self._mpm_linear,
            self._mpm_control,
            self._mpm_force_fields,
            self._mpm_config,
            self._mpm_params,
            self._gravity,
            rigid.dyn_state.geoms,
            rigid.dyn_info.geoms,
            rigid.dyn_state.links,
            rigid.dyn_info.links,
            rigid.rigid_info,
            rigid.collider.sdf.sdf_info,
            f,
            self._sim.cur_t,
        )
        status = kernel_substep(*args)
        while status.yielded:
            if status.checkpoint == MpmCheckpoint.GRID:
                self._grow_grid()
            elif status.checkpoint == MpmCheckpoint.ASSEMBLE:
                self._grow_triplets()
            else:
                gs.raise_exception(f"The implicit MPM substep yielded at an unexpected checkpoint: {status.checkpoint}")
            status = kernel_substep.resume(*args, from_checkpoint=status.checkpoint)

    def substep_post_coupling_grad(self, f):
        gs.raise_exception("Differentiable simulation is not supported by the implicit MPM solver yet.")

    @qd.kernel
    def copy_frame(self, source: qd.i32, target: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles.pos[target, i_p, i_b] = self.particles.pos[source, i_p, i_b]
            self.particles.vel[target, i_p, i_b] = self.particles.vel[source, i_p, i_b]
            self.particles.F[target, i_p, i_b] = self.particles.F[source, i_p, i_b]
            self.particles.C[target, i_p, i_b] = self.particles.C[source, i_p, i_b]
            self.particles.Jp[target, i_p, i_b] = self.particles.Jp[source, i_p, i_b]
            self.particles.actu[target, i_p, i_b] = self.particles.actu[source, i_p, i_b]

            self.particles_ng.active[target, i_p, i_b] = self.particles_ng.active[source, i_p, i_b]

    @qd.kernel
    def copy_grad(self, source: qd.i32, target: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles.grad[target, i_p, i_b].pos = self.particles.grad[source, i_p, i_b].pos
            self.particles.grad[target, i_p, i_b].vel = self.particles.grad[source, i_p, i_b].vel
            self.particles.grad[target, i_p, i_b].F = self.particles.grad[source, i_p, i_b].F
            self.particles.grad[target, i_p, i_b].C = self.particles.grad[source, i_p, i_b].C
            self.particles.grad[target, i_p, i_b].Jp = self.particles.grad[source, i_p, i_b].Jp
            self.particles.grad[target, i_p, i_b].actu = self.particles.grad[source, i_p, i_b].actu
            self.particles_ng.active[target, i_p, i_b] = self.particles_ng.active[source, i_p, i_b]

    @qd.kernel
    def reset_grad_till_frame(self, f: qd.i32):
        # Zero out particle grads in frames [0, f-1], for all particles, all batch indices
        for i_f, i_p, i_b in qd.ndrange(f, self._n_particles, self._B):
            self.particles.grad[i_f, i_p, i_b].pos = qd.Vector.zero(gs.qd_float, 3)
            self.particles.grad[i_f, i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            self.particles.grad[i_f, i_p, i_b].C = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].F = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].Jp = gs.qd_float(0.0)
            self.particles.grad[i_f, i_p, i_b].actu = gs.qd_float(0.0)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def add_grad_from_state(self, state):
        if self.is_active:
            if state.pos.grad is not None:
                assert state.pos.grad.is_contiguous()
                self.add_grad_from_pos(self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                assert state.vel.grad.is_contiguous()
                self.add_grad_from_vel(self._sim.cur_substep_local, state.vel.grad)

            if state.C.grad is not None:
                assert state.C.grad.is_contiguous()
                self.add_grad_from_C(self._sim.cur_substep_local, state.C.grad)

            if state.F.grad is not None:
                assert state.F.grad.is_contiguous()
                self.add_grad_from_F(self._sim.cur_substep_local, state.F.grad)

            if state.Jp.grad is not None:
                assert state.Jp.grad.is_contiguous()
                self.add_grad_from_Jp(self._sim.cur_substep_local, state.Jp.grad)

    @qd.kernel
    def add_grad_from_pos(self, f: qd.i32, pos_grad: qd.types.ndarray()):
        # pos_grad shape: [B, n_particles, 3]
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                self.particles.grad[f, i_p, i_b].pos[j] += pos_grad[i_b, i_p, j]

    @qd.kernel
    def add_grad_from_vel(self, f: qd.i32, vel_grad: qd.types.ndarray()):
        # vel_grad shape: [B, n_particles, 3]
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                self.particles.grad[f, i_p, i_b].vel[j] += vel_grad[i_b, i_p, j]

    @qd.kernel
    def add_grad_from_C(self, f: qd.i32, C_grad: qd.types.ndarray()):
        # C_grad shape: [B, n_particles, 3, 3]
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                for k in qd.static(range(3)):
                    self.particles.grad[f, i_p, i_b].C[j, k] += C_grad[i_b, i_p, j, k]

    @qd.kernel
    def add_grad_from_F(self, f: qd.i32, F_grad: qd.types.ndarray()):
        # F_grad shape: [B, n_particles, 3, 3]
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                for k in qd.static(range(3)):
                    self.particles.grad[f, i_p, i_b].F[j, k] += F_grad[i_b, i_p, j, k]

    @qd.kernel
    def add_grad_from_Jp(self, f: qd.i32, Jp_grad: qd.types.ndarray()):
        # Jp_grad shape: [B, n_particles]
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles.grad[f, i_p, i_b].Jp += Jp_grad[i_b, i_p]

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = {
                    "pos": torch.zeros((self._B, self._n_particles, 3), dtype=gs.tc_float, device=gs.device),
                    "vel": torch.zeros((self._B, self._n_particles, 3), dtype=gs.tc_float, device=gs.device),
                    "C": torch.zeros((self._B, self._n_particles, 3, 3), dtype=gs.tc_float, device=gs.device),
                    "F": torch.zeros((self._B, self._n_particles, 3, 3), dtype=gs.tc_float, device=gs.device),
                    "Jp": torch.zeros((self._B, self._n_particles), dtype=gs.tc_float, device=gs.device),
                    "actu": torch.zeros((self._B, self._n_particles), dtype=gs.tc_float, device=gs.device),
                    "active": torch.zeros((self._B, self._n_particles), dtype=gs.tc_bool, device=gs.device),
                }

            self._kernel_get_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["actu"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

        # restart from frame 0 in memory
        self.copy_frame(self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)

        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)

            self._kernel_set_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["actu"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name=ckpt_name)

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.actu, state.active)

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
                self.particles.pos[f, i_p, i_b][j] = pos[i_b, i_p, j]
                self.particles.vel[f, i_p, i_b][j] = vel[i_b, i_p, j]
                # Write C, F
                for k in qd.static(range(3)):
                    self.particles.C[f, i_p, i_b][j, k] = C[i_b, i_p, j, k]
                    self.particles.F[f, i_p, i_b][j, k] = F[i_b, i_p, j, k]
            # Write Jp, actu, active
            self.particles.Jp[f, i_p, i_b] = Jp[i_b, i_p]
            self.particles.actu[f, i_p, i_b] = actu[i_b, i_p]
            self.particles_ng.active[f, i_p, i_b] = active[i_b, i_p]

    def get_state(self, f):
        if not self.is_active:
            return None

        state = MPMSolverState(self._scene)
        self._kernel_get_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.actu, state.active)
        self._queried_states.append(state)
        return state

    @qd.kernel
    def _kernel_get_state(
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
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self.particles.pos[f, i_p, i_b][j]
                vel[i_b, i_p, j] = self.particles.vel[f, i_p, i_b][j]
                for k in qd.static(range(3)):
                    C[i_b, i_p, j, k] = self.particles.C[f, i_p, i_b][j, k]
                    F[i_b, i_p, j, k] = self.particles.F[f, i_p, i_b][j, k]
            Jp[i_b, i_p] = self.particles.Jp[f, i_p, i_b]
            actu[i_b, i_p] = self.particles.actu[f, i_p, i_b]
            active[i_b, i_p] = qd.cast(self.particles_ng.active[f, i_p, i_b], gs.qd_bool)

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng.active[f, i_p, i_b]:
                self.particles_render[i_p, i_b].pos = self.particles.pos[f, i_p, i_b]
                self.particles_render[i_p, i_b].vel = self.particles.vel[f, i_p, i_b]
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng.active[f, i_p, i_b]

        for i_v, i_b in qd.ndrange(self._n_vverts, self._B):
            vvert_pos = qd.Vector.zero(gs.qd_float, 3)
            for j in range(self._n_vvert_supports):
                vvert_pos += (
                    self.particles.pos[f, self.vverts_info.support_idxs[i_v][j], i_b]
                    * self.vverts_info.support_weights[i_v][j]
                )
            self.vverts_render[i_v, i_b].pos = vvert_pos
            self.vverts_render[i_v, i_b].active = self.particles_render[
                self.vverts_info.support_idxs[i_v][0], i_b
            ].active

    @qd.kernel
    def _kernel_add_particles(
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

            self.particles_info.material_idx[i_p] = material_idx
            self.particles_info.default_Jp[i_p] = mat_default_Jp
            self.particles_info.mass[i_p] = self._particle_volume * mat_rho
            self.particles_info.free[i_p] = True
            self.particles_info.muscle_group[i_p] = 0
            self.particles_info.muscle_direction[i_p] = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start

            self.particles_ng.active[f, i_p, i_b] = qd.cast(active, gs.qd_bool)
            for i in qd.static(range(3)):
                self.particles.pos[f, i_p, i_b][i] = pos[i_p_, i]

            self.particles.vel[f, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.particles.F[f, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
            self.particles.C[f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles.Jp[f, i_p, i_b] = mat_default_Jp
            self.particles.actu[f, i_p, i_b] = gs.qd_float(0.0)

    @qd.kernel
    def _kernel_set_particles_pos(
        self, f: qd.i32, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]

            for i in qd.static(range(3)):
                self.particles.pos[f, i_p, i_b][i] = poss[i_b_, i_p_, i]

            # Reset these attributes whenever overwritting particle positions manually
            self.particles.vel[f, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.particles.F[f, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
            self.particles.C[f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles.Jp[f, i_p, i_b] = self.particles_info.default_Jp[i_p]

    @qd.kernel
    def _kernel_set_particles_pos_grad(
        self,
        f: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        poss_grad: qd.types.ndarray(),  # shape [B, n_particles, 3]
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            for i in qd.static(range(3)):
                poss_grad[i_b, i_p_, i] = self.particles.grad[f, i_p, i_b].pos[i]

    @qd.kernel
    def _kernel_get_particles_pos(
        self,
        f: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles.pos[f, i_p, i_b][i]

    @qd.kernel
    def _kernel_set_particles_vel(
        self,
        f: qd.i32,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),  # shape [B, n_particles, 3]
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles.vel[f, i_p, i_b][i] = vels[i_b_, i_p_, i]

    @qd.kernel
    def _kernel_set_particles_vel_grad(
        self,
        f: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        vels_grad: qd.types.ndarray(),  # shape [B, n_particles, 3]
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            for i in qd.static(range(3)):
                vels_grad[i_b, i_p_, i] = self.particles.grad[f, i_p, i_b].vel[i]

    @qd.kernel
    def _kernel_get_particles_vel(
        self,
        f: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles.vel[f, i_p, i_b][i]

    @qd.kernel
    def _kernel_set_particles_active(
        self,
        f: qd.i32,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles_ng.active[f, i_p, i_b] = qd.cast(actives[i_b_, i_p_], gs.qd_bool)

    @qd.kernel
    def _kernel_get_particles_active(
        self,
        f: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng.active[f, i_p, i_b]

    @qd.kernel
    def _kernel_set_particles_actu(
        self,
        f: qd.i32,
        n_groups: qd.i32,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actus: qd.types.ndarray(),  # shape [B, n_particles, n_groups]
    ):
        for i_p_, i_g, i_b_ in qd.ndrange(particles_idx.shape[1], n_groups, envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            if self.particles_info.muscle_group[i_p] == i_g:
                self.particles.actu[f, i_p, i_b] = actus[i_b_, i_p_, i_g]

    @qd.kernel
    def _kernel_set_particles_actu_grad(
        self,
        f: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        actus_grad: qd.types.ndarray(),  # shape [B, n_particles, n_groups]
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            # Each particle reads the column of its muscle group (see _kernel_set_particles_actu)
            i_g = self.particles_info.muscle_group[i_p]
            actus_grad[i_b, i_p_, i_g] = self.particles.grad[f, i_p, i_b].actu
            # The input overwrites the actuation, so the value it replaces has no effect past this substep
            self.particles.grad[f, i_p, i_b].actu = 0.0

    @qd.kernel
    def _kernel_get_particles_actu(
        self,
        f: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        actus: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actus[i_b_, i_p_] = self.particles.actu[f, i_p, i_b]

    @qd.kernel
    def _kernel_set_particles_muscle_group(self, particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            self.particles_info.muscle_group[i_p] = muscle_group[i_p_]

    @qd.kernel
    def _kernel_get_particles_muscle_group(
        self, particle_start: qd.i32, n_particles: qd.i32, muscle_group: qd.types.ndarray()
    ):
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            muscle_group[i_p_] = self.particles_info.muscle_group[i_p]

    @qd.kernel
    def _kernel_set_particles_muscle_direction(
        self, particles_idx: qd.types.ndarray(), muscle_direction: qd.types.ndarray()
    ):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            for i in qd.static(range(3)):
                self.particles_info.muscle_direction[i_p][i] = muscle_direction[i_p_, i]

    @qd.kernel
    def _kernel_set_particles_free(self, particles_idx: qd.types.ndarray(), free: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            self.particles_info.free[i_p] = free[i_p_]

    @qd.kernel
    def _kernel_get_particles_free(self, particle_start: qd.i32, n_particles: qd.i32, free: qd.types.ndarray()):
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            free[i_p_] = self.particles_info.free[i_p]

    @qd.kernel
    def _kernel_get_mass(
        self, particle_start: qd.i32, n_particles: qd.i32, mass: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        total_mass = gs.qd_float(0.0)
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            total_mass += self.particles_info.mass[i_p]
        total_mass = total_mass / self._particle_volume_scale
        for i_b_ in range(envs_idx.shape[0]):
            mass[i_b_] = total_mass

    # ------------------------------------------------------------------------------------
    # -------------------------------- particle constraints ------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_set_particle_constraints(
        self,
        f: qd.i32,
        particles_mask: qd.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
        particle_start: qd.i32,
        stiffness: qd.f32,
        link_idx: qd.i32,
        link_pos: qd.types.ndarray(),  # shape [n_envs, 3]
        link_quat: qd.types.ndarray(),  # shape [n_envs, 4]
    ):
        for i_p_local, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_local]:
                i_p = i_p_local + particle_start

                # Get current particle position
                pos = self.particles.pos[f, i_p, i_b]

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
        particles_mask: qd.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
        particle_start: qd.i32,
    ):
        for i_p_local, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_local]:
                i_p = i_p_local + particle_start
                self.particle_constraints[i_p, i_b].is_constrained = False
                self.particle_constraints[i_p, i_b].link_idx = -1

    @qd.kernel
    def apply_particle_constraints(self, f: qd.i32, links_state: array_class.LinksState):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particle_constraints[i_p, i_b].is_constrained:
                # Update target position from link pose
                i_l = self.particle_constraints[i_p, i_b].link_idx
                if i_l >= 0:
                    link_pos = links_state.pos[i_l, i_b]
                    link_quat = links_state.quat[i_l, i_b]
                    local_pos = self.particle_constraints[i_p, i_b].link_local_pos
                    target = gu.qd_transform_by_trans_quat(local_pos, link_pos, link_quat)
                    self.particle_constraints[i_p, i_b].target_pos = target

                # Apply spring force to velocity
                target_pos = self.particle_constraints[i_p, i_b].target_pos
                stiffness = self.particle_constraints[i_p, i_b].stiffness
                mass = self.particles_info.mass[i_p] / self._particle_volume_scale

                pos = self.particles.pos[f + 1, i_p, i_b]
                vel = self.particles.vel[f + 1, i_p, i_b]

                pos_error = pos - target_pos
                spring_force = -stiffness * pos_error
                damping_force = -2.0 * qd.math.sqrt(stiffness * mass) * vel

                dv = self.substep_dt * (spring_force + damping_force) / mass
                self.particles.vel[f + 1, i_p, i_b] = vel + dv

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


class MpmCheckpoint(IntEnum):
    """Checkpoints of the substep graph. GRID and ASSEMBLE pause on a capacity overflow; the others only gate a
    resume, so that a resumed launch skips every stage before its checkpoint."""

    GRID = 0
    PREDICT = 1
    ASSEMBLE = 2
    SOLVE = 3
    PCG = 4
    SOLVE_POST = 5
    LINE_SEARCH_TRIAL = 6
    NEWTON_POST = 7
    FINALIZE = 8


@qd.kernel(graph=True, checkpoints=True)
def kernel_substep(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    control: qd.template(),
    force_fields: qd.template(),
    config: qd.template(),
    params: qd.template(),
    gravity: qd.Tensor,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    f: qd.i32,
    t: qd.f32,
):
    """One implicit MPM substep, from frame f to frame f + 1, as one graph.

    Every stage sits under a checkpoint: a launch resumed after a capacity yield must not run any stage before the
    resume point (Quadrants #956 may otherwise run a top-level func after a yielding checkpoint once). Checkpoint
    bodies wrap their calls in a static branch, as checkpoint lowering rejects bare top-level func calls.
    """
    dt = params.substep_dt[()]
    with qd.checkpoint(MpmCheckpoint.GRID, yield_on=grid.grid_overflow):
        if qd.static(True):
            mpm_func.func_build_grid(particles, particles_ng, grid, params, f)
    with qd.checkpoint(MpmCheckpoint.PREDICT, yield_on=control.never_yield):
        if qd.static(True):
            mpm_func.func_p2g(particles, particles_info, grid, params, gravity, f, dt)
            mpm_func.func_apply_force_fields(grid, force_fields, params, t, dt)
            mpm_func.func_fix_nodes(grid)
            mpm_func.func_reset_contact_rows(grid, newton)
            mpm_func.func_newton_reset(grid, newton)
            # The physical Hessian at u = 0, whose node diagonal blocks give the contact rows their inverse weights
            mpm_func.func_report_extent(grid, linear)
            mpm_func.func_assemble(particles, particles_info, materials, grid, newton, linear, params, f, dt)
            func_sort_reduce_bcoo(linear.matrix)
            mpm_func.func_gather_node_diag(grid, linear)
            mpm_func.func_build_wall_rows(particles, grid, newton, params, f, dt)
            if qd.static(config.has_rigid):
                func_build_rigid_rows(
                    particles,
                    grid,
                    newton,
                    geoms_state,
                    geoms_info,
                    links_state,
                    links_info,
                    rigid_info,
                    sdf_info,
                    config,
                    config,
                    params,
                    f,
                    dt,
                )
            mpm_func.func_newton_start(grid, newton, params, dt)
            func_reset_newton_control(control)
            mpm_func.func_energy(particles, particles_info, materials, grid, newton, control, params, f, dt, False)

    while qd.graph.do_while(control.newton_cond):
        with qd.checkpoint(MpmCheckpoint.ASSEMBLE, yield_on=linear.matrix.triplet_overflow):
            if qd.static(True):
                func_reset_newton_measures(control)
                mpm_func.func_report_extent(grid, linear)
                mpm_func.func_assemble(particles, particles_info, materials, grid, newton, linear, params, f, dt)
        with qd.checkpoint(MpmCheckpoint.SOLVE, yield_on=control.never_yield):
            if qd.static(True):
                func_sort_reduce_bcoo(linear.matrix)
                mpm_func.func_reset_brick(grid, linear)
                mpm_func.func_gather_brick(linear)
                mpm_func.func_invert_brick(grid, linear)
                func_init_linear_pcg_residual(linear.pcg, linear.b_rhs, linear.solution)
                mpm_func.func_apply_brick(grid, linear)
                func_init_linear_pcg_direction(linear.pcg)
        with qd.checkpoint(MpmCheckpoint.PCG, yield_on=control.never_yield):
            if qd.static(True):
                mpm_func.func_solve_brick_pcg(grid, linear)
        with qd.checkpoint(MpmCheckpoint.SOLVE_POST, yield_on=control.never_yield):
            if qd.static(True):
                mpm_func.func_negate_du(grid, newton, linear)
                mpm_func.func_record_start_point(grid, newton)
                mpm_func.func_contribute_newton_max_disp(particles, grid, newton, control, params, f)
                func_check_convergence(control)
                func_init_line_search(control)
        while qd.graph.do_while(control.ls_cond):
            with qd.checkpoint(MpmCheckpoint.LINE_SEARCH_TRIAL, yield_on=control.never_yield):
                if qd.static(True):
                    mpm_func.func_step_forward(grid, newton, control)
                    mpm_func.func_energy(
                        particles, particles_info, materials, grid, newton, control, params, f, dt, True
                    )
                    func_accept_or_backtrack(control)
        with qd.checkpoint(MpmCheckpoint.NEWTON_POST, yield_on=control.never_yield):
            if qd.static(True):
                func_update_convergence(control)

    with qd.checkpoint(MpmCheckpoint.FINALIZE, yield_on=control.never_yield):
        if qd.static(True):
            mpm_func.func_grid_velocity(grid, newton, dt)
            mpm_func.func_g2p_advect(particles, particles_ng, grid, params, f, dt)


@qd.kernel
def kernel_grid_keys(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    grid: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    mpm_func.func_reset_grid(grid)
    mpm_func.func_compose_particle_keys(particles, particles_ng, grid, params, f)


@qd.kernel
def kernel_grid_blocks(
    grid: qd.template(),
    params: qd.template(),
):
    mpm_func.func_append_particle_blocks(grid, params)
    mpm_func.func_gen_neighbor_keys(grid)


@qd.kernel
def kernel_grid_block_flags(grid: qd.template()):
    mpm_func.func_block_flags(grid)


@qd.kernel
def kernel_grid_publish_blocks(grid: qd.template()):
    mpm_func.func_publish_blocks(grid)
    mpm_func.func_publish_grid_demand(grid)
    mpm_func.func_cell_flags(grid)


@qd.kernel
def kernel_grid_cells(grid: qd.template()):
    mpm_func.func_publish_cells(grid)
    mpm_func.func_reset_cell_segs(grid)
    mpm_func.func_fill_cell_segs(grid)


def build_grid(particles, particles_ng, grid, params, f):
    """Host-driven func_build_grid, its sorts and scans shared launches."""
    kernel_grid_keys(particles, particles_ng, grid, params, f)
    dynamic_radix_sort(
        grid.particle_sort, grid.sort_keys, grid.sort_keys_tmp, grid.sort_ids, grid.sort_ids_tmp, grid.n_sort_keys
    )
    kernel_grid_blocks(grid, params)
    dynamic_radix_sort(
        grid.block_sort, grid.nbr_keys, grid.nbr_keys_tmp, grid.nbr_perm, grid.nbr_perm_tmp, grid.n_neighbor_keys
    )
    kernel_grid_block_flags(grid)
    kernel_dynamic_exclusive_sum(grid.block_scan, grid.block_flags, grid.block_ranks, grid.n_neighbor_keys)
    kernel_grid_publish_blocks(grid)
    kernel_dynamic_exclusive_sum(grid.cell_scan, grid.cell_flags, grid.cell_ranks, grid.n_valid)
    kernel_dynamic_exclusive_sum(grid.task_scan, grid.task_flags, grid.task_ranks, grid.n_valid)
    kernel_grid_cells(grid)


@qd.kernel
def kernel_predict(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),
    newton: qd.template(),
    force_fields: qd.template(),
    params: qd.template(),
    gravity: qd.Tensor,
    f: qd.i32,
    t: qd.f32,
):
    dt = params.substep_dt[()]
    mpm_func.func_p2g(particles, particles_info, grid, params, gravity, f, dt)
    mpm_func.func_apply_force_fields(grid, force_fields, params, t, dt)
    mpm_func.func_fix_nodes(grid)
    mpm_func.func_reset_contact_rows(grid, newton)
    mpm_func.func_newton_reset(grid, newton)


@qd.kernel
def kernel_contact_rows(
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    control: qd.template(),
    config: qd.template(),
    params: qd.template(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    f: qd.i32,
):
    """Contact rows from the reduced physical Hessian at u = 0, then arm the Newton loop."""
    dt = params.substep_dt[()]
    mpm_func.func_gather_node_diag(grid, linear)
    mpm_func.func_build_wall_rows(particles, grid, newton, params, f, dt)
    if qd.static(config.has_rigid):
        func_build_rigid_rows(
            particles,
            grid,
            newton,
            geoms_state,
            geoms_info,
            links_state,
            links_info,
            rigid_info,
            sdf_info,
            config,
            config,
            params,
            f,
            dt,
        )
    mpm_func.func_newton_start(grid, newton, params, dt)
    func_reset_newton_control(control)


@qd.kernel
def kernel_energy(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),
    newton: qd.template(),
    control: qd.template(),
    params: qd.template(),
    f: qd.i32,
    is_trial: qd.i32,
):
    mpm_func.func_energy(
        particles, particles_info, materials, grid, newton, control, params, f, params.substep_dt[()], is_trial
    )


@qd.kernel
def kernel_assemble(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    control: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    dt = params.substep_dt[()]
    func_reset_newton_measures(control)
    mpm_func.func_report_extent(grid, linear)
    mpm_func.func_assemble(particles, particles_info, materials, grid, newton, linear, params, f, dt)


@qd.kernel
def kernel_pcg_start(
    grid: qd.template(),
    linear: qd.template(),
):
    """Brick preconditioner of the reduced matrix, and the PCG start from x = 0: r = b, z = M^-1 r, p = z."""
    mpm_func.func_reset_brick(grid, linear)
    mpm_func.func_gather_brick(linear)
    mpm_func.func_invert_brick(grid, linear)
    func_init_linear_pcg_residual(linear.pcg, linear.b_rhs, linear.solution)
    mpm_func.func_apply_brick(grid, linear)
    func_init_linear_pcg_direction(linear.pcg)


@qd.kernel(graph=True)
def kernel_pcg_loop(
    grid: qd.template(),
    linear: qd.template(),
):
    """The PCG iterations after kernel_pcg_start, until convergence or the iteration budget, as one graph."""
    while qd.graph.do_while(linear.pcg.cond):
        func_linear_pcg_spmv(linear.pcg, linear.matrix)
        func_linear_pcg_update_xr(linear.pcg, linear.solution)
        mpm_func.func_apply_brick(grid, linear)
        func_linear_pcg_direction_update(linear.pcg)


@qd.kernel
def kernel_start_line_search(
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    control: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    mpm_func.func_negate_du(grid, newton, linear)
    mpm_func.func_record_start_point(grid, newton)
    mpm_func.func_contribute_newton_max_disp(particles, grid, newton, control, params, f)
    func_check_convergence(control)
    func_init_line_search(control)


@qd.kernel
def kernel_step_forward(
    grid: qd.template(),
    newton: qd.template(),
    control: qd.template(),
):
    mpm_func.func_step_forward(grid, newton, control)


@qd.kernel
def kernel_accept_or_backtrack(control: qd.template()):
    func_accept_or_backtrack(control)


@qd.kernel
def kernel_update_convergence(control: qd.template()):
    func_update_convergence(control)


@qd.kernel(graph=True)
def kernel_newton_loop(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    control: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    """The Newton iterations of a substep as one graph, after the start energy armed the loop.

    The triplet demand of an assembly depends on the grid topology only, fixed for the substep, so the assembly that
    sized the storage before the loop covers every assembly inside it and the graph needs no capacity checkpoint.
    """
    dt = params.substep_dt[()]
    while qd.graph.do_while(control.newton_cond):
        func_reset_newton_measures(control)
        mpm_func.func_report_extent(grid, linear)
        mpm_func.func_assemble(particles, particles_info, materials, grid, newton, linear, params, f, dt)
        func_sort_reduce_bcoo_graph(linear.matrix)

        mpm_func.func_reset_brick(grid, linear)
        mpm_func.func_gather_brick(linear)
        mpm_func.func_invert_brick(grid, linear)
        func_init_linear_pcg_residual(linear.pcg, linear.b_rhs, linear.solution)
        mpm_func.func_apply_brick(grid, linear)
        func_init_linear_pcg_direction(linear.pcg)
        while qd.graph.do_while(linear.pcg.cond):
            func_linear_pcg_spmv(linear.pcg, linear.matrix)
            func_linear_pcg_update_xr(linear.pcg, linear.solution)
            mpm_func.func_apply_brick(grid, linear)
            func_linear_pcg_direction_update(linear.pcg)

        mpm_func.func_negate_du(grid, newton, linear)
        mpm_func.func_record_start_point(grid, newton)
        mpm_func.func_contribute_newton_max_disp(particles, grid, newton, control, params, f)
        func_check_convergence(control)
        func_init_line_search(control)
        while qd.graph.do_while(control.ls_cond):
            mpm_func.func_step_forward(grid, newton, control)
            mpm_func.func_energy(particles, particles_info, materials, grid, newton, control, params, f, dt, 1)
            func_accept_or_backtrack(control)
        func_update_convergence(control)


@qd.kernel
def kernel_exchange_particles(
    grid: qd.template(),
    params: qd.template(),
    partner: qd.template(),
    stencil: qd.template(),
    only_free: qd.template(),
):
    mpm_exchange_func.func_exchange_particles(grid, params, partner, stencil, only_free)


@qd.kernel
def kernel_exchange_fem(
    grid: qd.template(),
    params: qd.template(),
    fem: qd.template(),
    f: qd.i32,
):
    mpm_exchange_func.func_exchange_fem(grid, params, fem, f)


@qd.kernel
def kernel_finalize(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    grid: qd.template(),
    newton: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    dt = params.substep_dt[()]
    mpm_func.func_grid_velocity(grid, newton, dt)
    mpm_func.func_g2p_advect(particles, particles_ng, grid, params, f, dt)


# ------------------------------------------ joint solve with Rigid ------------------------------------------


@qd.kernel
def kernel_rigid_begin(
    coupling: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
    params: qd.template(),
):
    mpm_rigid_func.func_rigid_begin(coupling, dyn_state.dofs, constraint_state, rigid_config, config, params)


@qd.kernel
def kernel_rigid_point_disp(
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    coupling: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    mpm_rigid_func.func_link_spatial_disp(coupling, coupling.x, dyn_info.links, dyn_state.dofs, config)
    mpm_rigid_func.func_contact_point_disp(particles, grid, newton, coupling, dyn_state.links, params, f)


@qd.kernel
def kernel_rigid_energy(
    coupling: qd.template(),
    control: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
    params: qd.template(),
    is_trial: qd.i32,
):
    mpm_rigid_func.func_rigid_evaluate(
        coupling, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, config, params
    )
    mpm_rigid_func.func_rigid_energy(coupling, control, params, is_trial)


@qd.kernel
def kernel_rigid_newton_prepare(
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    coupling: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    mpm_rigid_func.func_rigid_evaluate(
        coupling, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, config, params
    )
    mpm_rigid_func.func_rigid_gradient(
        particles,
        grid,
        newton,
        coupling,
        dyn_state,
        constraint_state,
        dyn_info,
        rigid_info,
        rigid_config,
        config,
        params,
        f,
    )


@qd.kernel
def kernel_coupled_pcg_start(
    grid: qd.template(),
    linear: qd.template(),
    coupling: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
    params: qd.template(),
):
    """kernel_pcg_start over the MPM rows and the Rigid dofs, preconditioned by the bricks and Rigid's factor."""
    mpm_func.func_reset_brick(grid, linear)
    mpm_func.func_gather_brick(linear)
    mpm_func.func_invert_brick(grid, linear)
    func_init_linear_pcg_residual(linear.pcg, linear.b_rhs, linear.solution)
    mpm_rigid_func.func_rigid_pcg_init(coupling)
    mpm_func.func_apply_brick(grid, linear)
    mpm_rigid_func.func_rigid_precondition(coupling, dyn_state, constraint_state, rigid_info, rigid_config, config)
    func_init_linear_pcg_direction(linear.pcg)
    mpm_rigid_func.func_rigid_pcg_init_direction(coupling, linear.pcg)


@qd.kernel(graph=True)
def kernel_coupled_pcg_loop(
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    coupling: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    """kernel_pcg_loop over the MPM rows and the Rigid dofs, with the joint operator."""
    while qd.graph.do_while(linear.pcg.cond):
        func_linear_pcg_spmv(linear.pcg, linear.matrix)
        mpm_rigid_func.func_rigid_pcg_matvec(
            particles,
            grid,
            newton,
            linear,
            coupling,
            dyn_state,
            constraint_state,
            dyn_info,
            rigid_info,
            rigid_config,
            config,
            params,
            f,
        )
        mpm_rigid_func.func_rigid_pcg_update_xr(coupling, linear.pcg)
        func_linear_pcg_update_xr(linear.pcg, linear.solution)
        mpm_func.func_apply_brick(grid, linear)
        mpm_rigid_func.func_rigid_precondition(coupling, dyn_state, constraint_state, rigid_info, rigid_config, config)
        mpm_rigid_func.func_rigid_pcg_dot_rz(coupling, linear.pcg)
        func_linear_pcg_direction_update(linear.pcg)
        mpm_rigid_func.func_rigid_pcg_direction(coupling, linear.pcg)


@qd.kernel
def kernel_rigid_start_line_search(coupling: qd.template(), control: qd.template()):
    mpm_rigid_func.func_rigid_start_line_search(coupling, control)


@qd.kernel
def kernel_rigid_step_forward(coupling: qd.template(), control: qd.template()):
    mpm_rigid_func.func_rigid_step_forward(coupling, control)


@qd.kernel(graph=True)
def kernel_coupled_newton_loop(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    control: qd.template(),
    coupling: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    """kernel_newton_loop over the grid displacement and Rigid's dofs together (design §6.2.3): one PCG over both,
    preconditioned by the bricks and Rigid's island factor, and a line search on the joint energy."""
    dt = params.substep_dt[()]
    while qd.graph.do_while(control.newton_cond):
        func_reset_newton_measures(control)
        mpm_func.func_report_extent(grid, linear)
        mpm_func.func_assemble(particles, particles_info, materials, grid, newton, linear, params, f, dt)
        func_sort_reduce_bcoo_graph(linear.matrix)
        mpm_rigid_func.func_rigid_evaluate(
            coupling, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, config, params
        )
        mpm_rigid_func.func_rigid_gradient(
            particles,
            grid,
            newton,
            coupling,
            dyn_state,
            constraint_state,
            dyn_info,
            rigid_info,
            rigid_config,
            config,
            params,
            f,
        )

        mpm_func.func_reset_brick(grid, linear)
        mpm_func.func_gather_brick(linear)
        mpm_func.func_invert_brick(grid, linear)
        func_init_linear_pcg_residual(linear.pcg, linear.b_rhs, linear.solution)
        mpm_rigid_func.func_rigid_pcg_init(coupling)
        mpm_func.func_apply_brick(grid, linear)
        mpm_rigid_func.func_rigid_precondition(coupling, dyn_state, constraint_state, rigid_info, rigid_config, config)
        func_init_linear_pcg_direction(linear.pcg)
        mpm_rigid_func.func_rigid_pcg_init_direction(coupling, linear.pcg)
        while qd.graph.do_while(linear.pcg.cond):
            func_linear_pcg_spmv(linear.pcg, linear.matrix)
            mpm_rigid_func.func_rigid_pcg_matvec(
                particles,
                grid,
                newton,
                linear,
                coupling,
                dyn_state,
                constraint_state,
                dyn_info,
                rigid_info,
                rigid_config,
                config,
                params,
                f,
            )
            mpm_rigid_func.func_rigid_pcg_update_xr(coupling, linear.pcg)
            func_linear_pcg_update_xr(linear.pcg, linear.solution)
            mpm_func.func_apply_brick(grid, linear)
            mpm_rigid_func.func_rigid_precondition(
                coupling, dyn_state, constraint_state, rigid_info, rigid_config, config
            )
            mpm_rigid_func.func_rigid_pcg_dot_rz(coupling, linear.pcg)
            func_linear_pcg_direction_update(linear.pcg)
            mpm_rigid_func.func_rigid_pcg_direction(coupling, linear.pcg)

        mpm_rigid_func.func_rigid_start_line_search(coupling, control)
        mpm_func.func_negate_du(grid, newton, linear)
        mpm_func.func_record_start_point(grid, newton)
        mpm_func.func_contribute_newton_max_disp(particles, grid, newton, control, params, f)
        func_check_convergence(control)
        func_init_line_search(control)
        while qd.graph.do_while(control.ls_cond):
            mpm_func.func_step_forward(grid, newton, control)
            mpm_rigid_func.func_rigid_step_forward(coupling, control)
            mpm_rigid_func.func_link_spatial_disp(coupling, coupling.x, dyn_info.links, dyn_state.dofs, config)
            mpm_rigid_func.func_contact_point_disp(particles, grid, newton, coupling, dyn_state.links, params, f)
            mpm_func.func_energy(particles, particles_info, materials, grid, newton, control, params, f, dt, 1)
            mpm_rigid_func.func_rigid_evaluate(
                coupling, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, config, params
            )
            mpm_rigid_func.func_rigid_energy(coupling, control, params, 1)
            func_accept_or_backtrack(control)
        func_update_convergence(control)


@qd.kernel
def kernel_rigid_write_back(
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    coupling: qd.template(),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
    params: qd.template(),
    f: qd.i32,
):
    mpm_rigid_func.func_rigid_evaluate(
        coupling, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, config, params
    )
    mpm_rigid_func.func_rigid_write_back(coupling, constraint_state, config)
    mpm_rigid_func.func_mpm_reaction(particles, grid, newton, coupling, dyn_state, constraint_state, params, f)


@qd.kernel
def kernel_add_mpm_contact_force(
    coupling: qd.template(),
    links_state: array_class.LinksState,
    params: qd.template(),
):
    mpm_rigid_func.func_add_mpm_contact_force(coupling, links_state, params)
