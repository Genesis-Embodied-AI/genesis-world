from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import MPMEntity
from genesis.engine.materials import MPM
from genesis.engine.numerics.dynamic_exclusive_sum import kernel_dynamic_exclusive_sum
from genesis.engine.numerics.dynamic_radix_sort import dynamic_radix_sort
from genesis.engine.numerics.newton_control import NewtonControl
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import MPMSolverState
from genesis.grad.bridge import no_grad
from genesis.options.solvers import LegacyCouplerOptions, MPMOptions
from genesis.utils.array_class import ErrorCode, LinksState
from genesis.utils.data import DataItem, Namespace, iter_data, iter_state_arrays, namespace
from genesis.utils.misc import (
    DeprecationError,
    broadcast_tensor,
    qd_destroy_data,
    qd_isfinite,
    qd_to_numpy,
    qd_to_torch,
    qd_zero_grad,
)

from . import mpm_adjoint_func, mpm_func
from .mpm_data import (
    BLOCK_CELLS,
    MIN_BLOCKS,
    MpmErrno,
    MpmForceFields,
    MpmGridScratch,
    MpmInfo,
    MpmMaterialTable,
    MpmNewtonScratch,
    MpmParticles,
    MpmParticlesInfo,
    MpmParticlesNG,
    MpmWallPairs,
    build_mpm_grid,
    build_mpm_newton,
    build_wall_pairs,
    initial_block_capacity,
)
from .newton_contact import VertexRange
from .newton_linear import (
    CAPACITY_GROW_FACTOR,
    MIN_SLOTS,
    BlockJacobiOwner,
    BlockJacobiScratch,
    LinearBlock,
    NewtonLinearScratch,
    build_block_jacobi,
    build_linear_block,
    grown_capacity,
    retire_scratch,
)
from .newton_solver import NewtonDevice
from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


def material_table_column(material):
    """Compute the column of a material in the material table, its energy family and its return map."""
    column = np.zeros(mpm_func.N_MATERIAL_PARAMS, dtype=gs.np_float)
    column[mpm_func.PARAM_MU] = material.mu
    column[mpm_func.PARAM_LAM] = material.lam
    if isinstance(material, MPM.Sand):
        column[mpm_func.PARAM_SAND_ALPHA] = material.alpha
        return column, mpm_func.FAMILY_EXTENDED_HENCKY, mpm_func.PLASTIC_SAND
    if isinstance(material, MPM.Liquid):
        return column, mpm_func.FAMILY_LIQUID, mpm_func.PLASTIC_LIQUID
    if isinstance(material, MPM.ElastoPlastic):
        column[mpm_func.PARAM_YIELD_LOWER] = material.yield_lower
        column[mpm_func.PARAM_YIELD_HIGHER] = material.yield_higher
        column[mpm_func.PARAM_YIELD_STRESS] = material.von_mises_yield_stress
        if isinstance(material, MPM.Snow):
            return column, mpm_func.FAMILY_FIXED_COROTATED, mpm_func.PLASTIC_SNOW
        if material.use_von_mises:
            return column, mpm_func.FAMILY_FIXED_COROTATED, mpm_func.PLASTIC_VON_MISES
        return column, mpm_func.FAMILY_FIXED_COROTATED, mpm_func.PLASTIC_CLAMP
    if isinstance(material, MPM.Muscle):
        column[mpm_func.PARAM_FIBRE] = material.stiffness
    if isinstance(material, MPM.Elastic) and material.model == "neohooken":
        return column, mpm_func.FAMILY_STABLE_NEOHOOKEAN, mpm_func.PLASTIC_NONE
    return column, mpm_func.FAMILY_FIXED_COROTATED, mpm_func.PLASTIC_NONE


@qd.data_oriented
class MpmNewtonParticipant(BlockJacobiOwner, NewtonDevice):
    """Grid displacement as a participant of the NewtonSolver."""

    # A wall pair writes the gradient and the Hessian of its row on its particle alone
    n_doublets = 1
    n_triplets = 1
    # The rows of a particle are those of the 8 nodes of its stencil
    n_vertex_rows = 8

    def __init__(
        self,
        particles: MpmParticles,
        particles_info: MpmParticlesInfo,
        materials: MpmMaterialTable,
        grid: MpmGridScratch,
        newton: MpmNewtonScratch,
        jacobi: BlockJacobiScratch,
        wall_pairs: MpmWallPairs,
        mpm_info: MpmInfo,
        n_sort: int,
    ):
        self.particles = particles
        self.particles_info = particles_info
        self.materials = materials
        self.grid = grid
        self.newton = newton
        self.jacobi = jacobi
        self.wall_pairs = wall_pairs
        self.mpm_info = mpm_info
        # The linear system of the solve, bound by the NewtonSolver
        self.linear: NewtonLinearScratch | None = None
        self.linear_block: LinearBlock = build_linear_block(0, 0)
        self.vertex_range = VertexRange(values={"offset": 0, "count": n_sort})
        self._scratches_replaced = []

    def replace_wall_pairs(self, wall_pairs: MpmWallPairs) -> None:
        """Swap the wall pairs, and free the arrays of those replaced (see retire_scratch)."""
        retire_scratch(self.wall_pairs, self._scratches_replaced)
        self.wall_pairs = wall_pairs

    def replace_scratch(self, grid: MpmGridScratch, newton: MpmNewtonScratch, jacobi: BlockJacobiScratch) -> None:
        """Swap the grid, the Newton state and the diagonal blocks, and free the arrays of those replaced (see
        retire_scratch), except the error flags, which the new grid binds again."""
        for scratch_old, scratch in zip((self.grid, self.newton, self.jacobi), (grid, newton, jacobi)):
            if scratch_old is not scratch:
                for path, array in iter_state_arrays(scratch_old):
                    if path[0] != "errors":
                        qd_destroy_data(array)
                self._scratches_replaced.append(scratch_old)
        self.grid, self.newton, self.jacobi = grid, newton, jacobi

    @property
    def max_linear_rows(self) -> int:
        """Rows of the nodes of the grid at its capacity."""
        return self.grid.blocks_key.shape[0] * BLOCK_CELLS

    @property
    def data(self) -> Iterator[DataItem]:
        yield from iter_data(self, "grid")
        yield from iter_data(self, "newton")
        yield from iter_data(self, "jacobi")
        yield from iter_data(self, "linear_block")
        yield from iter_data(self, "wall_pairs")
        yield from iter_data(self, "vertex_range")

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int, is_exact: qd.template()):
        mpm_func.func_assemble(
            f,
            self.mpm_info.substep_dt[()],
            self.particles,
            self.grid,
            self.newton,
            self.linear,
            self.linear_block,
            self.particles_info,
            self.materials,
            self.mpm_info,
            is_exact,
        )

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, f: int, control: NewtonControl):
        mpm_func.func_negate_du(self.grid, self.newton, self.linear, self.linear_block)
        mpm_func.func_record_start_point(self.grid, self.newton)
        mpm_func.func_contribute_newton_max_displacement(
            f, self.particles, self.grid, self.newton, control, self.mpm_info
        )
        mpm_func.func_cap_inversion_step(
            f, self.particles, self.grid, self.newton, control, self.particles_info, self.materials, self.mpm_info
        )

    @qd.func(requires_top_level=True)
    def func_step_forward(self, f: int, control: NewtonControl):
        mpm_func.func_step_forward(self.grid, self.newton, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, f: int, energies: qd.template()):
        dt = self.mpm_info.substep_dt[()]
        mpm_func.func_elastic_energy(
            f, dt, energies, self.particles, self.grid, self.newton, self.particles_info, self.materials, self.mpm_info
        )
        mpm_func.func_kinetic_energy(dt, energies, self.grid, self.newton)

    # ------------------------------------------- vertex map -------------------------------------------

    @qd.func
    def func_vertex_value(self, f: int, i_v: int):
        _, slots, w = mpm_func.particle_stencil(i_v, self.grid, self.newton, self.mpm_info)
        return mpm_func.particle_node_sum(slots, w, self.newton.u)

    @qd.func
    def func_vertex_rows(self, i_v: int):
        _, slots, w = mpm_func.particle_stencil(i_v, self.grid, self.newton, self.mpm_info)
        return 8, self.linear_block.row_start[()] + slots, w

    # ------------------------------------------- wall pairs -------------------------------------------

    @qd.func
    def func_pair_count(self):
        return self.wall_pairs.n_pairs[()]

    @qd.func
    def wall_pair_terms(self, i_pair: int):
        """Environment, energy, gradient and Hessian of the row of a wall pair at the current iterate."""
        i_b, slots, w = mpm_func.particle_stencil(
            self.wall_pairs.sorted_idx[i_pair], self.grid, self.newton, self.mpm_info
        )
        u_p = mpm_func.particle_node_sum(slots, w, self.newton.u)
        direction = mpm_func.wall_normal(self.wall_pairs.axis[i_pair], self.wall_pairs.sign[i_pair])
        energy, g, H = mpm_func.soft_row_terms(
            direction, self.wall_pairs.stiffness[i_pair], self.wall_pairs.target[i_pair], u_p
        )
        return i_b, energy, g, H

    @qd.func(requires_top_level=True)
    def func_pair_terms(self, f: int, i_kind: qd.template(), contact: qd.template()):
        """Write the gradient and Hessian of the row of every wall pair at the current iterate on its particle."""
        qd.loop_config(name="mpm_wall_pair_terms")
        for i_pair in range(self.wall_pairs.n_pairs[()]):
            i_b, energy, g, H = self.wall_pair_terms(i_pair)
            i_v = self.vertex_range.offset[()] + self.wall_pairs.sorted_idx[i_pair]
            contact.write_doublet(contact.matrices.doublet_offsets[i_kind] + i_pair, i_v, g)
            contact.write_triplet(contact.matrices.triplet_offsets[i_kind] + i_pair, i_v, i_v, H)

    @qd.func(requires_top_level=True)
    def func_pair_energy(self, f: int, energies: qd.template()):
        """Add the energy of the row of every wall pair at the current iterate into the entry of its environment."""
        qd.loop_config(name="mpm_wall_pair_energy")
        for i_pair in range(self.wall_pairs.n_pairs[()]):
            i_b, energy, g, H = self.wall_pair_terms(i_pair)
            qd.atomic_add(energies[i_b], energy)


@qd.data_oriented
class MPMSolver(ParticleSolver):
    """Implicit material point method."""

    material_cls = MPM.Base
    has_frames = True
    # The configurations without an adjoint raise at build (see '_check_supported')
    is_differentiable = True

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
        self._constraints_initialized = False
        # The namespace of the structs the grid growth builds again, which find the error flags of the solver in it
        self._namespace = Namespace("mpm")

        self._n_vvert_supports = self.scene.options.vis.n_support_neighbors

        # `_particle_volume_scale` is used to avoid potential numerical instability, as the actual `_particle_volume` may be very small.
        # Particle masses are stored scaled by it, and the implicit step divides it back out to work in kilograms.
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
        # Particle state of the frames, its gradient-free part and the static info, as dynamic tensors in the default
        # array mode, so that a scene with another particle count reuses the compiled kernels
        sizes_state = dict(frames=self.n_frames, particles=self._n_particles, envs=self._B)
        self.particles = MpmParticles(sizes=sizes_state, requires_grad=self._requires_grad)
        self.particles_ng = MpmParticlesNG(sizes=sizes_state)
        self.particles_info = MpmParticlesInfo(sizes=dict(particles=self._n_particles))

        # The errors of each env, raised at the beginning of the next step (see 'Solver.register_errors')
        with namespace(self._namespace):
            self.mpm_errno = MpmErrno(sizes=dict(envs=self._B))
        self.register_errors(
            self.mpm_errno.errno,
            {
                ErrorCode.INVALID_VELOCITY_NAN: (
                    "Invalid MPM particle velocities causing 'nan' or 'inf'. Please decrease the simulation timestep or "
                    "adjust the material parameters."
                ),
                ErrorCode.INVALID_MPM_SOLVE_NAN: (
                    "The implicit MPM Newton solve produced non-finite values. Try reducing the time step size or "
                    "adjusting simulation parameters."
                ),
                ErrorCode.OVERFLOW_MPM_GRID_RANGE: (
                    "MPM particles have non-finite positions or left the range the sparse grid can index."
                ),
                ErrorCode.INVALID_MPM_GRID: (
                    "Internal error of the implicit MPM solver: its sparse grid or matrix storage is inconsistent. "
                    "Rerun with `gs.init(debug=True)` to stop at the faulty kernel."
                ),
            },
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(pos=gs.qd_vec3, vel=gs.qd_vec3, active=gs.qd_bool)
        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

    def init_implicit_data(self):
        """Allocate the data of the implicit step at the smallest capacity."""
        n_sort = self._B * self._n_particles
        max_blocks = MIN_BLOCKS
        self._is_mpm_sized = False
        materials = {material.idx: material for material in self._materials}
        n_materials = len(materials)
        table = np.zeros((mpm_func.N_MATERIAL_PARAMS, n_materials), dtype=gs.np_float)
        family = np.zeros(n_materials, dtype=gs.np_int)
        plastic = np.zeros(n_materials, dtype=gs.np_int)
        for idx, material in materials.items():
            table[:, idx], family[idx], plastic[idx] = material_table_column(material)

        self.mpm_info = MpmInfo(
            values=dict(
                n_envs=self._B,
                n_particles=self._n_particles,
                substep_dt=self.substep_dt,
                dx=self._dx,
                inv_dx=1.0 / self._dx,
                particle_volume=self._particle_volume_real,
                mass_scale=self._particle_volume_scale,
                n_materials=n_materials,
                liquid_band=self._liquid_band(),
                hencky_sigma_min=0.05,
                extension_sigma_min=0.4,
                extension_stiffness=20.0,
                von_mises_sigma_min=0.05,
                sand_sigma_min=1e-4,
                liquid_j_min=0.2,
                inversion_step_slack=0.9,
                snow_hardening=10.0,
                contact_min_friction=1e-2,
                wall_sol_params=self._wall_sol_params(),
                wall_lower=tuple(self.boundary.lower),
                wall_upper=tuple(self.boundary.upper),
            )
        )
        self.mpm_materials = MpmMaterialTable(
            sizes=dict(materials=n_materials, material_params=mpm_func.N_MATERIAL_PARAMS)
        )
        self.mpm_materials.material_table.from_numpy(table.reshape(-1))
        self.mpm_materials.materials_family.from_numpy(family)
        self.mpm_materials.materials_plastic.from_numpy(plastic)
        with namespace(self._namespace):
            grid = build_mpm_grid(n_sort, max_blocks)
        self._newton_participant = MpmNewtonParticipant(
            self.particles,
            self.particles_info,
            self.mpm_materials,
            grid,
            build_mpm_newton(n_sort, max_blocks),
            build_block_jacobi(max_blocks * BLOCK_CELLS),
            build_wall_pairs(MIN_SLOTS),
            self.mpm_info,
            n_sort,
        )
        self._mpm_force_fields = MpmForceFields(self._ffs)

    def build_newton_participant(self) -> tuple["MPMSolver", MpmNewtonParticipant]:
        """Build the participant of the grid displacement in the joint Newton solve, with this solver as its host (see
        LegacyCoupler)."""
        return self, self._newton_participant

    def _liquid_band(self):
        velocity = self._options.liquid_smoothing_velocity
        if velocity is None:
            velocity = self._scene.options.newton.velocity_tol
        band = velocity * self.substep_dt / self._dx
        if band >= 0.5:
            gs.raise_exception(
                f"The smoothing band of the liquid energy, {band:.3g} in volume ratio, would soften the liquid down to "
                "half its volume: lower MPMOptions(liquid_smoothing_velocity), NewtonOptions(velocity_tol) or the "
                "substep."
            )
        return band

    def _wall_sol_params(self):
        params = gu.default_solver_params()
        params[0] = max(self._options.wall_timeconst, 2.0 * self.substep_dt)
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

    def init_constraints(self):
        """Lazy initialization of particle constraint fields."""
        # Memory check: ensure index fits in int32
        if self._n_particles * self._B * 3 > np.iinfo(gs.np_int).max:
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
        if self._requires_grad and self.is_active:
            qd_zero_grad(self.particles)
            if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
                torch.mps.synchronize()

    def _check_supported(self):
        if self._B > mpm_func.MAX_KEY_ENVS:
            gs.raise_exception(f"The implicit MPM solver supports {mpm_func.MAX_KEY_ENVS} environments at most.")
        cells = np.concatenate([self._lower_bound, self._upper_bound]) * self._grid_density
        if cells.min() < mpm_func.CELL_MIN or cells.max() >= mpm_func.CELL_END:
            gs.raise_exception(
                f"The MPM domain spans the cells [{cells.min():.0f}, {cells.max():.0f}], past the range "
                f"[{mpm_func.CELL_MIN}, {mpm_func.CELL_END}) the sparse grid of the implicit solver indexes."
            )
        for material in self._materials:
            if not isinstance(material, (MPM.Elastic, MPM.ElastoPlastic, MPM.Liquid, MPM.Sand)):
                gs.raise_exception(
                    f"The implicit MPM solver has no energy for the material {type(material).__name__}. It supports "
                    "the materials of genesis.materials.MPM."
                )
        # The legacy coupler registers the Newton solve of every substep (see LegacyCoupler)
        if not isinstance(self._sim.coupler.options, LegacyCouplerOptions):
            gs.raise_exception(
                "The implicit MPM solver requires LegacyCouplerOptions, got "
                f"{type(self._sim.coupler.options).__name__}."
            )
        if self._requires_grad:
            for force_field in self._ffs:
                if not force_field.differentiable:
                    gs.raise_exception(
                        f"Differentiable simulation does not support the force field {type(force_field).__name__}."
                    )

    def build(self):
        super().build()

        # particles and entities
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        if self.is_active:
            self._check_supported()

            self.init_particle_fields()
            self.init_vvert_fields()

            for entity in self._entities:
                entity._add_to_solver()

            self.init_implicit_data()

        self._build_gravity()
        if self.is_active:
            self._init_contact_invweight()

    def _init_contact_invweight(self):
        self._predict(0, t=0.0)
        kernel_contact_invweight(
            0,
            self.particles,
            self._newton_participant.grid,
            self._newton_participant.newton,
            self.particles_info,
            self.mpm_materials,
            self.mpm_info,
        )

    def _predict(self, f, t):
        if not self._is_mpm_sized:
            self._size_to_particles(f)
        build_grid(self.particles, self.particles_ng, self._newton_participant.grid, self.mpm_info, f)
        while int(qd_to_numpy(self._newton_participant.grid.has_grid_overflow)):
            self._grow_grid()
            build_grid(self.particles, self.particles_ng, self._newton_participant.grid, self.mpm_info, f)
        kernel_predict(
            f,
            t,
            self._gravity,
            self.particles,
            self._newton_participant.grid,
            self._newton_participant.newton,
            self.particles_info,
            self._mpm_force_fields,
            self.mpm_info,
        )

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
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        pass

    def _resize_blocks(self, max_blocks):
        n_sort = self._B * self._n_particles
        with namespace(self._namespace):
            grid = build_mpm_grid(n_sort, max_blocks)
        self._newton_participant.replace_scratch(
            grid, build_mpm_newton(n_sort, max_blocks), build_block_jacobi(max_blocks * BLOCK_CELLS)
        )

    def _size_to_particles(self, f):
        positions = qd_to_numpy(self.particles.pos, col_mask=f, keepdim=False, transpose=True)
        self._resize_blocks(initial_block_capacity(positions, self._inv_dx))
        self._is_mpm_sized = True

    def _grow_grid(self):
        demand = max(
            int(qd_to_numpy(self._newton_participant.grid.grid_demand)),
            int(qd_to_numpy(self._newton_participant.grid.n_particle_blocks)),
        )
        max_blocks = max(
            int(np.ceil(demand * CAPACITY_GROW_FACTOR)), self._newton_participant.grid.blocks_key.shape[0] + 1
        )
        self._resize_blocks(max_blocks)

    def substep_post_coupling(self, f):
        """Finish the substep with the check of the particle positions.

        The grid-to-particle transfer of the substep ran earlier, in the substep of the Newton solver."""
        # The positions a step ends with are checked once, an error being raised at the beginning of the next step
        if (self._sim.cur_substep_global + 1) % self._substeps == 0:
            kernel_flag_non_finite_pos(f + 1, self.particles, self.mpm_errno)

    # ------------------------------------------------------------------------------------
    # --------------------------- Newton solver participant ------------------------------
    # ------------------------------------------------------------------------------------

    def newton_begin(self, f):
        pass

    def newton_prepare(self, f):
        """Run the substep up to the Newton solve: the grid, the particle-to-grid transfer with the force fields, the
        physical Hessian at u = 0 and the wall pairs."""
        self._predict(f, self._sim.cur_t)
        if self._constraints_initialized:
            kernel_constraint_rows(
                self.particle_constraints,
                self._newton_participant.grid,
                self._newton_participant.newton,
                self._sim.rigid_solver.dyn_state.links,
                self.particles_info,
                self.mpm_info,
            )
        kernel_report_extent(self._newton_participant.grid, self._newton_participant.linear_block)
        kernel_wall_pairs(
            f,
            self.particles,
            self._newton_participant.grid,
            self._newton_participant.wall_pairs,
            self.particles_info,
            self.mpm_info,
        )
        n_pairs = int(qd_to_numpy(self._newton_participant.wall_pairs.n_pairs))
        if n_pairs > self._newton_participant.wall_pairs.sorted_idx.shape[0]:
            self._newton_participant.replace_wall_pairs(build_wall_pairs(grown_capacity(n_pairs)))
            kernel_wall_pairs(
                f,
                self.particles,
                self._newton_participant.grid,
                self._newton_participant.wall_pairs,
                self.particles_info,
                self.mpm_info,
            )

    def newton_finish(self, f, control):
        """Transfer the solved grid displacement back to the particles, those of an environment whose solve stopped on
        a non-finite measure keeping their state."""
        kernel_finalize(
            f,
            control.has_failed_non_finite,
            self.particles,
            self.particles_ng,
            self._newton_participant.grid,
            self._newton_participant.newton,
            self.particles_info,
            self.mpm_materials,
            self.mpm_info,
        )

    def newton_after(self, f):
        pass

    def newton_end(self, f, control):
        pass

    def newton_backward(self, f, solve):
        """Reverse a substep from the state left by the replay of its forward pass (see mpm_adjoint_func).

        The reverse runs the reverse of the grid-to-particle transfer, the adjoint solve `solve`, the pullback and the
        reverse of the particle-to-grid transfer into frame `f`."""
        kernel_adjoint_g2p(
            f,
            self.particles,
            self._newton_participant.grid,
            self._newton_participant.newton,
            self.particles_info,
            self.mpm_materials,
            self.mpm_info,
        )
        solve()
        if self._constraints_initialized:
            kernel_constraint_adjoint(
                f,
                self.particle_constraints,
                self.particles,
                self._newton_participant.grid,
                self._newton_participant.newton,
                self._newton_participant.linear,
                self._newton_participant.linear_block,
                self._sim.rigid_solver.dyn_state.links,
                self.mpm_info,
            )
        kernel_adjoint_pullback(
            f,
            self._sim.cur_t,
            self._gravity,
            self.particles,
            self.particles_ng,
            self._newton_participant.grid,
            self._newton_participant.newton,
            self._newton_participant.linear,
            self._newton_participant.linear_block,
            self._newton_participant.wall_pairs,
            self.particles_info,
            self.mpm_materials,
            self._mpm_force_fields,
            self.mpm_info,
        )

    def newton_adjoint_rhs(self, f):
        """Stage the adjoint of u* left by the reverse of the grid-to-particle transfer as the right-hand side."""
        kernel_adjoint_rhs(
            self._newton_participant.grid,
            self._newton_participant.newton,
            self._newton_participant.linear,
            self._newton_participant.linear_block,
        )

    def newton_adjoint_finish(self, f):
        pass

    def newton_flag_non_finite(self, control):
        kernel_flag_newton_non_finite(control, self.mpm_errno)

    @qd.kernel
    def copy_frame(self, source: int, target: int):
        for i_p, i_b in qd.ndrange(self.particles.pos.shape[1], self.particles.pos.shape[2]):
            self.particles.pos[target, i_p, i_b] = self.particles.pos[source, i_p, i_b]
            self.particles.vel[target, i_p, i_b] = self.particles.vel[source, i_p, i_b]
            self.particles.F[target, i_p, i_b] = self.particles.F[source, i_p, i_b]
            self.particles.C[target, i_p, i_b] = self.particles.C[source, i_p, i_b]
            self.particles.Jp[target, i_p, i_b] = self.particles.Jp[source, i_p, i_b]
            self.particles.actu[target, i_p, i_b] = self.particles.actu[source, i_p, i_b]

            self.particles_ng.active[target, i_p, i_b] = self.particles_ng.active[source, i_p, i_b]

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def data(self):
        """Yield the state of the particles, then the particle and material parameters (see 'Solver.data').

        A boundary checkpoint holds the frame 0 of the state. The grid, Newton and linear-system arrays of the implicit
        step are those of its participant of the Newton solve (see MpmNewtonParticipant)."""
        yield from iter_data(self, "particles")
        yield from iter_data(self, "particles_ng")
        if not self.is_built or not self.is_active:
            return
        yield from iter_data(self, "particles_info")
        yield from iter_data(self, "mpm_info")
        yield from iter_data(self, "mpm_materials")

    @mutates()
    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.actu, state.active)
            self.clear_errno(envs_idx)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: int,
        pos: qd.types.ndarray(),  # shape [B, n_particles, 3]
        vel: qd.types.ndarray(),  # shape [B, n_particles, 3]
        C: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
        F: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
        Jp: qd.types.ndarray(),  # shape [B, n_particles]
        actu: qd.types.ndarray(),  # shape [B, n_particles]
        active: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p, i_b in qd.ndrange(self.particles.pos.shape[1], self.particles.pos.shape[2]):
            for j in qd.static(range(3)):
                self.particles.pos[f, i_p, i_b][j] = pos[i_b, i_p, j]
                self.particles.vel[f, i_p, i_b][j] = vel[i_b, i_p, j]
                for k in qd.static(range(3)):
                    self.particles.C[f, i_p, i_b][j, k] = C[i_b, i_p, j, k]
                    self.particles.F[f, i_p, i_b][j, k] = F[i_b, i_p, j, k]
            self.particles.Jp[f, i_p, i_b] = Jp[i_b, i_p]
            self.particles.actu[f, i_p, i_b] = actu[i_b, i_p]
            self.particles_ng.active[f, i_p, i_b] = active[i_b, i_p]

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
    def _kernel_update_render_fields(self, f: int):
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
                i_p_support = self.vverts_info.support_idxs[i_v][j]
                vvert_pos += self.particles.pos[f, i_p_support, i_b] * self.vverts_info.support_weights[i_v][j]
            self.vverts_render[i_v, i_b].pos = vvert_pos
            i_p_first = self.vverts_info.support_idxs[i_v][0]
            self.vverts_render[i_v, i_b].active = self.particles_render[i_p_first, i_b].active

    @qd.kernel
    def add_particles(
        self,
        f: int,
        active: int,
        particle_start: int,
        n_particles: int,
        material_idx: int,
        mat_default_Jp: float,
        mat_rho: float,
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
            # The inverse mass of the particle alone, kept by a particle inactive in the configuration that the contact
            # inverse weights are computed from (see _init_contact_invweight)
            self.particles_info.contact_invweight[i_p] = qd.Matrix.identity(gs.qd_float, 3) / (
                self._particle_volume_real * mat_rho
            )

        for i_p_, i_b in qd.ndrange(n_particles, self.particles.pos.shape[2]):
            i_p = i_p_ + particle_start

            self.particles_ng.active[f, i_p, i_b] = qd.cast(active, gs.qd_bool)
            for i in qd.static(range(3)):
                self.particles.pos[f, i_p, i_b][i] = pos[i_p_, i]

            self.particles.vel[f, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.particles.F[f, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
            self.particles.C[f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles.Jp[f, i_p, i_b] = mat_default_Jp
            self.particles.actu[f, i_p, i_b] = gs.qd_float(0.0)

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
                    self.particles.pos[0, i_p, i_b][i] = poss[i_b_, i_p_, i]

                # Reset these attributes whenever overwriting particle positions manually
                self.particles.vel[0, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
                self.particles.F[0, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
                self.particles.C[0, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
                self.particles.Jp[0, i_p, i_b] = self.particles_info.default_Jp[i_p]

    @qd.kernel
    def _kernel_get_particles_pos(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles.pos[0, i_p, i_b][i]

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
                    self.particles.vel[0, i_p, i_b][i] = vels[i_b_, i_p_, i]

    @qd.kernel
    def _kernel_get_particles_vel(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles.vel[0, i_p, i_b][i]

    @qd.kernel
    def _kernel_set_particles_active(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_ng.active[0, i_p, i_b] = qd.cast(actives[i_b_, i_p_], gs.qd_bool)

    @qd.kernel
    def _kernel_get_particles_active(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng.active[0, i_p, i_b]

    @qd.kernel
    def _kernel_set_particles_actu(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actus: qd.types.ndarray(), n_groups: int
    ):
        for i_p_, i_g, i_b_ in qd.ndrange(particles_idx.shape[1], n_groups, envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                if self.particles_info.muscle_group[i_p] == i_g:
                    self.particles.actu[0, i_p, i_b] = actus[i_b_, i_p_, i_g]

    @qd.kernel
    def _kernel_get_particles_actu(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actus: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            actus[i_b_, i_p_] = self.particles.actu[0, i_p, i_b]

    @qd.kernel
    def _kernel_set_particles_muscle_group(self, particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_info.muscle_group[i_p] = muscle_group[i_p_]

    @qd.kernel
    def _kernel_get_particles_muscle_group(self, particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            muscle_group[i_p_] = self.particles_info.muscle_group[i_p]

    @qd.kernel
    def _kernel_set_particles_muscle_direction(
        self, particles_idx: qd.types.ndarray(), muscle_direction: qd.types.ndarray()
    ):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles_info.muscle_direction[i_p][i] = muscle_direction[i_p_, i]

    @qd.kernel
    def _kernel_set_particles_free(self, particles_idx: qd.types.ndarray(), free: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_info.free[i_p] = free[i_p_]

    @qd.kernel
    def _kernel_get_particles_free(self, particles_idx: qd.types.ndarray(), free: qd.types.ndarray()):
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            free[i_p_] = self.particles_info.free[i_p]

    @qd.kernel
    def _kernel_get_particles_mass(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), masses: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            masses[i_b_, i_p_] = self.particles_info.mass[i_p] / self._particle_volume_scale

    # ------------------------------------------------------------------------------------
    # -------------------------------- particle constraints ------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_particle_constraints(self, particles_mask, link_idx, stiffness, *, particles_start=0):
        """
        Attach some particles to a rigid link with critically damped springs.

        Each particle is pulled toward the position it holds now, expressed in the frame of the link. The springs are
        integrated implicitly, stable at any stiffness, and act on the particles alone: the link feels no reaction.

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
        particles_mask: qd.types.ndarray(),
        link_pos: qd.types.ndarray(),
        link_quat: qd.types.ndarray(),
    ):
        for i_p_, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_]:
                i_p = i_p_ + particles_start
                # Between two steps, the state sits in frame 0 (see 'Solver.has_frames')
                pos = self.particles.pos[0, i_p, i_b]

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
    def _kernel_remove_particle_constraints(self, particles_start: int, particles_mask: qd.types.ndarray()):
        for i_p_, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
            if particles_mask[i_b, i_p_]:
                i_p = i_p_ + particles_start
                self.particle_constraints[i_p, i_b].is_constrained = False
                self.particle_constraints[i_p, i_b].link_idx = -1

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
        raise DeprecationError("This property has been removed.")

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


@no_grad
@qd.kernel
def kernel_flag_non_finite_pos(f: int, particles: MpmParticles, mpm_errno: MpmErrno):
    for i_p, i_b in qd.ndrange(particles.pos.shape[1], particles.pos.shape[2]):
        if not qd_isfinite(particles.pos[f, i_p, i_b]):
            mpm_errno.errno[i_b] = mpm_errno.errno[i_b] | ErrorCode.INVALID_VELOCITY_NAN


@no_grad
@qd.kernel
def kernel_flag_newton_non_finite(control: NewtonControl, mpm_errno: MpmErrno):
    for i_b in range(mpm_errno.errno.shape[0]):
        if control.has_failed_non_finite[i_b] != 0:
            mpm_errno.errno[i_b] = mpm_errno.errno[i_b] | ErrorCode.INVALID_MPM_SOLVE_NAN


@qd.kernel
def kernel_grid_keys(
    f: int, particles: MpmParticles, particles_ng: MpmParticlesNG, grid: MpmGridScratch, mpm_info: MpmInfo
):
    mpm_func.func_reset_grid(grid)
    mpm_func.func_compose_particle_keys(f, particles, particles_ng, grid, mpm_info)


@qd.kernel
def kernel_grid_blocks(grid: MpmGridScratch, mpm_info: MpmInfo):
    mpm_func.func_append_particle_blocks(grid, mpm_info)
    mpm_func.func_gen_neighbor_keys(grid)


@qd.kernel
def kernel_grid_block_flags(grid: MpmGridScratch):
    mpm_func.func_block_flags(grid)


@qd.kernel
def kernel_grid_publish_blocks(grid: MpmGridScratch):
    mpm_func.func_publish_blocks(grid)
    mpm_func.func_publish_grid_demand(grid)
    mpm_func.func_cell_flags(grid)


@qd.kernel
def kernel_grid_cells(grid: MpmGridScratch):
    mpm_func.func_publish_cells(grid)
    mpm_func.func_reset_cell_segs(grid)
    mpm_func.func_fill_cell_segs(grid)


def build_grid(particles, particles_ng, grid, mpm_info, f):
    """Rebuild the block-sparse grid of the substep from the particles, its topology fixed until the next substep."""
    kernel_grid_keys(f, particles, particles_ng, grid, mpm_info)
    dynamic_radix_sort(
        grid.particle_sort,
        grid.sort_keys,
        grid.sort_keys_buffer,
        grid.sort_ids,
        grid.sort_ids_buffer,
        grid.n_sort_keys,
        key_bits=64,
    )
    kernel_grid_blocks(grid, mpm_info)
    dynamic_radix_sort(
        grid.block_sort,
        grid.neighbors_key,
        grid.neighbors_key_buffer,
        grid.neighbors_source_idx,
        grid.neighbors_source_idx_buffer,
        grid.n_neighbor_keys,
        key_bits=64,
    )
    kernel_grid_block_flags(grid)
    kernel_dynamic_exclusive_sum(
        grid.neighbors_is_new_block, grid.neighbors_block_idx, grid.n_neighbor_keys, grid.block_scan
    )
    kernel_grid_publish_blocks(grid)
    kernel_dynamic_exclusive_sum(grid.sort_is_cell_start, grid.sort_cell_idx, grid.n_valid, grid.cell_scan)
    kernel_dynamic_exclusive_sum(grid.sort_is_task_start, grid.sort_task_idx, grid.n_valid, grid.task_scan)
    kernel_grid_cells(grid)


@qd.kernel
def kernel_predict(
    f: int,
    t: float,
    gravity: qd.Tensor,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    force_fields: qd.template(),
    mpm_info: MpmInfo,
):
    """Transfer the particles to the grid with the force fields, and reset the Newton state."""
    dt = mpm_info.substep_dt[()]
    mpm_func.func_p2g(f, dt, gravity, particles, grid, particles_info, mpm_info)
    mpm_func.func_apply_force_fields(t, dt, grid, force_fields, mpm_info)
    mpm_func.func_stage_particle_pos(f, particles, grid, newton, mpm_info)
    mpm_func.func_newton_reset(grid, newton)


@qd.kernel
def kernel_wall_pairs(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    pairs: MpmWallPairs,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    """Build the wall pairs of the substep."""
    dt = mpm_info.substep_dt[()]
    mpm_func.func_build_wall_pairs(f, dt, particles, grid, pairs, particles_info, mpm_info)


@qd.kernel
def kernel_constraint_rows(
    constraints: qd.template(),
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    links_state: LinksState,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    mpm_func.func_build_constraint_rows(constraints, grid, newton, links_state, particles_info, mpm_info)


@qd.kernel
def kernel_constraint_adjoint(
    f: int,
    constraints: qd.template(),
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    links_state: LinksState,
    mpm_info: MpmInfo,
):
    mpm_adjoint_func.func_constraint_adjoint(
        f, constraints, particles, grid, newton, linear, rows, links_state, mpm_info
    )


@qd.kernel
def kernel_contact_invweight(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    mpm_func.func_contact_invweight(f, particles, grid, newton, particles_info, materials, mpm_info)


@qd.kernel
def kernel_report_extent(grid: MpmGridScratch, rows: LinearBlock):
    mpm_func.func_report_extent(grid, rows)


@qd.kernel
def kernel_finalize(
    f: int,
    is_env_frozen: qd.Tensor,
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    dt = mpm_info.substep_dt[()]
    mpm_func.func_g2p_advect(
        f, dt, is_env_frozen, particles, particles_ng, grid, newton, particles_info, materials, mpm_info
    )


@qd.kernel
def kernel_adjoint_g2p(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Compute the adjoint of u* into `newton.du` and the direct terms of the adjoint of the start frame of the
    substep."""
    dt = mpm_info.substep_dt[()]
    mpm_adjoint_func.func_adjoint_reset_u_bar(grid, newton)
    mpm_adjoint_func.func_adjoint_g2p(f, dt, particles, grid, newton, particles_info, materials, mpm_info)


@qd.kernel
def kernel_adjoint_rhs(grid: MpmGridScratch, newton: MpmNewtonScratch, linear: NewtonLinearScratch, rows: LinearBlock):
    mpm_adjoint_func.func_adjoint_rhs(grid, newton, linear, rows)


@qd.kernel
def kernel_adjoint_pullback(
    f: int,
    t: float,
    gravity: qd.Tensor,
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    wall_pairs: MpmWallPairs,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    force_fields: qd.template(),
    mpm_info: MpmInfo,
):
    """Pull the adjoint solution back into the start frame of a substep, then clear the adjoint of its end frame."""
    dt = mpm_info.substep_dt[()]
    mpm_adjoint_func.func_adjoint_nodes(t, dt, grid, newton, linear, rows, force_fields, mpm_info)
    mpm_adjoint_func.func_adjoint_particles(
        f, dt, gravity, particles, grid, newton, linear, rows, particles_info, materials, mpm_info
    )
    mpm_adjoint_func.func_wall_contact_adjoint(
        f, dt, particles, grid, newton, linear, rows, wall_pairs, particles_info, mpm_info
    )
    mpm_adjoint_func.func_adjoint_consume(f, particles, particles_ng, mpm_info)
